# =============================================================================
# CMB-LLM Step 8c - RAG-Faithfulness CROSS-MODEL (probe + gap, 4 families)
# One resumable walk-away job. For each model: layer-17 probe (full + length-
# matched + layer-0 leakage) AND the generation gap under BOTH a strict document
# prompt and a weak helpful prompt. Per-model JSON saved after each model, so a
# disconnect never loses a finished model; re-running skips completed models.
#
# VERIFY THE MODEL IDS BELOW before running - especially OLMo. These should match
# the four families your paper used (Qwen, Mistral 7B v0.3, OLMo-2 7B, Llama-3.1
# 8B). Llama is gated: make sure your HF token is set in this Colab session.
#
# RUN ORDER (fresh kernel): pipeline_bootstrap -> step8a_writefile -> THIS.
# ~25-40 min depending on download + generation. Walk away.
# =============================================================================

MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct",   # <-- verify matches your paper's OLMo-2 7B
    "meta-llama/Llama-3.1-8B-Instruct",  # <-- gated; needs your HF token
]
MATCH_TOK = 4   # length-matched control: keep pairs within this token delta

import sys, os, json, time, re, gc
import numpy as np

if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception:
    pass

OUT_DIR = '/content/drive/MyDrive/cmb_llm_intervention'
os.makedirs(OUT_DIR, exist_ok=True)
PROBE_LAYER = 17

from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT   # strict "from the document"
from harness.rag_faithfulness import PILOT_PAIRS, all_labeled_prompts
from harness.inference import load_model
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

WEAK_SYSTEM_PROMPT = "You are a helpful assistant."

rows   = all_labeled_prompts()
y      = np.array([r[1] for r in rows], dtype=int)
groups = np.array([r[2] for r in rows])

DECLINE_PATTERNS = [
    r"\bdoes ?n'?t (mention|state|say|specify|provide|contain|include|indicate|give)\b",
    r"\bdoes not (mention|state|say|specify|provide|contain|include|indicate|give)\b",
    r"\bnot (mentioned|stated|specified|provided|given|included|indicated|contained)\b",
    r"\bno (mention|information|indication|reference|details?) (of|about|regarding|on)\b",
    r"\b(passage|document|text|context|provided text) does ?n'?t\b",
    r"\bisn'?t (mentioned|stated|specified|provided)\b",
    r"\bcannot be (determined|found|inferred|answered) (from|based)\b",
    r"\bnot (possible|able) to (determine|answer)\b",
    r"\bthere (is|'s) no (mention|information|indication)\b",
    r"\bnot (explicitly )?(mentioned|stated|provided|given|specified) in\b",
    r"\bthe (passage|document|text) (provided )?does not\b",
]
DECLINE_RE = re.compile("|".join(DECLINE_PATTERNS), re.IGNORECASE)
def declined(r): return bool(DECLINE_RE.search(r))

def safe(name): return name.split("/")[-1].replace(".", "_")

def free_model():
    for nm in ("model", "tokenizer"):
        if nm in globals():
            del globals()[nm]
    gc.collect(); torch.cuda.empty_cache()

def chat_text(tok, sys_prompt, user_msg):
    return tok.apply_chat_template(
        [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_msg}],
        tokenize=False, add_generation_prompt=True)

def logo_auc(X, yy, gg):
    logo = LeaveOneGroupOut(); s = np.zeros(len(yy))
    for tr, te in logo.split(X, yy, gg):
        s[te] = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], yy[tr]).decision_function(X[te])
    return roc_auc_score(yy, s)

def run_model(model_id):
    mdl, tok = load_model(model_id, load_in_4bit=False)

    # ---- probe (L17 + L0) + token lengths under the strict/document framing ----
    X17, X0, tlen = [], [], []
    for prompt, label, pid, cond in rows:
        pt = chat_text(tok, NEUTRAL_SYSTEM_PROMPT, prompt)
        inp = tok(pt, return_tensors="pt").to(mdl.device)
        with torch.no_grad():
            out = mdl(**inp, output_hidden_states=True, return_dict=True)
        X17.append(out.hidden_states[PROBE_LAYER][0, -1, :].float().cpu().numpy())
        X0.append(out.hidden_states[0][0, -1, :].float().cpu().numpy())
        tlen.append(int(inp["input_ids"].shape[1]))
        del out; torch.cuda.empty_cache()
    X17 = np.stack(X17); X0 = np.stack(X0); tlen = np.array(tlen)
    auc_full = logo_auc(X17, y, groups)
    auc_leak = logo_auc(X0, y, groups)
    pair_ids = sorted(set(groups))
    deltas = {pid: abs(int(tlen[np.where(groups == pid)[0][0]] - tlen[np.where(groups == pid)[0][1]]))
              for pid in pair_ids}
    keep = [pid for pid in pair_ids if deltas[pid] <= MATCH_TOK]
    mask = np.isin(groups, keep)
    auc_match = logo_auc(X17[mask], y[mask], groups[mask]) if len(keep) >= 4 else None

    # ---- generation gap: unsupported under strict & weak; supported under strict ----
    def gen(user_msg, sys_prompt):
        inp = tok(chat_text(tok, sys_prompt, user_msg), return_tensors="pt").to(mdl.device)
        n = inp["input_ids"].shape[1]
        with torch.no_grad():
            o = mdl.generate(**inp, max_new_tokens=120, do_sample=False,
                             temperature=1.0, pad_token_id=tok.eos_token_id)
        return tok.decode(o[0, n:], skip_special_tokens=True).strip()

    recs = []
    for p in PILOT_PAIRS:
        rs = gen(p.unsupported_prompt(), NEUTRAL_SYSTEM_PROMPT)
        rw = gen(p.unsupported_prompt(), WEAK_SYSTEM_PROMPT)
        rc = gen(p.supported_prompt(),   NEUTRAL_SYSTEM_PROMPT)
        recs.append({"pair_id": p.pair_id, "hint": p.answer_hint,
                     "strict_decl": declined(rs), "strict_resp": rs[:200],
                     "weak_decl": declined(rw),
                     "weak_unfaithful": (not declined(rw)) and (p.answer_hint.lower() in rw.lower()),
                     "weak_resp": rw[:200],
                     "sup_answered": p.answer_hint.lower() in rc.lower()})
    n = len(recs)
    decl_strict = sum(r["strict_decl"] for r in recs) / n
    decl_weak   = sum(r["weak_decl"] for r in recs) / n
    unfaith_weak = sum(r["weak_unfaithful"] for r in recs) / n
    sup_ans = sum(r["sup_answered"] for r in recs) / n

    res = {"model": model_id, "n_pairs": n,
           "auc_full_l17": float(auc_full), "auc_match_l17": (float(auc_match) if auc_match else None),
           "n_matched_pairs": len(keep), "auc_layer0_leak": float(auc_leak),
           "decline_strict": decl_strict, "decline_weak": decl_weak,
           "unfaithful_weak": unfaith_weak, "supported_answered": sup_ans,
           "gap_strict": float(auc_full) - decl_strict, "gap_weak": float(auc_full) - decl_weak,
           "records": recs}
    free_model()
    return res

# Free whatever model is currently in the kernel before the loop
free_model()

results = []
for mid in MODELS:
    path = f"{OUT_DIR}/rag_crossmodel_{safe(mid)}.json"
    if os.path.exists(path):
        print(f"[skip] {mid} (already saved)")
        results.append(json.load(open(path))); continue
    print(f"\n{'='*70}\n{mid}\n{'='*70}")
    t0 = time.time()
    try:
        res = run_model(mid)
        json.dump(res, open(path, "w"), indent=2)
        results.append(res)
        print(f"  L17 full {res['auc_full_l17']:.3f} | matched {res['auc_match_l17']} "
              f"| leak {res['auc_layer0_leak']:.3f}")
        print(f"  decline strict {res['decline_strict']:.3f} -> weak {res['decline_weak']:.3f} "
              f"| gap_weak {res['gap_weak']:+.3f} | unfaithful_weak {res['unfaithful_weak']:.3f}")
        print(f"  ({time.time()-t0:.0f}s)  saved -> {path}")
    except Exception as e:
        print(f"  ERROR on {mid}: {e}\n  (skipping; re-run to retry this model)")
        free_model()

print("\n" + "=" * 78)
print(f"{'model':<34}{'L17':>6}{'match':>7}{'leak':>6}{'decl_str':>9}{'decl_wk':>8}{'gap_wk':>8}")
print("=" * 78)
for r in results:
    m = r['auc_match_l17']; mstr = f"{m:.3f}" if m else "  -  "
    print(f"{r['model'].split('/')[-1]:<34}{r['auc_full_l17']:>6.3f}{mstr:>7}"
          f"{r['auc_layer0_leak']:>6.3f}{r['decline_strict']:>9.3f}{r['decline_weak']:>8.3f}{r['gap_weak']:>+8.3f}")
print("=" * 78)
print("\nRead: L17 high + leak ~0.5 across models = internal representation is general.")
print("decline strict high, decline weak lower = instruction-gated gap. gap_wk is the")
print("per-model expression gap under the deployment-realistic (weak) prompt.")
