# =============================================================================
# CMB-LLM Step 8b - RAG-Faithfulness Probe at scale + controls (Qwen)
# 40 pairs (80 prompts). Three numbers, to earn the pilot's 1.000:
#   (1) full LOGO-CV AUC at layer 17.
#   (2) LENGTH-MATCHED control: AUC restricted to pairs whose supported/unsupported
#       prompts are within a small token-count delta (kills the length confound).
#   (3) LEAKAGE check: layer-0 (input-embedding) AUC, should be ~0.5.
# Validate here BEFORE spending four-model GPU. ~2-4 min incl. model load.
#
# RUN ORDER (fresh kernel): pipeline_bootstrap -> step8a_writefile -> THIS.
# =============================================================================

import sys, os, json, time
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

from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT

if 'model' not in globals() or 'tokenizer' not in globals():
    from harness.inference import load_model
    print('Loading model...')
    model, tokenizer = load_model('Qwen/Qwen2.5-7B-Instruct', load_in_4bit=False)

pilot_path = '/content/cmb_llm/harness/rag_faithfulness.py'
if not os.path.exists(pilot_path):
    raise FileNotFoundError("rag_faithfulness.py missing. Run step8a_writefile first.")

from harness.rag_faithfulness import PILOT_PAIRS, all_labeled_prompts
rows = all_labeled_prompts()
print(f'Loaded {len(rows)} prompts from {len(PILOT_PAIRS)} pairs')

import torch

def build_prompt_text(user_msg):
    messages = [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
                {"role": "user", "content": user_msg}]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def collect(layer):
    X, tok_lens = [], []
    for prompt, label, pid, cond in rows:
        pt = build_prompt_text(prompt)
        inputs = tokenizer(pt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        h = out.hidden_states[layer][0, -1, :].to(torch.float32).cpu().numpy()
        X.append(h); tok_lens.append(int(inputs["input_ids"].shape[1]))
        del out; torch.cuda.empty_cache()
    return np.stack(X), tok_lens

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

y      = np.array([r[1] for r in rows], dtype=int)
groups = np.array([r[2] for r in rows])

def logo_auc(X, y, groups):
    logo = LeaveOneGroupOut()
    scores = np.zeros(len(y))
    for tr, te in logo.split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
        scores[te] = clf.decision_function(X[te])
    return roc_auc_score(y, scores)

print('\nCollecting layer-17 activations (80 prompts)...')
t0 = time.time()
X17, tok_lens = collect(PROBE_LAYER)
print(f'  done in {time.time()-t0:.0f}s')
auc_full = logo_auc(X17, y, groups)
print(f'\n(1) full LOGO-CV AUC @ L17 = {auc_full:.3f}')

# (2) length-matched control: per-pair |token delta| between the two conditions
tok = np.array(tok_lens)
pair_ids = sorted(set(groups))
deltas = {}
for pid in pair_ids:
    idx = np.where(groups == pid)[0]
    deltas[pid] = abs(int(tok[idx[0]] - tok[idx[1]]))
dvals = np.array(list(deltas.values()))
print(f'\n  per-pair |token delta|: max {dvals.max()}, median {int(np.median(dvals))}, mean {dvals.mean():.1f}')
for T in (2, 4):
    keep = [pid for pid in pair_ids if deltas[pid] <= T]
    mask = np.isin(groups, keep)
    if len(keep) >= 4:
        auc_m = logo_auc(X17[mask], y[mask], groups[mask])
        print(f'(2) length-matched AUC (|delta|<= {T} tok, {len(keep)}/{len(pair_ids)} pairs) = {auc_m:.3f}')
    else:
        print(f'(2) only {len(keep)} pairs within |delta|<= {T} - too few')

# (3) leakage: layer 0
print('\nCollecting layer-0 activations (leakage check)...')
X0, _ = collect(0)
auc_l0 = logo_auc(X0, y, groups)
print(f'(3) layer-0 AUC (should be ~0.5) = {auc_l0:.3f}')

out_path = f'{OUT_DIR}/rag_full_probe_results.json'
with open(out_path, 'w') as f:
    json.dump({"n_prompts": len(rows), "n_pairs": len(pair_ids),
               "auc_full_l17": float(auc_full), "auc_layer0_leakage": float(auc_l0),
               "token_delta": {int(k): int(v) for k, v in deltas.items()}}, f, indent=2)
print(f'\nSaved -> {out_path}')

print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
if auc_full >= 0.85 and auc_l0 <= 0.62:
    print(f'L17 AUC {auc_full:.3f}, leakage {auc_l0:.3f} -> signal is real and contextual.')
    print('Scaled dataset validated. Proceed to the cross-model + gap-at-scale pair.')
elif auc_full >= 0.85:
    print(f'L17 AUC {auc_full:.3f} strong, but layer-0 {auc_l0:.3f} is elevated -> some')
    print('surface signal. Check the length-matched AUC above before trusting it.')
else:
    print(f'L17 AUC {auc_full:.3f} -> weaker at scale than the 12-pair pilot. Inspect')
    print('which pairs the probe misses before committing four-model GPU.')
print('=' * 70)
