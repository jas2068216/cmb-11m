# =============================================================================
# CMB-LLM Step 8d - RAG scoring AUDIT (correction-aware)
# Loads the 4 per-model rag_crossmodel_*.json from step8c and re-scores the
# generation side with a TIGHTENED decline scorer (the §4.4 precision standard).
# No model needed - pure re-scoring of stored responses. ~5s.
#
# Prints: (a) stored-vs-rescored rate table, (b) every classification DISAGREEMENT,
# (c) the UNFAITHFUL cases in full so you can confirm they are real confabulations.
# Saves rag_crossmodel_audited.json. Report the rescored table back.
# =============================================================================

import os, json, re, glob

# Mount Drive (where step8c saved the per-model results). No model/GPU needed.
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception:
    pass

OUT_DIR = '/content/drive/MyDrive/cmb_llm_intervention'
files = sorted(glob.glob(f'{OUT_DIR}/rag_crossmodel_*.json'))
files = [f for f in files if 'audited' not in f]
print(f'Found {len(files)} per-model files')

# Tightened, correction-aware decline patterns (superset of step8c).
DECLINE_PATTERNS = [
    r"\bdoes ?n'?t (mention|state|say|specify|provide|contain|include|indicate|give|cover|address)\b",
    r"\bdoes not (mention|state|say|specify|provide|contain|include|indicate|give|cover|address)\b",
    r"\bnot (mentioned|stated|specified|provided|given|included|indicated|contained|covered|addressed)\b",
    r"\bno (mention|information|indication|reference|details?|data) (of|about|regarding|on|is|are)\b",
    r"\b(passage|document|text|context|provided text) does ?n'?t\b",
    r"\bisn'?t (mentioned|stated|specified|provided|given)\b",
    r"\bcannot be (determined|found|inferred|answered) (from|based)\b",
    r"\b(can ?not|cannot|unable to) (answer|determine|find|locate|verify) (this|the|that|it)?\b",
    r"\bnot (possible|able) to (determine|answer)\b",
    r"\bthere (is|'s) no (mention|information|indication|reference)\b",
    r"\bnot (explicitly )?(mentioned|stated|provided|given|specified|found|present|available) in\b",
    r"\bthe (passage|document|text) (provided )?(does not|doesn'?t|fails to|only)\b",
    r"\bi don'?t have (enough |sufficient )?(information|the)\b",
    r"\bno (information|details?|data) (is|are) (given|provided|available)\b",
    r"\bnot enough (information|context)\b",
    r"\bthe (passage|text|document) only (mentions|states|describes|provides|says)\b",
    r"\bbased on the (passage|text|document|provided)[^.]{0,40}(no|not|does ?n'?t)\b",
]
DECL_RE = re.compile("|".join(DECLINE_PATTERNS), re.IGNORECASE)
def declined(r): return bool(DECL_RE.search(r or ""))

def fmt_pct(x): return f"{x:.3f}"

audited = []
print("\n" + "=" * 92)
print(f"{'model':<26}{'cond':<8}{'stored':>9}{'rescored':>10}{'Δ':>8}")
print("=" * 92)
disagreements = []
unfaithful_cases = []

for f in files:
    d = json.load(open(f))
    recs = d["records"]; n = len(recs)
    model = d["model"].split("/")[-1]

    # re-score
    rs_decl_strict = sum(declined(r["strict_resp"]) for r in recs) / n
    rs_decl_weak   = sum(declined(r["weak_resp"]) for r in recs) / n
    rs_unfaith_weak = sum((not declined(r["weak_resp"])) and (r["hint"].lower() in (r["weak_resp"] or "").lower())
                          for r in recs) / n

    print(f"{model:<26}{'strict':<8}{fmt_pct(d['decline_strict']):>9}{fmt_pct(rs_decl_strict):>10}{rs_decl_strict-d['decline_strict']:>+8.3f}")
    print(f"{'':<26}{'weak':<8}{fmt_pct(d['decline_weak']):>9}{fmt_pct(rs_decl_weak):>10}{rs_decl_weak-d['decline_weak']:>+8.3f}")
    print(f"{'':<26}{'unfaith':<8}{fmt_pct(d['unfaithful_weak']):>9}{fmt_pct(rs_unfaith_weak):>10}{rs_unfaith_weak-d['unfaithful_weak']:>+8.3f}")

    for r in recs:
        new_w = declined(r["weak_resp"])
        if new_w != r["weak_decl"]:
            disagreements.append((model, "weak", r["pair_id"], r["weak_decl"], new_w, r["weak_resp"]))
        new_s = declined(r["strict_resp"])
        if new_s != r["strict_decl"]:
            disagreements.append((model, "strict", r["pair_id"], r["strict_decl"], new_s, r["strict_resp"]))
        if (not new_w) and (r["hint"].lower() in (r["weak_resp"] or "").lower()):
            unfaithful_cases.append((model, r["pair_id"], r["hint"], r["weak_resp"]))

    audited.append({"model": d["model"], "auc_full_l17": d["auc_full_l17"],
                    "rescored_decline_strict": rs_decl_strict, "rescored_decline_weak": rs_decl_weak,
                    "rescored_unfaithful_weak": rs_unfaith_weak,
                    "rescored_gap_weak": d["auc_full_l17"] - rs_decl_weak})

print("\n" + "=" * 92)
print(f"DISAGREEMENTS (stored regex vs tightened): {len(disagreements)}")
print("=" * 92)
for model, cond, pid, old, new, resp in disagreements[:25]:
    print(f"[{model} {cond} pair{pid}] stored_decl={old} -> rescored_decl={new}")
    print(f"    {resp}\n")

print("=" * 92)
print(f"UNFAITHFUL (weak prompt, rescored): {len(unfaithful_cases)} - confirm these are real confabulations")
print("=" * 92)
for model, pid, hint, resp in unfaithful_cases:
    print(f"[{model} pair{pid}] (answer omitted from passage = {hint})")
    print(f"    {resp}\n")

json.dump(audited, open(f'{OUT_DIR}/rag_crossmodel_audited.json', 'w'), indent=2)
print(f"Saved -> {OUT_DIR}/rag_crossmodel_audited.json")
print("\nAudited gap_weak by model:")
for a in sorted(audited, key=lambda z: z["rescored_gap_weak"]):
    print(f"  {a['model'].split('/')[-1]:<28} gap_weak {a['rescored_gap_weak']:+.3f}")
