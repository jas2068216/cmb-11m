# =============================================================================
# CMB-LLM Step 6b — Uncertainty Probe Pilot (PATCHED)
# Skips model load (assumes step6 already loaded it). Builds chat messages
# inline so we don't depend on TestCase.to_prompt() — the uncertainty pilot
# has no document context, just a bare question.
# Paste in a NEW cell after the failed step6.
# =============================================================================

import sys, os, json, time
import numpy as np

if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')

OUT_DIR = '/content/drive/MyDrive/cmb_llm_intervention'
os.makedirs(OUT_DIR, exist_ok=True)
PROBE_LAYER = 17

assert 'model' in globals() and 'tokenizer' in globals(), \
    "Model not loaded — run step6 first (it loaded the model before erroring)."

from harness.uncertainty_pilot import PILOT_PAIRS, all_labeled_prompts
from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT

rows = all_labeled_prompts()
print(f'Loaded {len(rows)} prompts from {len(PILOT_PAIRS)} pairs')

import torch

def score_input(prompt_text: str, layer: int = PROBE_LAYER) -> np.ndarray:
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer][0, -1, :].to(torch.float32).cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return h

print('\nCollecting activations for 20 prompts...')
X, y, meta = [], [], []
t0 = time.time()
for prompt, label, pid, cond in rows:
    # Build messages inline — no document, just the bare question.
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    h = score_input(prompt_text, layer=PROBE_LAYER)
    X.append(h)
    y.append(label)
    meta.append({"pair_id": pid, "condition": cond, "prompt": prompt})

X = np.stack(X, axis=0)
y = np.array(y, dtype=int)
elapsed = time.time() - t0
print(f'  collected X={X.shape}  y_pos={int(y.sum())}/{len(y)}  in {elapsed:.0f}s')

# Train LogReg with leave-one-pair-out CV
print('\nTraining probe (leave-one-pair-out CV)...')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

groups = np.array([m["pair_id"] for m in meta])
logo = LeaveOneGroupOut()

fold_scores, fold_indices = [], []
for fold, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups)):
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X[tr_idx], y[tr_idx])
    s = clf.decision_function(X[te_idx])
    fold_scores.extend(s.tolist())
    fold_indices.extend(te_idx.tolist())
print(f'  {logo.get_n_splits(X, y, groups)} folds x 2 held-out samples each')

all_scores = np.zeros(len(y))
for s, i in zip(fold_scores, fold_indices):
    all_scores[i] = s
auc_cv = roc_auc_score(y, all_scores)
print(f'\n[uncertainty pilot] cross-validated AUC = {auc_cv:.3f}')

clf_full = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
auc_train = roc_auc_score(y, clf_full.decision_function(X))
print(f'                     training-set AUC    = {auc_train:.3f}')

out_path = f'{OUT_DIR}/uncertainty_pilot_results.json'
result = {
    "n_prompts":  int(len(y)),
    "n_pairs":    int(len(set(groups))),
    "probe_layer": int(PROBE_LAYER),
    "auc_cv":     float(auc_cv),
    "auc_train":  float(auc_train),
    "per_sample": [
        {**m, "score": float(all_scores[i])} for i, m in enumerate(meta)
    ],
}
with open(out_path, 'w') as f:
    json.dump(result, f, indent=2)
print(f'\nSaved -> {out_path}')

print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
if auc_cv >= 0.90:
    print(f'AUC = {auc_cv:.3f}  ->  STRONG signal. Expand multi-task generalization.')
elif auc_cv >= 0.70:
    print(f'AUC = {auc_cv:.3f}  ->  MODERATE signal. Worth expanding.')
elif auc_cv >= 0.55:
    print(f'AUC = {auc_cv:.3f}  ->  WEAK signal. Possibly real, possibly noise.')
else:
    print(f'AUC = {auc_cv:.3f}  ->  NO SIGNAL. Drop multi-task plan.')
print('=' * 70)

order = np.argsort(all_scores)
print('\nMost "KNOWN-looking" (lowest probe scores):')
for i in order[:3]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:90]}...')
print('\nMost "UNCERTAIN-looking" (highest probe scores):')
for i in order[-3:][::-1]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:90]}...')
