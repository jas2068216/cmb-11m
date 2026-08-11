# =============================================================================
# CMB-LLM Step 6 — Uncertainty Probe Pilot
# Tests whether the expression gap generalizes from contradiction → uncertainty.
# 20 prompts (10 paired KNOWN/UNCERTAIN). Trains a linear probe on layer 17
# last-input-token hidden states. AUC > 0.7 = green light to expand the
# multi-task generalization story (path to 5/5 novelty).
# Paste in a fresh Colab cell AFTER pipeline_bootstrap_cell + uncertainty_pilot.py
# is materialized into harness/. ~5-10 min on A100.
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

# Ensure model is loaded
if 'model' not in globals() or 'tokenizer' not in globals():
    from harness.inference import load_model
    print('Loading model...')
    model, tokenizer = load_model('Qwen/Qwen2.5-7B-Instruct', load_in_4bit=False)

# Make sure uncertainty_pilot.py exists in harness/
import os.path
pilot_path = '/content/cmb_llm/harness/uncertainty_pilot.py'
if not os.path.exists(pilot_path):
    raise FileNotFoundError(
        f"uncertainty_pilot.py not found at {pilot_path}. "
        f"Materialize it first (it's in your project's harness/ folder)."
    )

from harness.uncertainty_pilot import PILOT_PAIRS, all_labeled_prompts
from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT
from harness.inference import _build_chat_messages
import harness.inference as inf_mod
inf_mod.SYSTEM_PROMPT = NEUTRAL_SYSTEM_PROMPT

rows = all_labeled_prompts()
print(f'Loaded {len(rows)} prompts from {len(PILOT_PAIRS)} pairs')

import torch

def score_input(prompt_text: str, layer: int = PROBE_LAYER) -> np.ndarray:
    """Single forward pass; return layer-{layer} last-input-token hidden state."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer][0, -1, :].to(torch.float32).cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return h

# Build a minimal chat-template wrapper consistent with the contradiction work
class _PromptCase:
    """Lightweight stand-in compatible with _build_chat_messages."""
    def __init__(self, question, pair_id):
        self.case_id = f"unc_pair{pair_id}"
        self.document = ""    # no long context here — uncertainty pilot is short Q-only
        self.question = question

print('\nCollecting activations for 20 prompts...')
X = []
y = []
meta = []
t0 = time.time()
for prompt, label, pid, cond in rows:
    case = _PromptCase(prompt, pid)
    messages = _build_chat_messages(case)
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

# Train a logistic-regression probe with leave-one-pair-out CV
print('\nTraining probe (leave-one-pair-out CV)...')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

groups = np.array([m["pair_id"] for m in meta])
logo = LeaveOneGroupOut()

fold_aucs = []
fold_scores = []          # per-sample held-out scores
fold_indices = []         # corresponding indices
for fold, (tr_idx, te_idx) in enumerate(logo.split(X, y, groups)):
    clf = LogisticRegression(C=1.0, max_iter=1000)
    clf.fit(X[tr_idx], y[tr_idx])
    s = clf.decision_function(X[te_idx])
    fold_scores.extend(s.tolist())
    fold_indices.extend(te_idx.tolist())
    # AUC at fold level is degenerate (2 samples, both pos+neg), so compute
    # one aggregate at the end across all held-out predictions.
print(f'  {logo.get_n_splits(X, y, groups)} folds × 2 held-out samples each')

# Reconstruct aligned arrays
all_scores = np.zeros(len(y))
for s, i in zip(fold_scores, fold_indices):
    all_scores[i] = s
auc_cv = roc_auc_score(y, all_scores)
print(f'\n[uncertainty pilot] cross-validated AUC = {auc_cv:.3f}')

# Also fit on full data for a "training-set" AUC (sanity)
clf_full = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
auc_train = roc_auc_score(y, clf_full.decision_function(X))
print(f'                     training-set AUC    = {auc_train:.3f}')

# Save results
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
print(f'\nSaved → {out_path}')

# Decision guidance
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
if auc_cv >= 0.90:
    print(f'AUC = {auc_cv:.3f}  →  STRONG signal. Expand multi-task generalization.')
    print('Next: build full-scale uncertainty paired-contrast dataset (~120 cases).')
elif auc_cv >= 0.70:
    print(f'AUC = {auc_cv:.3f}  →  MODERATE signal. Worth expanding.')
    print('Next: scale to ~60 pairs and re-test before committing fully.')
elif auc_cv >= 0.55:
    print(f'AUC = {auc_cv:.3f}  →  WEAK signal. Possibly real, possibly noise.')
    print('Next: try other layers, different probe design, or pivot to path 2 (mechanistic).')
else:
    print(f'AUC = {auc_cv:.3f}  →  NO SIGNAL. Drop multi-task plan.')
    print('Pivot to path 2: mechanistic causal analysis of contradiction circuit only.')
print('=' * 70)

# Spot-check most-confident calls
order = np.argsort(all_scores)
print('\nMost "KNOWN-looking" (lowest probe scores):')
for i in order[:3]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:90]}...')
print('\nMost "UNCERTAIN-looking" (highest probe scores):')
for i in order[-3:][::-1]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:90]}...')
