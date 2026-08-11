# =============================================================================
# CMB-LLM Step 7f - RAG-Faithfulness Probe Pilot (mode 4)
# 24 prompts (12 paired SUPPORTED/UNSUPPORTED). Trains a linear probe on layer 17
# last-input-token hidden states. Contrast is structure-matched by design (only
# the answer-bearing sentence differs), so a high AUC is not a length artifact.
#
# AUC > 0.7 (leave-one-pair-out CV) = green light: the model internally registers
#   whether the answer is present in the provided context. Next: measure the gap
#   (on UNSUPPORTED, does it decline or answer from parametric memory anyway?).
#
# RUN ORDER in a fresh Colab session:
#   1. your pipeline_bootstrap_cell.py
#   2. step7e_writefile_rag_faithfulness_cell.py
#   3. THIS cell                              (~2-4 min incl. model load on A100)
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

# Document framing is exactly right for RAG-faithfulness (there IS a document).
from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT

if 'model' not in globals() or 'tokenizer' not in globals():
    from harness.inference import load_model
    print('Loading model...')
    model, tokenizer = load_model('Qwen/Qwen2.5-7B-Instruct', load_in_4bit=False)

pilot_path = '/content/cmb_llm/harness/rag_faithfulness_pilot.py'
if not os.path.exists(pilot_path):
    raise FileNotFoundError(
        f"rag_faithfulness_pilot.py not found at {pilot_path}. "
        f"Run step7e_writefile_rag_faithfulness_cell.py first."
    )

from harness.rag_faithfulness_pilot import PILOT_PAIRS, all_labeled_prompts

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

def build_prompt_text(user_msg: str) -> str:
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user",   "content": user_msg},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )

print('\nCollecting activations for 24 prompts...')
X, y, meta = [], [], []
t0 = time.time()
for prompt, label, pid, cond in rows:
    h = score_input(build_prompt_text(prompt), layer=PROBE_LAYER)
    X.append(h); y.append(label)
    meta.append({"pair_id": pid, "condition": cond, "prompt": prompt})

X = np.stack(X, axis=0)
y = np.array(y, dtype=int)
print(f'  collected X={X.shape}  y_pos={int(y.sum())}/{len(y)}  in {time.time()-t0:.0f}s')

print('\nTraining probe (leave-one-pair-out CV)...')
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

groups = np.array([m["pair_id"] for m in meta])
logo = LeaveOneGroupOut()
fold_scores, fold_indices = [], []
for tr_idx, te_idx in logo.split(X, y, groups):
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
print(f'\n[rag-faithfulness pilot] cross-validated AUC = {auc_cv:.3f}')

clf_full = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
auc_train = roc_auc_score(y, clf_full.decision_function(X))
print(f'                         training-set AUC    = {auc_train:.3f}')

out_path = f'{OUT_DIR}/rag_faithfulness_pilot_results.json'
with open(out_path, 'w') as f:
    json.dump({
        "n_prompts": int(len(y)), "n_pairs": int(len(set(groups))),
        "probe_layer": int(PROBE_LAYER), "system_prompt": NEUTRAL_SYSTEM_PROMPT,
        "auc_cv": float(auc_cv), "auc_train": float(auc_train),
        "per_sample": [{**m, "score": float(all_scores[i])} for i, m in enumerate(meta)],
    }, f, indent=2)
print(f'\nSaved -> {out_path}')

print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
if auc_cv >= 0.90:
    print(f'AUC = {auc_cv:.3f}  ->  STRONG signal. The model registers answer-in-context.')
    print('Next: measure the gap -- on UNSUPPORTED, does it decline or answer anyway?')
elif auc_cv >= 0.70:
    print(f'AUC = {auc_cv:.3f}  ->  MODERATE signal. Worth taking to the generation gap.')
elif auc_cv >= 0.55:
    print(f'AUC = {auc_cv:.3f}  ->  WEAK signal. Try a layer sweep before committing.')
else:
    print(f'AUC = {auc_cv:.3f}  ->  NO SIGNAL at layer 17. Sweep layers before dropping.')
print('=' * 70)

order = np.argsort(all_scores)
print('\nMost "SUPPORTED-looking" (lowest scores):')
for i in order[:3]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
print('\nMost "UNSUPPORTED-looking" (highest scores):')
for i in order[-3:][::-1]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
