# =============================================================================
# CMB-LLM Step 6d — Entity-Matched Uncertainty Control
# Both K and U reference the SAME well-known real entity (Einstein, Apple,
# Tokyo, DNA, Shakespeare). K asks a known fact about that entity; U asks
# a genuinely unknowable specific (private moments, never-recorded details).
# If AUC still > 0.70, the signal is epistemic state, not entity-familiarity.
# If AUC collapses, step6c was mostly detecting "unfamiliar string in prompt."
# Assumes model + tokenizer already in globals. Single cell, ~2 min.
# =============================================================================

import sys, os, json, time
import numpy as np

if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')

OUT_DIR = '/content/drive/MyDrive/cmb_llm_intervention'
os.makedirs(OUT_DIR, exist_ok=True)
PROBE_LAYER = 17

assert 'model' in globals() and 'tokenizer' in globals(), \
    "Model not loaded — run step6/6b first."

from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT

# Entity-matched pairs: BOTH halves reference the same real, well-known entity.
# K = known public fact. U = unknowable specific (private/never-recorded).
ENTITY_PAIRS = [
    # 1. Einstein
    ("What was the title of Einstein's most famous 1905 paper, the one introducing special relativity, that became foundational to modern physics?",
     "What did Einstein write in his private diary entry from March 17, 1905, the one describing his personal thoughts that morning, that was never published anywhere?"),
    # 2. Apple Inc.
    ("Who founded Apple Inc. in 1976, the company that built the original Apple I computer, in a garage in Los Altos California?",
     "What was the exact air temperature inside Steve Jobs's garage on the morning Apple Inc. was incorporated, the day the founders signed the papers, on April 1st 1976?"),
    # 3. Tokyo
    ("What is the approximate population of Tokyo's metropolitan area, the one comprising the 23 special wards plus surrounding cities, in recent census data?",
     "How many people in Tokyo's metropolitan area woke up before sunrise this morning, the ones living within the 23 special wards, as of the current date?"),
    # 4. DNA
    ("Which scientist co-discovered the double helix structure of DNA, the one collaborating with Francis Crick at Cambridge University, in 1953?",
     "What was Francis Crick wearing on the exact afternoon he and Watson confirmed the DNA double helix, the day they ran to the Eagle pub to celebrate, in February 1953?"),
    # 5. Shakespeare
    ("Who is the protagonist of Shakespeare's tragedy Hamlet, the one set in Denmark in the late medieval period, that explores themes of revenge and madness?",
     "What were Shakespeare's exact words to his wife Anne on the morning he left Stratford to write Hamlet, the private conversation in their kitchen, in early 1600?"),
]

# Token-length parity
print('Token-length parity check (K, U, |diff|):')
diffs = []
for i, (k, u) in enumerate(ENTITY_PAIRS):
    nk = len(tokenizer(k)["input_ids"])
    nu = len(tokenizer(u)["input_ids"])
    diffs.append(abs(nk - nu))
    print(f'  pair {i}: K={nk:3d}  U={nu:3d}  |diff|={abs(nk - nu)}')
print(f'  median |diff|={int(np.median(diffs))}  max |diff|={max(diffs)}')

rows = []
for pid, (k, u) in enumerate(ENTITY_PAIRS):
    rows.append((k, 0, pid, 'known'))
    rows.append((u, 1, pid, 'uncertain'))
print(f'\nLoaded {len(rows)} prompts from {len(ENTITY_PAIRS)} entity-matched pairs')

import torch

def score_input(prompt_text, layer=PROBE_LAYER):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer][0, -1, :].to(torch.float32).cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return h

print('\nCollecting activations...')
X, y, meta = [], [], []
t0 = time.time()
for prompt, label, pid, cond in rows:
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    h = score_input(pt, layer=PROBE_LAYER)
    X.append(h); y.append(label)
    meta.append({"pair_id": pid, "condition": cond, "prompt": prompt})

X = np.stack(X, axis=0); y = np.array(y, dtype=int)
print(f'  collected X={X.shape} in {time.time()-t0:.0f}s')

# Probe
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

groups = np.array([m["pair_id"] for m in meta])
logo = LeaveOneGroupOut()
fold_scores, fold_indices = [], []
for tr_idx, te_idx in logo.split(X, y, groups):
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr_idx], y[tr_idx])
    s = clf.decision_function(X[te_idx])
    fold_scores.extend(s.tolist()); fold_indices.extend(te_idx.tolist())

all_scores = np.zeros(len(y))
for s, i in zip(fold_scores, fold_indices):
    all_scores[i] = s
auc_cv = roc_auc_score(y, all_scores)
auc_train = roc_auc_score(y, LogisticRegression(C=1.0, max_iter=1000).fit(X, y).decision_function(X))

print(f'\n[entity-matched control] cross-validated AUC = {auc_cv:.3f}')
print(f'                          training-set AUC    = {auc_train:.3f}')

# Compare to step6b and step6c
comparisons = []
for name, path in [
    ("step6b (unmatched)",        f'{OUT_DIR}/uncertainty_pilot_results.json'),
    ("step6c (structure-matched)", f'{OUT_DIR}/uncertainty_matched_results.json'),
]:
    if os.path.exists(path):
        with open(path) as f:
            prev = json.load(f).get("auc_cv")
        comparisons.append((name, prev))
        print(f'  {name} AUC was {prev:.3f}')

print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
if auc_cv >= 0.85:
    print(f'AUC = {auc_cv:.3f}  ->  Signal survives entity-matching too. Genuine epistemic state.')
    print('Next: scale to 60 pairs with mixed stratification (fabricated + entity-unknowable).')
elif auc_cv >= 0.65:
    print(f'AUC = {auc_cv:.3f}  ->  Partial signal. Epistemic component real but entity-familiarity contributes.')
    print('Next: scale stratified, expect harder regime than step6c.')
elif auc_cv >= 0.55:
    print(f'AUC = {auc_cv:.3f}  ->  Weak. Most of step6c was likely entity-familiarity.')
    print('Reframe paper: signal is "entity novelty detection" not "epistemic uncertainty."')
else:
    print(f'AUC = {auc_cv:.3f}  ->  No signal here. Step6c was entity-familiarity confound.')
    print('Reframe paper: limited to entity-novelty; do NOT claim general uncertainty.')
print('=' * 70)
print('NOTE: n=10 here. AUC has wide CI. Use as directional, not definitive.')
print('=' * 70)

# Save
out_path = f'{OUT_DIR}/uncertainty_entity_matched_results.json'
with open(out_path, 'w') as f:
    json.dump({
        "n_prompts": len(y), "n_pairs": len(ENTITY_PAIRS),
        "probe_layer": PROBE_LAYER,
        "auc_cv": float(auc_cv), "auc_train": float(auc_train),
        "step6b_auc_cv": next((v for n, v in comparisons if "unmatched" in n), None),
        "step6c_auc_cv": next((v for n, v in comparisons if "structure" in n), None),
        "token_length_diffs": diffs,
        "per_sample": [{**m, "score": float(all_scores[i])} for i, m in enumerate(meta)],
    }, f, indent=2)
print(f'\nSaved -> {out_path}')

# Spot-check
order = np.argsort(all_scores)
print('\nMost "KNOWN-looking":')
for i in order[:3]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:110]}...')
print('\nMost "UNCERTAIN-looking":')
for i in order[-3:][::-1]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:110]}...')
