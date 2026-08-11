# =============================================================================
# CMB-LLM Step 6c — Uncertainty Pilot, Length/Structure-Matched Control
# Each pair has IDENTICAL surface structure (noun + embedded relative clause +
# modifier). Only the embedded presupposition flips: TRUE (known) vs FALSE
# (uncertain). If the probe still separates at AUC >> 0.7, the layer-17 signal
# is epistemic, not surface.
# Assumes model + tokenizer already in globals from step6b. Single cell, ~3 min.
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

# Matched pairs: same noun-phrase template, same modifier clause structure.
# K = known (true presupposition), U = uncertain (false presupposition).
MATCHED_PAIRS = [
    # 1. Einstein
    ("What was the title of Einstein's 1905 paper, the one introducing special relativity, that revolutionized physics?",
     "What was the title of Einstein's 1905 paper, the one introducing quantum gravity theory, that revolutionized physics?"),
    # 2. Leonardo
    ("What is the name of Leonardo da Vinci's portrait, the one painted around 1503 in Florence, that hangs in the Louvre?",
     "What is the name of Leonardo da Vinci's portrait, the one Pope Leo X commissioned in 1402 in Rome, that hangs in the Louvre?"),
    # 3. Apple
    ("Who was the co-founder of Apple Inc., the one who left the company early and sold his stake, often forgotten in the founding story?",
     "Who was the fourth co-founder of Apple Inc., the one Jobs met at a Cupertino synagogue in 1974, often forgotten in the founding story?"),
    # 4. Marie Curie
    ("What scientific field did Marie Curie win her second Nobel Prize in, the one she received in 1911 alone, while working in Paris?",
     "What scientific field did Marie Curie win her third Nobel Prize in, the one she received in 1923 alone, while working in Paris?"),
    # 5. Tokyo
    ("What is the approximate population of Tokyo's metropolitan area, the one including the 23 special wards, as measured in recent census data?",
     "What is the approximate population of Tokyo's underground district Shibuya-Kita, the one beneath the Hakuro Tunnel system, as measured in recent census data?"),
    # 6. Beethoven
    ("What is the name of Beethoven's ninth symphony, the one premiering in Vienna in 1824, that incorporates a choral finale?",
     "What is the name of Beethoven's tenth symphony, the one premiering in Vienna in 1828, that incorporates a choral finale?"),
    # 7. Olympic Games
    ("In what year were the first modern Olympic Games held, the ones organized by Pierre de Coubertin's committee, that took place in Athens?",
     "In what year were the second secret modern Olympic Games held, the ones organized by Pierre de Coubertin's brother, that took place in Athens?"),
    # 8. DNA
    ("Which scientist co-discovered the double helix structure of DNA, the one collaborating with Francis Crick at Cambridge, in 1953?",
     "Which scientist co-discovered the triple helix structure of DNA, the one collaborating with Francis Crick at Cambridge, in 1949?"),
    # 9. Shakespeare
    ("Who is the protagonist of Shakespeare's play Hamlet, the one set in Denmark in the late medieval period, that explores themes of revenge?",
     "Who is the protagonist of Shakespeare's lost play Cardenio Part Two, the one set in Denmark in the late medieval period, that explores themes of revenge?"),
    # 10. SpaceX
    ("What is the name of the SpaceX rocket, the one used for crewed missions to the ISS, that first launched astronauts in 2020?",
     "What is the name of the SpaceX rocket, the one used for the secret 2018 Mars colonization mission, that first launched astronauts in 2018?"),
]

# Sanity: report token-length parity per pair
print('Token-length parity check (K, U, |diff|):')
diffs = []
for i, (k, u) in enumerate(MATCHED_PAIRS):
    nk = len(tokenizer(k)["input_ids"])
    nu = len(tokenizer(u)["input_ids"])
    diffs.append(abs(nk - nu))
    print(f'  pair {i}: K={nk:3d}  U={nu:3d}  |diff|={abs(nk - nu)}')
print(f'  median |diff|={int(np.median(diffs))}  max |diff|={max(diffs)}')

# Build labeled rows in same shape as before
rows = []
for pid, (k, u) in enumerate(MATCHED_PAIRS):
    rows.append((k, 0, pid, 'known'))
    rows.append((u, 1, pid, 'uncertain'))

print(f'\nLoaded {len(rows)} prompts from {len(MATCHED_PAIRS)} matched pairs')

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

# Probe: LogReg, leave-one-pair-out
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

print(f'\n[matched control] cross-validated AUC = {auc_cv:.3f}')
print(f'                  training-set AUC    = {auc_train:.3f}')

# Compare to step6b
prev_path = f'{OUT_DIR}/uncertainty_pilot_results.json'
prev_auc = None
if os.path.exists(prev_path):
    with open(prev_path) as f:
        prev_auc = json.load(f).get("auc_cv")
    print(f'\nStep6b (unmatched) AUC was {prev_auc:.3f}')

print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
if auc_cv >= 0.85:
    print(f'AUC = {auc_cv:.3f}  ->  Signal SURVIVES structure-matching. Real epistemic feature.')
    print('Next: scale to 60 pairs with same matched-structure design.')
elif auc_cv >= 0.65:
    print(f'AUC = {auc_cv:.3f}  ->  Partial signal. Some surface contribution but epistemic component likely real.')
    print('Next: examine fold-level errors before scaling.')
elif auc_cv >= 0.55:
    print(f'AUC = {auc_cv:.3f}  ->  Weak/ambiguous. Surface confound probably explained most of step6b.')
    print('Next: probe other layers or pivot.')
else:
    print(f'AUC = {auc_cv:.3f}  ->  No epistemic signal. Step6b was surface confound.')
    print('Drop multi-task expansion plan; revisit single-task contradiction story.')
print('=' * 70)

# Save
out_path = f'{OUT_DIR}/uncertainty_matched_results.json'
with open(out_path, 'w') as f:
    json.dump({
        "n_prompts": len(y), "n_pairs": len(MATCHED_PAIRS),
        "probe_layer": PROBE_LAYER,
        "auc_cv": float(auc_cv), "auc_train": float(auc_train),
        "step6b_auc_cv": prev_auc,
        "token_length_diffs": diffs,
        "per_sample": [{**m, "score": float(all_scores[i])} for i, m in enumerate(meta)],
    }, f, indent=2)
print(f'\nSaved -> {out_path}')

# Spot-check
order = np.argsort(all_scores)
print('\nMost "KNOWN-looking":')
for i in order[:3]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:100]}...')
print('\nMost "UNCERTAIN-looking":')
for i in order[-3:][::-1]:
    print(f'  score={all_scores[i]:+.2f}  cond={meta[i]["condition"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:100]}...')
