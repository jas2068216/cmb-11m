# =============================================================================
# CMB-LLM Step 2a V4 — distance_kind null probe
# =============================================================================
# Tests whether the layer-2 peak in the within-V probe is real contradiction-
# encoding or just distance_kind structure.
#
# Setup: same exact case selection as the within-V probe (V≥16k, partial or
# detected only), but with the label switched from outcome to distance_kind
# (short=0, long=1). If THIS probe scores ~0.85 at layer 2, the within-V
# probe was reading distance_kind, not contradiction. If it scores ~0.55-0.65,
# the within-V signal is real and beyond distance_kind.
#
# Paste below the V4 cell. ~30 seconds, no model call.

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Load V4 activations
loaded = np.load(f'{RESULTS_DIR}/activations_v4.npz', allow_pickle=True)
all_hs_v4 = loaded['hidden_states'].astype(np.float32)
meta_v4 = json.loads(str(loaded['meta']))
print(f'Loaded V4 activations: shape={all_hs_v4.shape}')

# Load judgments_v4 from disk (in case kernel restarted)
with open(f'{RESULTS_DIR}/judgments_v4.json') as f:
    judg_v4 = json.load(f)
outcome_by_id = {j["case_id"]: j["outcome"] for j in judg_v4}

# Match V4 activations to outcomes and filter to partial+detected at V≥16k
# (same selection as the within-V probe — only the LABEL changes).
within_mask = []
distance_kind_labels = []   # 0 = short, 1 = long
outcome_labels = []         # 0 = partial, 1 = detected
X_list = []
for rec in meta_v4:
    cid = rec["case_id"]
    V_target = rec["V_target"]
    dk = rec["distance_kind"]
    outcome = outcome_by_id.get(cid)
    keep = (V_target >= 16000) and (outcome in ("partial", "detected"))
    within_mask.append(keep)
    if keep:
        distance_kind_labels.append(1 if dk == "long" else 0)
        outcome_labels.append(1 if outcome == "detected" else 0)

within_mask = np.array(within_mask)
distance_kind_labels = np.array(distance_kind_labels)
outcome_labels = np.array(outcome_labels)

X_within = all_hs_v4[within_mask].astype(np.float32)
n_layers_plus = X_within.shape[1]

print(f'\nWithin-V subset (V≥16k, partial+detected): n={X_within.shape[0]}')
print(f'  distance_kind: short={int((1-distance_kind_labels).sum())}  '
      f'long={int(distance_kind_labels.sum())}')
print(f'  outcome:       partial={int((1-outcome_labels).sum())}  '
      f'detected={int(outcome_labels.sum())}')

# Cross-tab to see if distance_kind and outcome are correlated within this subset
from collections import Counter
print('\n  outcome × distance_kind crosstab:')
ct = Counter()
for o, d in zip(outcome_labels, distance_kind_labels):
    ct[(o, d)] += 1
print(f'    partial   × short: {ct[(0,0)]}')
print(f'    partial   × long:  {ct[(0,1)]}')
print(f'    detected  × short: {ct[(1,0)]}')
print(f'    detected  × long:  {ct[(1,1)]}')

# -----------------------------------------------------------------------------
# distance_kind probe (null control)
# -----------------------------------------------------------------------------
print('\n' + '=' * 60)
print('Null probe: predict distance_kind from same hidden states')
print('=' * 60)

dk_results = []
n_min_class = min(int(distance_kind_labels.sum()),
                  int((1 - distance_kind_labels).sum()))
n_splits = 5 if n_min_class >= 5 else 3
print(f'Using {n_splits}-fold stratified CV (min class size = {n_min_class})')

for layer_idx in range(n_layers_plus):
    X = X_within[:, layer_idx, :]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=23)
    fold_aucs = []
    for tr, te in skf.split(X, distance_kind_labels):
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
        clf.fit(X[tr], distance_kind_labels[tr])
        proba = clf.predict_proba(X[te])[:, 1]
        if len(np.unique(distance_kind_labels[te])) > 1:
            fold_aucs.append(roc_auc_score(distance_kind_labels[te], proba))
    dk_results.append({
        "layer": layer_idx,
        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float('nan'),
        "auc_std":  float(np.std(fold_aucs))  if fold_aucs else float('nan'),
    })

# Load partial-vs-detected V4 results for side-by-side
with open(f'{RESULTS_DIR}/probe_within_v_v4.json') as f:
    pv_v4 = json.load(f)
pv_lookup = {r["layer"]: r["auc_mean"] for r in pv_v4}

print(f'\n{"layer":>6}  {"distance_kind AUC":>18}  {"partial-vs-det AUC":>22}  {"delta":>8}')
print('-' * 60)
for r in dk_results:
    pv = pv_lookup.get(r["layer"], float('nan'))
    delta = pv - r["auc_mean"]
    print(f'{r["layer"]:>6}  {r["auc_mean"]:>18.3f}  {pv:>22.3f}  {delta:>+8.3f}')

with open(f'{RESULTS_DIR}/probe_distance_kind_null.json', 'w') as f:
    json.dump(dk_results, f, indent=2)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
layers = [r["layer"] for r in dk_results]
ax.plot(layers, [r["auc_mean"] for r in dk_results],
        marker='s', linewidth=2, label='distance_kind probe (null)')
ax.plot(layers, [pv_lookup.get(l, np.nan) for l in layers],
        marker='o', linewidth=2, label='partial vs detected probe')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.axhline(0.7, color='green', linestyle=':', alpha=0.4)
ax.set_xlabel('Layer index')
ax.set_ylabel('AUC')
ax.set_title('Within-V probes — partial-vs-detected vs distance_kind null')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')

ax = axes[1]
deltas = [pv_lookup.get(r["layer"], np.nan) - r["auc_mean"] for r in dk_results]
ax.bar(layers, deltas, color=['green' if (not np.isnan(d) and d > 0)
                              else 'red' for d in deltas])
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Layer index')
ax.set_ylabel('AUC(partial-vs-det) − AUC(distance_kind)')
ax.set_title('Excess AUC beyond distance_kind null\n'
             '(positive = contradiction signal beyond distance structure)')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{RESULTS_DIR}/probe_distance_control.png',
            dpi=140, bbox_inches='tight')
fig

print('\nReading guide:')
print('  distance_kind AUC ≈ partial-vs-det AUC at layer 2  -> peak is distance, not contradiction')
print('  distance_kind AUC stays around 0.55-0.65            -> partial-vs-det signal is real')
print('  Excess AUC > 0.15 at any layer                      -> signal beyond distance there')
