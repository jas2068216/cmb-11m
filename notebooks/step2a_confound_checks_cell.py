# =============================================================================
# CMB-LLM Step 2a — confound checks for the partial-vs-detected probe
# =============================================================================
# The headline Step 2a result (AUC ~0.97) is confounded with V: detected cases
# are concentrated at V≥16k while partial cases skew toward V≤4k. A probe can
# trivially learn V from positional features and look like it's reading
# contradiction-encoding when it's actually reading input length.
#
# This cell runs two falsification tests:
#   Check 1 (V-only probe):  predict V_target ≥ 16k from same hidden states.
#                            If AUC ≈ probe result, V is doing the work.
#   Check 2 (within-V probe): restrict to V ≥ 16k cases (9 detected + 6 partial),
#                             re-train probe. If AUC stays high here, signal is
#                             real after V-controlling.
#
# Paste below the Step 2a cell. Assumes activations_v3.npz is on disk and
# probe_results from Step 2a are in memory.

import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Reload activations + meta
loaded = np.load(f'{RESULTS_DIR}/activations_v3.npz', allow_pickle=True)
all_hs = loaded['hidden_states'].astype(np.float32)   # [100, 29, 3584]
meta = json.loads(str(loaded['meta']))
n_cases, n_layers_plus, hidden_dim = all_hs.shape
print(f'Loaded activations: shape={all_hs.shape}')

# Distribution of outcomes by V_target (sanity check)
from collections import Counter
print('\nOutcome × V distribution (sanity check):')
for V_target in sorted(set(r["V_target"] for r in meta)):
    by_outcome = Counter(r["outcome"] for r in meta if r["V_target"] == V_target)
    print(f'  V={V_target:>6}: {dict(by_outcome)}')

# -----------------------------------------------------------------------------
# Check 1: V-only probe (null control). Predict V ≥ 16k vs < 16k from hidden states.
# -----------------------------------------------------------------------------
print('\n' + '=' * 60)
print('Check 1: V-only probe (null/falsification control)')
print('=' * 60)

y_V = np.array([1 if r["V_target"] >= 16000 else 0 for r in meta])
print(f'High-V (≥16k) cases: {y_V.sum()}  '
      f'Low-V (<16k) cases: {(1-y_V).sum()}')

v_only_results = []
for layer_idx in range(n_layers_plus):
    X = all_hs[:, layer_idx, :]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=17)
    fold_aucs = []
    for train_idx, test_idx in skf.split(X, y_V):
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
        clf.fit(X[train_idx], y_V[train_idx])
        proba = clf.predict_proba(X[test_idx])[:, 1]
        if len(np.unique(y_V[test_idx])) > 1:
            fold_aucs.append(roc_auc_score(y_V[test_idx], proba))
    v_only_results.append({
        "layer": layer_idx,
        "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float('nan'),
        "auc_std":  float(np.std(fold_aucs))  if fold_aucs else float('nan'),
    })

print(f'\n{"layer":>6}  {"AUC (V-only)":>14}  {"AUC (partial-vs-det)":>22}  {"delta":>8}')
print('-' * 58)
for v_only, p_vs_d in zip(v_only_results, probe_results):
    delta = p_vs_d['auc_mean'] - v_only['auc_mean']
    print(f'{v_only["layer"]:>6}  {v_only["auc_mean"]:>14.3f}  '
          f'{p_vs_d["auc_mean"]:>22.3f}  {delta:>+8.3f}')

with open(f'{RESULTS_DIR}/probe_v_only_control.json', 'w') as f:
    json.dump(v_only_results, f, indent=2)

# -----------------------------------------------------------------------------
# Check 2: within-V probe. Only V ≥ 16k cases, partial vs detected.
# -----------------------------------------------------------------------------
print('\n' + '=' * 60)
print('Check 2: within-V probe (controlled, V ≥ 16k only)')
print('=' * 60)

within_mask = np.array([
    (r["V_target"] >= 16000) and (r["outcome"] in ("partial", "detected"))
    for r in meta
])
X_within = all_hs[within_mask]
y_within = np.array([
    1 if r["outcome"] == "detected" else 0
    for r, keep in zip(meta, within_mask) if keep
])
print(f'Within-V dataset: n={X_within.shape[0]}  '
      f'detected={y_within.sum()}  partial={(1-y_within).sum()}')

# With n_pos likely ~9 and n_neg ~6, stratified 5-fold won't work cleanly.
# Use leave-one-out (LOO) which is appropriate for very small N.
within_results = []
for layer_idx in range(n_layers_plus):
    X = X_within[:, layer_idx, :]
    loo = LeaveOneOut()
    probas = np.zeros(len(y_within))
    valid = True
    for train_idx, test_idx in loo.split(X):
        # Need both classes in train
        if len(np.unique(y_within[train_idx])) < 2:
            valid = False
            break
        clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
        clf.fit(X[train_idx], y_within[train_idx])
        probas[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
    if valid and len(np.unique(y_within)) > 1:
        auc = float(roc_auc_score(y_within, probas))
    else:
        auc = float('nan')
    within_results.append({"layer": layer_idx, "auc_loo": auc})

print(f'\n{"layer":>6}  {"AUC (within-V, LOO)":>22}')
print('-' * 30)
for r in within_results:
    print(f'{r["layer"]:>6}  {r["auc_loo"]:>22.3f}')

with open(f'{RESULTS_DIR}/probe_within_v_control.json', 'w') as f:
    json.dump(within_results, f, indent=2)

# -----------------------------------------------------------------------------
# Combined plot
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw vs V-only vs within-V, all on one axis
ax = axes[0]
layers = list(range(n_layers_plus))
ax.plot(layers, [r['auc_mean'] for r in probe_results],
        marker='o', linewidth=2, label='partial vs detected (raw, n=44)')
ax.plot(layers, [r['auc_mean'] for r in v_only_results],
        marker='s', linewidth=2, label='V≥16k vs V<16k (V-only control)')
ax.plot(layers, [r['auc_loo'] for r in within_results],
        marker='^', linewidth=2, label=f'within V≥16k, partial vs detected (LOO, n={X_within.shape[0]})')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.set_xlabel('Layer index')
ax.set_ylabel('AUC')
ax.set_title('Probe AUC by layer — raw, V-only control, within-V control')
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

# Right: delta (raw - V-only). If delta is large positive, partial-vs-detected
# signal exceeds what V alone provides.
ax = axes[1]
deltas = [p['auc_mean'] - v['auc_mean']
          for p, v in zip(probe_results, v_only_results)]
ax.bar(layers, deltas, color=['green' if d > 0 else 'red' for d in deltas])
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Layer index')
ax.set_ylabel('AUC(partial-vs-det) − AUC(V-only)')
ax.set_title('Excess AUC beyond V-only baseline\n(positive = real contradiction signal beyond V)')
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{RESULTS_DIR}/probe_confound_checks.png', dpi=140, bbox_inches='tight')
fig

print('\nInterpretation guide:')
print('  V-only AUC ≈ raw AUC at all layers   -> V is doing all the work. Probe is artifact.')
print('  Excess AUC > 0 across mid-late layers -> real contradiction signal on top of V.')
print('  Within-V AUC stays > 0.7              -> signal survives V-controlling. The real result.')
print('  Within-V AUC ≈ 0.5                    -> no signal after V-controlling. Back to design.')
