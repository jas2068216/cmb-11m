# =============================================================================
# CMB-LLM Step 2a V4 — entity-identity null probe
# =============================================================================
# Tests whether the layer-2 partial-vs-detected signal is entity-driven.
#
# Setup: same case selection as within-V probe (V≥16k, partial+detected, n=43).
# Label switched from outcome to entity name (10-class). If entity is highly
# decodable at layer 2 AND outcomes are unevenly distributed across entities,
# the layer-2 signal is likely entity-driven (not contradiction-encoding).
#
# Paste below the distance-kind control cell. ~30 seconds, no model call.

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Reload V4 activations and outcomes
loaded = np.load(f'{RESULTS_DIR}/activations_v4.npz', allow_pickle=True)
all_hs_v4 = loaded['hidden_states'].astype(np.float32)
meta_v4 = json.loads(str(loaded['meta']))

with open(f'{RESULTS_DIR}/judgments_v4.json') as f:
    judg_v4 = json.load(f)
outcome_by_id = {j["case_id"]: j["outcome"] for j in judg_v4}

# Need entity name per case — pull from dataset_v4.json
with open(f'{RESULTS_DIR}/dataset_v4.json') as f:
    cases_v4_records = json.load(f)
entity_by_id = {c["case_id"]: c["entity_name"] for c in cases_v4_records}

# Build within-V subset
within_mask = []
X_list = []
y_outcome = []
y_entity = []
for rec in meta_v4:
    cid = rec["case_id"]
    V_target = rec["V_target"]
    outcome = outcome_by_id.get(cid)
    entity  = entity_by_id.get(cid)
    keep = (V_target >= 16000) and (outcome in ("partial", "detected")) and (entity is not None)
    within_mask.append(keep)
    if keep:
        y_outcome.append(1 if outcome == "detected" else 0)
        y_entity.append(entity)

within_mask = np.array(within_mask)
X_within = all_hs_v4[within_mask].astype(np.float32)
y_outcome = np.array(y_outcome)
n_layers_plus = X_within.shape[1]
print(f'Within-V subset: n={X_within.shape[0]}  '
      f'detected={int(y_outcome.sum())} partial={int((1-y_outcome).sum())}')

# Entity distribution + outcome breakdown
entity_counter = Counter(y_entity)
print(f'\nEntity distribution in within-V sample (n={len(y_entity)} entities, '
      f'{len(set(y_entity))} unique):')
print(f'{"entity":>22}  {"n":>4}  {"n_detected":>10}  {"n_partial":>10}  {"det_rate":>10}')
print('-' * 64)
for e, n in sorted(entity_counter.items(), key=lambda x: -x[1]):
    idxs = [i for i, ee in enumerate(y_entity) if ee == e]
    n_det = int(sum(y_outcome[i] for i in idxs))
    n_par = n - n_det
    rate = n_det / n if n > 0 else 0
    print(f'{e:>22}  {n:>4}  {n_det:>10}  {n_par:>10}  {rate:>10.2f}')

# Encode entity labels to integers
unique_entities = sorted(set(y_entity))
entity_to_id = {e: i for i, e in enumerate(unique_entities)}
y_entity_int = np.array([entity_to_id[e] for e in y_entity])
n_entities = len(unique_entities)
print(f'\n{n_entities} unique entities in subset')

# -----------------------------------------------------------------------------
# Multi-class entity probe (one-vs-rest macro AUC)
# -----------------------------------------------------------------------------
print('\n' + '=' * 60)
print('Null probe: predict entity identity from same hidden states')
print('=' * 60)

# With n=43 and 10 classes, some entities have very few samples. 3-fold to be safe.
# Macro-AUC requires at least 2 classes in each test fold for each one-vs-rest task.
# We'll compute per-fold per-entity AUC and macro-average.

min_class_count = min(entity_counter.values())
n_splits = 3 if min_class_count < 5 else 5
print(f'Using {n_splits}-fold stratified CV (min entity count = {min_class_count})')

entity_results = []
for layer_idx in range(n_layers_plus):
    X = X_within[:, layer_idx, :]
    # We'll use one-vs-rest with LogisticRegression(multi_class='ovr')
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=23)
    fold_macro_aucs = []
    for tr, te in skf.split(X, y_entity_int):
        try:
            clf = LogisticRegression(
                multi_class='ovr',
                class_weight='balanced',
                max_iter=2000, C=1.0, solver='lbfgs',
            )
            clf.fit(X[tr], y_entity_int[tr])
            proba = clf.predict_proba(X[te])
            # Compute macro AUC across entities present in test set
            test_entities = np.unique(y_entity_int[te])
            entity_aucs = []
            for e_id in test_entities:
                # Need this entity present in train AND test
                if e_id not in clf.classes_:
                    continue
                col = list(clf.classes_).index(e_id)
                y_te_bin = (y_entity_int[te] == e_id).astype(int)
                if len(np.unique(y_te_bin)) < 2:
                    continue
                entity_aucs.append(roc_auc_score(y_te_bin, proba[:, col]))
            if entity_aucs:
                fold_macro_aucs.append(float(np.mean(entity_aucs)))
        except Exception as e:
            continue
    entity_results.append({
        "layer": layer_idx,
        "macro_auc_mean": float(np.mean(fold_macro_aucs)) if fold_macro_aucs else float('nan'),
        "macro_auc_std":  float(np.std(fold_macro_aucs))  if fold_macro_aucs else float('nan'),
    })

# Load partial-vs-detected V4 results for side-by-side
with open(f'{RESULTS_DIR}/probe_within_v_v4.json') as f:
    pv_v4 = json.load(f)
pv_lookup = {r["layer"]: r["auc_mean"] for r in pv_v4}

# Load distance_kind probe results too
with open(f'{RESULTS_DIR}/probe_distance_kind_null.json') as f:
    dk_v4 = json.load(f)
dk_lookup = {r["layer"]: r["auc_mean"] for r in dk_v4}

print(f'\n{"layer":>6}  {"entity macro-AUC":>17}  '
      f'{"distance AUC":>13}  {"partial-vs-det":>15}  {"excess-over-entity":>20}')
print('-' * 80)
for r in entity_results:
    pv = pv_lookup.get(r["layer"], float('nan'))
    dk = dk_lookup.get(r["layer"], float('nan'))
    excess = pv - r["macro_auc_mean"]
    print(f'{r["layer"]:>6}  {r["macro_auc_mean"]:>17.3f}  '
          f'{dk:>13.3f}  {pv:>15.3f}  {excess:>+20.3f}')

with open(f'{RESULTS_DIR}/probe_entity_null.json', 'w') as f:
    json.dump(entity_results, f, indent=2)

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

ax = axes[0]
layers = [r["layer"] for r in entity_results]
ax.plot(layers, [r["macro_auc_mean"] for r in entity_results],
        marker='D', linewidth=2, label='entity macro-AUC (null)', color='C2')
ax.plot(layers, [dk_lookup.get(l, np.nan) for l in layers],
        marker='s', linewidth=2, label='distance_kind AUC (null)', color='C1')
ax.plot(layers, [pv_lookup.get(l, np.nan) for l in layers],
        marker='o', linewidth=2, label='partial vs detected', color='C0')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.set_xlabel('Layer index')
ax.set_ylabel('AUC')
ax.set_title('Within-V probes — outcome vs. distance_kind null vs. entity null')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')

ax = axes[1]
excess_entity = [pv_lookup.get(r["layer"], np.nan) - r["macro_auc_mean"]
                 for r in entity_results]
excess_distance = [pv_lookup.get(r["layer"], np.nan) - dk_lookup.get(r["layer"], np.nan)
                   for r in entity_results]
x = np.array(layers)
ax.bar(x - 0.2, excess_entity,   width=0.4, label='vs entity null',   color='C2', alpha=0.7)
ax.bar(x + 0.2, excess_distance, width=0.4, label='vs distance null', color='C1', alpha=0.7)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_xlabel('Layer index')
ax.set_ylabel('Excess AUC of partial-vs-det over null')
ax.set_title('Excess outcome signal over each null probe\n'
             '(positive = real contradiction signal beyond that confound)')
ax.grid(True, alpha=0.3)
ax.legend()

fig.tight_layout()
fig.savefig(f'{RESULTS_DIR}/probe_entity_control.png', dpi=140, bbox_inches='tight')
fig

print('\nReading guide:')
print('  entity macro-AUC ≈ partial-vs-det at layer 2  -> layer-2 signal is entity-driven.')
print('  entity macro-AUC stays around 0.5-0.65          -> outcome signal beyond entity.')
print('  Excess-over-entity > 0.15 at any layer          -> real contradiction signal there.')
