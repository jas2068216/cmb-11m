# =============================================================================
# CMB-LLM Step 2a — linear probes on hidden activations
# =============================================================================
# Goal: train per-layer probes to predict outcome (partial vs detected) from
# Qwen2.5-7B's hidden states at the last input-token position. If a probe in
# any layer scores AUC well above 0.5, R-restoration is achievable at decoding
# time — the model already encodes the contradiction internally, it just isn't
# surfacing it.
#
# Paste below the V3 cell in your Colab notebook. Assumes:
#   - model + tokenizer are loaded (cells 4-5 of the original notebook)
#   - cases_v3 and judgments_v3 are in memory (from the V3 cell)
#   - RESULTS_DIR is set
#
# Runtime: ~10-15 min on A100 (re-runs forward pass on 100 cases with hidden
# states enabled). Memory peak ~25 GB.

import torch
import numpy as np
import json
import time
from pathlib import Path
from collections import Counter

from harness.dataset import TestCase
from harness.inference import _build_chat_messages

# -----------------------------------------------------------------------------
# Step 2a.1 — Capture hidden states at last input token for every case
# -----------------------------------------------------------------------------
# We only need the hidden state at ONE position per case (the last input token),
# but at EVERY layer. Qwen2.5-7B has 28 transformer layers + the embedding layer.
# Storage: 100 cases × 29 layers × 3584 hidden_dim × 2 bytes (fp16) ≈ 20 MB. Trivial.

def capture_hidden_states(model, tokenizer, case):
    """Run forward pass, grab last-token hidden state from every layer."""
    messages = _build_chat_messages(case)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    # outputs.hidden_states is a tuple of (n_layers+1) tensors of shape
    # [batch, seq_len, hidden_dim]. Index 0 = embedding output, then one per layer.
    # We want the LAST position from each.
    last_pos_hs = torch.stack([
        h[0, -1, :].to(torch.float32).cpu() for h in outputs.hidden_states
    ])  # shape: [n_layers+1, hidden_dim]
    return last_pos_hs.numpy()


print('Capturing hidden states from all 100 cases...')
t0 = time.time()
case_by_id   = {c.case_id: c for c in cases_v3}
judg_by_id   = {j.case_id: j for j in judgments_v3}

hs_records = []   # list of dicts: case_id, outcome, V_target, distance_kind, hidden_states
for i, case in enumerate(cases_v3):
    j = judg_by_id.get(case.case_id)
    if j is None:
        continue
    hs = capture_hidden_states(model, tokenizer, case)
    hs_records.append({
        "case_id": case.case_id,
        "outcome": j.outcome,
        "V_target": case.V_target,
        "distance_kind": case.distance_kind,
        "hidden_states": hs,        # shape [n_layers+1, hidden_dim]
    })
    if (i + 1) % 10 == 0:
        elapsed = time.time() - t0
        print(f'  [{i+1}/100]  elapsed={elapsed:.0f}s')

print(f'Captured {len(hs_records)} hidden-state records in {time.time()-t0:.0f}s')

# Save activations (compact: float16, single npz)
n_layers_plus = hs_records[0]["hidden_states"].shape[0]
hidden_dim    = hs_records[0]["hidden_states"].shape[1]
print(f'Layers (including embedding): {n_layers_plus}  Hidden dim: {hidden_dim}')

all_hs = np.stack([r["hidden_states"] for r in hs_records]).astype(np.float16)
meta = [{k: r[k] for k in ("case_id", "outcome", "V_target", "distance_kind")}
        for r in hs_records]
np.savez_compressed(
    f'{RESULTS_DIR}/activations_v3.npz',
    hidden_states=all_hs,
    meta=json.dumps(meta),
)
print(f'Saved activations to {RESULTS_DIR}/activations_v3.npz  '
      f'shape={all_hs.shape}  size~{all_hs.nbytes/1e6:.1f}MB')

# -----------------------------------------------------------------------------
# Step 2a.2 — Train per-layer linear probes (partial vs detected)
# -----------------------------------------------------------------------------
# Binary task: predict outcome ∈ {partial, detected}. We exclude 'missed' and
# 'ambiguous' from the primary probe because:
#   - missed cases don't have year_first in their output; comparing them to
#     detected confounds "did the model retain the info" with "did the model
#     surface the conflict." That's Step 2b (separate probe).
#   - ambiguous count = 0 in V3, nothing to do.
#
# Class balance: ~35 partial vs ~9 detected. Use class_weight='balanced' and
# 5-fold stratified CV with AUC as the headline metric (robust to imbalance).

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Pull (X, y) from records
mask = np.array([r["outcome"] in ("partial", "detected") for r in hs_records])
X_all = all_hs[mask].astype(np.float32)              # [n_cases, n_layers+1, hidden_dim]
y_all = np.array([1 if r["outcome"] == "detected" else 0
                  for r, keep in zip(hs_records, mask) if keep])
print(f'\nProbe dataset: {X_all.shape[0]} cases  '
      f'(detected={y_all.sum()}, partial={(1-y_all).sum()})')

# Stratified 5-fold (or 3-fold if too few detected) CV per layer
n_pos = int(y_all.sum())
n_folds = 5 if n_pos >= 5 else 3
print(f'Using {n_folds}-fold stratified CV (n_positive={n_pos})')

probe_results = []
for layer_idx in range(n_layers_plus):
    X_layer = X_all[:, layer_idx, :]   # [n_cases, hidden_dim]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=17)
    fold_aucs = []
    for train_idx, test_idx in skf.split(X_layer, y_all):
        clf = LogisticRegression(
            class_weight='balanced',
            max_iter=2000,
            C=1.0,
            solver='lbfgs',
        )
        clf.fit(X_layer[train_idx], y_all[train_idx])
        proba = clf.predict_proba(X_layer[test_idx])[:, 1]
        # AUC only well-defined if both classes present in test split
        if len(np.unique(y_all[test_idx])) > 1:
            fold_aucs.append(roc_auc_score(y_all[test_idx], proba))

    mean_auc = float(np.mean(fold_aucs)) if fold_aucs else float('nan')
    std_auc  = float(np.std(fold_aucs))  if fold_aucs else float('nan')
    probe_results.append({
        "layer": layer_idx,
        "n_folds_valid": len(fold_aucs),
        "auc_mean": mean_auc,
        "auc_std": std_auc,
    })

# Print and save
print(f'\n{"layer":>6}  {"n_folds":>8}  {"AUC mean":>10}  {"AUC std":>10}')
print('-' * 40)
for r in probe_results:
    print(f'{r["layer"]:>6}  {r["n_folds_valid"]:>8}  '
          f'{r["auc_mean"]:>10.3f}  {r["auc_std"]:>10.3f}')

with open(f'{RESULTS_DIR}/probe_results_partial_vs_detected.json', 'w') as f:
    json.dump(probe_results, f, indent=2)

# -----------------------------------------------------------------------------
# Step 2a.3 — Plot AUC by layer
# -----------------------------------------------------------------------------
import matplotlib.pyplot as plt

layers = [r["layer"] for r in probe_results]
aucs   = [r["auc_mean"] for r in probe_results]
stds   = [r["auc_std"] for r in probe_results]

fig, ax = plt.subplots(figsize=(10, 5))
ax.errorbar(layers, aucs, yerr=stds, marker='o', capsize=4, linewidth=1.5,
            label='partial vs detected probe')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.6, label='chance (0.5)')
ax.axhline(0.7, color='green', linestyle=':', alpha=0.5,
           label='well above chance (0.7)')
ax.set_xlabel('Layer index (0 = embedding, 1..28 = transformer layers)')
ax.set_ylabel(f'CV AUC ({n_folds}-fold stratified)')
ax.set_title(f'{MODEL_NAME} — partial vs detected probe by layer\n'
             f'n={X_all.shape[0]} cases  '
             f'(detected={y_all.sum()}, partial={(1-y_all).sum()})')
ax.set_ylim(0.0, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(f'{RESULTS_DIR}/probe_auc_by_layer.png', dpi=140, bbox_inches='tight')
fig

print('\nReading guide:')
print('  AUC > 0.7 in any layer  -> model "knows" the conflict pre-output.')
print('                              R-restoration is decoding-time fixable.')
print('  AUC ~ 0.5 across layers -> no internal signal. Either:')
print('                              (a) probes can\'t capture the structure')
print('                              (b) the model genuinely doesn\'t track it')
print('  Peak in mid-to-late     -> typical: contradiction encoding emerges')
print('    layers (15-25 ish)      after semantic integration, before output.')
