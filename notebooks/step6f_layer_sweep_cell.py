# =============================================================================
# CMB-LLM Step 6f — Layer Sweep on 60-Pair Uncertainty Dataset
# Re-collects hidden states at multiple layers in a single forward pass per
# prompt, then trains a separate LOGO-CV probe at each layer. Confirms that
# layer 17 isn't arbitrary and gives the AUC-vs-layer curve for the paper.
# Assumes model + tokenizer + the 60 pairs from step6e are in globals.
# ~3-5 min total.
# =============================================================================

import sys, os, json, time
import numpy as np
import matplotlib.pyplot as plt

if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')

OUT_DIR = '/content/drive/MyDrive/cmb_llm_intervention'
os.makedirs(OUT_DIR, exist_ok=True)

assert 'model' in globals() and 'tokenizer' in globals(), \
    "Model not loaded — run step6/6b first."
assert 'FAB_PAIRS' in globals() and 'UNK_PAIRS' in globals(), \
    "Pair lists missing — re-run step6e in this kernel first."

from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT
import torch

# Layers to sweep. Qwen2.5-7B has 28 transformer layers; hidden_states has
# 29 entries (embeddings + 28 layers).
n_hidden = model.config.num_hidden_layers
print(f'Model has {n_hidden} transformer layers')
LAYERS = [0, 4, 8, 12, 16, 17, 18, 20, 24, n_hidden - 1]
LAYERS = sorted(set([L for L in LAYERS if 0 <= L <= n_hidden]))
print(f'Sweeping layers: {LAYERS}')

# Rebuild rows (in case step6e globals were cleared)
rows = []
for pid, (k, u) in enumerate(FAB_PAIRS):
    rows.append((k, 0, pid,        'known', 'fab'))
    rows.append((u, 1, pid,    'uncertain', 'fab'))
for pid_local, (k, u) in enumerate(UNK_PAIRS):
    pid = pid_local + 30
    rows.append((k, 0, pid,        'known', 'unk'))
    rows.append((u, 1, pid,    'uncertain', 'unk'))

def collect_multilayer(prompt_text, layers):
    """One forward pass, return [n_layers, hidden_dim] last-input-token states."""
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    arr = np.stack(
        [out.hidden_states[L][0, -1, :].to(torch.float32).cpu().numpy() for L in layers],
        axis=0,
    )
    del out
    torch.cuda.empty_cache()
    return arr  # [n_layers, hidden_dim]

print('\nCollecting multi-layer activations...')
t0 = time.time()
X_all = []  # [n_prompts, n_layers, hidden_dim]
y, meta = [], []
for i, (prompt, label, pid, cond, subgrp) in enumerate(rows):
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    X_all.append(collect_multilayer(pt, LAYERS))
    y.append(label)
    meta.append({"pair_id": pid, "condition": cond, "subgrp": subgrp})
    if (i + 1) % 30 == 0:
        print(f'  {i+1}/{len(rows)} ({time.time()-t0:.0f}s)')
X_all = np.stack(X_all, axis=0)  # [120, n_layers, hidden_dim]
y = np.array(y, dtype=int)
print(f'  collected X={X_all.shape} in {time.time()-t0:.0f}s')

# Probe each layer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

groups = np.array([m["pair_id"] for m in meta])
logo = LeaveOneGroupOut()

results = []
for li, L in enumerate(LAYERS):
    X = X_all[:, li, :]
    scores = np.zeros(len(y))
    for tr_idx, te_idx in logo.split(X, y, groups):
        clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr_idx], y[tr_idx])
        scores[te_idx] = clf.decision_function(X[te_idx])
    auc = roc_auc_score(y, scores)
    # Subgroup
    sg = np.array([m["subgrp"] for m in meta])
    auc_fab = roc_auc_score(y[sg == 'fab'], scores[sg == 'fab'])
    auc_unk = roc_auc_score(y[sg == 'unk'], scores[sg == 'unk'])
    results.append({"layer": L, "auc": float(auc),
                    "auc_fab": float(auc_fab), "auc_unk": float(auc_unk)})
    print(f'  L{L:2d}: AUC={auc:.3f}  fab={auc_fab:.3f}  unk={auc_unk:.3f}')

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
layers_arr = [r["layer"] for r in results]
ax.plot(layers_arr, [r["auc"]     for r in results], 'o-', label='overall',     linewidth=2)
ax.plot(layers_arr, [r["auc_fab"] for r in results], 's--', label='fab subset', alpha=0.7)
ax.plot(layers_arr, [r["auc_unk"] for r in results], '^--', label='unk subset', alpha=0.7)
ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5, label='chance')
ax.axvline(17,  color='red',  linestyle=':', alpha=0.5, label='layer 17 (contradiction work)')
ax.set_xlabel('Layer')
ax.set_ylabel('LOGO-CV AUC')
ax.set_title('Uncertainty probe AUC across Qwen2.5-7B layers (60 pairs)')
ax.set_ylim(0.4, 1.05)
ax.legend(loc='lower right')
ax.grid(alpha=0.3)
plt.tight_layout()
plot_path = f'{OUT_DIR}/uncertainty_layer_sweep.png'
plt.savefig(plot_path, dpi=120, bbox_inches='tight')
plt.show()
print(f'\nPlot saved -> {plot_path}')

# Save
out_path = f'{OUT_DIR}/uncertainty_layer_sweep.json'
with open(out_path, 'w') as f:
    json.dump({"layers_swept": LAYERS, "n_prompts": len(y),
               "n_pairs": len(set(groups)), "results": results}, f, indent=2)
print(f'Saved -> {out_path}')

# Verdict
best = max(results, key=lambda r: r["auc"])
l17  = next((r for r in results if r["layer"] == 17), None)
print('\n' + '=' * 70)
print('LAYER SWEEP VERDICT')
print('=' * 70)
print(f'Best layer: L{best["layer"]}  AUC={best["auc"]:.3f}')
if l17:
    print(f'Layer 17:   AUC={l17["auc"]:.3f}  (gap to best: {best["auc"] - l17["auc"]:+.3f})')
if best["layer"] == 17:
    print('Layer 17 IS the best. Choice was not arbitrary.')
elif l17 and abs(best["auc"] - l17["auc"]) < 0.02:
    print('Layer 17 is within 0.02 of the best — close enough to keep.')
else:
    print(f'Layer 17 is suboptimal. Consider switching to L{best["layer"]} for the paper.')
print('=' * 70)
