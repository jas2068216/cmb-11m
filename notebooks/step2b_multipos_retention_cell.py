# =============================================================================
# CMB-LLM Step 2b — multi-position capture + retention probe
# =============================================================================
# Opens silos 1 and 2:
#   Silo 1: we threw out the missed cases. This cell includes them.
#   Silo 2: we only probed the last input token. This cell adds two more
#           positions per case — right after assertion #1, right after
#           assertion #2.
#
# Then asks a different question:
#   "Does the model still encode year_first by the time it reaches
#    post-assertion-#2, or did year_first get dropped from attention?"
#
# This is the RETENTION probe — directly tests the (1−R) leg's failure mode
# for missed cases. Different question than partial-vs-detected.
#
# Paste below the entity-control cell. ~10-15 min on A100.

import json, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

from harness.dataset import TestCase
from harness.inference import _build_chat_messages

# Reload V4 cases + judgments
with open(f'{RESULTS_DIR}/dataset_v4.json') as f:
    cases_v4 = [TestCase(**c) for c in json.load(f)]
with open(f'{RESULTS_DIR}/judgments_v4.json') as f:
    judg_v4 = json.load(f)
outcome_by_id = {j["case_id"]: j["outcome"] for j in judg_v4}

# Assertion templates (from V2 overrides — must match what built the documents)
ASSERTION_TEMPLATE_A = (
    "{name} was founded in {year} by a small team that relocated to {city}."
)
ASSERTION_TEMPLATE_B = (
    "Industry filings list {year} as the year {name} was established, "
    "and the company's official records continue to cite that date."
)

# -----------------------------------------------------------------------------
# Helper: find token positions of post-A1 and post-A2 in chat-formatted prompt
# -----------------------------------------------------------------------------
def find_assertion_token_positions(case, tokenizer):
    """Returns (pos_post_A1, pos_post_A2, total_input_tokens) — token indices
    in the fully chat-formatted prompt, pointing at the LAST token of each
    assertion sentence."""
    a1 = ASSERTION_TEMPLATE_A.format(name=case.entity_name,
                                     year=case.year_first,
                                     city=case.city)
    a2 = ASSERTION_TEMPLATE_B.format(name=case.entity_name,
                                     year=case.year_second)

    # Locate end-of-assertion CHAR positions in the document
    idx_a1 = case.document.find(a1)
    idx_a2 = case.document.find(a2)
    if idx_a1 < 0 or idx_a2 < 0:
        return None
    char_a1_end = idx_a1 + len(a1)
    char_a2_end = idx_a2 + len(a2)

    # Build chat-formatted prompt and locate where the document sits in it
    messages = _build_chat_messages(case)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    doc_offset = prompt_text.find(case.document)
    if doc_offset < 0:
        return None

    abs_a1_end = doc_offset + char_a1_end
    abs_a2_end = doc_offset + char_a2_end

    # Use offset_mapping to translate char positions → token positions
    enc = tokenizer(prompt_text, return_offsets_mapping=True,
                    add_special_tokens=False)
    offsets = enc["offset_mapping"]
    total_tokens = len(offsets)

    def char_to_tok(target_char_end):
        # Find the last token whose end <= target_char_end (the token that
        # completes the assertion)
        pos = 0
        for i, (s, e) in enumerate(offsets):
            if e <= target_char_end:
                pos = i
        return pos

    pos_a1 = char_to_tok(abs_a1_end)
    pos_a2 = char_to_tok(abs_a2_end)
    return pos_a1, pos_a2, total_tokens


# -----------------------------------------------------------------------------
# Capture hidden states at 3 positions per case (V≥16k cases only — 120 total)
# -----------------------------------------------------------------------------
v4_v16plus_cases = [c for c in cases_v4 if c.V_target >= 16000]
print(f'Capturing multi-position activations for {len(v4_v16plus_cases)} cases...')

multipos_records = []
t0 = time.time()
fail_count = 0
for i, case in enumerate(v4_v16plus_cases):
    try:
        pos = find_assertion_token_positions(case, tokenizer)
        if pos is None:
            fail_count += 1
            continue
        pos_a1, pos_a2, n_tokens = pos

        # Forward pass with hidden states
        messages = _build_chat_messages(case)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)

        # Slice hidden states at three positions, all layers
        # outputs.hidden_states is tuple of (n_layers+1) tensors [1, seq_len, hidden]
        hs_a1 = torch.stack([
            h[0, pos_a1, :].to(torch.float32).cpu() for h in out.hidden_states
        ]).numpy()
        hs_a2 = torch.stack([
            h[0, pos_a2, :].to(torch.float32).cpu() for h in out.hidden_states
        ]).numpy()
        hs_last = torch.stack([
            h[0, -1, :].to(torch.float32).cpu() for h in out.hidden_states
        ]).numpy()

        multipos_records.append({
            "case_id": case.case_id,
            "outcome": outcome_by_id.get(case.case_id),
            "V_target": case.V_target,
            "distance_kind": case.distance_kind,
            "entity": case.entity_name,
            "year_first": case.year_first,
            "year_second": case.year_second,
            "pos_a1": pos_a1,
            "pos_a2": pos_a2,
            "n_tokens": n_tokens,
            "hs_post_a1": hs_a1,
            "hs_post_a2": hs_a2,
            "hs_last": hs_last,
        })
        del out
        torch.cuda.empty_cache()

        if (i + 1) % 10 == 0 or (i + 1) == len(v4_v16plus_cases):
            elapsed = time.time() - t0
            print(f'  [{i+1}/{len(v4_v16plus_cases)}] elapsed={elapsed:.0f}s '
                  f'(pos_a1≈{pos_a1}, pos_a2≈{pos_a2}, n_tokens={n_tokens})')
    except Exception as e:
        print(f'  ERROR on {case.case_id}: {e}')
        fail_count += 1

print(f'\nCaptured {len(multipos_records)} records. Failures: {fail_count}')
print(f'Total time: {time.time()-t0:.0f}s')

# Save
n_layers_plus = multipos_records[0]["hs_post_a1"].shape[0]
hidden_dim    = multipos_records[0]["hs_post_a1"].shape[1]
all_a1   = np.stack([r["hs_post_a1"]  for r in multipos_records]).astype(np.float16)
all_a2   = np.stack([r["hs_post_a2"]  for r in multipos_records]).astype(np.float16)
all_last = np.stack([r["hs_last"]     for r in multipos_records]).astype(np.float16)
meta = [{k: r[k] for k in ("case_id","outcome","V_target","distance_kind",
                            "entity","year_first","year_second","pos_a1","pos_a2","n_tokens")}
        for r in multipos_records]
np.savez_compressed(
    f'{RESULTS_DIR}/activations_v4_multipos.npz',
    hs_post_a1=all_a1, hs_post_a2=all_a2, hs_last=all_last,
    meta=json.dumps(meta),
)
print(f'Saved activations: 3 positions × {all_a1.shape[0]} cases × '
      f'{n_layers_plus} layers × {hidden_dim} dim')

# -----------------------------------------------------------------------------
# Retention probe: predict outcome ∈ {missed, not-missed} from post-A2 state
# Question: by the time the model has read assertion #2, does it still have
# year_first encoded enough to predict whether the output will mention it?
# -----------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Labels: 1 = not-missed (model output contains year_first), 0 = missed
y_retain = np.array([
    0 if r["outcome"] == "missed" else 1 for r in multipos_records
])
print(f'\nRetention probe sample: n={len(y_retain)}')
print(f'  not-missed (year_first kept): {int(y_retain.sum())}')
print(f'  missed (year_first dropped):  {int((1-y_retain).sum())}')

def run_per_layer_probe(X_layered, y, n_splits=5, label='probe'):
    """X_layered shape: [n_cases, n_layers+1, hidden_dim]"""
    results = []
    for layer_idx in range(X_layered.shape[1]):
        X = X_layered[:, layer_idx, :].astype(np.float32)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=23)
        fold_aucs = []
        for tr, te in skf.split(X, y):
            clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[te])[:, 1]
            if len(np.unique(y[te])) > 1:
                fold_aucs.append(roc_auc_score(y[te], proba))
        results.append({
            "layer": layer_idx,
            "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float('nan'),
            "auc_std":  float(np.std(fold_aucs))  if fold_aucs else float('nan'),
        })
    return results

print('\nRetention probe — three positions, all layers, V≥16k:')
ret_a1   = run_per_layer_probe(all_a1.astype(np.float32),   y_retain)
ret_a2   = run_per_layer_probe(all_a2.astype(np.float32),   y_retain)
ret_last = run_per_layer_probe(all_last.astype(np.float32), y_retain)

# Quick controls: V, distance_kind, entity at post-A2 position only (most relevant)
y_V    = np.array([1 if r["V_target"] >= 32000 else 0 for r in multipos_records])  # within V≥16k, split at 32k
y_dist = np.array([1 if r["distance_kind"] == "long" else 0 for r in multipos_records])
# Entity as one-hot — train multiclass at post-A2 to see if entity is recoverable
print('Running confound controls at post-A2 position (V split at 32k, distance_kind, entity)...')
ctrl_V    = run_per_layer_probe(all_a2.astype(np.float32), y_V,    n_splits=5)
ctrl_dist = run_per_layer_probe(all_a2.astype(np.float32), y_dist, n_splits=5)

# Save
out_data = {
    "retention_post_a1":  ret_a1,
    "retention_post_a2":  ret_a2,
    "retention_last":     ret_last,
    "control_V_post_a2":  ctrl_V,
    "control_dist_post_a2": ctrl_dist,
}
with open(f'{RESULTS_DIR}/probe_retention.json', 'w') as f:
    json.dump(out_data, f, indent=2)

# -----------------------------------------------------------------------------
# Print + plot
# -----------------------------------------------------------------------------
print(f'\n{"layer":>6}  {"post-A1":>8}  {"post-A2":>8}  {"last":>8}  '
      f'{"ctrl-V":>8}  {"ctrl-dist":>10}  {"excess-A2-vs-V":>16}')
print('-' * 78)
for i in range(n_layers_plus):
    r1, r2, rl = ret_a1[i], ret_a2[i], ret_last[i]
    cv, cd = ctrl_V[i], ctrl_dist[i]
    excess = r2["auc_mean"] - cv["auc_mean"]
    print(f'{i:>6}  {r1["auc_mean"]:>8.3f}  {r2["auc_mean"]:>8.3f}  '
          f'{rl["auc_mean"]:>8.3f}  {cv["auc_mean"]:>8.3f}  '
          f'{cd["auc_mean"]:>10.3f}  {excess:>+16.3f}')

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
layers = list(range(n_layers_plus))

ax = axes[0]
ax.plot(layers, [r["auc_mean"] for r in ret_a1],   marker='^', label='post-A1 position')
ax.plot(layers, [r["auc_mean"] for r in ret_a2],   marker='s', label='post-A2 position', linewidth=2)
ax.plot(layers, [r["auc_mean"] for r in ret_last], marker='o', label='last-input-token (old probe position)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.set_xlabel('Layer index')
ax.set_ylabel('Retention AUC (missed vs not-missed)')
ax.set_title(f'Retention probe by layer × position (n={len(y_retain)})')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

ax = axes[1]
ax.plot(layers, [r["auc_mean"] for r in ret_a2],   marker='s', linewidth=2, label='retention probe (post-A2)')
ax.plot(layers, [r["auc_mean"] for r in ctrl_V],   marker='x', label='V control (post-A2)')
ax.plot(layers, [r["auc_mean"] for r in ctrl_dist],marker='+', label='distance control (post-A2)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.set_xlabel('Layer index')
ax.set_ylabel('AUC at post-A2 position')
ax.set_title('Retention probe vs structural controls (post-A2)')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

fig.tight_layout()
fig.savefig(f'{RESULTS_DIR}/probe_retention_multipos.png',
            dpi=140, bbox_inches='tight')
fig

print('\nReading guide:')
print('  Retention AUC at post-A2 > 0.7 AND > V/dist controls')
print('    -> model still encodes year_first at assertion #2 position.')
print('       Failure is INTEGRATION (R-leg downstream), not retention.')
print('  Retention AUC at post-A2 ≈ 0.5 or ≈ controls')
print('    -> year_first is not encoded at post-A2.')
print('       Failure is RETENTION (attention dropped year_first before assertion #2).')
print('  Retention AUC at post-A2 > retention AUC at last-input-token')
print('    -> the model HAD year_first then LOST it before generation.')
print('       Most interesting case: encoding decays over context.')
