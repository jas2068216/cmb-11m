# =============================================================================
# CMB-LLM Step 3 — paired-contrast experiment
# =============================================================================
# For each V4 case, build a matched twin where the contradiction is REMOVED
# (year_second replaced with year_first, so both assertions agree). Same
# entity, same V, same distance, same filler — only the contradiction-present
# bit differs.
#
# Then train probes to discriminate Doc A (contradiction present) from
# Doc B (no contradiction). Because A and B are matched pair-wise, V,
# distance, and entity are perfectly controlled BY CONSTRUCTION. Any
# above-chance AUC is contradiction-encoding, not structural confound.
#
# Paste below the multipos retention cell. ~10-15 min on A100.

import random, json, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GroupKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import harness.dataset, harness.inference
from harness.dataset import (
    TestCase, ENTITY_TEMPLATES, CONTRADICTION_YEAR_PAIRS,
    FILLER_TEMPLATES, token_len, _make_filler,
)
from harness.inference import _build_chat_messages

# Re-apply V2 overrides defensively
harness.inference.SYSTEM_PROMPT = (
    "You are a careful analyst. Answer questions strictly from the provided "
    "document."
)
NEUTRAL_QUESTION_TEMPLATE = "Based on the document above, when was {name} founded?"
ASSERTION_TEMPLATE_A = (
    "{name} was founded in {year} by a small team that relocated to {city}."
)
ASSERTION_TEMPLATE_B = (
    "Industry filings list {year} as the year {name} was established, "
    "and the company's official records continue to cite that date."
)


# -----------------------------------------------------------------------------
# Build paired dataset: Doc A (contradiction) + Doc B (no contradiction)
# Both share the SAME filler (same RNG state) so structure is identical.
# -----------------------------------------------------------------------------
def _build_doc(entity, year_first, year_second, distance_kind, V_target, tokenizer, rng):
    """Same logic as _v4_build_document, returns document + position meta."""
    AVG_TOKENS_PER_SENTENCE = 18
    if distance_kind == "short":
        frac_first, frac_second = 0.45, 0.55
    else:
        frac_first, frac_second = 0.05, 0.85

    tokens_before  = int(V_target * frac_first)
    tokens_between = int(V_target * (frac_second - frac_first))
    tokens_after   = max(0, V_target - tokens_before - tokens_between)

    assertion1 = ASSERTION_TEMPLATE_A.format(name=entity["name"],
                                             year=year_first, city=entity["city"])
    assertion2 = ASSERTION_TEMPLATE_B.format(name=entity["name"], year=year_second)
    tokens_between = max(20, tokens_between
                         - token_len(tokenizer, assertion1)
                         - token_len(tokenizer, assertion2))

    n_before  = max(1, tokens_before  // AVG_TOKENS_PER_SENTENCE)
    n_between = max(1, tokens_between // AVG_TOKENS_PER_SENTENCE)
    n_after   = max(1, tokens_after   // AVG_TOKENS_PER_SENTENCE)

    fb = _make_filler(entity, n_before,  rng)
    fm = _make_filler(entity, n_between, rng)
    fa = _make_filler(entity, n_after,   rng)

    intro = (f"Internal briefing on {entity['name']}, a {entity['sector']} "
             f"company based in {entity['city']}.\n\n")
    document = f"{intro}{fb} {assertion1} {fm} {assertion2} {fa}"
    return document, assertion1, assertion2


def build_paired_dataset(tokenizer, V_targets, distance_kinds,
                         entities_per_cell, seed=23):
    """For each (V, distance, i), produce a pair: Doc A (year_first vs year_second)
    and Doc B (year_first vs year_first — no contradiction). Same filler."""
    rng = random.Random(seed)
    pairs = []
    pair_idx = 0
    for V_target in V_targets:
        for distance_kind in distance_kinds:
            for i in range(entities_per_cell):
                entity = rng.choice(ENTITY_TEMPLATES)
                yf, ys = rng.choice(CONTRADICTION_YEAR_PAIRS)

                # Snapshot RNG state before filler is drawn — Doc B will reuse it
                rng_state = rng.getstate()

                doc_a, a1_a, a2_a = _build_doc(entity, yf, ys, distance_kind,
                                                V_target, tokenizer, rng)
                rng.setstate(rng_state)
                doc_b, a1_b, a2_b = _build_doc(entity, yf, yf, distance_kind,
                                                V_target, tokenizer, rng)

                question = NEUTRAL_QUESTION_TEMPLATE.format(name=entity["name"])
                pairs.append({
                    "pair_id": pair_idx,
                    "entity": entity["name"],
                    "sector": entity["sector"],
                    "city": entity["city"],
                    "V_target": V_target,
                    "distance_kind": distance_kind,
                    "year_first": yf,
                    "year_second": ys,
                    "doc_a": doc_a, "doc_b": doc_b,
                    "assertion1": a1_a, "assertion2_a": a2_a, "assertion2_b": a2_b,
                    "question": question,
                })
                pair_idx += 1
    return pairs


pairs = build_paired_dataset(
    tokenizer=tokenizer,
    V_targets=[16000, 32000],
    distance_kinds=['short', 'long'],
    entities_per_cell=30,
    seed=23,
)
print(f'Built {len(pairs)} pairs ({len(pairs)*2} documents)')

# Confirm Doc B has no contradiction
sample = pairs[0]
print(f'\nSample pair {sample["pair_id"]}:')
print(f'  Doc A: ...year_first={sample["year_first"]} year_second={sample["year_second"]}')
print(f'    A2: {sample["assertion2_a"][:120]}...')
print(f'  Doc B: ...both years = {sample["year_first"]}')
print(f'    A2: {sample["assertion2_b"][:120]}...')

# -----------------------------------------------------------------------------
# Forward pass + capture at 3 positions for each document (both A and B)
# -----------------------------------------------------------------------------
def find_positions(document, assertion1, assertion2, tokenizer, question, entity_name):
    """Returns (pos_a1, pos_a2) — last-token indices of each assertion in
    chat-formatted prompt."""
    # Build a TestCase-like object just for the chat template
    class _MiniCase:
        def __init__(self, doc, q):
            self.document = doc
            self.question = q
        def to_prompt(self):
            return f"{self.document}\n\n---\n\nQuestion: {self.question}"

    mc = _MiniCase(document, question)
    messages = _build_chat_messages(mc)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    doc_offset = prompt_text.find(document)
    if doc_offset < 0:
        return None
    char_a1_end = doc_offset + document.find(assertion1) + len(assertion1)
    char_a2_end = doc_offset + document.find(assertion2) + len(assertion2)
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    def char_to_tok(target):
        pos = 0
        for i, (s, e) in enumerate(offsets):
            if e <= target:
                pos = i
        return pos
    return char_to_tok(char_a1_end), char_to_tok(char_a2_end), prompt_text


def capture_doc_hidden_states(doc, a1, a2, question, entity_name, tokenizer, model):
    """Forward pass; returns hidden states at (post-A1, post-A2, last) × all layers."""
    res = find_positions(doc, a1, a2, tokenizer, question, entity_name)
    if res is None:
        return None
    pos_a1, pos_a2, prompt_text = res
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    hs_a1   = torch.stack([h[0, pos_a1, :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    hs_a2   = torch.stack([h[0, pos_a2, :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    hs_last = torch.stack([h[0, -1,     :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    del out
    torch.cuda.empty_cache()
    return hs_a1, hs_a2, hs_last


print('\nCapturing activations for paired docs...')
records = []  # one per document (so 2 × n_pairs)
t0 = time.time()
for i, p in enumerate(pairs):
    for which in ('a', 'b'):
        doc  = p[f'doc_{which}']
        a1   = p['assertion1']
        a2   = p[f'assertion2_{which}']
        hs = capture_doc_hidden_states(doc, a1, a2, p['question'], p['entity'],
                                       tokenizer, model)
        if hs is None:
            print(f'  pair {p["pair_id"]} doc {which}: position lookup failed')
            continue
        hs_a1, hs_a2, hs_last = hs
        records.append({
            "pair_id": p["pair_id"],
            "doc_kind": which,                  # 'a' = contradiction, 'b' = none
            "entity": p["entity"],
            "V_target": p["V_target"],
            "distance_kind": p["distance_kind"],
            "year_first": p["year_first"],
            "year_second": p["year_second"] if which == 'a' else p["year_first"],
            "hs_post_a1": hs_a1,
            "hs_post_a2": hs_a2,
            "hs_last": hs_last,
        })
    if (i + 1) % 20 == 0:
        print(f'  [{i+1}/{len(pairs)}] elapsed={time.time()-t0:.0f}s')

print(f'\nCaptured {len(records)} documents in {time.time()-t0:.0f}s')

# Save
n_layers_plus = records[0]["hs_post_a1"].shape[0]
hidden_dim    = records[0]["hs_post_a1"].shape[1]
all_a1   = np.stack([r["hs_post_a1"]  for r in records]).astype(np.float16)
all_a2   = np.stack([r["hs_post_a2"]  for r in records]).astype(np.float16)
all_last = np.stack([r["hs_last"]     for r in records]).astype(np.float16)
meta_records = [{k: r[k] for k in ("pair_id","doc_kind","entity","V_target",
                                    "distance_kind","year_first","year_second")}
                for r in records]
np.savez_compressed(
    f'{RESULTS_DIR}/activations_paired.npz',
    hs_post_a1=all_a1, hs_post_a2=all_a2, hs_last=all_last,
    meta=json.dumps(meta_records),
)
print(f'Saved paired activations: shape={all_a1.shape}')

# -----------------------------------------------------------------------------
# Contradiction probe: predict doc_kind (A=1, B=0) from activations.
# Use GroupKFold by pair_id so paired A/B always end up in the same fold.
# -----------------------------------------------------------------------------
y = np.array([1 if r["doc_kind"] == 'a' else 0 for r in records])
groups = np.array([r["pair_id"] for r in records])
print(f'\nContradiction probe sample: n={len(y)}  '
      f'(A={int(y.sum())}, B={int((1-y).sum())})')
print(f'Pairs: {len(np.unique(groups))}, using GroupKFold (5 splits)')


def per_layer_probe_paired(X_layered, y, groups, n_splits=5):
    results = []
    for layer_idx in range(X_layered.shape[1]):
        X = X_layered[:, layer_idx, :].astype(np.float32)
        gkf = GroupKFold(n_splits=n_splits)
        fold_aucs = []
        for tr, te in gkf.split(X, y, groups):
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


print('\nContradiction probe — three positions, all layers:')
res_a1   = per_layer_probe_paired(all_a1.astype(np.float32),   y, groups)
res_a2   = per_layer_probe_paired(all_a2.astype(np.float32),   y, groups)
res_last = per_layer_probe_paired(all_last.astype(np.float32), y, groups)

# Print table
print(f'\n{"layer":>6}  {"post-A1":>9}  {"post-A2":>9}  {"last":>9}')
print('-' * 40)
for i in range(n_layers_plus):
    r1, r2, rl = res_a1[i], res_a2[i], res_last[i]
    print(f'{i:>6}  {r1["auc_mean"]:>9.3f}  {r2["auc_mean"]:>9.3f}  {rl["auc_mean"]:>9.3f}')

with open(f'{RESULTS_DIR}/probe_paired_contrast.json', 'w') as f:
    json.dump({"post_a1": res_a1, "post_a2": res_a2, "last": res_last}, f, indent=2)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
layers = list(range(n_layers_plus))
ax.plot(layers, [r["auc_mean"] for r in res_a1],   marker='^', label='post-A1 position')
ax.plot(layers, [r["auc_mean"] for r in res_a2],   marker='s', linewidth=2,
        label='post-A2 position (where contradiction would be encoded)')
ax.plot(layers, [r["auc_mean"] for r in res_last], marker='o', label='last-input-token')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.axhline(0.7, color='green', linestyle=':', alpha=0.4, label='well above chance')
ax.set_xlabel('Layer index')
ax.set_ylabel('Contradiction-probe AUC (paired GroupKFold)')
ax.set_title(f'Paired-contrast probe: Doc-A (contradiction) vs Doc-B (no contradiction)\n'
             f'n={len(y)} docs, {len(np.unique(groups))} pairs — V/distance/entity controlled by construction')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(f'{RESULTS_DIR}/probe_paired_contrast.png', dpi=140, bbox_inches='tight')
fig

print('\nReading guide:')
print('  AUC > 0.7 at any layer × position  -> contradiction IS encoded internally.')
print('     This survives all structural confounds because V/distance/entity are')
print('     matched within pairs. THIS would be the clean positive result.')
print('  AUC ≈ 0.5 across all layers × positions  -> contradiction is NOT linearly')
print('     encoded at any position we measured. Model behaves as if processing')
print('     the two assertions identically whether they conflict or not.')
print('  Highest signal at post-A2  -> integration happens at the moment of reading A2.')
print('  Highest signal at last-input-token  -> integration happens late, during answer prep.')
