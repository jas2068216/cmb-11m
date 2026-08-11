# =============================================================================
# CMB-LLM Step 3b — token-matched paired contrast (sharpening control)
# =============================================================================
# Step 3 showed AUC 1.000 distinguishing Doc A (contradiction: A1=X, A2=Y)
# from Doc B (no contradiction: A1=X, A2=X). At post-A2 the year-tokens differ
# trivially (Y vs X), so the probe could be reading literal token identity.
#
# This cell builds Doc B' instead: A1=Y, A2=Y. Now at post-A2 Doc A and Doc B'
# have the IDENTICAL year-token (both Y). The probe can't distinguish via
# literal token identity. The only thing that differs is year_first's value
# (X in Doc A, Y in Doc B'), which is far earlier in the document and only
# reachable via attention.
#
# AUC > 0.5 at post-A2 in this design  -> model has integrated year_first
#                                          via attention; retains conflict info.
# AUC ≈ 0.5 at post-A2 in this design  -> the Step 3 result was token-identity
#                                          trivial; model doesn't actually
#                                          integrate at post-A2.
#
# Paste below the Step 3 paired-contrast cell. ~10-12 min on A100.

import random, json, time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import harness.dataset, harness.inference
from harness.dataset import (
    ENTITY_TEMPLATES, CONTRADICTION_YEAR_PAIRS,
    FILLER_TEMPLATES, token_len, _make_filler,
)
from harness.inference import _build_chat_messages

# Re-apply overrides
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


def _build_doc(entity, year_first, year_second, distance_kind, V_target, tokenizer, rng):
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


# -----------------------------------------------------------------------------
# Build triples: Doc A (existing), Doc B (existing), Doc B' (new) — same filler
# -----------------------------------------------------------------------------
rng = random.Random(23)
triples = []
pair_idx = 0
for V_target in [16000, 32000]:
    for distance_kind in ['short', 'long']:
        for i in range(30):
            entity = rng.choice(ENTITY_TEMPLATES)
            yf, ys = rng.choice(CONTRADICTION_YEAR_PAIRS)

            rng_state = rng.getstate()
            doc_a, a1_a, a2_a = _build_doc(entity, yf, ys, distance_kind,
                                            V_target, tokenizer, rng)
            rng.setstate(rng_state)
            doc_b, a1_b, a2_b = _build_doc(entity, yf, yf, distance_kind,
                                            V_target, tokenizer, rng)
            rng.setstate(rng_state)
            # Doc B': BOTH assertions use year_second (the new control)
            doc_bp, a1_bp, a2_bp = _build_doc(entity, ys, ys, distance_kind,
                                               V_target, tokenizer, rng)

            triples.append({
                "pair_id": pair_idx,
                "entity": entity["name"], "city": entity["city"],
                "V_target": V_target, "distance_kind": distance_kind,
                "year_first": yf, "year_second": ys,
                "doc_a": doc_a,   "a1_a": a1_a, "a2_a": a2_a,
                "doc_b": doc_b,   "a1_b": a1_b, "a2_b": a2_b,
                "doc_bp": doc_bp, "a1_bp": a1_bp, "a2_bp": a2_bp,
            })
            pair_idx += 1

print(f'Built {len(triples)} triples')
sample = triples[0]
print(f'\nSample triple {sample["pair_id"]}:')
print(f'  Doc A : A1=year_first({sample["year_first"]}), A2=year_second({sample["year_second"]})')
print(f'  Doc B : A1=year_first({sample["year_first"]}), A2=year_first({sample["year_first"]})')
print(f'  Doc B\': A1=year_second({sample["year_second"]}), A2=year_second({sample["year_second"]})')
print(f'\nAt post-A2 token:')
print(f'  Doc A : "...Industry filings list {sample["year_second"]} as the year..."')
print(f'  Doc B\': "...Industry filings list {sample["year_second"]} as the year..."  ← SAME token at A2')
print(f'  difference is at A1 only: A has {sample["year_first"]}, B\' has {sample["year_second"]}')


# -----------------------------------------------------------------------------
# Helpers from Step 3
# -----------------------------------------------------------------------------
def find_positions(document, assertion1, assertion2, tokenizer, question):
    class _MiniCase:
        def __init__(self, doc, q): self.document = doc; self.question = q
        def to_prompt(self): return f"{self.document}\n\n---\n\nQuestion: {self.question}"
    mc = _MiniCase(document, question)
    messages = _build_chat_messages(mc)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    doc_offset = prompt_text.find(document)
    if doc_offset < 0: return None
    char_a1_end = doc_offset + document.find(assertion1) + len(assertion1)
    char_a2_end = doc_offset + document.find(assertion2) + len(assertion2)
    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]
    def c2t(target):
        pos = 0
        for i, (s, e) in enumerate(offsets):
            if e <= target: pos = i
        return pos
    return c2t(char_a1_end), c2t(char_a2_end), prompt_text


def capture_hs(doc, a1, a2, question, tokenizer, model):
    res = find_positions(doc, a1, a2, tokenizer, question)
    if res is None: return None
    pos_a1, pos_a2, prompt_text = res
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    hs_a1   = torch.stack([h[0, pos_a1, :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    hs_a2   = torch.stack([h[0, pos_a2, :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    hs_last = torch.stack([h[0, -1,     :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    del out; torch.cuda.empty_cache()
    return hs_a1, hs_a2, hs_last


# -----------------------------------------------------------------------------
# Capture Doc B' activations (Doc A activations already in activations_paired.npz)
# -----------------------------------------------------------------------------
print('\nCapturing Doc B\' activations (120 documents)...')
bp_records = []
t0 = time.time()
for i, t in enumerate(triples):
    hs = capture_hs(t["doc_bp"], t["a1_bp"], t["a2_bp"],
                    NEUTRAL_QUESTION_TEMPLATE.format(name=t["entity"]),
                    tokenizer, model)
    if hs is None:
        print(f'  pair {t["pair_id"]} Doc B\': position lookup failed')
        continue
    hs_a1, hs_a2, hs_last = hs
    bp_records.append({
        "pair_id": t["pair_id"],
        "V_target": t["V_target"], "distance_kind": t["distance_kind"],
        "entity": t["entity"],
        "year_first": t["year_second"],   # both = year_second in Doc B'
        "year_second": t["year_second"],
        "hs_post_a1": hs_a1, "hs_post_a2": hs_a2, "hs_last": hs_last,
    })
    if (i + 1) % 20 == 0:
        print(f'  [{i+1}/{len(triples)}] elapsed={time.time()-t0:.0f}s')
print(f'Captured {len(bp_records)} Doc B\' records in {time.time()-t0:.0f}s')

# Load existing Doc A activations from Step 3 (extract only the A entries)
loaded = np.load(f'{RESULTS_DIR}/activations_paired.npz', allow_pickle=True)
all_a1   = loaded['hs_post_a1']
all_a2   = loaded['hs_post_a2']
all_last = loaded['hs_last']
meta_step3 = json.loads(str(loaded['meta']))

a_mask = np.array([m["doc_kind"] == 'a' for m in meta_step3])
a_records = [m for m, k in zip(meta_step3, a_mask) if k]
A_a1   = all_a1[a_mask]
A_a2   = all_a2[a_mask]
A_last = all_last[a_mask]
# Pair_id order
a_pair_ids = [r["pair_id"] for r in a_records]
bp_pair_ids = [r["pair_id"] for r in bp_records]
print(f'\nDoc A records: {len(a_records)}  Doc B\' records: {len(bp_records)}')

# Align by pair_id
common_pids = sorted(set(a_pair_ids) & set(bp_pair_ids))
a_lookup  = {r["pair_id"]: i for i, r in enumerate(a_records)}
bp_lookup = {r["pair_id"]: i for i, r in enumerate(bp_records)}

X_a1  = np.concatenate([A_a1[[a_lookup[p]  for p in common_pids]],
                        np.stack([bp_records[bp_lookup[p]]["hs_post_a1"]  for p in common_pids]).astype(np.float16)])
X_a2  = np.concatenate([A_a2[[a_lookup[p]  for p in common_pids]],
                        np.stack([bp_records[bp_lookup[p]]["hs_post_a2"]  for p in common_pids]).astype(np.float16)])
X_last= np.concatenate([A_last[[a_lookup[p] for p in common_pids]],
                        np.stack([bp_records[bp_lookup[p]]["hs_last"]    for p in common_pids]).astype(np.float16)])

y = np.array([1] * len(common_pids) + [0] * len(common_pids))   # 1 = Doc A (contradiction), 0 = Doc B' (no, but year-token matched at A2)
groups = np.array(list(common_pids) + list(common_pids))
print(f'\nA-vs-B\' dataset: n={len(y)}  (A={int(y.sum())}, B\'={int((1-y).sum())})')

# Save
np.savez_compressed(
    f'{RESULTS_DIR}/activations_paired_bprime.npz',
    hs_post_a1=np.stack([bp_records[bp_lookup[p]]["hs_post_a1"] for p in common_pids]).astype(np.float16),
    hs_post_a2=np.stack([bp_records[bp_lookup[p]]["hs_post_a2"] for p in common_pids]).astype(np.float16),
    hs_last=np.stack([bp_records[bp_lookup[p]]["hs_last"] for p in common_pids]).astype(np.float16),
    meta=json.dumps([{k: r[k] for k in ("pair_id","V_target","distance_kind","entity","year_first","year_second")}
                     for r in bp_records]),
)

# -----------------------------------------------------------------------------
# Probe: A vs B' (token-matched at A2)
# -----------------------------------------------------------------------------
def per_layer_probe(X_layered, y, groups, n_splits=5):
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


n_layers_plus = X_a1.shape[1]
print('\nA vs B\' probe — three positions, all layers:')
res_a1   = per_layer_probe(X_a1.astype(np.float32),   y, groups)
res_a2   = per_layer_probe(X_a2.astype(np.float32),   y, groups)
res_last = per_layer_probe(X_last.astype(np.float32), y, groups)

# Load Step 3 (A vs B) results for comparison
with open(f'{RESULTS_DIR}/probe_paired_contrast.json') as f:
    step3 = json.load(f)
step3_a2 = {r["layer"]: r["auc_mean"] for r in step3["post_a2"]}
step3_last = {r["layer"]: r["auc_mean"] for r in step3["last"]}

print(f'\n{"layer":>6}  {"A vs B\' post-A2":>17}  {"A vs B post-A2":>16}  {"A vs B\' last":>14}  {"A vs B last":>13}')
print('-' * 75)
for i in range(n_layers_plus):
    r2 = res_a2[i]["auc_mean"]
    rl = res_last[i]["auc_mean"]
    s2 = step3_a2.get(i, float('nan'))
    sl = step3_last.get(i, float('nan'))
    print(f'{i:>6}  {r2:>17.3f}  {s2:>16.3f}  {rl:>14.3f}  {sl:>13.3f}')

with open(f'{RESULTS_DIR}/probe_paired_bprime.json', 'w') as f:
    json.dump({"post_a1": res_a1, "post_a2": res_a2, "last": res_last}, f, indent=2)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
layers = list(range(n_layers_plus))

ax = axes[0]
ax.plot(layers, [r["auc_mean"] for r in res_a1],   marker='^', label='A vs B\' post-A1')
ax.plot(layers, [r["auc_mean"] for r in res_a2],   marker='s', linewidth=2, label='A vs B\' post-A2 (token-matched)')
ax.plot(layers, [r["auc_mean"] for r in res_last], marker='o', label='A vs B\' last-input-token')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.axhline(0.7, color='green', linestyle=':', alpha=0.4)
ax.set_xlabel('Layer index')
ax.set_ylabel('AUC')
ax.set_title('Sharpening control: A vs B\' (token-matched at A2)')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=9)

ax = axes[1]
ax.plot(layers, [step3_a2.get(l, np.nan) for l in layers], marker='s', linestyle='--',
        alpha=0.7, label='Step 3 (A vs B) post-A2')
ax.plot(layers, [r["auc_mean"] for r in res_a2], marker='s', linewidth=2,
        label='Step 3b (A vs B\') post-A2')
ax.plot(layers, [step3_last.get(l, np.nan) for l in layers], marker='o', linestyle='--',
        alpha=0.7, label='Step 3 (A vs B) last-input-token')
ax.plot(layers, [r["auc_mean"] for r in res_last], marker='o', linewidth=2,
        label='Step 3b (A vs B\') last-input-token')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel('Layer index')
ax.set_ylabel('AUC')
ax.set_title('Step 3 vs Step 3b: trivial token signal vs propagated integration')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right', fontsize=8)

fig.tight_layout()
fig.savefig(f'{RESULTS_DIR}/probe_paired_bprime.png', dpi=140, bbox_inches='tight')
fig

print('\nReading guide:')
print('  A-vs-B\' post-A2 AUC > 0.7  -> model integrates year_first across the document')
print('    via attention. The Step 3 result is NOT just token-identity. Real positive.')
print('  A-vs-B\' post-A2 AUC ≈ 0.5  -> Step 3 was reading literal year-token at A2.')
print('    No clean evidence the model integrates year_first at this position.')
print('  A-vs-B\' last-input AUC remaining high  -> model carries year_first integrated')
print('    state to generation. Decoding-time intervention has a real tap point.')
