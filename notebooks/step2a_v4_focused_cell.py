# =============================================================================
# CMB-LLM Step 2a V4 — focused scale-up at V≥16k, tightens within-V LOO
# =============================================================================
# Generates 30 cases per cell at V=16k and V=32k only (120 cases total). For
# each case, runs the model forward to capture hidden states at the last input
# token, then generates the response. Judges, aggregates, and re-trains the
# within-V probe with bigger N (expect ~40-50 partial+detected cases vs prior n=14).
#
# Paste below the confound-checks cell. ~15-20 min on A100.

import random, json, time
import numpy as np
import torch
from pathlib import Path
from collections import Counter
import importlib

# Defensive: re-apply V2 overrides in case of kernel restart
import harness.dataset, harness.inference, harness.judge, harness.metrics
from harness.dataset import (
    TestCase, ENTITY_TEMPLATES, CONTRADICTION_YEAR_PAIRS,
    FILLER_TEMPLATES, token_len, _make_filler,
)
from harness.inference import _build_chat_messages, InferenceResult
from harness.judge import judge_all, save_judgments, Judgment
from harness.metrics import aggregate, summary_table, save_metrics

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


def _v4_build_document(entity, year_first, year_second, distance_kind,
                       V_target, tokenizer, rng):
    AVG_TOKENS_PER_SENTENCE = 18
    if distance_kind == "short":
        frac_first, frac_second = 0.45, 0.55
    else:
        frac_first, frac_second = 0.05, 0.85

    tokens_before  = int(V_target * frac_first)
    tokens_between = int(V_target * (frac_second - frac_first))
    tokens_after   = max(0, V_target - tokens_before - tokens_between)

    assertion1 = ASSERTION_TEMPLATE_A.format(name=entity["name"],
                                             year=year_first,
                                             city=entity["city"])
    assertion2 = ASSERTION_TEMPLATE_B.format(name=entity["name"],
                                             year=year_second)
    a1 = token_len(tokenizer, assertion1)
    a2 = token_len(tokenizer, assertion2)
    tokens_between = max(20, tokens_between - a1 - a2)

    n_before  = max(1, tokens_before  // AVG_TOKENS_PER_SENTENCE)
    n_between = max(1, tokens_between // AVG_TOKENS_PER_SENTENCE)
    n_after   = max(1, tokens_after   // AVG_TOKENS_PER_SENTENCE)

    fb = _make_filler(entity, n_before,  rng)
    fm = _make_filler(entity, n_between, rng)
    fa = _make_filler(entity, n_after,   rng)

    intro = (f"Internal briefing on {entity['name']}, a {entity['sector']} "
             f"company based in {entity['city']}.\n\n")
    document = f"{intro}{fb} {assertion1} {fm} {assertion2} {fa}"

    pre1 = f"{intro}{fb} "
    pre2 = f"{intro}{fb} {assertion1} {fm} "
    return document, {
        "pos_first_token": token_len(tokenizer, pre1),
        "pos_second_token": token_len(tokenizer, pre2),
        "V_actual": token_len(tokenizer, document),
        "distance_tokens": token_len(tokenizer, pre2) - token_len(tokenizer, pre1),
    }


def build_dataset_v4(tokenizer, V_targets, distance_kinds,
                     entities_per_cell, seed=23):
    """v4: per-case random draws of entity + year_pair (not cycled).
    Gives more diversity in entity-year combinations at large entities_per_cell.
    """
    rng = random.Random(seed)
    cases = []
    for V_target in V_targets:
        for distance_kind in distance_kinds:
            for i in range(entities_per_cell):
                entity = rng.choice(ENTITY_TEMPLATES)
                yf, ys = rng.choice(CONTRADICTION_YEAR_PAIRS)
                doc, meta = _v4_build_document(
                    entity, yf, ys, distance_kind, V_target, tokenizer, rng,
                )
                question = NEUTRAL_QUESTION_TEMPLATE.format(name=entity["name"])
                cid = f"v4_V{V_target}_{distance_kind}_{i:02d}_{entity['name'].replace(' ', '')}"
                cases.append(TestCase(
                    case_id=cid, entity_name=entity["name"],
                    sector=entity["sector"], city=entity["city"],
                    year_first=yf, year_second=ys,
                    distance_kind=distance_kind,
                    V_target=V_target, V_actual=meta["V_actual"],
                    distance_tokens=meta["distance_tokens"],
                    pos_first_token=meta["pos_first_token"],
                    pos_second_token=meta["pos_second_token"],
                    document=doc, question=question,
                ))
    return cases


def capture_and_generate(model, tokenizer, case, max_new_tokens=400):
    """Single-case: forward pass for hidden states, then generation for response.
    Returns (hidden_states_np, response_text, n_input, n_output, latency_s).
    Hidden states shape: [n_layers+1, hidden_dim], last input token position only.
    """
    messages = _build_chat_messages(case)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    n_input = inputs.input_ids.shape[1]

    t0 = time.time()
    # Pass 1: forward only, capture hidden states
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    last_hs = torch.stack([
        h[0, -1, :].to(torch.float32).cpu() for h in out.hidden_states
    ]).numpy()
    del out
    torch.cuda.empty_cache()

    # Pass 2: generation
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0
    new_tokens = output_ids[0][n_input:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return last_hs, response, n_input, len(new_tokens), latency


# -----------------------------------------------------------------------------
# Run V4
# -----------------------------------------------------------------------------
V_TARGETS = [16000, 32000]
DISTANCE_KINDS = ['short', 'long']
ENTITIES_PER_CELL = 30  # 2 V × 2 distance × 30 = 120 cases

cases_v4 = build_dataset_v4(
    tokenizer=tokenizer,
    V_targets=V_TARGETS,
    distance_kinds=DISTANCE_KINDS,
    entities_per_cell=ENTITIES_PER_CELL,
    seed=23,
)
print(f'V4: built {len(cases_v4)} test cases')

with open(f'{RESULTS_DIR}/dataset_v4.json', 'w') as f:
    json.dump([c.to_dict() for c in cases_v4], f, indent=2)

# Inference + activation capture, incremental save
results_v4 = []
hs_records_v4 = []
results_path = f'{RESULTS_DIR}/inference_v4.json'

# Resume from prior partial run if any
existing_ids = set()
if Path(results_path).exists():
    with open(results_path) as f:
        results_v4 = [InferenceResult(**r) for r in json.load(f)]
    existing_ids = {r.case_id for r in results_v4}
    print(f'[resume] {len(existing_ids)} prior results loaded')

t_start = time.time()
for i, case in enumerate(cases_v4):
    if case.case_id in existing_ids:
        continue
    try:
        hs, response, n_in, n_out, latency = capture_and_generate(
            model, tokenizer, case,
        )
        results_v4.append(InferenceResult(
            case_id=case.case_id,
            model_name=model.config._name_or_path,
            response=response,
            V_actual=case.V_actual,
            input_tokens=n_in,
            output_tokens=n_out,
            latency_s=latency,
        ))
        hs_records_v4.append({
            "case_id": case.case_id,
            "V_target": case.V_target,
            "distance_kind": case.distance_kind,
            "hidden_states": hs,
        })
        if (i + 1) % 10 == 0 or (i + 1) == len(cases_v4):
            with open(results_path, 'w') as f:
                json.dump([r.to_dict() for r in results_v4], f, indent=2)
            elapsed = time.time() - t_start
            print(f'  [{i+1}/{len(cases_v4)}] elapsed={elapsed:.0f}s  '
                  f'last: out={n_out} latency={latency:.1f}s')
    except Exception as e:
        print(f'  ERROR on {case.case_id}: {e}')
        continue

print(f'\nFinished {len(results_v4)} / {len(cases_v4)} cases in {time.time()-t_start:.0f}s')

# Judge
judgments_v4 = judge_all(cases_v4, results_v4)
save_judgments(judgments_v4, f'{RESULTS_DIR}/judgments_v4.json')
print('\nV4 outcome counts:', Counter(j.outcome for j in judgments_v4))

# Aggregate
metrics_v4 = aggregate(cases_v4, judgments_v4)
save_metrics(metrics_v4, f'{RESULTS_DIR}/metrics_v4.json')
print('\n' + summary_table(metrics_v4))

# Save activations
if hs_records_v4:
    all_hs_v4 = np.stack([r["hidden_states"] for r in hs_records_v4]).astype(np.float16)
    meta_v4 = [{k: r[k] for k in ("case_id", "V_target", "distance_kind")}
               for r in hs_records_v4]
    np.savez_compressed(
        f'{RESULTS_DIR}/activations_v4.npz',
        hidden_states=all_hs_v4,
        meta=json.dumps(meta_v4),
    )
    print(f'\nSaved activations: shape={all_hs_v4.shape}  '
          f'size~{all_hs_v4.nbytes/1e6:.1f}MB')

# -----------------------------------------------------------------------------
# Re-run within-V probe (V≥16k) with the larger dataset
# -----------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.metrics import roc_auc_score
import warnings, matplotlib.pyplot as plt
warnings.filterwarnings('ignore', category=UserWarning)

# Map activation -> outcome for V4 cases
case_outcome_v4 = {j.case_id: j.outcome for j in judgments_v4}
within_mask = []
y_within = []
X_within_list = []
for rec in hs_records_v4:
    outcome = case_outcome_v4.get(rec["case_id"])
    if outcome in ("partial", "detected"):
        within_mask.append(True)
        X_within_list.append(rec["hidden_states"])
        y_within.append(1 if outcome == "detected" else 0)
    else:
        within_mask.append(False)

X_within = np.stack(X_within_list).astype(np.float32) if X_within_list else None
y_within = np.array(y_within)
n_layers_plus = X_within.shape[1] if X_within is not None else 29

print(f'\nWithin-V probe (V≥16k, V4 only):')
print(f'  n_total={len(y_within)}  detected={int(y_within.sum())}  '
      f'partial={int((1-y_within).sum())}')

# Use stratified 5-fold if both classes have >=5 cases each, else LOO
n_pos = int(y_within.sum())
n_neg = len(y_within) - n_pos

within_results_v4 = []
if n_pos >= 5 and n_neg >= 5:
    print('  Using 5-fold stratified CV')
    for layer_idx in range(n_layers_plus):
        X = X_within[:, layer_idx, :]
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=23)
        fold_aucs = []
        for tr, te in skf.split(X, y_within):
            clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
            clf.fit(X[tr], y_within[tr])
            proba = clf.predict_proba(X[te])[:, 1]
            if len(np.unique(y_within[te])) > 1:
                fold_aucs.append(roc_auc_score(y_within[te], proba))
        within_results_v4.append({
            "layer": layer_idx,
            "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float('nan'),
            "auc_std":  float(np.std(fold_aucs))  if fold_aucs else float('nan'),
        })
else:
    print('  Using LOO (one class has < 5 samples)')
    for layer_idx in range(n_layers_plus):
        X = X_within[:, layer_idx, :]
        loo = LeaveOneOut()
        probas = np.zeros(len(y_within))
        for tr, te in loo.split(X):
            if len(np.unique(y_within[tr])) < 2:
                probas[te] = 0.5
                continue
            clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=1.0)
            clf.fit(X[tr], y_within[tr])
            probas[te] = clf.predict_proba(X[te])[:, 1]
        auc = float(roc_auc_score(y_within, probas)) if len(np.unique(y_within)) > 1 else float('nan')
        within_results_v4.append({"layer": layer_idx, "auc_mean": auc, "auc_std": 0.0})

print(f'\n{"layer":>6}  {"AUC V4 within-V":>18}  {"AUC V3 within-V (n=14)":>26}')
print('-' * 56)
# Load v3 within-V for comparison
try:
    with open(f'{RESULTS_DIR}/probe_within_v_control.json') as f:
        v3_within = json.load(f)
    v3_lookup = {r["layer"]: r["auc_loo"] for r in v3_within}
except Exception:
    v3_lookup = {}

for r in within_results_v4:
    v3_auc = v3_lookup.get(r["layer"], float('nan'))
    print(f'{r["layer"]:>6}  {r["auc_mean"]:>18.3f}  {v3_auc:>26.3f}')

with open(f'{RESULTS_DIR}/probe_within_v_v4.json', 'w') as f:
    json.dump(within_results_v4, f, indent=2)

# Plot
fig, ax = plt.subplots(figsize=(10, 5))
layers = [r["layer"] for r in within_results_v4]
ax.errorbar(layers,
            [r["auc_mean"] for r in within_results_v4],
            yerr=[r["auc_std"] for r in within_results_v4],
            marker='o', capsize=3, label=f'V4 within-V (n={len(y_within)})')
if v3_lookup:
    ax.plot(layers,
            [v3_lookup.get(l, np.nan) for l in layers],
            marker='s', linestyle='--', alpha=0.6,
            label='V3 within-V (n=14, LOO)')
ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5, label='chance')
ax.axhline(0.7, color='green', linestyle=':', alpha=0.4, label='well above chance')
ax.set_xlabel('Layer index')
ax.set_ylabel('Within-V AUC')
ax.set_title(f'{MODEL_NAME} — partial vs detected, V-controlled\n'
             f'V4 vs V3 comparison')
ax.set_ylim(0.3, 1.05)
ax.grid(True, alpha=0.3)
ax.legend(loc='lower right')
fig.tight_layout()
fig.savefig(f'{RESULTS_DIR}/probe_within_v_v4.png', dpi=140, bbox_inches='tight')
fig

print('\nReading guide:')
print('  V4 AUC near V3 AUC at peak layers (e.g. layer 6) -> signal is real.')
print('  V4 AUC regresses to ~0.5                          -> V3 peak was noise.')
print('  V4 peak shifts location                           -> earlier estimate had bias.')
