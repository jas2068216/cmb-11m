# =============================================================================
# CMB-LLM Step 1 V3 — scale-up of V2 (n=10 per cell, adds V=32k)
# =============================================================================
# Paste into a new cell below the V2 cell. Assumes V2 cell has already run
# (so the neutral prompt + paraphrased contradictions overrides are in place).
# If the kernel restarted, run the V2 cell first or this re-applies the
# overrides itself.
#
# Total: 5 V × 2 distance × 10 entities = 100 cases. ~10 min on A100.

import random, json, re
from collections import Counter, defaultdict
from pathlib import Path

# Re-apply V2 overrides defensively in case kernel was restarted
import harness.dataset, harness.inference, harness.judge, harness.metrics
from harness.dataset import (
    TestCase, ENTITY_TEMPLATES, CONTRADICTION_YEAR_PAIRS,
    FILLER_TEMPLATES, token_len, _make_filler,
)
from harness.inference import run_dataset, InferenceResult
from harness.judge import judge_all, save_judgments
from harness.metrics import aggregate, plot_rho_vs_V, summary_table, save_metrics

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


def _v3_build_document(entity, year_first, year_second, distance_kind,
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


def build_dataset_v3(tokenizer, V_targets, distance_kinds,
                     entities_per_cell, seed=17):
    rng = random.Random(seed)
    cases = []
    for V_target in V_targets:
        for distance_kind in distance_kinds:
            entity_pool = list(ENTITY_TEMPLATES); rng.shuffle(entity_pool)
            year_pool   = list(CONTRADICTION_YEAR_PAIRS); rng.shuffle(year_pool)
            for i in range(entities_per_cell):
                entity = entity_pool[i % len(entity_pool)]
                yf, ys = year_pool[i % len(year_pool)]
                doc, meta = _v3_build_document(
                    entity, yf, ys, distance_kind, V_target, tokenizer, rng,
                )
                question = NEUTRAL_QUESTION_TEMPLATE.format(name=entity["name"])
                cid = f"V{V_target}_{distance_kind}_{i}_{entity['name'].replace(' ', '')}"
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


# -----------------------------------------------------------------------------
# Run V3
# -----------------------------------------------------------------------------
V_TARGETS = [2000, 4000, 8000, 16000, 32000]
DISTANCE_KINDS = ['short', 'long']
ENTITIES_PER_CELL = 10

cases_v3 = build_dataset_v3(
    tokenizer=tokenizer,
    V_targets=V_TARGETS,
    distance_kinds=DISTANCE_KINDS,
    entities_per_cell=ENTITIES_PER_CELL,
    seed=17,
)
print(f'V3: built {len(cases_v3)} test cases')
print(f'Cells per V_target: {len(DISTANCE_KINDS) * ENTITIES_PER_CELL}')

# Quick sanity on V_actual distribution
from collections import defaultdict
v_dist = defaultdict(list)
for c in cases_v3:
    v_dist[c.V_target].append(c.V_actual)
for V_target in V_TARGETS:
    actuals = v_dist[V_target]
    print(f'  V_target={V_target:>6}: mean V_actual={sum(actuals)/len(actuals):.0f}  '
          f'min={min(actuals)}  max={max(actuals)}')

# Save dataset
import os
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f'{RESULTS_DIR}/dataset_v3.json', 'w') as f:
    json.dump([c.to_dict() for c in cases_v3], f, indent=2)

# Run inference (fresh path)
print('\nStarting inference...')
results_v3 = run_dataset(
    model, tokenizer, cases_v3,
    results_path=f'{RESULTS_DIR}/inference_v3.json',
    max_new_tokens=400,
    verbose=True,
)
print(f'\nCompleted {len(results_v3)} / {len(cases_v3)} cases')

# Judge
judgments_v3 = judge_all(cases_v3, results_v3)
save_judgments(judgments_v3, f'{RESULTS_DIR}/judgments_v3.json')
print('\nV3 outcome counts:', Counter(j.outcome for j in judgments_v3))

# Aggregate + plot
metrics_v3 = aggregate(cases_v3, judgments_v3)
save_metrics(metrics_v3, f'{RESULTS_DIR}/metrics_v3.json')
print('\n' + summary_table(metrics_v3))

fig_v3 = plot_rho_vs_V(
    metrics_v3,
    model_name=f'{MODEL_NAME} — V3 (n=10/cell, V up to 32k)',
    out_path=f'{RESULTS_DIR}/rho_vs_V_v3.png',
)
fig_v3
