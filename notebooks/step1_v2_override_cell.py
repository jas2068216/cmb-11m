# =============================================================================
# CMB-LLM Step 1 V2 — neutral prompt + paraphrased contradictions
# =============================================================================
# Paste this into a new cell at the bottom of step1_baseline_colab.ipynb and
# run it. It assumes cells 1-5 already ran (Drive mounted, model loaded).
# Writes to /content/drive/MyDrive/cmb_llm_results/ with _v2 suffix so V1 stays
# intact.

import random, json, re
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
import importlib

# Reload harness in case anything is stale
import harness.dataset
import harness.inference
import harness.judge
import harness.metrics
importlib.reload(harness.dataset)
importlib.reload(harness.inference)
importlib.reload(harness.judge)
importlib.reload(harness.metrics)

from harness.dataset import (
    TestCase, ENTITY_TEMPLATES, CONTRADICTION_YEAR_PAIRS,
    FILLER_TEMPLATES, token_len, _make_filler,
)
from harness.inference import run_dataset, InferenceResult
from harness.judge import judge_all, save_judgments, Judgment
from harness.metrics import aggregate, plot_rho_vs_V, summary_table, save_metrics

# -----------------------------------------------------------------------------
# Fix 1: Neutral system prompt + neutral question (no coaching toward detection)
# -----------------------------------------------------------------------------
harness.inference.SYSTEM_PROMPT = (
    "You are a careful analyst. Answer questions strictly from the provided "
    "document."
)

NEUTRAL_QUESTION_TEMPLATE = (
    "Based on the document above, when was {name} founded?"
)

# -----------------------------------------------------------------------------
# Fix 2: Paraphrased contradiction templates
# -----------------------------------------------------------------------------
# Two distinct surface forms so the contradiction can't be pattern-matched on
# sentence shape. The first puts the year in a founding-event clause; the
# second puts it in an "industry records list" frame.
ASSERTION_TEMPLATE_A = (
    "{name} was founded in {year} by a small team that relocated to {city}."
)
ASSERTION_TEMPLATE_B = (
    "Industry filings list {year} as the year {name} was established, "
    "and the company's official records continue to cite that date."
)


def _v2_build_document(entity, year_first, year_second, distance_kind,
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

    a1_tokens = token_len(tokenizer, assertion1)
    a2_tokens = token_len(tokenizer, assertion2)
    tokens_between = max(20, tokens_between - a1_tokens - a2_tokens)

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
    pos_first  = token_len(tokenizer, pre1)
    pos_second = token_len(tokenizer, pre2)
    V_actual = token_len(tokenizer, document)
    return document, {
        "pos_first_token": pos_first,
        "pos_second_token": pos_second,
        "V_actual": V_actual,
        "distance_tokens": pos_second - pos_first,
    }


def build_dataset_v2(tokenizer, V_targets, distance_kinds,
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
                doc, meta = _v2_build_document(
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
# Run V2
# -----------------------------------------------------------------------------
V_TARGETS = [2000, 4000, 8000, 16000]
DISTANCE_KINDS = ['short', 'long']
ENTITIES_PER_CELL = 3

cases_v2 = build_dataset_v2(
    tokenizer=tokenizer,
    V_targets=V_TARGETS,
    distance_kinds=DISTANCE_KINDS,
    entities_per_cell=ENTITIES_PER_CELL,
    seed=17,
)
print(f'V2: built {len(cases_v2)} test cases')

# Sample one so we can eyeball the prompt that goes to the model
sample = cases_v2[0]
print(f'\n--- sample case: {sample.case_id} ---')
print(f'V_actual={sample.V_actual}  distance_tokens={sample.distance_tokens}')
print(f'year_first={sample.year_first}  year_second={sample.year_second}')
print(f'question: {sample.question}')
print(f'doc[first 400]: {sample.document[:400]}...')
print(f'doc[last 400]:  ...{sample.document[-400:]}')

# Save dataset
import os
os.makedirs(RESULTS_DIR, exist_ok=True)
with open(f'{RESULTS_DIR}/dataset_v2.json', 'w') as f:
    json.dump([c.to_dict() for c in cases_v2], f, indent=2)

# Run inference (fresh results path so V1 isn't touched)
results_v2 = run_dataset(
    model, tokenizer, cases_v2,
    results_path=f'{RESULTS_DIR}/inference_v2.json',
    max_new_tokens=400,
    verbose=True,
)
print(f'\nCompleted {len(results_v2)} / {len(cases_v2)} cases')

# Judge
judgments_v2 = judge_all(cases_v2, results_v2)
save_judgments(judgments_v2, f'{RESULTS_DIR}/judgments_v2.json')
print('\nV2 outcome counts:', Counter(j.outcome for j in judgments_v2))

# Spot-check
case_by_id   = {c.case_id: c for c in cases_v2}
result_by_id = {r.case_id: r for r in results_v2}
for j in judgments_v2[:6]:
    c = case_by_id[j.case_id]; r = result_by_id[j.case_id]
    print(f'\n--- {j.case_id} ---')
    print(f'  truth: {c.year_first} vs {c.year_second} (distance={c.distance_kind})')
    print(f'  outcome: {j.outcome}  notes: {j.notes}')
    print(f'  response: {r.response[:300]}...')

# Aggregate + plot
metrics_v2 = aggregate(cases_v2, judgments_v2)
save_metrics(metrics_v2, f'{RESULTS_DIR}/metrics_v2.json')
print('\n' + summary_table(metrics_v2))

fig_v2 = plot_rho_vs_V(
    metrics_v2,
    model_name=f'{MODEL_NAME} — V2 (neutral prompt + paraphrased)',
    out_path=f'{RESULTS_DIR}/rho_vs_V_v2.png',
)
fig_v2
