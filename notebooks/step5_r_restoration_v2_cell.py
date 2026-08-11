# =============================================================================
# CMB-LLM Step 5 — R-Restoration v2 (probe-conditioned logit biasing).
# THE "BETTER AI" DEMO. Original goal: predict drift → intervene → better AI.
# This cell builds the intervention and measures whether it works.
#
# Procedure:
#   1. Load the trained probe direction (from seed=23 contradiction_monitor.npz)
#   2. Load V4 cases (120 contradiction-injection test cases at V≥16k)
#   3. For each case, run model.generate() WITH probe-conditioned logit bias
#      (boost contradiction-flagging tokens iff probe score above threshold)
#   4. Judge interventions same way as baseline (regex flag detection)
#   5. Compare ρ_undetected: baseline (no intervention) vs intervened
#
# Success metric: ρ_undetected drops measurably. Baseline = 0.92 (seed=23).
# Target: ρ_intervened < 0.5 (model now flags >50% of contradictions it missed).
#
# Paste below the seed=42 run cell. ~10-15 min on A100 (120 cases × 2 forward
# passes: one for probe scoring, one for generation).

import sys, os, json, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')

# Mount Drive (no-op if already mounted)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

# Configuration
SEED              = 23                                                  # use seed=23's probe direction (the reference)
RESULTS_DIR       = f'/content/drive/MyDrive/cmb_llm_pipeline/seed{SEED}'
OUT_DIR           = '/content/drive/MyDrive/cmb_llm_intervention'
os.makedirs(OUT_DIR, exist_ok=True)

THRESHOLD         = 0.0                                                 # probe score > 0 = "contradiction encoded" (since trained as A=positive, A's projections cluster at ~+3.7)
BIAS_STRENGTH     = 3.0                                                 # logit bias added to flag tokens
PROBE_LAYER       = 17

# -----------------------------------------------------------------------------
# 1. Load probe direction from saved monitor artifact
# -----------------------------------------------------------------------------
from drift_monitor import ContradictionProbeMonitor
monitor = ContradictionProbeMonitor.from_artifact(
    f'{RESULTS_DIR}/contradiction_monitor.npz'
)
print(f'Loaded probe direction from {RESULTS_DIR}/contradiction_monitor.npz')
print(f'  layer={monitor.probe.layer}  position={monitor.probe.position}')
print(f'  hidden_dim={monitor.probe.hidden_dim}')
print(f'  baseline mean projection (Doc A) = {monitor.baseline.mean:.3f}')

probe_weights = monitor.probe.weights
probe_bias    = monitor.probe.bias

# -----------------------------------------------------------------------------
# 2. Load V4 cases (use seed=23 dataset for direct comparison to baseline)
# -----------------------------------------------------------------------------
from harness.paired_contrast import load_paired_dataset
from harness.dataset import TestCase

triples = load_paired_dataset(f'{RESULTS_DIR}/paired_dataset.json')

# Convert PairedCase → TestCase (Doc A only, since that's what we want to intervene on)
cases = []
for t in triples:
    cases.append(TestCase(
        case_id=f"triple_{t.triple_id}",
        entity_name=t.entity_name, sector=t.sector, city=t.city,
        year_first=t.year_first, year_second=t.year_second,
        distance_kind=t.distance_kind, V_target=t.V_target,
        V_actual=0, distance_tokens=0,
        pos_first_token=0, pos_second_token=0,
        document=t.doc_a, question=t.question,
    ))
print(f'Loaded {len(cases)} V4 test cases')

# -----------------------------------------------------------------------------
# 3. Run R-restoration intervention on each case
# -----------------------------------------------------------------------------
from harness.r_restoration import evaluate_intervention, build_flag_token_ids

# Show flag tokens we'll be biasing
flag_ids = build_flag_token_ids(tokenizer)
print(f'\n{len(flag_ids)} flag-token IDs constructed. Sample decoded:')
sample = sorted(flag_ids)[:20]
for tid in sample:
    print(f'  {tid:>6}: {tokenizer.decode([tid])!r}')

# Re-apply neutral system prompt (intervention should not be coaching either)
import harness.inference as inf_mod
from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT
inf_mod.SYSTEM_PROMPT = NEUTRAL_SYSTEM_PROMPT

print('\n[R-restoration v2] running intervention on 120 cases...')
out_path = f'{OUT_DIR}/intervention_v2_results.json'
results = evaluate_intervention(
    model, tokenizer, cases,
    probe_weights=probe_weights, probe_bias=probe_bias,
    threshold=THRESHOLD, bias_strength=BIAS_STRENGTH, layer=PROBE_LAYER,
    out_path=out_path, verbose=True,
)
print(f'\nSaved intervention results → {out_path}')

# -----------------------------------------------------------------------------
# 4. Judge intervened responses
# -----------------------------------------------------------------------------
from harness.judge import judge_case, Judgment
from harness.inference import InferenceResult

case_by_id = {c.case_id: c for c in cases}
intervened_judgments = []
for r in results:
    case = case_by_id[r["case_id"]]
    fake_result = InferenceResult(
        case_id=r["case_id"], model_name=monitor.probe.model_name,
        response=r["response"], V_actual=r["n_input"],
        input_tokens=r["n_input"], output_tokens=r["n_output"],
        latency_s=r["latency_s"],
    )
    intervened_judgments.append(judge_case(case, fake_result))

with open(f'{OUT_DIR}/intervened_judgments.json', 'w') as f:
    json.dump([j.to_dict() for j in intervened_judgments], f, indent=2)

interv_outcomes = Counter(j.outcome for j in intervened_judgments)
print(f'\nIntervened outcome counts: {dict(interv_outcomes)}')

# -----------------------------------------------------------------------------
# 5. Load baseline judgments (no intervention) for comparison
# -----------------------------------------------------------------------------
with open(f'{RESULTS_DIR}/judgments_doca.json') as f:
    baseline_judgments = json.load(f)
baseline_outcomes = Counter(j["outcome"] for j in baseline_judgments)
print(f'Baseline outcome counts:   {dict(baseline_outcomes)}')

# Compute ρ for both
def rho_metrics(outcome_counter):
    total = sum(outcome_counter.values())
    detected = outcome_counter.get("detected", 0)
    return {
        "n_total":   total,
        "n_detected": detected,
        "n_partial":  outcome_counter.get("partial", 0),
        "n_missed":   outcome_counter.get("missed", 0),
        "rho_undetected": (total - detected) / max(1, total),
        "rho_committed_wrong": outcome_counter.get("missed", 0) / max(1, total),
        "detection_rate": detected / max(1, total),
    }

base_m = rho_metrics(baseline_outcomes)
interv_m = rho_metrics(interv_outcomes)

print('\n' + '=' * 70)
print('R-RESTORATION v2 — BEFORE vs AFTER COMPARISON')
print('=' * 70)
print(f'{"metric":<25}  {"baseline":>10}  {"intervened":>12}  {"delta":>10}')
print('-' * 70)
for k in ('n_detected', 'n_partial', 'n_missed',
          'detection_rate', 'rho_undetected', 'rho_committed_wrong'):
    b, i = base_m[k], interv_m[k]
    if isinstance(b, float):
        print(f'{k:<25}  {b:>10.3f}  {i:>12.3f}  {i-b:>+10.3f}')
    else:
        print(f'{k:<25}  {b:>10d}  {i:>12d}  {i-b:>+10d}')
print('=' * 70)

# Save summary
summary = {
    "config": {
        "seed": SEED,
        "threshold": THRESHOLD,
        "bias_strength": BIAS_STRENGTH,
        "probe_layer": PROBE_LAYER,
    },
    "baseline": base_m,
    "intervened": interv_m,
    "delta_detection_rate": interv_m["detection_rate"] - base_m["detection_rate"],
    "delta_rho_undetected": interv_m["rho_undetected"] - base_m["rho_undetected"],
}
with open(f'{OUT_DIR}/intervention_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

# -----------------------------------------------------------------------------
# 6. Probe-score histogram + intervention-active count
# -----------------------------------------------------------------------------
probe_scores = np.array([r["probe_score"] for r in results])
active_mask  = probe_scores > THRESHOLD
print(f'\nProbe score distribution (n={len(probe_scores)}):')
print(f'  mean={probe_scores.mean():.2f}  std={probe_scores.std():.2f}')
print(f'  min={probe_scores.min():.2f}  max={probe_scores.max():.2f}')
print(f'  intervention active on {active_mask.sum()}/{len(probe_scores)} cases '
      f'({100*active_mask.mean():.1f}%)')

# Plot
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
labels = ['detected', 'partial', 'missed']
b_vals = [baseline_outcomes.get(L, 0) for L in labels]
i_vals = [interv_outcomes.get(L, 0)  for L in labels]
x = np.arange(len(labels))
ax.bar(x - 0.2, b_vals, width=0.4, label='baseline (no intervention)', color='C0')
ax.bar(x + 0.2, i_vals, width=0.4, label='R-restoration v2', color='C2')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel('Count (out of 120)')
ax.set_title('Outcome distribution: baseline vs R-restoration v2')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

ax = axes[1]
ax.hist(probe_scores, bins=20, alpha=0.7, color='C3')
ax.axvline(THRESHOLD, color='black', linestyle='--', label=f'threshold = {THRESHOLD}')
ax.set_xlabel('Probe projection (layer 17 last-input-token)')
ax.set_ylabel('Count')
ax.set_title(f'Probe scores on V4 cases\n({active_mask.sum()}/{len(probe_scores)} '
             f'above threshold → intervention active)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.tight_layout()
fig.savefig(f'{OUT_DIR}/intervention_comparison.png', dpi=140, bbox_inches='tight')
fig

# -----------------------------------------------------------------------------
# 7. Spot-check: show a few intervened responses on previously-missed cases
# -----------------------------------------------------------------------------
case_by_id = {c.case_id: c for c in cases}
baseline_by_id = {j["case_id"]: j for j in baseline_judgments}
result_by_id   = {r["case_id"]: r for r in results}
intervened_by_id = {j.case_id: j for j in intervened_judgments}

flipped_cases = []
for cid in case_by_id:
    b_out = baseline_by_id[cid]["outcome"]
    i_out = intervened_by_id[cid].outcome
    if b_out in ("missed", "partial") and i_out == "detected":
        flipped_cases.append(cid)

print(f'\n{len(flipped_cases)} cases flipped from missed/partial → detected after R-restoration.')
print('Sample flipped cases:')
for cid in flipped_cases[:3]:
    c = case_by_id[cid]
    print(f'\n--- {cid} ---')
    print(f'  truth: year_first={c.year_first}  year_second={c.year_second}')
    print(f'  baseline outcome: {baseline_by_id[cid]["outcome"]}')
    print(f'    response: {[j for j in baseline_judgments if j["case_id"]==cid][0].get("notes", "")[:120]}')
    r = result_by_id[cid]
    print(f'  intervened outcome: {intervened_by_id[cid].outcome}')
    print(f'    probe_score: {r["probe_score"]:.2f}  active: {r["intervention_active"]}')
    print(f'    response: {r["response"][:300]}...')

print('\n' + '=' * 70)
print('R-RESTORATION v2: DONE')
print('=' * 70)
delta_pct = (interv_m["detection_rate"] - base_m["detection_rate"]) * 100
print(f'Detection rate: {base_m["detection_rate"]:.1%} → {interv_m["detection_rate"]:.1%}  '
      f'(Δ +{delta_pct:.1f} percentage points)')
print(f'ρ_undetected:   {base_m["rho_undetected"]:.3f} → {interv_m["rho_undetected"]:.3f}')
if interv_m["detection_rate"] > base_m["detection_rate"]:
    print('\n  ✓ R-restoration IMPROVED detection. Framework intervention works.')
else:
    print('\n  ✗ R-restoration did NOT improve detection. Iterate on bias_strength / threshold / flag vocabulary.')
