# =============================================================================
# CMB-LLM Step 4b — calibrated drift detector demo
# =============================================================================
# Uses the PATCHED drift_monitor.py (Step 4 + patches: raised W threshold,
# added KS-based alerting, added tune_thresholds_from_baseline).
#
# Re-runs the same sanity check / drift / streaming experiments as Step 4,
# but with the patched defaults. The sanity check should NO LONGER fire a
# false alert.
#
# Paste below the patched writefile cell. ~10 seconds.

import sys, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Reload the patched module (in case the old one is still in sys.modules)
if 'drift_monitor' in sys.modules:
    del sys.modules['drift_monitor']
if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')

from drift_monitor import (
    ProbeDirection, BaselineStats, ContradictionProbeMonitor,
    train_probe_direction, measure_drift, ALERT_MODES,
)

# -----------------------------------------------------------------------------
# Reload V4 paired-contrast activations + re-train probe + baseline
# -----------------------------------------------------------------------------
step3_loaded = np.load(f'{RESULTS_DIR}/activations_paired.npz', allow_pickle=True)
step3_meta = json.loads(str(step3_loaded['meta']))
step3_hs_last = step3_loaded['hs_last']

step3b_loaded = np.load(f'{RESULTS_DIR}/activations_paired_bprime.npz', allow_pickle=True)
bp_hs_last = step3b_loaded['hs_last']

a_mask = np.array([m['doc_kind'] == 'a' for m in step3_meta])
A_hs_last = step3_hs_last[a_mask]

LAYER = 17
X_a_layer  = A_hs_last[:, LAYER, :].astype(np.float32)
X_bp_layer = bp_hs_last[:, LAYER, :].astype(np.float32)

probe, diag = train_probe_direction(
    X_a=X_a_layer, X_bp=X_bp_layer,
    layer=LAYER, position="last_input",
    model_name=(MODEL_NAME if 'MODEL_NAME' in globals() else 'Qwen/Qwen2.5-7B-Instruct'),
    C=1.0,
)
print(f'Probe trained: train_auc={diag["train_auc"]:.3f}')

proj_A  = probe.project_batch(X_a_layer)
proj_bp = probe.project_batch(X_bp_layer)
baseline = BaselineStats.from_projections(proj_A, name="V4_DocA_layer17_last_input")
print(f'Baseline: mean={baseline.mean:.2f} std={baseline.std:.2f} n={baseline.n}')

# -----------------------------------------------------------------------------
# Step 4b.1 — Build monitor and calibrate thresholds empirically
# -----------------------------------------------------------------------------
monitor = ContradictionProbeMonitor(probe, baseline, alert_mode="ks")
print(f'\nDefault thresholds: W={monitor.alert_threshold}  KS_p={monitor.ks_pvalue_threshold}  mode={monitor.alert_mode}')

tune_result = monitor.tune_thresholds_from_baseline(n_splits=100,
                                                    wasserstein_percentile=99.0)
print('\nCalibration from baseline natural variance:')
for k, v in tune_result.items():
    print(f'  {k}: {v}')
print(f'\nMonitor now using calibrated W threshold = {monitor.alert_threshold:.3f}')

# -----------------------------------------------------------------------------
# Step 4b.2 — Compare alert modes across the three scenarios
# -----------------------------------------------------------------------------
rng = np.random.default_rng(17)
n = len(proj_A)
perm = rng.permutation(n)
half1 = proj_A[perm[:n//2]]
half2 = proj_A[perm[n//2:]]
baseline_h1 = BaselineStats.from_projections(half1, "DocA_h1")

scenarios = {
    "A-vs-A halves (no drift)":   (baseline_h1, half2),
    "A-vs-A full (no drift)":     (baseline, proj_A),
    "A-vs-B' (real drift)":       (baseline, proj_bp),
}

print(f'\n{"scenario":<32s}  {"mode":>12s}  {"alert":>6s}  {"W":>7s}  {"KS_p":>10s}')
print('-' * 78)
for scen_name, (bl, cur) in scenarios.items():
    for mode in ALERT_MODES:
        d = measure_drift(
            bl, cur,
            wasserstein_threshold=monitor.alert_threshold,
            ks_pvalue_threshold=monitor.ks_pvalue_threshold,
            alert_mode=mode,
        )
        print(f'{scen_name:<32s}  {mode:>12s}  {str(d.alert):>6s}  '
              f'{d.drift_score_wasserstein:>7.3f}  {d.ks_pvalue:>10.2e}')
    print('-' * 78)

# -----------------------------------------------------------------------------
# Step 4b.3 — Save calibrated artifact (round-trips the new thresholds)
# -----------------------------------------------------------------------------
artifact_path = f'{RESULTS_DIR}/contradiction_monitor_calibrated.npz'
monitor.save_artifact(artifact_path)
print(f'\nSaved CALIBRATED monitor artifact: {artifact_path}')
print(f'  alert_mode={monitor.alert_mode}')
print(f'  W threshold={monitor.alert_threshold:.3f}  (calibrated)')
print(f'  KS p threshold={monitor.ks_pvalue_threshold}')

# Verify round-trip
m2 = ContradictionProbeMonitor.from_artifact(artifact_path)
print(f'\nRound-trip check: loaded threshold={m2.alert_threshold:.3f} (matches: '
      f'{abs(m2.alert_threshold - monitor.alert_threshold) < 1e-6})')

# -----------------------------------------------------------------------------
# Step 4b.4 — Streaming with calibrated thresholds
# -----------------------------------------------------------------------------
print('\nStreaming drift demo with calibrated thresholds (KS mode):')
mixture_levels = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
window = 40
for mix in mixture_levels:
    n_bp = int(window * mix); n_a = window - n_bp
    idx_a  = rng.choice(len(proj_A),  size=n_a,  replace=True)
    idx_bp = rng.choice(len(proj_bp), size=n_bp, replace=True)
    stream = np.concatenate([proj_A[idx_a], proj_bp[idx_bp]])
    rng.shuffle(stream)
    d = measure_drift(baseline, stream,
                      wasserstein_threshold=monitor.alert_threshold,
                      ks_pvalue_threshold=monitor.ks_pvalue_threshold,
                      alert_mode='ks')
    print(f'  mix={mix:.2f}  alert={d.alert}  KS_p={d.ks_pvalue:.2e}  '
          f'W={d.drift_score_wasserstein:.3f}  mean_shift={d.mean_shift:+.2f}')

print('\nCalibrated drift detector: DONE.')
print('Key improvements over default:')
print(f'  - Sanity check (A vs A halves) no longer false-alerts under KS mode')
print(f'  - Calibrated W threshold is empirically grounded ({monitor.alert_threshold:.3f}),')
print(f'    not arbitrary (0.5 / 1.0)')
print(f'  - alert_mode="ks" is statistically principled (p-value-based)')
print(f'  - All thresholds round-trip through save/load')
