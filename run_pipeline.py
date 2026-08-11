"""End-to-end pipeline for the CMB-LLM (1−R) leg experiment.

Runs the full V4 → paired-contrast → token-matched control → drift detector
pipeline as a single reproducible script. Each phase saves its output to
disk; re-running picks up where the last run stopped.

Pipeline phases:
  1. build_dataset      — paired triples (Doc A, Doc B, Doc B') at V≥16k
  2. inference_judge    — model outputs for Doc A only (where outcomes matter),
                          regex-judged into detected / partial / missed / ambiguous
  3. capture_activations — multi-position hidden states for all three docs
  4. probes             — per-layer linear probes for A-vs-B and A-vs-B'
                          (paired GroupKFold) + outcome confound panel
  5. drift_detector     — train layer-17 probe direction on A vs B',
                          calibrate empirically against baseline natural variance,
                          save deployable monitor artifact
  6. report             — write summary metrics + key plots

Configuration:
    SEED               — global RNG seed (change to verify reproducibility)
    V_TARGETS          — list of context lengths
    DISTANCE_KINDS     — ["short", "long"]
    ENTITIES_PER_CELL  — number of cases per (V, distance) cell
    MODEL_NAME         — HuggingFace model id
    LOAD_IN_4BIT       — for low-VRAM GPUs
    PROBE_LAYER        — which layer to train the deployment probe at

Run from Colab:
    %run /content/cmb_llm/run_pipeline.py

Run standalone:
    python run_pipeline.py [--seed 23] [--phase all] [--results-dir /path]

For reproducibility verification, run twice with different seeds and
verify the framework conclusions stay stable:
    python run_pipeline.py --seed 23 --results-dir results_seed23
    python run_pipeline.py --seed 42 --results-dir results_seed42
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import List

import numpy as np


# =============================================================================
# Configuration (override via CLI / globals before calling main)
# =============================================================================
DEFAULT_CONFIG = {
    "seed":              23,
    "V_targets":         [16000, 32000],
    "distance_kinds":    ["short", "long"],
    "entities_per_cell": 30,
    "model_name":        "Qwen/Qwen2.5-7B-Instruct",
    "load_in_4bit":      False,
    "probe_layer":       17,
    "probe_position":    "last_input",
    "wasserstein_percentile": 99.0,
    "max_new_tokens":    400,
}


# =============================================================================
# Phase 1 — dataset
# =============================================================================
def phase_build_dataset(cfg, results_dir, tokenizer):
    out_path = Path(results_dir) / "paired_dataset.json"
    if out_path.exists():
        print(f"[phase 1] cached: {out_path}")
        from harness.paired_contrast import load_paired_dataset
        return load_paired_dataset(out_path)

    print(f"[phase 1] building paired dataset (seed={cfg['seed']})...")
    from harness.paired_contrast import build_paired_dataset, save_paired_dataset
    triples = build_paired_dataset(
        tokenizer=tokenizer,
        V_targets=cfg["V_targets"],
        distance_kinds=cfg["distance_kinds"],
        entities_per_cell=cfg["entities_per_cell"],
        seed=cfg["seed"],
    )
    save_paired_dataset(triples, out_path)
    print(f"[phase 1] built {len(triples)} triples → {out_path}")
    return triples


# =============================================================================
# Phase 2 — inference + judge on Doc A
# =============================================================================
def phase_inference_judge(cfg, results_dir, triples, model, tokenizer):
    out_inf  = Path(results_dir) / "inference_doca.json"
    out_judg = Path(results_dir) / "judgments_doca.json"

    if out_inf.exists() and out_judg.exists():
        print(f"[phase 2] cached: {out_inf}, {out_judg}")
        with out_inf.open() as f:
            results = json.load(f)
        with out_judg.open() as f:
            judgments = json.load(f)
        return results, judgments

    print(f"[phase 2] running inference on {len(triples)} Doc A cases...")
    import torch, time
    from harness.inference import _build_chat_messages, SYSTEM_PROMPT
    from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT, NEUTRAL_QUESTION_TEMPLATE
    from harness.dataset import TestCase
    from harness.judge import judge_case
    from harness.inference import InferenceResult

    # Use neutral system prompt for inference
    import harness.inference as inf_mod
    inf_mod.SYSTEM_PROMPT = NEUTRAL_SYSTEM_PROMPT

    results, judgments = [], []
    t0 = time.time()
    for i, t in enumerate(triples):
        case = TestCase(
            case_id=f"triple_{t.triple_id}",
            entity_name=t.entity_name, sector=t.sector, city=t.city,
            year_first=t.year_first, year_second=t.year_second,
            distance_kind=t.distance_kind,
            V_target=t.V_target, V_actual=0,
            distance_tokens=0, pos_first_token=0, pos_second_token=0,
            document=t.doc_a, question=t.question,
        )
        messages = _build_chat_messages(case)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        n_input = inputs.input_ids.shape[1]
        tt = time.time()
        with torch.no_grad():
            output_ids = model.generate(
                **inputs, max_new_tokens=cfg["max_new_tokens"],
                do_sample=False, pad_token_id=tokenizer.eos_token_id,
            )
        latency = time.time() - tt
        new_tokens = output_ids[0][n_input:]
        response = tokenizer.decode(new_tokens, skip_special_tokens=True)
        result = InferenceResult(
            case_id=case.case_id, model_name=model.config._name_or_path,
            response=response, V_actual=n_input,
            input_tokens=n_input, output_tokens=len(new_tokens),
            latency_s=latency,
        )
        results.append(result.to_dict())
        j = judge_case(case, result)
        judgments.append(j.to_dict())
        if (i + 1) % 20 == 0:
            with out_inf.open("w") as f: json.dump(results, f, indent=2)
            with out_judg.open("w") as f: json.dump(judgments, f, indent=2)
            print(f"  [{i+1}/{len(triples)}] elapsed={time.time()-t0:.0f}s")

    with out_inf.open("w") as f: json.dump(results, f, indent=2)
    with out_judg.open("w") as f: json.dump(judgments, f, indent=2)

    print(f"[phase 2] outcome distribution: {Counter(j['outcome'] for j in judgments)}")
    return results, judgments


# =============================================================================
# Phase 3 — multi-position activation capture
# =============================================================================
def phase_capture_activations(cfg, results_dir, triples, model, tokenizer):
    out_path = Path(results_dir) / "triples_activations.npz"
    print(f"[phase 3] capturing activations for {len(triples)*3} docs...")
    from harness.paired_contrast import capture_triples_activations
    info = capture_triples_activations(model, tokenizer, triples, out_path,
                                        verbose=True)
    print(f"[phase 3] saved {info['n_records']} records → {info['out_path']}")
    return info


# =============================================================================
# Phase 4 — probes
# =============================================================================
def phase_probes(cfg, results_dir, judgments):
    from harness.paired_contrast import load_triples_activations
    from harness.probes import (
        paired_contrast_probe, run_confound_panel,
        confound_excess_table, paired_results_table,
    )

    out_paired_ab  = Path(results_dir) / "probe_paired_a_vs_b.json"
    out_paired_abp = Path(results_dir) / "probe_paired_a_vs_bp.json"
    out_confound   = Path(results_dir) / "probe_outcome_confound.json"

    a1, a2, last, meta = load_triples_activations(
        Path(results_dir) / "triples_activations.npz"
    )

    # === A vs B (Step 3 result) ===
    if not out_paired_ab.exists():
        print(f"[phase 4] training A vs B paired probe (Step 3, at last-input)...")
        res_last_ab = paired_contrast_probe(last, meta, "a", "b")
        with out_paired_ab.open("w") as f:
            json.dump(res_last_ab, f, indent=2)
    else:
        with out_paired_ab.open() as f:
            res_last_ab = json.load(f)
        print(f"[phase 4] cached A-vs-B probe")

    # === A vs B' (Step 3b — the clean positive) ===
    if not out_paired_abp.exists():
        print(f"[phase 4] training A vs B' token-matched probe (Step 3b)...")
        # All three positions
        res_a1_abp   = paired_contrast_probe(a1,   meta, "a", "bp")
        res_a2_abp   = paired_contrast_probe(a2,   meta, "a", "bp")
        res_last_abp = paired_contrast_probe(last, meta, "a", "bp")
        bundle = {
            "post_a1": res_a1_abp,
            "post_a2": res_a2_abp,
            "last":    res_last_abp,
        }
        with out_paired_abp.open("w") as f:
            json.dump(bundle, f, indent=2)
    else:
        with out_paired_abp.open() as f:
            bundle = json.load(f)
        print(f"[phase 4] cached A-vs-B' probe")

    print('\n[phase 4] A vs B (Step 3) — last position:')
    print(paired_results_table(res_last_ab, "A-vs-B last"))
    print('\n[phase 4] A vs B\' (Step 3b) — three positions:')
    for pos in ('post_a1', 'post_a2', 'last'):
        print(f'\n  position={pos}:')
        print(paired_results_table(bundle[pos], f"A-vs-B' {pos}"))

    # === Outcome confound panel on Doc A activations only ===
    # The historical Step 2a + null probes — verify they're confounded as expected
    if not out_confound.exists():
        print(f"\n[phase 4] running outcome confound panel (on Doc A activations)...")
        a_mask = np.array([m["doc_kind"] == "a" for m in meta])
        a_meta = [m for m, k in zip(meta, a_mask) if k]
        a_last = last[a_mask]

        # Map each Doc A meta record to outcome from judgments
        outcome_by_triple_id = {}
        for j in judgments:
            # case_id was "triple_<id>"
            tid = int(j["case_id"].split("_")[1])
            outcome_by_triple_id[tid] = j["outcome"]

        # Filter to partial+detected for the classic Step 2a comparison
        for m in a_meta:
            m["outcome"] = outcome_by_triple_id.get(m["triple_id"], "missed")
        mask_pd = np.array([m["outcome"] in ("partial", "detected") for m in a_meta])
        a_meta_pd = [m for m, k in zip(a_meta, mask_pd) if k]
        a_last_pd = a_last[mask_pd]

        if a_last_pd.shape[0] >= 10 and any(m["outcome"] == "detected" for m in a_meta_pd):
            panel = run_confound_panel(
                a_last_pd, a_meta_pd,
                primary_label_fn=lambda r: 1 if r["outcome"] == "detected" else 0,
                primary_label_name="partial_vs_detected",
                n_splits=min(5, sum(m["outcome"] == "detected" for m in a_meta_pd)),
            )
            with out_confound.open("w") as f:
                json.dump(panel, f, indent=2)
            print(f"\n[phase 4] confound panel: n={panel['n_total']}  "
                  f"detected={panel['n_primary_positive']}  "
                  f"partial={panel['n_primary_negative']}")
            print(confound_excess_table(panel))
        else:
            print(f"[phase 4] skipping confound panel (not enough detected cases: "
                  f"{sum(m['outcome'] == 'detected' for m in a_meta_pd)})")
    else:
        print(f"[phase 4] cached outcome confound panel")

    return {"a_vs_b": res_last_ab, "a_vs_bp_bundle": bundle}


# =============================================================================
# Phase 5 — drift detector
# =============================================================================
def phase_drift_detector(cfg, results_dir):
    from drift_monitor import (
        ProbeDirection, BaselineStats, ContradictionProbeMonitor,
        train_probe_direction, measure_drift,
    )
    from harness.paired_contrast import load_triples_activations

    out_artifact = Path(results_dir) / "contradiction_monitor.npz"
    if out_artifact.exists():
        print(f"[phase 5] cached: {out_artifact}")
        return out_artifact

    print(f"[phase 5] training drift detector at layer {cfg['probe_layer']}...")
    a1, a2, last, meta = load_triples_activations(
        Path(results_dir) / "triples_activations.npz"
    )

    # Train deployment probe on A vs B' at configured layer/position
    a_mask  = np.array([m["doc_kind"] == "a"  for m in meta])
    bp_mask = np.array([m["doc_kind"] == "bp" for m in meta])
    hs_dict = {"last_input": last, "post_a2": a2, "post_a1": a1}
    hs = hs_dict[cfg["probe_position"]]

    X_a  = hs[a_mask][:,  cfg["probe_layer"], :].astype(np.float32)
    X_bp = hs[bp_mask][:, cfg["probe_layer"], :].astype(np.float32)

    probe, diag = train_probe_direction(
        X_a=X_a, X_bp=X_bp,
        layer=cfg["probe_layer"], position=cfg["probe_position"],
        model_name=cfg["model_name"], C=1.0,
    )
    print(f"[phase 5] probe train_auc={diag['train_auc']:.4f}  n_train={diag['n_train']}")

    # Baseline = projections of Doc A
    proj_A = probe.project_batch(X_a)
    baseline = BaselineStats.from_projections(proj_A,
        name=f"DocA_layer{cfg['probe_layer']}_{cfg['probe_position']}_seed{cfg['seed']}")

    monitor = ContradictionProbeMonitor(probe, baseline, alert_mode="ks")
    tune_result = monitor.tune_thresholds_from_baseline(
        n_splits=100, wasserstein_percentile=cfg["wasserstein_percentile"],
    )
    print(f"[phase 5] calibration: natural_variance_mean={tune_result['natural_variance_mean']:.3f}  "
          f"new W threshold={tune_result['new_threshold']:.3f}")

    monitor.save_artifact(out_artifact)
    print(f"[phase 5] saved monitor artifact → {out_artifact}")

    # Quick sanity + drift verification
    proj_bp = probe.project_batch(X_bp)
    d_self  = measure_drift(baseline, proj_A,
                            wasserstein_threshold=monitor.alert_threshold,
                            ks_pvalue_threshold=monitor.ks_pvalue_threshold,
                            alert_mode="ks")
    d_drift = measure_drift(baseline, proj_bp,
                            wasserstein_threshold=monitor.alert_threshold,
                            ks_pvalue_threshold=monitor.ks_pvalue_threshold,
                            alert_mode="ks")
    print(f"[phase 5] sanity (A vs A): alert={d_self.alert}  KS_p={d_self.ks_pvalue:.3f}")
    print(f"[phase 5] drift  (A vs B'): alert={d_drift.alert}  KS_p={d_drift.ks_pvalue:.2e}  W={d_drift.drift_score_wasserstein:.2f}")

    return out_artifact


# =============================================================================
# Phase 6 — final report
# =============================================================================
def phase_report(cfg, results_dir, judgments):
    out_path = Path(results_dir) / "summary_report.json"
    print(f"[phase 6] writing summary report...")

    outcomes = Counter(j["outcome"] for j in judgments)
    rho_undetected = (outcomes.get("partial", 0) + outcomes.get("missed", 0)
                      + outcomes.get("ambiguous", 0)) / max(1, sum(outcomes.values()))
    rho_committed_wrong = outcomes.get("missed", 0) / max(1, sum(outcomes.values()))

    # Load probe results to grab the headline AUC
    with (Path(results_dir) / "probe_paired_a_vs_bp.json").open() as f:
        abp_bundle = json.load(f)
    last_aucs = [r["auc_mean"] for r in abp_bundle["last"]["results"]]
    peak_layer = int(np.argmax(last_aucs))
    peak_auc = float(last_aucs[peak_layer])
    layer17_auc = float(last_aucs[cfg["probe_layer"]]) if cfg["probe_layer"] < len(last_aucs) else None

    summary = {
        "config":             cfg,
        "n_triples":          sum(outcomes.values()),
        "outcomes":           dict(outcomes),
        "rho_undetected":     rho_undetected,
        "rho_committed_wrong": rho_committed_wrong,
        "probe_headline": {
            "experiment":      "A vs B' (Step 3b token-matched paired contrast)",
            "position":        "last_input",
            "peak_layer":      peak_layer,
            "peak_auc":        peak_auc,
            "auc_at_probe_layer": layer17_auc,
            "probe_layer":     cfg["probe_layer"],
        },
    }
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)

    # Console summary
    print('\n' + '=' * 70)
    print('CMB-LLM PIPELINE SUMMARY')
    print('=' * 70)
    print(f"seed={cfg['seed']}  model={cfg['model_name']}  n={summary['n_triples']}")
    print(f"\nOutcome distribution: {dict(outcomes)}")
    print(f"ρ_undetected    = {rho_undetected:.3f}")
    print(f"ρ_committed_wrong = {rho_committed_wrong:.3f}")
    print(f"\nProbe (A vs B' token-matched, last-input-token):")
    print(f"  peak AUC = {peak_auc:.3f} at layer {peak_layer}")
    print(f"  AUC at deployment layer {cfg['probe_layer']}: {layer17_auc:.3f}" if layer17_auc else "")
    print('=' * 70)
    return summary


# =============================================================================
# Main
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    ap.add_argument("--results-dir", default="./pipeline_results")
    ap.add_argument("--phase", default="all",
                    choices=["all", "dataset", "inference", "activations",
                             "probes", "drift", "report"])
    ap.add_argument("--entities-per-cell", type=int,
                    default=DEFAULT_CONFIG["entities_per_cell"])
    ap.add_argument("--load-in-4bit", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    cfg = dict(DEFAULT_CONFIG)
    cfg["seed"] = args.seed
    cfg["entities_per_cell"] = args.entities_per_cell
    cfg["load_in_4bit"] = args.load_in_4bit
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Pipeline starting: seed={cfg['seed']}  results_dir={results_dir}")

    # Lazy imports so phases that don't need model/tokenizer can skip them
    needs_model = args.phase in ("all", "inference", "activations")
    model, tokenizer = None, None
    if needs_model:
        from harness.inference import load_model
        print(f"Loading model {cfg['model_name']} (4bit={cfg['load_in_4bit']})...")
        model, tokenizer = load_model(cfg["model_name"], load_in_4bit=cfg["load_in_4bit"])
    elif args.phase in ("dataset",):
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])

    triples = None
    if args.phase in ("all", "dataset", "inference", "activations"):
        triples = phase_build_dataset(cfg, results_dir, tokenizer)

    judgments = None
    if args.phase in ("all", "inference"):
        _, judgments = phase_inference_judge(cfg, results_dir, triples, model, tokenizer)
    elif args.phase in ("probes", "report"):
        with (results_dir / "judgments_doca.json").open() as f:
            judgments = json.load(f)

    if args.phase in ("all", "activations"):
        phase_capture_activations(cfg, results_dir, triples, model, tokenizer)

    if args.phase in ("all", "probes"):
        phase_probes(cfg, results_dir, judgments)

    if args.phase in ("all", "drift"):
        phase_drift_detector(cfg, results_dir)

    if args.phase in ("all", "report"):
        phase_report(cfg, results_dir, judgments)


if __name__ == "__main__":
    main()
