# CMB-LLM: The Expression Gap

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21881774.svg)](https://doi.org/10.5281/zenodo.21881774)

**Probing what language models know but don't say.**

Code, benchmarks, and intervention artifacts for a study showing that
instruction-tuned LLMs internally represent their own epistemic state — whether a
claim is uncertain, self-contradictory, or unsupported by a provided document —
more reliably than they reveal it in generation. We call the difference the
**expression gap**.

## Findings

1. **A universal probing locus.** A linear probe at one mid-network layer
   (layer 17, last-input-token) recovers internal epistemic state across failure
   modes that share no surface structure, and across four open-weights families
   (Qwen2.5-7B, Mistral-7B-v0.3, OLMo-2-7B, Llama-3.1-8B): paired
   known-vs-uncertain prompts (AUC 0.97–1.00), self-contradiction
   (WikiContradict between-passage AUC 0.991), and retrieval (un)faithfulness
   (AUC 1.000 on all four models).

2. **The directions are shared within a model.** Trained on one task and applied
   zero-shot to the others (no refitting), the probe keeps AUC 0.889–1.000
   (mean 0.959) across uncertainty, retrieval faithfulness, and sycophancy on
   Qwen2.5-7B — evidence for a single internal "something is epistemically
   wrong" direction rather than per-task tricks.

3. **The signal is epistemic, not lexical.** Chance at layer 0, survives
   token-length matching, and in a distractor control tracks whether an answer
   is genuinely *supported*, not whether its token appears.

4. **The gap is instruction-gated and model-dependent.** Under a permissive
   prompt, retrieval unfaithfulness ranges from a 0.00 to a 0.78 gap across
   models — and the model that fails least fails most *deceptively* (false
   attribution to the source).

5. **The gap is actionable.** R-Restoration, a probe-gated logit-bias
   intervention, recovers expression at zero KNOWN false-positive cost on a
   per-model basis (Llama +14.17pp, OLMo +15.83pp, Mistral +7.50pp, Qwen null).
   At k=4 models we report per-model existence proofs, not a pooled headline
   (HKSJ 95% CI includes zero, stated plainly in the paper). An ITI baseline
   (Li et al. 2023) is run and reported for comparison.

## Repository layout

```
cmb-11m/
├── harness/            # Core modules: datasets, inference, probes, judge,
│                       #   metrics, paired contrast, R-Restoration, pilots
├── notebooks/          # Colab-paste cells for every pipeline step
├── drift_monitor.py    # Probe + KS-based runtime drift alerting
├── run_pipeline.py     # End-to-end orchestrator
├── results/            # Result JSONs (being migrated from Drive; see note)
├── docs/               # Literature review, V4 design, failure atlas, future work
├── ITI_RESULTS.md      # ITI baseline numbers (Qwen, 120-prompt benchmark)
└── ITI_RUNBOOK.md      # How to reproduce the ITI baseline
```

## Reproducing

1. Open Google Colab with a GPU runtime (A100 best; T4 works with 4-bit loading).
2. Paste and run `notebooks/pipeline_bootstrap_cell.py` — materializes
   `/content/cmb_llm/` with all harness modules (~10 s).
3. Paste and run the writefile cells for the benchmarks you need
   (`step6e2_*` uncertainty, `step7a_*` sycophancy, `step8a_*` RAG).
4. Run the step cells in numeric order for the experiment of interest.
   The cross-task transfer matrix is `notebooks/step6j_crosstask_transfer_matrix_cell.py`.

Result JSONs are written to `MyDrive/cmb_llm_intervention/`.

**Note on `results/`:** canonical result JSONs currently live in the paper
authors' Drive and are being migrated into this repository; the notebooks
regenerate all of them from scratch.

## Benchmarks released

- 120-pair paired uncertainty benchmark (60 fabricated-premise + 60 unknowable),
  in `harness/uncertainty_scale.py` via `notebooks/step6e2_*`
- 40-pair RAG-faithfulness benchmark, in `harness/rag_faithfulness.py` via
  `notebooks/step8a_*`
- 20-pair sycophancy pilot, in `harness/sycophancy_pilot.py` via `notebooks/step7a_*`

Datasets: CC-BY-4.0. Code: MIT.

## Citation

Manuscript under review. Code and benchmarks are archived at Zenodo: DOI [10.5281/zenodo.21881774](https://doi.org/10.5281/zenodo.21881774) (v1.0.0, August 2026).

## Status

Active research code, submitted for ICLR 2027 review cycle. Reproducibility
verified across two seeds (23, 42).

