# Agentic goal-drift — future-work / LTFF grant seed

*Result held OUT of the ICLR paper by decision (June 2, 2026). Kept here as the
centerpiece of the post-submission LTFF extension. The paper ships with three
clean modes (contradiction, uncertainty, RAG-faithfulness); this one needs a
less-artificial paradigm before it is paper-grade.*

## What we found (Qwen / Mistral / OLMo / Llama, 40 items)

A standing instruction ("end every reply with the sentinel ZORP") is planted at
the top of the context. The probe (layer-17 last-input-token, instructed vs
control) detects that the instruction is internally represented at **AUC 1.000 on
all four models, leakage 0.500** — the goal is held internally everywhere. What
the model *does* with it splits three ways as competing sub-task directives (N,
the interference / D term) increase:

| Model | probe AUC | compliance at N=0,4,12,30 | profile |
|---|---|---|---|
| Qwen2.5-7B | 1.000 | 1.00 / 1.00 / 1.00 / 1.00 | robust — instruction alone suffices |
| Mistral-7B-v0.3 | 1.000 | 1.00 / 1.00 / 1.00 / 1.00 | robust |
| OLMo-2-7B | 1.000 | 1.00 / 0.90 / 0.95 / 0.55 | **monotonic D-driven drift**; loss → recency capture (0.45) |
| Llama-3.1-8B | 1.000 | 0.00 / 0.55 / 0.55 / 0.60 | **never executes the bare instruction**; examples partially rescue; decay |

Two mechanisms: **recency capture** (OLMo — the most recent competing directive
overwrites the standing one) vs **decay** (Llama — the sentinel is dropped with no
rival adopted). The probe-vs-output decoupling is the same expression-gap thesis
as the paper's three modes; the widest single gap in the whole study is Llama at
N=0 (probe 1.000, compliance 0.000).

Figure: `paper/latex/figures/agentic_drift_nsweep.png`. Per-model JSON:
`agentic_nsweep_*.json` in the Drive results folder.

## Why it's grant material, not a paper mode

- **Non-monotonic.** Only OLMo gives the clean "gap grows with interference"
  curve. Llama goes the other way (interference helps), so there is no single
  headline curve.
- **Paradigm is artificial.** The ZORP-sentinel measures *instruction-retention
  under interference*; a reviewer can fairly say Llama's N=0 failure is "declines
  an odd format instruction," not "agentic goal-drift." The paper's other three
  modes do not carry this critique.

## LTFF framing

"Extend the probe-based expression-gap framework to agentic instruction-retention.
Preliminary cross-model evidence shows the standing goal is internally represented
at ceiling (AUC 1.000) while output adherence collapses under interference in a
model-dependent way, via two distinct mechanisms (recency capture vs decay). The
funded work builds a *naturalistic* agent-trajectory benchmark (real tool-use and
sub-goals, not sentinel tokens), tests whether a probe-gated intervention restores
goal adherence, and measures the gap on frontier-scale models."

## To make it paper-grade later

1. Replace the sentinel paradigm with naturalistic standing goals over real
   multi-step agent trajectories (tool calls, sub-goals).
2. Disentangle instruction-following capability from goal-drift (Llama's N=0
   result is the confound to control).
3. Probe-gated intervention (R-restoration analogue) to re-inject the goal.

## Artifacts
- `harness/agentic_drift_pilot.py` (length/V sweep), `harness/agentic_interference_pilot.py` (N/D sweep, 40 items)
- `notebooks/step9b,9d,9e,9f` (pilots, cross-model, N-sweep)
- `paper/latex/figures/agentic_drift_nsweep.png/.pdf`
