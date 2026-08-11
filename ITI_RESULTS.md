# ITI baseline — results (run 2026-07-26, Qwen2.5-7B, 120-prompt uncertainty benchmark)

Raw JSON: Google Drive → MyDrive/cmb_llm_intervention/iti_baseline_results.json
Integrated into: paper/latex/sections/05c_intervention.tex ("ITI baseline (Qwen)" paragraph).

Baseline POS(uncertain)=0.150, FP(known)=0.000.

| K  | alpha | POS lift | FP delta |
|----|-------|----------|----------|
| 24 | 1.0   | +18.3pp  | +0.0pp   |  <- best at ZERO FP
| 24 | 2.0   | +20.0pp  | +3.3pp   |  <- fair frontier (best lift at <=5pp FP)
| 24 | 3.0   | +18.3pp  | +20.0pp  |
| 24 | 4.0   | +26.7pp  | +33.3pp  |
| 24 | 5.0   | +28.3pp  | +46.7pp  |
| 48 | 1.0   | +5.0pp   | +0.0pp   |
| 48 | 2.0   | +16.7pp  | +20.0pp  |
| 48 | 3.0   | +26.7pp  | +41.7pp  |
| 48 | 4.0   | +25.0pp  | +58.3pp  |
| 48 | 5.0   | +21.7pp  | +63.3pp  |

Per-head probe accuracy healthy: median 0.775, 326 heads >0.8 (capture correct).

## Comparison on Qwen (the model where R-Restoration is null)
- R-Restoration: -0.83pp @ 0 FP (null; saturated)
- ContextFocus (best): +8.3pp @ +2.5pp FP
- **ITI: +18.3pp @ 0 FP, or +20.0pp @ +3.3pp FP**  <- strongest baseline on Qwen

## Two remaining follow-ups (small, not blocking)
1. Re-score ITI generations with the paper's canonical correction-aware regex for the final Table-3 number (the lift ordering won't change; only the absolute POS basis).
2. Optional: run ITI on Llama/Mistral/OLMo to complete the frontier figure across all 4 models (this run was Qwen-only, the hardest case for our method).
