# ITI baseline — run this on Colab (the one reviewer-required baseline still unrun)

The code already exists. You just run it once on a GPU; I integrate the numbers after.

## Steps (paste each cell, run in order)
1. New Colab notebook → Runtime → Change runtime type → **GPU** (A100 best, T4 works).
2. Cell 1 — paste & run: `notebooks/pipeline_bootstrap_cell.py`  (materializes the harness)
3. Cell 2 — paste & run: `notebooks/step6e2_writefile_uncertainty_scale_cell.py`  (builds the V3 benchmark)
4. Cell 3 — paste & run: `notebooks/step6i_v2_iti_frontier_cell.py`  (the ITI frontier sweep, ~10–15 min)

## What it does
Runs Inference-Time Intervention (Li et al. 2023) on the same 120-prompt uncertainty benchmark,
measuring POS (hedge) lift and KNOWN false-positive delta — directly comparable to R-Restoration (Table 3).
Writes JSON to Drive: `MyDrive/cmb_llm_intervention/`.

## When it finishes
Send me the printed summary (or the JSON). I build the ITI-vs-R-Restoration comparison row/figure into the paper.
That closes the last open reviewer baseline for the paper.

## If the diagnostics say head-probe accuracies cluster near 0.50
That means the per-head activation reshape is off for that model. Just tell me — it's a known fix in the cell.
