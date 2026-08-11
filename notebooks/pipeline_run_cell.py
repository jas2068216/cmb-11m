# =============================================================================
# CMB-LLM Pipeline — RUN cell.
# Paste this in a SECOND Colab cell, AFTER pipeline_bootstrap_cell.py has run.
# This is the actual pipeline execution. ~60-90 min on A100.
# =============================================================================

import sys, os
if "/content/cmb_llm" not in sys.path:
    sys.path.insert(0, "/content/cmb_llm")

# Mount Drive for results persistence
try:
    from google.colab import drive
    drive.mount('/content/drive')
    RESULTS_BASE = '/content/drive/MyDrive/cmb_llm_pipeline'
except Exception:
    RESULTS_BASE = '/content/cmb_llm_pipeline'
os.makedirs(RESULTS_BASE, exist_ok=True)

# Pick a seed. Run twice with different seeds for reproducibility verification.
SEED = 23  # primary run. Change to 42 for the verification run.
RESULTS_DIR = f'{RESULTS_BASE}/seed{SEED}'
print(f'Results will land in: {RESULTS_DIR}')

# Build CLI args and invoke main()
sys.argv = [
    'run_pipeline.py',
    '--seed', str(SEED),
    '--results-dir', RESULTS_DIR,
    '--entities-per-cell', '30',
    '--phase', 'all',
]

# If you draw a T4, uncomment the next line for 4-bit loading
# sys.argv.append('--load-in-4bit')

# Run
import importlib
if 'run_pipeline' in sys.modules:
    importlib.reload(sys.modules['run_pipeline'])
import run_pipeline
run_pipeline.main()
