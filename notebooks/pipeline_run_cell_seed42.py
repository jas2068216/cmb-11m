# =============================================================================
# CMB-LLM Pipeline — RUN cell (SEED = 42, reproducibility verification).
# Paste this in a NEW Colab cell AFTER the bootstrap cell has run.
# Resumes from any cached phases on Drive at /content/drive/MyDrive/cmb_llm_pipeline/seed42/.
# =============================================================================

import sys, os
if "/content/cmb_llm" not in sys.path:
    sys.path.insert(0, "/content/cmb_llm")

# Mount Drive (no-op if already mounted)
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

SEED = 42
RESULTS_DIR = f'/content/drive/MyDrive/cmb_llm_pipeline/seed{SEED}'
os.makedirs(RESULTS_DIR, exist_ok=True)
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

import importlib
if 'run_pipeline' in sys.modules:
    importlib.reload(sys.modules['run_pipeline'])
import run_pipeline
run_pipeline.main()
