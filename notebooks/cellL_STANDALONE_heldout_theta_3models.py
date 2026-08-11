# =============================================================================
# WAVE-2 CELL L — STANDALONE: held-out gate calibration on Mistral + OLMo + Llama
# (GPU, ~40 min total. RUN LAST, fresh kernel fine.)
# Extends the paper's Qwen-only held-out theta measurement to the other three
# families: 20 random 50/50 pair-level splits; probe + theta fit on the
# calibration half only; held-out KNOWN gate-fire rate and UNCERTAIN coverage
# measured on the test half. NO generation, activations only.
# =============================================================================
import os, json
import numpy as np
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
CANON = '/content/drive/MyDrive/cmb_llm_intervention/v3_backup'
OUT   = '/content/drive/MyDrive/cmb_llm_intervention'
NEUTRAL_SYSTEM_PROMPT = "You are a careful analyst. Answer questions strictly from the provided document."
L_PROBE = 17
N_SPLITS = 20

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression

MODELS = [
    ("mistralai/Mistral-7B-Instruct-v0.3", f"{CANON}/baseline_mistral_7b_instruct_v03.json", "mistral_7b_instruct_v03"),
    ("allenai/OLMo-2-1124-7B-Instruct",    f"{CANON}/baseline_olmo_2_1124_7b_instruct.json", "olmo_2_1124_7b_instruct"),
    ("meta-llama/Llama-3.1-8B-Instruct",   f"{CANON}/baseline_llama_31_8b_instruct.json",    "llama_31_8b_instruct"),
]

out = {}
for MNAME, JPATH, SLUG in MODELS:
  print(f"\n{'='*70}\nMODEL: {MNAME}\n{'='*70}")
  recs = json.load(open(JPATH))['records']
  prompts = [r['prompt'] for r in recs]
  y = np.array([1 if r['condition'] == 'uncertain' else 0 for r in recs])
  pair_ids = [r['pair_id'] for r in recs]
  print(f"{len(recs)} canonical prompts, {len(set(pair_ids))} pairs")
  tokenizer = AutoTokenizer.from_pretrained(MNAME)
  model = AutoModelForCausalLM.from_pretrained(MNAME, device_map="auto", torch_dtype=torch.float16)
  model.eval()
  def chat(msg):
      return tokenizer.apply_chat_template(
          [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
           {"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)
  print("Capturing L17 activations (240)...")
  X = np.zeros((len(prompts), model.config.hidden_size), dtype=np.float32)
  for n, p in enumerate(prompts):
      inp = tokenizer(chat(p), return_tensors='pt').to(model.device)
      with torch.no_grad():
          o = model(**inp, output_hidden_states=True, return_dict=True)
      X[n] = o.hidden_states[L_PROBE][0, -1, :].float().cpu().numpy()
      del o
  torch.cuda.empty_cache()

  uniq_pairs = sorted(set(pair_ids))
  rng = np.random.default_rng(23)
  kn_rates, unc_covs = [], []
  for s in range(N_SPLITS):
      perm = rng.permutation(uniq_pairs)
      calib_pairs = set(perm[:len(perm)//2])
      calib = np.array([pid in calib_pairs for pid in pair_ids])
      test = ~calib
      probe = LogisticRegression(C=1.0, max_iter=2000).fit(X[calib], y[calib])
      sc_c = probe.predict_proba(X[calib])[:, 1]
      theta = (np.median(sc_c[y[calib] == 0]) + np.median(sc_c[y[calib] == 1])) / 2.0
      sc_t = probe.predict_proba(X[test])[:, 1]
      kn_fire = float(np.mean(sc_t[y[test] == 0] >= theta))
      unc_cov = float(np.mean(sc_t[y[test] == 1] >= theta))
      kn_rates.append(kn_fire); unc_covs.append(unc_cov)
  kn_rates = np.array(kn_rates); unc_covs = np.array(unc_covs)
  print(f"  HELD-OUT over {N_SPLITS} splits:")
  print(f"    KNOWN gate-fire: mean {100*kn_rates.mean():.1f}%  max {100*kn_rates.max():.1f}%  zero in {int((kn_rates==0).sum())}/{N_SPLITS} splits")
  print(f"    UNCERTAIN coverage: mean {100*unc_covs.mean():.1f}%  min {100*unc_covs.min():.1f}%")
  out[SLUG] = {'known_fire_mean': float(kn_rates.mean()), 'known_fire_max': float(kn_rates.max()),
               'known_fire_zero_splits': int((kn_rates==0).sum()), 'n_splits': N_SPLITS,
               'unc_cov_mean': float(unc_covs.mean()), 'unc_cov_min': float(unc_covs.min()),
               'known_rates': kn_rates.tolist(), 'unc_covs': unc_covs.tolist()}
  json.dump(out, open(f'{OUT}/heldout_theta_3models.json', 'w'), indent=2)
  print(f"  Saved (cumulative) -> heldout_theta_3models.json")
  del model; torch.cuda.empty_cache()
print("\nALL THREE MODELS DONE — PASTE THE PRINTED SUMMARIES BACK TO MrC.")
