# =============================================================================
# EXTRA CELL F2 — STANDALONE: canonical LOGO-CV probe AUC for OLMo + Llama
# (GPU, ~30 min). NO bootstrap, NO harness, NO other cells needed.
# Paste this whole cell into ANY fresh GPU kernel and run.
# Prompts are read directly from the canonical v3_backup baseline JSONs, so the
# CV AUCs corroborate tab:probe-auc on exactly the canonical 120-pair benchmark.
# =============================================================================
import os, json
import numpy as np
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
CANON = '/content/drive/MyDrive/cmb_llm_intervention/v3_backup'
OUT   = '/content/drive/MyDrive/cmb_llm_intervention'
NEUTRAL_SYSTEM_PROMPT = "You are a careful analyst. Answer questions strictly from the provided document."

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

L_PROBE = 17
MODELS = [
    ("allenai/OLMo-2-1124-7B-Instruct", f"{CANON}/baseline_olmo_2_1124_7b_instruct.json"),
    ("meta-llama/Llama-3.1-8B-Instruct", f"{CANON}/baseline_llama_31_8b_instruct.json"),
]

def logo(X, y, g, mask=None):
    if mask is not None: X, y, g = X[mask], y[mask], g[mask]
    lo = LeaveOneGroupOut(); s = np.zeros(len(y))
    for a, b in lo.split(X, y, g):
        s[b] = LogisticRegression(C=1.0, max_iter=2000).fit(X[a], y[a]).decision_function(X[b])
    return roc_auc_score(y, s)

out = {}
for mname, jpath in MODELS:
    recs = json.load(open(jpath))['records']
    prompts = [r['prompt'] for r in recs]
    y  = np.array([1 if r['condition'] == 'uncertain' else 0 for r in recs])
    g  = np.array([f"{r.get('subgrp','')}_{r['pair_id']}" for r in recs])
    sg = np.array([r.get('subgrp','') for r in recs])
    print(f"\n=== {mname} ===  ({len(recs)} prompts from canonical file)")
    tokenizer = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForCausalLM.from_pretrained(mname, device_map="auto", torch_dtype=torch.float16)
    model.eval()
    def act(prompt):
        pt = tokenizer.apply_chat_template(
            [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
             {"role": "user", "content": prompt}],
            tokenize=False, add_generation_prompt=True)
        inp = tokenizer(pt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            o = model(**inp, output_hidden_states=True, return_dict=True)
        h = o.hidden_states[L_PROBE][0, -1, :].float().cpu().numpy()
        del o; torch.cuda.empty_cache(); return h
    print("  collecting activations...")
    X = np.stack([act(p) for p in prompts])
    res = {"overall": logo(X, y, g),
           "fab": logo(X, y, g, mask=(sg == "fab")),
           "unk": logo(X, y, g, mask=(sg == "unk"))}
    out[mname] = res
    print(f"  LOGO-CV AUC: overall {res['overall']:.3f}  fab {res['fab']:.3f}  unk {res['unk']:.3f}")
    print(f"  (paper tab:probe-auc says: OLMo 0.988/0.966/1.000 | Llama 0.990/0.971/1.000)")
    del model; torch.cuda.empty_cache()

json.dump(out, open(f"{OUT}/cv_auc_olmo_llama.json", "w"), indent=2)
print(f"\nSaved -> {OUT}/cv_auc_olmo_llama.json   PASTE OUTPUT BACK TO MrC.")
