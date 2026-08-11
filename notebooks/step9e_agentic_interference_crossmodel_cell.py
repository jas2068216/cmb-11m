# =============================================================================
# CMB-LLM Step 9e - Agentic drift via INTERFERENCE, CROSS-MODEL (N=30 max pressure)
# Qwen held under both length and interference. RAG showed Qwen is the robust
# outlier; OLMo is the weak one. This screens all four families at the strongest
# interference level for instruction-retention drift.
#   per model: probe AUC (instructed vs control) | leak | compliance (ZORP kept)
#              | drift->competitor | gap = AUC - compliance
# Resumable per model; per-model JSON saved. VERIFY model ids (esp. OLMo) + HF
# token for Llama. RUN ORDER (fresh GPU): bootstrap -> step9c_writefile -> THIS.
# ~12-20 min.
# =============================================================================
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
N_SEG = 30; PROBE_LAYER = 17; SYS = "You are a helpful assistant."

import sys, os, json, time, gc
import numpy as np
if '/content/cmb_llm' not in sys.path: sys.path.insert(0, '/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT_DIR = '/content/drive/MyDrive/cmb_llm_intervention'; os.makedirs(OUT_DIR, exist_ok=True)

from harness.agentic_interference_pilot import (ITEMS, all_labeled_prompts,
    instructed_prompt, SENTINEL, COMPETITORS)
from harness.inference import load_model
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

rows = all_labeled_prompts(N_SEG)
y = np.array([r[1] for r in rows]); g = np.array([r[2] for r in rows])

def safe(n): return n.split("/")[-1].replace(".", "_")
def free_model():
    for nm in ("model","tokenizer"):
        if nm in globals(): del globals()[nm]
    gc.collect(); torch.cuda.empty_cache()
def logo_auc(X,yy,gg):
    lo=LeaveOneGroupOut(); s=np.zeros(len(yy))
    for tr,te in lo.split(X,yy,gg):
        s[te]=LogisticRegression(C=1.0,max_iter=1000).fit(X[tr],yy[tr]).decision_function(X[te])
    return roc_auc_score(yy,s)

def run_model(mid):
    mdl, tok = load_model(mid, load_in_4bit=False)
    def chat(msg):
        return tok.apply_chat_template(
            [{"role":"system","content":SYS},{"role":"user","content":msg}],
            tokenize=False, add_generation_prompt=True)
    def act(pt, layer):
        inp=tok(pt,return_tensors="pt").to(mdl.device)
        with torch.no_grad(): out=mdl(**inp,output_hidden_states=True,return_dict=True)
        h=out.hidden_states[layer][0,-1,:].float().cpu().numpy(); del out; torch.cuda.empty_cache(); return h
    X17=[]; X0=[]
    for prompt,label,pid,cond in rows:
        X17.append(act(chat(prompt),PROBE_LAYER)); X0.append(act(chat(prompt),0))
    X17=np.stack(X17); X0=np.stack(X0)
    auc=logo_auc(X17,y,g); leak=logo_auc(X0,y,g)
    comply=0; drift=0; recs=[]
    for it in ITEMS:
        inp=tok(chat(instructed_prompt(it,N_SEG)),return_tensors="pt").to(mdl.device); n=inp["input_ids"].shape[1]
        with torch.no_grad():
            o=mdl.generate(**inp,max_new_tokens=64,do_sample=False,temperature=1.0,pad_token_id=tok.eos_token_id)
        resp=tok.decode(o[0,n:],skip_special_tokens=True).strip(); U=resp.upper()
        c=SENTINEL in U; dr=(not c) and any(w in U for w in COMPETITORS)
        comply+=c; drift+=dr
        recs.append({"pair_id":it.pair_id,"complied":bool(c),"drift_competitor":bool(dr),"response":resp[:160]})
    nI=len(ITEMS); comply/=nI; drift/=nI
    free_model()
    return {"model":mid,"N":N_SEG,"probe_auc":float(auc),"leakage_auc":float(leak),
            "compliance":comply,"drift_to_competitor":drift,"gap":float(auc)-comply,"records":recs}

free_model()
res=[]
for mid in MODELS:
    path=f"{OUT_DIR}/agentic_interference_xmodel_{safe(mid)}.json"
    if os.path.exists(path):
        print(f"[skip] {mid}"); res.append(json.load(open(path))); continue
    print(f"\n{'='*70}\n{mid}\n{'='*70}"); t0=time.time()
    try:
        r=run_model(mid); json.dump(r,open(path,"w"),indent=2); res.append(r)
        print(f"  probe {r['probe_auc']:.3f} | leak {r['leakage_auc']:.3f} | compliance {r['compliance']:.3f} "
              f"| drift->comp {r['drift_to_competitor']:.3f} | gap {r['gap']:+.3f}  ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  ERROR {mid}: {e}"); free_model()

print("\n"+"="*88)
print(f"{'model':<30}{'probeAUC':>10}{'leak':>7}{'compliance':>12}{'drift':>8}{'gap':>8}")
print("="*88)
for r in res:
    print(f"{r['model'].split('/')[-1]:<30}{r['probe_auc']:>10.3f}{r['leakage_auc']:>7.3f}"
          f"{r['compliance']:>12.3f}{r['drift_to_competitor']:>8.3f}{r['gap']:>+8.3f}")
print("="*88)
print("\ngap>0 on any model = model-dependent agentic-drift (parallel to RAG heterogeneity).")
print("all compliance ~1.0 = robust cross-model negative; agentic drift joins sycophancy.")
