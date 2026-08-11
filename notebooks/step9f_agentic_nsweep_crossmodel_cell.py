# =============================================================================
# CMB-LLM Step 9f - Agentic goal-drift: cross-model N-sweep (paper-grade)
# 40 items x 4 families x N in [0,4,12,30] competing sub-tasks.
#   probe (at N=30): instructed vs control, L17 AUC + leakage -> instruction is
#                    internally represented regardless of drift.
#   compliance(N)  : final reply still ends with ZORP.
#   recency(N)     : non-compliant reply ends in a COMPETITOR word (recency capture).
#   decay(N)       : non-compliant reply drops the sentinel without a competitor.
# Shows the gap GROW with interference (the D-driven overrun). Resumable per model.
# RUN ORDER (fresh GPU): bootstrap -> step9c_writefile -> THIS. ~25-35 min.
# =============================================================================
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
N_SWEEP=[0,4,12,30]; PROBE_LAYER=17; SYS="You are a helpful assistant."

import sys, os, json, time, gc
import numpy as np
if '/content/cmb_llm' not in sys.path: sys.path.insert(0,'/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT_DIR='/content/drive/MyDrive/cmb_llm_intervention'; os.makedirs(OUT_DIR, exist_ok=True)

from harness.agentic_interference_pilot import (ITEMS, all_labeled_prompts,
    instructed_prompt, SENTINEL, COMPETITORS)
from harness.inference import load_model
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

def safe(n): return n.split("/")[-1].replace(".","_")
def free_model():
    for nm in ("model","tokenizer"):
        if nm in globals(): del globals()[nm]
    gc.collect(); torch.cuda.empty_cache()
def logo_auc(X,y,g):
    lo=LeaveOneGroupOut(); s=np.zeros(len(y))
    for tr,te in lo.split(X,y,g):
        s[te]=LogisticRegression(C=1.0,max_iter=1000).fit(X[tr],y[tr]).decision_function(X[te])
    return roc_auc_score(y,s)

def run_model(mid):
    mdl,tok=load_model(mid,load_in_4bit=False)
    def chat(msg):
        return tok.apply_chat_template(
            [{"role":"system","content":SYS},{"role":"user","content":msg}],
            tokenize=False, add_generation_prompt=True)
    def act(pt,layer):
        inp=tok(pt,return_tensors="pt").to(mdl.device)
        with torch.no_grad(): out=mdl(**inp,output_hidden_states=True,return_dict=True)
        h=out.hidden_states[layer][0,-1,:].float().cpu().numpy(); del out; torch.cuda.empty_cache(); return h
    def gen(pt):
        inp=tok(pt,return_tensors="pt").to(mdl.device); n=inp["input_ids"].shape[1]
        with torch.no_grad():
            o=mdl.generate(**inp,max_new_tokens=64,do_sample=False,temperature=1.0,pad_token_id=tok.eos_token_id)
        return tok.decode(o[0,n:],skip_special_tokens=True).strip()

    # probe at N=30
    rows=all_labeled_prompts(30); y=np.array([r[1] for r in rows]); g=np.array([r[2] for r in rows])
    X17=np.stack([act(chat(p),PROBE_LAYER) for p,_,_,_ in rows])
    X0 =np.stack([act(chat(p),0) for p,_,_,_ in rows])
    auc=logo_auc(X17,y,g); leak=logo_auc(X0,y,g)

    curve=[]
    for N in N_SWEEP:
        comply=recency=decay=0
        for it in ITEMS:
            resp=gen(chat(instructed_prompt(it,N))); U=resp.upper()
            if SENTINEL in U: comply+=1
            elif any(w in U for w in COMPETITORS): recency+=1
            else: decay+=1
        nI=len(ITEMS)
        curve.append({"N":N,"compliance":comply/nI,"recency":recency/nI,"decay":decay/nI,
                      "gap":1.0-comply/nI})
    free_model()
    return {"model":mid,"probe_auc_N30":float(auc),"leakage":float(leak),"curve":curve}

free_model(); res=[]
for mid in MODELS:
    path=f"{OUT_DIR}/agentic_nsweep_{safe(mid)}.json"
    if os.path.exists(path):
        print(f"[skip] {mid}"); res.append(json.load(open(path))); continue
    print(f"\n{'='*70}\n{mid}\n{'='*70}"); t0=time.time()
    try:
        r=run_model(mid); json.dump(r,open(path,"w"),indent=2); res.append(r)
        print(f"  probe AUC(N=30) {r['probe_auc_N30']:.3f} | leak {r['leakage']:.3f}")
        for c in r["curve"]:
            print(f"    N={c['N']:>2}: compliance {c['compliance']:.3f} | recency {c['recency']:.3f} "
                  f"| decay {c['decay']:.3f} | gap {c['gap']:+.3f}")
        print(f"  ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  ERROR {mid}: {e}"); free_model()

print("\n"+"="*78)
print(f"{'model':<26}{'probe':>7}  compliance by N "+str(N_SWEEP))
print("="*78)
for r in res:
    comps=" ".join(f"{c['compliance']:.2f}" for c in r["curve"])
    print(f"{r['model'].split('/')[-1]:<26}{r['probe_auc_N30']:>7.3f}  [{comps}]")
print("="*78)
print("\ncompliance falling as N rises = D-driven goal-drift gap (probe stays ~1.0).")
print("recency vs decay = the two failure mechanisms per model.")
