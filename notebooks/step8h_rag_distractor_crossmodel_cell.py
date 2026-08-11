# =============================================================================
# CMB-LLM Step 8h - RAG surface-confound control, CROSS-MODEL (4 families)
# Qwen result: AUC 1.000, surface-confound 0.062 (probe is epistemic). The
# AUC=1.000 claim spans all four families, so this rebuttal must too.
# Per model: train probe supported(0) vs unsupported(1); score distractors.
#   surface_confound = fraction of distractors classified SUPPORTED (token-presence).
#   epistemic        = fraction classified UNSUPPORTED (answer-support). LOW confound = good.
# Resumable per model. RUN ORDER (fresh GPU): bootstrap -> step8f_writefile -> THIS. ~8-12 min.
# =============================================================================
MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "allenai/OLMo-2-1124-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
]
PROBE_LAYER=17
import sys, os, json, time, gc
import numpy as np
if '/content/cmb_llm' not in sys.path: sys.path.insert(0,'/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT_DIR='/content/drive/MyDrive/cmb_llm_intervention'; os.makedirs(OUT_DIR, exist_ok=True)

from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT
from harness.rag_distractor_control import ITEMS, train_rows, distractor_prompts
from harness.inference import load_model
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

TR=train_rows(); YTR=np.array([l for _,l in TR]); GROUPS=np.repeat(np.arange(len(ITEMS)),2)
DP=distractor_prompts()
def safe(n): return n.split("/")[-1].replace(".","_")
def free_model():
    for nm in ("model","tokenizer"):
        if nm in globals(): del globals()[nm]
    gc.collect(); torch.cuda.empty_cache()

def run_model(mid):
    mdl,tok=load_model(mid,load_in_4bit=False)
    def act(msg):
        pt=tok.apply_chat_template([{"role":"system","content":NEUTRAL_SYSTEM_PROMPT},
            {"role":"user","content":msg}],tokenize=False,add_generation_prompt=True)
        inp=tok(pt,return_tensors="pt").to(mdl.device)
        with torch.no_grad(): out=mdl(**inp,output_hidden_states=True,return_dict=True)
        h=out.hidden_states[PROBE_LAYER][0,-1,:].float().cpu().numpy(); del out; torch.cuda.empty_cache(); return h
    Xtr=np.stack([act(p) for p,_ in TR])
    lo=LeaveOneGroupOut(); s=np.zeros(len(YTR))
    for trn,te in lo.split(Xtr,YTR,GROUPS):
        s[te]=LogisticRegression(C=1.0,max_iter=1000).fit(Xtr[trn],YTR[trn]).decision_function(Xtr[te])
    auc=roc_auc_score(YTR,s)
    clf=LogisticRegression(C=1.0,max_iter=1000).fit(Xtr,YTR)
    Xd=np.stack([act(p) for _,p,_ in DP]); dsc=clf.decision_function(Xd)
    free_model()
    return {"model":mid,"auc_sup_unsup":float(auc),
            "surface_confound_rate":float(np.mean(dsc<0)),
            "epistemic_rate":float(np.mean(dsc>0)),
            "mean_distractor":float(dsc.mean())}

free_model(); res=[]
for mid in MODELS:
    path=f"{OUT_DIR}/rag_distractor_xmodel_{safe(mid)}.json"
    if os.path.exists(path):
        print(f"[skip] {mid}"); res.append(json.load(open(path))); continue
    print(f"\n{'='*66}\n{mid}\n{'='*66}"); t0=time.time()
    try:
        r=run_model(mid); json.dump(r,open(path,"w"),indent=2); res.append(r)
        print(f"  AUC {r['auc_sup_unsup']:.3f} | surface-confound {r['surface_confound_rate']:.3f} "
              f"| epistemic {r['epistemic_rate']:.3f} | mean-distractor {r['mean_distractor']:+.2f}  ({time.time()-t0:.0f}s)")
    except Exception as e:
        print(f"  ERROR {mid}: {e}"); free_model()

print("\n"+"="*72)
print(f"{'model':<28}{'AUC':>7}{'surface-confound':>18}{'epistemic':>11}")
print("="*72)
for r in res:
    print(f"{r['model'].split('/')[-1]:<28}{r['auc_sup_unsup']:>7.3f}{r['surface_confound_rate']:>18.3f}{r['epistemic_rate']:>11.3f}")
print("="*72)
print("\nsurface-confound LOW on all models = the AUC=1.000 RAG probe is epistemic")
print("(reads answer-support, not token presence) across the whole cohort.")
