# =============================================================================
# WEEK-1 CELL D — Transfer matrix, second family + cosine matrix (GPU, ~10 min)
# Fixes R3 W5: Qwen-only matrix; "shared direction" never tested AS directions.
#
# Runs the 3x3 zero-shot transfer matrix on Mistral-7B-Instruct-v0.3 AND
# computes the cosine-similarity matrix between fitted probe directions for
# BOTH models. Run in a fresh kernel OR after Cell B/C (it reloads the model).
# Requires the same writefiles as Cell B (bootstrap, 6e2, 7a, 8a).
# =============================================================================
import sys, os, json
import numpy as np
if '/content/cmb_llm' not in sys.path: sys.path.insert(0,'/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT='/content/drive/MyDrive/cmb_llm_intervention'
from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT
from harness.uncertainty_scale import all_labeled_prompts as unc_rows
from harness.rag_faithfulness import all_labeled_prompts as rag_rows
from harness.sycophancy_pilot import all_labeled_prompts as syco_rows
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score
from harness.inference import load_model

HELP="You are a helpful assistant."
TASKS={"uncertainty":(unc_rows(),NEUTRAL_SYSTEM_PROMPT),
       "rag":(rag_rows(),NEUTRAL_SYSTEM_PROMPT),
       "sycophancy":(syco_rows(),HELP)}
L_PROBE=17
MODELS=["mistralai/Mistral-7B-Instruct-v0.3","Qwen/Qwen2.5-7B-Instruct"]  # Mistral first (new result)

def run_model(mname):
    global model,tokenizer
    print(f'\n=== {mname} ==='); model,tokenizer=load_model(mname,load_in_4bit=False)
    def act(prompt,sysp):
        pt=tokenizer.apply_chat_template([{"role":"system","content":sysp},
            {"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
        inp=tokenizer(pt,return_tensors="pt").to(model.device)
        with torch.no_grad(): out=model(**inp,output_hidden_states=True,return_dict=True)
        h=out.hidden_states[L_PROBE][0,-1,:].float().cpu().numpy()
        del out; torch.cuda.empty_cache(); return h
    data={}
    for name,(rows,sysp) in TASKS.items():
        print(f'  collecting {name} ({len(rows)})...')
        X=np.stack([act(r[0],sysp) for r in rows])
        data[name]=(X,np.array([r[1] for r in rows]),np.array([r[2] for r in rows]))
    names=list(TASKS); k=len(names)
    M=np.zeros((k,k)); dirs={}
    for i,tr in enumerate(names):
        Xt,yt,gt=data[tr]
        clf=LogisticRegression(C=1.0,max_iter=2000).fit(Xt,yt)
        dirs[tr]=clf.coef_[0]/np.linalg.norm(clf.coef_[0])
        for j,te in enumerate(names):
            Xe,ye,ge=data[te]
            if i==j:
                lo=LeaveOneGroupOut(); s=np.zeros(len(ye))
                for a,b in lo.split(Xe,ye,ge):
                    s[b]=LogisticRegression(C=1.0,max_iter=2000).fit(Xe[a],ye[a]).decision_function(Xe[b])
                M[i,j]=roc_auc_score(ye,s)
            else:
                M[i,j]=roc_auc_score(ye,clf.decision_function(Xe))
    C=np.array([[float(np.dot(dirs[a],dirs[b])) for b in names] for a in names])
    print('  TRANSFER (rows=train, cols=test):')
    print('  '+''.join(n[:10].rjust(12) for n in names))
    for i,n in enumerate(names): print(f'  {n:<12}'+''.join(f'{M[i,j]:>12.3f}' for j in range(k)))
    off=[M[i,j] for i in range(k) for j in range(k) if i!=j]
    print(f'  off-diag mean {np.mean(off):.3f}  min {min(off):.3f}')
    print('  DIRECTION COSINES (the actual "shared direction" test):')
    for i,n in enumerate(names): print(f'  {n:<12}'+''.join(f'{C[i,j]:>12.3f}' for j in range(k)))
    del model; torch.cuda.empty_cache()
    return {'matrix':M.tolist(),'cosines':C.tolist(),'tasks':names}

out={m:run_model(m) for m in MODELS}
json.dump(out,open(f'{OUT}/crosstask_transfer_v2.json','w'),indent=2)
print(f'\nSaved -> {OUT}/crosstask_transfer_v2.json')
print('READ: high off-diag AUC + HIGH cosines (>0.5) => genuinely shared direction.')
print('      high off-diag AUC + LOW cosines => shared separability, not shared direction')
print('      — report cosines either way; that is the honest version of the claim.')
print('PASTE OUTPUT BACK TO MrC.')
