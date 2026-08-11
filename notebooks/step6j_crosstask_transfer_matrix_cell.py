# =============================================================================
# CMB-LLM Step 6j - Cross-task ZERO-SHOT probe transfer matrix (Qwen)
# The keystone test for "universal direction": train the layer-17 probe on one
# task, apply it WITHOUT refitting to the others. High off-diagonal AUC => a
# shared epistemic direction (claim "single direction" earns it). Off-diagonal
# ~0.5 => probe is a per-task locus, and we keep the honest "universal recipe"
# framing. Labels are aligned: 1 = the epistemically-problematic condition
# (uncertain / unsupported / user-asserts-falsehood).
#
# RUN ORDER (fresh GPU): bootstrap -> step6e2_writefile (uncertainty_scale)
#   -> step7a_writefile (sycophancy) -> step8a_writefile (rag_faithfulness) -> THIS. ~2-3 min.
# =============================================================================
import sys, os, json
import numpy as np
if '/content/cmb_llm' not in sys.path: sys.path.insert(0,'/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT='/content/drive/MyDrive/cmb_llm_intervention'; os.makedirs(OUT, exist_ok=True)
PROBE_LAYER=17

from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT
from harness.uncertainty_scale import all_labeled_prompts as unc_rows
from harness.rag_faithfulness import all_labeled_prompts as rag_rows
from harness.sycophancy_pilot import all_labeled_prompts as syco_rows
if 'model' not in globals() or 'tokenizer' not in globals():
    from harness.inference import load_model
    print('Loading model...'); model,tokenizer=load_model('Qwen/Qwen2.5-7B-Instruct',load_in_4bit=False)
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

HELP="You are a helpful assistant."
# (rows, system_prompt). uncertainty/rag are document-framed; sycophancy is plain.
TASKS={
 "uncertainty": (unc_rows(),      NEUTRAL_SYSTEM_PROMPT),
 "rag":         (rag_rows(),      NEUTRAL_SYSTEM_PROMPT),
 "sycophancy":  (syco_rows(),     HELP),
}

def act(prompt, sysp):
    pt=tokenizer.apply_chat_template([{"role":"system","content":sysp},
        {"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    inp=tokenizer(pt,return_tensors="pt").to(model.device)
    with torch.no_grad(): out=model(**inp,output_hidden_states=True,return_dict=True)
    h=out.hidden_states[PROBE_LAYER][0,-1,:].float().cpu().numpy(); del out; torch.cuda.empty_cache(); return h

data={}
for name,(rows,sysp) in TASKS.items():
    print(f"collecting {name} ({len(rows)} prompts)...")
    X=np.stack([act(r[0],sysp) for r in rows])
    y=np.array([r[1] for r in rows]); g=np.array([r[2] for r in rows])
    data[name]=(X,y,g)

names=list(TASKS)
M=np.zeros((len(names),len(names)))
for i,tr in enumerate(names):
    Xtr,ytr,gtr=data[tr]
    clf=LogisticRegression(C=1.0,max_iter=1000).fit(Xtr,ytr)
    for j,te in enumerate(names):
        Xte,yte,gte=data[te]
        if tr==te:   # in-task: honest LOGO-CV
            lo=LeaveOneGroupOut(); s=np.zeros(len(yte))
            for a,b in lo.split(Xte,yte,gte):
                s[b]=LogisticRegression(C=1.0,max_iter=1000).fit(Xte[a],yte[a]).decision_function(Xte[b])
            M[i,j]=roc_auc_score(yte,s)
        else:        # zero-shot transfer
            M[i,j]=roc_auc_score(yte,clf.decision_function(Xte))

print("\n"+"="*60)
print("CROSS-TASK PROBE TRANSFER  (rows = trained on, cols = tested on)")
print("="*60)
print("train/test".ljust(14)+"".join(n[:10].rjust(12) for n in names))
for i,n in enumerate(names):
    print(f"{n:<14}"+"".join(f"{M[i,j]:>12.3f}" for j in range(len(names))))
print("="*60)
off=[M[i,j] for i in range(len(names)) for j in range(len(names)) if i!=j]
print(f"diagonal (in-task) mean: {np.mean([M[i,i] for i in range(len(names))]):.3f}")
print(f"off-diagonal (zero-shot transfer) mean: {np.mean(off):.3f}  min {min(off):.3f}  max {max(off):.3f}")
json.dump({"tasks":names,"matrix":M.tolist()},open(f'{OUT}/crosstask_transfer_matrix.json','w'),indent=2)
print(f"Saved -> {OUT}/crosstask_transfer_matrix.json")
print("\nREAD: off-diagonal >~0.75 => a SHARED epistemic direction (earns 'single direction').")
print("      off-diagonal ~0.5  => per-task directions; keep 'universal locus/recipe'.")
