# =============================================================================
# WEEK-1 CELL C — Held-out theta calibration (GPU, ~5 min; same kernel as B)
# Fixes R3 fatal #2: "zero FP cost" was an in-sample property of theta.
#
# 50/50 pair-level split, 20 seeds: fit probe + theta (KNOWN/UNCERTAIN median
# midpoint) on the CALIBRATION half only; report gate behavior on the TEST half.
# Because intervention output on a non-gated prompt is bit-identical to
# baseline, the held-out KNOWN gate-fire rate IS the held-out FP exposure.
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
if 'model' not in globals() or 'tokenizer' not in globals():
    from harness.inference import load_model
    print('Loading model...'); model,tokenizer=load_model('Qwen/Qwen2.5-7B-Instruct',load_in_4bit=False)
import torch
from sklearn.linear_model import LogisticRegression
L_PROBE=17

rows=unc_rows()
def act(prompt):
    pt=tokenizer.apply_chat_template([{"role":"system","content":NEUTRAL_SYSTEM_PROMPT},
        {"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    inp=tokenizer(pt,return_tensors="pt").to(model.device)
    with torch.no_grad(): out=model(**inp,output_hidden_states=True,return_dict=True)
    h=out.hidden_states[L_PROBE][0,-1,:].float().cpu().numpy()
    del out; torch.cuda.empty_cache(); return h

# reuse activations if Cell B already collected them for uncertainty
print('collecting activations (120 prompts)...')
X=np.stack([act(r[0]) for r in rows]); y=np.array([r[1] for r in rows]); g=np.array([r[2] for r in rows])

pair_ids=sorted(set(g)); rng=np.random.RandomState(23)
res=[]
for seed in range(20):
    perm=rng.permutation(pair_ids); half=len(perm)//2
    cal=set(perm[:half]); tst=set(perm[half:])
    ci=np.array([i for i,pid in enumerate(g) if pid in cal])
    ti=np.array([i for i,pid in enumerate(g) if pid in tst])
    clf=LogisticRegression(C=1.0,max_iter=2000).fit(X[ci],y[ci])
    s_cal=clf.decision_function(X[ci]); s_tst=clf.decision_function(X[ti])
    theta=(np.median(s_cal[y[ci]==0])+np.median(s_cal[y[ci]==1]))/2
    known_t=(y[ti]==0); unc_t=(y[ti]==1)
    fp_gate=float(np.mean(s_tst[known_t]>theta))     # held-out KNOWN gate-fire = FP exposure
    tp_gate=float(np.mean(s_tst[unc_t]>theta))       # held-out UNCERTAIN gate coverage
    res.append((fp_gate,tp_gate))
fp=np.array([r[0] for r in res]); tp=np.array([r[1] for r in res])
print('\nHELD-OUT THETA (20 random 50/50 pair splits):')
print(f'  KNOWN gate-fire (FP exposure): mean {fp.mean()*100:.2f}%  max {fp.max()*100:.2f}%  '
      f'(splits with zero: {int((fp==0).sum())}/20)')
print(f'  UNCERTAIN gate coverage:       mean {tp.mean()*100:.2f}%  min {tp.min()*100:.2f}%')
print('\nPAPER TEXT: if mean FP ~0 and coverage ~>=95%, the zero-FP property is held-out-')
print('confirmed at the gate level (state it that way). If FP > 0, report the number and')
print('soften "zero false-positive cost" to the measured held-out rate.')
json.dump({'fp_gate':fp.tolist(),'tp_gate':tp.tolist()},open(f'{OUT}/heldout_theta.json','w'),indent=2)
print(f'Saved -> {OUT}/heldout_theta.json   PASTE OUTPUT BACK TO MrC.')
