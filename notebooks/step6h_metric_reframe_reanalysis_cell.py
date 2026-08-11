# =============================================================================
# CMB-LLM Step 6h - Metric reframe: gap as RATE - RATE (no GPU, re-analysis)
# Reviewers: gap = AUC - POS mixes a ranking statistic with a rate. Reframe:
#   theta   = midpoint of median(KNOWN probe scores) and median(UNCERTAIN scores)
#   P_probe = fraction of UNCERTAIN prompts with probe score > theta  (gate would flag)
#   P_gen   = POS hedging rate on UNCERTAIN prompts                    (model surfaces)
#   gap_new = P_probe - P_gen        (both rates on the same prompts)
# Reads saved JSONs only. AUC kept separately as probe separability.
# RUN: any runtime; just needs Drive. ~5 sec.
# =============================================================================
import os, json
import numpy as np
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT='/content/drive/MyDrive/cmb_llm_intervention'

probe=json.load(open(f'{OUT}/uncertainty_scale60_results.json'))   # per_sample: score, condition
gap  =json.load(open(f'{OUT}/expression_gap_results.json'))        # records: condition, expressed_uncertainty

ps=probe['per_sample']
kn=np.array([r['score'] for r in ps if r['condition']=='known'])
un=np.array([r['score'] for r in ps if r['condition']=='uncertain'])
theta=0.5*(np.median(kn)+np.median(un))
P_probe=float(np.mean(un>theta))                    # gate flags uncertain
# also the symmetric KNOWN side (gate should NOT flag known) -> specificity
P_probe_known_fire=float(np.mean(kn>theta))

recs=[r for r in gap['records'] if r['condition']=='uncertain']
P_gen=float(np.mean([r['expressed_uncertainty'] for r in recs]))

from sklearn.metrics import roc_auc_score
y=np.array([0]*len(kn)+[1]*len(un)); s=np.concatenate([kn,un])
auc=roc_auc_score(y,s)
old_gap=auc-P_gen
new_gap=P_probe-P_gen

# subgroups
def sub(sg):
    uns=np.array([r['score'] for r in ps if r['condition']=='uncertain' and r.get('subgrp',sg)==sg])
    # subgrp may not be in probe per_sample; fall back via gap records for POS
    pg=[r['expressed_uncertainty'] for r in recs if r.get('subgrp')==sg]
    return (float(np.mean(uns>theta)) if len(uns) else None,
            float(np.mean(pg)) if pg else None)

print("="*60)
print("METRIC REFRAME  (Qwen, 60-pair uncertainty)")
print("="*60)
print(f"  theta (gate)                 : {theta:+.3f}")
print(f"  probe AUC (separability)     : {auc:.3f}   [stays, but not the gap term]")
print(f"  P_probe  (gate flags UNCERT) : {P_probe:.3f}")
print(f"  P_probe on KNOWN (false fire): {P_probe_known_fire:.3f}   [gate specificity check]")
print(f"  P_gen    (POS hedging UNCERT): {P_gen:.3f}")
print("-"*60)
print(f"  OLD gap = AUC - POS          : {old_gap:+.3f}")
print(f"  NEW gap = P_probe - P_gen    : {new_gap:+.3f}   <-- rate vs rate")
print("="*60)
json.dump({"theta":theta,"auc":auc,"P_probe":P_probe,"P_probe_known_fire":P_probe_known_fire,
           "P_gen":P_gen,"gap_old_auc_minus_pos":old_gap,"gap_new_rate_minus_rate":new_gap},
          open(f'{OUT}/metric_reframe_results.json','w'),indent=2)
print(f"Saved -> {OUT}/metric_reframe_results.json")
print("\nReport P_probe, P_gen, NEW gap back; I update Section 3 + the results tables.")
