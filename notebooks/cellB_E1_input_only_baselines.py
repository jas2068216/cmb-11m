# =============================================================================
# WEEK-1 CELL B — E1: the deciding experiment (GPU, ~10-15 min)
# R3's "single most damaging point": can input-only baselines match the probe?
#
# Compares, on V3 uncertainty / RAG faithfulness / sycophancy (LOGO-CV by pair):
#   1. TF-IDF text classifier on the RAW PROMPT (no model access at all)
#   2. Prompt log-perplexity under the SAME model (no internal access)
#   3. Layer-0 MEAN-over-tokens probe (the real lexical control replacing
#      the vacuous last-token layer-0 row)
#   4. Layer-17 last-token probe (the paper's method, recomputed same split)
#
# RUN ORDER (fresh GPU kernel): pipeline_bootstrap_cell.py ->
#   step6e2_writefile -> step7a_writefile -> step8a_writefile -> THIS.
# =============================================================================
import sys, os, json
import numpy as np
if '/content/cmb_llm' not in sys.path: sys.path.insert(0,'/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT='/content/drive/MyDrive/cmb_llm_intervention'; os.makedirs(OUT,exist_ok=True)

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
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import hstack

HELP="You are a helpful assistant."
TASKS={"uncertainty":(unc_rows(),NEUTRAL_SYSTEM_PROMPT),
       "rag":(rag_rows(),NEUTRAL_SYSTEM_PROMPT),
       "sycophancy":(syco_rows(),HELP)}
L_PROBE=17

def acts(prompt,sysp):
    pt=tokenizer.apply_chat_template([{"role":"system","content":sysp},
        {"role":"user","content":prompt}],tokenize=False,add_generation_prompt=True)
    inp=tokenizer(pt,return_tensors="pt").to(model.device)
    with torch.no_grad(): out=model(**inp,output_hidden_states=True,return_dict=True)
    h17=out.hidden_states[L_PROBE][0,-1,:].float().cpu().numpy()
    h0m=out.hidden_states[0][0].float().mean(dim=0).cpu().numpy()   # MEAN over all tokens
    # prompt log-perplexity (user prompt tokens only, plain — no chat template,
    # so the score reflects the text itself, not template interactions)
    ids=tokenizer(prompt,return_tensors="pt").to(model.device)
    with torch.no_grad(): o2=model(**ids,labels=ids['input_ids'])
    nll=o2.loss.item()
    del out,o2; torch.cuda.empty_cache()
    return h17,h0m,nll

def logo_auc(X,y,g):
    lo=LeaveOneGroupOut(); s=np.zeros(len(y))
    for a,b in lo.split(X,y,g):
        s[b]=LogisticRegression(C=1.0,max_iter=2000).fit(X[a],y[a]).decision_function(X[b])
    return roc_auc_score(y,s)

def logo_auc_sparse(texts,y,g):
    lo=LeaveOneGroupOut(); s=np.zeros(len(y)); texts=np.array(texts,dtype=object)
    for a,b in lo.split(np.zeros(len(y)),y,g):
        vw=TfidfVectorizer(ngram_range=(1,2),min_df=1)
        vc=TfidfVectorizer(analyzer='char_wb',ngram_range=(3,5),min_df=1)
        Xa=hstack([vw.fit_transform(texts[a]),vc.fit_transform(texts[a])])
        Xb=hstack([vw.transform(texts[b]),vc.transform(texts[b])])
        s[b]=LogisticRegression(C=1.0,max_iter=2000).fit(Xa,y[a]).decision_function(Xb)
    return roc_auc_score(y,s)

results={}
print(f"{'task':<13}{'n':<6}{'TF-IDF':<9}{'PPL':<9}{'L0-mean':<9}{'L17 probe':<10}")
for name,(rows,sysp) in TASKS.items():
    prompts=[r[0] for r in rows]; y=np.array([r[1] for r in rows]); g=np.array([r[2] for r in rows])
    H17,H0,NLL=[],[],[]
    for p in prompts:
        a,b,c=acts(p,sysp); H17.append(a); H0.append(b); NLL.append(c)
    H17=np.stack(H17); H0=np.stack(H0); NLL=np.array(NLL)
    a_tfidf=logo_auc_sparse(prompts,y,g)
    a_ppl=max(roc_auc_score(y,NLL),roc_auc_score(y,-NLL))   # direction-free
    a_l0=logo_auc(H0,y,g)
    a_l17=logo_auc(H17,y,g)
    results[name]={'n':len(y),'tfidf':a_tfidf,'ppl':a_ppl,'l0_mean':a_l0,'l17_probe':a_l17}
    print(f"{name:<13}{len(y):<6}{a_tfidf:<9.3f}{a_ppl:<9.3f}{a_l0:<9.3f}{a_l17:<10.3f}")

print("\nREAD:")
print("  input-only (TF-IDF / PPL / L0-mean) ~ L17  => deflationary reading stands;")
print("      reframe the representational claim; intervention/gating/dissociation survive.")
print("  L17 >> input-only (esp. where TF-IDF < 0.8) => the epistemic claim is EARNED;")
print("      report this table as the new lexical control and delete the old L0 row.")
json.dump(results,open(f'{OUT}/e1_input_only_baselines.json','w'),indent=2)
print(f"\nSaved -> {OUT}/e1_input_only_baselines.json   PASTE OUTPUT BACK TO MrC.")
