# =============================================================================
# CMB-LLM Step 8g - RAG surface-confound test (Qwen): is the AUC=1.000 probe
# epistemic, or just detecting answer-token presence?
# Train probe on supported(0) vs unsupported(1); score DISTRACTOR items (answer
# token present but in an irrelevant role -> answer NOT supported).
#   epistemic -> distractors classified UNSUPPORTED (score > 0).  surface-confound LOW.
#   surface   -> distractors classified SUPPORTED   (score < 0).  surface-confound HIGH.
# RUN ORDER (fresh GPU): bootstrap -> step8f_writefile -> THIS. ~2 min.
# =============================================================================
import sys, os, json, time
import numpy as np
if '/content/cmb_llm' not in sys.path: sys.path.insert(0,'/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT_DIR='/content/drive/MyDrive/cmb_llm_intervention'; os.makedirs(OUT_DIR, exist_ok=True)
PROBE_LAYER=17

from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT
if 'model' not in globals() or 'tokenizer' not in globals():
    from harness.inference import load_model
    print('Loading model...'); model,tokenizer=load_model('Qwen/Qwen2.5-7B-Instruct',load_in_4bit=False)

from harness.rag_distractor_control import ITEMS, train_rows, distractor_prompts
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

def chat(msg):
    return tokenizer.apply_chat_template(
        [{"role":"system","content":NEUTRAL_SYSTEM_PROMPT},{"role":"user","content":msg}],
        tokenize=False, add_generation_prompt=True)
def act(msg):
    inp=tokenizer(chat(msg),return_tensors="pt").to(model.device)
    with torch.no_grad(): out=model(**inp,output_hidden_states=True,return_dict=True)
    h=out.hidden_states[PROBE_LAYER][0,-1,:].float().cpu().numpy(); del out; torch.cuda.empty_cache(); return h

tr=train_rows(); ytr=np.array([l for _,l in tr])
groups=np.repeat(np.arange(len(ITEMS)),2)   # supported,unsupported per item
print(f"Collecting train activations ({len(tr)})...")
Xtr=np.stack([act(p) for p,_ in tr])
# replicate the supported-vs-unsupported AUC (LOGO by item)
lo=LeaveOneGroupOut(); s=np.zeros(len(ytr))
for trn,te in lo.split(Xtr,ytr,groups):
    s[te]=LogisticRegression(C=1.0,max_iter=1000).fit(Xtr[trn],ytr[trn]).decision_function(Xtr[te])
auc=roc_auc_score(ytr,s)

clf=LogisticRegression(C=1.0,max_iter=1000).fit(Xtr,ytr)
sup_sc=clf.decision_function(Xtr[ytr==0]); uns_sc=clf.decision_function(Xtr[ytr==1])
dp=distractor_prompts()
print(f"Collecting distractor activations ({len(dp)})...")
Xd=np.stack([act(p) for _,p,_ in dp]); dsc=clf.decision_function(Xd)

# label 1 = unsupported -> score>0 means classified UNSUPPORTED (epistemic-correct)
surface_confound=float(np.mean(dsc<0))   # distractor wrongly classified SUPPORTED
epistemic=float(np.mean(dsc>0))

print("\n"+"="*68)
print("SURFACE-CONFOUND TEST (does the probe read meaning or token presence?)")
print("="*68)
print(f"  supported/unsupported probe AUC (LOGO):  {auc:.3f}  (replicates the headline)")
print(f"  mean score  supported   : {sup_sc.mean():+.2f}  (target: negative)")
print(f"  mean score  unsupported : {uns_sc.mean():+.2f}  (target: positive)")
print(f"  mean score  DISTRACTOR  : {dsc.mean():+.2f}")
print(f"  distractors classified UNSUPPORTED (epistemic-correct): {epistemic:.3f}  ({int(epistemic*len(dp))}/{len(dp)})")
print(f"  distractors classified SUPPORTED  (surface confound)  : {surface_confound:.3f}")
print("="*68)
json.dump({"auc_sup_unsup":auc,"surface_confound_rate":surface_confound,"epistemic_rate":epistemic,
           "mean_supported":float(sup_sc.mean()),"mean_unsupported":float(uns_sc.mean()),
           "mean_distractor":float(dsc.mean()),
           "per_distractor":[{"pair_id":pid,"hint":h,"score":float(sc),
                              "classified":"unsupported" if sc>0 else "supported"}
                             for (pid,_,h),sc in zip(dp,dsc)]},
          open(f"{OUT_DIR}/rag_distractor_control_results.json","w"),indent=2)
print(f"Saved -> {OUT_DIR}/rag_distractor_control_results.json")

print("\nVERDICT")
if surface_confound<=0.25:
    print("Distractors land with UNSUPPORTED -> the probe reads ANSWER-SUPPORT, not token")
    print("presence. The AUC=1.000 result is epistemic; the reviewers' confound is RULED OUT.")
elif surface_confound>=0.6:
    print("Distractors land with SUPPORTED -> the probe is largely reading TOKEN PRESENCE.")
    print("The headline AUC is confounded; the RAG claim needs reframing.")
else:
    print("Mixed -> partial confound. Inspect per-distractor scores; report honestly.")

print("\n--- per distractor (token present but answer NOT supported) ---")
for (pid,_,h),sc in zip(dp,dsc):
    print(f"  pair={pid:>2} hint={h:<6} score={sc:+.2f} -> {'unsupported (good)' if sc>0 else 'SUPPORTED (confound)'}")
