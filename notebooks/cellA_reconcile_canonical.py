# =============================================================================
# WEEK-1 CELL A — Canonical reconciliation (NO GPU NEEDED, run first, ~1 min)
# Fixes R3 fatal #1: "two scoring runs mixed in headline tables."
#
# What it does:
#   1. Finds EVERY baseline_*.json / intervention_*.json under MyDrive
#      (they live in at least two folders — this cell surfaces all candidates).
#   2. Picks ONE canonical file per (model, kind): most records, then newest.
#   3. Re-scores every stored response with the ONE canonical correction-aware
#      regex below (same as step6q_v2).
#   4. Recomputes FROM RAW RECORDS: probe AUC (from stored probe_score — this
#      settles tab:probe-auc 0.969 vs tab:gap 0.997), baseline POS,
#      intervention POS, lift, paired SE, KNOWN FP + delta, and the full
#      meta-analysis (FE / DL / HKSJ, BOTH HKSJ variants — settles W8).
#   5. Writes canonical_results.json + prints the reconciliation report.
# Paste the printed report back to MrC.
# =============================================================================
import os, json, glob, re, math
from collections import defaultdict

try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
MYDRIVE = '/content/drive/MyDrive'
OUT = f'{MYDRIVE}/cmb_llm_intervention'

# ---- THE canonical correction-aware regex (step6q_v2, verbatim) -------------
OLD_PATTERNS = [
    r"\bi don't (know|have)\b", r"\bi do not (know|have)\b",
    r"\bi'm not (sure|certain|aware)\b", r"\bi am not (sure|certain|aware)\b",
    r"\bi'm unable to\b", r"\bi am unable to\b",
    r"\bi cannot (provide|determine|confirm|verify|answer|say)\b",
    r"\bi can't (provide|determine|confirm|verify|answer|say)\b",
    r"\bunable to (confirm|verify|determine|answer)\b",
    r"\bno (record|reliable|definitive|verifiable|known|documented|public)\b",
    r"\bnot (recorded|documented|publicly known|available|preserved|captured)\b",
    r"\bwithout (access|reliable|verifiable)\b",
    r"\b(there is|there's) no (record|evidence|information|documentation)\b",
    r"\bdoes not exist\b",
    r"\bnever (occurred|happened|published|recorded|existed|wrote|composed)\b",
    r"\bnot a (real|known|documented|published)\b",
    r"\bthis (premise|presupposition|question|claim) (is|appears) (false|incorrect|inaccurate)\b",
    r"\bthe premise\b.*\b(false|incorrect|inaccurate|wrong|mistaken)\b",
    r"\bappears to (be )?(false|incorrect|fabricat|fiction)\b",
    r"\b(no such|fictional|fabricated|fictitious|made-up|invented)\b",
    r"\b(i must|i should|i need to) (clarify|correct|note|point out)\b",
    r"\bi (cannot|can't) confirm\b",
    r"\bbased on .* (i don't|i do not|i'm not|i am not)\b",
    r"\bnot (something|information|data) (i|that)\b",
]
P1 = r"\bno\s+(\w+\s+){0,3}(record|records|evidence|information|details?|data|documentation|account|accounts|source|sources|publication|publications|note|notes|reference|references|mention|mentions|knowledge|history|writing|writings)\b"
P2 = r"\b(no\s+\w*\s*(record|records|evidence|information|details?|data|documentation|account|accounts|source|sources|reference|references)|none\s+of\s+\w+\s+\w+)\s+(is|are|was|were)\s+(provided|available|recorded|documented|known|public|cited|listed|mentioned|preserved|captured)\b"
P3_RAW = r"\bthe\s+(document|text|passage|passages|source|sources|provided\s+information|context)\s+(does\s+not|doesn'?t|do\s+not|don'?t)\s+(provide|contain|mention|specify|detail|include|describe|reference|address|cover|state)\b"
P5 = r"\b(unfortunately|regrettably|sadly)\b[^.]{0,80}?\b(no|cannot|unable|don'?t|do\s+not|n'?t)\b"
P10 = r"\b(actually|in\s+fact|to\s+clarify|to\s+correct|to\s+note)\b[^.]{0,40}\b(no|never|not|incorrect|wrong|false|fictional|fabricated)\b"
P12 = r"\bno\s+(information|details?|evidence|record|records|documentation|data)\s+(about|of|regarding|on|for|concerning)\b"
BRIDGE_PATTERNS = [
    r"\bhowever\b[^.]{0,100}\bit\s+(is|'?s)\s+(widely|commonly|generally|well|broadly)\s+(known|recognized|accepted|understood)\b",
    r"\b(however|but|nevertheless|nonetheless)\b[^.]{0,80}\bthe\s+(correct|actual|established|commonly|widely|true|accepted|right|real)\b",
    r"\bit\s+(is|'?s)\s+(widely|commonly|generally|well|broadly)\s+(known|recognized|accepted|understood)\b",
    r"\bin\s+(fact|reality|truth)\b",
    r"\bis\s+(widely|commonly|generally)\s+(recognized|accepted|known|established)\s+as\b",
    r"\bthe\s+(correct|actual|true|established|accepted|right)\s+answer\s+is\b",
    r"\bthe\s+commonly\s+recognized\b",
]
CORRECTION_PATTERNS = [
    r"\bthere\s+(seems|appears)\s+to\s+be\s+(some|an?\s+|the\s+)?(incorrect|inaccurate|wrong|mistaken|false|misleading|missing)\s+(information|claim|statement|premise|assumption|detail|fact)\b",
    r"\bthere\s+(is|are)\s+(some|an?\s+|the\s+)?(incorrect|inaccurate|wrong|mistaken|false|misleading|missing)\s+(information|claim|statement|premise|assumption|detail|fact)\b",
    r"\b(this|that|your\s+(statement|claim|premise|assumption|question|prompt|information))\s+(is|appears|seems|contains)\s+(incorrect|inaccurate|wrong|mistaken|false)\b",
    r"\bthe\s+(correct|actual|true|accurate|right)\s+(answer|information|date|location|name|year|fact|premise|version|details?)\s+(is|was|are|were)\b",
    r"\b(actually|in\s+fact|in\s+reality|in\s+truth)\b[^.]{0,60}\b(is|was|were|are|did|does)\s+(not|never|no)\b",
    r"\b(however|but|nevertheless)\b[^.]{0,60}\b(actually|in\s+fact)\b",
    r"\b(is|was|were|are)\s+not\s+(actually|in\s+fact|in\s+reality|correct|accurate|true|right)\b",
    r"\bno\s+such\s+(event|occurrence|incident|paper|book|symphony|painting|record|reference|conversation|meeting|discovery|invention|publication|article|study|individual|person|kingdom|dynasty|composition|work|piece|opera|attempt|election|founder|co-founder|cofounder)\b",
    r"\bthere\s+(is|was|are|were)\s+no\s+(such|specific|particular|known|documented|verified)\b",
    r"\b(to\s+clarify|let\s+me\s+correct|i\s+need\s+to\s+correct|i\s+should\s+correct|allow\s+me\s+to\s+clarify|i\s+must\s+correct)\b",
    r"\b(misconception|misinformation|inaccuracy|inaccuracies|mistaken\s+belief)\b",
    r"\b(did|do|does)\s+not\s+exist\b",
    r"\bnever\s+(existed|happened|occurred|took\s+place|published|wrote|composed|invented|discovered|won|received)\b",
    r"\b(the\s+(date|year|location|name|claim|premise|statement|assumption|year|number)|that\s+\w+)\s+is\s+(incorrect|inaccurate|wrong|false|a\s+misconception)\b",
    r"\b(i'm\s+sorry|i\s+am\s+sorry)\s*,?\s+but\s+(i|i'?m|i\s+am)\b",
    r"\b(only|just)\s+(composed|wrote|published|painted|invented|discovered|created|made|recorded)\s+(\d+|nine|eight|seven|six|five|four|three|two|one)\b",
]
OLD_RE=re.compile("|".join(OLD_PATTERNS),re.IGNORECASE)
KEPT_RE=re.compile("|".join([P1,P2,P5,P10,P12]),re.IGNORECASE)
P3_RE=re.compile(P3_RAW,re.IGNORECASE); BRIDGE_RE=re.compile("|".join(BRIDGE_PATTERNS),re.IGNORECASE)
CORR_RE=re.compile("|".join(CORRECTION_PATTERNS),re.IGNORECASE)
def flagged(t):
    if OLD_RE.search(t) or KEPT_RE.search(t): return True
    if P3_RE.search(t) and not BRIDGE_RE.search(t): return True
    return bool(CORR_RE.search(t))

# ---- 1. inventory every candidate file --------------------------------------
def find(kind):
    hits=[]
    for p in glob.glob(f'{MYDRIVE}/**/{kind}_*.json', recursive=True):
        try:
            d=json.load(open(p)); recs=d.get('records',[])
            hits.append({'path':p,'n':len(recs),'mtime':os.path.getmtime(p),
                         'model':d.get('model','?'),'data':d})
        except Exception as e: print(f'  ! unreadable {p}: {e}')
    return hits

print('='*80); print('INVENTORY — every candidate results file found on Drive'); print('='*80)
inv={'baseline':find('baseline'),'intervention':find('intervention')}
by_model=defaultdict(dict)
for kind,hits in inv.items():
    for h in sorted(hits,key=lambda x:(-x['n'],-x['mtime'])):
        print(f"  {kind:<13} {h['model']:<40} n={h['n']:<4} {h['path'].replace(MYDRIVE,'~')}")
        if kind not in by_model[h['model']]:      # first = most records, newest
            by_model[h['model']][kind]=h
print('\nCANONICAL CHOICES (most records, then newest — all others are the stale run):')
for m,kk in by_model.items():
    for kind,h in kk.items(): print(f"  {m:<40} {kind:<13} n={h['n']}  {h['path'].replace(MYDRIVE,'~')}")

# ---- 2. recompute everything from raw records -------------------------------
def auc(scores,labels):
    pairs=sorted(zip(scores,labels)); pos=sum(labels); neg=len(labels)-pos
    if not pos or not neg: return float('nan')
    rank=0; i=0
    # rank-sum AUC with tie handling
    import itertools
    s=sorted(range(len(scores)),key=lambda i:scores[i])
    ranks=[0]*len(scores); i=0
    while i<len(s):
        j=i
        while j+1<len(s) and scores[s[j+1]]==scores[s[i]]: j+=1
        r=(i+j)/2+1
        for k in range(i,j+1): ranks[s[k]]=r
        i=j+1
    rp=sum(ranks[i] for i in range(len(labels)) if labels[i])
    return (rp-pos*(pos+1)/2)/(pos*neg)

print('\n'+'='*80); print('CANONICAL PER-MODEL TABLE (one scorer, one run — paste into paper)'); print('='*80)
rows=[]
for m,kk in sorted(by_model.items()):
    if 'baseline' not in kk or 'intervention' not in kk:
        print(f'  ! {m}: missing a file, skipped'); continue
    b=kk['baseline']['data']['records']; iv=kk['intervention']['data']['records']
    key=lambda r:(r['pair_id'],r.get('subgrp',''),r['condition'])
    bb={key(r):r for r in b}; ii={key(r):r for r in iv}
    common=sorted(set(bb)&set(ii))
    unc=[k for k in common if k[2]=='uncertain']; kno=[k for k in common if k[2]=='known']
    # probe AUC from stored scores (baseline file)
    sc=[bb[k]['probe_score'] for k in common]; lb=[1 if k[2]=='uncertain' else 0 for k in common]
    A=auc(sc,lb)
    bpos=sum(flagged(bb[k]['response']) for k in unc)/len(unc)
    ipos=sum(flagged(ii[k]['response']) for k in unc)/len(unc)
    bfp =sum(flagged(bb[k]['response']) for k in kno)/len(kno)
    ifp =sum(flagged(ii[k]['response']) for k in kno)/len(kno)
    diffs=[(1 if flagged(ii[k]['response']) else 0)-(1 if flagged(bb[k]['response']) else 0) for k in unc]
    md=sum(diffs)/len(diffs); var=sum((d-md)**2 for d in diffs)/(len(diffs)-1)
    se=(var/len(diffs))**.5 if var>0 else 1e-9
    rows.append({'model':m,'n_unc':len(unc),'probe_auc':A,'base_pos':bpos,'int_pos':ipos,
                 'lift':md,'se':se,'base_fp':bfp,'int_fp':ifp,'fp_delta':ifp-bfp,
                 'gap_auc_minus_pos':A-bpos,'gap_rate':None})
    print(f"  {m:<38} AUC={A:.3f}  basePOS={bpos*100:5.1f}%  intPOS={ipos*100:5.1f}%  "
          f"lift={md*100:+6.2f}pp (SE {se*100:.2f})  FPdelta={100*(ifp-bfp):+.1f}pp  gap(AUC-POS)={A-bpos:.3f}")

# ---- 3. meta-analysis, both HKSJ variants -----------------------------------
if len(rows)>=3:
    y=[r['lift'] for r in rows]; se=[r['se'] for r in rows]; k=len(y)
    w=[1/s**2 for s in se]; fe=sum(a*b for a,b in zip(w,y))/sum(w); fese=(1/sum(w))**.5
    Q=sum(wi*(yi-fe)**2 for wi,yi in zip(w,y)); df=k-1
    C=sum(w)-sum(wi**2 for wi in w)/sum(w); tau2=max(0,(Q-df)/C)
    wr=[1/(s**2+tau2) for s in se]; reM=sum(a*b for a,b in zip(wr,y))/sum(wr); rese=(1/sum(wr))**.5
    q=sum(wi*(yi-reM)**2 for wi,yi in zip(wr,y))/((k-1)*sum(wr))
    hks_std=q**.5; hks_mod=max(hks_std,rese)   # standard vs modified (truncated) HKSJ
    try:
        from scipy import stats as st; t=st.t.ppf(.975,df)
    except Exception: t={2:4.303,3:3.182}.get(df,2.776)
    print('\nMETA-ANALYSIS (canonical run):')
    print(f'  FE  {fe*100:+.2f}pp [{(fe-1.96*fese)*100:+.2f},{(fe+1.96*fese)*100:+.2f}]   '
          f'DL {reM*100:+.2f}pp [{(reM-1.96*rese)*100:+.2f},{(reM+1.96*rese)*100:+.2f}]   Q={Q:.2f} I2={max(0,(Q-df)/Q)*100:.1f}%')
    print(f'  HKSJ standard  [{(reM-t*hks_std)*100:+.2f},{(reM+t*hks_std)*100:+.2f}]')
    print(f'  HKSJ modified  [{(reM-t*hks_mod)*100:+.2f},{(reM+t*hks_mod)*100:+.2f}]   <- state THIS variant in the paper')

json.dump({'canonical_files':{m:{k2:v['path'] for k2,v in kk.items()} for m,kk in by_model.items()},
           'table':rows}, open(f'{OUT}/canonical_results.json','w'), indent=2, default=str)
print(f'\nSaved -> {OUT}/canonical_results.json')
print('\nPASTE THIS WHOLE OUTPUT BACK TO MrC.')
