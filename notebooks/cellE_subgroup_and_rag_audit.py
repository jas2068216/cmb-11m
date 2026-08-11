# =============================================================================
# EXTRA CELL E — canonical subgroup rates + RAG table audit (NO GPU, ~1 min)
# 1. Recomputes fab/unk baseline hedge rates for all four models from the
#    canonical v3_backup files with the canonical regex (fixes the one stale
#    subgroup sentence in §5.2 — Mistral's fab/unk pair).
# 2. Finds every RAG results file on Drive and prints its structure + category
#    counts, so the tab:rag-gap "rows don't sum to 1" bug can be located.
# =============================================================================
import os, json, glob, re
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
MYDRIVE='/content/drive/MyDrive'
CANON=f'{MYDRIVE}/cmb_llm_intervention/v3_backup'

# canonical correction-aware regex (same as Cell A) — compact import trick:
# if Cell A ran in this kernel, reuse; else paste-run Cell A first for `flagged`.
try:
    flagged
    print('using flagged() from Cell A (same kernel)')
except NameError:
    raise SystemExit('Run Cell A first in this kernel (it defines the canonical scorer), then re-run this cell.')

print('='*70); print('1. CANONICAL SUBGROUP BASELINE HEDGE RATES (fab / unk)'); print('='*70)
for p in sorted(glob.glob(f'{CANON}/baseline_*.json')):
    d=json.load(open(p)); recs=d['records']
    for sg in ('fab','unk'):
        u=[r for r in recs if r['condition']=='uncertain' and r.get('subgrp')==sg]
        rate=sum(flagged(r['response']) for r in u)/len(u) if u else float('nan')
        print(f"  {d.get('model','?'):<40} {sg}: {rate*100:5.1f}%  (n={len(u)})")

print('\n'+'='*70); print('2. RAG RESULTS FILES — structure + category audit'); print('='*70)
hits=sorted(glob.glob(f'{MYDRIVE}/**/rag*​.json', recursive=True)) or sorted(glob.glob(f'{MYDRIVE}/**/rag*.json', recursive=True))
if not hits: print('  no rag*.json found — also trying *threeway* / *faithful*')
hits += sorted(glob.glob(f'{MYDRIVE}/**/*threeway*.json', recursive=True))
hits += sorted(glob.glob(f'{MYDRIVE}/**/*faithful*.json', recursive=True))
seen=set()
for p in hits:
    if p in seen: continue
    seen.add(p)
    try: d=json.load(open(p))
    except Exception as e: print(f'  ! {p}: {e}'); continue
    print(f'\n  FILE {p.replace(MYDRIVE,"~")}')
    if isinstance(d,dict):
        print(f'    keys: {list(d.keys())[:12]}')
        recs=d.get('records') or d.get('results') or []
        if isinstance(recs,list) and recs and isinstance(recs[0],dict):
            print(f'    n={len(recs)}  record keys: {list(recs[0].keys())}')
            # category count on any plausible label field
            for f in ('category','outcome','label','classification','verdict'):
                if f in recs[0]:
                    from collections import Counter
                    for grp in sorted(set(str(r.get('model','')) for r in recs)) or ['']:
                        sub=[r for r in recs if str(r.get('model',''))==grp]
                        print(f'    {f} counts {grp}: {dict(Counter(str(r[f]) for r in sub))}')
                    break
    elif isinstance(d,list) and d and isinstance(d[0],dict):
        print(f'    list n={len(d)}  keys: {list(d[0].keys())}')
print('\nPASTE THIS WHOLE OUTPUT BACK TO MrC.')
