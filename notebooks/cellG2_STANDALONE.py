# =============================================================================
# EXTRA CELL G2 — STANDALONE: ITI baseline with the CANONICAL scorer
# (GPU, ~40-60 min on A100 — pick Runtime > Change runtime type > A100 first!)
# NO other cells needed. Paste this whole cell into any fresh GPU kernel, run.
# Prompts come from the canonical v3_backup baseline JSON; scorer is the paper's
# canonical correction-aware regex, inlined below.
# =============================================================================
import os, json, re
import numpy as np
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
CANON = '/content/drive/MyDrive/cmb_llm_intervention/v3_backup'
OUT   = '/content/drive/MyDrive/cmb_llm_intervention'
NEUTRAL_SYSTEM_PROMPT = "You are a careful analyst. Answer questions strictly from the provided document."
K_SWEEP = [24]; ALPHA_SWEEP = [1.0, 2.0]   # the frontier region from the July run

# ---- canonical correction-aware regex (verbatim from the paper pipeline) ----
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


# ---- load canonical prompts -------------------------------------------------
recs = json.load(open(f"{CANON}/baseline_qwen25_7b_instruct.json"))['records']
prompts = [r['prompt'] for r in recs]
conds   = [r['condition'] for r in recs]
print(f"{len(recs)} canonical prompts loaded")

# ---- model ------------------------------------------------------------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
MNAME = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MNAME)
model = AutoModelForCausalLM.from_pretrained(MNAME, device_map="auto", torch_dtype=torch.float16)
model.eval()
cfg = model.config; L = cfg.num_hidden_layers; H = cfg.num_attention_heads; D = cfg.hidden_size // H
layers = model.model.layers
def chat(msg):
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
         {"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)

# ---- 1. per-head last-token activations -------------------------------------
cap = {}
def mk_hook(li):
    def hook(mod, inp):
        cap[li] = inp[0][0, -1, :].detach().float().cpu().view(H, D).numpy()
    return hook
handles = [layers[i].self_attn.o_proj.register_forward_pre_hook(mk_hook(i)) for i in range(L)]
print(f"Capturing per-head activations for {len(prompts)} prompts...")
A = np.zeros((len(prompts), L, H, D), dtype=np.float32)
y = np.array([1 if c == 'uncertain' else 0 for c in conds])
for n, p in enumerate(prompts):
    inp = tokenizer(chat(p), return_tensors='pt').to(model.device)
    with torch.no_grad(): model(**inp)
    for li in range(L): A[n, li] = cap[li]
for h in handles: h.remove()

# ---- 2. head ranking + directions -------------------------------------------
print("Ranking heads (few min)...")
acc = np.zeros((L, H))
for li in range(L):
    for hd in range(H):
        acc[li, hd] = cross_val_score(LogisticRegression(C=1.0, max_iter=500), A[:, li, hd, :], y, cv=5).mean()
flat = acc.reshape(-1); order = np.argsort(flat)[::-1]
mu_u = A[y == 1].mean(0); mu_k = A[y == 0].mean(0)
direction = mu_u - mu_k
direction = direction / (np.linalg.norm(direction, axis=-1, keepdims=True) + 1e-8)
proj = np.einsum('nlhd,lhd->nlh', A, direction); sigma = proj.std(0)
print(f"per-head acc median {np.median(flat):.3f} max {flat.max():.3f}")

# ---- 3. generation with/without ITI -----------------------------------------
def gen(prompt, sel, alpha):
    hooks = []
    if sel is not None:
        bylayer = {}
        for (li, hd) in sel: bylayer.setdefault(li, []).append(hd)
        def mk(li, hds):
            add = np.zeros((H, D), dtype=np.float32)
            for hd in hds: add[hd] = alpha * sigma[li, hd] * direction[li, hd]
            addt = torch.tensor(add.reshape(-1), device=model.device, dtype=next(model.parameters()).dtype)
            def hook(mod, inp):
                x = inp[0]; x[:, -1, :] = x[:, -1, :] + addt; return (x,) + inp[1:]
            return hook
        for li, hds in bylayer.items():
            hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(mk(li, hds)))
    inp = tokenizer(chat(prompt), return_tensors='pt').to(model.device); nlen = inp['input_ids'].shape[1]
    with torch.no_grad():
        o = model.generate(**inp, max_new_tokens=120, do_sample=False, temperature=1.0,
                           pad_token_id=tokenizer.eos_token_id)
    for h in hooks: h.remove()
    return tokenizer.decode(o[0, nlen:], skip_special_tokens=True).strip()

unc = [n for n, c in enumerate(conds) if c == 'uncertain']
kn  = [n for n, c in enumerate(conds) if c == 'known']
def run(sel, alpha, tag):
    gens = {}
    for n in unc + kn: gens[n] = gen(prompts[n], sel, alpha)
    pos = float(np.mean([flagged(gens[n]) for n in unc]))
    fp  = float(np.mean([flagged(gens[n]) for n in kn]))
    return pos, fp, {str(n): gens[n] for n in gens}

print("Baseline generations (no ITI)...")
b_pos, b_fp, b_gens = run(None, 0, 'base')
print(f"  baseline POS={b_pos:.3f} FP={b_fp:.3f}   (canonical regex)")
results = []; all_gens = {'baseline': b_gens}
for K in K_SWEEP:
    sel = [(o // H, o % H) for o in order[:K]]
    for alpha in ALPHA_SWEEP:
        print(f"ITI K={K} alpha={alpha} generations...")
        pu, fk, gg = run(sel, alpha, f'K{K}a{alpha}')
        all_gens[f'K{K}_a{alpha}'] = gg
        results.append({'K': K, 'alpha': alpha, 'POS': pu, 'FP': fk,
                        'POS_lift_pp': (pu - b_pos) * 100, 'FP_delta_pp': (fk - b_fp) * 100})
        print(f"  ITI K={K} a={alpha}: POS {pu:.3f} (lift {100*(pu-b_pos):+.1f}pp) "
              f"FP {fk:.3f} (delta {100*(fk-b_fp):+.1f}pp)")

json.dump({'scorer': 'canonical_correction_aware', 'baseline_POS': b_pos, 'baseline_FP': b_fp,
           'results': results}, open(f'{OUT}/iti_canonical_results.json', 'w'), indent=2)
json.dump(all_gens, open(f'{OUT}/iti_canonical_generations.json', 'w'))
print(f"\nSaved -> iti_canonical_results.json + iti_canonical_generations.json")
print("PASTE THE PRINTED SUMMARY BACK TO MrC.")
