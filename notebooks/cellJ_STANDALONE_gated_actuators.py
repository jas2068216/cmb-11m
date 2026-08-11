# =============================================================================
# WAVE-2 CELL J — STANDALONE: PROBE-GATED steering baselines (OLMo + Mistral)
# (GPU, ~1.5-2h on A100. RUN AFTER CELL K, fresh kernel is fine.)
# The decisive experiment for review finding "your win might just be the gate":
# puts OUR probe gate in front of ITI and CAA actuators at matched gate
# exposure, so the actuator comparison is finally apples-to-apples.
# NO other cells needed. Results save to Drive as each model finishes.
# =============================================================================
import os, json, re
import numpy as np
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
CANON = '/content/drive/MyDrive/cmb_llm_intervention/v3_backup'
OUT   = '/content/drive/MyDrive/cmb_llm_intervention'
NEUTRAL_SYSTEM_PROMPT = "You are a careful analyst. Answer questions strictly from the provided document."
L_PROBE = 17          # the paper's probing locus
K_ITI   = 24          # ITI heads (frontier region from the July sweep)
ALPHA   = 1.0         # ITI alpha
CAA_SCALES = [1.0, 5.0]

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


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

MODELS = [
    ("allenai/OLMo-2-1124-7B-Instruct",    f"{CANON}/baseline_olmo_2_1124_7b_instruct.json", "olmo_2_1124_7b_instruct"),
    ("mistralai/Mistral-7B-Instruct-v0.3", f"{CANON}/baseline_mistral_7b_instruct_v03.json", "mistral_7b_instruct_v03"),
]

for MNAME, JPATH, SLUG in MODELS:
  print(f"\n{'='*70}\nMODEL: {MNAME}\n{'='*70}")
  recs = json.load(open(JPATH))['records']
  prompts = [r['prompt'] for r in recs]; conds = [r['condition'] for r in recs]
  print(f"{len(recs)} canonical prompts loaded")
  tokenizer = AutoTokenizer.from_pretrained(MNAME)
  model = AutoModelForCausalLM.from_pretrained(MNAME, device_map="auto", torch_dtype=torch.float16)
  model.eval()
  cfg = model.config; L = cfg.num_hidden_layers; H = cfg.num_attention_heads; D = cfg.hidden_size // H
  layers = model.model.layers
  def chat(msg):
      return tokenizer.apply_chat_template(
          [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
           {"role": "user", "content": msg}], tokenize=False, add_generation_prompt=True)

  # ---- 1. one activation pass: L17 hidden state (gate+CAA) AND per-head (ITI) --
  cap = {}
  def mk_hook(li):
      def hook(mod, inp):
          cap[li] = inp[0][0, -1, :].detach().float().cpu().view(H, D).numpy()
      return hook
  handles = [layers[i].self_attn.o_proj.register_forward_pre_hook(mk_hook(i)) for i in range(L)]
  print(f"Capturing activations for {len(prompts)} prompts...")
  A = np.zeros((len(prompts), L, H, D), dtype=np.float32)
  X17 = np.zeros((len(prompts), cfg.hidden_size), dtype=np.float32)
  y = np.array([1 if c == 'uncertain' else 0 for c in conds])
  for n, p in enumerate(prompts):
      inp = tokenizer(chat(p), return_tensors='pt').to(model.device)
      with torch.no_grad():
          o = model(**inp, output_hidden_states=True, return_dict=True)
      X17[n] = o.hidden_states[L_PROBE][0, -1, :].float().cpu().numpy()
      for li in range(L): A[n, li] = cap[li]
      del o
  for h in handles: h.remove()
  torch.cuda.empty_cache()

  # ---- 2. THE GATE: deployed-config L17 probe + median-midpoint threshold ------
  probe = LogisticRegression(C=1.0, max_iter=2000).fit(X17, y)
  score = probe.predict_proba(X17)[:, 1]
  med_k = np.median(score[y == 0]); med_u = np.median(score[y == 1])
  theta = (med_k + med_u) / 2.0
  gate = score >= theta
  unc = [n for n, c in enumerate(conds) if c == 'uncertain']
  kn  = [n for n, c in enumerate(conds) if c == 'known']
  g_u = int(gate[unc].sum()); g_k = int(gate[kn].sum())
  print(f"GATE (theta={theta:.4f}): fires {g_u}/120 uncertain, {g_k}/120 known")
  print("  (matched gate exposure: every gated condition below uses THIS gate)")

  # ---- 3. actuator machinery ---------------------------------------------------
  print("Ranking heads for ITI (few min)...")
  acc = np.zeros((L, H))
  for li in range(L):
      for hd in range(H):
          acc[li, hd] = cross_val_score(LogisticRegression(C=1.0, max_iter=500), A[:, li, hd, :], y, cv=5).mean()
  order = np.argsort(acc.reshape(-1))[::-1]
  mu_u = A[y == 1].mean(0); mu_k = A[y == 0].mean(0)
  head_dir = mu_u - mu_k
  head_dir = head_dir / (np.linalg.norm(head_dir, axis=-1, keepdims=True) + 1e-8)
  proj = np.einsum('nlhd,lhd->nlh', A, head_dir); sigma = proj.std(0)
  iti_sel = [(o // H, o % H) for o in order[:K_ITI]]
  caa_vec = torch.tensor(X17[y == 1].mean(0) - X17[y == 0].mean(0),
                         device=model.device, dtype=next(model.parameters()).dtype)

  def iti_hooks():
      hooks = []
      bylayer = {}
      for (li, hd) in iti_sel: bylayer.setdefault(li, []).append(hd)
      def mk(li, hds):
          add = np.zeros((H, D), dtype=np.float32)
          for hd in hds: add[hd] = ALPHA * sigma[li, hd] * head_dir[li, hd]
          addt = torch.tensor(add.reshape(-1), device=model.device, dtype=next(model.parameters()).dtype)
          def hook(mod, inp):
              x = inp[0]; x[:, -1, :] = x[:, -1, :] + addt; return (x,) + inp[1:]
          return hook
      for li, hds in bylayer.items():
          hooks.append(layers[li].self_attn.o_proj.register_forward_pre_hook(mk(li, hds)))
      return hooks

  def caa_hooks(scale):
      # add scale * caa_vec to layer-L_PROBE output at EVERY position (CAA/ContextFocus actuator)
      def hook(mod, inp, out):
          if isinstance(out, tuple):
              return (out[0] + scale * caa_vec,) + out[1:]
          return out + scale * caa_vec
      return [layers[L_PROBE].register_forward_hook(hook)]

  def gen(prompt, hooks):
      inp = tokenizer(chat(prompt), return_tensors='pt').to(model.device); nlen = inp['input_ids'].shape[1]
      with torch.no_grad():
          o = model.generate(**inp, max_new_tokens=120, do_sample=False, temperature=1.0,
                             pad_token_id=tokenizer.eos_token_id)
      for h in hooks: h.remove()
      return tokenizer.decode(o[0, nlen:], skip_special_tokens=True).strip()

  # ---- 4. baseline once; gated conditions only regenerate gated prompts --------
  print("Baseline generations (240)...")
  base = {n: gen(prompts[n], []) for n in range(len(prompts))}
  b_pos = float(np.mean([flagged(base[n]) for n in unc]))
  b_fp  = float(np.mean([flagged(base[n]) for n in kn]))
  print(f"  baseline POS={b_pos:.3f} FP={b_fp:.3f}   (canonical regex)")

  results = {}; all_gens = {'baseline': {str(n): base[n] for n in base}}
  conds_to_run = [('gated_ITI_a1', iti_hooks, None)] + \
                 [(f'gated_CAA_s{int(s)}', caa_hooks, s) for s in CAA_SCALES]
  for tag, mk_hooks, arg in conds_to_run:
      print(f"{tag}: generating {int(gate.sum())} gated prompts (ungated = baseline copy)...")
      gens = {}
      for n in range(len(prompts)):
          if gate[n]:
              hooks = mk_hooks() if arg is None else mk_hooks(arg)
              gens[n] = gen(prompts[n], hooks)
          else:
              gens[n] = base[n]
      pos = float(np.mean([flagged(gens[n]) for n in unc]))
      fp  = float(np.mean([flagged(gens[n]) for n in kn]))
      results[tag] = {'POS': pos, 'FP': fp,
                      'POS_lift_pp': 100*(pos-b_pos), 'FP_delta_pp': 100*(fp-b_fp)}
      all_gens[tag] = {str(n): gens[n] for n in gens}
      print(f"  {tag}: POS {pos:.3f} (lift {100*(pos-b_pos):+.1f}pp)  FP {fp:.3f} (delta {100*(fp-b_fp):+.1f}pp)")

  json.dump({'model': MNAME, 'scorer': 'canonical_correction_aware',
             'gate': {'theta': float(theta), 'fires_uncertain': g_u, 'fires_known': g_k},
             'baseline_POS': b_pos, 'baseline_FP': b_fp, 'K_ITI': K_ITI, 'ALPHA': ALPHA,
             'results': results},
            open(f'{OUT}/gated_actuators_{SLUG}.json', 'w'))
  json.dump(all_gens, open(f'{OUT}/gated_actuators_generations_{SLUG}.json', 'w'))
  print(f"Saved -> gated_actuators_{SLUG}.json + generations")
  del model; torch.cuda.empty_cache()
print("\nBOTH MODELS DONE — PASTE THE PRINTED SUMMARIES BACK TO MrC.")
print("READ: if gated-ITI/CAA now match R-Restoration at the same gate exposure,")
print("      the actuator does not matter and the gate is the contribution;")
print("      if R-Restoration still wins on the FP/POS trade-off, the actuator")
print("      comparison is settled fairly. Either answer goes in the paper.")
