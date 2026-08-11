# =============================================================================
# CMB-LLM Step 5d — R-Restoration v2.3 (bias=10, no decay).
# v2:   bias=3,  decay=30  → +0.8 pp (null)
# v2.1: bias=15, decay=∞   → +20.0 pp, 17 ambiguous (too strong)
# v2.2: bias=8,  decay=20  → null (decay was wrong knob)
# v2.3: bias=10, decay=∞   → target: ~15 pp gain, ≤6 ambiguous (sweet spot)
# Paste after v2.2. ~12-15 min on A100.
# =============================================================================

import sys, os, json, time
import numpy as np
from collections import Counter
from pathlib import Path

if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass

SEED          = 23
RESULTS_DIR   = f'/content/drive/MyDrive/cmb_llm_pipeline/seed{SEED}'
OUT_DIR       = '/content/drive/MyDrive/cmb_llm_intervention'
os.makedirs(OUT_DIR, exist_ok=True)

# Sustained pressure, moderate strength
THRESHOLD     = 0.0
BIAS_STRENGTH = 10.0     # was 15 (v2.1) / 8 (v2.2)
DECAY_AFTER   = 9999     # effectively disabled
PROBE_LAYER   = 17

# Ensure model is loaded
if 'model' not in globals() or 'tokenizer' not in globals():
    from harness.inference import load_model
    print('Loading model...')
    model, tokenizer = load_model('Qwen/Qwen2.5-7B-Instruct', load_in_4bit=False)

# Load probe + cases
from drift_monitor import ContradictionProbeMonitor
monitor = ContradictionProbeMonitor.from_artifact(f'{RESULTS_DIR}/contradiction_monitor.npz')
probe_weights, probe_bias = monitor.probe.weights, monitor.probe.bias

from harness.paired_contrast import load_paired_dataset, NEUTRAL_SYSTEM_PROMPT
from harness.dataset import TestCase
triples = load_paired_dataset(f'{RESULTS_DIR}/paired_dataset.json')
cases = [TestCase(
    case_id=f"triple_{t.triple_id}", entity_name=t.entity_name, sector=t.sector, city=t.city,
    year_first=t.year_first, year_second=t.year_second, distance_kind=t.distance_kind,
    V_target=t.V_target, V_actual=0, distance_tokens=0,
    pos_first_token=0, pos_second_token=0, document=t.doc_a, question=t.question,
) for t in triples]
print(f'Loaded {len(cases)} cases')

import torch
from transformers import LogitsProcessor, LogitsProcessorList
from harness.r_restoration import build_flag_token_ids, score_input_with_probe
from harness.inference import _build_chat_messages, InferenceResult
import harness.inference as inf_mod
inf_mod.SYSTEM_PROMPT = NEUTRAL_SYSTEM_PROMPT

flag_token_ids = list(build_flag_token_ids(tokenizer))
print(f'{len(flag_token_ids)} flag tokens, bias={BIAS_STRENGTH}, decay_after={DECAY_AFTER}')

class RRProcessor(LogitsProcessor):
    def __init__(self, active, flag_ids, bias, decay_after):
        self.active = active; self.flag_ids = flag_ids
        self.bias = bias; self.decay_after = decay_after; self.step = 0
    def __call__(self, input_ids, scores):
        if not self.active: return scores
        decay = max(0.0, 1.0 - self.step / max(1, self.decay_after))
        cur = self.bias * decay
        if cur > 0: scores[..., self.flag_ids] += cur
        self.step += 1
        return scores

print(f'\n[v2.3] running on {len(cases)} cases...')
results, t0 = [], time.time()
out_path = f'{OUT_DIR}/intervention_v23_results.json'
for i, case in enumerate(cases):
    messages = _build_chat_messages(case)
    prompt_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    score = score_input_with_probe(model, tokenizer, prompt_text, probe_weights, probe_bias, layer=PROBE_LAYER)
    active = score > THRESHOLD
    proc = RRProcessor(active, flag_token_ids, BIAS_STRENGTH, DECAY_AFTER)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    n_in = inputs.input_ids.shape[1]
    tt = time.time()
    with torch.no_grad():
        out_ids = model.generate(
            **inputs, max_new_tokens=400, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([proc]),
        )
    new_tokens = out_ids[0][n_in:]
    resp = tokenizer.decode(new_tokens, skip_special_tokens=True)
    results.append({
        "case_id": case.case_id, "probe_score": float(score),
        "intervention_active": bool(active), "response": resp,
        "n_input": int(n_in), "n_output": int(len(new_tokens)), "latency_s": float(time.time()-tt),
    })
    if (i + 1) % 10 == 0 or (i + 1) == len(cases):
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)
        print(f'  [{i+1}/{len(cases)}] elapsed={time.time()-t0:.0f}s')

print(f'\nSaved → {out_path}')

# Judge + compare against baseline AND v2.1
from harness.judge import judge_case
case_by_id = {c.case_id: c for c in cases}
judg = []
for r in results:
    case = case_by_id[r["case_id"]]
    fr = InferenceResult(case_id=r["case_id"], model_name=monitor.probe.model_name,
                          response=r["response"], V_actual=r["n_input"],
                          input_tokens=r["n_input"], output_tokens=r["n_output"], latency_s=r["latency_s"])
    judg.append(judge_case(case, fr))
v23 = Counter(j.outcome for j in judg)

with open(f'{RESULTS_DIR}/judgments_doca.json') as f:
    baseline_j = json.load(f)
baseline = Counter(j["outcome"] for j in baseline_j)

# Load v2.1 for comparison
try:
    with open(f'{OUT_DIR}/intervention_v21_results.json') as f:
        v21_results = json.load(f)
    v21_judg = []
    for r in v21_results:
        case = case_by_id[r["case_id"]]
        fr = InferenceResult(case_id=r["case_id"], model_name=monitor.probe.model_name,
                              response=r["response"], V_actual=r["n_input"],
                              input_tokens=r["n_input"], output_tokens=r["n_output"], latency_s=r["latency_s"])
        v21_judg.append(judge_case(case, fr))
    v21 = Counter(j.outcome for j in v21_judg)
except Exception:
    v21 = None

def m(c):
    t = sum(c.values()); d = c.get("detected", 0)
    return {"det": d, "par": c.get("partial", 0), "mis": c.get("missed", 0),
            "amb": c.get("ambiguous", 0), "rate": d/max(1,t), "rho": (t-d)/max(1,t)}

b, c23 = m(baseline), m(v23)
print('\n' + '=' * 78)
print('R-RESTORATION v2.3 (bias=10, no decay) — COMPARISON')
print('=' * 78)
header = f'{"metric":<18}  {"baseline":>10}  {"v2.1 (b=15)":>12}  {"v2.3 (b=10)":>14}'
print(header); print('-' * len(header))
if v21:
    c21 = m(v21)
else:
    c21 = {"det": 34, "par": 0, "mis": 69, "amb": 17, "rate": 34/120, "rho": 86/120}
for k_disp, k in [("detected","det"),("partial","par"),("missed","mis"),("ambiguous","amb")]:
    print(f'{k_disp:<18}  {b[k]:>10d}  {c21[k]:>12d}  {c23[k]:>14d}')
print(f'{"detection rate":<18}  {b["rate"]:>10.3f}  {c21["rate"]:>12.3f}  {c23["rate"]:>14.3f}')
print(f'{"rho_undetected":<18}  {b["rho"]:>10.3f}  {c21["rho"]:>12.3f}  {c23["rho"]:>14.3f}')
print('=' * 78)

# Output coherence check
resp_lens = [len(r["response"]) for r in results]
ambig_cases = [r for r, j in zip(results, judg) if j.outcome == "ambiguous"]
print(f'\nOutput coherence:')
print(f'  avg response length: {sum(resp_lens)/len(resp_lens):.0f} chars  (max {max(resp_lens)})')
print(f'  ambiguous (degraded) responses: {len(ambig_cases)}/{len(results)}')

# Spot-check flipped cases
base_by_id = {j["case_id"]: j["outcome"] for j in baseline_j}
flipped = [j for j in judg if base_by_id.get(j.case_id) in ('missed','partial') and j.outcome == 'detected']
print(f'\n{len(flipped)} cases flipped → detected. First 3:')
for j in flipped[:3]:
    r = next(r for r in results if r["case_id"] == j.case_id)
    print(f'\n--- {j.case_id} (baseline={base_by_id[j.case_id]}) ---')
    print(f'  probe_score: {r["probe_score"]:.2f}')
    print(f'  response: {r["response"][:300]}...')

# Spot-check any ambiguous responses (degradation check)
if ambig_cases:
    print(f'\n--- First ambiguous response (degradation sample) ---')
    r = ambig_cases[0]
    print(f'  case: {r["case_id"]}, probe_score: {r["probe_score"]:.2f}')
    print(f'  response: {r["response"][:300]}...')
