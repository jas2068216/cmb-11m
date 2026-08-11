# =============================================================================
# CMB-LLM Step 5b — R-Restoration v2.1 (aggressive tuning).
# v2 gave +0.8 pp detection (1 case flipped). Bias too weak, decay too fast.
# v2.1 cranks bias 3 → 15 and effectively disables decay.
# Paste below step 5 r-restoration cell. ~10-15 min on A100.
# =============================================================================

import sys, os, json, time
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import Counter

if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')

# Mount Drive
try:
    from google.colab import drive
    drive.mount('/content/drive')
except Exception:
    pass

SEED              = 23
RESULTS_DIR       = f'/content/drive/MyDrive/cmb_llm_pipeline/seed{SEED}'
OUT_DIR           = '/content/drive/MyDrive/cmb_llm_intervention'
os.makedirs(OUT_DIR, exist_ok=True)

# Aggressive tuning
THRESHOLD         = 0.0
BIAS_STRENGTH     = 15.0           # was 3.0
DECAY_AFTER       = 9999           # effectively disabled (was 30)
PROBE_LAYER       = 17

# -----------------------------------------------------------------------------
# Ensure model is loaded
# -----------------------------------------------------------------------------
if 'model' not in globals() or 'tokenizer' not in globals():
    from harness.inference import load_model
    print('Loading model...')
    model, tokenizer = load_model('Qwen/Qwen2.5-7B-Instruct', load_in_4bit=False)

# -----------------------------------------------------------------------------
# Load probe + cases
# -----------------------------------------------------------------------------
from drift_monitor import ContradictionProbeMonitor
monitor = ContradictionProbeMonitor.from_artifact(
    f'{RESULTS_DIR}/contradiction_monitor.npz'
)
probe_weights = monitor.probe.weights
probe_bias    = monitor.probe.bias
print(f'Probe loaded: layer={monitor.probe.layer}  '
      f'baseline mean projection (Doc A) = {monitor.baseline.mean:.3f}')

from harness.paired_contrast import load_paired_dataset, NEUTRAL_SYSTEM_PROMPT
from harness.dataset import TestCase
triples = load_paired_dataset(f'{RESULTS_DIR}/paired_dataset.json')
cases = []
for t in triples:
    cases.append(TestCase(
        case_id=f"triple_{t.triple_id}",
        entity_name=t.entity_name, sector=t.sector, city=t.city,
        year_first=t.year_first, year_second=t.year_second,
        distance_kind=t.distance_kind, V_target=t.V_target,
        V_actual=0, distance_tokens=0,
        pos_first_token=0, pos_second_token=0,
        document=t.doc_a, question=t.question,
    ))
print(f'Loaded {len(cases)} V4 cases')

# -----------------------------------------------------------------------------
# Inline aggressive logits processor (bypasses r_restoration.py decay)
# -----------------------------------------------------------------------------
import torch
from transformers import LogitsProcessor, LogitsProcessorList
from harness.r_restoration import build_flag_token_ids, score_input_with_probe
from harness.inference import _build_chat_messages
import harness.inference as inf_mod
inf_mod.SYSTEM_PROMPT = NEUTRAL_SYSTEM_PROMPT

flag_token_ids = list(build_flag_token_ids(tokenizer))
print(f'{len(flag_token_ids)} flag tokens, bias={BIAS_STRENGTH}, decay_after={DECAY_AFTER}')

class AggressiveRRProcessor(LogitsProcessor):
    def __init__(self, active: bool, flag_ids, bias_strength: float, decay_after: int):
        self.active = active
        self.flag_ids = flag_ids
        self.bias_strength = bias_strength
        self.decay_after = decay_after
        self.step = 0
    def __call__(self, input_ids, scores):
        if not self.active:
            return scores
        decay = max(0.0, 1.0 - self.step / max(1, self.decay_after))
        cur = self.bias_strength * decay
        if cur > 0:
            scores[..., self.flag_ids] += cur
        self.step += 1
        return scores

# -----------------------------------------------------------------------------
# Run intervention
# -----------------------------------------------------------------------------
print(f'\n[R-restoration v2.1] running on {len(cases)} cases...')
results = []
out_path = f'{OUT_DIR}/intervention_v21_results.json'
t0 = time.time()
for i, case in enumerate(cases):
    messages = _build_chat_messages(case)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    probe_score = score_input_with_probe(model, tokenizer, prompt_text,
                                          probe_weights, probe_bias,
                                          layer=PROBE_LAYER)
    active = probe_score > THRESHOLD
    proc = AggressiveRRProcessor(active, flag_token_ids, BIAS_STRENGTH, DECAY_AFTER)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    n_input = inputs.input_ids.shape[1]
    tt = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs, max_new_tokens=400, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=LogitsProcessorList([proc]),
        )
    latency = time.time() - tt
    new_tokens = output_ids[0][n_input:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    results.append({
        "case_id": case.case_id,
        "probe_score": float(probe_score),
        "intervention_active": bool(active),
        "response": response,
        "n_input": int(n_input),
        "n_output": int(len(new_tokens)),
        "latency_s": float(latency),
    })
    if (i + 1) % 10 == 0 or (i + 1) == len(cases):
        with open(out_path, 'w') as f: json.dump(results, f, indent=2)
        print(f'  [{i+1}/{len(cases)}] elapsed={time.time()-t0:.0f}s')

print(f'\nSaved results → {out_path}')

# -----------------------------------------------------------------------------
# Judge + compare
# -----------------------------------------------------------------------------
from harness.judge import judge_case
from harness.inference import InferenceResult
case_by_id = {c.case_id: c for c in cases}
judg = []
for r in results:
    case = case_by_id[r["case_id"]]
    fr = InferenceResult(
        case_id=r["case_id"], model_name=monitor.probe.model_name,
        response=r["response"], V_actual=r["n_input"],
        input_tokens=r["n_input"], output_tokens=r["n_output"],
        latency_s=r["latency_s"],
    )
    judg.append(judge_case(case, fr))
interv = Counter(j.outcome for j in judg)
print(f'\nv2.1 intervened outcomes: {dict(interv)}')

with open(f'{RESULTS_DIR}/judgments_doca.json') as f:
    baseline_j = json.load(f)
baseline = Counter(j["outcome"] for j in baseline_j)
print(f'Baseline outcomes:        {dict(baseline)}')

def m(c):
    t = sum(c.values())
    d = c.get("detected", 0)
    return {"detected": d, "partial": c.get("partial", 0), "missed": c.get("missed", 0),
            "rate": d / max(1, t), "rho": (t - d) / max(1, t)}

b = m(baseline); i = m(interv)
print('\n' + '=' * 70)
print('R-RESTORATION v2.1 (bias=15, no decay) — BEFORE vs AFTER')
print('=' * 70)
print(f'{"metric":<18}  {"baseline":>10}  {"intervened":>12}  {"delta":>10}')
print('-' * 70)
for k in ('detected', 'partial', 'missed'):
    print(f'{k:<18}  {b[k]:>10d}  {i[k]:>12d}  {i[k]-b[k]:>+10d}')
print(f'{"detection rate":<18}  {b["rate"]:>10.3f}  {i["rate"]:>12.3f}  {i["rate"]-b["rate"]:>+10.3f}')
print(f'{"rho_undetected":<18}  {b["rho"]:>10.3f}  {i["rho"]:>12.3f}  {i["rho"]-b["rho"]:>+10.3f}')
print('=' * 70)

# Spot-check flipped cases
base_by_id = {j["case_id"]: j["outcome"] for j in baseline_j}
flipped = [j for j in judg if base_by_id.get(j.case_id) in ('missed','partial') and j.outcome == 'detected']
print(f'\n{len(flipped)} cases flipped to detected. First 3:')
for j in flipped[:3]:
    r = next(r for r in results if r["case_id"] == j.case_id)
    print(f'\n--- {j.case_id} (baseline={base_by_id[j.case_id]}, intervened=detected) ---')
    print(f'  probe_score: {r["probe_score"]:.2f}')
    print(f'  response: {r["response"][:300]}...')
