"""R-Restoration intervention v2: probe-conditioned logit biasing.

The framework's central empirical claim:
    "The model encodes contradiction-relevant information internally
     (layer 17 last-input-token, paired-contrast probe AUC = 1.000),
     but fails to surface it in output 91% of the time."

R-Restoration applies that internal signal to steer generation:

    1. Run forward pass on the input.
    2. Project layer-17 last-input-token activation onto the probe direction.
    3. If projection exceeds threshold (model HAS the contradiction internally),
       bias the logits during generation toward contradiction-flagging tokens.
    4. Otherwise, generate normally.

If R-Restoration drops ρ_undetected measurably, we have direct evidence
that the framework's diagnostic ("model knows but doesn't surface")
implies a working intervention ("force the model to surface what it knows").
That's the "better AI" demonstration the original goal demanded.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Set

import numpy as np


# -----------------------------------------------------------------------------
# Token vocabulary for "contradiction-flagging language"
# -----------------------------------------------------------------------------
FLAG_WORDS = [
    # Direct contradiction language
    "contradict", "Contradict", "contradicts", "contradiction",
    "inconsistent", "Inconsistent", "inconsistency", "inconsistencies",
    "conflict", "Conflict", "conflicts", "conflicting",
    "discrepancy", "Discrepancy", "discrepancies",
    "mismatch", "Mismatch", "mismatched",
    # Contrast connectives
    "However", "however", "But", "Although", "although",
    "Whereas", "whereas",
    # Multiplicity / both-sides phrasing
    "both", "Both", "two", "Two", "different", "Different",
    "either", "Either",
]


def build_flag_token_ids(tokenizer, words: List[str] = FLAG_WORDS) -> Set[int]:
    """Tokenize each word in multiple variants (with/without leading space) and
    collect the FIRST token id of each variant. This gives us the bias targets:
    we boost tokens that, if selected, START a flagging phrase.
    """
    ids: Set[int] = set()
    for w in words:
        for variant in [w, " " + w]:
            toks = tokenizer.encode(variant, add_special_tokens=False)
            if toks:
                ids.add(toks[0])
    return ids


# -----------------------------------------------------------------------------
# Probe-conditioned logits processor
# -----------------------------------------------------------------------------
def make_logits_processor(flag_token_ids: Set[int],
                          probe_score: float,
                          threshold: float,
                          bias_strength: float = 3.0,
                          decay_after: int = 30):
    """Returns a LogitsProcessor that adds `bias_strength` to logits for the
    given token ids, but ONLY when probe_score > threshold.

    - decay_after: linearly decay bias to 0 over this many generated tokens.
      Prevents over-steering of long generations.
    """
    from transformers import LogitsProcessor

    class _RRProcessor(LogitsProcessor):
        def __init__(self):
            self.step = 0
            self.active = probe_score > threshold
            self.flag_ids = list(flag_token_ids)

        def __call__(self, input_ids, scores):
            if not self.active or not self.flag_ids:
                return scores
            # Linear decay
            decay = max(0.0, 1.0 - self.step / max(1, decay_after))
            current_bias = bias_strength * decay
            if current_bias > 0:
                scores[..., self.flag_ids] += current_bias
            self.step += 1
            return scores

    return _RRProcessor()


# -----------------------------------------------------------------------------
# Probe scoring (single forward pass)
# -----------------------------------------------------------------------------
def score_input_with_probe(model, tokenizer, prompt_text: str,
                            probe_weights: np.ndarray,
                            probe_bias: float,
                            layer: int = 17) -> float:
    """Run one forward pass; project layer-{layer} last-input-token hidden
    state onto the probe direction; return the scalar score."""
    import torch
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer][0, -1, :].to(torch.float32).cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return float(np.dot(probe_weights, h) + probe_bias)


# -----------------------------------------------------------------------------
# Run an intervention on a single case
# -----------------------------------------------------------------------------
def run_intervention(model, tokenizer,
                      prompt_text: str,
                      probe_weights: np.ndarray,
                      probe_bias: float,
                      flag_token_ids: Set[int],
                      threshold: float,
                      bias_strength: float = 3.0,
                      max_new_tokens: int = 400,
                      layer: int = 17) -> dict:
    """Run R-restoration on a single prompt. Returns dict with:
        probe_score, intervention_active, response, latency_s
    """
    import torch
    from transformers import LogitsProcessorList

    probe_score = score_input_with_probe(
        model, tokenizer, prompt_text, probe_weights, probe_bias, layer=layer,
    )
    intervention_active = probe_score > threshold

    processor = make_logits_processor(
        flag_token_ids=flag_token_ids,
        probe_score=probe_score,
        threshold=threshold,
        bias_strength=bias_strength,
    )
    processors = LogitsProcessorList([processor])

    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    n_input = inputs.input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            logits_processor=processors,
        )
    latency = time.time() - t0
    new_tokens = output_ids[0][n_input:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return {
        "probe_score":         probe_score,
        "intervention_active": bool(intervention_active),
        "response":            response,
        "n_input":             int(n_input),
        "n_output":            int(len(new_tokens)),
        "latency_s":           latency,
        "threshold":           threshold,
        "bias_strength":       bias_strength,
    }


# -----------------------------------------------------------------------------
# Batch evaluation
# -----------------------------------------------------------------------------
def evaluate_intervention(model, tokenizer,
                           cases,
                           probe_weights: np.ndarray,
                           probe_bias: float,
                           threshold: float,
                           bias_strength: float = 3.0,
                           layer: int = 17,
                           out_path = None,
                           verbose: bool = True) -> List[dict]:
    """Run R-restoration on a list of TestCase-like records. Returns list of
    intervention results plus the case_id."""
    from harness.inference import _build_chat_messages

    flag_token_ids = build_flag_token_ids(tokenizer)
    if verbose:
        print(f"[R-restoration] {len(flag_token_ids)} flag-token IDs constructed")

    results = []
    t_start = time.time()
    for i, case in enumerate(cases):
        # Build chat-formatted prompt the same way as baseline inference
        messages = _build_chat_messages(case)
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        info = run_intervention(
            model, tokenizer, prompt_text,
            probe_weights=probe_weights, probe_bias=probe_bias,
            flag_token_ids=flag_token_ids,
            threshold=threshold, bias_strength=bias_strength, layer=layer,
        )
        info["case_id"] = case.case_id
        results.append(info)

        if (i + 1) % 10 == 0 or (i + 1) == len(cases):
            elapsed = time.time() - t_start
            n_active = sum(1 for r in results if r["intervention_active"])
            if verbose:
                print(f"  [{i+1}/{len(cases)}] elapsed={elapsed:.0f}s  "
                      f"interventions_active={n_active}/{len(results)}")
            if out_path is not None:
                with open(out_path, "w") as f:
                    json.dump(results, f, indent=2)

    if out_path is not None:
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    return results
