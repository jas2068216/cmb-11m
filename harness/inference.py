"""Model inference for CMB-LLM Step 1.

Runs each TestCase through a chat model and saves the response. Designed for
Colab Pro: handles 4-bit quantization fallback, resumable to disk, no
multi-GPU assumptions.

Default model: Qwen/Qwen2.5-7B-Instruct (128k native context with rope scaling).
Fallback: meta-llama/Llama-3.2-3B-Instruct (128k native, smaller, T4-friendly).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional

from .dataset import TestCase


@dataclass
class InferenceResult:
    case_id: str
    model_name: str
    response: str
    V_actual: int
    input_tokens: int
    output_tokens: int
    latency_s: float

    def to_dict(self) -> dict:
        return asdict(self)


SYSTEM_PROMPT = (
    "You are a careful analyst. Answer questions strictly from the provided "
    "document. If the document contains contradictory information, you must "
    "explicitly identify the contradiction rather than picking one side "
    "silently."
)


def _build_chat_messages(case: TestCase) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": case.to_prompt()},
    ]


def load_model(model_name: str = "Qwen/Qwen2.5-7B-Instruct",
               load_in_4bit: bool = False,
               device_map: str = "auto"):
    """Load a HuggingFace causal LM and its tokenizer.

    Returns (model, tokenizer). On Colab T4, set load_in_4bit=True.
    On A100, fp16 works directly.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs = {"device_map": device_map}
    if load_in_4bit:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
    else:
        kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def run_case(model, tokenizer, case: TestCase,
             max_new_tokens: int = 400) -> InferenceResult:
    """Run a single test case. Returns the model's response."""
    import torch

    messages = _build_chat_messages(case)
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    n_input = inputs.input_ids.shape[1]

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,                # deterministic for reproducibility
            temperature=1.0,                # ignored when do_sample=False
            pad_token_id=tokenizer.eos_token_id,
        )
    latency = time.time() - t0

    # Strip prompt tokens from the output
    new_tokens = output_ids[0][n_input:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)

    return InferenceResult(
        case_id=case.case_id,
        model_name=model.config._name_or_path,
        response=response,
        V_actual=case.V_actual,
        input_tokens=n_input,
        output_tokens=len(new_tokens),
        latency_s=latency,
    )


def run_dataset(model, tokenizer, cases: List[TestCase],
                results_path: str | Path,
                max_new_tokens: int = 400,
                verbose: bool = True) -> List[InferenceResult]:
    """Run all cases, saving incrementally so a crash doesn't lose progress."""
    results_path = Path(results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing results for resumability
    completed = {}
    if results_path.exists():
        with results_path.open() as f:
            for record in json.load(f):
                completed[record["case_id"]] = InferenceResult(**record)
        if verbose:
            print(f"[resume] loaded {len(completed)} prior results")

    results: List[InferenceResult] = list(completed.values())
    for i, case in enumerate(cases):
        if case.case_id in completed:
            if verbose:
                print(f"[skip] {case.case_id} (already done)")
            continue
        if verbose:
            print(f"[{i+1}/{len(cases)}] running {case.case_id} "
                  f"(V_actual={case.V_actual})")
        try:
            result = run_case(model, tokenizer, case, max_new_tokens=max_new_tokens)
            results.append(result)
            # Incremental save
            with results_path.open("w") as f:
                json.dump([r.to_dict() for r in results], f, indent=2)
            if verbose:
                print(f"    out_tokens={result.output_tokens} "
                      f"latency={result.latency_s:.1f}s")
        except Exception as e:
            print(f"    ERROR on {case.case_id}: {e}")
            # Don't abort the whole run; just skip this case
            continue

    return results


def load_results(path: str | Path) -> List[InferenceResult]:
    with Path(path).open() as f:
        records = json.load(f)
    return [InferenceResult(**r) for r in records]
