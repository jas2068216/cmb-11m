"""Paired-contrast dataset construction + multi-position activation capture.

For every (entity, V, distance) configuration we produce a TRIPLE of documents:

  Doc A : A1 = year_first  (X),  A2 = year_second (Y)   ← contradiction
  Doc B : A1 = year_first  (X),  A2 = year_first  (X)   ← no contradiction (A1-matched)
  Doc B': A1 = year_second (Y),  A2 = year_second (Y)   ← no contradiction (A2-token-matched)

All three share identical filler text (same RNG state when generating). The
ONLY differences are which year-values appear in the two assertion slots.

Step 3  experiment compares A vs B  (controls V/distance/entity but not the
        literal token at A2 — partly trivial signal).
Step 3b experiment compares A vs B' (also controls the literal token at A2 —
        any AUC > 0.5 must come from year_first propagation via attention).

Both versions plus the multi-position activation capture live here so the
full pipeline can be re-run from one orchestrator.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import numpy as np

from .dataset import (
    ENTITY_TEMPLATES, CONTRADICTION_YEAR_PAIRS,
    FILLER_TEMPLATES, token_len, _make_filler,
)


# --------------------------------------------------------------------------- #
# Templates (mirror the V2/V3/V4 overrides)
# --------------------------------------------------------------------------- #
NEUTRAL_SYSTEM_PROMPT = (
    "You are a careful analyst. Answer questions strictly from the provided "
    "document."
)
NEUTRAL_QUESTION_TEMPLATE = "Based on the document above, when was {name} founded?"
ASSERTION_TEMPLATE_A = (
    "{name} was founded in {year} by a small team that relocated to {city}."
)
ASSERTION_TEMPLATE_B = (
    "Industry filings list {year} as the year {name} was established, "
    "and the company's official records continue to cite that date."
)


@dataclass
class PairedCase:
    """A triple of documents that share filler but differ in assertion content."""
    triple_id: int
    entity_name: str
    sector: str
    city: str
    year_first: str
    year_second: str
    distance_kind: str
    V_target: int

    doc_a: str
    doc_b: str
    doc_bp: str

    assertion_a1: str
    assertion_a2_a: str         # Doc A's A2 (uses year_second)
    assertion_a2_b: str         # Doc B's A2 (uses year_first)
    assertion_a1_bp: str        # Doc B''s A1 (uses year_second)
    assertion_a2_bp: str        # Doc B''s A2 (uses year_second)

    question: str

    def to_dict(self):
        return asdict(self)


# --------------------------------------------------------------------------- #
# Document building (shared filler)
# --------------------------------------------------------------------------- #
def _build_one_document(entity, year_first_val, year_second_val,
                        distance_kind, V_target, tokenizer, rng):
    """Build a document with the given year values at A1 and A2.
    Returns (document_text, assertion1_text, assertion2_text)."""
    AVG_TOKENS_PER_SENTENCE = 18
    if distance_kind == "short":
        frac_first, frac_second = 0.45, 0.55
    elif distance_kind == "long":
        frac_first, frac_second = 0.05, 0.85
    else:
        raise ValueError(f"unknown distance_kind={distance_kind!r}")

    tokens_before  = int(V_target * frac_first)
    tokens_between = int(V_target * (frac_second - frac_first))
    tokens_after   = max(0, V_target - tokens_before - tokens_between)

    assertion1 = ASSERTION_TEMPLATE_A.format(name=entity["name"],
                                             year=year_first_val,
                                             city=entity["city"])
    assertion2 = ASSERTION_TEMPLATE_B.format(name=entity["name"],
                                             year=year_second_val)
    tokens_between = max(20, tokens_between
                         - token_len(tokenizer, assertion1)
                         - token_len(tokenizer, assertion2))

    n_before  = max(1, tokens_before  // AVG_TOKENS_PER_SENTENCE)
    n_between = max(1, tokens_between // AVG_TOKENS_PER_SENTENCE)
    n_after   = max(1, tokens_after   // AVG_TOKENS_PER_SENTENCE)

    fb = _make_filler(entity, n_before,  rng)
    fm = _make_filler(entity, n_between, rng)
    fa = _make_filler(entity, n_after,   rng)
    intro = (f"Internal briefing on {entity['name']}, a {entity['sector']} "
             f"company based in {entity['city']}.\n\n")
    document = f"{intro}{fb} {assertion1} {fm} {assertion2} {fa}"
    return document, assertion1, assertion2


def build_paired_dataset(tokenizer,
                         V_targets: List[int],
                         distance_kinds: List[str],
                         entities_per_cell: int,
                         seed: int = 23) -> List[PairedCase]:
    """Build paired triples (Doc A, Doc B, Doc B') with identical filler per
    triple. Per-case random draws of entity + year_pair (seeded)."""
    rng = random.Random(seed)
    triples: List[PairedCase] = []
    triple_idx = 0

    for V_target in V_targets:
        for distance_kind in distance_kinds:
            for i in range(entities_per_cell):
                entity = rng.choice(ENTITY_TEMPLATES)
                yf, ys = rng.choice(CONTRADICTION_YEAR_PAIRS)

                # Snapshot RNG state so all three docs share the SAME filler
                rng_state = rng.getstate()

                doc_a,  a1_a,  a2_a  = _build_one_document(
                    entity, yf, ys, distance_kind, V_target, tokenizer, rng)
                rng.setstate(rng_state)
                doc_b,  a1_b,  a2_b  = _build_one_document(
                    entity, yf, yf, distance_kind, V_target, tokenizer, rng)
                rng.setstate(rng_state)
                doc_bp, a1_bp, a2_bp = _build_one_document(
                    entity, ys, ys, distance_kind, V_target, tokenizer, rng)

                question = NEUTRAL_QUESTION_TEMPLATE.format(name=entity["name"])
                triples.append(PairedCase(
                    triple_id=triple_idx,
                    entity_name=entity["name"],
                    sector=entity["sector"], city=entity["city"],
                    year_first=yf, year_second=ys,
                    distance_kind=distance_kind, V_target=V_target,
                    doc_a=doc_a, doc_b=doc_b, doc_bp=doc_bp,
                    assertion_a1=a1_a,
                    assertion_a2_a=a2_a, assertion_a2_b=a2_b,
                    assertion_a1_bp=a1_bp, assertion_a2_bp=a2_bp,
                    question=question,
                ))
                triple_idx += 1
    return triples


def save_paired_dataset(triples: List[PairedCase], path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([t.to_dict() for t in triples], f, indent=2)


def load_paired_dataset(path) -> List[PairedCase]:
    with Path(path).open() as f:
        records = json.load(f)
    return [PairedCase(**r) for r in records]


# --------------------------------------------------------------------------- #
# Position finding (char→token via offset_mapping)
# --------------------------------------------------------------------------- #
def find_assertion_positions(document: str, assertion1: str, assertion2: str,
                              question: str, tokenizer) -> Optional[Tuple[int, int, str]]:
    """Locate the last-token indices of A1 and A2 in the chat-formatted prompt.
    Returns (pos_a1, pos_a2, prompt_text)."""
    # Build a minimal chat prompt (system + user content)
    prompt_text = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
            {"role": "user",
             "content": f"{document}\n\n---\n\nQuestion: {question}"},
        ],
        tokenize=False, add_generation_prompt=True,
    )
    doc_offset = prompt_text.find(document)
    if doc_offset < 0:
        return None
    char_a1_end = doc_offset + document.find(assertion1) + len(assertion1)
    char_a2_end = doc_offset + document.find(assertion2) + len(assertion2)
    if document.find(assertion1) < 0 or document.find(assertion2) < 0:
        return None

    enc = tokenizer(prompt_text, return_offsets_mapping=True, add_special_tokens=False)
    offsets = enc["offset_mapping"]

    def char_to_tok(target_char_end: int) -> int:
        pos = 0
        for i, (s, e) in enumerate(offsets):
            if e <= target_char_end:
                pos = i
        return pos

    return char_to_tok(char_a1_end), char_to_tok(char_a2_end), prompt_text


# --------------------------------------------------------------------------- #
# Activation capture (one forward pass per document, 3 positions × all layers)
# --------------------------------------------------------------------------- #
def capture_document_activations(model, tokenizer,
                                  document: str, assertion1: str, assertion2: str,
                                  question: str) -> Optional[dict]:
    """Returns a dict with hs_post_a1, hs_post_a2, hs_last — each a numpy
    array of shape [n_layers+1, hidden_dim]. Returns None on failure."""
    import torch
    res = find_assertion_positions(document, assertion1, assertion2, question, tokenizer)
    if res is None:
        return None
    pos_a1, pos_a2, prompt_text = res
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    hs_a1   = torch.stack([h[0, pos_a1, :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    hs_a2   = torch.stack([h[0, pos_a2, :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    hs_last = torch.stack([h[0, -1,     :].to(torch.float32).cpu() for h in out.hidden_states]).numpy()
    del out
    torch.cuda.empty_cache()
    return {
        "hs_post_a1": hs_a1,
        "hs_post_a2": hs_a2,
        "hs_last":    hs_last,
        "pos_a1":     pos_a1,
        "pos_a2":     pos_a2,
        "n_tokens":   inputs.input_ids.shape[1],
    }


def capture_triples_activations(model, tokenizer,
                                triples: List[PairedCase],
                                out_path,
                                verbose: bool = True) -> dict:
    """For each triple, capture activations for Doc A, Doc B, Doc B'.
    Saves a single .npz with three arrays (per doc_kind) and meta.
    Resumable: skips triples whose case_id is already in the cache.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    records: List[dict] = []   # one per (triple, doc_kind) pair
    if out_path.exists():
        try:
            loaded = np.load(out_path, allow_pickle=True)
            meta_cached = json.loads(str(loaded["meta"]))
            for i, m in enumerate(meta_cached):
                records.append({
                    "triple_id":     m["triple_id"],
                    "doc_kind":      m["doc_kind"],
                    "V_target":      m["V_target"],
                    "distance_kind": m["distance_kind"],
                    "entity":        m["entity"],
                    "year_first":    m["year_first"],
                    "year_second":   m["year_second"],
                    "hs_post_a1":    loaded["hs_post_a1"][i],
                    "hs_post_a2":    loaded["hs_post_a2"][i],
                    "hs_last":       loaded["hs_last"][i],
                })
            if verbose:
                print(f"[resume] loaded {len(records)} cached records")
        except Exception as e:
            print(f"[resume] cache load failed ({e}), starting fresh")
            records = []

    done_keys = {(r["triple_id"], r["doc_kind"]) for r in records}
    t_start = time.time()

    for i, triple in enumerate(triples):
        for which in ('a', 'b', 'bp'):
            key = (triple.triple_id, which)
            if key in done_keys:
                continue
            if which == 'a':
                doc, a1, a2 = triple.doc_a,  triple.assertion_a1,    triple.assertion_a2_a
            elif which == 'b':
                doc, a1, a2 = triple.doc_b,  triple.assertion_a1,    triple.assertion_a2_b
            else:
                doc, a1, a2 = triple.doc_bp, triple.assertion_a1_bp, triple.assertion_a2_bp
            try:
                hs = capture_document_activations(model, tokenizer, doc, a1, a2, triple.question)
                if hs is None:
                    print(f"[skip] triple {triple.triple_id} doc {which}: position lookup failed")
                    continue
                records.append({
                    "triple_id":     triple.triple_id,
                    "doc_kind":      which,
                    "V_target":      triple.V_target,
                    "distance_kind": triple.distance_kind,
                    "entity":        triple.entity_name,
                    "year_first":    triple.year_first,
                    "year_second":   triple.year_second,
                    "hs_post_a1":    hs["hs_post_a1"],
                    "hs_post_a2":    hs["hs_post_a2"],
                    "hs_last":       hs["hs_last"],
                })
            except Exception as e:
                print(f"[err] triple {triple.triple_id} doc {which}: {e}")
                continue
        # Save periodically
        if (i + 1) % 10 == 0 or (i + 1) == len(triples):
            _save_records_npz(records, out_path)
            if verbose:
                print(f"  [{i+1}/{len(triples)} triples] {len(records)} doc records  "
                      f"elapsed={time.time()-t_start:.0f}s")

    _save_records_npz(records, out_path)
    return {"n_records": len(records), "out_path": str(out_path)}


def _save_records_npz(records: List[dict], path):
    if not records:
        return
    a1   = np.stack([r["hs_post_a1"] for r in records]).astype(np.float16)
    a2   = np.stack([r["hs_post_a2"] for r in records]).astype(np.float16)
    last = np.stack([r["hs_last"]    for r in records]).astype(np.float16)
    meta = [{k: r[k] for k in ("triple_id","doc_kind","V_target","distance_kind",
                               "entity","year_first","year_second")}
            for r in records]
    np.savez_compressed(path,
                        hs_post_a1=a1, hs_post_a2=a2, hs_last=last,
                        meta=json.dumps(meta))


def load_triples_activations(path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[dict]]:
    loaded = np.load(path, allow_pickle=True)
    meta = json.loads(str(loaded["meta"]))
    return (loaded["hs_post_a1"], loaded["hs_post_a2"], loaded["hs_last"], meta)
