"""Contradiction-injection dataset for measuring long-context self-contradiction.

Builds synthetic test cases over fictional entities so there's no parametric
knowledge leak. Each case has:
  - a long document (varies in token length V)
  - two contradicting assertions about the same fact, at known token positions
  - a question that requires resolving the contradiction
  - ground-truth metadata: V, contradiction distance, asserted values

Design choices
--------------
- Fact type: founding year of a fictional company. Integer answer, unambiguous,
  easy to detect in model output via regex.
- Filler text: procedurally generated news-style paragraphs about milestones,
  partnerships, market context. Filler must NOT mention any year (we strip
  numerics from filler to avoid accidental contradictions).
- Two contradiction positions: 'short' (assertions ~10% of V apart) and
  'long' (~90% of V apart). Tests whether distance, not just absolute V,
  drives the failure.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, List, Protocol


# --------------------------------------------------------------------------- #
# Tokenizer protocol — accepts anything with .encode() returning a list of ints
# --------------------------------------------------------------------------- #
class TokenizerLike(Protocol):
    def encode(self, text: str) -> List[int]: ...


def token_len(tokenizer: TokenizerLike, text: str) -> int:
    return len(tokenizer.encode(text))


# --------------------------------------------------------------------------- #
# Entity templates — fictional companies, sectors, locations
# --------------------------------------------------------------------------- #
ENTITY_TEMPLATES = [
    {"name": "Voltara Energy", "sector": "grid-scale battery storage", "city": "Reno"},
    {"name": "Helix Biotic", "sector": "synthetic biology", "city": "Cambridge"},
    {"name": "Mirador Robotics", "sector": "warehouse automation", "city": "Austin"},
    {"name": "Sable Quantum", "sector": "cryogenic computing", "city": "Boulder"},
    {"name": "Northwind Optics", "sector": "satellite imaging", "city": "Tacoma"},
    {"name": "Cardia Bioloop", "sector": "implantable cardiac sensors", "city": "Rochester"},
    {"name": "Tessera Materials", "sector": "ceramic composites", "city": "Akron"},
    {"name": "Lumen Flux", "sector": "industrial photonics", "city": "Tucson"},
    {"name": "Halberd Defense", "sector": "counter-UAS systems", "city": "Huntsville"},
    {"name": "Rhizome Agro", "sector": "vertical farming", "city": "Des Moines"},
]

# Year pairs that contradict each other. Both plausible founding years for a
# modern tech company. Picked to be 3+ years apart so the contradiction is
# unambiguous on judgment.
CONTRADICTION_YEAR_PAIRS = [
    ("2014", "2019"),
    ("2016", "2021"),
    ("2013", "2018"),
    ("2017", "2022"),
    ("2015", "2020"),
]

# Filler sentence templates. None reference years — that's enforced by review.
FILLER_TEMPLATES = [
    "The {city} office expanded its engineering team to support the {sector} roadmap.",
    "Customers in the {sector} space have praised {name} for its operational rigor.",
    "Industry analysts covering {sector} note that {name} has carved out a defensible niche.",
    "A recent partnership extended {name}'s reach into adjacent {sector} verticals.",
    "Recruiting in {city} has accelerated as {name} scales its {sector} platform.",
    "Supply-chain investments by {name} are expected to harden delivery across {sector} customers.",
    "Internal documentation describes {name}'s manufacturing approach as deliberately conservative.",
    "Several executives have remarked that {name} prioritizes durability over novelty.",
    "Press coverage of {name} has emphasized its restrained public posture.",
    "Engineering teams at {name} are organized around platform reliability metrics.",
    "Sales cycles in {sector} tend to be long, and {name} has built a process to match.",
    "Field engineers deployed by {name} report consistently high uptime.",
    "Procurement teams evaluating {sector} vendors frequently shortlist {name}.",
    "The leadership team at {name} has emphasized retention over rapid hiring.",
    "Operational reviews at {name} center on per-unit margin discipline.",
    "Customer success teams at {name} run quarterly health checks across the {sector} install base.",
    "Trade publications covering {sector} have profiled {name}'s approach to validation.",
    "Hiring plans for the {city} headquarters focus on senior {sector} specialists.",
    "Vendor consolidation across {sector} has favored mid-sized firms like {name}.",
    "Quality assurance practices at {name} are audited by an external third party.",
]


# --------------------------------------------------------------------------- #
# Test case dataclass
# --------------------------------------------------------------------------- #
@dataclass
class TestCase:
    case_id: str
    entity_name: str
    sector: str
    city: str
    year_first: str           # asserted earlier in the document
    year_second: str          # asserted later (contradicts year_first)
    distance_kind: str        # "short" or "long"
    V_target: int             # target context length in tokens
    V_actual: int             # measured context length
    distance_tokens: int      # tokens between the two assertions
    pos_first_token: int      # token offset of first assertion
    pos_second_token: int     # token offset of second assertion
    document: str             # full document with embedded contradictions
    question: str             # the prompt question

    def to_prompt(self) -> str:
        return f"{self.document}\n\n---\n\nQuestion: {self.question}"

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Document construction
# --------------------------------------------------------------------------- #
def _make_filler(entity: dict, n_sentences: int, rng: random.Random) -> str:
    """Generate filler text with no year references."""
    sentences = []
    for _ in range(n_sentences):
        template = rng.choice(FILLER_TEMPLATES)
        sentences.append(template.format(**entity))
    return " ".join(sentences)


def _assertion_sentence(entity: dict, year: str) -> str:
    return f"{entity['name']} was founded in {year} and is headquartered in {entity['city']}."


def _build_document(
    entity: dict,
    year_first: str,
    year_second: str,
    distance_kind: str,
    V_target: int,
    tokenizer: TokenizerLike,
    rng: random.Random,
) -> tuple[str, dict]:
    """Build a document of approximately V_target tokens with two contradicting
    assertions at the requested distance.

    Returns (document_text, metadata) where metadata has actual token positions.
    """
    # Approx sentences-per-100-tokens heuristic for sizing filler
    AVG_TOKENS_PER_SENTENCE = 18  # filler sentences are short
    target_total = V_target

    # Decide positions in token-fraction terms
    if distance_kind == "short":
        # Both assertions near the middle, ~10% of V apart
        frac_first, frac_second = 0.45, 0.55
    elif distance_kind == "long":
        # First near start, second near end, ~80% apart
        frac_first, frac_second = 0.05, 0.85
    else:
        raise ValueError(f"unknown distance_kind={distance_kind!r}")

    # Build in three filler segments: before, between, after
    tokens_before = int(target_total * frac_first)
    tokens_between = int(target_total * (frac_second - frac_first))
    tokens_after = max(0, target_total - tokens_before - tokens_between)

    assertion1 = _assertion_sentence(entity, year_first)
    assertion2 = _assertion_sentence(entity, year_second)
    a1_tokens = token_len(tokenizer, assertion1)
    a2_tokens = token_len(tokenizer, assertion2)

    # Account for assertion tokens in filler budgets
    tokens_between = max(20, tokens_between - a1_tokens - a2_tokens)

    n_before = max(1, tokens_before // AVG_TOKENS_PER_SENTENCE)
    n_between = max(1, tokens_between // AVG_TOKENS_PER_SENTENCE)
    n_after = max(1, tokens_after // AVG_TOKENS_PER_SENTENCE)

    filler_before = _make_filler(entity, n_before, rng)
    filler_between = _make_filler(entity, n_between, rng)
    filler_after = _make_filler(entity, n_after, rng)

    # Stitch into a document with an intro line for plausibility
    intro = (
        f"Internal briefing on {entity['name']}, a {entity['sector']} company "
        f"based in {entity['city']}.\n\n"
    )
    document = (
        f"{intro}{filler_before} {assertion1} {filler_between} "
        f"{assertion2} {filler_after}"
    )

    # Measure actual token positions of the two assertions in the final document
    # Strategy: tokenize the prefix up to (and not including) each assertion
    pre_assertion1 = f"{intro}{filler_before} "
    pre_assertion2 = f"{intro}{filler_before} {assertion1} {filler_between} "

    pos_first = token_len(tokenizer, pre_assertion1)
    pos_second = token_len(tokenizer, pre_assertion2)
    V_actual = token_len(tokenizer, document)
    distance_tokens = pos_second - pos_first

    meta = {
        "pos_first_token": pos_first,
        "pos_second_token": pos_second,
        "V_actual": V_actual,
        "distance_tokens": distance_tokens,
    }
    return document, meta


def build_dataset(
    tokenizer: TokenizerLike,
    V_targets: List[int] = (2000, 4000, 8000, 16000),
    distance_kinds: List[str] = ("short", "long"),
    entities_per_cell: int = 3,
    seed: int = 17,
) -> List[TestCase]:
    """Build a full evaluation set.

    Default config produces |V_targets| * |distance_kinds| * entities_per_cell
    cases. For the smoke test, use small V_targets and entities_per_cell=1.
    """
    rng = random.Random(seed)
    cases: List[TestCase] = []

    for V_target in V_targets:
        for distance_kind in distance_kinds:
            # Sample entities and year pairs without replacement-per-cell
            entity_pool = list(ENTITY_TEMPLATES)
            year_pool = list(CONTRADICTION_YEAR_PAIRS)
            rng.shuffle(entity_pool)
            rng.shuffle(year_pool)

            for i in range(entities_per_cell):
                entity = entity_pool[i % len(entity_pool)]
                year_first, year_second = year_pool[i % len(year_pool)]

                doc, meta = _build_document(
                    entity, year_first, year_second,
                    distance_kind, V_target, tokenizer, rng,
                )
                question = (
                    f"Based strictly on the document above, when was "
                    f"{entity['name']} founded? Quote the specific year or years "
                    f"that appear in the document. If the document contains any "
                    f"inconsistencies on this point, identify them explicitly."
                )
                case_id = f"V{V_target}_{distance_kind}_{i}_{entity['name'].replace(' ', '')}"
                cases.append(TestCase(
                    case_id=case_id,
                    entity_name=entity["name"],
                    sector=entity["sector"],
                    city=entity["city"],
                    year_first=year_first,
                    year_second=year_second,
                    distance_kind=distance_kind,
                    V_target=V_target,
                    V_actual=meta["V_actual"],
                    distance_tokens=meta["distance_tokens"],
                    pos_first_token=meta["pos_first_token"],
                    pos_second_token=meta["pos_second_token"],
                    document=doc,
                    question=question,
                ))
    return cases


def save_dataset(cases: List[TestCase], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([c.to_dict() for c in cases], f, indent=2)


def load_dataset(path: str | Path) -> List[TestCase]:
    with Path(path).open() as f:
        records = json.load(f)
    return [TestCase(**r) for r in records]


# --------------------------------------------------------------------------- #
# Simple verification helpers
# --------------------------------------------------------------------------- #
def verify_case(case: TestCase, tokenizer: TokenizerLike) -> dict:
    """Sanity-check a single test case. Returns a dict of verification results.

    Confirms:
    - Both year strings actually appear in the document
    - Token positions are close (within ±10 tokens) to where the tokenizer
      now reports the assertions sit
    - The two years are different
    """
    issues = []
    if case.year_first == case.year_second:
        issues.append("year_first == year_second (no contradiction)")
    if case.year_first not in case.document:
        issues.append(f"year_first {case.year_first!r} not in document")
    if case.year_second not in case.document:
        issues.append(f"year_second {case.year_second!r} not in document")

    # Recompute positions to confirm metadata
    prefix1_end = case.document.find(f"in {case.year_first}")
    prefix2_end = case.document.find(f"in {case.year_second}")
    if prefix1_end == -1 or prefix2_end == -1:
        issues.append("could not locate assertion sentences in document")
    elif prefix1_end >= prefix2_end:
        issues.append("year_first appears AFTER year_second in document order")

    V_actual_now = token_len(tokenizer, case.document)
    drift = abs(V_actual_now - case.V_actual)
    if drift > 5:
        issues.append(f"V_actual drift > 5 tokens ({drift})")

    return {
        "case_id": case.case_id,
        "ok": len(issues) == 0,
        "issues": issues,
        "V_actual_now": V_actual_now,
    }
