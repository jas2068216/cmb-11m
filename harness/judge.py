"""Judgment for CMB-LLM Step 1.

Scores each InferenceResult against its TestCase ground truth on three axes:

1. mentioned_first  — did the model cite year_first?
2. mentioned_second — did the model cite year_second?
3. flagged         — did the model explicitly call out the inconsistency?

Detection outcome (the headline metric):
- "detected"      : both years mentioned AND inconsistency flagged
- "partial"       : both years mentioned but no flagging language
- "missed"        : only one year cited, or neither — model committed to a
                    single answer without surfacing the conflict
- "ambiguous"     : output doesn't clearly contain either year (rare;
                    usually means the model refused or went off-topic)

Rate ρ for Step 1 = fraction of cases that are NOT "detected".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import json

from .dataset import TestCase
from .inference import InferenceResult


# Detection-language patterns. Case-insensitive; word boundaries where useful.
FLAG_PATTERNS = [
    r"\bcontradict",
    r"\binconsisten",
    r"\bconflict",
    r"\bdiscrepanc",
    r"\bmismatch",
    r"\bdiffer(s|ing|ent)\b",
    r"\bboth\b.*\byears?\b",
    r"\btwo\s+different\s+(years?|dates?)",
    r"\bappears?\s+twice",
    r"\bstates?\s+two\b",
    r"\bcannot\s+(both\s+)?be\s+(true|correct)",
    r"\bone\s+says.*another\s+says",
]

FLAG_RE = re.compile("|".join(FLAG_PATTERNS), re.IGNORECASE)


@dataclass
class Judgment:
    case_id: str
    mentioned_first: bool
    mentioned_second: bool
    flagged: bool
    outcome: str       # detected | partial | missed | ambiguous
    notes: str         # short explanation for spot-checks

    def to_dict(self) -> dict:
        return asdict(self)


def _year_mentioned(response: str, year: str) -> bool:
    # Anchor on word boundaries so e.g. "2018" doesn't match "20180" if it
    # ever appeared. Years are 4 digits; regex \b is fine.
    return bool(re.search(rf"\b{re.escape(year)}\b", response))


def judge_case(case: TestCase, result: InferenceResult) -> Judgment:
    resp = result.response
    mentioned_first = _year_mentioned(resp, case.year_first)
    mentioned_second = _year_mentioned(resp, case.year_second)
    flagged = bool(FLAG_RE.search(resp))

    if not (mentioned_first or mentioned_second):
        outcome = "ambiguous"
        notes = "neither year cited in response"
    elif mentioned_first and mentioned_second and flagged:
        outcome = "detected"
        notes = "both years cited and flagged"
    elif mentioned_first and mentioned_second and not flagged:
        outcome = "partial"
        notes = "both years cited but no explicit flag"
    else:
        outcome = "missed"
        which = case.year_first if mentioned_first else case.year_second
        notes = f"committed to single year {which}; conflict not surfaced"

    return Judgment(
        case_id=case.case_id,
        mentioned_first=mentioned_first,
        mentioned_second=mentioned_second,
        flagged=flagged,
        outcome=outcome,
        notes=notes,
    )


def judge_all(cases: List[TestCase],
              results: List[InferenceResult]) -> List[Judgment]:
    """Pair each result with its case and produce judgments."""
    case_by_id = {c.case_id: c for c in cases}
    judgments: List[Judgment] = []
    for r in results:
        case = case_by_id.get(r.case_id)
        if case is None:
            continue
        judgments.append(judge_case(case, r))
    return judgments


def save_judgments(judgments: List[Judgment], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([j.to_dict() for j in judgments], f, indent=2)


def load_judgments(path: str | Path) -> List[Judgment]:
    with Path(path).open() as f:
        records = json.load(f)
    return [Judgment(**r) for r in records]
