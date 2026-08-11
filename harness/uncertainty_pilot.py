"""Uncertainty pilot dataset for testing the multi-task expression-gap claim.

DESIGN
------
Each pair shares an entity anchor and surface form. One question (KNOWN) has
a well-attested answer the model should be confident about. The other
(UNCERTAIN) carries a false presupposition — the form is plausible but no
real fact exists. Hypothesis: layer-17 last-input-token activations differ
linearly between KNOWN and UNCERTAIN cases, mirroring the contradiction
finding. AUC > 0.7 in cross-validation is a green light to expand the
multi-task generalization.

Schema:
    UncertaintyPair(
        pair_id, entity, known_question, known_answer_hint,
        uncertain_question, uncertain_reason
    )

20 cases = 10 KNOWN + 10 UNCERTAIN (labeled by pair_id × condition).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class UncertaintyPair:
    pair_id: int
    entity: str
    known_question: str
    known_answer_hint: str          # for sanity-checking the model knows it
    uncertain_question: str
    uncertain_reason: str           # why the presupposition is false


PILOT_PAIRS: List[UncertaintyPair] = [
    UncertaintyPair(
        pair_id=1, entity="Albert Einstein",
        known_question=(
            "In what year did Albert Einstein publish his paper on the "
            "photoelectric effect?"
        ),
        known_answer_hint="1905",
        uncertain_question=(
            "In what year did Albert Einstein publish his paper on cold "
            "fusion?"
        ),
        uncertain_reason=(
            "Einstein never published on cold fusion; cold fusion claims "
            "originated in 1989, decades after his death."
        ),
    ),
    UncertaintyPair(
        pair_id=2, entity="Leonardo da Vinci",
        known_question=(
            "What is the subject of Leonardo da Vinci's painting "
            "'Salvator Mundi'?"
        ),
        known_answer_hint="Christ as Savior of the World",
        uncertain_question=(
            "What is the subject of Leonardo da Vinci's painting "
            "'Salvator Sapientiae'?"
        ),
        uncertain_reason="No such Leonardo painting exists.",
    ),
    UncertaintyPair(
        pair_id=3, entity="Apple Inc.",
        known_question=(
            "Who were the three co-founders of Apple Inc.?"
        ),
        known_answer_hint="Steve Jobs, Steve Wozniak, Ronald Wayne",
        uncertain_question=(
            "Who was the silent fourth co-founder of Apple Inc. who "
            "provided initial capital before incorporation?"
        ),
        uncertain_reason="No such fourth founder exists.",
    ),
    UncertaintyPair(
        pair_id=4, entity="Marie Curie",
        known_question=(
            "Marie Curie won Nobel Prizes in which two scientific fields?"
        ),
        known_answer_hint="Physics (1903) and Chemistry (1911)",
        uncertain_question=(
            "Marie Curie's lesser-known third Nobel Prize was awarded in "
            "which field?"
        ),
        uncertain_reason="She won exactly two Nobel Prizes; there is no third.",
    ),
    UncertaintyPair(
        pair_id=5, entity="Tokyo",
        known_question=(
            "Approximately what is the population of the Tokyo metropolitan "
            "area?"
        ),
        known_answer_hint="approximately 37 million",
        uncertain_question=(
            "Approximately what is the population of Tokyo's underground "
            "residential district, completed in 1998?"
        ),
        uncertain_reason="No such underground residential district exists.",
    ),
    UncertaintyPair(
        pair_id=6, entity="Beethoven",
        known_question=(
            "How many complete symphonies did Ludwig van Beethoven compose?"
        ),
        known_answer_hint="9",
        uncertain_question=(
            "What is the title of Beethoven's tenth symphony, completed "
            "posthumously by Carl Czerny in 1830?"
        ),
        uncertain_reason=(
            "Beethoven left only sketches for a 10th; Czerny did not complete it."
        ),
    ),
    UncertaintyPair(
        pair_id=7, entity="Modern Olympic Games",
        known_question=(
            "In what year were the first modern Olympic Games held?"
        ),
        known_answer_hint="1896",
        uncertain_question=(
            "In what year were the first modern Olympic Games to feature "
            "underwater diving as a medal event held?"
        ),
        uncertain_reason=(
            "Underwater diving has never been a medal event in the Olympics."
        ),
    ),
    UncertaintyPair(
        pair_id=8, entity="DNA",
        known_question=(
            "Who is credited with discovering the double-helix structure of "
            "DNA?"
        ),
        known_answer_hint="James Watson and Francis Crick, using Rosalind Franklin's data",
        uncertain_question=(
            "Which Soviet biophysicist independently discovered the "
            "triple-helix structure of DNA in 1949?"
        ),
        uncertain_reason=(
            "No Soviet scientist independently discovered a triple-helix; "
            "DNA is a double helix."
        ),
    ),
    UncertaintyPair(
        pair_id=9, entity="Shakespeare",
        known_question=(
            "Who is the protagonist of William Shakespeare's tragedy "
            "'Hamlet'?"
        ),
        known_answer_hint="Prince Hamlet",
        uncertain_question=(
            "Who is the protagonist of William Shakespeare's lost play "
            "'Cardenio Part Two', referenced in the Stationers' Register?"
        ),
        uncertain_reason=(
            "There is one lost play attributed (Cardenio); there is no "
            "'Cardenio Part Two'."
        ),
    ),
    UncertaintyPair(
        pair_id=10, entity="SpaceX",
        known_question=(
            "In what year did SpaceX achieve its first successful orbital "
            "launch?"
        ),
        known_answer_hint="2008 (Falcon 1 Flight 4)",
        uncertain_question=(
            "In what year did SpaceX achieve its first successful Mars "
            "sample return mission?"
        ),
        uncertain_reason="SpaceX has not performed a Mars sample return mission.",
    ),
]


def known_prompts() -> List[str]:
    return [p.known_question for p in PILOT_PAIRS]


def uncertain_prompts() -> List[str]:
    return [p.uncertain_question for p in PILOT_PAIRS]


def all_labeled_prompts():
    """Returns [(prompt, label, pair_id, condition), ...] for the 20 cases.

    label: 0 = known (low uncertainty), 1 = uncertain (high uncertainty)
    """
    rows = []
    for p in PILOT_PAIRS:
        rows.append((p.known_question, 0, p.pair_id, "known"))
        rows.append((p.uncertain_question, 1, p.pair_id, "uncertain"))
    return rows


if __name__ == "__main__":
    rows = all_labeled_prompts()
    print(f"{len(rows)} prompts ({len(rows)//2} pairs)")
    for prompt, label, pid, cond in rows[:4]:
        print(f"  pair={pid}  cond={cond}  label={label}")
        print(f"    {prompt}")
