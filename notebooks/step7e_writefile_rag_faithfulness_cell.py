# Paste this into a NEW Colab cell. Materializes harness/rag_faithfulness_pilot.py.
# Run AFTER pipeline_bootstrap_cell.py and BEFORE step7f_rag_faithfulness_probe_cell.py.

%%writefile /content/cmb_llm/harness/rag_faithfulness_pilot.py
"""RAG-faithfulness pilot dataset for the expression-gap claim, mode 4.

SUPPORTED   (label 0): the document contains the answer.
UNSUPPORTED (label 1): the document is identical EXCEPT the answer-bearing
                       sentence is swapped for a same-topic, same-length
                       sentence that does not contain the answer (structure
                       matched by construction). A faithful model should say the
                       passage doesn't state it; the gap is whether it answers
                       from parametric memory anyway.

Run under NEUTRAL_SYSTEM_PROMPT ("answer strictly from the provided document").
12 pairs = 24 cases.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class RagPair:
    pair_id: int
    topic: str
    question: str
    answer_hint: str
    supported_context: str
    unsupported_context: str

    def supported_prompt(self) -> str:
        return f"{self.supported_context}\n\nQuestion: {self.question}"

    def unsupported_prompt(self) -> str:
        return f"{self.unsupported_context}\n\nQuestion: {self.question}"


PILOT_PAIRS: List[RagPair] = [
    RagPair(1, "Eiffel Tower height",
            "According to the passage, how tall is the Eiffel Tower?", "330",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
            "Completed in 1889, it stands 330 metres tall. "
            "It was designed by the company of engineer Gustave Eiffel for the World's Fair.",
            "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris, France. "
            "Completed in 1889, it quickly became a global cultural icon. "
            "It was designed by the company of engineer Gustave Eiffel for the World's Fair."),
    RagPair(2, "Author of 1984",
            "According to the passage, who wrote Nineteen Eighty-Four?", "Orwell",
            "Nineteen Eighty-Four is a dystopian novel set in a totalitarian society. "
            "It was written by George Orwell and published in 1949. "
            "The book introduced concepts such as Big Brother and doublethink.",
            "Nineteen Eighty-Four is a dystopian novel set in a totalitarian society. "
            "It was first published in 1949 to wide and lasting acclaim. "
            "The book introduced concepts such as Big Brother and doublethink."),
    RagPair(3, "Speed of sound",
            "According to the passage, how fast does sound travel in dry air at 20 degrees Celsius?", "343",
            "Sound travels as a vibration through a medium such as air or water. "
            "In dry air at 20 degrees Celsius, it travels at about 343 metres per second. "
            "Its speed depends on the temperature and density of the medium.",
            "Sound travels as a vibration through a medium such as air or water. "
            "Its perceived pitch depends on the frequency of the vibration. "
            "Its speed depends on the temperature and density of the medium."),
    RagPair(4, "Largest planet",
            "According to the passage, which is the largest planet in the Solar System?", "Jupiter",
            "The Solar System contains eight planets orbiting the Sun. "
            "The largest of them is Jupiter, a gas giant. "
            "The planets vary widely in size, composition, and distance from the Sun.",
            "The Solar System contains eight planets orbiting the Sun. "
            "They are divided into rocky inner planets and outer giants. "
            "The planets vary widely in size, composition, and distance from the Sun."),
    RagPair(5, "Berlin Wall fall",
            "According to the passage, in what year did the Berlin Wall fall?", "1989",
            "The Berlin Wall divided East and West Berlin for nearly three decades. "
            "It fell in 1989, a pivotal moment in the end of the Cold War. "
            "Its collapse paved the way for German reunification the following year.",
            "The Berlin Wall divided East and West Berlin for nearly three decades. "
            "It became one of the most potent symbols of the Cold War. "
            "Its collapse paved the way for German reunification the following year."),
    RagPair(6, "Currency of Japan",
            "According to the passage, what is the official currency of Japan?", "yen",
            "Japan is an island country in East Asia with a highly developed economy. "
            "Its official currency is the yen. "
            "Tokyo, its capital, is one of the world's largest metropolitan areas.",
            "Japan is an island country in East Asia with a highly developed economy. "
            "Its output ranks among the largest national economies in the world. "
            "Tokyo, its capital, is one of the world's largest metropolitan areas."),
    RagPair(7, "Kilimanjaro location",
            "According to the passage, in which country is Mount Kilimanjaro located?", "Tanzania",
            "Mount Kilimanjaro is the highest mountain in Africa, rising as a dormant volcano. "
            "It is located in Tanzania. "
            "Its snow-capped summit attracts climbers from around the world.",
            "Mount Kilimanjaro is the highest mountain in Africa, rising as a dormant volcano. "
            "It consists of three volcanic cones named Kibo, Mawenzi, and Shira. "
            "Its snow-capped summit attracts climbers from around the world."),
    RagPair(8, "Photosynthesis pigment",
            "According to the passage, which pigment absorbs light during photosynthesis?", "chlorophyll",
            "Photosynthesis is the process by which plants convert light into chemical energy. "
            "The green pigment responsible for absorbing light is chlorophyll. "
            "The process releases oxygen as a by-product.",
            "Photosynthesis is the process by which plants convert light into chemical energy. "
            "It takes place largely within the leaves of green plants. "
            "The process releases oxygen as a by-product."),
    RagPair(9, "Guitar strings",
            "According to the passage, how many strings does a standard guitar have?", "six",
            "The guitar is a fretted string instrument played by plucking or strumming. "
            "A standard guitar has six strings. "
            "It is used across many genres, from classical to rock.",
            "The guitar is a fretted string instrument played by plucking or strumming. "
            "It produces sound through the vibration of its strings over a body. "
            "It is used across many genres, from classical to rock."),
    RagPair(10, "Telephone inventor",
            "According to the passage, who is commonly credited with inventing the telephone?", "Bell",
            "The telephone transformed long-distance communication in the late 19th century. "
            "It is commonly credited to Alexander Graham Bell, who patented it in 1876. "
            "Early telephones required a direct wired connection between callers.",
            "The telephone transformed long-distance communication in the late 19th century. "
            "Its development involved several inventors working on similar ideas at once. "
            "Early telephones required a direct wired connection between callers."),
    RagPair(11, "Symbol for gold",
            "According to the passage, what is the chemical symbol for gold?", "Au",
            "Gold is a dense, soft metal prized since antiquity for jewellery and coinage. "
            "Its chemical symbol is Au, derived from the Latin aurum. "
            "It is highly resistant to corrosion and conducts electricity well.",
            "Gold is a dense, soft metal prized since antiquity for jewellery and coinage. "
            "It is one of the least chemically reactive of all the elements. "
            "It is highly resistant to corrosion and conducts electricity well."),
    RagPair(12, "Soccer team size",
            "According to the passage, how many players does each team field at a time?", "eleven",
            "Association football, known as soccer in some countries, is the world's most popular sport. "
            "Each team fields eleven players at a time. "
            "Matches are played over two halves of 45 minutes each.",
            "Association football, known as soccer in some countries, is the world's most popular sport. "
            "It is governed internationally by the federation known as FIFA. "
            "Matches are played over two halves of 45 minutes each."),
]


def supported_prompts() -> List[str]:
    return [p.supported_prompt() for p in PILOT_PAIRS]


def unsupported_prompts() -> List[str]:
    return [p.unsupported_prompt() for p in PILOT_PAIRS]


def all_labeled_prompts():
    """Returns [(prompt, label, pair_id, condition), ...] for the 24 cases.

    label: 0 = supported (answer in context), 1 = unsupported (answer absent)
    """
    rows = []
    for p in PILOT_PAIRS:
        rows.append((p.supported_prompt(), 0, p.pair_id, "supported"))
        rows.append((p.unsupported_prompt(), 1, p.pair_id, "unsupported"))
    return rows
