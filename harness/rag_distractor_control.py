"""RAG surface-confound control — does the probe read 'answer supported' or just
'answer token present'? (Addresses the reviewers' AUC=1.000 triviality objection.)

Each item has THREE contexts, all 3 sentences, structure-matched:
  supported   : the answer is stated in its proper role.
  unsupported : the answer is omitted.
  distractor  : the answer TOKEN appears, but in an IRRELEVANT role (e.g. "330" as
                a visitor count, not the tower's height) -- so the answer to the
                question is NOT supported.

Test: train the probe on supported(0) vs unsupported(1); then score distractors.
  - epistemic probe  -> distractors classified UNSUPPORTED (answer not given).
  - surface probe    -> distractors classified SUPPORTED (token is present).
All answers are numeric/recurring so the token can sit naturally in another role.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List


@dataclass
class DItem:
    pair_id: int
    question: str
    answer_hint: str
    supported: str
    unsupported: str
    distractor: str

    def s_prompt(self): return f"{self.supported}\n\nQuestion: {self.question}"
    def u_prompt(self): return f"{self.unsupported}\n\nQuestion: {self.question}"
    def d_prompt(self): return f"{self.distractor}\n\nQuestion: {self.question}"


ITEMS: List[DItem] = [
    DItem(1, "According to the passage, how tall is the Eiffel Tower?", "330",
          "The Eiffel Tower is a wrought-iron tower in Paris. It stands 330 metres tall. It was completed in 1889.",
          "The Eiffel Tower is a wrought-iron tower in Paris. It is among the most visited monuments on Earth. It was completed in 1889.",
          "The Eiffel Tower is a wrought-iron tower in Paris. In its opening months it welcomed about 330 thousand visitors. It was completed in 1889."),
    DItem(2, "According to the passage, how fast does sound travel in dry air at 20 degrees Celsius?", "343",
          "Sound travels as a vibration through air. In dry air at 20 degrees Celsius it moves at about 343 metres per second. Its speed varies with temperature.",
          "Sound travels as a vibration through air. Its pitch depends on the frequency of the vibration. Its speed varies with temperature.",
          "Sound travels as a vibration through air. The textbook devotes 343 pages to acoustics. Its speed varies with temperature."),
    DItem(3, "According to the passage, in what year did the Berlin Wall fall?", "1989",
          "The Berlin Wall divided the city for decades. It fell in 1989, near the end of the Cold War. Its collapse led to reunification.",
          "The Berlin Wall divided the city for decades. It became a stark symbol of the Cold War. Its collapse led to reunification.",
          "The Berlin Wall divided the city for decades. A nearby museum preserves 1989 fragments of its concrete. Its collapse led to reunification."),
    DItem(4, "According to the passage, how many strings does a standard guitar have?", "six",
          "The guitar is a fretted string instrument. A standard guitar has six strings. It spans many genres.",
          "The guitar is a fretted string instrument. It produces sound through vibrating strings over a body. It spans many genres.",
          "The guitar is a fretted string instrument. The shop displayed it beside six other instruments. It spans many genres."),
    DItem(5, "According to the passage, how many players does each team field at a time?", "eleven",
          "Association football is the world's most popular sport. Each team fields eleven players at a time. Matches last ninety minutes.",
          "Association football is the world's most popular sport. It is governed internationally by FIFA. Matches last ninety minutes.",
          "Association football is the world's most popular sport. The club won eleven trophies in one decade. Matches last ninety minutes."),
    DItem(6, "According to the passage, in what year did the first crewed Moon landing occur?", "1969",
          "The Apollo program sent crews toward the Moon. The first crewed landing happened in 1969. It followed years of test flights.",
          "The Apollo program sent crews toward the Moon. It involved many uncrewed test missions first. It followed years of test flights.",
          "The Apollo program sent crews toward the Moon. The archive holds 1969 photographs from the missions. It followed years of test flights."),
    DItem(7, "According to the passage, what is the boiling point of water in Celsius at sea level?", "100",
          "Water is essential to life. At sea level it boils at 100 degrees Celsius. Its boiling point falls at altitude.",
          "Water is essential to life. It exists as a solid, a liquid, and a gas. Its boiling point falls at altitude.",
          "Water is essential to life. The survey sampled 100 lakes across the region. Its boiling point falls at altitude."),
    DItem(8, "According to the passage, at what Celsius temperature does water freeze?", "0",
          "The Celsius scale is widely used for temperature. Water freezes at 0 degrees on it. It is named after a Swedish astronomer.",
          "The Celsius scale is widely used for temperature. It divides a key range into one hundred degrees. It is named after a Swedish astronomer.",
          "The Celsius scale is widely used for temperature. The instrument logged 0 errors during calibration. It is named after a Swedish astronomer."),
    DItem(9, "According to the passage, how many moons does Mars have?", "two",
          "Mars is the fourth planet from the Sun. It has two moons, Phobos and Deimos. Its surface is rusty red.",
          "Mars is the fourth planet from the Sun. It hosts the tallest volcano in the solar system. Its surface is rusty red.",
          "Mars is the fourth planet from the Sun. Two rovers currently operate on its surface. Its surface is rusty red."),
    DItem(10, "According to the passage, how many planets are in the solar system?", "eight",
          "Our solar system orbits a single star. It contains eight planets. They range from rocky worlds to gas giants.",
          "Our solar system orbits a single star. It formed long ago from a disc of gas. They range from rocky worlds to gas giants.",
          "Our solar system orbits a single star. Eight spacecraft have left it entirely. They range from rocky worlds to gas giants."),
    DItem(11, "According to the passage, how many continents are there on Earth?", "seven",
          "Earth's land is divided into large landmasses. There are seven continents. They vary widely in size and climate.",
          "Earth's land is divided into large landmasses. They drift slowly over geological time. They vary widely in size and climate.",
          "Earth's land is divided into large landmasses. The atlas devotes seven chapters to oceans. They vary widely in size and climate."),
    DItem(12, "According to the passage, at what Fahrenheit temperature does water freeze?", "32",
          "The Fahrenheit scale is used mainly in the United States. On it water freezes at 32 degrees. It is named after a physicist.",
          "The Fahrenheit scale is used mainly in the United States. It divides a reference range into many degrees. It is named after a physicist.",
          "The Fahrenheit scale is used mainly in the United States. The manual lists 32 conversion examples. It is named after a physicist."),
    DItem(13, "According to the passage, how many degrees are in a right angle?", "90",
          "Angles are measured in degrees. A right angle is exactly 90 degrees. They appear throughout geometry.",
          "Angles are measured in degrees. They are formed wherever two lines meet. They appear throughout geometry.",
          "Angles are measured in degrees. The course covered 90 practice problems on them. They appear throughout geometry."),
    DItem(14, "According to the passage, how many hours are in a day?", "24",
          "A day is the basic unit of the calendar. It contains 24 hours. It is set by Earth's rotation.",
          "A day is the basic unit of the calendar. It is divided into morning and evening. It is set by Earth's rotation.",
          "A day is the basic unit of the calendar. The exhibit ran for 24 weeks. It is set by Earth's rotation."),
    DItem(15, "According to the passage, how many players does each basketball team field at a time?", "five",
          "Basketball is a fast team sport. Each team fields five players at a time. Points are scored through a raised hoop.",
          "Basketball is a fast team sport. It is played indoors and outdoors worldwide. Points are scored through a raised hoop.",
          "Basketball is a fast team sport. The league added five new teams last year. Points are scored through a raised hoop."),
    DItem(16, "According to the passage, how many strings does a standard violin have?", "four",
          "The violin is a bowed string instrument. A standard violin has four strings. It anchors the string section.",
          "The violin is a bowed string instrument. It is the smallest of its family. It anchors the string section.",
          "The violin is a bowed string instrument. The soloist owned four of them. It anchors the string section."),
]


def train_rows():
    """[(prompt, label), ...]; label 0 = supported, 1 = unsupported."""
    rows = []
    for it in ITEMS:
        rows.append((it.s_prompt(), 0)); rows.append((it.u_prompt(), 1))
    return rows


def distractor_prompts():
    return [(it.pair_id, it.d_prompt(), it.answer_hint) for it in ITEMS]
