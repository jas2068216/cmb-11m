"""RAG-faithfulness dataset (scaled, 40 pairs) — expression-gap mode 4.

Each pair holds a question constant and gives a short context passage. SUPPORTED
states the answer in its middle sentence; UNSUPPORTED is identical except that
single answer-bearing sentence is swapped for a same-topic, similar-length
sentence that does not contain the answer. Structure is therefore matched by
construction (one sentence differs); the probe cell additionally reports a
token-length-matched-subset AUC to earn any near-ceiling number.

    SUPPORTED   (label 0): the document contains the answer.
    UNSUPPORTED (label 1): the document does not. A faithful model says so; the
                           gap is whether it answers from parametric memory.

Run under NEUTRAL_SYSTEM_PROMPT ("answer strictly from the provided document")
for the faithful baseline, and under a plain helpful prompt for the weak-prompt
(instruction-gated) gap.

40 pairs = 80 cases (40 SUPPORTED + 40 UNSUPPORTED).
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


def _mk(pid, topic, q, hint, intro, ans, filler, close):
    return RagPair(pid, topic, q, hint,
                   f"{intro} {ans} {close}",
                   f"{intro} {filler} {close}")


PILOT_PAIRS: List[RagPair] = [
    _mk(1, "Eiffel Tower height", "According to the passage, how tall is the Eiffel Tower?", "330",
        "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in Paris.",
        "Completed in 1889, it stands 330 metres tall.",
        "Completed in 1889, it quickly became a global cultural icon.",
        "It was designed by the company of engineer Gustave Eiffel."),
    _mk(2, "Author of 1984", "According to the passage, who wrote Nineteen Eighty-Four?", "Orwell",
        "Nineteen Eighty-Four is a dystopian novel set in a totalitarian society.",
        "It was written by George Orwell and published in 1949.",
        "It was first published in 1949 to wide and lasting acclaim.",
        "The book introduced concepts such as Big Brother and doublethink."),
    _mk(3, "Speed of sound", "According to the passage, how fast does sound travel in dry air at 20 degrees Celsius?", "343",
        "Sound travels as a vibration through a medium such as air or water.",
        "In dry air at 20 degrees Celsius, it travels at about 343 metres per second.",
        "Its perceived pitch depends on the frequency of the vibration.",
        "Its speed depends on the temperature and density of the medium."),
    _mk(4, "Largest planet", "According to the passage, which is the largest planet in the Solar System?", "Jupiter",
        "The Solar System contains eight planets orbiting the Sun.",
        "The largest of them is Jupiter, a gas giant.",
        "They are divided into rocky inner planets and outer giants.",
        "The planets vary widely in size, composition, and distance from the Sun."),
    _mk(5, "Berlin Wall fall", "According to the passage, in what year did the Berlin Wall fall?", "1989",
        "The Berlin Wall divided East and West Berlin for nearly three decades.",
        "It fell in 1989, a pivotal moment in the end of the Cold War.",
        "It became one of the most potent symbols of the Cold War.",
        "Its collapse paved the way for German reunification the following year."),
    _mk(6, "Currency of Japan", "According to the passage, what is the official currency of Japan?", "yen",
        "Japan is an island country in East Asia with a highly developed economy.",
        "Its official currency is the yen.",
        "Its output ranks among the largest national economies in the world.",
        "Tokyo, its capital, is one of the world's largest metropolitan areas."),
    _mk(7, "Kilimanjaro location", "According to the passage, in which country is Mount Kilimanjaro located?", "Tanzania",
        "Mount Kilimanjaro is the highest mountain in Africa, rising as a dormant volcano.",
        "It is located in Tanzania.",
        "It consists of three volcanic cones named Kibo, Mawenzi, and Shira.",
        "Its snow-capped summit attracts climbers from around the world."),
    _mk(8, "Photosynthesis pigment", "According to the passage, which pigment absorbs light during photosynthesis?", "chlorophyll",
        "Photosynthesis is the process by which plants convert light into chemical energy.",
        "The green pigment responsible for absorbing light is chlorophyll.",
        "It takes place largely within the leaves of green plants.",
        "The process releases oxygen as a by-product."),
    _mk(9, "Guitar strings", "According to the passage, how many strings does a standard guitar have?", "six",
        "The guitar is a fretted string instrument played by plucking or strumming.",
        "A standard guitar has six strings.",
        "It produces sound through the vibration of its strings over a body.",
        "It is used across many genres, from classical to rock."),
    _mk(10, "Telephone inventor", "According to the passage, who is commonly credited with inventing the telephone?", "Bell",
        "The telephone transformed long-distance communication in the late 19th century.",
        "It is commonly credited to Alexander Graham Bell, who patented it in 1876.",
        "Its development involved several inventors working on similar ideas at once.",
        "Early telephones required a direct wired connection between callers."),
    _mk(11, "Symbol for gold", "According to the passage, what is the chemical symbol for gold?", "Au",
        "Gold is a dense, soft metal prized since antiquity for jewellery and coinage.",
        "Its chemical symbol is Au, derived from the Latin aurum.",
        "It is one of the least chemically reactive of all the elements.",
        "It is highly resistant to corrosion and conducts electricity well."),
    _mk(12, "Soccer team size", "According to the passage, how many players does each team field at a time?", "eleven",
        "Association football, known as soccer in some countries, is the world's most popular sport.",
        "Each team fields eleven players at a time.",
        "It is governed internationally by the federation known as FIFA.",
        "Matches are played over two halves of 45 minutes each."),
    _mk(13, "Mona Lisa museum", "According to the passage, in which museum is the Mona Lisa housed?", "Louvre",
        "The Mona Lisa is a half-length portrait renowned for its enigmatic expression.",
        "It is housed in the Louvre Museum in Paris.",
        "It has been studied and admired by visitors for centuries.",
        "The work is among the most recognized paintings in the world."),
    _mk(14, "Great Wall location", "According to the passage, in which country is the Great Wall located?", "China",
        "The Great Wall is a series of fortifications built across hilly terrain.",
        "It is located in northern China.",
        "It was constructed and rebuilt by many successive dynasties.",
        "Today it draws millions of visitors each year."),
    _mk(15, "Largest ocean", "According to the passage, which is the largest ocean?", "Pacific",
        "The world's oceans form a connected body of salt water covering most of the planet.",
        "The largest of them is the Pacific Ocean.",
        "They are home to an enormous diversity of marine life.",
        "Their currents play a major role in regulating climate."),
    _mk(16, "Capital of Australia", "According to the passage, what is the capital of Australia?", "Canberra",
        "Australia is a country and continent in the Southern Hemisphere.",
        "Its capital city is Canberra.",
        "It is known for its distinctive wildlife and landscapes.",
        "Most of its population lives along the coast."),
    _mk(17, "Formula of water", "According to the passage, what is the chemical formula of water?", "H2O",
        "Water is a transparent, odourless substance essential to all known life.",
        "Its chemical formula is H2O.",
        "It exists naturally as a solid, a liquid, and a gas.",
        "It covers the majority of the Earth's surface."),
    _mk(18, "Tallest land animal", "According to the passage, what is the tallest land animal?", "giraffe",
        "The savannas of Africa are home to many large herbivores.",
        "The tallest land animal is the giraffe.",
        "These animals have adapted in many ways to their environment.",
        "They are a popular sight on safaris."),
    _mk(19, "Moons of Mars", "According to the passage, how many moons does Mars have?", "two",
        "Mars is the fourth planet from the Sun and is often called the Red Planet.",
        "It has two moons, Phobos and Deimos.",
        "Its surface features the largest volcano in the Solar System.",
        "Several missions have explored it with rovers."),
    _mk(20, "Light bulb inventor", "According to the passage, who is commonly credited with inventing the light bulb?", "Edison",
        "The incandescent light bulb transformed everyday life in the late 19th century.",
        "It is commonly credited to Thomas Edison.",
        "Its development drew on the work of many earlier experimenters.",
        "Early bulbs used a glowing filament inside a glass casing."),
    _mk(21, "Pride and Prejudice author", "According to the passage, who wrote Pride and Prejudice?", "Austen",
        "Pride and Prejudice is a classic novel of manners set in rural England.",
        "It was written by Jane Austen.",
        "It explores themes of marriage, class, and reputation.",
        "It remains widely read and adapted today."),
    _mk(22, "Freezing point Fahrenheit", "According to the passage, at what Fahrenheit temperature does water freeze?", "32",
        "The Fahrenheit scale is a temperature scale used mainly in the United States.",
        "On it, water freezes at 32 degrees.",
        "It divides the range between key reference points into degrees.",
        "It is named after the physicist who proposed it."),
    _mk(23, "Largest mammal", "According to the passage, what is the largest mammal?", "blue whale",
        "Mammals range enormously in size, from tiny shrews to ocean giants.",
        "The largest mammal is the blue whale.",
        "They are found in nearly every habitat on Earth.",
        "Many species are the focus of conservation efforts."),
    _mk(24, "Element symbol O", "According to the passage, which element has the symbol O?", "oxygen",
        "The periodic table organizes the chemical elements by their properties.",
        "The element with the symbol O is oxygen.",
        "Each element is represented by a one- or two-letter symbol.",
        "The table is a cornerstone of modern chemistry."),
    _mk(25, "First Moon landing year", "According to the passage, in what year did the first crewed Moon landing occur?", "1969",
        "The Apollo program was a series of crewed spaceflights run by NASA.",
        "The first crewed Moon landing took place in 1969.",
        "It involved years of preparation and many test missions.",
        "It remains a landmark achievement in exploration."),
    _mk(26, "Currency of the UK", "According to the passage, what is the official currency of the United Kingdom?", "pound",
        "The United Kingdom is a country in north-western Europe.",
        "Its official currency is the pound sterling.",
        "It comprises four constituent nations.",
        "London is its capital and largest city."),
    _mk(27, "Sides of a hexagon", "According to the passage, how many sides does a hexagon have?", "six",
        "Polygons are flat shapes bounded by straight sides.",
        "A hexagon has six sides.",
        "They are classified by the number of their sides.",
        "They appear throughout geometry and nature."),
    _mk(28, "Ringed planet", "According to the passage, which planet is famous for its prominent rings?", "Saturn",
        "The outer Solar System contains several large gas giants.",
        "The planet famous for its prominent rings is Saturn.",
        "These planets are composed mostly of hydrogen and helium.",
        "They have been visited by robotic spacecraft."),
    _mk(29, "Sistine Chapel painter", "According to the passage, who painted the ceiling of the Sistine Chapel?", "Michelangelo",
        "The Sistine Chapel is celebrated for its Renaissance frescoes.",
        "Its ceiling was painted by Michelangelo.",
        "Its artwork attracts vast numbers of visitors each year.",
        "It is located within Vatican City."),
    _mk(30, "Hardest natural material", "According to the passage, what is the hardest known natural material?", "diamond",
        "Minerals vary widely in hardness, ranked on a standard scale.",
        "The hardest known natural material is diamond.",
        "Hardness affects how a material is used and worked.",
        "Geologists use hardness to help identify minerals."),
    _mk(31, "Basketball team size", "According to the passage, how many players does each basketball team field at a time?", "five",
        "Basketball is a team sport played on a rectangular court.",
        "Each team fields five players at a time.",
        "Points are scored by shooting a ball through a hoop.",
        "It is played both indoors and outdoors worldwide."),
    _mk(32, "Capital of Canada", "According to the passage, what is the capital of Canada?", "Ottawa",
        "Canada is a country occupying much of northern North America.",
        "Its capital city is Ottawa.",
        "It is known for its vast forests and lakes.",
        "It has two official languages, English and French."),
    _mk(33, "Gas humans exhale", "According to the passage, which gas do humans primarily exhale?", "carbon dioxide",
        "Human respiration exchanges gases between the body and the air.",
        "The gas humans primarily exhale is carbon dioxide.",
        "This exchange takes place within the lungs.",
        "It occurs continuously throughout life."),
    _mk(34, "Violin strings", "According to the passage, how many strings does a standard violin have?", "four",
        "The violin is a wooden string instrument played with a bow.",
        "A standard violin has four strings.",
        "It is the smallest member of its instrument family.",
        "It features prominently in orchestras and folk music."),
    _mk(35, "Evolution theory author", "According to the passage, who proposed the theory of evolution by natural selection?", "Darwin",
        "The theory of evolution by natural selection reshaped the biological sciences.",
        "It was proposed by Charles Darwin.",
        "It explains how species change over many generations.",
        "It is supported by extensive evidence from many fields."),
    _mk(36, "Largest internal organ", "According to the passage, what is the largest internal organ in the human body?", "liver",
        "The human body contains many organs with specialized roles.",
        "The largest internal organ is the liver.",
        "Each organ contributes to the body's overall function.",
        "Together they keep the body in balance."),
    _mk(37, "Adult teeth count", "According to the passage, how many teeth are in a full set of adult teeth?", "32",
        "Human teeth develop in two sets over a lifetime.",
        "A full set of adult teeth numbers 32.",
        "They are used for biting and chewing food.",
        "Dental care helps preserve them through life."),
    _mk(38, "Pyramids of Giza country", "According to the passage, in which country are the Pyramids of Giza?", "Egypt",
        "The Pyramids of Giza are among the most famous ancient monuments.",
        "They are located in Egypt.",
        "They were built as tombs for ancient rulers.",
        "They have fascinated travellers for millennia."),
    _mk(39, "Fastest land animal", "According to the passage, what is the fastest land animal?", "cheetah",
        "Speed is a vital adaptation for many hunting animals.",
        "The fastest land animal is the cheetah.",
        "Such animals rely on speed to catch their prey.",
        "They are found in several regions of the world."),
    _mk(40, "Colours in a rainbow", "According to the passage, how many colours is a rainbow traditionally described as having?", "seven",
        "A rainbow forms when light is refracted and dispersed by water droplets.",
        "It is traditionally described as having seven colours.",
        "It appears as an arc across the sky.",
        "Its colours always follow the same order."),
]


def supported_prompts() -> List[str]:
    return [p.supported_prompt() for p in PILOT_PAIRS]


def unsupported_prompts() -> List[str]:
    return [p.unsupported_prompt() for p in PILOT_PAIRS]


def all_labeled_prompts():
    """[(prompt, label, pair_id, condition), ...]. label 0=supported, 1=unsupported."""
    rows = []
    for p in PILOT_PAIRS:
        rows.append((p.supported_prompt(), 0, p.pair_id, "supported"))
        rows.append((p.unsupported_prompt(), 1, p.pair_id, "unsupported"))
    return rows


if __name__ == "__main__":
    rows = all_labeled_prompts()
    print(f"{len(rows)} prompts ({len(rows)//2} pairs)")
