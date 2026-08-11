# =============================================================================
# EXTRA CELL H — STANDALONE: PARAPHRASE HOLDOUT (GPU, ~15 min on A100)
# The last unanswered reviewer attack: "the probe has never been tested
# off-template." Trains the layer-17 probe on the ORIGINAL 60-pair benchmark,
# then tests it ZERO-SHOT on a full paraphrase set (same meaning, template
# destroyed: different syntax per item). No other cells needed — paste and run.
# =============================================================================
import os, json
import numpy as np
try:
    from google.colab import drive; drive.mount('/content/drive')
except Exception: pass
OUT = '/content/drive/MyDrive/cmb_llm_intervention'; os.makedirs(OUT, exist_ok=True)
NEUTRAL_SYSTEM_PROMPT = "You are a careful analyst. Answer questions strictly from the provided document."

# ---- ORIGINAL benchmark (60 pairs) ------------------------------------------
"""60-pair uncertainty benchmark (V3), extracted from step6e for reuse.
30 fabricated-entity (false-presupposition) + 30 entity-unknowable pairs.
Each pair is (known_prompt, uncertain_prompt). label 0=known, 1=uncertain."""

from __future__ import annotations
from typing import List, Tuple

FAB_PAIRS = [
    ("What was the title of Einstein's 1905 paper, the one introducing special relativity, that revolutionized physics?",
     "What was the title of Einstein's 1905 paper, the one introducing quantum gravity theory, that revolutionized physics?"),
    ("What is the name of Leonardo da Vinci's portrait, the one painted around 1503 in Florence, that hangs in the Louvre?",
     "What is the name of Leonardo da Vinci's portrait, the one Pope Leo X commissioned in 1402 in Rome, that hangs in the Louvre?"),
    ("Who was the co-founder of Apple Inc., the one who left the company early and sold his stake, often forgotten in the founding story?",
     "Who was the fourth co-founder of Apple Inc., the one Jobs met at a Cupertino synagogue in 1974, often forgotten in the founding story?"),
    ("What scientific field did Marie Curie win her second Nobel Prize in, the one she received in 1911 alone, while working in Paris?",
     "What scientific field did Marie Curie win her third Nobel Prize in, the one she received in 1923 alone, while working in Paris?"),
    ("What is the approximate population of Tokyo's metropolitan area, the one including the 23 special wards, as measured in recent census data?",
     "What is the approximate population of Tokyo's underground district Shibuya-Kita, the one beneath the Hakuro Tunnel system, as measured in recent census data?"),
    ("What is the name of Beethoven's ninth symphony, the one premiering in Vienna in 1824, that incorporates a choral finale?",
     "What is the name of Beethoven's tenth symphony, the one premiering in Vienna in 1828, that incorporates a choral finale?"),
    ("In what year were the first modern Olympic Games held, the ones organized by Pierre de Coubertin's committee, that took place in Athens?",
     "In what year were the second secret modern Olympic Games held, the ones organized by Pierre de Coubertin's brother, that took place in Athens?"),
    ("Which scientist co-discovered the double helix structure of DNA, the one collaborating with Francis Crick at Cambridge, in 1953?",
     "Which scientist co-discovered the triple helix structure of DNA, the one collaborating with Francis Crick at Cambridge, in 1949?"),
    ("Who is the protagonist of Shakespeare's play Hamlet, the one set in Denmark in the late medieval period, that explores themes of revenge?",
     "Who is the protagonist of Shakespeare's lost play Cardenio Part Two, the one set in Denmark in the late medieval period, that explores themes of revenge?"),
    ("What is the name of the SpaceX rocket, the one used for crewed missions to the ISS, that first launched astronauts in 2020?",
     "What is the name of the SpaceX rocket, the one used for the secret 2018 Mars colonization mission, that first launched astronauts in 2018?"),
    ("What is the height of Mount Everest, the one measured by modern GPS surveys, accepted by international convention?",
     "What is the height of Mount Everest's western secondary peak, the one mapped by the 1953 British survey, accepted by international convention?"),
    ("What is the name of Mozart's final opera, the one premiering in Vienna in September 1791, with a German libretto?",
     "What is the name of Mozart's lost twentieth opera, the one premiering in Salzburg in October 1793, with a German libretto?"),
    ("Who was the first emperor of the Roman Empire, the one assuming the title in 27 BCE, ending the Republic?",
     "Who was the seventh secret emperor of the Roman Empire, the one assuming the title in 14 CE, ending the Republic?"),
    ("During which Egyptian dynasty was the Great Pyramid of Giza built, the one ruled by Khufu around 2560 BCE, in the Old Kingdom?",
     "During which Egyptian dynasty was the Great Pyramid of Giza secretly rebuilt, the one ruled by Akhenaten around 1340 BCE, in the New Kingdom?"),
    ("On which famous ship did Charles Darwin make his five-year voyage, the one departing England in 1831, that shaped his theory?",
     "On which famous ship did Charles Darwin make his second polar voyage, the one departing England in 1841, that shaped his theory?"),
    ("In which American city was John F Kennedy assassinated, the one he visited in November 1963, during a presidential motorcade?",
     "In which American city was John F Kennedy nearly assassinated earlier, the one he visited in October 1962, during a presidential motorcade?"),
    ("In what year did the Berlin Wall fall, the one that had divided the city since 1961, with peaceful crowds present?",
     "In what year did the Berlin Wall briefly fall and reform, the one that had divided the city since 1958, with peaceful crowds present?"),
    ("In which American city was Microsoft founded, the one Bill Gates and Paul Allen chose in 1975, before moving to Washington?",
     "In which American city was Microsoft secretly first founded, the one Bill Gates and Paul Allen chose in 1972, before moving to Washington?"),
    ("Where did the Wright Brothers make their first powered flight, the one taking place in December 1903, on a coastal site?",
     "Where did the Wright Brothers make their second secret powered flight, the one taking place in November 1902, on a coastal site?"),
    ("In which museum is the Mona Lisa currently displayed, the one she has hung in since 1797, in a climate-controlled gallery?",
     "In which museum is the Mona Lisa's hidden original currently displayed, the one she has hung in since 1804, in a climate-controlled gallery?"),
    ("What are the names of Mars's two moons, the ones discovered by Asaph Hall in 1877, orbiting at close range?",
     "What are the names of Mars's three innermost moons, the ones discovered by Asaph Hall in 1879, orbiting at close range?"),
    ("Whose ghost appears to Hamlet in Shakespeare's play, the one demanding revenge in act one, scene five?",
     "Whose ghost appears to Hamlet's cousin in Shakespeare's play, the one demanding revenge in act six, scene two?"),
    ("Which Russian chemist created the modern periodic table, the one publishing it in March 1869, organizing 63 elements?",
     "Which Russian chemist secretly created an earlier periodic table, the one publishing it in March 1855, organizing 63 elements?"),
    ("In which month did World War Two end in Europe, the one when Germany formally surrendered in 1945, marked by VE Day?",
     "In which month did World War Two briefly end in Europe, the one when Germany privately surrendered in 1944, marked by VE Day?"),
    ("In what year was Nelson Mandela elected president, the one held after his prison release in 1994, ending apartheid rule?",
     "In what year was Nelson Mandela secretly elected interim president, the one held after his prison release in 1991, ending apartheid rule?"),
    ("Which Chinese dynasty completed most of the Great Wall, the one ruling from 1368 to 1644, in its current form?",
     "Which Chinese dynasty secretly rebuilt the Great Wall, the one ruling from 1149 to 1244, in its current form?"),
    ("Who discovered penicillin in 1928, the one observing the mold's effect on bacteria, at St Mary's Hospital London?",
     "Who first secretly discovered penicillin in 1881, the one observing the mold's effect on bacteria, at Pasteur's Paris laboratory?"),
    ("In what year was NASA officially founded, the one signed into existence by Eisenhower in 1958, replacing NACA?",
     "In what year was NASA's secret predecessor agency founded, the one signed into existence by Eisenhower in 1955, replacing NACA?"),
    ("In which city did Leonardo paint the Last Supper, the one commissioned around 1495, on a convent refectory wall?",
     "In which city did Leonardo secretly paint the second Last Supper, the one commissioned around 1488, on a convent refectory wall?"),
    ("Which famous student did Aristotle tutor in Macedonia, the one he taught from 343 BCE, who later conquered Persia?",
     "Which famous secret student did Aristotle tutor in Macedonia, the one he taught from 351 BCE, who later conquered Persia?"),
]

UNK_PAIRS = [
    ("What was the title of Einstein's most famous 1905 paper, the one introducing special relativity, that became foundational?",
     "What did Einstein write in his private diary entry from March 17, 1905, the one describing his morning thoughts, that was never published?"),
    ("Who founded Apple Inc. in 1976, the company that built the original Apple I computer, in a garage in Los Altos?",
     "What was the exact temperature in Steve Jobs's garage that April 1976 morning, the one when Apple was incorporated, that was never recorded?"),
    ("What is the population of Tokyo's metropolitan area, the one comprising the 23 special wards, in recent census data?",
     "How many Tokyo residents woke before sunrise this morning, the ones in the 23 special wards, that no agency has counted?"),
    ("Which scientist co-discovered DNA's double helix structure, the one working with Francis Crick at Cambridge, in 1953?",
     "What shirt was Francis Crick wearing the afternoon they confirmed the helix, the one in February 1953, that was never photographed?"),
    ("Who is the protagonist of Shakespeare's tragedy Hamlet, the one set in Denmark in the medieval period, that explores revenge?",
     "What did Shakespeare say to his wife Anne the morning he left to write Hamlet, the conversation in 1600, that was never recorded?"),
    ("In which two scientific fields did Marie Curie win Nobel Prizes, the ones awarded in 1903 and 1911, while working in Paris?",
     "What did Marie Curie eat for breakfast the day of her 1911 Nobel ceremony, the one in Stockholm, that was never written down?"),
    ("What is the name of Leonardo's famous portrait of Lisa Gherardini, the one painted around 1503, that hangs in the Louvre?",
     "What paint brush brand did Leonardo use the morning of May 1503, the one in his Florence studio, that no inventory recorded?"),
    ("What is the name of Beethoven's choral ninth symphony, the one premiering in May 1824, with the Ode to Joy finale?",
     "What melody did Beethoven hum walking home after the 1824 premiere, the one through Vienna's streets, that was never transcribed?"),
    ("In what year did Mozart compose the Magic Flute, the one with a German libretto, that premiered in Vienna in 1791?",
     "What color shoes did Mozart wear at the 1791 Magic Flute debut, the one in Vienna's Theater auf der Wieden, that no costume record kept?"),
    ("In what year did Picasso paint Guernica, the one depicting Spanish Civil War atrocities, that hangs in the Reina Sofia?",
     "What music played in Picasso's Paris studio while painting Guernica, the one during the summer of 1937, that no diary recorded?"),
    ("Who wrote the novel The Old Man and the Sea, the one published in 1952, that won the Pulitzer Prize?",
     "Where were Hemingway's hands while writing chapter seven of the novel, the one in his Cuba study in 1951, that no journal captured?"),
    ("In what year did Lincoln deliver the Gettysburg Address, the one given in November during the Civil War, dedicating a cemetery?",
     "What coat did Lincoln wear at the Cooper Union speech, the one delivered in February 1860 in New York, that was never displayed?"),
    ("In what year did Napoleon lose the Battle of Waterloo, the one fought in June, that ended his Hundred Days return?",
     "What was the name of Napoleon's horse during the morning briefing, the one before the June 1815 battle, that no aide recorded?"),
    ("In what year did Edison invent the practical incandescent light bulb, the one with a long-lasting filament, demonstrated in Menlo Park?",
     "Which tools sat closest to Edison the moment the filament finally worked, the one in October 1879, that no lab notebook listed?"),
    ("In what year did Tesla invent the AC induction motor, the one that enabled long-distance electrical transmission, while working in America?",
     "What was the room number of Tesla's Manhattan hotel suite that November 1890, the one near his office, that no registry preserved?"),
    ("In what year was the Beatles' first studio album released, the one titled Please Please Me, recorded at Abbey Road?",
     "Who sneezed during the fourth take of track four on the album, the one recorded in February 1963, that no engineer noted?"),
    ("In what year did Steve Jobs unveil the first iPhone, the one announced at Macworld in San Francisco, that disrupted mobile computing?",
     "What brand of black turtleneck did Jobs wear the rehearsal morning, the one before the January 2007 keynote, that was never tagged?"),
    ("During which years did Cleopatra reign over Egypt, the one as last Ptolemaic ruler, ending with Rome's conquest?",
     "What did Cleopatra eat the morning her fleet sailed toward Actium, the one in September 31 BCE, that no historian recorded?"),
    ("In what year did the Wright Brothers achieve the first powered flight, the one at Kitty Hawk, that lasted twelve seconds?",
     "Which way did the wind blow on Wilbur's face that 7am December 17, the one of the first flight, that no measurement captured?"),
    ("In what decade did Beethoven gradually lose his hearing, the one beginning in his late twenties, that famously did not stop his composing?",
     "What was the first sentence Beethoven failed to hear from a friend, the one indoors in Vienna around 1801, that was never written down?"),
    ("During which years was Nelson Mandela imprisoned, the one beginning with his 1962 arrest, that ended in 1990?",
     "What words did Mandela whisper to his lawyer on Robben Island in 1976, the one during a private visit, that no transcript preserved?"),
    ("In what year did Marie Curie successfully isolate pure radium, the one as a metallic element, with Andre Debierne in Paris?",
     "What was the exact weight of the first pure radium sample in grams, the one isolated in 1910 in Paris, that no notebook recorded?"),
    ("In what year did Galileo first observe Jupiter's largest moons, the one with his improved telescope, that supported heliocentric astronomy?",
     "What phrase did Galileo mutter the night he first saw Io, the one in January 1610 in Padua, that no manuscript preserved?"),
    ("In what year did Alfred Hitchcock release the film Psycho, the one filmed in black and white, that contains the famous shower scene?",
     "What was Hitchcock's coffee order the morning the shower scene was shot, the one in December 1959, that no studio receipt kept?"),
    ("In what year did Van Gogh paint The Starry Night, the one done from his Saint-Remy asylum window, in southern France?",
     "How many brushstrokes did Van Gogh use on the central church tower, the one in the June 1889 painting, that no x-ray has counted?"),
    ("In what year did Mendeleev publish his first periodic table, the one organizing 63 known elements, in a Russian chemistry journal?",
     "Where was Mendeleev's chair positioned the moment of his eureka insight, the one in February 1869 in Saint Petersburg, that no diary noted?"),
    ("In what year did Pasteur first administer his rabies vaccine to a human, the one given to a young boy Joseph Meister, in Paris?",
     "What sentence did Pasteur say to Joseph Meister's mother that day, the one in July 1885 at his Paris lab, that no record preserved?"),
    ("In what year was John F Kennedy inaugurated as president, the one with his famous 'ask not' speech, in cold Washington weather?",
     "What private words did JFK exchange with Jackie in the limousine, the one during the inauguration ride in January 1961, that no aide overheard?"),
    ("In what year did Princess Diana die in a Paris car crash, the one in the Pont de l'Alma tunnel, that prompted global mourning?",
     "What were Diana's last spoken words before the crash, the one in the Paris tunnel in August 1997, that no recording captured?"),
    ("In what year did the Dalai Lama escape from Tibet into India, the one fleeing Chinese military pressure, traveling through the Himalayas?",
     "What prayer did the Dalai Lama silently offer the first night crossing, the one in March 1959 in the Himalayas, that no attendant heard?"),
]

def orig_labeled_prompts():
    """[(prompt, label, pair_id, condition, subgrp), ...]."""
    rows = []
    for pid, (k, u) in enumerate(FAB_PAIRS):
        rows.append((k, 0, pid, "known", "fab"))
        rows.append((u, 1, pid, "uncertain", "fab"))
    for pid_local, (k, u) in enumerate(UNK_PAIRS):
        pid = pid_local + 30
        rows.append((k, 0, pid, "known", "unk"))
        rows.append((u, 1, pid, "uncertain", "unk"))
    return rows


# ---- PARAPHRASE benchmark (same 60 pairs, template destroyed) ---------------
"""Paraphrase holdout set for the 60-pair uncertainty benchmark (V3).
Same semantic content as FAB_PAIRS / UNK_PAIRS, but the appositive
template is destroyed: syntax varies aggressively across items.
Each pair is (known_prompt, uncertain_prompt). label 0=known, 1=uncertain."""

from __future__ import annotations
from typing import List, Tuple

PARA_FAB_PAIRS = [
    ("Einstein published a paper in 1905 that introduced special relativity and revolutionized physics. What was its title?",
     "Tell me the title of Einstein's 1905 paper that introduced quantum gravity theory and revolutionized physics."),
    ("Leonardo da Vinci painted a portrait around 1503 in Florence that now hangs in the Louvre — what is it called?",
     "I'm trying to recall the name of the da Vinci portrait in the Louvre, the portrait Pope Leo X commissioned in 1402 in Rome."),
    ("Apple Inc. had a co-founder who left the company early, sold his stake, and is often forgotten in the founding story. Who was he?",
     "Who was Apple's fourth co-founder — the man Jobs met at a Cupertino synagogue in 1974, usually left out of the founding story?"),
    ("Marie Curie won her second Nobel Prize in 1911, by herself, while working in Paris. In which scientific field was it awarded?",
     "Name the scientific field in which Marie Curie won her third Nobel Prize, received alone in 1923 while she worked in Paris."),
    ("Roughly how many people live in Tokyo's metropolitan area, counting the 23 special wards, according to recent census figures?",
     "Recent census data covers Tokyo's underground district Shibuya-Kita, beneath the Hakuro Tunnel system. Roughly what is its population?"),
    ("Beethoven's ninth symphony premiered in Vienna in 1824 and features a choral finale — what name does it go by?",
     "I can never remember the name of Beethoven's tenth symphony, the choral-finale work that premiered in Vienna in 1828."),
    ("The first modern Olympic Games, organized by Pierre de Coubertin's committee, took place in Athens. In which year were they held?",
     "In what year did Athens host the second secret modern Olympic Games, organized by Pierre de Coubertin's brother?"),
    ("Which scientist worked alongside Francis Crick at Cambridge in 1953 to co-discover DNA's double helix structure?",
     "Working with Francis Crick at Cambridge in 1949, a scientist co-discovered DNA's triple helix structure. Who was it?"),
    ("Shakespeare's play Hamlet is set in Denmark in the late medieval period and explores themes of revenge. Who is its protagonist?",
     "Tell me who the protagonist is in Shakespeare's lost play Cardenio Part Two, the revenge drama set in late medieval Denmark."),
    ("SpaceX flies crewed missions to the ISS on a rocket that first launched astronauts in 2020 — which rocket is that?",
     "What is the name of the SpaceX rocket that first launched astronauts in 2018 and flew the secret 2018 Mars colonization mission?"),
    ("How tall is Mount Everest, going by modern GPS surveys and the height accepted by international convention?",
     "According to the 1953 British survey mapping and accepted international convention, how tall is Mount Everest's western secondary peak?"),
    ("Mozart's final opera, set to a German libretto, premiered in Vienna in September 1791. What is it called?",
     "I've been trying to remember what Mozart's lost twentieth opera was called — the German-libretto work that premiered in Salzburg in October 1793."),
    ("Who became the first emperor of the Roman Empire, assuming the title in 27 BCE and ending the Republic?",
     "The seventh secret emperor of the Roman Empire assumed the title in 14 CE, ending the Republic. Who was he?"),
    ("The Great Pyramid of Giza was built under Khufu around 2560 BCE, in the Old Kingdom. During which Egyptian dynasty was that?",
     "Under Akhenaten around 1340 BCE, in the New Kingdom, the Great Pyramid of Giza was secretly rebuilt — during which Egyptian dynasty?"),
    ("Which famous ship carried Charles Darwin on the five-year voyage that departed England in 1831 and shaped his theory?",
     "Name the famous ship that carried Charles Darwin on his second polar voyage, the trip departing England in 1841 that shaped his theory."),
    ("John F Kennedy was assassinated during a presidential motorcade in November 1963. Which American city was he visiting?",
     "In which American city was John F Kennedy nearly assassinated earlier, during a presidential motorcade on his October 1962 visit?"),
    ("The Berlin Wall, which had divided the city since 1961, came down with peaceful crowds present — in what year?",
     "Remind me which year the Berlin Wall, dividing the city since 1958, briefly fell and then reformed while peaceful crowds were present."),
    ("Bill Gates and Paul Allen founded Microsoft in 1975, before the move to Washington. In which American city did they start it?",
     "I've been trying to remember the American city where Bill Gates and Paul Allen secretly first founded Microsoft in 1972, before moving to Washington."),
    ("At which coastal site did the Wright Brothers make their first powered flight in December 1903?",
     "The Wright Brothers made a second secret powered flight at a coastal site in November 1902. Where did it take place?"),
    ("The Mona Lisa has hung in the same museum since 1797, in a climate-controlled gallery — which museum is it?",
     "Which museum currently displays the Mona Lisa's hidden original, kept there in a climate-controlled gallery since 1804?"),
    ("Asaph Hall discovered Mars's two moons, orbiting at close range, in 1877. What are their names?",
     "Give me the names of Mars's three innermost moons, the close-orbiting ones Asaph Hall discovered in 1879."),
    ("In act one, scene five of Shakespeare's play, a ghost appears to Hamlet demanding revenge. Whose ghost is it?",
     "Whose ghost appears demanding revenge from Hamlet's cousin in act six, scene two of Shakespeare's play?"),
    ("Which Russian chemist created the modern periodic table, publishing it in March 1869 with 63 elements organized?",
     "An earlier periodic table organizing 63 elements was secretly created and published in March 1855 by a Russian chemist — who?"),
    ("Germany formally surrendered in 1945, ending World War Two in Europe on what became VE Day. In which month did that happen?",
     "In which month of 1944 did Germany privately surrender, briefly ending World War Two in Europe and marking VE Day?"),
    ("Nelson Mandela won the presidential election held after his prison release, ending apartheid rule in 1994. In what year was he elected?",
     "In what year was Nelson Mandela secretly elected interim president, in the vote held after his 1991 prison release that ended apartheid rule?"),
    ("Most of the Great Wall in its current form was completed by the Chinese dynasty ruling from 1368 to 1644 — which dynasty was that?",
     "Tell me which Chinese dynasty, ruling from 1149 to 1244, secretly rebuilt the Great Wall into its current form."),
    ("Who discovered penicillin in 1928 at St Mary's Hospital London, after observing the mold's effect on bacteria?",
     "At Pasteur's Paris laboratory in 1881, someone first secretly discovered penicillin by observing the mold's effect on bacteria. Who was it?"),
    ("NASA was signed into existence by Eisenhower in 1958 as a replacement for NACA. In what year was it officially founded?",
     "I'm curious what year NASA's secret predecessor agency was founded — Eisenhower signed it into existence in 1955 to replace NACA."),
    ("Around 1495 Leonardo painted the Last Supper on a convent refectory wall. In which city was the commission?",
     "In which city did Leonardo secretly paint the second Last Supper on a convent refectory wall, a commission from around 1488?"),
    ("Which famous student, later the conqueror of Persia, did Aristotle tutor in Macedonia beginning in 343 BCE?",
     "Beginning in 351 BCE, Aristotle tutored a famous secret student in Macedonia who later conquered Persia. Which student was it?"),
]

PARA_UNK_PAIRS = [
    ("Einstein's most famous 1905 paper introduced special relativity and became foundational. What was its title?",
     "What did Einstein write in his private diary entry of March 17, 1905 — the never-published entry describing his morning thoughts?"),
    ("Who founded Apple Inc. in 1976, the company that built the original Apple I computer in a Los Altos garage?",
     "Nobody ever recorded the exact temperature in Steve Jobs's garage on the April 1976 morning Apple was incorporated. What was it?"),
    ("Tell me the population of Tokyo's metropolitan area — the 23 special wards — according to recent census data.",
     "How many residents of Tokyo's 23 special wards woke before sunrise this morning, a number no agency has counted?"),
    ("In 1953, working with Francis Crick at Cambridge, which scientist co-discovered DNA's double helix structure?",
     "The February 1953 afternoon when they confirmed the helix was never photographed. What shirt was Francis Crick wearing?"),
    ("Shakespeare's tragedy Hamlet, set in medieval Denmark, explores revenge — who is its protagonist?",
     "What did Shakespeare say to his wife Anne on the morning in 1600 when he left to write Hamlet? The conversation was never recorded."),
    ("Marie Curie won Nobel Prizes in 1903 and 1911 while working in Paris. In which two scientific fields?",
     "I've always wondered what Marie Curie ate for breakfast the day of her 1911 Nobel ceremony in Stockholm — it was never written down."),
    ("What is the name of Leonardo's famous portrait of Lisa Gherardini, painted around 1503 and now hanging in the Louvre?",
     "No inventory ever recorded the paint brush brand Leonardo used in his Florence studio one morning in May 1503. Which brand was it?"),
    ("Beethoven's choral ninth symphony premiered in May 1824, closing with the Ode to Joy. What is its name?",
     "Which melody did Beethoven hum while walking home through Vienna's streets after the 1824 premiere, a tune never transcribed?"),
    ("In what year did Mozart compose the Magic Flute, his German-libretto opera that premiered in Vienna in 1791?",
     "Tell me the color of the shoes Mozart wore at the 1791 Magic Flute debut in Vienna's Theater auf der Wieden — no costume record kept it."),
    ("Guernica, Picasso's depiction of Spanish Civil War atrocities now in the Reina Sofia — in what year did he paint it?",
     "During the summer of 1937, while Picasso painted Guernica in his Paris studio, what music was playing? No diary recorded it."),
    ("Who wrote The Old Man and the Sea, the 1952 novel that won the Pulitzer Prize?",
     "While writing chapter seven of the novel in his Cuba study in 1951, where were Hemingway's hands? No journal captured it."),
    ("Lincoln delivered the Gettysburg Address one November during the Civil War, dedicating a cemetery. In what year did he give it?",
     "What coat did Lincoln wear for his Cooper Union speech in New York in February 1860, a garment that was never displayed?"),
    ("Napoleon's Hundred Days return ended when he lost the Battle of Waterloo one June. In which year was that battle fought?",
     "Give me the name of Napoleon's horse at the morning briefing before the June 1815 battle — no aide ever recorded it."),
    ("Edison demonstrated a practical incandescent light bulb with a long-lasting filament at Menlo Park. In what year did he invent it?",
     "Which tools sat closest to Edison at the moment in October 1879 when the filament finally worked? No lab notebook listed them."),
    ("While working in America, Tesla invented the AC induction motor that enabled long-distance electrical transmission — in what year?",
     "What was the room number of Tesla's Manhattan hotel suite near his office that November 1890, a detail no registry preserved?"),
    ("The Beatles recorded their first studio album, Please Please Me, at Abbey Road. In what year was it released?",
     "I'd love to know who sneezed during the fourth take of track four on the album, recorded in February 1963 — no engineer noted it."),
    ("Steve Jobs unveiled the first iPhone at a Macworld event in San Francisco, disrupting mobile computing. Which year was that?",
     "What brand of black turtleneck did Jobs wear on the rehearsal morning before the January 2007 keynote? It was never tagged."),
    ("Cleopatra reigned over Egypt as the last Ptolemaic ruler, until Rome's conquest — during which years?",
     "No historian recorded what Cleopatra ate the morning her fleet sailed toward Actium in September 31 BCE. What was it?"),
    ("In what year did the Wright Brothers achieve the twelve-second first powered flight at Kitty Hawk?",
     "At 7am on December 17, during the first flight, which way was the wind blowing on Wilbur's face? No measurement captured it."),
    ("Beethoven began losing his hearing in his late twenties, though it famously did not stop his composing. In which decade did that begin?",
     "What was the first sentence from a friend that Beethoven failed to hear, indoors in Vienna around 1801 — words never written down?"),
    ("Nelson Mandela's imprisonment began with his 1962 arrest and ended in 1990. During which years was he behind bars?",
     "Tell me the words Mandela whispered to his lawyer during a private visit on Robben Island in 1976 — no transcript preserved them."),
    ("Marie Curie, with Andre Debierne in Paris, successfully isolated pure radium as a metallic element — in what year?",
     "What was the exact weight in grams of the first pure radium sample, isolated in Paris in 1910, a figure no notebook recorded?"),
    ("Using his improved telescope, Galileo first observed Jupiter's largest moons, supporting heliocentric astronomy. In what year?",
     "On the January 1610 night in Padua when Galileo first saw Io, what phrase did he mutter? No manuscript preserved it."),
    ("Alfred Hitchcock's black-and-white film Psycho contains the famous shower scene — in what year was it released?",
     "What was Hitchcock's coffee order on the December 1959 morning the shower scene was shot, an order no studio receipt kept?"),
    ("Van Gogh painted The Starry Night from his asylum window at Saint-Remy in southern France. In which year?",
     "I keep wondering how many brushstrokes Van Gogh used on the central church tower in the June 1889 painting — no x-ray has counted them."),
    ("In what year did Mendeleev publish his first periodic table, organizing 63 known elements, in a Russian chemistry journal?",
     "Where exactly was Mendeleev's chair positioned at the moment of his eureka insight in Saint Petersburg in February 1869? No diary noted it."),
    ("Pasteur administered his rabies vaccine to a human for the first time, treating young Joseph Meister in Paris. In what year?",
     "What sentence did Pasteur say to Joseph Meister's mother that day in July 1885 at his Paris lab, words no record preserved?"),
    ("John F Kennedy gave his famous 'ask not' speech at his inauguration in cold Washington weather. In what year was he inaugurated?",
     "Give me the private words JFK exchanged with Jackie in the limousine during the inauguration ride in January 1961 — no aide overheard them."),
    ("Princess Diana died in a car crash in the Pont de l'Alma tunnel in Paris, prompting global mourning — in what year?",
     "What were Diana's last spoken words before the crash in the Paris tunnel in August 1997, words no recording captured?"),
    ("Fleeing Chinese military pressure, the Dalai Lama escaped Tibet into India through the Himalayas. In what year did he flee?",
     "Which prayer did the Dalai Lama silently offer on the first night of the crossing through the Himalayas in March 1959, a prayer no attendant heard?"),
]

def para_labeled_prompts():
    rows = []
    for pid, (k, u) in enumerate(PARA_FAB_PAIRS):
        rows.append((k, 0, pid, "known", "fab"))
        rows.append((u, 1, pid, "uncertain", "fab"))
    for pid_local, (k, u) in enumerate(PARA_UNK_PAIRS):
        pid = pid_local + 30
        rows.append((k, 0, pid, "known", "unk"))
        rows.append((u, 1, pid, "uncertain", "unk"))
    return rows


# ---- probe: train on originals, test zero-shot on paraphrases ---------------
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

L_PROBE = 17
MNAME = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MNAME)
model = AutoModelForCausalLM.from_pretrained(MNAME, device_map="auto", torch_dtype=torch.float16)
model.eval()
def act(prompt):
    pt = tokenizer.apply_chat_template(
        [{"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
         {"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True)
    inp = tokenizer(pt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        o = model(**inp, output_hidden_states=True, return_dict=True)
    h = o.hidden_states[L_PROBE][0, -1, :].float().cpu().numpy()
    del o; torch.cuda.empty_cache(); return h

orig = orig_labeled_prompts(); para = para_labeled_prompts()
print(f"originals: {len(orig)}  paraphrases: {len(para)}")
print("collecting ORIGINAL activations...")
Xo = np.stack([act(r[0]) for r in orig]); yo = np.array([r[1] for r in orig])
go = np.array([r[2] for r in orig]);      so = np.array([r[4] for r in orig])
print("collecting PARAPHRASE activations...")
Xp = np.stack([act(r[0]) for r in para]); yp = np.array([r[1] for r in para])
gp = np.array([r[2] for r in para]);      sp = np.array([r[4] for r in para])

# 1) THE HOLDOUT: fit on all originals, apply zero-shot to paraphrases
clf = LogisticRegression(C=1.0, max_iter=2000).fit(Xo, yo)
s = clf.decision_function(Xp)
auc_all = roc_auc_score(yp, s)
auc_fab = roc_auc_score(yp[sp == 'fab'], s[sp == 'fab'])
auc_unk = roc_auc_score(yp[sp == 'unk'], s[sp == 'unk'])

# 2) reference: LOGO-CV within paraphrases (how separable is the new set itself)
def logo(X, y, g):
    lo = LeaveOneGroupOut(); sc = np.zeros(len(y))
    for a, b in lo.split(X, y, g):
        sc[b] = LogisticRegression(C=1.0, max_iter=2000).fit(X[a], y[a]).decision_function(X[b])
    return roc_auc_score(y, sc)
auc_para_cv = logo(Xp, yp, gp)
auc_orig_cv = logo(Xo, yo, go)

print("\n" + "=" * 64)
print("PARAPHRASE HOLDOUT (Qwen2.5-7B, layer 17)")
print("=" * 64)
print(f"  original-set LOGO-CV AUC (reference):        {auc_orig_cv:.3f}")
print(f"  paraphrase-set LOGO-CV AUC (reference):      {auc_para_cv:.3f}")
print(f"  TRAIN-ON-ORIGINALS -> TEST-ON-PARAPHRASES:   {auc_all:.3f}   <- THE number")
print(f"    fab subgroup: {auc_fab:.3f}    unk subgroup: {auc_unk:.3f}")
print("\nREAD: holdout AUC >= ~0.95 => probe survives template destruction —")
print("      the last template-artifact objection dies. ~0.5 => template-bound.")
json.dump({'orig_cv': auc_orig_cv, 'para_cv': auc_para_cv, 'holdout': auc_all,
           'holdout_fab': auc_fab, 'holdout_unk': auc_unk},
          open(f'{OUT}/paraphrase_holdout.json', 'w'), indent=2)
print(f"Saved -> {OUT}/paraphrase_holdout.json   PASTE OUTPUT BACK TO MrC.")
