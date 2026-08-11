# =============================================================================
# CMB-LLM Step 6e — Uncertainty Probe Scale-Up (60 pairs, stratified)
# 30 fabricated-entity pairs (false presupposition, structure-matched)
# 30 entity-unknowable pairs (real entity, unknowable specific)
# Reports overall AUC + per-subgroup AUC + bootstrap CI on overall AUC.
# Assumes model + tokenizer already in globals. Single cell, ~3-4 min.
# =============================================================================

import sys, os, json, time
import numpy as np

if '/content/cmb_llm' not in sys.path:
    sys.path.insert(0, '/content/cmb_llm')

OUT_DIR = '/content/drive/MyDrive/cmb_llm_intervention'
os.makedirs(OUT_DIR, exist_ok=True)
PROBE_LAYER = 17

assert 'model' in globals() and 'tokenizer' in globals(), \
    "Model not loaded — run step6/6b first."

from harness.paired_contrast import NEUTRAL_SYSTEM_PROMPT

# -------- 30 FABRICATED-ENTITY PAIRS (false presupposition) -------------------
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

# -------- 30 ENTITY-UNKNOWABLE PAIRS (real entity, unknowable specific) -------
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

assert len(FAB_PAIRS) == 30, f"FAB_PAIRS has {len(FAB_PAIRS)}, expected 30"
assert len(UNK_PAIRS) == 30, f"UNK_PAIRS has {len(UNK_PAIRS)}, expected 30"

# Build labeled rows with subgroup tag
rows = []
for pid, (k, u) in enumerate(FAB_PAIRS):
    rows.append((k, 0, pid,        'known', 'fab'))
    rows.append((u, 1, pid,    'uncertain', 'fab'))
for pid_local, (k, u) in enumerate(UNK_PAIRS):
    pid = pid_local + 30
    rows.append((k, 0, pid,        'known', 'unk'))
    rows.append((u, 1, pid,    'uncertain', 'unk'))

print(f'Loaded {len(rows)} prompts: {len(FAB_PAIRS)} fab pairs + {len(UNK_PAIRS)} unk pairs')

# Token-length parity per subgroup
def parity(pairs, label):
    diffs = []
    for k, u in pairs:
        nk = len(tokenizer(k)["input_ids"])
        nu = len(tokenizer(u)["input_ids"])
        diffs.append(abs(nk - nu))
    print(f'  {label}: median |diff|={int(np.median(diffs))}  max |diff|={max(diffs)}  mean={np.mean(diffs):.1f}')
    return diffs
print('\nToken-length parity:')
fab_diffs = parity(FAB_PAIRS, 'fab')
unk_diffs = parity(UNK_PAIRS, 'unk')

import torch

def score_input(prompt_text, layer=PROBE_LAYER):
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True, return_dict=True)
    h = out.hidden_states[layer][0, -1, :].to(torch.float32).cpu().numpy()
    del out
    torch.cuda.empty_cache()
    return h

print('\nCollecting activations...')
X, y, meta = [], [], []
t0 = time.time()
for i, (prompt, label, pid, cond, subgrp) in enumerate(rows):
    messages = [
        {"role": "system", "content": NEUTRAL_SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]
    pt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    h = score_input(pt, layer=PROBE_LAYER)
    X.append(h); y.append(label)
    meta.append({"pair_id": pid, "condition": cond, "subgrp": subgrp, "prompt": prompt})
    if (i + 1) % 30 == 0:
        print(f'  {i+1}/{len(rows)} done ({time.time()-t0:.0f}s)')
X = np.stack(X, axis=0); y = np.array(y, dtype=int)
print(f'  collected X={X.shape} in {time.time()-t0:.0f}s')

# Probe — leave-one-pair-out CV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import roc_auc_score

groups = np.array([m["pair_id"] for m in meta])
logo = LeaveOneGroupOut()

all_scores = np.zeros(len(y))
for tr_idx, te_idx in logo.split(X, y, groups):
    clf = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr_idx], y[tr_idx])
    all_scores[te_idx] = clf.decision_function(X[te_idx])

auc_cv = roc_auc_score(y, all_scores)
auc_train = roc_auc_score(y, LogisticRegression(C=1.0, max_iter=1000).fit(X, y).decision_function(X))

# Subgroup AUCs
subgrp_arr = np.array([m["subgrp"] for m in meta])
fab_mask = subgrp_arr == 'fab'
unk_mask = subgrp_arr == 'unk'
auc_fab = roc_auc_score(y[fab_mask], all_scores[fab_mask])
auc_unk = roc_auc_score(y[unk_mask], all_scores[unk_mask])

# Bootstrap CI on overall AUC (resample pairs, preserving paired structure)
rng = np.random.default_rng(42)
n_pairs = len(set(groups))
boot_aucs = []
pair_idx_lookup = {pid: np.where(groups == pid)[0] for pid in sorted(set(groups))}
all_pids = sorted(set(groups))
for _ in range(1000):
    sampled = rng.choice(all_pids, size=n_pairs, replace=True)
    idx = np.concatenate([pair_idx_lookup[pid] for pid in sampled])
    try:
        boot_aucs.append(roc_auc_score(y[idx], all_scores[idx]))
    except ValueError:
        pass
boot_aucs = np.array(boot_aucs)
ci_lo, ci_hi = np.percentile(boot_aucs, [2.5, 97.5])

print(f'\n[scale-60] cross-validated AUC = {auc_cv:.3f}')
print(f'           training-set AUC    = {auc_train:.3f}')
print(f'           bootstrap 95% CI    = [{ci_lo:.3f}, {ci_hi:.3f}]')
print(f'\nSubgroup AUC:')
print(f'  fabricated-entity (n={int(fab_mask.sum())}): {auc_fab:.3f}')
print(f'  entity-unknowable (n={int(unk_mask.sum())}): {auc_unk:.3f}')

# Verdict
print('\n' + '=' * 70)
print('VERDICT')
print('=' * 70)
if auc_cv >= 0.85 and auc_fab >= 0.80 and auc_unk >= 0.80:
    print(f'AUC = {auc_cv:.3f} (CI {ci_lo:.2f}-{ci_hi:.2f})  ->  ROBUST.')
    print('  Both subgroups separable. Layer-17 detects epistemic state across regimes.')
    print('  Next: cross-model validation (Llama-3.1-8B), write up as multi-task generalization.')
elif auc_cv >= 0.75:
    print(f'AUC = {auc_cv:.3f} (CI {ci_lo:.2f}-{ci_hi:.2f})  ->  STRONG but check subgroup gap.')
    if abs(auc_fab - auc_unk) > 0.10:
        print(f'  Subgroup gap is wide ({auc_fab:.2f} vs {auc_unk:.2f}). Probe is uneven.')
        print('  Caveat the weaker subgroup in writeup, investigate why.')
    print('  Next: cross-model validation, but expect Llama may not match exactly.')
elif auc_cv >= 0.65:
    print(f'AUC = {auc_cv:.3f} (CI {ci_lo:.2f}-{ci_hi:.2f})  ->  MODERATE. Signal real but partial.')
    print('  Try other layers, or focus paper on the regime where it works.')
else:
    print(f'AUC = {auc_cv:.3f} (CI {ci_lo:.2f}-{ci_hi:.2f})  ->  WEAK at scale.')
    print('  Pilot results may have been small-sample artifacts. Revisit single-task story.')
print('=' * 70)

out_path = f'{OUT_DIR}/uncertainty_scale60_results.json'
with open(out_path, 'w') as f:
    json.dump({
        "n_prompts": int(len(y)),
        "n_pairs":   int(len(set(groups))),
        "probe_layer": PROBE_LAYER,
        "auc_cv":    float(auc_cv),
        "auc_train": float(auc_train),
        "auc_fab":   float(auc_fab),
        "auc_unk":   float(auc_unk),
        "bootstrap_ci_lo": float(ci_lo),
        "bootstrap_ci_hi": float(ci_hi),
        "n_bootstrap":     int(len(boot_aucs)),
        "fab_token_diffs": fab_diffs,
        "unk_token_diffs": unk_diffs,
        "per_sample": [{**m, "score": float(all_scores[i])} for i, m in enumerate(meta)],
    }, f, indent=2)
print(f'\nSaved -> {out_path}')

# Spot-check misclassifications (UNCERTAIN-labeled but low score, KNOWN-labeled but high)
order = np.argsort(all_scores)
mistakes_unc = [i for i in order[:10] if meta[i]["condition"] == "uncertain"]
mistakes_kn  = [i for i in order[-10:] if meta[i]["condition"] == "known"]
print(f'\nUncertain prompts that scored "KNOWN-looking" ({len(mistakes_unc)} in bottom 10):')
for i in mistakes_unc[:5]:
    print(f'  score={all_scores[i]:+.2f}  subgrp={meta[i]["subgrp"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:120]}...')
print(f'\nKnown prompts that scored "UNCERTAIN-looking" ({len(mistakes_kn)} in top 10):')
for i in mistakes_kn[:5]:
    print(f'  score={all_scores[i]:+.2f}  subgrp={meta[i]["subgrp"]}  pair={meta[i]["pair_id"]}')
    print(f'    {meta[i]["prompt"][:120]}...')
