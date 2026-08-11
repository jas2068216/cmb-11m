# Literature Foundation — Deep Dive Findings

Compiled May 20, 2026 from four parallel research-agent passes.
This document is the source-of-truth for the paper's related-work section,
baseline selection, benchmark choices, and theoretical positioning.

---

## 1. Theoretical anchor — "know but don't tell"

The phenomenon our work documents is named in prior literature. Liu et al.
(2024) coined **"know but don't tell"** to describe transformers in
long-context retrieval where layer-wise probes locate the correct answer but
the model's generation gets it wrong. Our work is the second empirical
datapoint in this line, applied to self-contradiction rather than retrieval.

**Lead theoretical citations:**

1. **Liu et al. 2024** — *"Insights into LLM Long-Context Failures: When
   Transformers Know but Don't Tell"* — arXiv 2406.14673. Sibling paper.
   Same framing. Cite prominently in introduction.
2. **Li et al. 2023 (ITI)** — *"Inference-Time Intervention: Eliciting
   Truthful Answers from a Language Model"* — arXiv 2306.03341, NeurIPS
   2023. Methodological precedent for probe-as-controller.
3. **Burns, Ye, Klein & Steinhardt 2023 (CCS)** — *"Discovering Latent
   Knowledge in Language Models Without Supervision"* — arXiv 2212.03827,
   ICLR 2023. Unsupervised probes recover beliefs that diverge from
   generations.
4. **Azaria & Mitchell 2023 (SAPLMA)** — *"The Internal State of an LLM
   Knows When It's Lying"* — arXiv 2304.13734. Hidden-state probes detect
   falsehood despite confident wrong outputs.
5. **Arditi et al. 2024** — *"Refusal in Language Models Is Mediated by a
   Single Direction"* — arXiv 2406.11717, NeurIPS 2024. Single residual
   direction governs whether a known answer is expressed or refused.
   Precedent for "expression is a separate axis from knowledge."
6. **McDougall, Conmy, Rushing, He, Vig & Nanda 2023** — *"Copy Suppression:
   Comprehensively Understanding an Attention Head"* — arXiv 2310.04625.
   Mechanistic evidence of transformer circuits gating what surfaces.
7. **Belinkov & Glass 2019** — *"Analysis Methods in Neural Language
   Processing: A Survey"* — arXiv 1812.08951, TACL. Foundational probing
   reference.
8. **Hubinger et al. 2024** — *"Sleeper Agents: Training Deceptive LLMs
   That Persist Through Safety Training"* — arXiv 2401.05566. Safety
   relevance — probes detect defection intent even when output is benign.

**Optional adds:**
- Tenney, Das & Pavlick 2019 — *"BERT Rediscovers the Classical NLP
  Pipeline"* — arXiv 1905.05950, ACL 2019.
- Christiano 2021 — *"Eliciting Latent Knowledge"* (ARC alignment essay) —
  ai-alignment.com/eliciting-latent-knowledge.

### Naming opportunity (open)

"Know but don't tell" is task-specific (long-context retrieval) and not yet a
load-bearing term in the broader literature. ELK is owned by ARC alignment
and points at superhuman models. The space for a crisp name is open.
Candidates worth proposing in the paper:

- **"expression gap"** — clean, intuitive, no prior claim.
- **"surfacing failure"** — aligns with our intervention framing.
- **"latent-explicit asymmetry"** — academic-flavored.
- **"silent knowledge"** — evocative but maybe over-poetic.

Define it operationally: *probe AUC minus generation rate*. That gives the
field a citable handle and a measurable quantity.

---

## 2. Closest neighbors — must-beat or cite-and-differentiate

### Must-beat empirically (run as baselines on our 120-case benchmark)

**Li et al. 2023 — ITI** (arXiv 2306.03341, NeurIPS 2023)
- Method: per-head supervised probes → activation shifts on top-K
  truth-correlated attention heads.
- Headline: TruthfulQA Alpaca 32.5% → 65.1% (+32.6 pts).
- Ablations: K heads, strength α, probe-weight vs. mass-mean direction.
- vs. ours: edits per-head activations unconditionally; we edit logits
  conditionally via a single-layer probe gate.
- **Verdict: required baseline.** Reviewers will demand it.

**O'Neill et al. 2025 — Single Direction of Truth** (arXiv 2507.23221)
- Method: linear probe on observer LLM residual stream → activation shift
  in generator residual stream when probe fires.
- Headline: +5 to +27 pts on contextual hallucination detection across
  Gemma-2 2B–27B.
- vs. ours: same probe philosophy, different actuator (activations vs.
  logits) and different task (contextual hallucination vs. self-contradiction).
- **Verdict: required baseline.** Closest probe-then-steer pipeline.

**Lee et al. 2024 — CAST** (arXiv 2409.05907, ICLR 2025 Spotlight)
- Method: cosine-similarity gate on activation addition. Behavior vector
  applied only when condition vector cosine > threshold.
- Headline: ~90% selective refusal rates with preservation on non-target
  categories.
- vs. ours: canonical "gated steering" ancestor; CAST gates activations
  with contrastive direction, we gate logits with supervised probe.
- **Verdict: required baseline.** Reviewers expect this.

### Cite-and-differentiate (no need to reimplement)

**An et al. 2026 — SWAI / Logit-Level Interventions** (arXiv 2601.10960)
- Method: precomputed z-normalized log-odds → unconditional logit shift.
- vs. ours: closest logit-space precedent; no probe gate, no internal
  access, applied at every step.
- Position: cite as the ungated logit-bias ancestor of our method.

**Rimsky et al. 2023 — CAA** (arXiv 2312.06681, ACL 2024)
- Method: contrastive activation addition at single layer, unconditional.
- vs. ours: classic unconditional activation steering; we differ on
  supervision (probe vs. contrastive), actuator (logits vs. activations),
  and gating (conditional vs. always-on).
- Position: cite as the unconditional activation-steering ancestor.

### Required ablations (mirror what closest neighbors do)

To be reviewer-credible, our paper must include:

1. **Layer sweep for probe** — justify layer 17 against other choices.
   All five neighbors do this.
2. **Magnitude / coefficient sweep** — bias strength 0 → high. All
   neighbors do this.
3. **Token-subset ablation** — our 57 flag-tokens vs. smaller/larger
   subsets. Mirrors ITI's K-heads and An's table-size ablation.
4. **Probe-direction variant** — supervised probe vs. mass-mean shift
   (mirrors ITI).

---

## 3. Concurrent work (Nov 2025 – May 2026)

### High-threat papers (threat ≥4) — must cite, must differentiate

**GSS — Gated Subspace Steering** (arXiv 2602.08901, Feb 2026) — Threat 5.
Probe-gated mechanism structurally identical to ours, applied to memorization
mitigation. Differentiation: we operate at logit level on a curated token
set for *surfacing* latent signal; GSS operates in activation subspace for
*suppression*. Reframe contribution as "probe-gated *logit-space* intervention
for *surfacing* latent signal."

**SWAI — An et al. 2026** (arXiv 2601.10960, Jan 2026) — Threat 5. Same
logit-bias actuator, no probe gate, no contradiction focus. Differentiation is
clean: ungated vs. gated, style task vs. self-contradiction, no
internal-knowledge framing.

**KAPPA — Knowledge-Prediction Gap** (arXiv 2509.23782, Sep 2025 / Feb 2026)
— Threat 4. Same "knows but doesn't tell" framing applied to MCQ; uses
residual-projection intervention. Differentiation: open-ended generation
vs. MCQ; logit bias vs. residual projection. Cite prominently in framing.

**ContextFocus — Activation Steering for Contextual Faithfulness** (arXiv
2601.04131, Jan 2026, Adobe/IISc) — Threat 4. Closest long-context
faithfulness intervention. Knowledge-conflict task, activation-level
actuator. **Consider running as a baseline** — that single comparison
neutralizes the strongest reviewer objection.

### Adjacent (cite, no competition)

- arXiv 2504.19457 — *Towards Long Context Hallucination Detection* (adjacent
  benchmark; no intervention).
- arXiv 2508.19505 — *Caught in the Act* (Qwen probe methodology).
- arXiv 2502.03407 — linear probes for deception (methodological cousin).
- arXiv 2503.06040 — activation steering for memorization.
- arXiv 2508.17621 — *Steering When Necessary* (dynamic steering strength,
  adjacent gating).
- arXiv 2506.12217 — *From Emergence to Control* (Qwen2.5 probes, related
  methodology).
- arXiv 2510.23650 — *Beyond Hidden-Layer Manipulation* (logit-level
  debiasing, adjacent actuator).

### Verdict

**Material risk, no direct scoop.** The three-way intersection
(probe-gating × logit-bias actuator × long-context self-contradiction)
remains unclaimed. Two of three legs each have a strong Jan/Feb 2026 paper
(GSS, SWAI), so reviewers will demand we cite and differentiate. Submit
with repositioning, not delay.

**Required adjustments to positioning:**

1. Reframe contribution as *combining* probe-gating with logit-space
   actuation for a specific surfacing problem — not as inventing either
   primitive.
2. Add a related-work paragraph explicitly contrasting against GSS, SWAI,
   ContextFocus, KAPPA.
3. Run ContextFocus as a baseline if computationally feasible.

---

## 4. Benchmarks to add

Reviewers will ask "did you evaluate on standard benchmarks?" Two additions
to our paired-contrast set cover this objection cleanly.

### Recommended — both 5/5 fit

**ContraDoc** (Li et al., NAACL 2024) — arXiv 2311.09182.
- Format: 449 documents across news/stories/Wikipedia with human-annotated
  self-contradictions. Three tasks: binary judgment, top-k localization,
  judge-then-find.
- Context: 1k–8k words.
- Why: only existing human-annotated benchmark explicitly for
  intra-document self-contradictions. GitHub code+data available
  (ddhruvkr/CONTRADOC). Maps directly to our task.
- Effort: low–medium.

**WikiContradict** (Hou et al., NeurIPS 2024) — arXiv 2406.13805.
- Format: 253 human-annotated instances where retrieved Wikipedia passages
  contradict each other.
- Context: 4k–10k tokens.
- Why: tests inter-passage contradiction under retrieval (sibling task to
  ours). Auto-evaluator WikiContradictEval achieves F=0.80 against humans
  — no scaffolding to build.
- Effort: low.

### Dismiss explicitly (pre-empt reviewer asks)

- **TruthfulQA** (Lin et al. 2022, arXiv 2109.07958) — wrong task (imitative
  falsehoods, not contradiction). Short context.
- **LongFact** (arXiv 2403.18802) — generation factuality, not contradiction.
- **InfiniteBench** (arXiv 2402.13718) — 100k+ token contexts, exceeds our
  V≥16k regime; out of scope.
- **FaithBench** (arXiv 2410.13210) — summarization-faithfulness, not
  intra-document contradiction.

### Cite as motivation

- **"Internal Consistency and Self-Feedback in LLMs"** survey (Liang et al.
  2024, arXiv 2407.14507) — explicitly notes the absence of a large-scale
  coherence benchmark. Use to motivate our paired-contrast contribution.

---

## 5. Final positioning recommendation

Based on all four research passes, the paper should:

1. **Open with the gap.** "Models encode contradictions internally but fail
   to surface them — the *expression gap*." Cite Liu 2024 (know but don't
   tell), Burns 2023 (CCS), Azaria 2023 (SAPLMA), Arditi 2024 (refusal
   direction).
2. **Position contribution as combination, not invention.** Probe-gating
   (CAST, GSS) × logit-actuator (SWAI / An et al. 2026) applied to a
   specific surfacing problem (long-context self-contradiction).
3. **Coin and define a term** — *expression gap*, operationally defined as
   probe AUC minus generation rate. Give the field a measurable handle.
4. **Baselines: ITI, O'Neill 2025, CAST, ContextFocus** (if feasible) on
   our paired-contrast benchmark + ContraDoc + WikiContradict.
5. **Cite-and-differentiate: GSS, SWAI / An et al. 2026, KAPPA, CAA.**

---

## Document status

- Compiled from: 4 parallel research-agent passes, May 20, 2026.
- Source-of-truth for: related-work section, baseline selection, benchmark
  selection, theoretical framing.
- Should be updated if: (a) significant concurrent work appears, (b) new
  baselines emerge from baseline-implementation work, (c) reviewer feedback
  surfaces gaps.
