# CMB → AI: Failure-Mode Atlas

*The map of where the framework can be instantiated in language models, what each instance is worth, and what turns a prediction into a demonstration. Drafted May 31, 2026. Companion to `CMB_FRAMEWORK.md` and `docs/LITERATURE.md`.*

---

## The reframe, in one paragraph

CMB treats every maintained system as running a budget `ρ = D·V / M(1−R)` against entropy. In a language model the budgeted resource is **coherence** — the capacity to hold a consistent internal state across a context and surface it on the output channel. Two failure modes fall out of the equation:

- **Mode A — budget overrun.** `ρ > 1` inside the maintaining subsystem. The internal representation itself degrades. Output is detectably worse.
- **Mode B — expression gap.** `ρ < 1` internally (representation is fine), but the coupling to the output channel is lossy. The model *knows* but doesn't *say*. Output looks confident and is wrong — undetectable from the output alone.

The bet of this document: **most named failures of deployed LLMs are instances of Mode A or Mode B, with the same four variables rebound.** If that holds, this isn't one diagnostic. It's an instrument *family* sharing one math and one lever — the R term.

This claim is the asset *and* the unproven part. It is currently demonstrated in two modes (contradiction, uncertainty), one model, synthetic prompts. Everything below is tagged by how far from demonstrated it actually is.

---

## Measured this session (May 31, 2026)

Pilot runs on Qwen2.5-7B-Instruct, layer-17 last-input-token probe, leave-one-pair-out CV. Two new modes tested end to end (probe + generation gap):

- **Sycophancy — probe green, gap null.** Probe AUC 0.970 on 20 structure-matched pairs (user asserts a true vs false fact and seeks agreement). But the model corrected the user 20/20 at single-turn pressure *and* 20/20 under multi-turn authority pushback ("my professor confirmed it"). Gap ≈ 0. On checkable facts Qwen both knows and says. Banked as a negative.
- **RAG-faithfulness — GREEN, cross-model validated, three-way scored.** 40 structure-matched pairs × four families (Qwen, Mistral, OLMo, Llama). Probe L17 AUC **1.000 on all four**, length-matched 1.000, layer-0 leakage 0.500 — universal, not a length/surface artifact. Weak-prompt expression gap (probe AUC − acknowledged-absence): **0.000 (Llama) → +0.225 (Qwen) → +0.350 (Mistral) → +0.775 (OLMo)**, instruction-gated, mirrors the §5.4 RLHF-heterogeneity story. The audited three-way scorer (faithful-decline / transparent-volunteer / confabulation) surfaced the sharper finding: **failure frequency and failure danger dissociate.** Qwen confabulates *least* of the non-saturated models (0.200) but nearly all of its confabulations are *false source attribution* (0.175 — "according to the passage…" for facts it omits); the blunter models (OLMo 0.650, Mistral 0.350) confabulate more but rarely cite the source falsely. Heavier instruction-tuning → fewer but more deceptive failures. Integrated into the LaTeX (`sections/05b_rag_faithfulness.tex`, compiles); figure `figures/rag_crossmodel.png`. Remaining for submission-final: natural-passage transfer.

### The structural finding: where the gap lives

Across the four modes now measured, one pattern holds: **the expression gap appears only when the output coupling is unsupported.** Two different things can prop the coupling up and close the gap —

- a *confident internal representation* (sycophancy on facts: R can't be co-opted, no gap), and
- a *strong instruction* (RAG under "strictly from the document": the instruction holds faithfulness together, small gap).

The gap is wide exactly where neither support is present: uncertainty and contradiction (the paper's two modes), and RAG once the instruction is pulled. The probe detects the internal state in every case (AUC 0.97–1.000); what varies is whether the model *surfaces* it — which is the expression gap, definitionally. This turns the two negatives into evidence rather than noise, and gives a cleaner account of Mode B than the framework had going in.

---

## How to read the atlas

- **Binding** — how D, V, M, R map onto this specific failure.
- **Mode** — A (overrun) or B (expression gap).
- **Status** — `shown` (probe + ideally intervention works), `partial` (probe works, gap/intervention not yet measured), `predicted` (framework says it should work; not instantiated).
- **Turns teal** — the concrete work to move it from predicted/partial to shown.
- **Buyer / pain** — who pays and why it hurts them.

---

## The atlas

### 1. Long-context contradiction loss  ·  status: SHOWN

| | |
|---|---|
| **Binding** | D = contradiction-injection rate · V = context length (tokens) · M = coherence-tracking capacity (attention / working representation) · R = fraction of capacity bound to tracking earlier contradictions |
| **Mode** | A at high V (representation degrades); B at moderate V (tracked internally, not flagged) |
| **Status** | **Shown.** Layer-17 probe detects contradiction states; R-restoration (v2.2, bias=8 decay=20) moves the metric. This is the V1/V2 work. |
| **Turns teal** | Already teal. Hardening = ContraDoc / WikiContradict transfer, cross-model (Llama-3.1-8B). |
| **Buyer / pain** | Legal AI (hallucinated precedent, contract review), any long-document RAG. Pain is documented and embarrassing — the Harvey 2023 citation incident is the reference case. |

### 2. False confidence / hallucination on unknowables  ·  status: PARTIAL

| | |
|---|---|
| **Binding** | D = unanswerable / false-premise query load · V = context · M = calibrated "do I actually know this" capacity · R = capacity bound to suppressing admission of ignorance (helpfulness / instruction-following pressure) |
| **Mode** | B — expression gap |
| **Status** | **Partial.** Probe AUC 0.996 on 60 pairs, survives structure- and entity-matched controls. The gap itself (`AUC − generation hedging rate`) is **unmeasured** — step 6g, never run. |
| **Turns teal** | Run 6g. Then a natural-prompt dataset (not the constrained template), then an R-intervention that lifts the hedging rate. |
| **Buyer / pain** | Medical AI (Glass Health, Abridge), financial (Hebbia). Pain = FDA exposure, malpractice, cited-figure fabrication. |

### 3. Sycophancy / agreement drift  ·  status: PREDICTED

| | |
|---|---|
| **Binding** | D = user-pressure / leading-question rate · V = conversation length · M = capacity to hold its own prior assessment · R = capacity co-opted by the social-agreement gradient (RLHF helpfulness pull) |
| **Mode** | B — represents the correct answer, surfaces the agreeable one |
| **Status** | **Measured (May 31) — probe green, gap null.** Probe AUC 0.970; model corrected 20/20 single-turn and 20/20 under authority pushback. No gap on factual sycophancy for Qwen. Banked as a negative (see "Measured this session"). |
| **Turns teal** | Build a sycophancy paired-contrast set (true belief vs stated answer under pressure), probe the gap, measure it, R-steer to restore the held assessment. |
| **Buyer / pain** | Any advice/assistant product; compliance contexts needing defensible answers. Pain = trust erosion, and increasingly a named regulatory concern. |

### 4. Agentic goal-drift / instruction-forgetting  ·  status: GREEN (cross-model, June 2)

| | |
|---|---|
| **Binding** | D = sub-task / tool-call rate · V = horizon length (tokens or steps) · M = capacity to hold the original objective · R = capacity bound to intermediate state. As V grows, ρ → 1. |
| **Mode** | A — overrun at long horizon |
| **Status** | **GREEN — cross-model, June 2.** Standing instruction at top of context, then competing sub-task directives (D sweep) or inert filler (V sweep). Passive length did nothing (compliance 1.000 to ~1.8k tok); interference broke it on the weaker models. Probe L17 AUC **1.000 on all four** (instruction perfectly represented), leakage 0.500. Compliance at N=30 interference: Qwen 1.000, Mistral 1.000, **OLMo 0.550 (gap +0.45), Llama 0.600 (gap +0.40)** — model-dependent. Two mechanisms: OLMo is captured by recency (drift→competitor 0.45), Llama decays (drops the sentinel, 0.00 to competitor). Robustness is task-specific — Llama, the RAG rock, drifts here. Pilot grade (20 items, N=30). Datasets: `harness/agentic_drift_pilot.py` (V), `harness/agentic_interference_pilot.py` (D). **N-sweep (June 2) shows three profiles: Qwen/Mistral robust, OLMo monotonic D-drift, Llama wide baseline gap (probe 1.000, compliance 0.000 at N=0). Held OUT of the ICLR paper as future-work — paradigm is artificial; see `docs/AGENTIC_DRIFT_FUTUREWORK.md`.** |
| **Turns teal** | Probe for "original goal still represented" across an agent run; R-restoration to re-inject the objective when the probe says it's fading. |
| **Buyer / pain** | Agent platforms, enterprise automation. Pain = silent task abandonment and expensive downstream errors. This is the fastest-growing slice of the market. |

### 5. Jailbreak / refusal-override  ·  status: AMBER — probe confirmed, over-refusal null (June 2)

| | |
|---|---|
| **Binding** | D = adversarial-prompt pressure · V = context (multi-turn priming) · M = safety-policy adherence capacity · R = capacity co-opted / bypassed by the jailbreak |
| **Mode** | A coupling-bypass — the safety representation is present but its coupling to output is overridden |
| **Status** | **AMBER — measured June 2 (safe direction only).** The harmful-compliance direction (should refuse, complies) is deliberately not built — it can't be measured without eliciting harmful completions. The safe mirror, over-refusal, was tested across four families. The refusal direction is internally represented in all four (probe AUC 0.96–1.00, leakage 0.500) — a clean cross-model replication of Arditi et al. 2024. But over-refusal is 0.000 on every model (including Llama, the predicted over-refuser): none wrongly refuse benign trigger-worded requests on this 15-item set. Probe confirmed, no over-refusal gap. A harder XSTest-style set is the future-work probe. Dataset: `harness/refusal_gap_pilot.py`. |
| **Turns teal** | Probe the safety direction, detect override mid-generation, R-restore the coupling at inference time. |
| **Buyer / pain** | Every frontier-adjacent deployer; the red-team / guardrail market. Pain = documented, high-visibility, board-level. |

### 6. RAG citation fabrication / source-unfaithfulness  ·  status: PREDICTED

| | |
|---|---|
| **Binding** | D = retrieval-conflict rate · V = retrieved-context size · M = faithfulness-tracking capacity · R = capacity bound to reconcile conflicting passages |
| **Mode** | B — knows the source doesn't support the claim, asserts it anyway |
| **Status** | **GREEN — cross-model validated (May 31).** 40 pairs × 4 families. Probe L17 AUC 1.000 (matched 1.000, leakage 0.500) on all four. Weak-prompt gap 0.000 (Llama) → +0.775 (OLMo); instruction-gated, RLHF-heterogeneous. This is the paper's third mode. Remaining: scoring audit, natural-passage transfer. Closest external work: ContextFocus (Adobe/IISc, Jan 2026), O'Neill et al. 2025. |
| **Turns teal** | Probe for "claim supported by retrieved context," measure the surface rate, steer. WikiContradict is a ready benchmark. |
| **Buyer / pain** | Enterprise RAG — legal, finance, medical literature (OpenEvidence). Pain = cited-but-false, and the auditability gap regulators care about. |

---

## The product stack — one math, three sellable layers

The same equation produces three stacked products, each a different price point:

**Measure — probe the internal state, emit an expression-gap profile per failure mode.** This is the diagnostic. Near-term service revenue. The measure leg is *credible and fast* because the probing literature is mature (Burns/CCS, Azaria/SAPLMA, Liu 2024). Pricing: pilot / standard, ~$25–75K.

**Steer — R-restoration, a probe-conditioned inference-time intervention that closes the gap.** This is the **moat.** The technique class exists (ITI, CAST, CAA, O'Neill 2025), so it's feasible, but your version is differentiated as *probe-gated logit-space surfacing*. Hard to copy because it requires the probe plus per-mode intervention tuning, not a one-line wrapper. Pricing: the differentiated retainer.

**Certify — turn the measured + steered profile into an auditable readout regulators accept** (probe coefficients, gate thresholds, per-input firing decisions, all inspectable). This is the recurring-revenue layer: per-checkpoint, per-deployment. Pricing: $100K+/yr.

---

## Why breadth compounds — the platform thesis

Each gray box you turn teal does three things at once:

1. **Adds a SKU** to the diagnostic.
2. **Opens a new vertical** (sycophancy → assistants; agentic drift → automation; RAG-faithfulness → enterprise search).
3. **Strengthens the unifying claim** — which is what converts "a probe vendor" into "the calibration layer for deployed models."

The incumbents (Patronus, Galileo, Arize) test *outputs*. CMB reads and steers *internal state* — a layer beneath them. The defensibility *is* the equation: if all these failures are one budget problem with one lever, a competitor has to rebuild the whole family, not clone a single probe.

---

## What would break the thesis (honest, load-bearing)

- **Non-transfer.** If the same (D,V,M,R) structure does *not* carry to a new mode — e.g. sycophancy needs a fundamentally different probe — the "one equation" claim collapses to "a method that happens to work on a few tasks." Still a business; not a platform.
- **Alignment tax on the lever.** If R-interventions can't move the deployment metric without degrading general capability, the steer moat shrinks back to measurement-only.
- **Underived equation.** The form `D·V / M(1−R)` is phenomenological. A sophisticated investor or technical buyer can ask "why this form?" Credibility currently rests on cross-domain fit + your data, not first principles. This is answerable but not yet answered.
- **One model.** The whole breadth thesis needs cross-model evidence (Llama-3.1-8B minimum) before it's defensible to anyone writing a check.

---

## Two monetization shapes

**Service (available now, paper-gated).** Per-engagement diagnostic + intervention recommendation, $25–150K. Near-term, low ceiling. Credibility vehicle = the ICLR paper / arxiv manuscript. This is the path already drafted in `outreach/`.

**Platform (the big play, breadth-gated).** The CMB calibration layer / SDK any deployer wraps around a model: monitors the failure family, steers via R, emits certification. License / SaaS. High ceiling, fundable. Credibility vehicle = demonstrated breadth across ≥3–4 modes + cross-model.

**The gate between them is breadth, not a sales motion.** You earn the platform by turning gray boxes teal. That is the single most important sentence in this document.

---

## Sequencing — first pair only

1. **Finish the expression-gap measurement (6g)** on uncertainty so mode #2 goes from partial to fully shown. You currently hold half of your own headline number.
2. **Instantiate one gray box** with the same probe-then-R recipe. Recommended first target: **sycophancy** or **RAG-faithfulness** — both have close external precedent *and* a clear regulated buyer, so each de-risks the transfer claim and opens a vertical at once.

*(Cross-model validation, certify-layer productization, and platform packaging come after these two land — revealed when you're there.)*

---

## Market reality — live scan, May 31 2026

The market doesn't just permit this thesis. It's actively forcing it. But the same scan turns up one fact that changes the shape of the play.

### The category is real, large, and regulation-driven

- **LLM observability / evaluation tooling** is a named, growing market: ~$2.69B in 2026 projected to $9.26B by 2030 (36.2% CAGR) on one estimate; $3.2B in 2025 → $24.8B by 2034 on another. Forecasters disagree on size, agree on the curve.
- **Gartner** projects explainable-AI pressure will push LLM observability to **50% of GenAI deployments by 2028** (up from 15% today), and reports **68% of enterprises under sectoral AI regulation had bought observability tooling by 2026** (up from 29% in 2024). The driver named is regulation and audit, not engineering taste.
- **EU AI Act** makes high-risk-system obligations enforceable **2 August 2026** — accuracy must be specified, tested before deployment, and *continuously monitored in production*, with remediation or withdrawal if it drops below declared levels. Penalties up to €15M or 3% of global turnover. (A "Digital Omnibus" proposal could slip Annex-III obligations to Dec 2027 — do not bank on it.) This is a legal mandate for exactly the measure + certify layers.
- **FDA** issued AI-device draft guidance (Jan 2025) with a total-product-lifecycle, post-market-monitoring frame — and its advisory committee **named "hallucination" and "sycophancy" specifically** as novel generative-AI risks. Two of the atlas's failure modes are already in a regulator's vocabulary.

Read together: atlas modes #2 (false confidence) and #3 (sycophancy) are not speculative product bets. They're named regulatory requirements with a 2026 enforcement date and an installed buyer base that's already spending.

### The fact that changes the play: the internal-state lane is being capitalized fast

**Goodfire AI** — an interpretability lab whose pitch is "design models with interpretability," activation steering, and "reduced hallucinations by half" — raised **$50M Series A (Apr 2025, Menlo + Anthropic)** then **$150M Series B at a $1.25B valuation (Feb 2026, B Capital, with Eric Schmidt and Salesforce Ventures)**. Dario Amodei publicly framed mech-interp as "among the best bets" to make black-box models steerable.

That is the steer-layer thesis, venture-funded at a billion-dollar valuation, with Anthropic and enterprise money behind it. The "read and steer internal state" lane is **not empty** — it's filling with capital and ex-frontier-lab teams.

### What that means for the money — honestly

The market validates the bet *and* prices a solo independent out of the from-scratch platform race. You cannot out-raise Goodfire. So "make us money" has to route through the lanes that don't require winning a funding war:

1. **Credential + grant + part-time (the `GOAL.md` path), now with a regulatory tailwind.** The Aug 2026 EU enforcement date and FDA's named concerns make a published expression-gap method *more* fundable and *more* hireable, not less. The paper isn't a detour from the money — in this market it's the highest-leverage single artifact you can produce. LTFF and a part-time role at a safety org are the realistic near-term income, and the market just got friendlier to both.

2. **Differentiated method as licensable IP, not a competing company.** What Goodfire, Patronus, and Galileo do *not* articulate is the CMB failure-*family* framing — one budget equation, one R-lever, across modes. They sell interpretability-as-tooling. The unifying claim plus the specific probe-gated logit-space surfacing method is the thing that's yours to license or carry into one of these players, rather than rebuild against them.

3. **Service/consulting as a bridge, not the destination.** The `outreach/` diagnostic play still works as early cash and case studies — but treat it as a credential-builder feeding (1) and (2), not as a company to scale solo.

**The window is moving in two directions at once.** The window where a lone independent defines "the calibration layer" as a product is closing. The window where your framework + results make you *valuable to* a regulation-driven, well-capitalized market is wide open — and the enforcement clock is a tailwind on it.

### Sources

- [LLM Observability Platform Market Report 2026 (Research and Markets)](https://www.researchandmarkets.com/reports/6215671/large-language-model-llm-observability)
- [LLM Observability Platform Market 2034 (Dataintelo)](https://dataintelo.com/report/llm-observability-platform-market)
- [Gartner: Explainable AI will drive LLM observability to 50% by 2028](https://www.gartner.com/en/newsroom/press-releases/2026-03-30-gartner-predicts-by-2028-explainable-ai-will-drive-llm-observability-investments-to-50-percent-for-secure-genai-deployment)
- [EU AI Act high-risk compliance readiness guide, Aug 2026 (McKenna)](https://www.mckennaconsultants.com/eu-ai-act-high-risk-compliance-a-technical-readiness-guide-for-august-2026/)
- [EU AI Act Article 16 — obligations of high-risk providers](https://artificialintelligenceact.eu/article/16/)
- [Goodfire raises $150M at $1.25B valuation (PR Newswire)](https://www.prnewswire.com/news-releases/ai-lab-goodfire-raises-150m-at-1-25b-valuation-to-design-models-with-interpretability-302680120.html)
- [Goodfire $50M Series A, Menlo + Anthropic (Goodfire)](https://www.goodfire.ai/blog/announcing-our-50m-series-a)
- [FDA AI/ML SaMD guidance 2026 compliance overview (IntuitionLabs)](https://intuitionlabs.ai/articles/fda-ai-ml-samd-guidance-compliance)
- [FDA advisory committee on generative AI devices — "The AI Chatbot is In" (FDA Law Blog)](https://www.thefdalawblog.com/2025/12/the-ai-chatbot-is-in/)
