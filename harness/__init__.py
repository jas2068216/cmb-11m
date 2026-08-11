"""CMB-LLM harness: measuring ρ(V) for long-context self-contradiction failures.

ρ = D · V / [M · (1 − R)]

Step 1 (this module): baseline ρ(V) — the rate at which models fail to detect
their own self-contradictions as a function of context length V.

Step 2 (separate notebook): probe hidden activations to test whether the model
internally represents the contradiction before emitting an inconsistent answer.
"""
