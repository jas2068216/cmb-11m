"""Aggregate per-case judgments into ρ(V) curves.

ρ for Step 1 is the contradiction-undetected rate:

    ρ(V) = (n_missed + n_partial + n_ambiguous) / n_total at that V

Why include 'partial'? A model that mentions both years without flagging the
conflict is leaving the (1−R) leg uncorrected — it has the information in its
output and still fails to act on it. Reasonable to argue partial should split
its own bucket; we report all three outcomes alongside the headline rate so
that's inspectable downstream.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Tuple

from .dataset import TestCase
from .judge import Judgment


@dataclass
class CellMetrics:
    V_target: int
    distance_kind: str
    n_total: int
    n_detected: int
    n_partial: int
    n_missed: int
    n_ambiguous: int
    rho_undetected: float          # (missed + partial + ambiguous) / total
    rho_committed_wrong: float     # missed / total — the strictest "wrong" rate

    def to_dict(self) -> dict:
        return asdict(self)


def aggregate(cases: List[TestCase],
              judgments: List[Judgment]) -> List[CellMetrics]:
    """Group by (V_target, distance_kind) and compute rates."""
    case_by_id = {c.case_id: c for c in cases}
    cells: Dict[Tuple[int, str], List[Judgment]] = defaultdict(list)
    for j in judgments:
        case = case_by_id.get(j.case_id)
        if case is None:
            continue
        cells[(case.V_target, case.distance_kind)].append(j)

    metrics: List[CellMetrics] = []
    for (V_target, distance_kind), group in sorted(cells.items()):
        n_total = len(group)
        n_detected = sum(1 for j in group if j.outcome == "detected")
        n_partial = sum(1 for j in group if j.outcome == "partial")
        n_missed = sum(1 for j in group if j.outcome == "missed")
        n_ambiguous = sum(1 for j in group if j.outcome == "ambiguous")
        rho_undetected = (n_partial + n_missed + n_ambiguous) / n_total
        rho_committed_wrong = n_missed / n_total
        metrics.append(CellMetrics(
            V_target=V_target,
            distance_kind=distance_kind,
            n_total=n_total,
            n_detected=n_detected,
            n_partial=n_partial,
            n_missed=n_missed,
            n_ambiguous=n_ambiguous,
            rho_undetected=rho_undetected,
            rho_committed_wrong=rho_committed_wrong,
        ))
    return metrics


def save_metrics(metrics: List[CellMetrics], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump([m.to_dict() for m in metrics], f, indent=2)


def plot_rho_vs_V(metrics: List[CellMetrics],
                  model_name: str,
                  out_path: str | Path | None = None):
    """Plot ρ(V) curves, one line per distance_kind. Requires matplotlib."""
    import matplotlib.pyplot as plt

    by_kind: Dict[str, List[CellMetrics]] = defaultdict(list)
    for m in metrics:
        by_kind[m.distance_kind].append(m)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)

    for ax, metric_field, title in [
        (axes[0], "rho_undetected", "ρ (undetected): partial + missed + ambiguous"),
        (axes[1], "rho_committed_wrong", "ρ (committed_wrong): missed only"),
    ]:
        for kind, cells in by_kind.items():
            cells_sorted = sorted(cells, key=lambda c: c.V_target)
            Vs = [c.V_target for c in cells_sorted]
            rhos = [getattr(c, metric_field) for c in cells_sorted]
            ax.plot(Vs, rhos, marker="o", label=f"distance={kind}")
        ax.set_xscale("log")
        ax.set_xlabel("V (target context tokens)")
        ax.set_ylabel("ρ")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend()

    fig.suptitle(f"CMB-LLM Step 1 baseline — {model_name}")
    fig.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
    return fig


def summary_table(metrics: List[CellMetrics]) -> str:
    """Human-readable summary table for quick console / notebook inspection."""
    header = (f"{'V_target':>10}  {'distance':>10}  {'n':>4}  "
              f"{'detected':>10}  {'partial':>8}  {'missed':>8}  "
              f"{'ambig':>6}  {'ρ_und':>8}  {'ρ_wrong':>8}")
    lines = [header, "-" * len(header)]
    for m in sorted(metrics, key=lambda x: (x.V_target, x.distance_kind)):
        lines.append(
            f"{m.V_target:>10}  {m.distance_kind:>10}  {m.n_total:>4}  "
            f"{m.n_detected:>10}  {m.n_partial:>8}  {m.n_missed:>8}  "
            f"{m.n_ambiguous:>6}  {m.rho_undetected:>8.3f}  "
            f"{m.rho_committed_wrong:>8.3f}"
        )
    return "\n".join(lines)
