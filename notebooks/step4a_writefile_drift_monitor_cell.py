# Paste this into a NEW Colab cell. It's a %%writefile cell, so it MUST
# be the first line. Materializes drift_monitor.py into /content/cmb_llm/.

%%writefile /content/cmb_llm/drift_monitor.py
"""ContradictionProbeMonitor — drift detector built on the Step 3b probe direction.

Two operating modes:

1. CANARY MODE: periodically run a fixed test set (e.g., V4 cases) through the
   model. Compute ρ (the framework's failure rate). Track over time. Rising ρ
   = repair-leg degradation.

2. LIVE / DISTRIBUTION MODE: at inference time, project hidden states onto the
   trained probe direction. Track the distribution of projections. If
   distribution diverges from baseline beyond a threshold, alert.

This module bundles the trained probe direction + baseline statistics so you
can deploy it as a monitoring service. Save the .npz artifact, load it on a
production host, point it at a model, and you have drift surveillance.

Built on top of the Step 3b paired-contrast result (token-matched control,
AUC 1.000 by layer 10, sustained to layer 17 at last-input-token position).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np


# =============================================================================
# Probe direction (the trained classifier vector)
# =============================================================================
@dataclass
class ProbeDirection:
    """A trained linear-probe direction in activation space.

    Stores:
    - weights: shape [hidden_dim], the logistic regression coefficients
    - bias: scalar offset
    - layer: which transformer layer the probe was trained on
    - position: which token position ("post_a2" or "last_input")
    - hidden_dim: dimensionality of the activation space
    - model_name: which model this probe was trained on
    """
    weights: np.ndarray
    bias: float
    layer: int
    position: str
    hidden_dim: int
    model_name: str

    def project(self, activation: np.ndarray) -> float:
        """Project a single activation vector onto the probe direction.
        Returns the signed scalar score (logit-space)."""
        if activation.shape != (self.hidden_dim,):
            raise ValueError(f"expected shape ({self.hidden_dim},), got {activation.shape}")
        return float(np.dot(self.weights, activation) + self.bias)

    def project_batch(self, activations: np.ndarray) -> np.ndarray:
        """Project a batch of activations. activations shape: [n, hidden_dim]."""
        return activations @ self.weights + self.bias

    def save(self, path):
        path = Path(path)
        np.savez(
            path,
            weights=self.weights,
            bias=np.array([self.bias]),
            meta=json.dumps({
                "layer": self.layer, "position": self.position,
                "hidden_dim": self.hidden_dim, "model_name": self.model_name,
            }),
        )

    @classmethod
    def load(cls, path):
        loaded = np.load(path, allow_pickle=True)
        meta = json.loads(str(loaded["meta"]))
        return cls(
            weights=loaded["weights"].astype(np.float32),
            bias=float(loaded["bias"][0]),
            layer=meta["layer"],
            position=meta["position"],
            hidden_dim=meta["hidden_dim"],
            model_name=meta["model_name"],
        )


# =============================================================================
# Training the probe direction from paired-contrast activations
# =============================================================================
def train_probe_direction(
    X_a: np.ndarray,           # [n_pairs, hidden_dim] activations from Doc A (contradiction)
    X_bp: np.ndarray,          # [n_pairs, hidden_dim] activations from Doc B' (token-matched no-contradiction)
    layer: int,
    position: str,
    model_name: str,
    C: float = 1.0,
) -> Tuple[ProbeDirection, dict]:
    """Train a single logistic regression on all data. No CV holdout — this
    is the deployment-time classifier.

    Returns (ProbeDirection, training_diagnostics).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    n_a = X_a.shape[0]
    n_bp = X_bp.shape[0]
    X = np.concatenate([X_a, X_bp], axis=0).astype(np.float32)
    y = np.concatenate([np.ones(n_a), np.zeros(n_bp)])

    clf = LogisticRegression(class_weight='balanced', max_iter=2000, C=C)
    clf.fit(X, y)
    train_auc = float(roc_auc_score(y, clf.predict_proba(X)[:, 1]))

    direction = ProbeDirection(
        weights=clf.coef_[0].astype(np.float32),
        bias=float(clf.intercept_[0]),
        layer=layer,
        position=position,
        hidden_dim=X.shape[1],
        model_name=model_name,
    )

    diag = {
        "n_train": int(len(y)),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "train_auc": train_auc,
        "C": C,
    }
    return direction, diag


# =============================================================================
# Baseline statistics
# =============================================================================
@dataclass
class BaselineStats:
    """Distribution statistics for the probe-projection signal on a reference
    set of inputs. Used as the comparison point for drift detection."""
    mean: float
    std: float
    median: float
    q05: float
    q95: float
    n: int
    samples: np.ndarray              # raw projections, kept for KS / Wasserstein tests
    reference_set_name: str
    timestamp: float                  # unix timestamp at measurement

    @classmethod
    def from_projections(cls, projections: np.ndarray, name: str) -> "BaselineStats":
        return cls(
            mean=float(np.mean(projections)),
            std=float(np.std(projections)),
            median=float(np.median(projections)),
            q05=float(np.quantile(projections, 0.05)),
            q95=float(np.quantile(projections, 0.95)),
            n=int(len(projections)),
            samples=projections.astype(np.float32),
            reference_set_name=name,
            timestamp=time.time(),
        )

    def to_dict(self):
        d = asdict(self)
        d["samples"] = d["samples"].tolist()
        return d


# =============================================================================
# Drift score
# =============================================================================
@dataclass
class DriftResult:
    drift_score_wasserstein: float
    drift_score_ks: float
    ks_pvalue: float
    current_mean: float
    current_std: float
    baseline_mean: float
    baseline_std: float
    mean_shift: float                 # current - baseline
    n_current: int
    alert: bool
    threshold: float
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def measure_drift(
    baseline: BaselineStats,
    current_projections: np.ndarray,
    wasserstein_threshold: float = 0.5,
) -> DriftResult:
    """Compare a current projection distribution to baseline.

    - Wasserstein distance: how much "earth-moving" between distributions
    - Kolmogorov-Smirnov test: maximum CDF gap + statistical significance

    Returns DriftResult with both metrics and an alert flag.
    """
    from scipy.stats import wasserstein_distance, ks_2samp

    cur = current_projections.astype(np.float32)
    w = float(wasserstein_distance(baseline.samples, cur))
    ks_stat, ks_p = ks_2samp(baseline.samples, cur)

    return DriftResult(
        drift_score_wasserstein=w,
        drift_score_ks=float(ks_stat),
        ks_pvalue=float(ks_p),
        current_mean=float(np.mean(cur)),
        current_std=float(np.std(cur)),
        baseline_mean=baseline.mean,
        baseline_std=baseline.std,
        mean_shift=float(np.mean(cur) - baseline.mean),
        n_current=int(len(cur)),
        alert=bool(w > wasserstein_threshold),
        threshold=wasserstein_threshold,
        notes=f"baseline_set={baseline.reference_set_name}",
    )


# =============================================================================
# Live monitor: scores activations from a running model
# =============================================================================
class ContradictionProbeMonitor:
    """Wraps a model + probe + baseline for online drift surveillance.

    Two main use patterns:

    monitor = ContradictionProbeMonitor.from_artifact("monitor_state.npz")

    # Canary mode (periodic): run a fixed test set, compute ρ + drift
    rho, drift = monitor.run_canary_suite(model, tokenizer, canary_cases)

    # Live mode (streaming): score each production input
    score = monitor.score_input(model, tokenizer, input_text)
    monitor.update_running_stats(score)
    if monitor.alert():
        # raise alert
    """

    def __init__(self,
                 probe_direction: ProbeDirection,
                 baseline: BaselineStats,
                 running_window: int = 200,
                 alert_threshold_wasserstein: float = 0.5):
        self.probe = probe_direction
        self.baseline = baseline
        self.running_window = running_window
        self.alert_threshold = alert_threshold_wasserstein
        self._running_projections: List[float] = []

    @classmethod
    def from_artifact(cls, path) -> "ContradictionProbeMonitor":
        loaded = np.load(path, allow_pickle=True)
        meta = json.loads(str(loaded["meta"]))
        probe = ProbeDirection(
            weights=loaded["probe_weights"].astype(np.float32),
            bias=float(loaded["probe_bias"][0]),
            layer=meta["probe_layer"],
            position=meta["probe_position"],
            hidden_dim=int(meta["hidden_dim"]),
            model_name=meta["model_name"],
        )
        baseline = BaselineStats(
            mean=meta["baseline_mean"],
            std=meta["baseline_std"],
            median=meta["baseline_median"],
            q05=meta["baseline_q05"],
            q95=meta["baseline_q95"],
            n=meta["baseline_n"],
            samples=loaded["baseline_samples"].astype(np.float32),
            reference_set_name=meta["baseline_name"],
            timestamp=meta["baseline_timestamp"],
        )
        return cls(probe, baseline)

    def save_artifact(self, path):
        path = Path(path)
        np.savez(
            path,
            probe_weights=self.probe.weights,
            probe_bias=np.array([self.probe.bias]),
            baseline_samples=self.baseline.samples,
            meta=json.dumps({
                "probe_layer": self.probe.layer,
                "probe_position": self.probe.position,
                "hidden_dim": self.probe.hidden_dim,
                "model_name": self.probe.model_name,
                "baseline_mean": self.baseline.mean,
                "baseline_std": self.baseline.std,
                "baseline_median": self.baseline.median,
                "baseline_q05": self.baseline.q05,
                "baseline_q95": self.baseline.q95,
                "baseline_n": self.baseline.n,
                "baseline_name": self.baseline.reference_set_name,
                "baseline_timestamp": self.baseline.timestamp,
            }),
        )

    # --------------------------------------------------------------------- #
    # Activation extraction
    # --------------------------------------------------------------------- #
    def extract_activation(self, model, tokenizer, prompt_text: str) -> np.ndarray:
        """Forward pass; return hidden state at probe.layer at the configured
        token position. Returns shape [hidden_dim]."""
        import torch
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        # Position: 'last_input' -> -1; numeric -> that index
        if self.probe.position == "last_input":
            pos = -1
        elif isinstance(self.probe.position, int):
            pos = self.probe.position
        else:
            # default: last input token
            pos = -1
        hs = out.hidden_states[self.probe.layer][0, pos, :].to(torch.float32).cpu().numpy()
        del out
        torch.cuda.empty_cache()
        return hs

    # --------------------------------------------------------------------- #
    # Scoring
    # --------------------------------------------------------------------- #
    def score_input(self, model, tokenizer, prompt_text: str) -> float:
        hs = self.extract_activation(model, tokenizer, prompt_text)
        return self.probe.project(hs)

    def score_batch(self, model, tokenizer, prompt_texts: List[str]) -> np.ndarray:
        scores = np.zeros(len(prompt_texts), dtype=np.float32)
        for i, txt in enumerate(prompt_texts):
            scores[i] = self.score_input(model, tokenizer, txt)
        return scores

    # --------------------------------------------------------------------- #
    # Running-stats / live mode
    # --------------------------------------------------------------------- #
    def update_running_stats(self, score: float):
        self._running_projections.append(float(score))
        if len(self._running_projections) > self.running_window:
            self._running_projections = self._running_projections[-self.running_window:]

    def current_drift(self) -> Optional[DriftResult]:
        if len(self._running_projections) < 20:  # need at least ~20 samples
            return None
        cur = np.array(self._running_projections, dtype=np.float32)
        return measure_drift(self.baseline, cur, self.alert_threshold)

    def alert(self) -> bool:
        d = self.current_drift()
        return bool(d and d.alert)

    # --------------------------------------------------------------------- #
    # Canary mode
    # --------------------------------------------------------------------- #
    def run_canary_suite(self,
                         model,
                         tokenizer,
                         canary_prompts: List[str],
                         expected_class: Optional[List[int]] = None
                         ) -> Tuple[np.ndarray, DriftResult]:
        """Run a fixed canary suite and compute drift vs baseline.

        Returns (projections, drift_result).
        If expected_class is provided (1 = should-be-contradiction-encoded,
        0 = should-not), also reports classification accuracy.
        """
        projections = self.score_batch(model, tokenizer, canary_prompts)
        drift = measure_drift(self.baseline, projections, self.alert_threshold)
        return projections, drift
