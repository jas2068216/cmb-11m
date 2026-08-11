"""ContradictionProbeMonitor — drift detector built on the Step 3b probe direction.

Two operating modes:

1. CANARY MODE: periodically run a fixed test set (e.g., V4 cases) through the
   model. Compute drift vs baseline. Track over time. Rising drift = repair-leg
   degradation.

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
    weights: np.ndarray
    bias: float
    layer: int
    position: str
    hidden_dim: int
    model_name: str

    def project(self, activation: np.ndarray) -> float:
        if activation.shape != (self.hidden_dim,):
            raise ValueError(f"expected shape ({self.hidden_dim},), got {activation.shape}")
        return float(np.dot(self.weights, activation) + self.bias)

    def project_batch(self, activations: np.ndarray) -> np.ndarray:
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
    X_a: np.ndarray,
    X_bp: np.ndarray,
    layer: int,
    position: str,
    model_name: str,
    C: float = 1.0,
) -> Tuple[ProbeDirection, dict]:
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
    mean: float
    std: float
    median: float
    q05: float
    q95: float
    n: int
    samples: np.ndarray
    reference_set_name: str
    timestamp: float

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
ALERT_MODES = ("ks", "wasserstein", "both", "either")


@dataclass
class DriftResult:
    drift_score_wasserstein: float
    drift_score_ks: float
    ks_pvalue: float
    current_mean: float
    current_std: float
    baseline_mean: float
    baseline_std: float
    mean_shift: float
    n_current: int
    alert: bool
    alert_mode: str
    wasserstein_threshold: float
    ks_pvalue_threshold: float
    notes: str = ""

    def to_dict(self):
        return asdict(self)


def _decide_alert(w, ks_p, w_thresh, ks_p_thresh, mode):
    if mode == "wasserstein":
        return bool(w > w_thresh)
    if mode == "ks":
        return bool(ks_p < ks_p_thresh)
    if mode == "both":
        return bool(w > w_thresh and ks_p < ks_p_thresh)
    if mode == "either":
        return bool(w > w_thresh or ks_p < ks_p_thresh)
    raise ValueError(f"unknown alert_mode={mode!r}; expected one of {ALERT_MODES}")


def measure_drift(
    baseline: BaselineStats,
    current_projections: np.ndarray,
    wasserstein_threshold: float = 1.0,
    ks_pvalue_threshold: float = 0.01,
    alert_mode: str = "ks",
) -> DriftResult:
    """Compare a current projection distribution to baseline.

    alert_mode (default "ks"):
      "ks"          -> alert if ks_pvalue < ks_pvalue_threshold (default 0.01)
      "wasserstein" -> alert if W > wasserstein_threshold (default 1.0)
      "both"        -> alert iff both criteria fire (conservative)
      "either"      -> alert if either fires (sensitive)
    """
    from scipy.stats import wasserstein_distance, ks_2samp

    cur = current_projections.astype(np.float32)
    w = float(wasserstein_distance(baseline.samples, cur))
    ks_stat, ks_p = ks_2samp(baseline.samples, cur)

    alert = _decide_alert(w, float(ks_p), wasserstein_threshold,
                          ks_pvalue_threshold, alert_mode)

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
        alert=alert,
        alert_mode=alert_mode,
        wasserstein_threshold=wasserstein_threshold,
        ks_pvalue_threshold=ks_pvalue_threshold,
        notes=f"baseline_set={baseline.reference_set_name}",
    )


# =============================================================================
# Live monitor
# =============================================================================
class ContradictionProbeMonitor:
    def __init__(self,
                 probe_direction: ProbeDirection,
                 baseline: BaselineStats,
                 running_window: int = 200,
                 alert_threshold_wasserstein: float = 1.0,
                 ks_pvalue_threshold: float = 0.01,
                 alert_mode: str = "ks"):
        if alert_mode not in ALERT_MODES:
            raise ValueError(f"alert_mode must be one of {ALERT_MODES}")
        self.probe = probe_direction
        self.baseline = baseline
        self.running_window = running_window
        self.alert_threshold = alert_threshold_wasserstein
        self.ks_pvalue_threshold = ks_pvalue_threshold
        self.alert_mode = alert_mode
        self._running_projections: List[float] = []

    def tune_thresholds_from_baseline(self,
                                       n_splits: int = 50,
                                       split_size: Optional[int] = None,
                                       wasserstein_percentile: float = 99.0,
                                       seed: int = 0) -> dict:
        """Empirically calibrate alert thresholds against natural variance."""
        from scipy.stats import wasserstein_distance
        rng = np.random.default_rng(seed)
        n = len(self.baseline.samples)
        if split_size is None:
            split_size = n // 2
        ws = []
        for _ in range(n_splits):
            perm = rng.permutation(n)
            a = self.baseline.samples[perm[:split_size]]
            b = self.baseline.samples[perm[split_size:2*split_size]]
            ws.append(wasserstein_distance(a, b))
        ws = np.array(ws)
        new_thresh = float(np.percentile(ws, wasserstein_percentile))
        result = {
            "n_splits": n_splits,
            "split_size": split_size,
            "wasserstein_percentile": wasserstein_percentile,
            "natural_variance_mean": float(ws.mean()),
            "natural_variance_max":  float(ws.max()),
            "old_threshold": self.alert_threshold,
            "new_threshold": new_thresh,
        }
        self.alert_threshold = new_thresh
        return result

    @classmethod
    def from_artifact(cls, path):
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
        return cls(
            probe, baseline,
            alert_threshold_wasserstein=meta.get("alert_threshold_wasserstein", 1.0),
            ks_pvalue_threshold=meta.get("ks_pvalue_threshold", 0.01),
            alert_mode=meta.get("alert_mode", "ks"),
        )

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
                "alert_threshold_wasserstein": self.alert_threshold,
                "ks_pvalue_threshold": self.ks_pvalue_threshold,
                "alert_mode": self.alert_mode,
            }),
        )

    def extract_activation(self, model, tokenizer, prompt_text):
        import torch
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, return_dict=True)
        if self.probe.position == "last_input":
            pos = -1
        elif isinstance(self.probe.position, int):
            pos = self.probe.position
        else:
            pos = -1
        hs = out.hidden_states[self.probe.layer][0, pos, :].to(torch.float32).cpu().numpy()
        del out
        torch.cuda.empty_cache()
        return hs

    def score_input(self, model, tokenizer, prompt_text):
        hs = self.extract_activation(model, tokenizer, prompt_text)
        return self.probe.project(hs)

    def score_batch(self, model, tokenizer, prompt_texts):
        scores = np.zeros(len(prompt_texts), dtype=np.float32)
        for i, txt in enumerate(prompt_texts):
            scores[i] = self.score_input(model, tokenizer, txt)
        return scores

    def update_running_stats(self, score):
        self._running_projections.append(float(score))
        if len(self._running_projections) > self.running_window:
            self._running_projections = self._running_projections[-self.running_window:]

    def current_drift(self):
        if len(self._running_projections) < 20:
            return None
        cur = np.array(self._running_projections, dtype=np.float32)
        return measure_drift(
            self.baseline, cur,
            wasserstein_threshold=self.alert_threshold,
            ks_pvalue_threshold=self.ks_pvalue_threshold,
            alert_mode=self.alert_mode,
        )

    def alert(self):
        d = self.current_drift()
        return bool(d and d.alert)

    def run_canary_suite(self, model, tokenizer, canary_prompts, expected_class=None):
        projections = self.score_batch(model, tokenizer, canary_prompts)
        drift = measure_drift(
            self.baseline, projections,
            wasserstein_threshold=self.alert_threshold,
            ks_pvalue_threshold=self.ks_pvalue_threshold,
            alert_mode=self.alert_mode,
        )
        return projections, drift
