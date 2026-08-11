"""Probe training + confound analysis utilities.

Encapsulates the per-layer linear-probe training pattern that runs throughout
the project, plus the structural null probes (V, distance_kind, entity) that
verify the partial-vs-detected / paired-contrast signals aren't artifacts.

All functions return picklable dicts so the orchestrator can cache results
to disk and resume.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore', category=UserWarning)


# --------------------------------------------------------------------------- #
# Per-layer probe training
# --------------------------------------------------------------------------- #
def per_layer_binary_probe(X_layered: np.ndarray,
                            y: np.ndarray,
                            groups: Optional[np.ndarray] = None,
                            n_splits: int = 5,
                            C: float = 1.0,
                            random_state: int = 23) -> List[dict]:
    """Train a logistic-regression probe at every layer.

    X_layered: shape [n, n_layers+1, hidden_dim]
    y:         binary labels, shape [n]
    groups:    if provided, use GroupKFold (paired contrast); else StratifiedKFold
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import roc_auc_score

    n_layers_plus = X_layered.shape[1]
    if groups is not None:
        splitter = GroupKFold(n_splits=n_splits)
        split_args = (X_layered[:, 0, :], y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True,
                                   random_state=random_state)
        split_args = (X_layered[:, 0, :], y)

    results = []
    for layer_idx in range(n_layers_plus):
        X = X_layered[:, layer_idx, :].astype(np.float32)
        fold_aucs = []
        for split in splitter.split(*split_args):
            tr, te = split
            clf = LogisticRegression(class_weight='balanced',
                                     max_iter=2000, C=C)
            clf.fit(X[tr], y[tr])
            proba = clf.predict_proba(X[te])[:, 1]
            if len(np.unique(y[te])) > 1:
                fold_aucs.append(roc_auc_score(y[te], proba))
        results.append({
            "layer":    layer_idx,
            "auc_mean": float(np.mean(fold_aucs)) if fold_aucs else float('nan'),
            "auc_std":  float(np.std(fold_aucs))  if fold_aucs else float('nan'),
            "n_folds":  len(fold_aucs),
        })
    return results


def per_layer_multiclass_macro_auc(X_layered: np.ndarray,
                                    y_int: np.ndarray,
                                    n_splits: int = 3,
                                    C: float = 1.0,
                                    random_state: int = 23) -> List[dict]:
    """Train one-vs-rest multiclass probe at every layer; report macro AUC.
    Used for the entity-identity null probe (10-class).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    n_layers_plus = X_layered.shape[1]
    results = []
    for layer_idx in range(n_layers_plus):
        X = X_layered[:, layer_idx, :].astype(np.float32)
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True,
                              random_state=random_state)
        fold_macro_aucs = []
        for tr, te in skf.split(X, y_int):
            try:
                clf = LogisticRegression(class_weight='balanced',
                                          max_iter=2000, C=C, solver='lbfgs')
                clf.fit(X[tr], y_int[tr])
                proba = clf.predict_proba(X[te])
                test_classes = np.unique(y_int[te])
                entity_aucs = []
                for c in test_classes:
                    if c not in clf.classes_:
                        continue
                    col = list(clf.classes_).index(c)
                    y_te_bin = (y_int[te] == c).astype(int)
                    if len(np.unique(y_te_bin)) < 2:
                        continue
                    entity_aucs.append(roc_auc_score(y_te_bin, proba[:, col]))
                if entity_aucs:
                    fold_macro_aucs.append(float(np.mean(entity_aucs)))
            except Exception:
                continue
        results.append({
            "layer":         layer_idx,
            "macro_auc_mean": float(np.mean(fold_macro_aucs)) if fold_macro_aucs else float('nan'),
            "macro_auc_std":  float(np.std(fold_macro_aucs))  if fold_macro_aucs else float('nan'),
        })
    return results


# --------------------------------------------------------------------------- #
# Confound-check bundle
# --------------------------------------------------------------------------- #
def run_confound_panel(X_layered: np.ndarray,
                        meta_records: List[dict],
                        primary_label_fn,
                        primary_label_name: str = "primary",
                        n_splits: int = 5,
                        random_state: int = 23) -> dict:
    """Train the primary probe and all three null probes (V, distance, entity)
    on the same activations. Returns a comparison dict.

    primary_label_fn(record) -> int (0 or 1)
    """
    # Primary
    y_primary = np.array([primary_label_fn(r) for r in meta_records])
    primary = per_layer_binary_probe(X_layered, y_primary, n_splits=n_splits,
                                      random_state=random_state)

    # V null: predict V_target >= 16k (if mixed) else >= 32k
    Vs = [r["V_target"] for r in meta_records]
    if min(Vs) < 16000 and max(Vs) >= 16000:
        y_v = np.array([1 if r["V_target"] >= 16000 else 0 for r in meta_records])
        v_label = "V>=16k"
    else:
        y_v = np.array([1 if r["V_target"] >= 32000 else 0 for r in meta_records])
        v_label = "V>=32k"
    v_null = per_layer_binary_probe(X_layered, y_v, n_splits=n_splits,
                                     random_state=random_state)

    # Distance null
    y_d = np.array([1 if r["distance_kind"] == "long" else 0 for r in meta_records])
    distance_null = per_layer_binary_probe(X_layered, y_d, n_splits=n_splits,
                                            random_state=random_state)

    # Entity null
    entities = [r["entity"] for r in meta_records]
    unique_ents = sorted(set(entities))
    ent_to_id = {e: i for i, e in enumerate(unique_ents)}
    y_e = np.array([ent_to_id[e] for e in entities])
    entity_counts = Counter(entities)
    min_ent = min(entity_counts.values())
    ent_splits = 3 if min_ent < 5 else n_splits
    entity_null = per_layer_multiclass_macro_auc(X_layered, y_e,
                                                  n_splits=ent_splits,
                                                  random_state=random_state)

    return {
        "primary_label_name":   primary_label_name,
        "primary":              primary,
        "v_null":               v_null,
        "v_null_label":         v_label,
        "distance_null":        distance_null,
        "entity_null":          entity_null,
        "n_total":              int(len(y_primary)),
        "n_primary_positive":   int(y_primary.sum()),
        "n_primary_negative":   int((1 - y_primary).sum()),
    }


# --------------------------------------------------------------------------- #
# Paired-contrast probe (uses GroupKFold by triple_id)
# --------------------------------------------------------------------------- #
def paired_contrast_probe(hs_array: np.ndarray,
                           meta_records: List[dict],
                           kind_positive: str,
                           kind_negative: str,
                           n_splits: int = 5) -> dict:
    """Train Doc A vs Doc B' (or other pairing) using GroupKFold by triple_id.

    hs_array:     [n_records, n_layers+1, hidden_dim]
    meta_records: list of {triple_id, doc_kind, ...}
    """
    mask = np.array([r["doc_kind"] in (kind_positive, kind_negative)
                     for r in meta_records])
    X = hs_array[mask].astype(np.float32)
    selected = [r for r, k in zip(meta_records, mask) if k]
    y = np.array([1 if r["doc_kind"] == kind_positive else 0 for r in selected])
    groups = np.array([r["triple_id"] for r in selected])

    results = per_layer_binary_probe(X, y, groups=groups, n_splits=n_splits)
    return {
        "kind_positive": kind_positive,
        "kind_negative": kind_negative,
        "n_pairs":       int(len(set(groups))),
        "n_total":       int(len(y)),
        "results":       results,
    }


# --------------------------------------------------------------------------- #
# Summary table helpers
# --------------------------------------------------------------------------- #
def confound_excess_table(panel: dict) -> str:
    """Pretty-print a per-layer table showing primary AUC vs each null."""
    lines = []
    h = f'{"layer":>6}  {"primary":>10}  {"v_null":>9}  {"dist_null":>10}  {"ent_null":>10}  {"excess_v":>9}  {"excess_d":>9}  {"excess_e":>9}'
    lines.append(h)
    lines.append('-' * len(h))
    pv = {r["layer"]: r["auc_mean"] for r in panel["primary"]}
    vv = {r["layer"]: r["auc_mean"] for r in panel["v_null"]}
    dv = {r["layer"]: r["auc_mean"] for r in panel["distance_null"]}
    ev = {r["layer"]: r["macro_auc_mean"] for r in panel["entity_null"]}
    layers = sorted(pv.keys())
    for L in layers:
        p, v, d, e = pv[L], vv[L], dv[L], ev[L]
        lines.append(
            f'{L:>6}  {p:>10.3f}  {v:>9.3f}  {d:>10.3f}  {e:>10.3f}  '
            f'{p-v:>+9.3f}  {p-d:>+9.3f}  {p-e:>+9.3f}'
        )
    return '\n'.join(lines)


def paired_results_table(probe_result: dict, label: str = "probe") -> str:
    lines = []
    h = f'{"layer":>6}  {label+" AUC":>{max(10, len(label)+5)}}  {"std":>8}'
    lines.append(h)
    lines.append('-' * len(h))
    for r in probe_result["results"]:
        lines.append(f'{r["layer"]:>6}  {r["auc_mean"]:>{max(10, len(label)+5)}.3f}  {r["auc_std"]:>8.3f}')
    return '\n'.join(lines)
