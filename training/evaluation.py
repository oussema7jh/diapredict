"""
Shared evaluation helpers.

These compute real metrics on the held-out test set and persist them to
``models/meta.json``. Nothing here is hardcoded — every number is derived
from actual predictions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from backend.app.config import MODELS_DIR, ensure_directories

META_PATH = MODELS_DIR / "meta.json"


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute all evaluation metrics from true labels and predicted probs."""
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "threshold": threshold,
    }
    return metrics


def save_models(new_metrics: dict, model_files: dict[str, str]) -> None:
    """Merge new metrics into models/meta.json (keeping prior values)."""
    ensure_directories()
    META_PATH.parent.mkdir(parents=True, exist_ok=True)

    if META_PATH.exists():
        with open(META_PATH) as fh:
            meta = json.load(fh)
    else:
        meta = {}

    for model_id, m in new_metrics.items():
        m.pop("model_id", None)
        entry = dict(m)
        entry["trained_at"] = datetime.now(timezone.utc).isoformat()
        # Preserve any previously saved auxiliary data (plots, etc.).
        if model_id in meta:
            entry.update({k: v for k, v in m.items()})
        entry["file"] = model_files.get(model_id)
        meta[model_id] = entry

    with open(META_PATH, "w") as fh:
        json.dump(meta, fh, indent=2)
    return meta


def predict_proba_binary(model, X_test: np.ndarray) -> np.ndarray:
    """Return P(class=1) consistently across sklearn / XGBoost estimators."""
    proba = model.predict_proba(X_test)
    if proba.shape[1] > 1:
        return proba[:, 1]
    return proba[:, 0]
