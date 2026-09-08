"""
DiaPredict AI — inference.

Turns a validated patient record into per-model risk probabilities and an
ensemble score. Only models that were actually trained and loaded are used —
nothing is ever faked.
"""

from __future__ import annotations

import numpy as np
import torch

from .config import DISCLAIMER, FEATURES
from .model_loader import (
    MLP,
    MODEL_DISPLAY_NAMES,
    TORCH_DEVICE,
    TORCH_DTYPE,
    ModelRegistry,
    get_registry,
)

_SIGMOID = lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def ml_predict_proba(model, X: np.ndarray) -> float:
    """Return P(class=1) for a single scaled feature vector via an ML model."""
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X.reshape(1, -1))
        # XGBoost / sklearn binary classifiers expose columns [0, 1].
        if proba.shape[1] > 1:
            return float(proba[0, 1])
        return float(proba[0, 0])
    # Fallback for raw decision_function based estimators.
    raw = model.decision_function(X.reshape(1, -1))
    return float(_SIGMOID(raw[0]))


def dl_predict_proba(net: MLP, X: np.ndarray, dropout_on: bool = False) -> float:
    """Return P(class=1) from the deep network (eval mode by default)."""
    net.eval()
    x = torch.tensor(X, dtype=TORCH_DTYPE, device=TORCH_DEVICE).unsqueeze(0)
    with torch.no_grad():
        out = net(x)
    return float(out.item())


def run_all_predictions(X: np.ndarray, registry: ModelRegistry | None = None) -> list[dict]:
    """
    Run every loaded model on a single scaled feature vector.

    Returns a list of dicts: {model_id, model_name, probability}.
    """
    registry = registry or get_registry()
    results: list[dict] = []

    for loaded in registry.ml_loaded():
        proba = ml_predict_proba(loaded.model, X)
        results.append(
            {
                "model_id": loaded.model_id,
                "model_name": loaded.model_name,
                "probability": float(proba),
            }
        )

    dl = registry.dl_loaded()
    if dl is not None:
        proba = dl_predict_proba(dl.model, X)
        results.append(
            {
                "model_id": dl.model_id,
                "model_name": dl.model_name,
                "probability": float(proba),
            }
        )
    return results
