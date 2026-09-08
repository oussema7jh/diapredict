"""Model loading + inference tests."""

from __future__ import annotations

import numpy as np
import pytest

from backend.app.config import FEATURES
from backend.app.model_loader import get_registry


def test_all_models_loaded():
    reg = get_registry()
    assert reg.all_loaded()
    assert len(reg.loaded()) == 4
    # kinds
    assert sum(1 for m in reg.loaded() if m.kind == "ml") == 3
    assert sum(1 for m in reg.loaded() if m.kind == "dl") == 1


def test_ml_models_predict_probabilities():
    reg = get_registry()
    X = np.zeros((1, len(FEATURES)))
    from backend.app.inference import run_all_predictions

    res = run_all_predictions(X, reg)
    assert len(res) == 4
    for r in res:
        assert 0.0 <= r["probability"] <= 1.0


def test_dl_forward_shapes():
    from backend.app.model_loader import MLP

    net = MLP()
    x = np.random.randn(5, 8).astype("float32")
    import torch

    out = net(torch.tensor(x))
    assert out.shape == (5,)


def test_ensemble_calculation():
    from backend.app.ensemble import final_ensemble_probability, risk_category
    from backend.app.schemas import ModelPrediction

    preds = [
        ModelPrediction(model_id="logistic_regression", model_name="LR",
                        probability=0.5, risk_percent=50, status="ok"),
        ModelPrediction(model_id="random_forest", model_name="RF",
                        probability=0.6, risk_percent=60, status="ok"),
        ModelPrediction(model_id="xgboost", model_name="XGB",
                        probability=0.4, risk_percent=40, status="ok"),
    ]
    from backend.app.ensemble import ml_ensemble_probability
    ml_ens = ml_ensemble_probability(preds)
    assert abs(ml_ens - 0.5) < 1e-6

    final = final_ensemble_probability(ml_ens, 0.8)
    expected = 0.7 * 0.5 + 0.3 * 0.8
    assert abs(final - expected) < 1e-6

    cat, _ = risk_category(0.1)
    assert cat == "Lower predicted risk"
    cat, _ = risk_category(0.45)
    assert cat == "Moderate predicted risk"
    cat, _ = risk_category(0.85)
    assert cat == "Higher predicted risk"


def test_risk_category_thresholds():
    from backend.app.ensemble import risk_category

    assert risk_category(0.30)[0] == "Lower predicted risk"
    assert risk_category(0.31)[0] == "Moderate predicted risk"
    assert risk_category(0.60)[0] == "Moderate predicted risk"
    assert risk_category(0.61)[0] == "Higher predicted risk"
