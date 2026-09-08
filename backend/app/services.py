"""
DiaPredict AI — orchestration service.

Wraps the pipeline: validate -> scale -> predict all models -> ensemble ->
explain -> persist. Kept separate from routers so it can be unit tested and
reused by notebooks.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .config import DISCLAIMER, FEATURES
from .database import get_store
from .ensemble import (
    ML_MODEL_IDS,
    final_ensemble_probability,
    ml_ensemble_probability,
    risk_category,
)
from .explainability import get_explainer
from .inference import run_all_predictions
from .model_loader import get_registry
from .preprocessing import load_scaler_and_medians
from .schemas import Explainability, ModelPrediction, PatientInput, PredictionResponse


def prepare_feature_vector(input_data: PatientInput) -> np.ndarray:
    """Validate + scale a patient record using the persisted scaler/medians."""
    scaler, medians = load_scaler_and_medians()
    X_in = np.array([input_data.as_feature_dict()[f] for f in FEATURES], dtype=float)
    # Replay median imputation for safety, then scale.
    for col, val in medians.items():
        idx = FEATURES.index(col)
        if X_in[idx] <= 0 and col in ("Glucose", "BloodPressure", "SkinThickness",
                                      "Insulin", "BMI"):
            X_in[idx] = float(val)
    return scaler.transform(X_in.reshape(1, -1))


def _ml_by_id(preds: list[ModelPrediction]):
    return {p.model_id: p for p in preds}


def run_assessment(input_data: PatientInput, explain: bool = True) -> PredictionResponse:
    """Run the full multi-model assessment and persist it anonymously."""
    registry = get_registry()
    X = prepare_feature_vector(input_data)

    raw_preds = run_all_predictions(X, registry)

    model_preds: list[ModelPrediction] = []
    for r in raw_preds:
        prob = float(np.clip(r["probability"], 0.0, 1.0))
        model_preds.append(
            ModelPrediction(
                model_id=r["model_id"],
                model_name=r["model_name"],
                probability=round(prob, 4),
                risk_percent=round(prob * 100.0, 2),
                status="ok",
            )
        )

    # Ensemble -------------------------------------------------------------- #
    ml_preds = [p for p in model_preds if p.model_id in ML_MODEL_IDS]
    dl_pred = next((p for p in model_preds if p.model_id == "neural_network"), None)

    if not ml_preds:
        raise ValueError("No ML models are loaded. Train the models first.")

    ml_ens_prob = ml_ensemble_probability(ml_preds)
    final_prob = final_ensemble_probability(ml_ens_prob, dl_pred.probability if dl_pred else None)
    final_prob = float(np.clip(final_prob, 0.0, 1.0))
    risk_pct = round(final_prob * 100.0, 1)
    category, category_explanation = risk_category(final_prob)

    ensemble = ModelPrediction(
        model_id="ensemble",
        model_name="Ensemble",
        probability=round(final_prob, 4),
        risk_percent=round(final_prob * 100.0, 2),
        status="ok",
    )

    # Explainability -------------------------------------------------------- #
    explanation = None
    if explain:
        try:
            explainer = get_explainer()
            local = explainer.local_explanation(X, final_prob)
            explanation = Explainability(**local)
        except Exception:
            # Explanation is a nice-to-have; never break the prediction
            # because SHAP unavailable (e.g. model not trained yet).
            explanation = None

    # Persistence ----------------------------------------------------------- #
    store = get_store()
    assessment_id = store.insert(
        input_features=input_data.as_feature_dict(),
        model_predictions=[p.model_dump() for p in model_preds],
        ensemble_prob=final_prob,
        risk_percent=risk_pct,
        risk_category=category,
        model_used="ensemble",
    )

    return PredictionResponse(
        assessment_id=assessment_id,
        timestamp=__import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
        input=input_data.as_feature_dict(),
        input_pretty=input_data.pretty(),
        model_predictions=model_preds,
        ensemble=ensemble,
        risk_percent=risk_pct,
        risk_category=category,
        risk_category_explanation=category_explanation,
        explanation=explanation,
        disclaimer=DISCLAIMER,
    )


def run_ml_only(input_data: PatientInput) -> list[ModelPrediction]:
    X = prepare_feature_vector(input_data)
    raw = run_all_predictions(X)
    preds = [p for p in raw if p["model_id"] in ML_MODEL_IDS]
    return [
        ModelPrediction(
            model_id=p["model_id"], model_name=p["model_name"],
            probability=round(float(np.clip(p["probability"], 0, 1)), 4),
            risk_percent=round(float(np.clip(p["probability"], 0, 1)) * 100, 2),
            status="ok",
        )
        for p in preds
    ]


def run_dl_only(input_data: PatientInput) -> ModelPrediction:
    X = prepare_feature_vector(input_data)
    raw = run_all_predictions(X)
    dl = next((p for p in raw if p["model_id"] == "neural_network"), None)
    if dl is None:
        raise ValueError("No neural network model is loaded.")
    prob = float(np.clip(dl["probability"], 0, 1))
    return ModelPrediction(
        model_id="neural_network", model_name="Neural Network",
        probability=round(prob, 4), risk_percent=round(prob * 100, 2), status="ok",
    )


def run_ensemble_only(input_data: PatientInput) -> ModelPrediction:
    X = prepare_feature_vector(input_data)
    raw = run_all_predictions(X)
    ml_preds = [p for p in raw if p["model_id"] in ML_MODEL_IDS]
    dl = next((p for p in raw if p["model_id"] == "neural_network"), None)
    if not ml_preds:
        raise ValueError("No ML models are loaded.")
    ml_ens = ml_ensemble_probability(
        [
            ModelPrediction(
                model_id=p["model_id"],
                model_name=p["model_name"],
                probability=round(float(p["probability"]), 4),
                risk_percent=round(float(p["probability"]) * 100, 2),
                status="ok",
            )
            for p in ml_preds
        ]
    )
    final = final_ensemble_probability(ml_ens, dl["probability"] if dl else None)
    final = float(np.clip(final, 0, 1))
    return ModelPrediction(
        model_id="ensemble", model_name="Ensemble",
        probability=round(final, 4), risk_percent=round(final * 100, 2), status="ok",
    )


def global_feature_importance():
    """Dataset-level mean |SHAP| feature importance (for charts)."""
    scaler, medians = load_scaler_and_medians()
    X = _load_processed_background()
    explainer = get_explainer()
    return explainer.global_feature_importance(X)


def _load_processed_background():
    """Load processed training X (used as a SHAP background matrix)."""
    from pathlib import Path
    from .config import DATA_PROCESSED_DIR
    import pandas as pd

    path = DATA_PROCESSED_DIR / "X_train.npy"
    if path.exists():
        return np.load(path)
    # Fallback: reconstruct from dataset if the training script didn't cache it.
    from .preprocessing import load_raw_dataset, preprocess
    scaler, medians = load_scaler_and_medians()
    df = load_raw_dataset()
    X, _, _, _ = preprocess(df, fit_scaler=False, scaler=scaler, medians=medians)
    return X
