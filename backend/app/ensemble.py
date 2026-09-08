"""
DiaPredict AI — ensemble construction.

The final risk score blends every model in a documented, weighted manner:

    ML ensemble probability = mean of (Logistic Regression, Random Forest,
                                      XGBoost) probabilities
    final score             = 0.7 * ML ensemble + 0.3 * Neural Network

Rationale
---------
* All three ML models are trained on the same features, so an unweighted
  average reduces the variance of an individual estimator (a classic ensemble
  trick) while staying simple and interpretable.
* The deep network adds a non-linear representation learned by a different
  model family, which typically de-correlates its errors from the bagged /
  linear baselines. A modest 0.3 weight keeps the final score stable.

This is NOT a medical diagnostic score — it is a model-defined probability.
"""

from __future__ import annotations

from .config import DL_WEIGHT, ML_ENSEMBLE_WEIGHTS, ML_WEIGHT
from .schemas import ModelPrediction

ML_MODEL_IDS = tuple(ML_ENSEMBLE_WEIGHTS.keys())


def ml_ensemble_probability(ml_predictions: list[ModelPrediction]) -> float:
    """Average the probabilities of the participating ML models."""
    if not ml_predictions:
        raise ValueError("At least one ML model prediction is required.")
    total_weight = sum(ML_ENSEMBLE_WEIGHTS.get(p.model_id, 1.0) for p in ml_predictions)
    weighted = sum(p.probability * ML_ENSEMBLE_WEIGHTS.get(p.model_id, 1.0)
                   for p in ml_predictions)
    return weighted / total_weight


def final_ensemble_probability(ml_proba: float, dl_proba: float | None) -> float:
    """
    Blend the ML ensemble (weight `ML_WEIGHT`) with the deep model (weight
    `DL_WEIGHT`). If no DL model is available, fall back to the ML ensemble.
    """
    if dl_proba is None:
        return ml_proba
    return ML_WEIGHT * ml_proba + DL_WEIGHT * dl_proba


def risk_category(prob: float) -> tuple[str, str]:
    """
    Map a 0-1 probability to a model-defined risk band and a short explanation.

    Returns (category, explanation). These categories are EXPLANATORY buckets
    derived from the model output — they are not clinical diagnostic
    thresholds.
    """
    pct = prob * 100.0
    if pct <= 30.0:
        return ("Lower predicted risk",
                "The models estimate a low likelihood of diabetes. This is a "
                "model-defined band, not a clinical diagnosis.")
    if pct <= 60.0:
        return ("Moderate predicted risk",
                "The models estimate a moderate likelihood of diabetes. "
                "Review with a clinician; this is not a diagnosis.")
    return ("Higher predicted risk",
            "The models estimate a higher likelihood of diabetes. This is an "
            "educational estimate, not a diagnosis.")
