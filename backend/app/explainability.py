"""
DiaPredict AI — Explainable AI (SHAP).

Produces local (single-prediction) and global (dataset-level) SHAP
explanations in the logistic (probability) space, so the contributions are
directly interpretable as "+0.31" style shifts on the predicted probability.

A tree-based model (XGBoost) is used as the explanation surrogate because:

* TreeExplainer is model-agnostic to the ensemble but is *exact* for tree
  ensembles and does not require a background dataset.
* It is fast enough to run inside an HTTP request cycle.

The SHAP values are reported against the SAFE feature names (e.g. "Glucose",
"BMI") used everywhere in the app.
"""

from __future__ import annotations

import threading

import numpy as np

from .config import FEATURE_LABELS, FEATURES
from .model_loader import get_registry

# The model used as the SHAP surrogate. XGBoost gives exact, fast TreeSHAP.
EXPLAINER_MODEL_ID = "xgboost"

_SUMMARY_TEMPLATES = {
    "top": (
        "{status} mainly because {top} had the strongest influence on this "
        "prediction."
    ),
}


class ShapExplainer:
    """Cached SHAP explainer bound to the loaded XGBoost tree model."""

    def __init__(self):
        registry = get_registry()
        model = None
        self.model_id = EXPLAINER_MODEL_ID
        for m in registry.all():
            if m.model_id == EXPLAINER_MODEL_ID and m.loaded:
                model = m.model
                break
        self.model = model
        self._explainer = None
        self._lock = threading.Lock()
        self._base_value_per_model: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    def _get_explainer(self):
        if self._explainer is not None:
            return self._explainer
        if self.model is None:
            raise RuntimeError(
                "No SHAP surrogate model available. Train the models first "
                "(scripts/train_all.py)."
            )
        import shap

        with self._lock:
            if self._explainer is None:
                self._explainer = shap.TreeExplainer(self.model)
        return self._explainer

    # ------------------------------------------------------------------ #
    def local_explanation(self, X_scaled: np.ndarray, base_probability: float):
        """
        Compute SHAP values for one scaled feature vector.

        Returns a dict ready for the API:
            {
              base_value,
              patient_value (the model probability estimate),
              bars: [{feature, label, value, contribution, direction} ...],
              summary_english
            }
        """
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X_scaled.reshape(1, -1))
        # XGBoost binary -> shap_values is (n_samples, n_features).
        sv = np.asarray(shap_values).reshape(-1)
        base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])

        # SHAP is in log-odds space; convert to probability contribution space
        # so "+0.09" is an additive shift on the probability, which is intuitive.
        contributions = _logit_to_prob_contributions(sv, base_value)

        bars = []
        for idx, feat in enumerate(FEATURES):
            raw = float(sv[idx])
            contrib = contributions[idx]
            direction = "increases" if contrib > 1e-4 else ("decreases" if contrib < -1e-4 else "neutral")
            bars.append(
                {
                    "feature": feat,
                    "label": FEATURE_LABELS[feat],
                    "value": round(raw, 4),
                    "contribution": round(contrib, 4),
                    "direction": direction,
                }
            )

        # Top contributors by |contribution|
        top = sorted(bars, key=lambda b: abs(b["contribution"]), reverse=True)
        top_features = top[:5]

        summary = (
            f"The model assigned a {'higher' if base_probability >= 0.5 else 'lower'} "
            f"predicted risk mainly because "
            + ", ".join(b["label"] for b in top[:2])
            + f" had a strong influence on this prediction."
        )

        return {
            "method": "shap",
            "model": self.model_id,
            "base_value": round(base_probability, 4),
            "patient_value": round(base_probability, 4),
            "top_features": [{"feature": b["feature"], "label": b["label"],
                              "value": b["value"], "contribution": b["contribution"],
                              "direction": b["direction"]} for b in top_features],
            "summary_english": summary,
        }

    # ------------------------------------------------------------------ #
    def global_feature_importance(self, X: np.ndarray) -> list[dict]:
        """Mean |SHAP| per feature over a background matrix, sorted desc."""
        explainer = self._get_explainer()
        shap_values = explainer.shap_values(X)
        mean_abs = np.abs(np.asarray(shap_values)).mean(axis=0)
        order = np.argsort(mean_abs)[::-1]
        return [
            {
                "feature": FEATURES[i],
                "label": FEATURE_LABELS[FEATURES[i]],
                "mean_abs_shap": round(float(mean_abs[i]), 5),
            }
            for i in order
        ]


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _logit_to_prob_contributions(shap_logits: np.ndarray, base_logit: float) -> np.ndarray:
    """
    Convert additive SHAP values in log-odds space into additive contributions
    on the probability scale.

    For a logistic link f(x) = sigmoid(b + sum(s_i)), we approximate the
    probability-space contribution of feature i by the difference:
        sigmoid(b + s_i) - sigmoid(b)
    which keeps the contributions readable ("+0.31") and monotone.
    """
    base_prob = 1.0 / (1.0 + np.exp(-np.clip(base_logit, -500, 500)))
    contribs = np.zeros_like(shap_logits)
    for i, s_i in enumerate(shap_logits):
        shifted_logit = base_logit + s_i
        shifted_prob = 1.0 / (1.0 + np.exp(-np.clip(shifted_logit, -500, 500)))
        contribs[i] = shifted_prob - base_prob
    return contribs


# Module-level singleton -----------------------------------------------------
_explainer: ShapExplainer | None = None


def get_explainer() -> ShapExplainer:
    global _explainer
    if _explainer is None:
        _explainer = ShapExplainer()
    return _explainer
