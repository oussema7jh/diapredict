"""
DiaPredict AI — Pydantic schemas (API contract).

These models define the request/response shapes exposed by the REST API and
perform the input validation that prevents impossible physiological values
from reaching the models.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from .config import FEATURE_LABELS, FEATURES

# --------------------------------------------------------------------------- #
# Input
# --------------------------------------------------------------------------- #
# Human-friendly name -> (min, max, validator message) for every feature.
# These are wide but physiologically sensible ranges used purely to catch
# typos / impossible values, NOT clinical ranges.
_VALUE_LIMITS: dict[str, tuple[float, float]] = {
    "Pregnancies": (0, 20),
    "Glucose": (40, 400),
    "BloodPressure": (30, 250),
    "SkinThickness": (5, 100),
    "Insulin": (5, 900),
    "BMI": (10, 90),
    "DiabetesPedigreeFunction": (0.02, 3.0),
    "Age": (1, 120),
}


class PatientInput(BaseModel):
    """A single patient's health information as collected from the form."""

    Pregnancies: float = Field(ge=0, le=20)
    Glucose: float = Field(ge=40, le=400)
    BloodPressure: float = Field(ge=30, le=250)
    SkinThickness: float = Field(ge=5, le=100)
    Insulin: float = Field(ge=5, le=900)
    BMI: float = Field(ge=10, le=90)
    DiabetesPedigreeFunction: float = Field(ge=0.02, le=3.0)
    Age: float = Field(ge=1, le=120)

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pregnancies": 2,
                "Glucose": 145,
                "BloodPressure": 85,
                "SkinThickness": 32,
                "Insulin": 120,
                "BMI": 31.2,
                "DiabetesPedigreeFunction": 0.52,
                "Age": 45,
            }
        }
    }

    @field_validator("Pregnancies", "Age")
    @classmethod
    def _no_negative(cls, v: float, info):
        if v < 0:
            raise ValueError(f"{info.field_name} cannot be negative")
        return v

    def as_feature_dict(self) -> dict[str, float]:
        """Return the input as an ordered dict aligned with FEATURES."""
        return {f: float(getattr(self, f)) for f in FEATURES}

    def pretty(self) -> list[dict[str, str]]:
        """Human-readable list of {label, value, unit} for the UI."""
        out = []
        for f in FEATURES:
            out.append(
                {
                    "feature": f,
                    "label": FEATURE_LABELS[f],
                    "value": _fmt(getattr(self, f)),
                    "unit": _units(f),
                }
            )
        return out


def _units(feature: str) -> str:
    from .config import FEATURE_UNITS  # avoid circular import noise

    return FEATURE_UNITS[feature]


def _fmt(v: float) -> str:
    return f"{v:g}"


# --------------------------------------------------------------------------- #
# Model / prediction helper shapes
# --------------------------------------------------------------------------- #
class ModelPrediction(BaseModel):
    model_id: str
    model_name: str
    probability: float = Field(ge=0, le=1)
    risk_percent: float = Field(ge=0, le=100)
    status: str = "ok"


class ShapBar(BaseModel):
    feature: str
    label: str
    value: float  # SHAP value (raw)
    contribution: float  # contribution to probability (probability space)
    direction: Literal["increases", "decreases", "neutral"]


class Explainability(BaseModel):
    method: str = "shap"
    model: str
    base_value: float
    patient_value: float
    top_features: list[ShapBar]
    summary_english: str


class PredictionResponse(BaseModel):
    assessment_id: int
    timestamp: str
    input: dict[str, float]
    input_pretty: list[dict[str, str]]
    model_predictions: list[ModelPrediction]
    ensemble: ModelPrediction
    risk_percent: float
    risk_category: str
    risk_category_explanation: str
    explanation: Optional[Explainability] = None
    disclaimer: str


# --------------------------------------------------------------------------- #
# Models / performance
# --------------------------------------------------------------------------- #
class ModelPerformance(BaseModel):
    model_id: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    confusion_matrix: dict[str, int]
    trained_at: Optional[str] = None
    completed: bool
    message: Optional[str] = None


class ModelStatus(BaseModel):
    model_id: str
    model_name: str
    type: Literal["ml", "dl"]
    loaded: bool
    status: str
    file: str
    message: Optional[str] = None


class ModelsStatusResponse(BaseModel):
    models: list[ModelStatus]
    all_loaded: bool
    disclaimer: str


class HealthResponse(BaseModel):
    status: str
    version: str
    time: str
    models_loaded: int
    disclaimer: str
