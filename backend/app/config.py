"""
DiaPredict AI — configuration and paths.

Central place for project-wide settings, paths, model constants and the
medical disclaimer. Keeping these constants here makes the rest of the code
clean and avoids magic strings scattered around the codebase.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------- #
# Project paths
# --------------------------------------------------------------------------- #
# backend/app/config.py -> two levels up is the project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_RAW_DIR = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
ML_MODELS_DIR = MODELS_DIR / "ml"
DL_MODELS_DIR = MODELS_DIR / "dl"

DATABASE_PATH = PROJECT_ROOT / "data" / "diapredict.db"
META_PATH = PROJECT_ROOT / "models" / "meta.json"  # stores validation metrics

# --------------------------------------------------------------------------- #
# Feature specification
# --------------------------------------------------------------------------- #
# Column names must match the Pima Indians Diabetes dataset exactly.
# ``FLOAT_FEATURES`` are mapped to 0..1 probability space for ML / DL / SHAP.
FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

FEATURE_LABELS = {
    "Pregnancies": "Pregnancies",
    "Glucose": "Glucose",
    "BloodPressure": "Blood Pressure",
    "SkinThickness": "Skin Thickness",
    "Insulin": "Insulin",
    "BMI": "BMI",
    "DiabetesPedigreeFunction": "Diabetes Pedigree Function",
    "Age": "Age",
}

FEATURE_UNITS = {
    "Pregnancies": "count",
    "Glucose": "mg/dL",
    "BloodPressure": "mmHg",
    "SkinThickness": "mm",
    "Insulin": "mu U/mL",
    "BMI": "kg/m^2",
    "DiabetesPedigreeFunction": "score",
    "Age": "years",
}

TARGET = "Outcome"

# --------------------------------------------------------------------------- #
# Risk categories (model-defined, NOT clinical thresholds)
# --------------------------------------------------------------------------- #
LOW_RISK_THRESHOLD = 0.30   # <= 30% -> lower predicted risk
MODERATE_RISK_THRESHOLD = 0.61  # 31-60% -> moderate, >= 61% -> higher

# --------------------------------------------------------------------------- #
# Ensemble weighting
# --------------------------------------------------------------------------- #
# Documented weighting scheme. The ML ensemble uses simple averaging of the
# three tree / linear models; the final ensemble blends the ML average with the
# deep-learning probability. Weights are justified in docs/ARCHITECTURE.md.
ML_ENSEMBLE_WEIGHTS = {"logistic_regression": 1.0, "random_forest": 1.0, "xgboost": 1.0}
ML_WEIGHT = 0.7   # contribution of the ML ensemble to the final score
DL_WEIGHT = 0.3   # contribution of the deep network to the final score

# --------------------------------------------------------------------------- #
# Deep learning constants
# --------------------------------------------------------------------------- #
DL_INPUT_DIM = len(FEATURES)
DL_HIDDEN_DIMS = (24, 16)   # two hidden layers (see spec architecture)
DL_DROPOUT = 0.3
DL_SEED = 42

# --------------------------------------------------------------------------- #
# Medical / ethical disclaimer
# --------------------------------------------------------------------------- #
DISCLAIMER = (
    "DiaPredict AI is an educational and research prototype. It provides "
    "machine-learning-based risk estimates and is not a medical diagnosis. "
    "The results should not be used to make medical decisions. Always consult "
    "a qualified healthcare professional for medical advice."
)


def ensure_directories() -> None:
    """Create all directories that must exist before models can be saved."""
    for path in (
        DATA_RAW_DIR,
        DATA_PROCESSED_DIR,
        ML_MODELS_DIR,
        DL_MODELS_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)
