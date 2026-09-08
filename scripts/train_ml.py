"""
DiaPredict AI — Machine Learning training script.

Trains Logistic Regression, Random Forest and XGBoost on the Pima dataset,
evaluates each on the held-out test set and saves:
    models/ml/<model>.joblib
    models/meta.json   (metrics + metadata)

Usage:
    python scripts/train_ml.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import ML_MODELS_DIR, ensure_directories  # noqa: E402
from backend.app.preprocessing import load_raw_dataset, make_train_test  # noqa: E402
from training.evaluation import compute_metrics, predict_proba_binary, save_models  # noqa: E402

ensure_directories()

MODEL_BUILDERS = {
    "logistic_regression": lambda: LogisticRegression(max_iter=2000, random_state=42),
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300, random_state=42, n_jobs=-1, class_weight="balanced"
    ),
    "xgboost": lambda: XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.9,
        colsample_bytree=0.9,
    ),
}

MODEL_FILES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
}

MODEL_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
}


def cross_validate(model, X, y, cv: int = 5) -> dict:
    """5-fold stratified cross-validation scores for the report."""
    from sklearn.model_selection import cross_val_score, StratifiedKFold

    kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kf, scoring="roc_auc", n_jobs=-1)
    return {"cv_splits": cv, "mean_roc_auc": float(scores.mean()), "std_roc_auc": float(scores.std())}


def train_all(verbose: bool = True) -> dict:
    raw = load_raw_dataset()
    X_train, y_train, X_test, y_test, scaler, medians = make_train_test(raw)

    # Cache the scaled training matrix so the SHAP background can be reused.
    data_dir = PROJECT_ROOT / "data" / "processed"
    np.save(data_dir / "X_train.npy", X_train)
    np.save(data_dir / "X_test.npy", X_test)
    np.save(data_dir / "y_train.npy", y_train)
    np.save(data_dir / "y_test.npy", y_test)

    all_metrics = {}
    for model_id, builder in MODEL_BUILDERS.items():
        if verbose:
            print(f"\n=== Training {MODEL_NAMES[model_id]} ===")
        model = builder()
        model.fit(X_train, y_train)
        y_prob = predict_proba_binary(model, X_test)
        metrics = compute_metrics(y_test, y_prob)
        metrics["model_id"] = model_id
        metrics["model_name"] = MODEL_NAMES[model_id]
        metrics["cv"] = cross_validate(model, X_train, y_train)
        all_metrics[model_id] = metrics

        path = ML_MODELS_DIR / MODEL_FILES[model_id]
        joblib.dump(model, path)
        if verbose:
            print(f"  Test ROC-AUC:   {metrics['roc_auc']:.4f}")
            print(f"  Test accuracy:  {metrics['accuracy']:.4f}")
            print(f"  Saved:          {path}")

    save_models(all_metrics, {k: f"models/ml/{v}" for k, v in MODEL_FILES.items()})
    if verbose:
        print("\nAll ML models trained and evaluated.")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiaPredict ML models.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    train_all(verbose=not args.quiet)
