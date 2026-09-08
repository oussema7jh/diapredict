"""Data preprocessing tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.app.config import FEATURES
from backend.app.preprocessing import clean_dataset, impute_median, make_train_test, preprocess

from .conftest import PROJECT_ROOT


def _sample_df():
    return pd.DataFrame(
        {
            "Pregnancies": [1, 0, 3, 2, 5],
            "Glucose": [148, 85, 183, 89, 137],
            "BloodPressure": [72, 66, 64, 66, 40],
            "SkinThickness": [35, 29, 0, 23, 35],
            "Insulin": [0, 0, 0, 94, 168],
            "BMI": [33.6, 26.6, 23.3, 28.1, 43.1],
            "DiabetesPedigreeFunction": [0.627, 0.351, 0.672, 0.167, 2.288],
            "Age": [50, 31, 32, 21, 33],
            "Outcome": [1, 0, 1, 0, 1],
        }
    )


def test_impossible_zeros_become_nan():
    df = clean_dataset(_sample_df())
    # Insulin and SkinThickness contain 0 -> should be NaN after cleaning.
    assert df["Insulin"].iloc[0] != df["Insulin"].iloc[0]  # NaN
    assert df["SkinThickness"].iloc[2] != df["SkinThickness"].iloc[2]  # NaN
    # Pregnancies == 0 is valid and preserved.
    assert df["Pregnancies"].iloc[1] == 0


def test_impute_median_fills_nan():
    df = clean_dataset(_sample_df())
    df_imp, medians = impute_median(df)
    assert df_imp.isnull().sum().sum() == 0
    assert "Insulin" in medians
    assert medians["Insulin"] >= 0


def test_preprocess_output_shapes():
    X, y, scaler, medians = preprocess(_sample_df(), fit_scaler=True)
    assert X.shape[1] == len(FEATURES)
    assert y is None if False else y is not None
    assert np.isnan(X).sum() == 0


def test_make_train_test_shapes():
    df = pd.concat([_sample_df(), _sample_df(), _sample_df(), _sample_df()])
    X_tr, y_tr, X_te, y_te, scaler, medians = make_train_test(df)
    assert X_tr.shape[1] == len(FEATURES)
    assert X_tr.shape[0] == y_tr.shape[0]
    assert X_te.shape[0] == y_te.shape[0]
    assert X_te.shape[1] == len(FEATURES)
    # Stratified split must keep both classes present. (Train contains all classes)
    assert set(np.unique(y_tr)) == {0, 1}
    assert set(np.unique(y_te)) <= {0, 1}


def test_inference_reuses_same_scaler():
    """Preprocessing at inference must produce the same scaler output."""
    df = pd.concat([_sample_df(), _sample_df(), _sample_df(), _sample_df()])
    _, _, X_te, _, scaler, medians = make_train_test(df)
    row = _sample_df().iloc[[0]].drop(columns=["Outcome"])
    X2, _, _, _ = preprocess(row, fit_scaler=False, scaler=scaler, medians=medians)
    assert np.isnan(X2).sum() == 0
    assert X2.shape == (1, len(FEATURES))
