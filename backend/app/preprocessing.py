"""
DiaPredict AI — data preprocessing.

This module owns every transformation applied to the raw Pima Indians dataset
as well as the exact same transformations applied to a single incoming
patient record during real-time inference. Keeping a single source of truth
guarantees that training and serving stay consistent.

Pipeline
--------
Raw dataset
    -> drop missing / invalid values (physically impossible zeros)
    -> reset indices
    -> train/test split (stratified)
    -> fit StandardScaler on the training split only
    -> transform both splits and any new patient record
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    FEATURES,
    TARGET,
    ensure_directories,
)

# --------------------------------------------------------------------------- #
# Invalid-value handling
# --------------------------------------------------------------------------- #
# In the original Pima dataset a value of 0 is used as a stand-in for a
# missing measurement for variables where 0 is physiologically impossible.
# These are replaced with NaN and then imputed with the median from the
# training split (a robust, common choice for skewed medical variables).
ZERO_IS_INVALID = {
    "Glucose": True,
    "BloodPressure": True,
    "SkinThickness": True,
    "Insulin": True,
    "BMI": True,
    "Pregnancies": False,  # 0 pregnancies is legitimate
    "DiabetesPedigreeFunction": False,  # 0 is possible but rare
    "Age": False,
}

SPLIT_SEED = 42
TEST_SIZE = 0.25

SCALER_FILENAME = "scaler.joblib"


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    """Load the Pima dataset CSV (handles either known CSV layout)."""
    candidates = list(DATA_RAW_DIR.glob("*.csv"))
    if path is None and len(candidates) == 0:
        raise FileNotFoundError(
            "No dataset found in data/raw/. Follow data/README.md to place "
            "the Pima Indians Diabetes dataset there."
        )
    if path is None:
        # Prefer the named column version if present.
        named = [p for p in candidates if p.name == "diabetes.csv"]
        path = named[0] if named else candidates[0]

    df = pd.read_csv(path)
    # The JBrownlee file has no column names; give it the standard ones.
    if df.shape[1] == 9 and list(df.columns)[0].lower() == "pregnancies":
        # Already named -> keep.
        pass
    if list(df.columns) == list(range(9)):
        df.columns = FEATURES + [TARGET]
    # Normalise column casing/spacing.
    df.columns = [str(c).strip() for c in df.columns]
    return df


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert physically impossible zeros to NaN and return a clean frame.

    Returns a deep copy so callers never mutate the original input.
    """
    df = df.copy()

    for col in FEATURES:
        if ZERO_IS_INVALID.get(col, False):
            df.loc[df[col] <= 0, col] = np.nan

    # Remove rows with no valid measurements at all in the key clinical
    # variables (would otherwise be pure noise).
    needed = ["Glucose", "BMI"]
    df = df.dropna(subset=[c for c in needed if c in df.columns])

    # Drop the small number of fully duplicate rows (harmless but tidy).
    df = df.drop_duplicates().reset_index(drop=True)
    return df


def impute_median(df: pd.DataFrame, medians: dict[str, float] | None = None):
    """
    Impute NaNs with the median. When ``medians`` is provided (inference path)
    the SAME medians learned at training time are reused — this is essential to
    avoid data leakage.
    """
    df = df.copy()
    if medians is None:
        medians = {c: float(df[c].median()) for c in FEATURES if c in df.columns}
    for col, val in medians.items():
        if col in df.columns:
            df[col] = df[col].fillna(val)
    return df, medians


def preprocess(
    df: pd.DataFrame,
    fit_scaler: bool = True,
    scaler: StandardScaler | None = None,
    medians: dict[str, float] | None = None,
):
    """
    End-to-end preprocessing.

    Parameters
    ----------
    fit_scaler:
        True when training (fit the scaler on the given X), False at inference
        (reuse the provided fitted scaler).
    scaler:
        A fitted StandardScaler used when ``fit_scaler`` is False.
    medians:
        Medians used for imputation at inference time.

    Returns
    -------
    (X, y, scaler, medians)
        X is the scaled feature matrix (numpy), y the target (or None if the
        input frame had no target column), plus the fitted scaler/medians.
    """
    if scaler is not None and fit_scaler:
        raise ValueError("Cannot both fit and reuse a scaler.")

    df = clean_dataset(df)
    has_target = TARGET in df.columns
    X_df = df[FEATURES] if not has_target else df[FEATURES]

    X_imputed, medians = impute_median(X_df, medians)

    if fit_scaler:
        scaler = StandardScaler().fit(X_imputed)
    elif scaler is None:
        raise ValueError("At inference time a fitted scaler must be provided.")

    X = scaler.transform(X_imputed)
    y = df[TARGET].to_numpy().astype(int) if has_target else None
    return X, y, scaler, medians


def make_train_test(df: pd.DataFrame):
    """
    Return (X_tr, y_tr, X_te, y_te, scaler, medians).

    The scaler and medians are fitted on the TRAINING split only; the test
    split is transformed with those same parameters to avoid data leakage.
    """
    clean = clean_dataset(df)
    X = clean[FEATURES]
    y = clean[TARGET].to_numpy().astype(int)

    X_train_df, X_test_df, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=SPLIT_SEED, stratify=y
    )

    X_train_df, medians = impute_median(X_train_df)
    X_test_df = X_test_df.copy()
    for col, val in medians.items():
        X_test_df[col] = X_test_df[col].fillna(val)

    # Fit on numpy arrays so the scaler is not bound to feature names (this
    # keeps single-row numpy inference paths warning-free).
    scaler = StandardScaler().fit(X_train_df.to_numpy())
    X_train = scaler.transform(X_train_df.to_numpy())
    X_test = scaler.transform(X_test_df.to_numpy())

    # Persist median / scaler so inference can replay the same transforms.
    ensure_directories()
    joblib.dump(scaler, DATA_PROCESSED_DIR / SCALER_FILENAME)
    with open(DATA_PROCESSED_DIR / "medians.json", "w") as fh:
        json.dump(medians, fh, indent=2)

    return X_train, y_train, X_test, y_test, scaler, medians


def load_scaler_and_medians():
    """Load the persisted fitted scaler + medians (used at inference time)."""
    ensure_directories()
    scaler = joblib.load(DATA_PROCESSED_DIR / SCALER_FILENAME)
    with open(DATA_PROCESSED_DIR / "medians.json") as fh:
        medians = json.load(fh)
    return scaler, medians
