"""Input-validation tests for the Pydantic schema."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from backend.app.schemas import PatientInput

GOOD = {
    "Pregnancies": 2,
    "Glucose": 145,
    "BloodPressure": 85,
    "SkinThickness": 32,
    "Insulin": 120,
    "BMI": 31.2,
    "DiabetesPedigreeFunction": 0.52,
    "Age": 45,
}


def test_valid_input_accepted():
    p = PatientInput(**GOOD)
    assert p.Glucose == 145


@pytest.mark.parametrize(
    "field,bad",
    [
        ("Age", -5),
        ("Pregnancies", -1),
        ("Glucose", 20),      # below min
        ("Glucose", 500),     # above max
        ("BMI", 300),         # beyond physiological
        ("BloodPressure", 0),
    ],
)
def test_impossible_values_rejected(field, bad):
    data = dict(GOOD)
    data[field] = bad
    with pytest.raises(ValidationError):
        PatientInput(**data)


def test_nan_rejected():
    import math

    data = dict(GOOD)
    data["Glucose"] = float("nan")
    with pytest.raises(ValidationError):
        PatientInput(**data)


def test_as_feature_dict_ordered():
    p = PatientInput(**GOOD)
    fd = p.as_feature_dict()
    from backend.app.config import FEATURES

    assert list(fd.keys()) == FEATURES
