"""Shared pytest fixtures."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture(scope="session")
def valid_patient():
    from backend.app.schemas import PatientInput

    return PatientInput(
        Pregnancies=2,
        Glucose=145,
        BloodPressure=85,
        SkinThickness=32,
        Insulin=120,
        BMI=31.2,
        DiabetesPedigreeFunction=0.52,
        Age=45,
    )


@pytest.fixture(scope="session")
def test_client():
    from fastapi.testclient import TestClient

    from backend.app.main import app

    return TestClient(app)
