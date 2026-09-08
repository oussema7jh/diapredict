"""PREDICT router — all prediction-focused endpoints."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from ..config import DISCLAIMER
from ..database import get_store
from ..schemas import (
    ModelPrediction,
    PatientInput,
    PredictionResponse,
    ShapBar,
    Explainability,
)
from ..services import (
    run_assessment,
    run_dl_only,
    run_ensemble_only,
    run_ml_only,
)

router = APIRouter(prefix="/predict", tags=["predict"])


@router.post("", response_model=PredictionResponse)
def predict(input_data: PatientInput) -> PredictionResponse:
    """Run the full assessment across every trained model + ensemble + SHAP."""
    try:
        return run_assessment(input_data)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/ml", response_model=list[ModelPrediction])
def predict_ml(input_data: PatientInput) -> list[ModelPrediction]:
    """Run only the machine-learning models."""
    try:
        return run_ml_only(input_data)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/dl", response_model=ModelPrediction)
def predict_dl(input_data: PatientInput) -> ModelPrediction:
    """Run only the neural network."""
    try:
        return run_dl_only(input_data)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/ensemble", response_model=ModelPrediction)
def predict_ensemble(input_data: PatientInput) -> ModelPrediction:
    """Run the weighted ensemble only."""
    try:
        return run_ensemble_only(input_data)
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))


@router.post("/explain", response_model=Explainability)
async def explain(input_data: PatientInput) -> Explainability:
    """Return the SHAP local explanation for a given input."""
    try:
        resp = run_assessment(input_data, explain=True)
        if resp.explanation is None:
            raise HTTPException(status_code=503, detail="SHAP explanation unavailable.")
        return resp.explanation
    except ValueError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
