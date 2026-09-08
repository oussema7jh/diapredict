"""HEALTH router — simple liveness probe."""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from ..config import DISCLAIMER
from ..model_loader import get_registry
from ..schemas import HealthResponse

router = APIRouter(tags=["health"])

APP_VERSION = "1.0.0"


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    registry = get_registry()
    return HealthResponse(
        status="ok",
        version=APP_VERSION,
        time=datetime.now(timezone.utc).isoformat(),
        models_loaded=len(registry.loaded()),
        disclaimer=DISCLAIMER,
    )
