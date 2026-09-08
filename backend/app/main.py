"""
DiaPredict AI — FastAPI application entry point.

Run with:
    uvicorn backend.app.main:app --reload
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .config import DISCLAIMER
from .routers import health, models, predict

app = FastAPI(
    title="DiaPredict AI API",
    description=(
        "Explainable diabetes risk prediction platform. "
        "Educational/research prototype — NOT a medical diagnosis."
    ),
    version="1.0.0",
    openapi_tags=[
        {"name": "predict", "description": "Run model / ensemble predictions."},
        {"name": "models", "description": "Model status and performance metrics."},
        {"name": "health", "description": "Liveness / health probe."},
    ],
)

# CORS — allow the Vite dev server and any local origin during development.
# (Open by default for a local research prototype; restrict in production.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(predict.router)
app.include_router(models.router)


@app.get("/", tags=["health"])
def root():
    return {
        "name": "DiaPredict AI",
        "version": "1.0.0",
        "docs": "/docs",
        "disclaimer": DISCLAIMER,
    }
