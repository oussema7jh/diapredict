"""MODELS router — model status, performance metrics and dashboard stats."""

from __future__ import annotations

from fastapi import APIRouter

from ..config import DISCLAIMER
from ..database import get_store
from ..model_loader import MetricsStore, get_registry
from ..schemas import (
    ModelPerformance,
    ModelsStatusResponse,
    ModelStatus,
)

router = APIRouter(prefix="/models", tags=["models"])
metrics_store = MetricsStore()


@router.get("/status", response_model=ModelsStatusResponse)
def models_status() -> ModelsStatusResponse:
    """Report which trained models are currently loaded and ready."""
    registry = get_registry()
    statuses: list[ModelStatus] = []
    for m in registry.all():
        statuses.append(
            ModelStatus(
                model_id=m.model_id,
                model_name=m.model_name,
                type=m.kind,
                loaded=m.loaded,
                status="ready" if m.loaded else "unavailable",
                file=m.file,
                message=m.error,
            )
        )
    return ModelsStatusResponse(
        models=statuses,
        all_loaded=registry.all_loaded(),
        disclaimer=DISCLAIMER,
    )


@router.get("/performance", response_model=list[ModelPerformance])
def models_performance() -> list[ModelPerformance]:
    """Return real validation metrics computed on the held-out test set."""
    registry = get_registry()
    all_metrics = metrics_store.all_metrics()

    results: list[ModelPerformance] = []
    for m in registry.all():
        met = all_metrics.get(m.model_id)
        if met is None or not m.loaded:
            results.append(
                ModelPerformance(
                    model_id=m.model_id,
                    model_name=m.model_name,
                    accuracy=0.0,
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    roc_auc=0.0,
                    confusion_matrix={"tn": 0, "fp": 0, "fn": 0, "tp": 0},
                    completed=False,
                    message="Model not trained / metrics unavailable.",
                )
            )
            continue

        cm = met.get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
        results.append(
            ModelPerformance(
                model_id=m.model_id,
                model_name=m.model_name,
                accuracy=round(float(met["accuracy"]), 4),
                precision=round(float(met["precision"]), 4),
                recall=round(float(met["recall"]), 4),
                f1=round(float(met["f1"]), 4),
                roc_auc=round(float(met["roc_auc"]), 4),
                confusion_matrix=cm,
                trained_at=met.get("trained_at"),
                completed=True,
            )
        )
    return results


@router.get("/curves")
def curves():
    """ROC and precision-recall curve data for each model, computed on the
    held-out test set. Used by the frontend to render comparison charts."""
    import numpy as np
    from sklearn.metrics import roc_curve, precision_recall_curve

    from ..config import DATA_PROCESSED_DIR
    from ..model_loader import get_registry

    X_test = np.load(DATA_PROCESSED_DIR / "X_test.npy")
    y_test = np.load(DATA_PROCESSED_DIR / "y_test.npy")

    registry = get_registry()
    out = []
    for m in registry.loaded():
        if m.kind == "dl":
            import torch

            from ..model_loader import TORCH_DEVICE, TORCH_DTYPE
            x = torch.tensor(X_test, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
            m.model.eval()
            with torch.no_grad():
                p = m.model(x).numpy()
        else:
            p = m.model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, p)
        prec, rec, _ = precision_recall_curve(y_test, p)
        out.append(
            {
                "model_id": m.model_id,
                "model_name": m.model_name,
                "roc": [
                    {"fpr": float(a), "tpr": float(b)}
                    for a, b in zip(fpr, tpr)
                ],
                "pr": [
                    {"recall": float(a), "precision": float(b)}
                    for a, b in zip(rec, prec)
                ],
            }
        )
    return {"curves": out}


@router.get("/dl-training")
def dl_training():
    """Neural network training/validation curves (saved during training)."""
    from pathlib import Path
    from backend.app.config import MODELS_DIR

    p = MODELS_DIR / "dl_training_history.json"
    if not p.exists():
        return {"history": None}
    import json

    return {"history": json.loads(p.read_text())}


@router.get("/feature-importance")
def feature_importance():
    """Global mean |SHAP| feature importance (for the dashboard / charts)."""
    from ..services import global_feature_importance

    try:
        return {"importance": global_feature_importance()}
    except Exception as exc:
        return {"importance": [], "message": str(exc)}


@router.get("/stats")
def stats():
    """Aggregate dashboard statistics."""
    store = get_store()
    stats = store.stats()
    recent = store.recent(limit=200)
    return {
        **stats,
        "recent_assessments": recent,
    }


@router.get("/assessments/{assessment_id}")
def get_assessment(assessment_id: int):
    store = get_store()
    rec = store.get_assessment(assessment_id)
    if rec is None:
        return {"error": "Assessment not found", "id": assessment_id}
    return rec
