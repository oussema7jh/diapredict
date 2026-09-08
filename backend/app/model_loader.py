"""
DiaPredict AI — model registry / loader.

Discovers trained models on disk, loads them into memory and exposes a
uniform, typed interface for the three ML models and the deep network.
Also provides access to the persisted validation metrics (stored by the
training scripts in ``models/meta.json``).
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import torch

from .config import (
    DL_DROPOUT,
    DL_HIDDEN_DIMS,
    DL_INPUT_DIM,
    DL_MODELS_DIR,
    META_PATH,
    ML_MODELS_DIR,
)

# torch tensor / dtype helpers ---------------------------------------------
TORCH_DEVICE = torch.device("cpu")
TORCH_DTYPE = torch.float32


# --------------------------------------------------------------------------- #
# Model file names
# --------------------------------------------------------------------------- #
ML_FILE_NAMES = {
    "logistic_regression": "logistic_regression.joblib",
    "random_forest": "random_forest.joblib",
    "xgboost": "xgboost.joblib",
}

DL_FILE_NAME = "mlp.pt"

MODEL_DISPLAY_NAMES = {
    "logistic_regression": "Logistic Regression",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "neural_network": "Neural Network",
}


class MLP(torch.nn.Module):
    """Fully-connected feed-forward network (the deep model).

    Architecture (matching section 4 of the spec):
        Input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout
              -> Linear -> Sigmoid -> probability
    """

    def __init__(self, input_dim: int = DL_INPUT_DIM, hidden: tuple[int, int] = DL_HIDDEN_DIMS,
                 dropout: float = DL_DROPOUT, seed: int = 42):
        super().__init__()
        torch.manual_seed(seed)
        i1, i2 = input_dim, hidden[0]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(i1, hidden[0]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[0], hidden[1]),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden[1], 1),
            torch.nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


@dataclass
class LoadedModel:
    model_id: str
    model_name: str
    kind: str  # 'ml' or 'dl'
    model: object
    loaded: bool
    path: Path
    file: str
    error: str | None = None


class ModelRegistry:
    """Loads and caches every model once per process."""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *_a, **_k):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        self._models: dict[str, LoadedModel] = {}
        self._load_all()

    # ------------------------------------------------------------------ #
    def _load_all(self) -> None:
        self._models.clear()
        for model_id, fname in ML_FILE_NAMES.items():
            self._models[model_id] = self._load_ml(model_id, ML_MODELS_DIR / fname)
        self._models["neural_network"] = self._load_dl(DL_MODELS_DIR / DL_FILE_NAME)

    def _load_ml(self, model_id: str, path: Path) -> LoadedModel:
        try:
            if not path.exists():
                return LoadedModel(model_id, MODEL_DISPLAY_NAMES[model_id], "ml",
                                   None, False, path, path.name, "model file missing")
            model = joblib.load(path)
            return LoadedModel(model_id, MODEL_DISPLAY_NAMES[model_id], "ml",
                               model, True, path, path.name)
        except Exception as exc:  # pragma: no cover - defensive
            return LoadedModel(model_id, MODEL_DISPLAY_NAMES[model_id], "ml",
                               None, False, path, path.name, str(exc))

    def _load_dl(self, path: Path) -> LoadedModel:
        model_id = "neural_network"
        try:
            if not path.exists():
                return LoadedModel(model_id, MODEL_DISPLAY_NAMES[model_id], "dl",
                                   None, False, path, path.name, "checkpoint missing")
            net = MLP()
            state = torch.load(path, map_location=TORCH_DEVICE, weights_only=True)
            net.load_state_dict(state)
            net.eval()
            return LoadedModel(model_id, MODEL_DISPLAY_NAMES[model_id], "dl",
                               net, True, path, path.name)
        except Exception as exc:  # pragma: no cover - defensive
            return LoadedModel(model_id, MODEL_DISPLAY_NAMES[model_id], "dl",
                               None, False, path, path.name, str(exc))

    # ------------------------------------------------------------------ #
    def all(self) -> list[LoadedModel]:
        return list(self._models.values())

    def get(self, model_id: str) -> LoadedModel | None:
        return self._models.get(model_id)

    def loaded(self) -> list[LoadedModel]:
        return [m for m in self._models.values() if m.loaded]

    def ml_loaded(self) -> list[LoadedModel]:
        return [m for m in self.loaded() if m.kind == "ml"]

    def dl_loaded(self) -> LoadedModel | None:
        for m in self.loaded():
            if m.kind == "dl":
                return m
        return None

    def all_loaded(self) -> bool:
        return len(self.loaded()) == 4


# --------------------------------------------------------------------------- #
# Metrics store
# --------------------------------------------------------------------------- #
class MetricsStore:
    """Reads validation metrics written by the training scripts."""

    def __init__(self, path: Path = META_PATH):
        self.path = path

    def _read(self) -> dict:
        if not self.path.exists():
            return {}
        with open(self.path) as fh:
            return json.load(fh)

    def get_metrics(self, model_id: str) -> dict | None:
        return self._read().get(model_id)

    def all_metrics(self) -> dict:
        return self._read()

    def write_full(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w") as fh:
            json.dump(data, fh, indent=2, default=_json_default)


def _json_default(obj):
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (Path,)):
        return str(obj)
    return str(obj)


# Module-level singletons ---------------------------------------------------
_registry: ModelRegistry | None = None


def get_registry() -> ModelRegistry:
    global _registry
    if _registry is None:
        _registry = ModelRegistry()
    return _registry


def reload_registry() -> ModelRegistry:
    global _registry
    _registry = ModelRegistry()
    return _registry
