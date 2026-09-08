"""
DiaPredict AI — Deep Learning training script (PyTorch MLP).

Trains the fully-connected network described in the spec:
    Input -> Linear -> ReLU -> Dropout -> Linear -> ReLU -> Dropout
          -> Linear -> Sigmoid -> probability

Implements a real training/validation loop, Adam optimiser, BCELoss,
early stopping, model checkpointing and test-set evaluation.

Usage:
    python scripts/train_dl.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import (  # noqa: E402
    DL_DROPOUT,
    DL_HIDDEN_DIMS,
    DL_MODELS_DIR,
    DL_SEED,
    ensure_directories,
)
from backend.app.model_loader import MLP, TORCH_DEVICE, TORCH_DTYPE  # noqa: E402
from backend.app.preprocessing import load_raw_dataset, make_train_test  # noqa: E402
from training.evaluation import compute_metrics, save_models  # noqa: E402

ensure_directories()

SEED = DL_SEED
torch.manual_seed(SEED)
np.random.seed(SEED)

MODEL_FILES = {"neural_network": "mlp.pt"}


class Trainer:
    def __init__(
        self,
        input_dim: int,
        hidden=DL_HIDDEN_DIMS,
        dropout: float = DL_DROPOUT,
        lr: float = 1e-3,
        epochs: int = 200,
        batch_size: int = 64,
        patience: int = 25,
        val_size: float = 0.20,
    ):
        self.net = MLP(input_dim=input_dim, hidden=hidden, dropout=dropout, seed=SEED).to(TORCH_DEVICE)
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_size = val_size
        self.history = {"train_loss": [], "val_loss": [], "val_auc": []}

    def _tensors(self, X: np.ndarray, y: np.ndarray):
        x = torch.tensor(X, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        y = torch.tensor(y, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        return x, y

    def fit(self, X_tr, y_tr, X_va, y_va):
        x_train, y_train = self._tensors(X_tr, y_tr)
        x_val, y_val = self._tensors(X_va, y_va)
        n = x_train.shape[0]

        best_val_auc = -1.0
        best_state = None
        epochs_no_improve = 0
        e0 = time.time()

        for epoch in range(1, self.epochs + 1):
            # Training ----------------------------------------------------- #
            self.net.train()
            perm = torch.randperm(n)
            train_loss = 0.0
            for i in range(0, n, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb, yb = x_train[idx], y_train[idx]
                self.optimizer.zero_grad()
                out = self.net(xb)
                loss = self.criterion(out, yb)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item() * xb.shape[0]
            train_loss /= n

            # Validation ---------------------------------------------------- #
            self.net.eval()
            with torch.no_grad():
                val_out = self.net(x_val)
                val_loss = self.criterion(val_out, y_val).item()
                y_prob = val_out.cpu().numpy()
            y_true_np = y_val.cpu().numpy()
            val_auc = roc_auc_score(y_true_np, y_prob)

            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            self.history["val_auc"].append(float(val_auc))

            if epoch % 10 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:3d} | train_loss {train_loss:.4f} | "
                    f"val_loss {val_loss:.4f} | val_auc {val_auc:.4f}"
                )

            # Early stopping / best checkpoint ------------------------------ #
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= self.patience:
                    print(f"Early stopping at epoch {epoch} (no improvement for {self.patience}).")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        elapsed = time.time() - e0
        self.best_epoch = self.history["val_auc"].index(max(self.history["val_auc"])) + 1
        print(f"Training finished in {elapsed:.1f}s, best val_auc {best_val_auc:.4f}.")
        return self

    def evaluate(self, X_te, y_te):
        self.net.eval()
        x = torch.tensor(X_te, dtype=TORCH_DTYPE, device=TORCH_DEVICE)
        with torch.no_grad():
            y_prob = self.net(x).cpu().numpy()
        return compute_metrics(y_te, y_prob)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), path)
        print(f"Saved checkpoint: {path}")


def main(verbose: bool = True):
    raw = load_raw_dataset()
    # Build train/test split, then carve out a validation set from training.
    X_train, y_train, X_test, y_test, scaler, medians = make_train_test(raw)

    from sklearn.model_selection import train_test_split
    X_tr, X_va, y_tr, y_va = train_test_split(
        X_train, y_train, test_size=0.20, random_state=SEED, stratify=y_train
    )

    trainer = Trainer(input_dim=X_tr.shape[1], lr=1e-3, epochs=200, patience=25, batch_size=64)
    trainer.fit(X_tr, y_tr, X_va, y_va)

    metrics = trainer.evaluate(X_test, y_test)
    metrics["model_id"] = "neural_network"
    metrics["model_name"] = "Neural Network"
    metrics["history"] = {
        "train_loss": [round(v, 6) for v in trainer.history["train_loss"]],
        "val_loss": [round(v, 6) for v in trainer.history["val_loss"]],
        "val_auc": [round(v, 6) for v in trainer.history["val_auc"]],
    }
    metrics["best_epoch"] = getattr(trainer, "best_epoch", None)

    trainer.save(DL_MODELS_DIR / MODEL_FILES["neural_network"])
    save_models(
        {"neural_network": metrics},
        {"neural_network": f"models/dl/{MODEL_FILES['neural_network']}"},
    )

    # Persist training curves for the frontend charts.
    from backend.app.config import MODELS_DIR
    (MODELS_DIR / "dl_training_history.json").write_text(
        json.dumps(metrics["history"], indent=2)
    )

    if verbose:
        print("\n=== Neural Network evaluation (test set) ===")
        print(f"  ROC-AUC:  {metrics['roc_auc']:.4f}")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision:{metrics['precision']:.4f}")
        print(f"  Recall:   {metrics['recall']:.4f}")
        print(f"  F1:       {metrics['f1']:.4f}")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DiaPredict neural network.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    main(verbose=not args.quiet)
