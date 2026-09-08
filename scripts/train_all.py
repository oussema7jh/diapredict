"""
DiaPredict AI — train every model.

Usage:
    python scripts/train_all.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_ml import train_all as train_ml  # noqa: E402
from scripts.train_dl import main as train_dl  # noqa: E402


if __name__ == "__main__":
    print("=" * 60)
    print("Training Machine Learning models ...")
    print("=" * 60)
    train_ml()

    print("\n" + "=" * 60)
    print("Training Deep Learning model ...")
    print("=" * 60)
    train_dl()

    print("\nAll models trained. See models/ for artifacts.")
