"""DiaPredict AI backend application package."""

from .config import DISCLAIMER, ensure_directories

ensure_directories()

__all__ = ["DISCLAIMER", "ensure_directories"]
