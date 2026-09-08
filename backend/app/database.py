"""
DiaPredict AI — SQLite persistence.

Stores anonymous assessment records (by numeric assessment id, never any
personally identifiable information). The assessment id is auto-generated and
is the only identifier stored.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from typing import Any

from .config import DATABASE_PATH

_SCHEMA = """
CREATE TABLE IF NOT EXISTS assessments (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at       TEXT NOT NULL,
    input_features   TEXT NOT NULL,   -- JSON of the feature dict
    model_predictions TEXT NOT NULL,  -- JSON of per-model probabilities
    ensemble_prob    REAL NOT NULL,
    risk_percent     REAL NOT NULL,
    risk_category    TEXT NOT NULL,
    model_used       TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class AssessmentStore:
    """Thread-safe, minimal SQLite wrapper."""

    def __init__(self, path=None):
        self.path = str(path or DATABASE_PATH)
        self._lock = threading.Lock()
        self._init()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init(self) -> None:
        with self._connect() as conn:
            conn.execute(_SCHEMA)
            conn.commit()

    def insert(
        self,
        input_features: dict[str, Any],
        model_predictions: list[dict],
        ensemble_prob: float,
        risk_percent: float,
        risk_category: str,
        model_used: str,
    ) -> int:
        with self._lock:
            with self._connect() as conn:
                cur = conn.execute(
                    "INSERT INTO assessments "
                    "(created_at, input_features, model_predictions, ensemble_prob, "
                    " risk_percent, risk_category, model_used) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        _now(),
                        json.dumps(input_features),
                        json.dumps(model_predictions),
                        float(ensemble_prob),
                        float(risk_percent),
                        risk_category,
                        model_used,
                    ),
                )
                conn.commit()
                return int(cur.lastrowid)

    def count(self) -> int:
        with self._connect() as conn:
            return int(conn.execute("SELECT COUNT(*) FROM assessments").fetchone()[0])

    def get_assessment(self, assessment_id: int) -> dict | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM assessments WHERE id = ?", (assessment_id,)
            ).fetchone()
        if row is None:
            return None
        d = dict(row)
        d["input_features"] = json.loads(d["input_features"])
        d["model_predictions"] = json.loads(d["model_predictions"])
        return d

    def recent(self, limit: int = 100) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM assessments ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            d["input_features"] = json.loads(d["input_features"])
            d["model_predictions"] = json.loads(d["model_predictions"])
            out.append(d)
        return out

    def stats(self) -> dict:
        """Aggregate counts used by the dashboard."""
        rows = self.recent(limit=100000)
        total = len(rows)
        if total == 0:
            return {
                "total_assessments": 0,
                "avg_risk_percent": 0.0,
                "higher_risk_count": 0,
            }
        avg = sum(r["risk_percent"] for r in rows) / total
        higher = sum(1 for r in rows if r["risk_category"] == "Higher predicted risk")
        return {
            "total_assessments": total,
            "avg_risk_percent": round(avg, 2),
            "higher_risk_count": higher,
        }


# Singleton ----------------------------------------------------------------
_store: AssessmentStore | None = None


def get_store() -> AssessmentStore:
    global _store
    if _store is None:
        _store = AssessmentStore()
    return _store
