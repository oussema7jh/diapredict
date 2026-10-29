"""Database dashboard regressions; no model files or third-party packages needed."""

import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from backend.app.database import AssessmentStore


class AssessmentStatsTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(dir=".", ignore_cleanup_errors=True)
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "assessments.db"
        self.store = AssessmentStore(self.path)

    def add_rows(self, rows):
        with sqlite3.connect(self.path) as conn:
            conn.executemany(
                "INSERT INTO assessments "
                "(created_at, input_features, model_predictions, ensemble_prob, "
                "risk_percent, risk_category, model_used) VALUES (?, ?, ?, ?, ?, ?, ?)",
                [("2026-01-01T00:00:00Z", "{}", "[]", percent / 100, percent, category, "test")
                 for percent, category in rows],
            )

    def test_empty_database(self):
        self.assertEqual(self.store.stats(), {
            "total_assessments": 0,
            "avg_risk_percent": 0.0,
            "higher_risk_count": 0,
        })

    def test_mixed_categories_and_rounding(self):
        self.add_rows([
            (10.0, "Lower predicted risk"),
            (40.0, "Moderate predicted risk"),
            (80.0, "Higher predicted risk"),
        ])
        self.assertEqual(self.store.stats(), {
            "total_assessments": 3,
            "avg_risk_percent": 43.33,
            "higher_risk_count": 1,
        })

    def test_includes_history_beyond_old_limit(self):
        self.add_rows([(90.0, "Higher predicted risk")])
        self.add_rows([(10.0, "Lower predicted risk")] * 100000)
        self.assertEqual(self.store.stats(), {
            "total_assessments": 100001,
            "avg_risk_percent": 10.0,
            "higher_risk_count": 1,
        })

    def test_stats_do_not_load_full_assessment_records(self):
        self.add_rows([(20.0, "Lower predicted risk")])
        with patch.object(self.store, "recent", side_effect=AssertionError("records loaded")):
            self.assertEqual(self.store.stats()["total_assessments"], 1)


if __name__ == "__main__":
    unittest.main()
