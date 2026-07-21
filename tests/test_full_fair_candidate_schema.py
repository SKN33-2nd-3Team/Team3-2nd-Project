"""Contracts for the completed full-fair candidate artifact."""

from __future__ import annotations

import json
import subprocess
import sys
import unittest
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FullFairCandidateSchemaTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.candidate_dir = ROOT / "models" / "candidates" / "20260721_full_fair_v1"
        cls.metadata = json.loads((cls.candidate_dir / "metadata.json").read_text(encoding="utf-8"))
        cls.model = joblib.load(cls.candidate_dir / "candidate_pipeline.joblib")
        cls.test = pd.read_csv(ROOT / "data" / "test.csv").head(5)

    def test_fresh_process_reload(self) -> None:
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import joblib,pandas as pd; "
                    "m=joblib.load(r'models/candidates/20260721_full_fair_v1/candidate_pipeline.joblib'); "
                    "p=m.predict_proba(pd.read_csv(r'data/test.csv').head(1))[:,1]; assert len(p)==1"
                ),
            ],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_reordered_columns_preserve_prediction(self) -> None:
        normal = self.model.predict_proba(self.test)[:, 1]
        reordered = self.model.predict_proba(self.test.loc[:, list(reversed(self.test.columns))])[:, 1]
        np.testing.assert_allclose(normal, reordered, rtol=0, atol=1e-12)

    def test_unseen_category_is_supported(self) -> None:
        frame = self.test.copy()
        frame.loc[:, "subscription_type"] = "__UNSEEN_PLAN__"
        probability = self.model.predict_proba(frame)[:, 1]
        self.assertEqual(probability.shape, (len(frame),))
        self.assertTrue(((probability >= 0) & (probability <= 1)).all())

    def test_missing_required_column_is_rejected(self) -> None:
        with self.assertRaises(Exception):
            self.model.predict_proba(self.test.drop(columns=["weekly_hours"]))

    def test_invalid_numeric_type_is_rejected(self) -> None:
        frame = self.test.copy()
        frame["weekly_hours"] = frame["weekly_hours"].astype(object)
        frame.loc[:, "weekly_hours"] = "not-a-number"
        with self.assertRaises((TypeError, ValueError)):
            self.model.predict_proba(frame)

    def test_metadata_records_validation_boundary(self) -> None:
        self.assertTrue(self.metadata["fresh_process_reload_verified"])
        self.assertIn("독립된 외부 라벨 Holdout", self.metadata["known_limitations"][0])
        self.assertEqual(self.metadata["candidate_model"], "catboost")


if __name__ == "__main__":
    unittest.main()
