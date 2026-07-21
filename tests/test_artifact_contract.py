"""Lightweight contracts for the committed model and decision-support artifacts."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class ArtifactContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = ROOT / "artifacts"
        cls.metadata = json.loads((cls.artifacts / "model" / "metadata.json").read_text(encoding="utf-8"))

    def test_operating_scenarios_match_metadata(self) -> None:
        scenarios = pd.read_csv(self.artifacts / "operating_scenarios.csv")
        self.assertEqual(scenarios["scenario_id"].tolist(), [
            "aggressive_cost_3_to_1",
            "balanced_f1",
            "precision_first",
        ])
        self.assertTrue(scenarios["threshold"].is_monotonic_increasing)
        self.assertEqual(self.metadata["app_default_scenario_id"], "balanced_f1")
        self.assertEqual(
            scenarios["scenario_id"].tolist(),
            [row["scenario_id"] for row in self.metadata["operating_scenarios"]],
        )

    def test_targeting_and_calibration_are_internal_holdout_only(self) -> None:
        targeting = pd.read_csv(self.artifacts / "targeting_metrics_test.csv")
        deciles = pd.read_csv(self.artifacts / "decile_calibration_test.csv")
        self.assertEqual(set(targeting["evaluation_split"]), {"internal_test_holdout"})
        self.assertEqual(set(deciles["evaluation_split"]), {"internal_test_holdout"})
        self.assertEqual(len(deciles), 10)
        self.assertTrue(targeting["capture_rate"].is_monotonic_increasing)

    def test_saved_pipeline_scores_the_committed_schema(self) -> None:
        from src.features import MusicFeatureEngineer  # noqa: F401

        model = joblib.load(self.artifacts / "model" / "music_churn_pipeline.joblib")
        test = pd.read_csv(ROOT / "data" / "test.csv")
        row = test.loc[:, self.metadata["input_columns"]].head(1)
        probability = model.predict_proba(row)[:, 1]
        self.assertEqual(probability.shape, (1,))
        self.assertGreaterEqual(float(probability[0]), 0.0)
        self.assertLessEqual(float(probability[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
