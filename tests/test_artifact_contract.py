"""Contracts for the locally promoted full-fair CatBoost decision artifacts."""

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

    def test_promoted_model_and_scenarios_are_oof_scoped(self) -> None:
        scenarios = pd.read_csv(self.artifacts / "operating_scenarios_oof_catboost.csv")
        self.assertEqual(self.metadata["model"], "catboost")
        self.assertEqual(self.metadata["evaluation_scope"], "five_fold_oof")
        self.assertEqual(scenarios["scenario_id"].tolist(), ["recall_first", "balanced_f1", "precision_first"])
        self.assertTrue(scenarios["threshold"].is_monotonic_increasing)
        self.assertEqual(self.metadata["app_default_scenario_id"], "balanced_f1")
        self.assertEqual(set(scenarios["evaluation_split"]), {"five_fold_oof"})

    def test_topk_and_deciles_are_truthfully_labeled_oof_evidence(self) -> None:
        targeting = pd.read_csv(self.artifacts / "topk_lift_oof_catboost.csv")
        deciles = pd.read_csv(self.artifacts / "risk_decile_oof_catboost.csv")
        self.assertEqual(set(targeting["model"]), {"catboost"})
        self.assertEqual(set(targeting["evaluation_split"]), {"five_fold_oof"})
        self.assertEqual(set(deciles["evaluation_split"]), {"five_fold_oof"})
        self.assertEqual(len(deciles), 10)
        self.assertTrue(targeting["capture_rate"].is_monotonic_increasing)

    def test_saved_pipeline_scores_the_committed_schema(self) -> None:
        from scripts.run_full_fair_comparison import Features

        setattr(sys.modules["__main__"], "Features", Features)
        model = joblib.load(self.artifacts / "model" / "music_churn_pipeline.joblib")
        test = pd.read_csv(ROOT / "data" / "test.csv")
        row = test.loc[:, self.metadata["input_columns"]].head(1)
        probability = model.predict_proba(row)[:, 1]
        self.assertEqual(probability.shape, (1,))
        self.assertGreaterEqual(float(probability[0]), 0.0)
        self.assertLessEqual(float(probability[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
