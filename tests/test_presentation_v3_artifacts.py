"""Presentation-v3 figure/source contracts."""

from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


class PresentationV3ArtifactTest(unittest.TestCase):
    def test_every_figure_has_a_nonempty_source_csv(self) -> None:
        inventory = pd.read_csv(ROOT / "figures" / "presentation_v3" / "figure_inventory.csv")
        self.assertEqual(len(inventory), 20)
        self.assertEqual(inventory["figure"].nunique(), 20)
        for row in inventory.itertuples(index=False):
            figure = ROOT / row.figure
            source = ROOT / row.source_csv
            self.assertTrue(figure.is_file(), figure)
            self.assertGreater(figure.stat().st_size, 30_000, figure)
            self.assertTrue(source.is_file(), source)
            self.assertFalse(pd.read_csv(source).empty, source)

    def test_required_instructor_views_are_present(self) -> None:
        required = {
            "01_performance_progression.png",
            "04_eight_model_fair_comparison.png",
            "10_threshold_precision_recall_f1.png",
            "14_topk_capture_lift.png",
            "15_risk_decile.png",
            "17_final_feature_importance.png",
            "19_numeric_target_correlations.png",
        }
        actual = {path.name for path in (ROOT / "figures" / "presentation_v3").glob("*.png")}
        self.assertTrue(required.issubset(actual))


if __name__ == "__main__":
    unittest.main()
