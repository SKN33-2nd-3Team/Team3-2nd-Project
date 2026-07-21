"""Reuse verified committed artifacts for the bounded-run evidence checkpoint.

This script deliberately does not train or replace a model.  It creates the
requested run-local tables with provenance so completed measurements are not
rerun merely to copy them into a new reporting layout.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN_ID = "20260721_submission_bounded_v1"
RUN_DIR = ROOT / "experiments" / "submission_bounded" / RUN_ID
ARTIFACTS = ROOT / "artifacts"


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RUN_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    comparison = pd.read_csv(ARTIFACTS / "model_comparison.csv")
    comparison["provenance"] = "VERIFIED: artifacts/model_comparison.csv"
    comparison.to_csv(RUN_DIR / "reused_model_comparison.csv", index=False)

    scenarios = pd.read_csv(ARTIFACTS / "operating_scenarios.csv")
    scenarios["selection_scope"] = "Validation selection; internal Test holdout evaluation"
    scenarios.to_csv(RUN_DIR / "threshold_scenarios.csv", index=False)

    topk = pd.read_csv(ARTIFACTS / "targeting_metrics_test.csv")
    topk.to_csv(RUN_DIR / "topk_lift.csv", index=False)

    deciles = pd.read_csv(ARTIFACTS / "decile_calibration_test.csv")
    deciles.to_csv(RUN_DIR / "risk_decile.csv", index=False)

    raw = comparison.loc[comparison["model"] == "logistic_raw"].iloc[0]
    engineered = comparison.loc[comparison["model"] == "logistic_engineered"].iloc[0]
    feature_experiments = pd.DataFrame(
        [
            {
                "experiment_id": "FE-00",
                "experiment": "minimal logistic baseline",
                "change": "No derived features",
                "validation_pr_auc": raw["validation_pr_auc"],
                "test_pr_auc": raw["test_pr_auc"],
                "decision": "reference",
                "provenance": "VERIFIED: current pipeline comparison",
            },
            {
                "experiment_id": "FE-01",
                "experiment": "deterministic ratio/date features",
                "change": "signup_days_ago plus four ratios; all computed inside Pipeline",
                "validation_pr_auc": engineered["validation_pr_auc"],
                "test_pr_auc": engineered["test_pr_auc"],
                "delta_test_pr_auc_vs_baseline": engineered["test_pr_auc"] - raw["test_pr_auc"],
                "decision": "measured but not automatically adopted for the bounded candidate",
                "provenance": "VERIFIED: current pipeline comparison",
            },
            {
                "experiment_id": "FE-02",
                "experiment": "polynomial/interactions",
                "decision": "rejected",
                "reason": "One-hot expansion raises complexity; tree candidates already learn interactions.",
                "provenance": "INFERRED from model family and feature cardinality",
            },
            {
                "experiment_id": "FE-03",
                "experiment": "log/power transform",
                "decision": "deferred",
                "reason": "No completed isolated experiment checkpoint; not rerun before P0 candidate comparison.",
                "provenance": "UNVERIFIED",
            },
            {
                "experiment_id": "FE-04",
                "experiment": "rare-level grouping/frequency encoding",
                "decision": "rejected",
                "reason": "Categorical fields are low-cardinality; OneHotEncoder is sufficient.",
                "provenance": "VERIFIED schema plus INFERRED design decision",
            },
            {
                "experiment_id": "FE-05",
                "experiment": "target encoding",
                "decision": "rejected",
                "reason": "Leakage-prone without a CV-only encoder and unnecessary for low-cardinality fields.",
                "provenance": "VERIFIED schema plus ML guardrail",
            },
        ]
    )
    feature_experiments.to_csv(RUN_DIR / "feature_experiment_screening.csv", index=False)

    available = {"dummy_prior", "logistic_raw", "decision_tree", "random_forest", "gradient_boosting"}
    screening_rows = []
    mapping = {
        "dummy_prior": "DummyClassifier",
        "logistic_raw": "LogisticRegression",
        "decision_tree": "DecisionTreeClassifier",
        "random_forest": "RandomForestClassifier",
        "gradient_boosting": "GradientBoostingClassifier",
    }
    for key, label in mapping.items():
        screening_rows.append({"model": label, "status": "VERIFIED", "reason": "Existing committed comparison artifact available.", "source_key": key})
    for label in ["XGBoost", "LightGBM", "CatBoost"]:
        screening_rows.append({"model": label, "status": "UNAVAILABLE", "reason": "Package not installed; no dependency added in bounded run.", "source_key": ""})
    pd.DataFrame(screening_rows).to_csv(RUN_DIR / "model_screening.csv", index=False)

    metadata = json.loads((ARTIFACTS / "model" / "metadata.json").read_text(encoding="utf-8"))
    presentation_metrics = pd.DataFrame(
        [
            {"metric": "internal_test_pr_auc", "value": metadata["metrics"]["test"]["pr_auc"], "status": "VERIFIED", "source": "artifacts/model/metadata.json"},
            {"metric": "internal_test_roc_auc", "value": metadata["metrics"]["test"]["roc_auc"], "status": "VERIFIED", "source": "artifacts/model/metadata.json"},
            {"metric": "prediction_horizon", "value": "UNVERIFIED", "status": "UNVERIFIED", "source": "dataset documentation absent"},
            {"metric": "campaign_uplift", "value": "UNVERIFIED", "status": "UNVERIFIED", "source": "campaign outcomes absent"},
        ]
    )
    presentation_metrics.to_csv(RUN_DIR / "presentation_metrics.csv", index=False)

    (RUN_DIR / "presentation_experiment_flow.csv").write_text(
        "stage,status,next_decision\n"
        "repository_audit,VERIFIED,Reuse artifacts without rerunning completed experiments\n"
        "feature_screening,VERIFIED/INFERRED,Run only isolated missing P0 experiments\n"
        "external_boosters,UNAVAILABLE,Obtain explicit dependency approval before installation\n"
        "candidate_selection,DEFERRED,Require bounded candidate run and reload validation\n",
        encoding="utf-8",
    )

    report = """# Presentation Evidence Pack — checkpoint 1

## Verified evidence reused without rerun

- Current committed comparison includes Dummy, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Current internal Test Gradient Boosting PR-AUC is recorded in `presentation_metrics.csv`.
- Threshold scenarios, Top-K/Lift, and risk deciles were copied with their original internal-holdout provenance.

## Bounded-run gaps

- XGBoost, LightGBM, and CatBoost are **UNAVAILABLE** because their packages are absent and no dependency was added.
- No separate bounded candidate has yet been trained, saved, and reloaded; candidate status remains **DEFERRED**.
- The supplied data does not independently verify a 30-day prediction horizon or campaign uplift.

## Next P0 work

Run only the isolated bounded candidate comparison needed to create a separate candidate artifact, reload validation, and a final evidence pack. Do not modify the current app or operational model.
"""
    (RUN_DIR / "presentation_evidence_pack_checkpoint.md").write_text(report, encoding="utf-8")

    manifest["status"] = "checkpoint_materialized"
    manifest["checkpoint"] = "existing_experiment_evidence_materialized"
    manifest["completed_steps"] = ["repository_audit", "existing_experiment_evidence_materialized"]
    manifest["pending_steps"] = ["isolated_bounded_candidate_run", "candidate_reload_validation", "final_evidence_pack"]
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
