"""Build the final PlaylistPro submission allowlist manifest."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INCLUDE_FILES = {
    "README.md", "requirements.txt", "models/churn_pipeline.joblib",
    "artifacts/feature_schema.json", "artifacts/model_metadata.json",
    "artifacts/metrics.csv", "artifacts/requirements_traceability.csv",
    "artifacts/model/music_churn_pipeline.joblib", "artifacts/model/metadata.json",
    "reports/preprocessing_report.md", "reports/preprocessing_report.pdf",
    "reports/training_report.md", "reports/training_report.pdf",
    "reports/final_model_selection_decision.md", "reports/final_local_model_integration.md",
    "docs/data_card.md", "docs/data_dictionary.md",
    "docs/data_source.md", "docs/requirements.md", "docs/validation_plan.md",
    "docs/final_submission_checklist.md", "docs/human_confirmation_required.md",
    "docs/project_structure_audit.md",
    "tools/validate_submission.py", "tools/validate_streamlit.py",
    "tools/build_submission_manifest.py",
}

INCLUDE_DIRS = {
    "app", "assets/screenshots", "data", "notebooks", "figures/preprocessing",
    "figures/training", "figures/presentation_v3", "artifacts/presentation_v3", "tests",
    "experiments/submission_full_comparison/20260721_full_fair_v1",
}

FINAL_ARTIFACTS = {
    "app_schema_smoke_test.csv", "bootstrap_confidence_intervals.csv",
    "calibration_summary.csv", "current_insight_inventory.csv",
    "equal_contact_comparison.csv", "equal_recall_comparison.csv",
    "feature_importance.csv", "final_model_selection_matrix.csv",
    "insight_preprocessing_register.csv", "model_comparison_fair.csv",
    "operating_scenarios_oof_catboost.csv",
    "preprocessing_experiment_results.csv", "random_search_summary.csv",
    "risk_decile_oof_catboost.csv", "seed_stability_summary.csv",
    "segment_error_analysis.csv", "test_predictions.csv",
    "threshold_operating_scenarios_v2.csv", "threshold_sweep_oof_catboost.csv",
    "top3_fine_tuning_summary.csv", "top3_selection_matrix.csv",
    "topk_lift_oof_catboost.csv",
}

FINAL_SCRIPTS = {
    "scripts/build_full_oof_diagnostics.py", "scripts/finalize_full_fair_candidate.py",
    "scripts/promote_full_fair_candidate.py",
    "scripts/run_full_fair_comparison.py",
}

FINAL_SRC = {"src/features.py", "src/full_fair_features.py", "src/retention_strategy.py"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT).as_posix()
    if rel in INCLUDE_FILES or rel in FINAL_SCRIPTS or rel in FINAL_SRC:
        return True
    if rel.startswith("artifacts/") and path.name in FINAL_ARTIFACTS:
        return True
    return any(rel == directory or rel.startswith(directory + "/") for directory in INCLUDE_DIRS)


def main() -> None:
    files = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or path.name == "submission_manifest.json" or not included(path):
            continue
        rel = path.relative_to(ROOT).as_posix()
        if "__pycache__" in rel or rel.endswith((".pyc", ".pyo", ".inspect.ndjson")):
            continue
        if "prompt" in rel.lower():
            continue
        files.append({"path": rel, "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {"package": "PlaylistPro_final_submission", "policy": "explicit_allowlist", "files": files}
    (ROOT / "submission_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"manifest_files": len(files)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
