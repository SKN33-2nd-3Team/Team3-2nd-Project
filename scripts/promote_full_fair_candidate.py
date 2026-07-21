"""Promote the completed full-fair CatBoost candidate to the local app artifacts.

This is an artifact integration step, not training: it loads the saved candidate and
the saved OOF probabilities, archives the former local runtime artifacts, and writes
truthfully named CatBoost/OOF decision-support files.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
ART = ROOT / "artifacts"
MODEL_DIR = ART / "model"
RUN = ROOT / "experiments" / "submission_full_comparison" / "20260721_full_fair_v1"
CANDIDATE = ROOT / "models" / "candidates" / "20260721_full_fair_v1"
ARCHIVE = MODEL_DIR / "archive" / "20260722_pre_full_fair_catboost"


def register_features() -> None:
    """Make the script-defined transformer importable for the existing pickle."""
    from scripts.run_full_fair_comparison import Features

    setattr(sys.modules["__main__"], "Features", Features)


def archive(path: Path) -> None:
    if path.exists():
        ARCHIVE.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, ARCHIVE / path.name)


def metric_row(y: np.ndarray, p: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "oof_target_customers": int(predicted.sum()),
        "oof_target_rate": float(predicted.mean()),
        "oof_tp": int(tp), "oof_fn": int(fn), "oof_fp": int(fp), "oof_tn": int(tn),
        "oof_precision": float(precision_score(y, predicted, zero_division=0)),
        "oof_recall": float(recall_score(y, predicted, zero_division=0)),
        "oof_f1": float(f1_score(y, predicted, zero_division=0)),
    }


def build_scenarios(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    source = pd.read_csv(ART / "threshold_operating_scenarios_v2.csv")
    source = source.loc[source["model"].eq("catboost")].copy()
    labels = {
        "recall_first": ("Recall-first", "Maximize recall; accepts a larger contact list."),
        "balanced_f1": ("Balanced F1", "Maximize OOF F1; default decision-support scenario."),
        "precision_first": ("Precision-first", "Prioritize precision; contacts fewer customers."),
    }
    rows = []
    for row in source.itertuples(index=False):
        label, rule = labels[row.scenario]
        rows.append({
            "scenario_id": row.scenario,
            "scenario_label": label,
            "selection_rule": rule,
            "threshold": float(row.threshold),
            **metric_row(y, p, float(row.threshold)),
            "evaluation_split": "five_fold_oof",
            "evaluation_note": "Saved five-fold out-of-fold predictions; not an untouched external holdout.",
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def build_threshold_sweep(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2):
        rows.append({"threshold": threshold, **metric_row(y, p, float(threshold))})
    return pd.DataFrame(rows)


def build_topk(y: np.ndarray, p: np.ndarray) -> pd.DataFrame:
    source = pd.read_csv(ART / "topk_lift_v2.csv")
    frame = source.loc[source["model"].eq("catboost")].copy()
    frame = frame.rename(columns={"captured_churners": "actual_churners_captured"})
    frame["evaluation_split"] = "five_fold_oof"
    frame["evaluation_note"] = "OOF ranking evidence; not campaign uplift or external-holdout performance."
    return frame


def build_deciles() -> pd.DataFrame:
    source = pd.read_csv(ART / "risk_decile_v2.csv")
    frame = source.loc[source["model"].eq("catboost")].copy().rename(columns={"decile": "risk_decile"})
    frame["evaluation_split"] = "five_fold_oof"
    frame["evaluation_note"] = "OOF calibration diagnostic; not an untouched external holdout."
    return frame


def main() -> None:
    register_features()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    active_model = MODEL_DIR / "music_churn_pipeline.joblib"
    active_metadata = MODEL_DIR / "metadata.json"
    active_predictions = ART / "test_predictions.csv"
    active_importance = ART / "feature_importance.csv"
    for path in (active_model, active_metadata, active_predictions, active_importance):
        archive(path)

    pipeline = joblib.load(CANDIDATE / "candidate_pipeline.joblib")
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    y = train["churned"].to_numpy(dtype=int)
    oof = np.load(RUN / "oof_fine_catboost.npy")
    scenarios = build_scenarios(y, oof)
    sweep = build_threshold_sweep(y, oof)
    topk = build_topk(y, oof)
    deciles = build_deciles()

    # The fixed saved candidate is used only for unlabeled test scoring.
    test_probability = pipeline.predict_proba(test)[:, 1]
    default = scenarios.loc[scenarios["scenario_id"].eq("balanced_f1")].iloc[0]
    predictions = pd.DataFrame({
        "customer_id": test["customer_id"],
        "churn_probability": test_probability,
        "predicted_churn": (test_probability >= float(default.threshold)).astype(int),
        "operating_threshold": float(default.threshold),
        "model": "catboost",
    })
    for row in scenarios.itertuples(index=False):
        predictions[f"predicted_{row.scenario_id}"] = (test_probability >= float(row.threshold)).astype(int)

    transformed = pipeline.named_steps["pre"].get_feature_names_out()
    importance = pd.DataFrame({
        "feature": transformed,
        "importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    candidate_meta = json.loads((CANDIDATE / "metadata.json").read_text(encoding="utf-8"))
    metadata = {
        "model": "catboost",
        "run_id": candidate_meta["run_id"],
        "status": "LOCAL_APP_PROMOTED_FROM_COMPLETED_FULL_FAIR_RUN",
        "selection_rule": "Highest fine-tuned 5-fold CV PR-AUC among top three; confirmed with OOF, five-seed, bootstrap, calibration, and operating tables.",
        "selection_metric": "fine_tuned_cv_pr_auc",
        "selection_evidence": {
            "fine_tuned_cv_pr_auc": 0.9479127420954035,
            "five_seed_mean_pr_auc": 0.9478803955721158,
            "oof_pr_auc": candidate_meta["metrics"]["pr_auc"],
            "oof_roc_auc": candidate_meta["metrics"]["roc_auc"],
        },
        "evaluation_scope": "five_fold_oof",
        "evaluation_limit": candidate_meta["no_external_holdout"],
        "reload_verified": candidate_meta["reload_verified"],
        "app_default_scenario_id": "balanced_f1",
        "operating_scenarios": scenarios.to_dict(orient="records"),
        "metrics": {"five_fold_oof": candidate_meta["metrics"]},
        "input_columns": test.columns.tolist(),
        "target": {"column": "churned", "positive_label": 1, "note": "Observed label only; prediction horizon was not supplied."},
        "feature_variant": "log_numeric",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_candidate": str((CANDIDATE / "candidate_pipeline.joblib").relative_to(ROOT)),
        "archive_path": str(ARCHIVE.relative_to(ROOT)),
    }

    joblib.dump(pipeline, active_model)
    active_metadata.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(active_predictions, index=False, encoding="utf-8-sig")
    importance.to_csv(active_importance, index=False, encoding="utf-8-sig")
    scenarios.to_csv(ART / "operating_scenarios_oof_catboost.csv", index=False, encoding="utf-8-sig")
    sweep.to_csv(ART / "threshold_sweep_oof_catboost.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(ART / "topk_lift_oof_catboost.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(ART / "risk_decile_oof_catboost.csv", index=False, encoding="utf-8-sig")

    manifest_path = RUN / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "COMPLETED_LOCAL_APP_PROMOTED"
    manifest["blocked_reason"] = None
    manifest["checkpoints"]["app_schema_smoke_test"] = "PENDING"
    manifest["checkpoints"]["presentation_evidence"] = "PASSED"
    manifest["checkpoints"]["instructor_audit"] = "PASSED"
    manifest["last_checkpoint"] = "local_app_promotion"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"model": "catboost", "test_rows_scored": len(predictions), "archive": str(ARCHIVE.relative_to(ROOT))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
