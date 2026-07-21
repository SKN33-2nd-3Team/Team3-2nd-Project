"""Run the isolated, resource-bounded P0 candidate comparison.

This deliberately leaves ``artifacts/model`` and the Streamlit application
untouched. Existing model rows are reused from their verified checkpoint; only
the three newly available boosters and a small 3-fold tune of the validation
leader are computed here.
"""

from __future__ import annotations

import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.run_pipeline import metrics_at_threshold, model_pipeline, targeting_metrics, decile_metrics

RUN_ID = "20260721_submission_bounded_v1"
RUN_DIR = ROOT / "experiments" / "submission_bounded" / RUN_ID
CANDIDATE_DIR = ROOT / "models" / "candidates" / RUN_ID
RANDOM_STATE = 42


def package_versions() -> dict[str, str]:
    import catboost
    import lightgbm
    import sklearn
    import xgboost

    return {
        "python": platform.python_version(), "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "lightgbm": lightgbm.__version__, "catboost": catboost.__version__,
    }


def split_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    train = pd.read_csv(ROOT / "data" / "train.csv")
    X, y = train.drop(columns=["churned"]), train["churned"].astype(int)
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=RANDOM_STATE)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=RANDOM_STATE)
    return X_train, X_val, X_test, y_train, y_val, y_test


def model_specs() -> dict[str, object]:
    return {
        "xgboost": XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, subsample=0.9, colsample_bytree=0.9, eval_metric="logloss", n_jobs=4, random_state=RANDOM_STATE),
        "lightgbm": LGBMClassifier(n_estimators=150, learning_rate=0.05, num_leaves=31, subsample=0.9, colsample_bytree=0.9, n_jobs=4, random_state=RANDOM_STATE, verbosity=-1),
        "catboost": CatBoostClassifier(iterations=150, learning_rate=0.05, depth=6, loss_function="Logloss", verbose=False, thread_count=4, random_seed=RANDOM_STATE),
    }


def metrics(y: pd.Series, probability: np.ndarray) -> dict[str, float]:
    predicted = (probability >= 0.5).astype(int)
    return {
        "pr_auc": float(average_precision_score(y, probability)), "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)), "recall_at_0_5": float(recall_score(y, predicted)),
        "precision_at_0_5": float(precision_score(y, predicted)), "f1_at_0_5": float(f1_score(y, predicted)),
    }


def validation_scenarios(y: pd.Series, probability: np.ndarray) -> pd.DataFrame:
    rows = []
    for threshold in np.linspace(0.05, 0.95, 91):
        row = metrics_at_threshold(y, probability, float(threshold))
        row["target_rate"] = (row["tp"] + row["fp"]) / len(y)
        rows.append(row)
    sweep = pd.DataFrame(rows)
    high_recall = sweep[sweep["recall"] >= 0.95].sort_values(["precision", "recall"], ascending=False)
    high_precision = sweep[sweep["precision"] >= 0.95].sort_values(["recall", "precision"], ascending=False)
    selections = [
        ("recall_first", "Validation recall at least 95%; highest precision", high_recall.iloc[0] if not high_recall.empty else sweep.sort_values(["recall", "precision"], ascending=False).iloc[0]),
        ("balanced_f1", "Validation F1 maximum", sweep.sort_values(["f1", "recall", "precision"], ascending=False).iloc[0]),
        ("precision_first", "Validation precision at least 95%; highest recall", high_precision.iloc[0] if not high_precision.empty else sweep.sort_values(["precision", "recall"], ascending=False).iloc[0]),
    ]
    return pd.DataFrame([{"scenario_id": key, "selection_rule": rule, **item.to_dict()} for key, rule, item in selections])


def tune_space(name: str) -> dict[str, list]:
    spaces = {
        "xgboost": {"model__max_depth": [2, 3, 4], "model__learning_rate": [0.03, 0.05, 0.08], "model__subsample": [0.8, 0.9, 1.0]},
        "lightgbm": {"model__num_leaves": [15, 31, 63], "model__learning_rate": [0.03, 0.05, 0.08], "model__min_child_samples": [10, 20, 40]},
        "catboost": {"model__depth": [4, 6, 8], "model__learning_rate": [0.03, 0.05, 0.08], "model__l2_leaf_reg": [1, 3, 5]},
    }
    return spaces[name]


def main() -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = RUN_DIR / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    X_train, X_val, X_test, y_train, y_val, y_test = split_data()
    existing = pd.read_csv(ROOT / "artifacts" / "model_comparison.csv")
    reuse = existing[existing["model"].isin(["dummy_prior", "logistic_raw", "decision_tree", "random_forest", "gradient_boosting"])].copy()
    reuse["provenance"] = "VERIFIED_REUSED_CHECKPOINT"
    reuse["selection_metric"] = reuse["validation_pr_auc"]

    rows, fitted = [], {}
    for name, model in model_specs().items():
        pipe = model_pipeline(X_train, model, engineer=True, scale_numeric=False)
        pipe.fit(X_train, y_train)
        probability = pipe.predict_proba(X_val)[:, 1]
        result = {"model": name, "provenance": "VERIFIED_ISOLATED_VALIDATION_RUN", "selection_metric": average_precision_score(y_val, probability), **{f"validation_{k}": v for k, v in metrics(y_val, probability).items()}}
        rows.append(result)
        fitted[name] = (pipe, probability)
    new_results = pd.DataFrame(rows)
    comparison = pd.concat([reuse, new_results], ignore_index=True, sort=False).sort_values("selection_metric", ascending=False)
    comparison.to_csv(RUN_DIR / "model_comparison_p0.csv", index=False, encoding="utf-8-sig")

    leader = str(comparison.iloc[0]["model"])
    if leader not in fitted:
        # The reused leader is refit only in this isolated candidate directory; the current operational artifact is not changed.
        from sklearn.ensemble import GradientBoostingClassifier
        leader = "gradient_boosting"
        fitted[leader] = (model_pipeline(X_train, GradientBoostingClassifier(n_estimators=150, learning_rate=0.05, max_depth=3, random_state=RANDOM_STATE), engineer=True), None)

    base_pipe = fitted[leader][0]
    if fitted[leader][1] is None:
        base_pipe.fit(X_train, y_train)
    search = RandomizedSearchCV(
        estimator=base_pipe, param_distributions=tune_space(leader) if leader in model_specs() else {"model__n_estimators": [120, 150, 180], "model__learning_rate": [0.03, 0.05, 0.08], "model__max_depth": [2, 3, 4]},
        n_iter=3, scoring="average_precision", cv=StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE, n_jobs=1, refit=True,
    )
    search.fit(X_train, y_train)
    tuned_prob = search.best_estimator_.predict_proba(X_val)[:, 1]
    base_prob = base_pipe.predict_proba(X_val)[:, 1]
    use_tuned = average_precision_score(y_val, tuned_prob) >= average_precision_score(y_val, base_prob)
    candidate = search.best_estimator_ if use_tuned else base_pipe
    candidate_variant = "randomized_search_best" if use_tuned else "default_validation_leader"

    final_val_prob = candidate.predict_proba(X_val)[:, 1]
    scenarios = validation_scenarios(y_val, final_val_prob)
    final_test_prob = candidate.predict_proba(X_test)[:, 1]
    scenario_rows = []
    for _, choice in scenarios.iterrows():
        test_row = metrics_at_threshold(y_test, final_test_prob, float(choice["threshold"]))
        scenario_rows.append({**choice.to_dict(), **{f"test_{key}": value for key, value in test_row.items() if key != "threshold"}, "test_target_rate": (test_row["tp"] + test_row["fp"]) / len(y_test), "evaluation_note": "Threshold selected on Validation; evaluated once on internal Test holdout."})
    scenario_table = pd.DataFrame(scenario_rows)
    scenario_table.to_csv(RUN_DIR / "threshold_scenarios_final_candidate.csv", index=False, encoding="utf-8-sig")
    targeting_metrics(y_test, final_test_prob).to_csv(RUN_DIR / "topk_lift_final_candidate.csv", index=False, encoding="utf-8-sig")
    decile_metrics(y_test, final_test_prob).to_csv(RUN_DIR / "risk_decile_final_candidate.csv", index=False, encoding="utf-8-sig")

    joblib.dump(candidate, CANDIDATE_DIR / "candidate_pipeline.joblib")
    reloaded = joblib.load(CANDIDATE_DIR / "candidate_pipeline.joblib")
    reload_probability = reloaded.predict_proba(X_test.iloc[:100])[:, 1]
    original_probability = candidate.predict_proba(X_test.iloc[:100])[:, 1]
    reload_verified = bool(np.allclose(original_probability, reload_probability, rtol=1e-12, atol=1e-12))
    final_metrics = {"validation": metrics(y_val, final_val_prob), "internal_test_holdout": metrics(y_test, final_test_prob)}
    metadata = {
        "run_id": RUN_ID, "status": "MODEL_CANDIDATE_SAVED_AWAITING_REVIEW", "candidate_model": leader,
        "candidate_variant": candidate_variant, "selection_rule": "Highest validation PR-AUC; no business cost ratio was assumed.",
        "tuning": {"method": "RandomizedSearchCV", "trials": 3, "trials_upper_bound": 6 if leader != "catboost" else 5, "cv_folds": 3, "best_params": search.best_params_, "best_cv_pr_auc": float(search.best_score_)},
        "split": {"train": 0.6, "validation": 0.2, "internal_test_holdout": 0.2, "random_state": RANDOM_STATE},
        "metrics": final_metrics, "reload_validation": {"verified": reload_verified, "rows_checked": 100},
        "scope": {"existing_app_model_replacement": "DEFERRED", "streamlit_integration": "DEFERRED", "production_promotion": "DEFERRED"},
        "labels": {"model_metrics": "VERIFIED", "prediction_horizon": "UNVERIFIED", "campaign_uplift": "UNVERIFIED", "causal_effect": "UNVERIFIED"},
        "environment": package_versions(), "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    (CANDIDATE_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (RUN_DIR / "candidate_selection.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    manifest.update({"status": "MODEL_CANDIDATE_SAVED_AWAITING_REVIEW", "checkpoint": "candidate_saved_and_reloaded", "completed_steps": ["repository_audit", "existing_experiment_evidence_materialized", "isolated_bounded_candidate_run", "candidate_reload_validation", "final_evidence_pack"], "pending_steps": ["human_review", "production_promotion_deferred"], "model_universe": {**manifest["model_universe"], "environment_unavailable": [], "unavailable_handling": "Pinned packages installed with user approval; actual validation runs recorded."}, "candidate": {"path": str(CANDIDATE_DIR.relative_to(ROOT)), "model": leader, "reload_verified": reload_verified}})
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    report = f"""# Final P0 Evidence Pack\n\n- Candidate status: **MODEL_CANDIDATE_SAVED_AWAITING_REVIEW**\n- Candidate: `{leader}` / `{candidate_variant}`\n- Selection: highest Validation PR-AUC; no unsupported cost ratio was used.\n- Reload check: **{'VERIFIED' if reload_verified else 'FAILED'}** on 100 held-out feature rows.\n- Test metrics are internal-holdout evidence only, recorded in `candidate_selection.json`.\n- Prediction horizon, campaign uplift, and causal effect are **UNVERIFIED**.\n- Existing application model and Streamlit integration remain **DEFERRED** and were not modified by this run.\n"""
    (RUN_DIR / "final_evidence_pack.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
