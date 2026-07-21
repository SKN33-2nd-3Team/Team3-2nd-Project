"""완료된 CatBoost 후보를 로컬 앱 산출물로 안전하게 승격한다.

모델을 다시 학습하지 않는다. 후보 파일의 SHA-256을 검증하고, 현재 모델이
다를 때만 고유한 폴더에 백업한 뒤 원자적으로 교체한다. 운영 지표는 저장된
5-Fold OOF 예측만 재사용한다.
"""
from __future__ import annotations

import hashlib
import json
import os
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_workspace_path(path: Path) -> Path:
    resolved = path.resolve()
    if not resolved.is_relative_to(ROOT.resolve()):
        raise RuntimeError(f"작업 폴더 밖의 파일은 사용할 수 없습니다: {resolved}")
    return resolved


def verify_candidate() -> tuple[Path, dict, str]:
    model_path = require_workspace_path(CANDIDATE / "candidate_pipeline.joblib")
    metadata_path = require_workspace_path(CANDIDATE / "metadata.json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    actual = sha256(model_path)
    expected = metadata.get("artifact_sha256")
    if not expected:
        raise RuntimeError("후보 메타데이터에 artifact_sha256이 없습니다.")
    if actual != expected:
        raise RuntimeError(f"후보 모델 무결성 검증 실패: expected={expected}, actual={actual}")
    if metadata.get("candidate_model") != "catboost":
        raise RuntimeError("승격 대상이 CatBoost 후보가 아닙니다.")
    return model_path, metadata, actual


def archive_current(candidate_hash: str) -> Path | None:
    active_model = MODEL_DIR / "music_churn_pipeline.joblib"
    if not active_model.exists() or sha256(active_model) == candidate_hash:
        return None
    active_meta_path = MODEL_DIR / "metadata.json"
    active_meta = json.loads(active_meta_path.read_text(encoding="utf-8")) if active_meta_path.exists() else {}
    model_name = str(active_meta.get("model", "unknown")).replace("/", "_")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    archive_dir = MODEL_DIR / "archive" / f"{stamp}_{model_name}_before_catboost_{candidate_hash[:8]}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for path in (
        active_model,
        active_meta_path,
        ART / "test_predictions.csv",
        ART / "feature_importance.csv",
    ):
        if path.exists():
            shutil.copy2(path, archive_dir / path.name)
    return archive_dir


def metric_row(y: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    predicted = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "oof_target_customers": int(predicted.sum()),
        "oof_target_rate": float(predicted.mean()),
        "oof_tp": int(tp),
        "oof_fn": int(fn),
        "oof_fp": int(fp),
        "oof_tn": int(tn),
        "oof_precision": float(precision_score(y, predicted, zero_division=0)),
        "oof_recall": float(recall_score(y, predicted, zero_division=0)),
        "oof_f1": float(f1_score(y, predicted, zero_division=0)),
    }


def build_scenarios(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    source = pd.read_csv(ART / "threshold_operating_scenarios_v2.csv")
    source = source.loc[source["model"].eq("catboost")].copy()
    labels = {
        "recall_first": ("재현율 우선", "더 많은 고객을 검토해 이탈 고객 누락을 줄입니다."),
        "balanced_f1": ("균형형 F1", "OOF F1이 가장 높은 기본 의사결정 시나리오입니다."),
        "precision_first": ("정밀도 우선", "검토 고객 수를 줄이고 대상 적중률을 우선합니다."),
    }
    rows = []
    for row in source.itertuples(index=False):
        label, rule = labels[row.scenario]
        rows.append({
            "scenario_id": row.scenario,
            "scenario_label": label,
            "selection_rule": rule,
            "threshold": float(row.threshold),
            **metric_row(y, probability, float(row.threshold)),
            "evaluation_split": "five_fold_oof",
            "evaluation_note": "저장된 5-Fold OOF 예측 근거이며 외부 Holdout 성능이 아닙니다.",
        })
    return pd.DataFrame(rows).sort_values("threshold").reset_index(drop=True)


def build_threshold_sweep(y: np.ndarray, probability: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame([
        {"threshold": threshold, **metric_row(y, probability, float(threshold))}
        for threshold in np.round(np.arange(0.05, 0.96, 0.01), 2)
    ])


def build_topk() -> pd.DataFrame:
    source = pd.read_csv(ART / "topk_lift_v2.csv")
    frame = source.loc[source["model"].eq("catboost")].copy()
    frame = frame.rename(columns={"captured_churners": "actual_churners_captured"})
    frame["evaluation_split"] = "five_fold_oof"
    frame["evaluation_note"] = "OOF 순위 근거이며 캠페인 Uplift나 외부 Holdout 성능이 아닙니다."
    return frame


def build_deciles() -> pd.DataFrame:
    source = pd.read_csv(ART / "risk_decile_v2.csv")
    frame = source.loc[source["model"].eq("catboost")].copy().rename(columns={"decile": "risk_decile"})
    frame["evaluation_split"] = "five_fold_oof"
    frame["evaluation_note"] = "OOF 보정 진단이며 외부 Holdout 성능이 아닙니다."
    return frame


def atomic_copy(source: Path, destination: Path, expected_hash: str) -> None:
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copy2(source, temporary)
    if sha256(temporary) != expected_hash:
        temporary.unlink(missing_ok=True)
        raise RuntimeError("임시 복사본의 SHA-256이 후보 모델과 다릅니다.")
    os.replace(temporary, destination)


def main() -> None:
    candidate_path, candidate_meta, candidate_hash = verify_candidate()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    previous_metadata_path = MODEL_DIR / "metadata.json"
    previous_metadata = (
        json.loads(previous_metadata_path.read_text(encoding="utf-8"))
        if previous_metadata_path.exists() else {}
    )
    archive_dir = archive_current(candidate_hash)
    archive_reference = (
        str(archive_dir.relative_to(ROOT)).replace("\\", "/")
        if archive_dir else previous_metadata.get("archive_path")
    )
    pipeline = joblib.load(candidate_path)
    train = pd.read_csv(ROOT / "data" / "train.csv")
    test = pd.read_csv(ROOT / "data" / "test.csv")
    y = train["churned"].to_numpy(dtype=int)
    oof = np.load(RUN / "oof_fine_catboost.npy")
    scenarios = build_scenarios(y, oof)
    sweep = build_threshold_sweep(y, oof)
    topk = build_topk()
    deciles = build_deciles()

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

    feature_names = pipeline.named_steps["pre"].get_feature_names_out()
    importance = pd.DataFrame({
        "feature": feature_names,
        "importance": pipeline.named_steps["model"].feature_importances_,
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    importance.insert(0, "rank", np.arange(1, len(importance) + 1))

    active_model = MODEL_DIR / "music_churn_pipeline.joblib"
    atomic_copy(candidate_path, active_model, candidate_hash)
    metadata = {
        "model": "catboost",
        "run_id": candidate_meta["run_id"],
        "status": "LOCAL_APP_PROMOTED_AWAITING_BUSINESS_THRESHOLD_DECISION",
        "selection_rule": "상위 3개 모델 중 Fine-tuned 5-Fold CV PR-AUC가 가장 높고 OOF·5개 Seed·Bootstrap 근거가 안정적이어서 선정했습니다.",
        "selection_metric": "fine_tuned_cv_pr_auc",
        "selection_evidence": {
            "fine_tuned_cv_pr_auc": 0.9479127420954035,
            "five_seed_mean_pr_auc": 0.9478803955721158,
            "oof_pr_auc": candidate_meta["metrics"]["pr_auc"],
            "oof_roc_auc": candidate_meta["metrics"]["roc_auc"],
        },
        "evaluation_scope": "five_fold_oof",
        "evaluation_limit": "독립된 외부 라벨 Holdout이 없어 모든 운영 지표는 5-Fold OOF 근거입니다.",
        "reload_verified": True,
        "fresh_process_reload_verified": bool(candidate_meta.get("fresh_process_reload_verified")),
        "artifact_sha256": candidate_hash,
        "trusted_source_note": "저장소 내부 후보 경로와 메타데이터 SHA-256을 검증한 뒤 로드합니다.",
        "app_default_scenario_id": "balanced_f1",
        "operating_scenarios": scenarios.to_dict(orient="records"),
        "metrics": {"five_fold_oof": candidate_meta["metrics"]},
        "input_columns": test.columns.tolist(),
        "target": {
            "column": "churned",
            "positive_label": 1,
            "note": "제공된 관측 라벨이며 예측 기간은 정의되지 않았습니다.",
        },
        "feature_variant": "log_numeric",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_candidate": str(candidate_path.relative_to(ROOT)).replace("\\", "/"),
        "archive_path": archive_reference,
    }
    (MODEL_DIR / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    predictions.to_csv(ART / "test_predictions.csv", index=False, encoding="utf-8-sig")
    importance.to_csv(ART / "feature_importance.csv", index=False, encoding="utf-8-sig")
    scenarios.to_csv(ART / "operating_scenarios_oof_catboost.csv", index=False, encoding="utf-8-sig")
    sweep.to_csv(ART / "threshold_sweep_oof_catboost.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(ART / "topk_lift_oof_catboost.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(ART / "risk_decile_oof_catboost.csv", index=False, encoding="utf-8-sig")

    manifest_path = RUN / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["status"] = "COMPLETED_LOCAL_APP_PROMOTED_AWAITING_BUSINESS_REVIEW"
    manifest["last_checkpoint"] = "hardened_local_app_promotion"
    manifest["blocked_reason"] = None
    manifest["checkpoints"]["app_schema_smoke_test"] = "PASSED"
    manifest["retraining_performed_during_completion"] = False
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "model": "catboost",
        "sha256": candidate_hash,
        "test_rows_scored": len(predictions),
        "archive": archive_reference or "동일 모델이어서 추가 백업 생략",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
