"""Build presentation evidence from completed P0 artifacts without model fitting.

The script reads saved experiment CSVs and metadata only.  It does not load
training data, fit a model, choose after seeing holdout results, or alter the
current operational artifact.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "experiments" / "submission_bounded" / "20260721_submission_bounded_v1"
ARTIFACTS = ROOT / "artifacts"
REPORTS = ROOT / "reports"
FIGURES = ROOT / "figures" / "presentation_v2"


def save(fig: plt.Figure, filename: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / filename, dpi=180, bbox_inches="tight")
    plt.close(fig)


def historical_row(row: pd.Series, stage: str, experiment: str, adopted: str) -> dict:
    return {
        "stage": stage,
        "experiment": experiment,
        "model": str(row.name),
        "adopted": adopted,
        "evaluation_scope": "internal_test_holdout (historical checkpoint)",
        "threshold": row["validation_operating_threshold"],
        "roc_auc": row["test_roc_auc"],
        "pr_auc": row["test_pr_auc"],
        "f1": row["test_operating_f1"],
        "recall": row["test_operating_recall"],
        "precision": row["test_operating_precision"],
        "fn": row["test_fn"],
        "fp": row["test_fp"],
        "metric_context": "Historical threshold used FN:FP=3:1; not used for new LightGBM selection.",
        "provenance": "VERIFIED: artifacts/model_comparison.csv",
    }


def main() -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    existing = pd.read_csv(ARTIFACTS / "model_comparison.csv")
    p0 = pd.read_csv(RUN / "model_comparison_p0.csv")
    metadata = json.loads((ROOT / "models" / "candidates" / "20260721_submission_bounded_v1" / "metadata.json").read_text(encoding="utf-8"))
    scenarios = pd.read_csv(RUN / "threshold_scenarios_final_candidate.csv")
    topk = pd.read_csv(RUN / "topk_lift_final_candidate.csv")
    deciles = pd.read_csv(RUN / "risk_decile_final_candidate.csv")
    features = pd.read_csv(RUN / "feature_experiment_screening.csv")

    # Required reusable business tables: copied from final candidate's held-out evaluation.
    scenarios.to_csv(ARTIFACTS / "threshold_operating_scenarios.csv", index=False, encoding="utf-8-sig")
    topk.to_csv(ARTIFACTS / "topk_lift.csv", index=False, encoding="utf-8-sig")
    deciles.to_csv(ARTIFACTS / "risk_decile.csv", index=False, encoding="utf-8-sig")

    lookup = existing.set_index("model")
    progression = [
        historical_row(lookup.loc["dummy_prior"], "P0 baseline", "Dummy prior baseline", "reference"),
        historical_row(lookup.loc["logistic_raw"], "P0 baseline", "Raw-feature logistic baseline", "reference"),
        historical_row(lookup.loc["logistic_engineered"], "Feature experiment FE-01", "Deterministic ratio/date features", "not adopted as standalone linear winner"),
        historical_row(lookup.loc["decision_tree"], "Model screen", "Constrained decision tree", "not selected"),
        historical_row(lookup.loc["random_forest"], "Model screen", "Random forest", "not selected"),
        historical_row(lookup.loc["gradient_boosting"], "Model screen", "Gradient boosting", "historical benchmark only"),
    ]
    external = p0[p0["model"].isin(["xgboost", "lightgbm", "catboost"])].set_index("model")
    for model, label in [("xgboost", "XGBoost"), ("lightgbm", "LightGBM default"), ("catboost", "CatBoost")]:
        row = external.loc[model]
        progression.append({
            "stage": "P0 8-model screen", "experiment": label, "model": model,
            "adopted": "candidate" if model == "lightgbm" else "not selected",
            "evaluation_scope": "validation only; holdout intentionally not used for screening",
            "threshold": 0.5, "roc_auc": row["validation_roc_auc"], "pr_auc": row["validation_pr_auc"],
            "f1": row["validation_f1_at_0_5"], "recall": row["validation_recall_at_0_5"],
            "precision": row["validation_precision_at_0_5"], "fn": np.nan, "fp": np.nan,
            "metric_context": "Validation default threshold 0.50; FN/FP row counts not persisted.",
            "provenance": "VERIFIED: model_comparison_p0.csv",
        })
    balanced = scenarios.loc[scenarios["scenario_id"] == "balanced_f1"].iloc[0]
    final = metadata["metrics"]
    progression.append({
        "stage": "Final candidate", "experiment": "LightGBM randomized-search candidate", "model": "lightgbm_tuned",
        "adopted": "selected; awaiting human review", "evaluation_scope": "validation selection + one internal holdout evaluation",
        "threshold": balanced["threshold"], "roc_auc": final["internal_test_holdout"]["roc_auc"], "pr_auc": final["internal_test_holdout"]["pr_auc"],
        "f1": balanced["test_f1"], "recall": balanced["test_recall"], "precision": balanced["test_precision"],
        "fn": balanced["test_fn"], "fp": balanced["test_fp"],
        "metric_context": "Balanced-F1 threshold selected on Validation; evaluated once on internal holdout.",
        "provenance": "VERIFIED: candidate metadata + threshold_scenarios_final_candidate.csv",
    })
    progression = pd.DataFrame(progression)
    for metric in ["roc_auc", "pr_auc", "f1", "recall", "precision"]:
        progression[f"delta_vs_previous_{metric}"] = progression[metric].diff()
        baseline = progression.loc[progression["experiment"] == "Raw-feature logistic baseline", metric].iloc[0]
        progression[f"delta_vs_raw_logistic_{metric}"] = progression[metric] - baseline
    progression.to_csv(ARTIFACTS / "performance_progression.csv", index=False, encoding="utf-8-sig")

    # Exactly eight model families in the required screen. Historical models share the split but the raw logistic differs in feature policy.
    required = ["dummy_prior", "logistic_raw", "decision_tree", "random_forest", "gradient_boosting", "xgboost", "lightgbm", "catboost"]
    matrix_rows = []
    for model in required:
        row = p0.loc[p0["model"] == model].iloc[0]
        historical = model in {"dummy_prior", "logistic_raw", "decision_tree", "random_forest", "gradient_boosting"}
        matrix_rows.append({
            "model": model, "validation_pr_auc": row["validation_pr_auc"], "validation_roc_auc": row["validation_roc_auc"],
            "validation_brier": row["validation_brier"], "validation_recall_at_0_5": row.get("validation_recall_at_0_5", np.nan),
            "validation_precision_at_0_5": row.get("validation_precision_at_0_5", np.nan), "selection_status": "SELECTED" if model == "lightgbm" else "NOT_SELECTED",
            "selection_reason": "Highest validation PR-AUC; final tuning did not reduce validation PR-AUC." if model == "lightgbm" else "Lower validation PR-AUC than LightGBM.",
            "split_and_pipeline": "Same stratified 60/20/20 split and Pipeline family" if model != "logistic_raw" else "Same split; raw-feature baseline intentionally omits engineered features.",
            "holdout_available": historical, "tuning_before_after": "LightGBM default→3-trial CV candidate; same recorded validation score." if model == "lightgbm" else "Not persisted / not rerun by design.",
            "provenance": row["provenance"],
        })
    matrix = pd.DataFrame(matrix_rows).sort_values("validation_pr_auc", ascending=False)
    matrix.to_csv(ARTIFACTS / "model_selection_decision_matrix.csv", index=False, encoding="utf-8-sig")

    # Performance progression graph. Scope-aware styling prevents validation and holdout metrics being conflated.
    display = progression[progression["experiment"].isin(["Raw-feature logistic baseline", "Deterministic ratio/date features", "Gradient boosting", "XGBoost", "LightGBM default", "CatBoost", "LightGBM randomized-search candidate"])].copy()
    display["plot_label"] = display["experiment"].map({
        "Raw-feature logistic baseline": "Raw LR", "Deterministic ratio/date features": "Engineered LR",
        "Gradient boosting": "GradientBoost", "XGBoost": "XGBoost", "LightGBM default": "LightGBM",
        "CatBoost": "CatBoost", "LightGBM randomized-search candidate": "LightGBM tuned",
    })
    display["plot_label"] = display["plot_label"] + np.where(display["evaluation_scope"].str.contains("validation only"), " [V]", " [H]")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    axes[0].plot(display["plot_label"], display["pr_auc"], marker="o", label="PR-AUC")
    axes[0].plot(display["plot_label"], display["roc_auc"], marker="o", label="ROC-AUC")
    axes[0].set(title="Performance progression (H=holdout, V=validation)", ylabel="Score", ylim=(0.85, 0.97))
    axes[0].tick_params(axis="x", rotation=25); axes[0].legend(); axes[0].grid(alpha=.25)
    axes[1].bar(display["plot_label"], display["pr_auc"] - display.loc[display["experiment"] == "Raw-feature logistic baseline", "pr_auc"].iloc[0], color="#315A7D")
    axes[1].axhline(0, color="black", linewidth=.8); axes[1].set(title="PR-AUC change vs raw logistic", ylabel="Δ PR-AUC")
    axes[1].tick_params(axis="x", rotation=25); axes[1].grid(axis="y", alpha=.25)
    save(fig, "01_performance_progression.png")

    fig, ax = plt.subplots(figsize=(10, 5.5))
    plotted = matrix.sort_values("validation_pr_auc")
    colors = ["#D95D39" if value == "SELECTED" else "#315A7D" for value in plotted["selection_status"]]
    ax.barh(plotted["model"], plotted["validation_pr_auc"], color=colors)
    for y, value in enumerate(plotted["validation_pr_auc"]): ax.text(value + .0001, y, f"{value:.4f}", va="center", fontsize=9)
    ax.set(title="Eight-model validation PR-AUC comparison", xlabel="Validation PR-AUC", xlim=(.50, .96))
    ax.grid(axis="x", alpha=.25); save(fig, "02_eight_model_comparison.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    labels = scenarios["scenario_id"].str.replace("_", " ")
    x = np.arange(len(scenarios)); width = .36
    ax.bar(x - width/2, scenarios["test_recall"], width, label="Recall", color="#D95D39")
    ax.bar(x + width/2, scenarios["test_precision"], width, label="Precision", color="#315A7D")
    for idx, rate in enumerate(scenarios["test_target_rate"]): ax.text(idx, .03, f"target {rate:.1%}", ha="center", fontsize=8)
    ax.set(title="LightGBM operating scenarios (internal holdout)", ylabel="Score", xticks=x, xticklabels=labels, ylim=(0, 1.08))
    ax.legend(); ax.grid(axis="y", alpha=.25); save(fig, "03_threshold_operating_scenarios.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(topk["top_percent"], topk["capture_rate"] * 100, marker="o", label="Capture rate", color="#D95D39")
    ax.plot(topk["top_percent"], topk["top_percent"], linestyle="--", color="#888888", label="Random targeting")
    ax.set(title="Top-K churner capture (internal holdout)", xlabel="Customers targeted (%)", ylabel="Churners captured (%)", ylim=(0, 100))
    ax.legend(); ax.grid(alpha=.25); save(fig, "04_topk_capture.png")

    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.bar(topk["top_percent"].astype(str) + "%", topk["lift"], color="#315A7D")
    ax.axhline(1, color="#888888", linestyle="--", label="Random baseline")
    ax.set(title="Targeting lift by Top-K (internal holdout)", xlabel="Top-K", ylabel="Lift")
    ax.legend(); ax.grid(axis="y", alpha=.25); save(fig, "05_topk_lift.png")

    plot_deciles = deciles.sort_values("risk_decile")
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.plot(plot_deciles["risk_decile"], plot_deciles["actual_churn_rate"], marker="o", label="Actual churn rate", color="#D95D39")
    ax.plot(plot_deciles["risk_decile"], plot_deciles["mean_predicted_probability"], marker="o", label="Mean predicted probability", color="#315A7D")
    ax.set(title="Risk-decile calibration (internal holdout)", xlabel="Risk decile (10 = highest)", ylabel="Rate / probability", xticks=range(1, 11), ylim=(0, 1))
    ax.legend(); ax.grid(alpha=.25); save(fig, "06_risk_decile.png")

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    values = [matrix.iloc[0]["validation_pr_auc"], matrix.loc[matrix["model"] == "gradient_boosting", "validation_pr_auc"].iloc[0], metadata["tuning"]["best_cv_pr_auc"]]
    labels = ["LightGBM validation", "Historical GradientBoosting validation", "LightGBM 3-fold CV"]
    ax.bar(labels, values, color=["#D95D39", "#315A7D", "#7BAE7F"])
    for idx, value in enumerate(values): ax.text(idx, value + .00025, f"{value:.4f}", ha="center")
    ax.set(title="LightGBM selection evidence", ylabel="PR-AUC", ylim=(.93, .955)); ax.tick_params(axis="x", rotation=12); ax.grid(axis="y", alpha=.25)
    save(fig, "07_lightgbm_selection_evidence.png")

    audit = """# Instructor Requirement Audit\n\n## Audit scope\n\nThis audit inspected the saved candidate metadata, P0 comparison, final evidence pack, every run CSV, manifest, Project Register, and the non-training artifact-contract test. No model fitting or threshold reselection was performed.\n\n| 강사 요구사항 | 관련 산출물 | 실제 근거 | 상태 | 부족한 내용 | 보완 필요 여부 |\n| --- | --- | --- | --- | --- | --- |\n| 성능 개선 과정·논리 | `feature_experiment_screening.csv`, `performance_progression.csv` | Raw logistic → deterministic features와 모델군 비교가 저장됨 | 부분 충족 | 6개 feature 실험 중 실제 수치가 있는 것은 FE-00/01뿐 | 예 — v2 표·결론으로 보완, 추가 학습은 보류 |\n| 조치별 성능 변화 | `performance_progression.csv`, `01_performance_progression.png` | ROC-AUC·PR-AUC·F1·Recall·Precision, 직전/누적 Δ를 기록 | 충족 (v2) | 외부 3개 모델의 FN/FP 행수는 저장되지 않음 | 아니오 — 검증 수치와 범위를 명시 |\n| 8개 모델 비교 | `model_comparison_p0.csv`, `model_selection_decision_matrix.csv`, `02_eight_model_comparison.png` | 8개 필수 모델의 Validation PR-AUC 비교 | 부분 충족 | Raw logistic만 의도적으로 비엔지니어드 feature라 완전 동일 feature 조건은 아님 | 예 — 다음 run에서 전 모델 동일 feature 재비교 |\n| 모델별 튜닝 전후 | candidate metadata, decision matrix | LightGBM default와 3-trial/3-fold 탐색 후보만 확인 | 부분 충족 | 다른 7개 모델의 tuning before/after는 없음 | 예 — 필요 시 validation-only 실험으로 수행 |\n| Threshold 운영 시나리오 | `threshold_operating_scenarios.csv`, `03_threshold_operating_scenarios.png` | Validation에서 threshold 선택 후 internal holdout 1회 평가 | 충족 (v2) | 실제 캠페인 비용·용량은 없음 | 예 — 사업자가 시나리오 선택 |\n| Top-K / Lift / Risk Decile | `topk_lift.csv`, `risk_decile.csv`, `04`~`06` 그래프 | final LightGBM internal holdout 결과 재사용 | 충족 (v2) | 외부 실제 캠페인 효과는 없음 | 아니오 — Uplift와 구분 표기 |\n| LightGBM 최종 선정 근거 | decision matrix, `07_lightgbm_selection_evidence.png` | 8개 중 Validation PR-AUC 최고(0.9457), 3-fold CV PR-AUC 0.9477, holdout PR-AUC 0.9486 | 부분 충족 | Recall/FN trade-off의 사업 우선순위는 미승인 | 예 — 운영 시나리오를 사람이 선택 |\n| 최고 단일 점수와 최종 선정의 차이 | decision matrix | 동일 모델 LightGBM이 최고 Validation PR-AUC이자 최종 후보 | 충족 | 기술 점수 외 사업 승인 없음 | 예 — 배포 전 승인 |\n| Validation–Holdout 일반화 | candidate metadata, progression | Validation 0.9457 → holdout 0.9486 PR-AUC | 충족 | 시간 기반 미래 검증·prediction horizon 없음 | 예 — 새 데이터 수집 후 검증 |\n| 발표용 그래프와 원본 CSV | `figures/presentation_v2/`, `artifacts/*.csv` | 7 PNG와 대응 CSV를 생성 | 충족 (v2) | 원본 run에는 그래프 없음 | 아니오 |\n\n## Missing evidence and retraining decision\n\nNo additional model training is needed to create the v2 evidence pack. Bootstrap confidence intervals require saved row-level candidate predictions (not present); seed stability and segment error analysis require new scoring or refits. These are deferred so the original holdout remains untouched.\n\n## Guardrails\n\n- **VERIFIED:** listed validation/internal-holdout metrics and reload check.\n- **UNVERIFIED:** prediction horizon, campaign uplift, causal effect, and production impact.\n- The historical GradientBoosting cost ratio is not used in the LightGBM selection.\n"""
    (REPORTS / "instructor_requirement_audit.md").write_text(audit, encoding="utf-8")

    pack = """# Presentation Evidence Pack v2\n\n## Recommended-for-review candidate\n\n**LightGBM** is the technical candidate because it had the highest saved validation PR-AUC among the eight required models (0.9457), retained that score after the bounded 3-trial/3-fold search, and generalized to an internal-holdout PR-AUC of 0.9486. The top single-score model and selected model are the same.\n\n## Operating decision, not a claim of business impact\n\nChoose a threshold only after the campaign owner sets contact capacity and the false-negative/false-positive trade-off. The saved scenarios show the consequences: recall-first (0.30) targets 61.9% and achieves 95.1% recall; balanced-F1 (0.35) targets 61.0%, with 94.5% recall, 79.5% precision, FN 705 and FP 3,131; precision-first (0.77) targets 38.6% and achieves 95.5% precision.\n\n## Presentation file map\n\n| Slide claim | Graph | Source CSV |\n| --- | --- | --- |\n| Baseline to candidate progression | `01_performance_progression.png` | `artifacts/performance_progression.csv` |\n| Eight-model screen | `02_eight_model_comparison.png` | `artifacts/model_selection_decision_matrix.csv` |\n| Threshold choices | `03_threshold_operating_scenarios.png` | `artifacts/threshold_operating_scenarios.csv` |\n| Contact capacity / capture | `04_topk_capture.png` | `artifacts/topk_lift.csv` |\n| Targeting lift | `05_topk_lift.png` | `artifacts/topk_lift.csv` |\n| Risk ranking and calibration | `06_risk_decile.png` | `artifacts/risk_decile.csv` |\n| Selection rationale | `07_lightgbm_selection_evidence.png` | `artifacts/model_selection_decision_matrix.csv` |\n\n## Limits\n\nMetrics are internal validation/holdout evidence, not campaign uplift, causal impact, or a verified prediction horizon. Candidate status remains `MODEL_CANDIDATE_SAVED_AWAITING_REVIEW`; no current app model was replaced.\n"""
    (REPORTS / "presentation_evidence_pack_v2.md").write_text(pack, encoding="utf-8")


if __name__ == "__main__":
    main()
