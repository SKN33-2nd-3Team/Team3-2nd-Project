"""Assemble the local PlaylistPro submission package without retraining models."""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import nbformat as nbf
import pandas as pd
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "submission_package" / "PlaylistPro_final_submission_20260722"


def copy_file(source: str | Path, target: str | Path) -> None:
    source_path = ROOT / source if not isinstance(source, Path) or not source.is_absolute() else source
    target_path = PACKAGE / target
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, target_path)


def copy_tree(source: str, target: str, pattern: str = "*") -> None:
    for path in (ROOT / source).rglob(pattern):
        if path.is_file() and "__pycache__" not in path.parts:
            copy_file(path, Path(target) / path.relative_to(ROOT / source))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def make_data_check_notebook() -> nbf.NotebookNode:
    cells = [
        markdown("""
# 01. 데이터 품질 점검

`eda 예시.ipynb`의 순서인 **구조 이해 → 품질 진단 → 처리 결정 → 해석**을 PlaylistPro 데이터에 적용합니다.

> 분석 단위는 고객 1명 = 1행이며, Target은 `churned`(0=유지, 1=이탈)입니다. 관측 기준일과 결과 기간은 제공되지 않아 미래 30일 이탈로 해석하지 않습니다.
"""),
        markdown("""
## 1. 데이터 로드와 분석 목적

- 원본 출처·라이선스·합성 판정 근거는 `docs/data_card.md`에서 관리합니다.
- 이 Notebook은 제출 패키지의 `data/processed/`에 복사된 최종 고객 단위 테이블을 읽습니다.
- 원본 값을 수정하지 않고 품질 문제를 계수합니다.
"""),
        code("""
from pathlib import Path
import pandas as pd

ROOT = Path.cwd()
train = pd.read_csv(ROOT / "data/processed/train.csv")
test = pd.read_csv(ROOT / "data/processed/test.csv")
print("train:", train.shape, "test:", test.shape)
train.head(3)
"""),
        markdown("""
## 2. 데이터 구조 및 변수 이해

행·열 수, 자료형, 고객 키의 유일성, Train/Test 고객 교집합을 확인합니다. 식별자 `customer_id`는 모델 입력에서 제외합니다.
"""),
        code("""
structure = pd.DataFrame({
    "dataset": ["train", "test"],
    "rows": [len(train), len(test)],
    "columns": [train.shape[1], test.shape[1]],
    "customer_id_unique": [train.customer_id.is_unique, test.customer_id.is_unique],
    "missing_cells": [int(train.isna().sum().sum()), int(test.isna().sum().sum())],
    "duplicate_rows": [int(train.duplicated().sum()), int(test.duplicated().sum())],
})
display(structure)
print("Train/Test customer_id 교집합:", len(set(train.customer_id) & set(test.customer_id)))
display(train.dtypes.rename("dtype").to_frame())
"""),
        markdown("""
**해석:** 고객 키 중복·Train/Test 교집합·결측·완전 중복은 없습니다. 따라서 행 삭제나 임의 대체보다, Pipeline 안에서 미등록 범주와 향후 결측에 대비한 안전한 전처리를 유지하는 것이 적절합니다.
"""),
        markdown("""
## 3. Target과 클래스 비율

Accuracy만으로 판단하지 않고 Recall, Precision, F1, PR-AUC를 함께 사용하는 이유를 확인합니다.
"""),
        code("""
target = train["churned"].value_counts().sort_index().rename_axis("churned").to_frame("customers")
target["rate"] = target["customers"] / len(train)
display(target)
"""),
        markdown("""
**해석:** 관측 이탈률은 약 51.3%로 극단적 불균형은 아닙니다. SMOTE를 기본 적용하지 않고, 동일 Fold의 PR-AUC와 Threshold별 오류량을 비교합니다.
"""),
        markdown("""
## 4. 결측·중복·이상값과 데이터 계약 위반

통계적 극단값을 일괄 삭제하지 않고, 도메인상 불가능한 조합을 별도로 계수합니다.
"""),
        code("""
quality = pd.DataFrame([
    {"check": "missing cells", "count": int(train.isna().sum().sum()), "action": "현재 없음; Pipeline 대치 유지"},
    {"check": "duplicate rows", "count": int(train.duplicated().sum()), "action": "삭제 없음"},
    {"check": "unique songs > played songs", "count": int((train.weekly_unique_songs > train.weekly_songs_played).sum()), "action": "수집 규칙 경고; 원본 보존"},
    {"check": "shared playlists > created", "count": int((train.num_shared_playlists > train.num_playlists_created).sum()), "action": "수집 규칙 경고; 원본 보존"},
])
quality["rate"] = quality["count"] / len(train)
display(quality)
"""),
        markdown("""
**해석:** 고유 곡 수와 공유 플레이리스트에는 현실적으로 불가능한 조합이 각각 약 29.6%, 24.6% 존재합니다. 이를 임의 수정하면 생성 규칙을 새로 주입할 수 있으므로 원본을 보존하고 실제 서비스 적용 한계로 기록합니다.
"""),
        markdown("""
## 5. 전처리 결정

| 항목 | 적용 | 근거 |
|---|---|---|
| 식별자 | `customer_id` 제외 | 일반화 불가능한 키 |
| 날짜 대용치 | `signup_date` → 경과일 | 실제 날짜가 아닌 음수 정수 |
| 결측 | 수치 median, 범주 최빈값 | 새 입력의 결측 대비 |
| 범주형 | One-Hot, unknown 허용 | 새 범주에서 추론 중단 방지 |
| 수치 변환 | `log_numeric` 채택 | 동일 Fold Logistic과 CatBoost 검증에서 근소하게 우수 |
| 불균형 | SMOTE 미적용 | 클래스 비율이 극단적이지 않고 Fold 내부 비교를 우선 |
| 누수 방지 | 모든 변환을 Pipeline/Fold 내부에서 fit | Validation 정보의 학습 유입 방지 |
"""),
        markdown("""
## 6. 최종 요약

1. 고객 125,000명의 Train과 75,000명의 무라벨 Test를 사용합니다.
2. 결측·중복·고객 교집합은 없지만, 도메인상 불가능한 조합이 존재합니다.
3. 규칙 기반 합성으로 판단되므로 높은 점수를 실서비스 성능으로 일반화하지 않습니다.
4. 데이터 수정 대신 Train Fold 내부 Pipeline과 검증 경계를 유지합니다.
5. 상세 근거는 `reports/preprocessing_report.md`와 `docs/data_card.md`에 연결됩니다.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def make_eda_notebook() -> nbf.NotebookNode:
    cells = [
        markdown("""
# 02. 탐색적 데이터 분석과 전처리 연결

예시 Notebook의 원칙대로 그래프를 나열하지 않고, 각 결과를 **관찰 → 다음 처리·실험 → 한계**로 연결합니다.
"""),
        code("""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path.cwd()
train = pd.read_csv(ROOT / "data/processed/train.csv")
plt.style.use("seaborn-v0_8-whitegrid")
"""),
        markdown("""
## 1. Target 분포

관측 라벨의 규모를 확인해 지표와 불균형 처리 방향을 결정합니다.
"""),
        code("""
counts = train.churned.value_counts().sort_index()
ax = counts.rename({0: "유지", 1: "이탈"}).plot.bar(color=["#315d78", "#e4573d"], rot=0, title="관측 Target 분포")
ax.set_ylabel("고객 수")
for patch in ax.patches:
    ax.annotate(f"{int(patch.get_height()):,}", (patch.get_x()+patch.get_width()/2, patch.get_height()), ha="center", va="bottom")
plt.show()
"""),
        markdown("""
**관찰:** 이탈이 51.3%로 과반이지만 극단적 희소 클래스는 아닙니다. **결정:** SMOTE보다 동일 Fold의 PR-AUC·Recall·오류량 비교를 우선합니다. **한계:** 제공 라벨의 관측 기간이 없어 실제 미래 이탈률로 해석하지 않습니다.
"""),
        markdown("""
## 2. 구독 유형과 문의 수준

운영에서 행동으로 연결할 수 있는 범주형 신호를 확인합니다.
"""),
        code("""
rates = (train.groupby(["subscription_type", "customer_service_inquiries"])["churned"]
         .mean().unstack().sort_index())
display(rates.style.format("{:.1%}"))
rates.plot.bar(figsize=(9, 4), color=["#2f7d62", "#e8a23a", "#e4573d"], title="구독 유형 × 문의 수준 관측 이탈률")
plt.ylabel("관측 이탈률")
plt.xticks(rotation=0)
plt.show()
"""),
        markdown("""
**관찰:** Free와 높은 문의 수준에서 관측 이탈률이 높습니다. **결정:** 구독 옵션 안내와 문의 해결을 행동 후보에 연결하되, 모델과 행동 규칙은 분리합니다. **한계:** 합성 데이터의 연관성이므로 행동 효과나 인과 원인으로 단정하지 않습니다.
"""),
        markdown("""
## 3. 주간 청취시간의 단계형 패턴

연속형 변수의 비선형성을 확인합니다.
"""),
        code("""
bins = [-1, 5, 10, 40, float("inf")]
labels = ["≤5h", "5-10h", "10-40h", ">40h"]
hours = train.assign(hours_band=pd.cut(train.weekly_hours, bins=bins, labels=labels))
hour_rate = hours.groupby("hours_band", observed=True).churned.agg(["size", "mean"])
display(hour_rate.style.format({"mean": "{:.1%}"}))
hour_rate["mean"].plot.bar(color="#e4573d", rot=0, title="주간 청취시간 구간별 관측 이탈률")
plt.ylabel("관측 이탈률")
plt.show()
"""),
        markdown("""
**관찰:** 5·10·40시간 경계에서 계단형 차이가 나타납니다. **결정:** 선형 가정만 두지 않고 Tree Boosting과 log 변환 실험을 비교합니다. **한계:** 매끄러운 실사용 분포보다 합성 생성 규칙에 가까워 실서비스 재현을 보장하지 않습니다.
"""),
        markdown("""
## 4. 신호와 노이즈의 구분

수치형 상관계수만으로 Feature를 제거하지 않고, 타깃 연관성과 예측 시점 가용성을 함께 봅니다.
"""),
        code("""
numeric = train.select_dtypes("number").drop(columns=["customer_id"])
corr = numeric.corr(numeric_only=True)["churned"].drop("churned").sort_values(key=abs, ascending=False)
display(corr.to_frame("target_correlation").head(12).style.format("{:.4f}"))
"""),
        markdown("""
**관찰:** 선형 상관은 전반적으로 작지만 구간형 신호가 존재합니다. **결정:** 상관계수만으로 제거하지 않고 동일 Fold Feature 실험으로 채택 여부를 결정합니다. **한계:** Feature Importance도 인과성이 아니라 예측 기여도입니다.
"""),
        markdown("""
## 5. EDA에서 모델링·운영으로 이어진 결정

| EDA 관찰 | 모델링 결정 | 운영 화면 연결 |
|---|---|---|
| 청취시간의 비선형 단계 | Logistic 기준선 + Boosting 비교 | 활동 저하 신호와 콘텐츠 재활성화 후보 |
| Free·Student 요금제 차이 | 범주형 처리와 CatBoost 후보 포함 | 구독 옵션 안내 |
| 높은 문의 수준 | 행동 가능 신호로 유지 | 문의 해결 우선 |
| 스킵률·일시정지 문턱 | Threshold형 Tree 모델 비교 | 추천 피드백·복귀 안내 |
| 합성·논리 위반 | 외부 Holdout과 인과 주장 제한 | KPI를 향후 검증 항목으로 표시 |

최종 결론은 `reports/preprocessing_report.md`, 모델 성능은 `reports/training_report.md`에서 확인합니다.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def make_model_notebook() -> nbf.NotebookNode:
    cells = [
        markdown("""
# 03. 모델 실험 결과 재현·감사

이 Notebook은 이미 완료된 `20260721_full_fair_v1` Run의 저장 결과를 읽어 비교합니다. **모델을 다시 학습하지 않으며**, 동일 Fold·동일 Feature 조건과 최종 모델 재로딩을 검증합니다.
"""),
        code("""
from pathlib import Path
import json
import joblib
import pandas as pd

ROOT = Path.cwd()
comparison = pd.read_csv(ROOT / "artifacts/model_comparison_fair.csv")
fine = pd.read_csv(ROOT / "artifacts/top3_fine_tuning_summary.csv")
scenarios = pd.read_csv(ROOT / "artifacts/threshold_operating_scenarios_v2.csv")
metadata = json.loads((ROOT / "artifacts/model_metadata.json").read_text(encoding="utf-8"))
"""),
        markdown("""
## 1. 동일 조건 8개 모델 비교

1차 지표는 PR-AUC, 보조 지표는 ROC-AUC·F1·Recall·Precision·FN·FP·Brier입니다.
"""),
        code("""
cols = ["model", "pr_auc", "roc_auc", "f1", "recall", "precision", "brier", "seconds"]
display(comparison[cols].sort_values("pr_auc", ascending=False).style.format({c: "{:.6f}" for c in cols[1:7]}))
ax = comparison.sort_values("pr_auc").plot.barh(x="model", y="pr_auc", color="#315d78", legend=False, title="8개 모델 동일 Fold PR-AUC")
ax.set_xlim(0.5, 0.96)
"""),
        markdown("""
**해석:** Dummy를 제외한 모든 모델이 기준선을 넘었고, CatBoost·LightGBM·XGBoost가 상위권입니다. 기본 Threshold의 F1·Recall이 가장 높다는 이유만으로 모델을 확정하지 않고 순위 품질과 안정성을 추가 검증했습니다.
"""),
        markdown("""
## 2. 상위 3개 정밀 탐색
"""),
        code("""
display(fine.sort_values("best_cv_pr_auc", ascending=False).style.format({
    "best_cv_pr_auc": "{:.6f}", "pr_auc": "{:.6f}", "roc_auc": "{:.6f}",
    "f1": "{:.6f}", "recall": "{:.6f}", "precision": "{:.6f}", "brier": "{:.6f}",
}))
"""),
        markdown("""
**해석:** CatBoost가 Fine-tuned CV PR-AUC 0.947913으로 근소한 1위입니다. Bootstrap 구간은 겹치므로 절대적 우월성이 아니라, 사전 선정 규칙에서의 기술 후보로 표현합니다.
"""),
        markdown("""
## 3. Threshold 운영 시나리오

Threshold는 모델 선택과 분리된 운영 결정입니다.
"""),
        code("""
cat = scenarios.query("model == 'catboost'")[["scenario", "threshold", "f1", "recall", "precision", "fn", "fp"]]
display(cat.style.format({"threshold": "{:.2f}", "f1": "{:.4f}", "recall": "{:.4f}", "precision": "{:.4f}", "fn": "{:,.0f}", "fp": "{:,.0f}"}))
"""),
        markdown("""
**해석:** 균형형 0.35는 Recall 0.9444와 Precision 0.7959를 제공하지만, 재현율 우선과 정밀도 우선은 접촉 규모·FN·FP를 크게 바꿉니다. 실제 운영안은 사업 담당자가 선택해야 합니다.
"""),
        markdown("""
## 4. 저장 모델 새 입력 예측
"""),
        code("""
model = joblib.load(ROOT / "models/churn_pipeline.joblib")
sample = pd.read_csv(ROOT / "data/processed/test.csv").head(1)
probability = float(model.predict_proba(sample)[0, 1])
print({"customer_id": int(sample.iloc[0].customer_id), "churn_probability": probability, "threshold": 0.35})
"""),
        markdown("""
**검증 결론:** 저장 Pipeline이 새 프로세스에서 로드되고 원본 스키마의 신규 고객을 예측합니다. 제공 Test에는 정답 라벨이 없으므로 이 확률은 추론 결과이며 성능 평가가 아닙니다.
"""),
        markdown("""
## 5. 최종 선정과 한계

- 최종 기술 후보: **CatBoost**
- 선정 근거: Fine-tuned CV PR-AUC, OOF PR-AUC, 5개 Seed 평균, Bootstrap 안정성
- 기본 화면 시나리오: Threshold 0.35(균형형 F1), 사업 확정값 아님
- 한계: 외부 라벨 Holdout·예측 기간·실제 캠페인 Uplift·ROI 미검증
"""),
    ]
    return nbf.v4.new_notebook(cells=cells, metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}})


def execute_notebook(notebook: nbf.NotebookNode, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    client = NotebookClient(notebook, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(PACKAGE)}})
    client.execute()
    nbf.write(notebook, path)


def build_canonical_artifacts() -> None:
    metadata_source = json.loads((ROOT / "artifacts/model/metadata.json").read_text(encoding="utf-8"))
    (PACKAGE / "artifacts/model_metadata.json").write_text(
        json.dumps(metadata_source, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    feature_schema = {
        "features": metadata_source["input_columns"],
        "target": metadata_source["target"]["column"],
        "positive_class": metadata_source["target"]["positive_label"],
        "thresholds": {row["scenario_id"]: row["threshold"] for row in metadata_source["operating_scenarios"]},
        "default_scenario": metadata_source["app_default_scenario_id"],
        "schema_note": "customer_id는 입력 계약에는 포함되지만 모델 Feature 변환 단계에서 제외됩니다.",
    }
    (PACKAGE / "artifacts/feature_schema.json").write_text(
        json.dumps(feature_schema, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    metrics = metadata_source["metrics"]["five_fold_oof"]
    metrics_row = {
        "model": metadata_source["model"],
        "evaluation_split": metadata_source["evaluation_scope"],
        "threshold": 0.5,
        **metrics,
        "evaluation_limit": metadata_source["evaluation_limit"],
    }
    pd.DataFrame([metrics_row]).to_csv(PACKAGE / "artifacts/metrics.csv", index=False)


def assemble_files() -> None:
    PACKAGE.mkdir(parents=True, exist_ok=True)
    for directory in [
        "app", "src", "tests", "docs", "reports", "notebooks", "models", "artifacts",
        "data/processed", "figures/preprocessing", "figures/training", "figures/presentation",
        "assets/screenshots", "presentation", "tools", "output/pdf",
    ]:
        (PACKAGE / directory).mkdir(parents=True, exist_ok=True)

    copy_file("requirements.txt", "requirements.txt")
    copy_file("app/streamlit_app.py", "app/streamlit_app.py")
    package_app = PACKAGE / "app/streamlit_app.py"
    app_source = package_app.read_text(encoding="utf-8")
    app_source = app_source.replace('DATA_DIR = ROOT / "data"', 'DATA_DIR = ROOT / "data" / "processed"', 1)
    app_source = app_source.replace(
        'MODEL_PATH = ARTIFACT_DIR / "model" / "music_churn_pipeline.joblib"',
        'MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"',
        1,
    )
    package_app.write_text(app_source, encoding="utf-8")
    copy_tree("src", "src", "*.py")
    copy_tree("tests", "tests", "*.py")
    copy_tree("docs", "docs", "*.md")

    copy_file("data/train.csv", "data/processed/train.csv")
    copy_file("data/test.csv", "data/processed/test.csv")
    # Keep compatibility paths used by the repository's original tests while
    # exposing the cleaner submission paths documented in README.md.
    copy_file("data/train.csv", "data/train.csv")
    copy_file("data/test.csv", "data/test.csv")
    copy_file("artifacts/model/music_churn_pipeline.joblib", "models/churn_pipeline.joblib")
    copy_file("artifacts/model/music_churn_pipeline.joblib", "artifacts/model/music_churn_pipeline.joblib")
    copy_file("artifacts/model/metadata.json", "artifacts/model/metadata.json")
    copy_tree("models/candidates/20260721_full_fair_v1", "models/candidates/20260721_full_fair_v1", "*.json")
    copy_file(
        "models/candidates/20260721_full_fair_v1/candidate_pipeline.joblib",
        "models/candidates/20260721_full_fair_v1/candidate_pipeline.joblib",
    )

    app_artifacts = [
        "test_predictions.csv", "feature_importance.csv", "topk_lift_oof_catboost.csv",
        "risk_decile_oof_catboost.csv", "operating_scenarios_oof_catboost.csv",
        "threshold_sweep_oof_catboost.csv", "model_comparison_fair.csv",
        "top3_fine_tuning_summary.csv", "insight_preprocessing_register.csv",
        "preprocessing_experiment_results.csv", "random_search_summary.csv",
        "current_insight_inventory.csv", "bootstrap_confidence_intervals.csv",
        "calibration_summary.csv", "equal_contact_comparison.csv",
        "equal_recall_comparison.csv", "seed_stability_summary.csv",
        "threshold_operating_scenarios_v2.csv", "topk_lift_v2.csv",
        "risk_decile_v2.csv", "final_model_selection_matrix.csv",
    ]
    for name in app_artifacts:
        copy_file(Path("artifacts") / name, Path("artifacts") / name)

    copy_tree("artifacts/presentation_v3", "artifacts/presentation_v3", "*.csv")
    copy_tree("figures/presentation_v3", "figures/presentation", "*.png")
    copy_file("figures/presentation_v3/figure_inventory.csv", "figures/presentation/figure_inventory.csv")
    copy_tree("figures/presentation_v3", "figures/presentation_v3", "*.png")
    copy_file("figures/presentation_v3/figure_inventory.csv", "figures/presentation_v3/figure_inventory.csv")

    preprocessing_figures = {
        "artifacts/eda/target_distribution.png": "01_target_distribution.png",
        "artifacts/eda/categorical_subscription_type.png": "02_subscription_type.png",
        "artifacts/eda/categorical_customer_service_inquiries.png": "03_service_inquiries.png",
        "artifacts/eda_insight/02_weekly_hours_steps.png": "04_weekly_hours_steps.png",
        "artifacts/eda_insight/03_signal_vs_noise.png": "05_signal_vs_noise.png",
        "artifacts/eda_insight/04_plan_x_inquiry_heatmap.png": "06_plan_inquiry_heatmap.png",
        "artifacts/eda_insight/06_model_ceiling.png": "07_model_ceiling.png",
    }
    for source, name in preprocessing_figures.items():
        copy_file(source, Path("figures/preprocessing") / name)

    training_figures = [
        "01_performance_progression.png", "03_adopted_rejected_experiments.png",
        "04_eight_model_fair_comparison.png", "06_top3_fine_tuning_before_after.png",
        "08_bootstrap_confidence_intervals.png", "10_threshold_precision_recall_f1.png",
        "11_threshold_volume_fn_fp.png", "14_topk_capture_lift.png",
        "15_risk_decile.png", "16_final_confusion_matrix.png",
        "17_final_feature_importance.png", "18_segment_error_analysis.png",
    ]
    for name in training_figures:
        copy_file(Path("figures/presentation_v3") / name, Path("figures/training") / name)

    build_canonical_artifacts()
    execute_notebook(make_data_check_notebook(), PACKAGE / "notebooks/01_data_check.ipynb")
    execute_notebook(make_eda_notebook(), PACKAGE / "notebooks/02_eda.ipynb")
    execute_notebook(make_model_notebook(), PACKAGE / "notebooks/03_model_experiments.ipynb")

    copy_file("scripts/build_final_submission.py", "tools/build_final_submission.py")


def write_inventory() -> None:
    rows = []
    for path in sorted(PACKAGE.rglob("*")):
        if path.is_file() and path.name != "submission_manifest.json":
            rows.append({
                "path": path.relative_to(PACKAGE).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    (PACKAGE / "submission_manifest.json").write_text(
        json.dumps({"package": PACKAGE.name, "files": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    assemble_files()
    write_inventory()
    print(PACKAGE)
