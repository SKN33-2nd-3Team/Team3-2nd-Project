"""PlaylistPro 고객 이탈 위험 의사결정 지원 대시보드."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model" / "music_churn_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model" / "metadata.json"
PREDICTIONS_PATH = ARTIFACT_DIR / "test_predictions.csv"
IMPORTANCE_PATH = ARTIFACT_DIR / "feature_importance.csv"
TARGETING_PATH = ARTIFACT_DIR / "topk_lift_oof_catboost.csv"
DECILE_PATH = ARTIFACT_DIR / "risk_decile_oof_catboost.csv"
SCENARIOS_PATH = ARTIFACT_DIR / "operating_scenarios_oof_catboost.csv"
FAIR_COMPARISON_PATH = ARTIFACT_DIR / "model_comparison_fair.csv"
FINE_TUNING_PATH = ARTIFACT_DIR / "top3_fine_tuning_summary.csv"
PREPROCESSING_PATH = ARTIFACT_DIR / "insight_preprocessing_register.csv"
INSIGHT_PATH = ARTIFACT_DIR / "current_insight_inventory.csv"

st.set_page_config(page_title="PlaylistPro 인사이트", page_icon="🎵", layout="wide")
st.markdown(
    """
    <style>
    .stApp {background:#f7f8fa;color:#172033}
    .hero {background:linear-gradient(115deg,#17324a,#315d78);color:white;border-radius:18px;padding:26px 30px;margin-bottom:20px}
    .hero h1 {color:white;margin:0 0 8px 0}.hero p {color:#dce8f1;margin:0;line-height:1.6}
    .notice {background:#fff8f5;border:1px solid #fed6cc;border-radius:12px;padding:14px 16px;line-height:1.55}
    [data-testid="stSidebar"] {background:#102235}
    [data-testid="stSidebar"] * {color:#eef5fa}
    </style>
    """,
    unsafe_allow_html=True,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@st.cache_data(show_spinner=False)
def load_tables():
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "train": train,
        "test": test,
        "metadata": metadata,
        "predictions": pd.read_csv(PREDICTIONS_PATH),
        "importance": pd.read_csv(IMPORTANCE_PATH),
        "targeting": pd.read_csv(TARGETING_PATH),
        "deciles": pd.read_csv(DECILE_PATH),
        "scenarios": pd.read_csv(SCENARIOS_PATH),
        "comparison": pd.read_csv(FAIR_COMPARISON_PATH),
        "fine_tuning": pd.read_csv(FINE_TUNING_PATH),
        "preprocessing": pd.read_csv(PREPROCESSING_PATH),
        "insights": pd.read_csv(INSIGHT_PATH),
    }


@st.cache_resource(show_spinner=False)
def load_model(expected_hash: str):
    model_path = MODEL_PATH.resolve()
    if not model_path.is_relative_to(ROOT.resolve()):
        raise RuntimeError("허용된 프로젝트 폴더 밖의 모델은 로드할 수 없습니다.")
    actual_hash = sha256(model_path)
    if not expected_hash or actual_hash != expected_hash:
        raise RuntimeError("모델 SHA-256이 메타데이터와 다릅니다. 파일 무결성을 확인해 주세요.")
    # joblib/pickle은 코드 실행이 가능한 형식이므로 저장소 내부에서 검증된 파일만 로드한다.
    return joblib.load(model_path)


def load_workspace():
    required = [
        DATA_DIR / "train.csv", DATA_DIR / "test.csv", MODEL_PATH, METADATA_PATH,
        PREDICTIONS_PATH, IMPORTANCE_PATH, TARGETING_PATH, DECILE_PATH,
        SCENARIOS_PATH, FAIR_COMPARISON_PATH, FINE_TUNING_PATH,
        PREPROCESSING_PATH, INSIGHT_PATH,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("필수 파일이 없습니다: " + ", ".join(missing))
        st.stop()
    tables = load_tables()
    model = load_model(str(tables["metadata"].get("artifact_sha256", "")))
    return tables, model


def section_header(title: str, description: str) -> None:
    st.title(title)
    st.caption(description)


def plot_style(fig):
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
    )
    return fig


def selected_scenario(scenarios: pd.DataFrame, scenario_id: str) -> pd.Series:
    found = scenarios.loc[scenarios["scenario_id"].eq(scenario_id)]
    return found.iloc[0] if not found.empty else scenarios.iloc[0]


def model_boundary() -> None:
    st.markdown(
        '<div class="notice"><b>해석 범위</b><br>'
        '모델 성능과 운영 표는 저장된 5-Fold OOF 예측 근거입니다. 독립 외부 Holdout, 향후 이탈 기간, '
        '캠페인 Uplift·ROI·인과효과를 검증한 결과가 아닙니다. 실제 CRM 발송과 고객 접촉은 실행하지 않습니다.</div>',
        unsafe_allow_html=True,
    )


def project_summary_page(tables: dict) -> None:
    train, test, metadata, targeting = (
        tables["train"], tables["test"], tables["metadata"], tables["targeting"]
    )
    metrics = metadata["metrics"]["five_fold_oof"]
    st.markdown(
        '<div class="hero"><h1>고객 이탈 위험을 순위화하고, 검토 범위를 선택합니다.</h1>'
        '<p>최종 CatBoost 모델과 고객 행동 인사이트를 한 화면에 통합한 의사결정 지원 도구입니다. '
        '모델은 고객 조치를 자동 실행하지 않습니다.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("학습 고객", f"{len(train):,}명")
    c2.metric("점수 산출 고객", f"{len(test):,}명", "정답 라벨 없음")
    c3.metric("최종 모델", "CatBoost")
    c4.metric("OOF PR-AUC", f"{metrics['pr_auc']:.4f}")
    top30 = targeting.loc[targeting["top_percent"].eq(30)].iloc[0]
    st.info(
        f"OOF 위험 점수 상위 30%({int(top30.target_customers):,}명)는 관측된 이탈 라벨의 "
        f"{float(top30.capture_rate):.1%}를 포함했습니다. 이는 검토 용량 계획 근거이며 캠페인 효과가 아닙니다."
    )
    model_boundary()


def customer_insights_page(tables: dict) -> None:
    train = tables["train"]
    section_header("고객 인사이트", "관측된 연관성을 서비스 개선 가설로 정리합니다.")
    weekly = train.groupby(pd.qcut(train["weekly_hours"], 4, duplicates="drop"), observed=True)["churned"].mean()
    skip = train.groupby(pd.qcut(train["song_skip_rate"], 4, duplicates="drop"), observed=True)["churned"].mean()
    subscription = train.groupby("subscription_type")["churned"].mean().sort_values(ascending=False)
    inquiries = train.groupby("customer_service_inquiries")["churned"].mean().sort_values(ascending=False)
    cards = [
        ("청취 활동", f"주간 청취시간 사분위 간 이탈률 차이 {weekly.max()-weekly.min():.1%}", "활동 저하 고객을 진단 후보로 검토"),
        ("스킵 행동", f"스킵률 사분위 간 이탈률 차이 {skip.max()-skip.min():.1%}", "추천 품질 개선 실험 가설"),
        ("구독 유형", f"최고 관측 이탈률: {subscription.index[0]} {subscription.iloc[0]:.1%}", "전환 제안 효과는 실험으로 확인"),
        ("고객 문의", f"최고 관측 이탈률: {inquiries.index[0]} {inquiries.iloc[0]:.1%}", "문의 해결 품질·속도 개선 가설"),
    ]
    for column, (title, observation, action) in zip(st.columns(4), cards):
        with column:
            st.subheader(title)
            st.write(observation)
            st.caption(action)
    st.warning("위 차이는 연관성입니다. 해당 변수를 바꾸면 이탈이 감소한다는 인과관계로 해석할 수 없습니다.")
    importance = tables["importance"].head(15).sort_values("importance")
    fig = px.bar(importance, x="importance", y="feature", orientation="h")
    fig.update_layout(title="CatBoost 주요 입력 신호", xaxis_title="중요도", yaxis_title=None)
    st.plotly_chart(plot_style(fig), width="stretch")


def improvement_page(tables: dict) -> None:
    section_header("모델 개선 과정", "인사이트 기반 전처리와 동일 조건 모델 비교 과정을 확인합니다.")
    prep = tables["preprocessing"].copy()
    prep["decision"] = prep["decision"].map({"provisional_adopt": "채택", "exclude": "제외"}).fillna(prep["decision"])
    prep = prep.rename(columns={
        "variant": "전처리안", "prediction_time_available": "예측 시점 사용 가능",
        "logistic_pr_auc": "Logistic OOF PR-AUC", "decision": "결정", "reason": "근거",
    })
    st.dataframe(prep, hide_index=True, width="stretch")
    comparison = tables["comparison"].sort_values("pr_auc", ascending=False)
    fig = px.bar(comparison, x="model", y="pr_auc", color="model", text=comparison["pr_auc"].map(lambda x: f"{x:.4f}"))
    fig.update_layout(title="동일 log_numeric Feature · 동일 5-Fold OOF 비교", showlegend=False, xaxis_title="모델", yaxis_title="PR-AUC")
    st.plotly_chart(plot_style(fig), width="stretch")
    st.info("log_numeric 전처리를 공통 Feature로 채택한 뒤 8개 모델을 같은 Fold에서 비교했습니다. Target 기반 인코딩은 사용하지 않았습니다.")


def model_comparison_page(tables: dict) -> None:
    section_header("모델 비교·선정", "기본 비교와 실제 상위 3개 정밀 탐색 결과를 함께 봅니다.")
    comparison = tables["comparison"].sort_values("pr_auc", ascending=False)
    st.subheader("8개 모델 동일 조건 비교")
    st.dataframe(comparison[["model", "pr_auc", "roc_auc", "f1", "recall", "precision", "fn", "fp", "seconds"]], hide_index=True, width="stretch")
    fine = tables["fine_tuning"].sort_values("best_cv_pr_auc", ascending=False)
    st.subheader("상위 3개 정밀 탐색")
    st.dataframe(fine, hide_index=True, width="stretch")
    best = fine.iloc[0]
    st.success(f"최종 선정: {str(best.model).upper()} · Fine-tuned CV PR-AUC {float(best.best_cv_pr_auc):.6f}")
    st.caption("CatBoost·XGBoost·LightGBM의 차이는 작습니다. CatBoost는 PR-AUC와 Seed 안정성을 중심으로 선정했으며, 운영 Threshold는 별도 사업 결정입니다.")
    model_boundary()


def operations_page(tables: dict, scenario: pd.Series) -> None:
    section_header("운영 시나리오", "Recall·Precision·FN·FP와 검토 고객 수의 균형을 비교합니다.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("검토 대상 비율", f"{float(scenario.oof_target_rate):.1%}")
    c2.metric("Recall", f"{float(scenario.oof_recall):.1%}")
    c3.metric("Precision", f"{float(scenario.oof_precision):.1%}")
    c4.metric("FN / FP", f"{int(scenario.oof_fn):,} / {int(scenario.oof_fp):,}")
    st.info(f"{scenario.scenario_label} · Threshold {float(scenario.threshold):.2f} · {scenario.selection_rule}")
    scenarios = tables["scenarios"][[
        "scenario_label", "threshold", "oof_target_rate", "oof_precision", "oof_recall", "oof_f1", "oof_fn", "oof_fp"
    ]].rename(columns={
        "scenario_label": "시나리오", "threshold": "Threshold", "oof_target_rate": "대상 비율",
        "oof_precision": "Precision", "oof_recall": "Recall", "oof_f1": "F1", "oof_fn": "FN", "oof_fp": "FP",
    })
    st.dataframe(scenarios, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        topk = tables["targeting"]
        fig = go.Figure(go.Bar(
            x=topk["top_percent"].map(lambda x: f"상위 {x:g}%"), y=topk["capture_rate"],
            text=topk["capture_rate"].map(lambda x: f"{x:.1%}"), textposition="outside",
        ))
        fig.update_layout(title="OOF 상위 K% 포착률", yaxis_tickformat=".0%", yaxis_range=[0, 1.08])
        st.plotly_chart(plot_style(fig), width="stretch")
    with right:
        deciles = tables["deciles"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["actual_churn_rate"], name="관측 라벨 비율"))
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["mean_predicted_probability"], name="평균 예측 확률"))
        fig.update_layout(title="OOF 위험 Decile 진단", barmode="group", yaxis_tickformat=".0%", xaxis_title="Decile (10=최고 위험)")
        st.plotly_chart(plot_style(fig), width="stretch")
    model_boundary()


def prioritization_page(tables: dict, model, scenario: pd.Series) -> None:
    section_header("고객 우선순위", "검증된 CatBoost 모델로 단건을 재추론하거나 저장된 배치 순위를 확인합니다.")
    test, predictions = tables["test"], tables["predictions"]
    single_tab, batch_tab = st.tabs(["단건 확인", "배치 목록"])
    with single_tab:
        customer_id = st.number_input(
            "고객 ID", min_value=int(test.customer_id.min()), max_value=int(test.customer_id.max()),
            value=int(test.customer_id.iloc[0]), step=1,
        )
        row = test.loc[test["customer_id"].eq(customer_id)]
        if row.empty:
            st.warning("해당 고객 ID가 test.csv에 없습니다.")
        else:
            probability = float(model.predict_proba(row)[:, 1][0])
            selected = probability >= float(scenario.threshold)
            c1, c2, c3 = st.columns(3)
            c1.metric("이탈 위험 점수", f"{probability:.1%}")
            c2.metric("현재 시나리오", str(scenario.scenario_label))
            c3.metric("검토 대상", "예" if selected else "아니오")
            st.dataframe(row, hide_index=True, width="stretch")
    with batch_tab:
        threshold = st.slider("최소 위험 점수", 0.0, 1.0, float(scenario.threshold), 0.01)
        filtered = predictions.loc[predictions["churn_probability"].ge(threshold)].sort_values("churn_probability", ascending=False)
        c1, c2 = st.columns(2)
        c1.metric("검토 고객", f"{len(filtered):,}명")
        c2.metric("전체 대비", f"{len(filtered)/len(predictions):.1%}")
        st.dataframe(filtered.head(1000), hide_index=True, width="stretch")
        st.download_button(
            "검토 목록 CSV 다운로드", filtered.to_csv(index=False).encode("utf-8-sig"),
            "catboost_customer_priority.csv", "text/csv",
        )
    with st.expander("가정 기반 캠페인 계산기"):
        st.caption("실제 Uplift가 아닌 계획 비교용 가정 계산입니다.")
        contact_cost = st.number_input("고객 1명당 접촉 비용(원)", min_value=0, value=5000, step=500)
        customer_value = st.number_input("유지 고객 1명 가치(원)", min_value=0, value=120000, step=5000)
        success_rate = st.slider("접촉 성공 시 유지 전환율 가정", 0.0, 1.0, 0.10, 0.01)
        capacity = st.number_input("최대 접촉 인원", min_value=1, max_value=len(predictions), value=min(10000, len(predictions)), step=500)
        audience = predictions.sort_values("churn_probability", ascending=False).head(int(capacity))
        assumed_saves = float(audience["churn_probability"].sum() * success_rate)
        assumed_net = assumed_saves * customer_value - len(audience) * contact_cost
        st.metric("가정 순편익", f"{assumed_net:,.0f}원", f"가정 유지 전환 {assumed_saves:,.1f}명")
    model_boundary()


def main() -> None:
    tables, model = load_workspace()
    metadata = tables["metadata"]
    scenarios = tables["scenarios"]
    with st.sidebar:
        st.title("🎵 PlaylistPro")
        st.caption("고객 이탈 위험 의사결정 지원")
        page = st.radio(
            "메뉴",
            ["프로젝트 요약", "고객 인사이트", "모델 개선 과정", "모델 비교·선정", "운영 시나리오", "고객 우선순위"],
            label_visibility="collapsed",
        )
        scenario_id = st.selectbox(
            "검토 시나리오",
            scenarios["scenario_id"].tolist(),
            index=scenarios["scenario_id"].tolist().index(metadata.get("app_default_scenario_id", "balanced_f1")),
            format_func=lambda value: str(selected_scenario(scenarios, value).scenario_label),
        )
        st.divider()
        st.write("현재 모델: **CatBoost**")
        st.caption("모델 파일 SHA-256 검증 완료 후 로드")
    scenario = selected_scenario(scenarios, scenario_id)
    if page == "프로젝트 요약":
        project_summary_page(tables)
    elif page == "고객 인사이트":
        customer_insights_page(tables)
    elif page == "모델 개선 과정":
        improvement_page(tables)
    elif page == "모델 비교·선정":
        model_comparison_page(tables)
    elif page == "운영 시나리오":
        operations_page(tables, scenario)
    else:
        prioritization_page(tables, model, scenario)


if __name__ == "__main__":
    main()
