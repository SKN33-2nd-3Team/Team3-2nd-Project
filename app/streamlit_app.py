"""PlaylistPro Insight — Streamlit decision-support dashboard."""

from __future__ import annotations

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
TARGETING_PATH = ARTIFACT_DIR / "targeting_metrics_test.csv"
DECILE_PATH = ARTIFACT_DIR / "decile_calibration_test.csv"
SCENARIOS_PATH = ARTIFACT_DIR / "operating_scenarios.csv"
CANDIDATE_METADATA_PATH = ROOT / "models" / "candidates" / "20260721_submission_bounded_v1" / "metadata.json"
PROGRESSION_PATH = ARTIFACT_DIR / "performance_progression.csv"
DECISION_MATRIX_PATH = ARTIFACT_DIR / "model_selection_decision_matrix.csv"
INSIGHT_INVENTORY_PATH = ARTIFACT_DIR / "current_insight_inventory.csv"
FAIR_COMPARISON_PATH = ARTIFACT_DIR / "model_comparison_fair.csv"
PREPROCESSING_REGISTER_PATH = ARTIFACT_DIR / "insight_preprocessing_register.csv"

st.set_page_config(
    page_title="PlaylistPro Insight",
    page_icon="♫",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --ink:#111827; --muted:#64748b; --line:#e5e7eb; --panel:#ffffff; --accent:#e4573d; --accent-soft:#fff1ed; --navy:#18324a; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3, h4 { font-family: 'Space Grotesk', sans-serif; color: var(--ink); letter-spacing:-.02em; }
    .stApp { background: #f7f8fa; color: var(--ink); }
    [data-testid="stSidebar"] { background: #102235; border-right: 0; }
    /* Keep navigation and explanatory copy legible on the fixed dark sidebar,
       while letting Streamlit form controls use the active app theme. */
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] label *,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary,
    [data-testid="stSidebar"] [data-testid="stExpander"] summary * { color: #e8eef5 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div * {
        color: var(--text-color, #111827) !important;
    }
    [data-testid="stSidebar"] code {
        background: rgba(255,255,255,.14) !important;
        border: 1px solid rgba(255,255,255,.20);
        border-radius: .35rem;
        color: #ffffff !important;
        padding: .08rem .32rem;
    }
    [data-testid="stSidebar"] .stRadio label { padding: .55rem .65rem; border-radius: .55rem; }
    [data-testid="stSidebar"] .stRadio label:hover { background: rgba(255,255,255,.08); }
    .brand { padding: .4rem 0 1.4rem 0; }
    .brand-mark { display:inline-flex; align-items:center; justify-content:center; width:38px; height:38px; background:#e4573d; color:white; border-radius:12px; font-size:23px; font-weight:700; margin-right:10px; }
    .brand-name { font-family:'Space Grotesk'; font-weight:700; font-size:20px; color:white; vertical-align:middle; }
    .eyebrow { color:var(--accent); text-transform:uppercase; letter-spacing:.12em; font-size:11px; font-weight:700; margin-bottom:.45rem; }
    .hero { background:linear-gradient(115deg,#18324a 0%,#214765 65%,#315d78 100%); color:white; border-radius:20px; padding:26px 30px; margin-bottom:22px; box-shadow:0 12px 32px rgba(24,50,74,.12); }
    .hero h1 { color:white; margin:0; font-size:34px; }
    .hero p { color:#d9e4ed; max-width:780px; margin:.6rem 0 0; font-size:15px; line-height:1.6; }
    .hero .tag { display:inline-block; background:rgba(255,255,255,.13); color:#f4f8fb; border-radius:999px; padding:6px 10px; margin-top:14px; font-size:12px; }
    .metric-card { background:white; border:1px solid var(--line); border-radius:16px; padding:17px 18px; min-height:112px; box-shadow:0 5px 18px rgba(15,23,42,.035); }
    .metric-label { color:var(--muted); font-size:12px; font-weight:600; }
    .metric-value { color:var(--ink); font-family:'Space Grotesk'; font-size:28px; font-weight:700; margin-top:8px; }
    .metric-delta { color:var(--accent); font-size:12px; margin-top:4px; }
    .section-label { margin-top:25px; margin-bottom:8px; font-family:'Space Grotesk'; font-size:18px; font-weight:600; }
    .info-card { border:1px solid #fed6cc; background:#fff8f5; border-radius:14px; padding:15px 17px; line-height:1.55; color:#63352d; }
    .note-card { border:1px solid #dbe5ef; background:#f3f7fb; border-radius:14px; padding:15px 17px; line-height:1.55; color:#29445d; }
    .risk-high { color:#b93827; background:#fff0ec; border:1px solid #ffc7ba; border-radius:999px; padding:5px 10px; font-weight:700; display:inline-block; }
    .risk-mid { color:#9a6413; background:#fff8e9; border:1px solid #f7dc9b; border-radius:999px; padding:5px 10px; font-weight:700; display:inline-block; }
    .risk-low { color:#267152; background:#edf9f3; border:1px solid #bce6d0; border-radius:999px; padding:5px 10px; font-weight:700; display:inline-block; }
    div[data-testid="stDataFrame"] { border-radius:12px; overflow:hidden; }
    .small-caption { color:var(--muted); font-size:12px; line-height:1.5; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    return pd.read_csv(DATA_DIR / "train.csv"), pd.read_csv(DATA_DIR / "test.csv")


@st.cache_resource(show_spinner=False)
def load_model():
    # Importing the module registers the serialized custom transformer class.
    from src.features import MusicFeatureEngineer  # noqa: F401

    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def load_artifacts() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    comparison = pd.read_csv(ARTIFACT_DIR / "model_comparison.csv")
    threshold = pd.read_csv(ARTIFACT_DIR / "threshold_sweep_validation.csv")
    importance_path = ARTIFACT_DIR / "feature_importance.csv"
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame(columns=["feature", "importance", "rank"])
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    targeting = pd.read_csv(TARGETING_PATH)
    deciles = pd.read_csv(DECILE_PATH)
    scenarios = pd.read_csv(SCENARIOS_PATH)
    return comparison, threshold, metadata, predictions, importance, targeting, deciles, scenarios


@st.cache_data(show_spinner=False)
def load_presentation_evidence() -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load saved evidence only; never fit or promote a candidate from the app."""
    candidate = json.loads(CANDIDATE_METADATA_PATH.read_text(encoding="utf-8")) if CANDIDATE_METADATA_PATH.exists() else {}
    progression = pd.read_csv(PROGRESSION_PATH) if PROGRESSION_PATH.exists() else pd.DataFrame()
    decision_matrix = pd.read_csv(DECISION_MATRIX_PATH) if DECISION_MATRIX_PATH.exists() else pd.DataFrame()
    insights = pd.read_csv(INSIGHT_INVENTORY_PATH) if INSIGHT_INVENTORY_PATH.exists() else pd.DataFrame()
    fair_comparison = pd.read_csv(FAIR_COMPARISON_PATH) if FAIR_COMPARISON_PATH.exists() else pd.DataFrame()
    preprocessing_register = pd.read_csv(PREPROCESSING_REGISTER_PATH) if PREPROCESSING_REGISTER_PATH.exists() else pd.DataFrame()
    return candidate, progression, decision_matrix, insights, fair_comparison, preprocessing_register


def safe_load():
    required = [
        DATA_DIR / "train.csv",
        DATA_DIR / "test.csv",
        MODEL_PATH,
        METADATA_PATH,
        TARGETING_PATH,
        DECILE_PATH,
        SCENARIOS_PATH,
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("필수 파일이 없습니다: " + ", ".join(missing))
        st.stop()
    return load_data(), load_model(), load_artifacts()


def metric_card(label: str, value: str, sub: str = ""):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-delta">{sub}</div></div>',
        unsafe_allow_html=True,
    )


def scenario_label(row: pd.Series) -> str:
    return f"{row['scenario_label']} (임계값 {float(row['threshold']):.2f})"


def get_scenario(scenarios: pd.DataFrame, scenario_id: str) -> pd.Series:
    selected = scenarios.loc[scenarios["scenario_id"] == scenario_id]
    if selected.empty:
        return scenarios.iloc[0]
    return selected.iloc[0]


def risk_level(probability: float, scenarios: pd.DataFrame) -> tuple[str, str]:
    ordered = scenarios.sort_values("threshold")
    aggressive, balanced, precision = ordered.iloc[0], ordered.iloc[1], ordered.iloc[-1]
    if probability >= float(precision["threshold"]):
        return "정밀 우선", "risk-high"
    if probability >= float(balanced["threshold"]):
        return "균형 대응", "risk-high"
    if probability >= float(aggressive["threshold"]):
        return "관찰 후보", "risk-mid"
    return "저위험", "risk-low"


def plotly_theme(fig):
    fig.update_layout(
        template="plotly_white",
        font=dict(family="DM Sans", color="#111827"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def header(title: str, description: str):
    st.markdown(f'<div class="eyebrow">PLAYLISTPRO / {title.upper()}</div>', unsafe_allow_html=True)
    st.title(title)
    st.caption(description)


def candidate_status_card(candidate: dict):
    """Make the candidate boundary visible wherever model evidence is discussed."""
    if not candidate:
        st.info("저장된 기술 후보 모델 메타데이터가 없습니다.")
        return
    validation = candidate.get("metrics", {}).get("validation", {})
    holdout = candidate.get("metrics", {}).get("internal_test_holdout", {})
    candidate_name = str(candidate.get("candidate_model", "unknown")).upper()
    status = str(candidate.get("status", "UNKNOWN"))
    st.markdown(
        f'<div class="info-card"><b>기술 후보 모델 — {candidate_name}</b><br>'
        f'상태: <b>{status}</b>. 이는 저장된 검증 근거이며, 현재 데모 점수 산출에 쓰이는 모델이 아닙니다. '
        '운영 승격, 예측 기간, 캠페인 효과는 아직 검증되지 않았습니다.</div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("후보 Validation PR-AUC", f"{validation.get('pr_auc', float('nan')):.3f}", "저장된 독립 실행")
    with c2:
        metric_card("후보 Holdout PR-AUC", f"{holdout.get('pr_auc', float('nan')):.3f}", "내부 Holdout")
    with c3:
        metric_card("재로딩 검증", "통과" if candidate.get("reload_validation", {}).get("verified") else "미검증", "저장된 후보 산출물")


def overview_page(
    train: pd.DataFrame,
    test: pd.DataFrame,
    comparison: pd.DataFrame,
    metadata: dict,
    predictions: pd.DataFrame,
    targeting: pd.DataFrame,
    scenarios: pd.DataFrame,
    active_scenario: pd.Series,
    candidate: dict,
):
    st.markdown(
        '<div class="hero"><div class="eyebrow" style="color:#ffb3a4">MUSIC RETENTION INTELLIGENCE</div><h1>이탈 신호를 읽고, 다음 행동을 설계하세요.</h1><p>구독·결제·청취 행동을 한 화면에서 탐색하고, 데이터셋 라벨 기준 이탈 위험을 고객 단위로 확인하는 PlaylistPro 분석 workspace입니다.</p><span class="tag">Light mode · Model inference only · FN 우선</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="info-card"><b>해석 범위 안내</b><br>현재 파일에는 관측 기준일·해지일·30일 라벨 생성 규칙이 없습니다. 따라서 아래 확률은 <b>데이터셋의 churned 라벨 기준</b>이며, 실제 향후 30일 해지 확률로 확정 표시하지 않습니다.</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="note-card"><b>현재 선택: {active_scenario["scenario_label"]}</b><br>'
        f'{active_scenario["selection_rule"]} · threshold {float(active_scenario["threshold"]):.2f} · '
        f'내부 Test 대상 비율 {float(active_scenario["test_target_rate"]):.1%}. '
        '이 선택은 연락 우선순위의 운영 가정이며, 자동 할인·차단·발송을 실행하지 않습니다.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="section-label">오늘의 데이터 snapshot</div>', unsafe_allow_html=True)
    with st.expander("Model state and candidate evidence"):
        st.markdown(f"**Demo inference model:** `{metadata['model']}`")
        candidate_status_card(candidate)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("학습 고객", f"{len(train):,}", "target 포함")
    with c2:
        metric_card("제공 test", f"{len(test):,}", "정답 없는 추론용")
    with c3:
        metric_card("관찰 이탈률", f"{train['churned'].mean():.1%}", "churned=1")
    with c4:
        metric_card("비교 모델", f"{len(comparison):,}개", "최종 모델 선정 완료")

    st.markdown('<div class="section-label">핵심 신호</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    plan_rates = train.groupby("payment_plan")["churned"].mean()
    type_rates = train.groupby("subscription_type")["churned"].mean().sort_values(ascending=False)
    with a:
        metric_card("결제주기 격차", f"{plan_rates.max() - plan_rates.min():+.1%}", f"{plan_rates.idxmax()} vs {plan_rates.idxmin()}")
    with b:
        metric_card("최고위험 요금제", f"{type_rates.iloc[0]:.1%}", f"{type_rates.index[0]} 관찰 이탈률")
    with c:
        metric_card("모델 Test PR-AUC", f"{comparison.iloc[0]['test_pr_auc']:.3f}", f"{comparison.iloc[0]['model']} 기준")

    left, right = st.columns([1.05, 1])
    with left:
        st.markdown('<div class="section-label">Target 구성</div>', unsafe_allow_html=True)
        counts = train["churned"].map({0: "Active", 1: "Churned"}).value_counts().reset_index()
        counts.columns = ["status", "customers"]
        fig = px.pie(counts, names="status", values="customers", hole=.62, color="status", color_discrete_map={"Active": "#315A7D", "Churned": "#E4573D"})
        fig.update_traces(textinfo="percent+label", marker=dict(line=dict(color="#ffffff", width=2)))
        st.plotly_chart(plotly_theme(fig), width="stretch")
    with right:
        st.markdown('<div class="section-label">제공 test의 예측 snapshot</div>', unsafe_allow_html=True)
        ordered = scenarios.sort_values("threshold")
        cuts = ordered["threshold"].tolist()
        bins = pd.cut(
            predictions["churn_probability"],
            bins=[-0.01, *cuts, 1.0],
            labels=["모니터링", *ordered["scenario_label"].tolist()],
        )
        risk_counts = bins.value_counts().sort_index().reset_index()
        risk_counts.columns = ["risk_band", "customers"]
        fig = px.bar(risk_counts, x="risk_band", y="customers", color="risk_band", color_discrete_sequence=["#8ac5a6", "#f2cb74", "#f19b78", "#d94a3a"])
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Customers")
        st.plotly_chart(plotly_theme(fig), width="stretch")
        st.markdown('<div class="small-caption">정답이 없는 test이므로 이 분포는 예측 결과 분포이며 실제 이탈률이 아닙니다.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">운영자가 먼저 볼 인사이트</div>', unsafe_allow_html=True)
    heldout = targeting.copy()
    heldout["top_percent"] = heldout["top_percent"].map(lambda value: f"상위 {value:g}%")
    heldout["precision"] = heldout["precision"].map(lambda value: f"{value:.1%}")
    heldout["capture_rate"] = heldout["capture_rate"].map(lambda value: f"{value:.1%}")
    st.dataframe(
        heldout[["top_percent", "target_customers", "actual_churners_captured", "precision", "capture_rate", "lift"]],
        column_config={
            "top_percent": "내부 Test 상위 구간",
            "target_customers": "대상 고객 수",
            "actual_churners_captured": "포착된 실제 라벨 이탈자",
            "precision": "Precision",
            "capture_rate": "Capture rate",
            "lift": st.column_config.NumberColumn("Lift", format="%.2f×"),
        },
        hide_index=True,
        width="stretch",
    )
    st.caption("위 표는 내부 Test holdout에서 한 번 평가한 랭킹 성능입니다. 제공 test.csv의 실제 이탈이나 캠페인 효과를 뜻하지 않습니다.")
    inquiries = train.groupby("customer_service_inquiries")["churned"].mean().sort_values(ascending=False)
    st.dataframe(
        pd.DataFrame(
            [
                ["구독 유형", str(type_rates.index[0]), f"{type_rates.iloc[0]:.1%}", "최고위험 요금제 — 전환·혜택 실험 1순위"],
                ["구독 유형", str(type_rates.index[-1]), f"{type_rates.iloc[-1]:.1%}", "최저위험 요금제 — 유지 요인 벤치마크"],
                ["상담 문의", str(inquiries.index[0]), f"{inquiries.iloc[0]:.1%}", "불만 신호와 이탈의 연관 — 상담 이슈 해결을 유지 활동에 연결"],
                ["결제 주기", str(plan_rates.idxmax()), f"{plan_rates.max():.1%}", f"주기 간 격차 {plan_rates.max() - plan_rates.min():+.1%} — 연관성 점검"],
            ],
            columns=["관점", "세그먼트/필드", "관찰값", "다음 확인"],
        ),
        hide_index=True,
        width="stretch",
    )


def eda_page(train: pd.DataFrame):
    header("Data Explorer", "각 입력 변수와 churned 라벨의 관계를 분포·이탈률·표본 수로 확인합니다.")
    feature_options = [c for c in train.columns if c not in ["customer_id", "churned"]]
    st.markdown(
        '<div class="info-card"><b>Customer insights: observed associations only</b><br>'
        'Use the cards below as hypotheses for service and feature design, not as causal claims or automatic campaign rules.</div>',
        unsafe_allow_html=True,
    )
    support_rates = train.groupby("customer_service_inquiries")["churned"].mean()
    subscription_rates = train.groupby("subscription_type")["churned"].mean()
    plan_rates = train.groupby("payment_plan")["churned"].mean()
    listening_band = pd.qcut(train["weekly_hours"], q=4, duplicates="drop")
    listening_rates = train.groupby(listening_band, observed=True)["churned"].mean()
    insight_columns = st.columns(4)
    insight_columns[0].metric("Support-contact spread", f"{support_rates.max() - support_rates.min():.1%}", "observed label-rate range")
    insight_columns[1].metric("Subscription-type spread", f"{subscription_rates.max() - subscription_rates.min():.1%}", "segment association")
    insight_columns[2].metric("Payment-plan spread", f"{plan_rates.max() - plan_rates.min():.1%}", "weak signal if near zero")
    insight_columns[3].metric("Listening-quartile spread", f"{listening_rates.max() - listening_rates.min():.1%}", "behavioral association")
    st.caption("Observed result → feature/design hypothesis → validate in an experiment. These associations do not establish churn causes.")
    # Open on the strongest signal; payment_plan is now the weakest (spread 0.001).
    default_column = "weekly_hours" if "weekly_hours" in feature_options else feature_options[0]
    selected = st.selectbox("분석할 컬럼", feature_options, index=feature_options.index(default_column))
    st.markdown('<div class="small-caption">모든 컬럼을 선택해 개별 target 관계를 확인할 수 있습니다. 그래프는 연관관계이며 인과관계가 아닙니다.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.45, 1])
    with left:
        if selected == "signup_date" and not pd.api.types.is_numeric_dtype(train[selected]):
            view = train.assign(signup_year=pd.to_datetime(train[selected]).dt.year).groupby("signup_year")["churned"].agg(["mean", "size"]).reset_index()
            fig = px.bar(view, x="signup_year", y="mean", text=view["mean"].map(lambda x: f"{x:.1%}"), labels={"mean": "Churn rate", "signup_year": "Signup year"}, title="Signup cohort vs churn rate")
            fig.update_traces(marker_color="#E4573D", textposition="outside")
        elif pd.api.types.is_numeric_dtype(train[selected]):
            view = train[[selected, "churned"]].copy()
            view["label"] = view["churned"].map({0: "Active", 1: "Churned"})
            fig = px.histogram(view, x=selected, color="label", marginal="box", barmode="overlay", opacity=.7, color_discrete_map={"Active": "#315A7D", "Churned": "#E4573D"}, title=f"{selected} distribution by churn label")
        else:
            rates = train.groupby(selected, dropna=False)["churned"].agg(["mean", "size"]).reset_index().sort_values("mean", ascending=True)
            rates["label"] = rates["mean"].map(lambda x: f"{x:.1%}")
            fig = px.bar(rates, x="mean", y=selected, text="label", orientation="h", title=f"{selected} vs churn rate", labels={"mean": "Churn rate"})
            fig.update_traces(marker_color="#E4573D", textposition="outside")
        st.plotly_chart(plotly_theme(fig), width="stretch")
    with right:
        st.markdown(f'<div class="section-label">{selected} quick read</div>', unsafe_allow_html=True)
        if selected == "signup_date" and not pd.api.types.is_numeric_dtype(train[selected]):
            dates = pd.to_datetime(train[selected])
            st.metric("기간", f"{dates.min():%Y-%m-%d} → {dates.max():%Y-%m-%d}")
            st.metric("고유 날짜", f"{dates.nunique():,}")
            st.warning("snapshot/해지일이 없어 30일 해지 변수로 사용할 수 없습니다.")
        elif pd.api.types.is_numeric_dtype(train[selected]):
            corr = train[[selected, "churned"]].corr().iloc[0, 1]
            st.metric("Pearson correlation", f"{corr:+.4f}")
            st.metric("중앙값", f"{train[selected].median():,.3f}")
            st.metric("범위", f"{train[selected].min():,.3f} ~ {train[selected].max():,.3f}")
        else:
            grouped = train.groupby(selected)["churned"].agg(["mean", "size"])
            st.metric("범주 수", f"{grouped.shape[0]:,}")
            st.metric("최고 이탈률", f"{grouped['mean'].max():.1%}")
            st.metric("최저 이탈률", f"{grouped['mean'].min():.1%}")

    st.markdown('<div class="section-label">전체 컬럼 관계 요약</div>', unsafe_allow_html=True)
    summary_rows = []
    for column in feature_options:
        if column == "signup_date" and not pd.api.types.is_numeric_dtype(train[column]):
            series = pd.to_datetime(train[column])
            summary_rows.append([column, "date", f"{series.min():%Y-%m-%d} ~ {series.max():%Y-%m-%d}", "cohort only", "30일 라벨 미검증"])
        elif pd.api.types.is_numeric_dtype(train[column]):
            corr = train[[column, "churned"]].corr().iloc[0, 1]
            summary_rows.append([column, "numeric", f"corr {corr:+.4f}", f"median {train[column].median():,.2f}", "association"])
        else:
            rates = train.groupby(column)["churned"].mean()
            summary_rows.append([column, "categorical", f"max {rates.max():.1%}", f"min {rates.min():.1%}", "association"])
    st.dataframe(pd.DataFrame(summary_rows, columns=["컬럼", "타입", "핵심 신호", "분포/범위", "해석"]), hide_index=True, width="stretch")


def improvement_page(progression: pd.DataFrame, candidate: dict, decision_matrix: pd.DataFrame):
    header(
        "모델 개선 과정",
        "저장된 실험 결과로 Baseline, Feature 작업, 모델 비교가 순위 품질을 어떻게 바꿨는지 확인합니다.",
    )
    st.markdown(
        '<div class="info-card"><b>근거 범위</b><br>'
        '이 화면은 저장된 과거 실험 결과만 읽습니다. 학습 재실행, Threshold 재선정, 인과 효과 주장은 하지 않습니다.</div>',
        unsafe_allow_html=True,
    )
    if progression.empty:
        st.warning("저장된 성능 개선 근거를 찾을 수 없습니다.")
        return

    raw_logistic = progression.loc[progression["model"].eq("logistic_raw")]
    raw_pr_auc = float(raw_logistic.iloc[0]["pr_auc"]) if not raw_logistic.empty else float("nan")
    candidate_pr_auc = float(candidate.get("metrics", {}).get("validation", {}).get("pr_auc", float("nan")))
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("Raw Logistic PR-AUC", f"{raw_pr_auc:.3f}", "과거 Baseline")
    with c2:
        metric_card("후보 Validation PR-AUC", f"{candidate_pr_auc:.3f}", "저장된 LightGBM 후보")
    with c3:
        metric_card("순위 품질 변화", f"{candidate_pr_auc - raw_pr_auc:+.3f}", "저장된 평가 Run이 달라 방향성 참고")

    progress_view = progression[["experiment", "model", "adopted", "pr_auc", "roc_auc", "f1", "recall", "precision"]].copy()
    progress_view.columns = ["실험", "모델", "결정", "PR-AUC", "ROC-AUC", "F1", "Recall", "Precision"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=progress_view["실험"], y=progress_view["PR-AUC"], mode="lines+markers",
        marker_color="#E4573D", line_color="#E4573D", name="PR-AUC",
    ))
    fig.add_trace(go.Scatter(
        x=progress_view["실험"], y=progress_view["ROC-AUC"], mode="lines+markers",
        marker_color="#315A7D", line_color="#315A7D", name="ROC-AUC",
    ))
    fig.update_layout(title="저장된 성능 개선 과정", xaxis_title=None, yaxis_title="점수", xaxis_tickangle=-28)
    st.plotly_chart(plotly_theme(fig), width="stretch")

    st.markdown('<div class="section-label">채택·제외 실험</div>', unsafe_allow_html=True)
    st.dataframe(
        progress_view.style.format({metric: "{:.3f}" for metric in ["PR-AUC", "ROC-AUC", "F1", "Recall", "Precision"]}),
        hide_index=True,
        width="stretch",
    )
    st.caption("발표에서는 관찰 결과 → 가설/Feature·모델 변경 → 지표 변화 → 채택·제외 순서로 설명합니다. FN·FP는 동일 Threshold가 보장된 운영 시나리오에서만 비교합니다. Feature Importance는 인과 설명이 아닙니다.")

    if not decision_matrix.empty:
        st.markdown('<div class="section-label">저장된 후보 선정 근거</div>', unsafe_allow_html=True)
        selection_columns = ["model", "validation_pr_auc", "validation_roc_auc", "selection_status", "selection_reason"]
        st.dataframe(decision_matrix[selection_columns], hide_index=True, width="stretch")
        st.caption("제한된 후보 선정은 Validation PR-AUC를 사용했습니다. 더 넓은 공정 비교 Run은 아직 완료되지 않아 최종 결과로 표시하지 않습니다.")


def project_summary_page(train: pd.DataFrame, test: pd.DataFrame, metadata: dict, candidate: dict):
    header("프로젝트 요약", "관측된 이탈 라벨을 검토 우선순위로 연결하되, 모델 상태와 한계를 먼저 공개합니다.")
    st.markdown(
        '<div class="hero"><div class="eyebrow" style="color:#ffb3a4">발표 요약</div>'
        '<h1>리텐션 팀은 누구를 먼저 검토해야 할까?</h1>'
        '<p>PlaylistPro는 제공된 데이터셋 라벨을 기준으로 고객을 순위화하고, 대상 범위의 균형을 보여줍니다. '
        '검증된 미래 이탈 기간, 캠페인 효과, 자동 리텐션 조치는 주장하지 않습니다.</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("학습 고객", f"{len(train):,}", "라벨 있음")
    with c2:
        metric_card("배치 고객", f"{len(test):,}", "라벨 없음")
    with c3:
        metric_card("관측 라벨 비율", f"{train['churned'].mean():.1%}", "데이터셋 churned=1")
    with c4:
        metric_card("데모 추론", "부스팅 모델", "저장된 Pipeline")
    st.markdown('<div class="section-label">발표 핵심 메시지</div>', unsafe_allow_html=True)
    st.markdown(
        '- 고객 행동 패턴은 **관측된 연관성**이며, 서비스·Feature 가설을 만드는 데 사용합니다.\n'
        '- 모델 근거는 단일 점수가 아니라 순위 품질과 오류 Trade-off로 평가합니다.\n'
        '- Threshold는 검토 대상을 고르는 값이며, 운영 용량과 FN/FP 비용은 사업 판단이 필요합니다.\n'
        '- 개별 점수는 저장된 Pipeline의 실제 추론 결과이고, 배치 고객은 정답 라벨이 없습니다.',
    )
    st.markdown('<div class="section-label">모델 상태</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="note-card"><b>데모 추론 모델: {metadata.get("model", "unknown")}</b><br>'
        '저장된 Gradient Boosting Pipeline이 고객 점수 화면을 구동합니다. 후보 모델 산출물로 교체하지 않았습니다.</div>',
        unsafe_allow_html=True,
    )
    candidate_status_card(candidate)


def customer_insights_page(insights: pd.DataFrame):
    header("고객 인사이트", "근거가 있는 관찰 4개와 활용 가능성, 데이터가 뒷받침하지 못하는 한계를 함께 봅니다.")
    st.markdown(
        '<div class="info-card"><b>해석 원칙</b><br>'
        '각 카드는 관찰 → 가설 또는 Feature 활용 → 한계 순서입니다. Feature를 바꾸면 이탈이 줄어든다는 주장은 아닙니다.</div>',
        unsafe_allow_html=True,
    )
    if insights.empty:
        st.warning("저장된 인사이트 목록을 찾을 수 없습니다.")
        return
    visible = insights.loc[insights["insight_id"].isin(["I01", "I02", "I03", "I04"])].copy()
    korean_copy = {
        "I01": {
            "insight": "주간 청취 시간이 낮고 곡 건너뛰기 비율이 높을수록 이탈 라벨과 연관성이 관찰됩니다.",
            "actionability": "진단 우선 세그먼트로 활용하되, 원인이라고 단정하지 않습니다.",
            "limitation": "시간적 선후관계가 없어 인과관계를 판단할 수 없습니다.",
            "evidence": "EDA 이탈률 차트와 Feature Importance",
            "status": "관찰 연관성 검증",
        },
        "I02": {
            "insight": "Free 구독 유형에서 높은 관측 이탈률이 나타납니다.",
            "actionability": "혜택·전환 제안의 가설로만 활용하고 실험으로 검증합니다.",
            "limitation": "제안이나 혜택의 실제 효과는 알려져 있지 않습니다.",
            "evidence": "구독 유형별 EDA 이탈률",
            "status": "관찰 연관성 검증",
        },
        "I03": {
            "insight": "고객 서비스 문의가 많을수록 이탈 라벨과 연관성이 관찰됩니다.",
            "actionability": "상담 해결 품질·처리 속도 개선 실험의 가설로 활용합니다.",
            "limitation": "문의가 이탈의 원인인지, 이탈 위험의 결과인지 알 수 없습니다.",
            "evidence": "고객 서비스 문의별 EDA 이탈률",
            "status": "관찰 연관성 검증",
        },
        "I04": {
            "insight": "구독 일시정지 횟수와 연령은 모델의 예측 신호에 기여합니다.",
            "actionability": "일시정지는 진단 신호로 활용하고, 연령 활용은 공정성 검토가 필요합니다.",
            "limitation": "모델 중요도는 행동 가능한 원인이나 개입 효과를 뜻하지 않습니다.",
            "evidence": "저장된 Feature Importance",
            "status": "모델 신호 확인",
        },
    }
    for index, row in visible.iterrows():
        translated = korean_copy.get(row["insight_id"])
        if translated:
            for column, value in translated.items():
                visible.loc[index, column] = value
    for row_group in np.array_split(visible, max(1, int(np.ceil(len(visible) / 2)))):
        columns = st.columns(len(row_group))
        for column, (_, row) in zip(columns, row_group.iterrows()):
            with column:
                st.markdown(f"#### {row['insight_id']}")
                st.write(row["insight"])
                st.caption(f"활용: {row['actionability']}")
                st.caption(f"한계: {row['limitation']}")
    st.markdown('<div class="section-label">인사이트-Feature 연결 기록</div>', unsafe_allow_html=True)
    st.dataframe(
        visible[["insight_id", "related_features", "status", "evidence", "actionability", "limitation"]],
        hide_index=True,
        width="stretch",
    )
    st.caption("Feature Importance와 라벨 비율 차이는 모델·데이터 근거입니다. 인과 요인이나 자동 고객 접촉 권고가 아닙니다.")


def model_selection_page(fair_comparison: pd.DataFrame, candidate: dict, decision_matrix: pd.DataFrame):
    header("모델 비교·선정", "완료된 공통 Feature 모델 비교와 후보 선정 근거, 미완료 검증 작업을 분리해서 봅니다.")
    st.markdown(
        '<div class="info-card"><b>완료된 8개 모델 비교</b><br>'
        '아래 표는 동일한 `log_numeric` Feature Variant의 5-Fold OOF Baseline 비교입니다. 튜닝, Seed 안정성, 운영 승격이 완료되었다는 뜻은 아닙니다.</div>',
        unsafe_allow_html=True,
    )
    if not fair_comparison.empty:
        ordered = fair_comparison.sort_values("pr_auc", ascending=False).copy()
        fig = px.bar(ordered, x="model", y="pr_auc", color="model", text=ordered["pr_auc"].map(lambda value: f"{value:.3f}"))
        fig.update_layout(title="5-Fold OOF PR-AUC — 공통 Feature 비교", showlegend=False, yaxis_title="PR-AUC", xaxis_title=None)
        st.plotly_chart(plotly_theme(fig), width="stretch")
        st.dataframe(
            ordered[["model", "pr_auc", "roc_auc", "recall", "precision", "fn", "fp", "seconds"]],
            hide_index=True,
            width="stretch",
        )
    else:
        st.warning("저장된 공통 Feature 모델 비교 결과를 찾을 수 없습니다.")
    st.markdown('<div class="section-label">저장된 기술 후보</div>', unsafe_allow_html=True)
    candidate_status_card(candidate)
    if not decision_matrix.empty:
        st.dataframe(
            decision_matrix[["model", "validation_pr_auc", "validation_roc_auc", "selection_status", "selection_reason"]],
            hide_index=True,
            width="stretch",
        )
    st.warning("최고 OOF Baseline을 최종 운영 모델이라고 말하면 안 됩니다. 더 넓은 공정 비교 Workflow는 아직 완료되지 않았습니다.")


def operations_page(targeting: pd.DataFrame, deciles: pd.DataFrame, scenarios: pd.DataFrame):
    scenario_ids = scenarios["scenario_id"].tolist()
    selected_id = st.selectbox(
        "비교할 운영 시나리오",
        scenario_ids,
        format_func=lambda scenario_id: scenario_label(get_scenario(scenarios, scenario_id)),
    )
    active_scenario = get_scenario(scenarios, selected_id)
    header("운영 시나리오", "사업 담당자가 Threshold를 고르기 전에 연락 대상 범위와 성능 Trade-off를 보여줍니다.")
    st.markdown(
        f'<div class="note-card"><b>선택 시나리오: {active_scenario["scenario_label"]}</b><br>'
        f'{active_scenario["selection_rule"]} · threshold {float(active_scenario["threshold"]):.2f} · '
        'Validation에서 선택하고 내부 Holdout에서 한 번 평가한 결과입니다. 실제 캠페인 성과가 아닙니다.</div>',
        unsafe_allow_html=True,
    )
    scenario_view = scenarios[["scenario_label", "threshold", "test_target_rate", "test_precision", "test_recall", "test_f1", "test_fn", "test_fp"]].copy()
    scenario_view.columns = ["시나리오", "Threshold", "대상 비율", "Precision", "Recall", "F1", "FN", "FP"]
    st.dataframe(scenario_view, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(
            x=targeting["top_percent"].map(lambda value: f"Top {value:g}%"),
            y=targeting["capture_rate"],
            marker_color="#E4573D",
            text=targeting["capture_rate"].map(lambda value: f"{value:.1%}"),
            textposition="outside",
        ))
        fig.update_layout(title="내부 Holdout Top-K Capture", yaxis_tickformat=".0%", yaxis_range=[0, 1.08], showlegend=False)
        st.plotly_chart(plotly_theme(fig), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["actual_churn_rate"], name="Observed", marker_color="#E4573D"))
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["mean_predicted_probability"], name="Predicted", marker_color="#315A7D"))
        fig.update_layout(title="내부 Holdout Risk Decile 점검", barmode="group", yaxis_tickformat=".0%", xaxis_title="Decile (10 = 최고 위험)")
        st.plotly_chart(plotly_theme(fig), width="stretch")
    st.markdown('<div class="section-label">의사결정 원칙</div>', unsafe_allow_html=True)
    st.write("연락 가능 인원과 이탈 고객을 놓치는 비용(FN), 비이탈 고객에게 연락하는 비용(FP)을 정한 뒤 시나리오를 선택합니다. Uplift와 ROI는 운영 후 통제된 캠페인으로 측정해야 합니다.")


def project_summary_v2(
    train: pd.DataFrame,
    test: pd.DataFrame,
    metadata: dict,
    candidate: dict,
    targeting: pd.DataFrame,
):
    header("프로젝트 요약", "이탈 고객을 단정하는 도구가 아니라, 제한된 검토 인력을 어디에 먼저 쓸지 돕는 의사결정 지원 화면입니다.")
    top30 = targeting.loc[targeting["top_percent"].eq(30)] if not targeting.empty else pd.DataFrame()
    if not top30.empty:
        evidence = top30.iloc[0]
        decision_text = (
            f"내부 Holdout에서 위험 점수 상위 30%({int(evidence['target_customers']):,}명)를 검토하면, "
            f"관측된 이탈 라벨의 {float(evidence['capture_rate']):.1%}를 포착했습니다. "
            f"이 값은 실제 캠페인 효과가 아닌 저장된 모델 평가 결과입니다."
        )
    else:
        decision_text = "저장된 내부 Holdout 대상화 근거를 찾을 수 없습니다."
    st.markdown(
        f'<div class="hero"><div class="eyebrow" style="color:#ffb3a4">핵심 의사결정</div>'
        f'<h1>누구를 먼저 검토할지, 점수 순위로 정합니다.</h1><p>{decision_text}</p></div>',
        unsafe_allow_html=True,
    )
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("학습 고객", f"{len(train):,}", "라벨 있음")
    with c2:
        metric_card("배치 고객", f"{len(test):,}", "라벨 없음")
    with c3:
        metric_card("관측 이탈 라벨 비율", f"{train['churned'].mean():.1%}", "churned=1")
    with c4:
        metric_card("현재 데모 추론", "Gradient Boosting", "저장된 Pipeline")
    st.markdown('<div class="section-label">사용자가 알아야 할 세 가지</div>', unsafe_allow_html=True)
    left, middle, right = st.columns(3)
    with left:
        st.markdown('<div class="note-card"><b>1. 인사이트</b><br>행동·구독·문의 신호의 차이는 이탈 라벨과의 관찰된 연관성입니다. 원인으로 단정하지 않습니다.</div>', unsafe_allow_html=True)
    with middle:
        st.markdown('<div class="note-card"><b>2. 모델</b><br>현재 앱은 Gradient Boosting으로 실제 점수를 산출합니다. LightGBM은 별도 검증 후보이며 아직 교체되지 않았습니다.</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="note-card"><b>3. 운영</b><br>Threshold는 확률을 바꾸지 않습니다. 검토 대상의 범위와 FN·FP의 균형을 정합니다.</div>', unsafe_allow_html=True)
    with st.expander("모델 상태와 검증 한계"):
        st.markdown(f"**현재 데모 추론 모델:** `{metadata.get('model', 'unknown')}`")
        candidate_status_card(candidate)
        st.caption("제공 데이터에는 관측 기준일과 미래 이탈 기간 정의가 없습니다. 30일 이탈, 캠페인 Uplift, ROI는 검증되지 않았습니다.")


def customer_insights_v2(train: pd.DataFrame, insights: pd.DataFrame):
    header("고객 인사이트", "관찰값 → 해석 → 다음 검증 행동의 순서로, 실제로 활용 가능한 신호만 압축했습니다.")
    weekly_band = pd.qcut(train["weekly_hours"], q=4, duplicates="drop")
    weekly_rates = train.groupby(weekly_band, observed=True)["churned"].mean()
    skip_band = pd.qcut(train["song_skip_rate"], q=4, duplicates="drop")
    skip_rates = train.groupby(skip_band, observed=True)["churned"].mean()
    subscription_rates = train.groupby("subscription_type")["churned"].mean().sort_values(ascending=False)
    inquiry_rates = train.groupby("customer_service_inquiries")["churned"].mean().sort_values()
    cards = [
        (
            "청취 저하·스킵 패턴",
            f"주간 청취 시간 사분위 간 이탈 라벨 차이 {weekly_rates.max() - weekly_rates.min():.1%}, "
            f"스킵률 사분위 간 차이 {skip_rates.max() - skip_rates.min():.1%}",
            "다음 행동: 사용 저하 고객을 진단 우선 세그먼트로 분류하고, 실제 개선 개입은 실험으로 검증합니다.",
            "한계: 행동 저하가 이탈의 원인인지 결과인지 알 수 없습니다.",
        ),
        (
            "구독 유형",
            f"가장 높은 관측 이탈률은 {subscription_rates.index[0]} {subscription_rates.iloc[0]:.1%}, "
            f"가장 낮은 유형과의 차이는 {subscription_rates.iloc[0] - subscription_rates.iloc[-1]:.1%}",
            "다음 행동: 혜택·전환 제안의 우선 실험군을 설계합니다.",
            "한계: 제안이 실제 이탈을 줄이는지는 이 데이터만으로 알 수 없습니다.",
        ),
        (
            "고객 서비스 문의",
            f"문의 수준 간 관측 이탈 라벨 차이 {inquiry_rates.iloc[-1] - inquiry_rates.iloc[0]:.1%}",
            "다음 행동: 문의 해결 속도와 재문의율을 KPI로 둔 서비스 개선 실험을 설계합니다.",
            "한계: 문의가 이탈의 원인인지, 위험 고객의 결과인지 분리되지 않습니다.",
        ),
    ]
    for column, (title, observation, action, limit) in zip(st.columns(3), cards):
        with column:
            st.markdown(f'<div class="metric-card"><div class="metric-label">관찰</div><div class="metric-value" style="font-size:20px">{title}</div><div class="small-caption">{observation}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="note-card"><b>{action}</b><br><span class="small-caption">{limit}</span></div>', unsafe_allow_html=True)
    with st.expander("모델 신호이지만 행동 판단에는 주의할 항목"):
        st.markdown("구독 일시정지 횟수는 진단 신호가 될 수 있습니다. 연령도 모델 신호에 기여하지만, 행동 대상 선정에 사용할 때는 공정성·정책 검토가 선행되어야 합니다.")
    if not insights.empty:
        st.caption("근거: 저장된 EDA·Feature Importance 인사이트 목록. 모든 수치는 제공된 학습 데이터의 라벨 기준 관찰값입니다.")


def improvement_journey_v2(preprocessing_register: pd.DataFrame, fair_comparison: pd.DataFrame, candidate: dict):
    header("모델 개선 과정", "같은 평가 조건에서 확인된 Feature 개선과 모델 비교를 분리해, 무엇이 실제로 개선됐는지 보여줍니다.")
    st.markdown('<div class="section-label">1. Feature 선택 — 동일 Logistic OOF 비교</div>', unsafe_allow_html=True)
    if preprocessing_register.empty:
        st.warning("저장된 전처리 실험 결과를 찾을 수 없습니다.")
    else:
        raw = preprocessing_register.loc[preprocessing_register["variant"].eq("raw")].iloc[0]
        selected = preprocessing_register.loc[preprocessing_register["variant"].eq("log_numeric")].iloc[0]
        c1, c2, c3 = st.columns(3)
        with c1:
            metric_card("Raw Feature PR-AUC", f"{float(raw['logistic_pr_auc']):.3f}", "동일 Logistic OOF")
        with c2:
            metric_card("채택 Feature PR-AUC", f"{float(selected['logistic_pr_auc']):.3f}", "log_numeric")
        with c3:
            metric_card("동일 조건 개선", f"{float(selected['logistic_pr_auc']) - float(raw['logistic_pr_auc']):+.3f}", "PR-AUC 차이")
        display = preprocessing_register[["variant", "logistic_pr_auc", "decision", "reason"]].copy()
        display.columns = ["Feature Variant", "Logistic OOF PR-AUC", "결정", "근거"]
        st.dataframe(display, hide_index=True, width="stretch")
        st.caption("비율·상호작용·신호 축소 Variant는 Raw보다 낮아 제외했고, `log_numeric`만 동일 조건에서 가장 높아 잠정 채택했습니다.")
    st.markdown('<div class="section-label">2. 모델 비교 — 동일 Feature 5-Fold OOF</div>', unsafe_allow_html=True)
    if not fair_comparison.empty:
        ranked = fair_comparison.sort_values("pr_auc", ascending=False).copy()
        fig = px.bar(ranked.head(4), x="model", y="pr_auc", color="model", text=ranked.head(4)["pr_auc"].map(lambda value: f"{value:.3f}"))
        fig.update_layout(title="상위 4개 모델의 공통 Feature OOF PR-AUC", showlegend=False, xaxis_title=None, yaxis_title="PR-AUC")
        st.plotly_chart(plotly_theme(fig), width="stretch")
        st.caption("이 비교에서는 CatBoost와 LightGBM의 PR-AUC 차이가 매우 작습니다. 단일 OOF 점수만으로 운영 모델을 확정하지 않습니다.")
    st.markdown('<div class="section-label">3. 저장된 후보 상태</div>', unsafe_allow_html=True)
    candidate_status_card(candidate)
    st.warning("현재 결론: 공정 비교·튜닝·안정성 검증이 모두 끝나기 전까지는 ‘최종 운영 모델’이 아니라 ‘검토 중인 기술 후보’로만 표현합니다.")


def model_comparison_v2(fair_comparison: pd.DataFrame, candidate: dict):
    header("모델 비교·검증 상태", "최고 점수와 운영 후보를 구분하고, 현재 확정 가능한 결론만 제시합니다.")
    if fair_comparison.empty:
        st.warning("저장된 공통 Feature 모델 비교 결과를 찾을 수 없습니다.")
        return
    ranked = fair_comparison.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    best, lightgbm = ranked.iloc[0], ranked.loc[ranked["model"].eq("lightgbm")].iloc[0]
    c1, c2, c3 = st.columns(3)
    with c1:
        metric_card("공정 OOF 최고 모델", str(best["model"]).upper(), f"PR-AUC {float(best['pr_auc']):.3f}")
    with c2:
        metric_card("LightGBM OOF PR-AUC", f"{float(lightgbm['pr_auc']):.3f}", f"최고 모델과 {float(best['pr_auc']) - float(lightgbm['pr_auc']):.4f} 차이")
    with c3:
        metric_card("현재 운영 모델", "미확정", "후보 검토 중")
    st.markdown('<div class="info-card"><b>이 화면의 결론</b><br>공통 Feature OOF 비교의 최고 점수는 CatBoost입니다. 그러나 차이가 작고, 저장된 LightGBM 후보는 별도 P0 검증 Run의 결과입니다. 두 결과를 합쳐 ‘LightGBM이 최종 선정됐다’고 말할 수는 없습니다.</div>', unsafe_allow_html=True)
    display = ranked[["model", "pr_auc", "roc_auc", "recall", "precision", "fn", "fp"]].copy()
    display.columns = ["모델", "PR-AUC", "ROC-AUC", "Recall", "Precision", "FN", "FP"]
    st.dataframe(display, hide_index=True, width="stretch")
    with st.expander("저장된 LightGBM 후보 근거"):
        candidate_status_card(candidate)
        st.caption("다음 결정: 동일 공정 조건의 튜닝·Seed 안정성·Bootstrap 검증이 완료된 뒤 기술 선정과 운영 승격을 별도로 승인합니다.")


def operations_v2(targeting: pd.DataFrame, deciles: pd.DataFrame, scenarios: pd.DataFrame):
    header("운영 시나리오", "확률을 바꾸지 않고, 제한된 인력으로 어느 범위까지 검토할지를 고르는 화면입니다.")
    scenario_ids = scenarios["scenario_id"].tolist()
    selected_id = st.selectbox("비교할 운영 시나리오", scenario_ids, format_func=lambda value: scenario_label(get_scenario(scenarios, value)))
    active = get_scenario(scenarios, selected_id)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("검토 대상 비율", f"{float(active['test_target_rate']):.1%}", "내부 Holdout")
    with c2:
        metric_card("Recall", f"{float(active['test_recall']):.1%}", "포착 범위")
    with c3:
        metric_card("Precision", f"{float(active['test_precision']):.1%}", "검토 효율")
    with c4:
        metric_card("FN / FP", f"{int(active['test_fn']):,} / {int(active['test_fp']):,}", "같은 시나리오 기준")
    st.markdown(f'<div class="note-card"><b>{active["scenario_label"]}</b> — {active["selection_rule"]}<br>이 선택은 위험 점수를 바꾸지 않고, 누구를 검토 대상으로 표시할지만 바꿉니다.</div>', unsafe_allow_html=True)
    scenario_view = scenarios[["scenario_label", "threshold", "test_target_rate", "test_precision", "test_recall", "test_f1", "test_fn", "test_fp"]].copy()
    scenario_view.insert(0, "현재 선택", scenario_view["scenario_label"].eq(active["scenario_label"]).map({True: "●", False: ""}))
    scenario_view.columns = ["선택", "시나리오", "Threshold", "대상 비율", "Precision", "Recall", "F1", "FN", "FP"]
    st.dataframe(scenario_view, hide_index=True, width="stretch")
    left, right = st.columns(2)
    with left:
        fig = go.Figure(go.Bar(x=targeting["top_percent"].map(lambda value: f"Top {value:g}%"), y=targeting["capture_rate"], marker_color="#E4573D", text=targeting["capture_rate"].map(lambda value: f"{value:.1%}"), textposition="outside"))
        fig.update_layout(title="내부 Holdout Top-K Capture", yaxis_tickformat=".0%", yaxis_range=[0, 1.08], showlegend=False)
        st.plotly_chart(plotly_theme(fig), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["actual_churn_rate"], name="관측 라벨", marker_color="#E4573D"))
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["mean_predicted_probability"], name="예측 점수", marker_color="#315A7D"))
        fig.update_layout(title="내부 Holdout Risk Decile 점검", barmode="group", yaxis_tickformat=".0%", xaxis_title="Decile (10 = 최고 위험)")
        st.plotly_chart(plotly_theme(fig), width="stretch")
    st.caption("운영 지표는 현재 저장된 Gradient Boosting 데모 Pipeline의 내부 Holdout 결과입니다. 실제 캠페인 Uplift·ROI는 통제 실험으로 별도 측정해야 합니다.")


def model_page(
    comparison: pd.DataFrame,
    threshold: pd.DataFrame,
    metadata: dict,
    importance: pd.DataFrame,
    targeting: pd.DataFrame,
    deciles: pd.DataFrame,
    scenarios: pd.DataFrame,
    active_scenario: pd.Series,
    candidate: dict,
):
    header("Model Lab", "후보 모델 성능과 FN 우선 threshold trade-off를 확인합니다. Gradient Boosting을 최종 모델로 선정했습니다.")
    st.markdown(
        '<div class="note-card"><b>Operational-demo evidence</b><br>'
        'The charts and threshold scenarios below belong to the saved Gradient Boosting demo pipeline. '
        'They are not evidence that the separate LightGBM candidate has been deployed.</div>',
        unsafe_allow_html=True,
    )
    with st.expander("Technical candidate status"):
        candidate_status_card(candidate)
    best = comparison.iloc[0]
    st.markdown(f'<div class="note-card"><b>최종 선정 모델: {best["model"]}</b><br>Validation expected cost 기준으로 최종 선정했습니다. 비용비 FN:FP={metadata.get("false_negative_cost", 3):g}:1과 운영 threshold는 팀 가정이며 실제 사업 수치로 재검토해야 합니다.</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Test PR-AUC", f"{best['test_pr_auc']:.3f}", "ranking quality")
    with m2: metric_card("Test Recall", f"{float(active_scenario['test_recall']):.1%}", str(active_scenario["scenario_label"]))
    with m3: metric_card("Test Precision", f"{float(active_scenario['test_precision']):.1%}", str(active_scenario["scenario_label"]))
    with m4: metric_card("Threshold", f"{float(active_scenario['threshold']):.2f}", "사용자 선택 시나리오")

    left, right = st.columns([1.1, 1])
    with left:
        display = comparison[["model", "validation_pr_auc", "test_pr_auc", "validation_operating_recall", "test_operating_recall", "test_operating_precision", "test_expected_cost_per_customer"]].copy()
        display.columns = ["model", "Val PR-AUC", "Test PR-AUC", "Val Recall", "Test Recall", "Test Precision", "Test cost/customer"]
        st.markdown('<div class="section-label">후보 비교</div>', unsafe_allow_html=True)
        st.dataframe(display.style.format({c: "{:.3f}" for c in display.columns if c != "model"}), hide_index=True, width="stretch")
    with right:
        chart_df = comparison.sort_values("test_pr_auc")
        fig = go.Figure()
        fig.add_trace(go.Bar(name="PR-AUC", x=chart_df["test_pr_auc"], y=chart_df["model"], orientation="h", marker_color="#315A7D"))
        fig.update_layout(title="Test PR-AUC", xaxis_title="PR-AUC", yaxis_title=None, showlegend=False)
        st.plotly_chart(plotly_theme(fig), width="stretch")

    if not importance.empty:
        st.markdown('<div class="section-label">Model interpretation</div>', unsafe_allow_html=True)
        top = importance.head(12).sort_values("importance")
        fig = go.Figure(go.Bar(x=top["importance"], y=top["feature"], orientation="h", marker_color="#e4573d"))
        fig.update_layout(title="Top encoded feature importance", xaxis_title="Importance", yaxis_title=None, height=430, showlegend=False)
        st.plotly_chart(plotly_theme(fig), width="stretch")
        st.markdown('<div class="small-caption">Importance는 Gradient Boosting의 모델 내부 연관 신호입니다. 원인·효과나 캠페인 효과를 의미하지 않으며, 범주형 값은 one-hot 인코딩된 상태로 표시됩니다.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Threshold trade-off</div>', unsafe_allow_html=True)
    curve = threshold.copy()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["recall"], name="Recall", line=dict(color="#E4573D", width=3)))
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["precision"], name="Precision", line=dict(color="#315A7D", width=3)))
    fig.add_trace(go.Scatter(x=curve["threshold"], y=curve["expected_cost_per_customer"], name="Expected cost/customer", line=dict(color="#8a6f2f", dash="dot")))
    fig.update_layout(title="Validation threshold curve", xaxis_title="Threshold", yaxis_title="Metric / cost", yaxis_range=[0, 1])
    st.plotly_chart(plotly_theme(fig), width="stretch")
    st.markdown('<div class="small-caption">Threshold가 낮아지면 Recall은 보통 올라가지만 FP도 늘어납니다. 현재 비용비 3:1은 팀 가정이며, 운영 수용량과 실제 FN/FP 비용으로 재설정해야 합니다.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">사용자가 고르는 운영 시나리오</div>', unsafe_allow_html=True)
    scenario_view = scenarios[[
        "scenario_label", "selection_rule", "threshold", "test_target_rate", "test_precision", "test_recall", "test_f1"
    ]].copy()
    scenario_view.columns = ["시나리오", "선택 기준", "Threshold", "내부 Test 대상 비율", "Test Precision", "Test Recall", "Test F1"]
    st.dataframe(
        scenario_view.style.format({
            "Threshold": "{:.2f}",
            "내부 Test 대상 비율": "{:.1%}",
            "Test Precision": "{:.1%}",
            "Test Recall": "{:.1%}",
            "Test F1": "{:.3f}",
        }),
        hide_index=True,
        width="stretch",
    )
    st.caption("세 시나리오는 모두 Validation에서 선택하고 내부 Test holdout에서 1회 평가했습니다. 어느 시나리오도 실제 캠페인 성과를 보장하지 않습니다.")

    left, right = st.columns(2)
    with left:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=targeting["top_percent"].map(lambda value: f"Top {value:g}%"),
            y=targeting["capture_rate"],
            name="Capture rate",
            marker_color="#E4573D",
            text=targeting["capture_rate"].map(lambda value: f"{value:.1%}"),
            textposition="outside",
        ))
        fig.update_layout(title="내부 Test: 상위 구간별 실제 라벨 포착률", yaxis_tickformat=".0%", yaxis_range=[0, 1.08], showlegend=False)
        st.plotly_chart(plotly_theme(fig), width="stretch")
    with right:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["actual_churn_rate"], name="Actual", marker_color="#E4573D"))
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["mean_predicted_probability"], name="Predicted", marker_color="#315A7D"))
        fig.update_layout(title="내부 Test: 위험 Decile 보정 진단", barmode="group", xaxis_title="위험 decile (10=최고)", yaxis_tickformat=".0%")
        st.plotly_chart(plotly_theme(fig), width="stretch")
    st.caption("확률과 실제 라벨 비율의 차이는 보정 오차 진단입니다. 따라서 앱의 확률은 ‘확정 해지 확률’이 아니라 우선순위 점수로 사용합니다.")

    cm = np.array([
        [int(active_scenario["test_tn"]), int(active_scenario["test_fp"])],
        [int(active_scenario["test_fn"]), int(active_scenario["test_tp"])],
    ])
    fig = px.imshow(
        cm,
        text_auto=True,
        x=["미대상", "대상"],
        y=["실제 라벨 유지", "실제 라벨 이탈"],
        color_continuous_scale="Blues",
        title=f"내부 Test 혼동행렬 — {active_scenario['scenario_label']}",
    )
    st.plotly_chart(plotly_theme(fig), width="stretch")

    with st.expander("모델 한계와 데이터 리스크"):
        st.markdown("""
        - 현재 Test는 내부 분할 Test이며, 제공 `data/test.csv`는 정답이 없어 실제 성능을 검증하지 못합니다.
        - `churned`가 향후 30일 내 해지를 뜻한다는 생성 규칙이 확인되지 않았습니다.
        - 성능이 낮거나 특정 세그먼트에서 달라질 수 있으므로 자동 할인·차단에 사용하지 않습니다.
        - 변수 중요도는 곧바로 해지 원인이나 캠페인 효과를 의미하지 않습니다.
        """)


def prediction_page(train: pd.DataFrame, model, metadata: dict, scenarios: pd.DataFrame, active_scenario: pd.Series, show_header: bool = True):
    if show_header:
        header("Customer Scoring", "고객 정보를 입력하면 저장된 Pipeline이 이탈 위험 점수와 리텐션 확인 포인트를 반환합니다.")
    threshold = float(active_scenario["threshold"])
    st.markdown(f'<div class="note-card"><b>모델 실행 방식</b><br>최종 선정된 `{metadata["model"]}` Pipeline만 로드합니다. 현재 선택은 <b>{active_scenario["scenario_label"]}</b> (`{threshold:.2f}`)이며, {active_scenario["selection_rule"]} 기준의 운영 가정입니다.</div>', unsafe_allow_html=True)

    with st.form("customer_form"):
        st.markdown('<div class="section-label">Subscription & profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=int(train.age.median()), step=1)
            location = st.selectbox("Location", sorted(train.location.unique()), index=sorted(train.location.unique()).index("California") if "California" in train.location.unique() else 0)
            subscription_type = st.selectbox("Subscription type", sorted(train.subscription_type.unique()))
        with c2:
            payment_plan = st.selectbox("Payment plan", sorted(train.payment_plan.unique()))
            pause_value = st.selectbox("num_subscription_pauses (실제 값)", sorted(train.num_subscription_pauses.unique()))
            payment_method = st.selectbox("Payment method", sorted(train.payment_method.unique()))
        with c3:
            customer_service = st.selectbox("Customer service inquiries", sorted(train.customer_service_inquiries.unique()))
            if pd.api.types.is_numeric_dtype(train.signup_date):
                signup_date = st.number_input(
                    "Signup date (기준일 대비 일수)",
                    min_value=int(train.signup_date.min()),
                    max_value=int(train.signup_date.max()),
                    value=int(train.signup_date.median()),
                    step=1,
                )
            else:
                signup_default = pd.to_datetime(train.signup_date).median().date()
                signup_date = st.date_input("Signup date", value=signup_default)
            st.caption("가입 시점은 파생 feature로만 사용됩니다.")

        st.markdown('<div class="section-label">Listening behavior</div>', unsafe_allow_html=True)
        numeric_cols = ["weekly_hours", "average_session_length", "song_skip_rate", "weekly_songs_played", "weekly_unique_songs", "num_favorite_artists", "num_platform_friends", "num_playlists_created", "num_shared_playlists", "notifications_clicked"]
        defaults = {c: float(train[c].median()) for c in numeric_cols}
        caps = {c: float(train[c].max()) * 1.5 for c in numeric_cols}
        row1 = st.columns(5)
        weekly_hours = row1[0].number_input("Weekly hours", min_value=0.0, max_value=caps["weekly_hours"], value=defaults["weekly_hours"], step=0.1)
        avg_session = row1[1].number_input("Avg session length", min_value=0.0, max_value=caps["average_session_length"], value=defaults["average_session_length"], step=0.05)
        skip_rate = row1[2].number_input("Song skip rate", min_value=0.0, max_value=1.0, value=defaults["song_skip_rate"], step=0.01)
        songs = row1[3].number_input("Weekly songs played", min_value=0, max_value=int(caps["weekly_songs_played"]), value=int(defaults["weekly_songs_played"]), step=1)
        unique_songs = row1[4].number_input("Weekly unique songs", min_value=0, max_value=int(caps["weekly_unique_songs"]), value=int(defaults["weekly_unique_songs"]), step=1)
        row2 = st.columns(5)
        favorites = row2[0].number_input("Favorite artists", min_value=0, max_value=int(caps["num_favorite_artists"]), value=int(defaults["num_favorite_artists"]), step=1)
        friends = row2[1].number_input("Platform friends", min_value=0, max_value=int(caps["num_platform_friends"]), value=int(defaults["num_platform_friends"]), step=1)
        playlists = row2[2].number_input("Playlists created", min_value=0, max_value=int(caps["num_playlists_created"]), value=int(defaults["num_playlists_created"]), step=1)
        shared = row2[3].number_input("Shared playlists", min_value=0, max_value=int(caps["num_shared_playlists"]), value=int(defaults["num_shared_playlists"]), step=1)
        notifications = row2[4].number_input("Notifications clicked", min_value=0, max_value=int(caps["notifications_clicked"]), value=int(defaults["notifications_clicked"]), step=1)
        submitted = st.form_submit_button("이 고객의 위험 점수 계산", type="primary", width="stretch")

    if submitted:
        row = pd.DataFrame(
            [{
                "customer_id": -1,
                "age": age,
                "location": location,
                "subscription_type": subscription_type,
                "payment_plan": payment_plan,
                "num_subscription_pauses": pause_value,
                "payment_method": payment_method,
                "customer_service_inquiries": customer_service,
                "signup_date": signup_date,
                "weekly_hours": weekly_hours,
                "average_session_length": avg_session,
                "song_skip_rate": skip_rate,
                "weekly_songs_played": songs,
                "weekly_unique_songs": unique_songs,
                "num_favorite_artists": favorites,
                "num_platform_friends": friends,
                "num_playlists_created": playlists,
                "num_shared_playlists": shared,
                "notifications_clicked": notifications,
            }]
        )
        input_columns = metadata.get("input_columns", [])
        missing = sorted(set(input_columns) - set(row.columns))
        unexpected = sorted(set(row.columns) - set(input_columns))
        if missing or unexpected:
            st.error(f"입력 스키마가 저장된 모델과 맞지 않습니다. 누락={missing}, 추가={unexpected}")
            st.stop()
        row = row.reindex(columns=input_columns)
        probability = float(model.predict_proba(row)[:, 1][0])
        level, css = risk_level(probability, scenarios)
        st.markdown('<div class="section-label">Scoring result</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1: metric_card("Dataset-label churn probability", f"{probability:.1%}", "model score")
        with r2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Risk band</div><div class="metric-value"><span class="{css}">{level}</span></div><div class="metric-delta">현재 선택 threshold {threshold:.2f}</div></div>', unsafe_allow_html=True)
        with r3: metric_card("Decision", "우선 검토 후보" if probability >= threshold else "일반 모니터링", "자동 조치 아님")
        st.markdown('<div class="section-label">리텐션 검토 포인트</div>', unsafe_allow_html=True)
        actions = []
        type_rates_all = train.groupby("subscription_type")["churned"].mean()
        if float(type_rates_all.get(subscription_type, 0)) >= train["churned"].mean() * 1.2:
            actions.append(f"{subscription_type} 요금제: 평균 대비 높은 이탈 세그먼트 — 전환·혜택 실험 후보")
        try:
            if float(pause_value) >= 3:
                actions.append("구독 일시정지 3회 이상: 관찰상 연관 신호 — 정지 흐름 개선 실험 후보")
        except (TypeError, ValueError):
            pass
        inquiry_rates = train.groupby("customer_service_inquiries")["churned"].mean()
        if float(inquiry_rates.get(customer_service, 0)) >= train["churned"].mean() * 1.2:
            actions.append("상담 문의 상위 구간: 상담 경험 개선을 위한 실험 후보")
        if skip_rate >= train.song_skip_rate.quantile(.75): actions.append("높은 skip rate: 개인화 추천·탐색 경험의 추가 진단 후보")
        if weekly_hours <= train.weekly_hours.quantile(.25): actions.append("낮은 주간 청취: 재활성화 메시지·콘텐츠 추천 실험 후보")
        if not actions: actions.append("현재 입력만으로는 강한 단일 신호가 없으므로 전체 세그먼트 맥락과 함께 검토")
        for action in actions:
            st.markdown(f"- {action}")
        st.caption("위 내용은 예측 신호에 기반한 실험 후보이며, 해지 원인·캠페인 효과를 의미하지 않습니다.")


def batch_page(predictions: pd.DataFrame, scenarios: pd.DataFrame, active_scenario: pd.Series, show_header: bool = True):
    if show_header:
        header("Batch Prioritization", "정답이 없는 제공 test 고객의 예측 분포와 우선 검토 대상을 확인합니다.")
    threshold = float(active_scenario["threshold"])
    st.markdown(
        f'<div class="note-card"><b>현재 선택: {active_scenario["scenario_label"]}</b><br>'
        f'{active_scenario["selection_rule"]} · threshold {threshold:.2f}. 아래 CSV는 검토 후보 목록일 뿐이며 외부 발송·할인·계정 조치를 실행하지 않습니다.</div>',
        unsafe_allow_html=True,
    )
    min_probability = st.slider("최소 이탈 확률", 0.0, 1.0, threshold, 0.01)
    filtered = predictions[predictions["churn_probability"] >= min_probability].sort_values("churn_probability", ascending=False)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("전체 test", f"{len(predictions):,}", "정답 없음")
    with c2: metric_card("현재 대상", f"{len(filtered):,}", f"probability ≥ {min_probability:.2f}")
    with c3: metric_card("비율", f"{len(filtered) / len(predictions):.1%}", "예측 대상 비중")
    st.markdown('<div class="info-card"><b>주의</b><br>제공 test에는 정답이 없으므로 고위험 고객 목록의 정확도나 실제 유지 효과는 여기서 검증할 수 없습니다. 운영 전에는 라벨과 캠페인 결과를 연결해야 합니다.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">우선 검토 고객</div>', unsafe_allow_html=True)
    view = filtered.head(500).copy()
    view["risk_band"] = view["churn_probability"].map(lambda value: risk_level(float(value), scenarios)[0])
    st.dataframe(view, hide_index=True, width="stretch")
    st.download_button("현재 필터 결과 CSV 다운로드", data=filtered.to_csv(index=False).encode("utf-8-sig"), file_name="playlistpro_churn_priorities.csv", mime="text/csv", width="stretch")


def prioritization_page(
    train: pd.DataFrame,
    model,
    metadata: dict,
    predictions: pd.DataFrame,
    scenarios: pd.DataFrame,
    active_scenario: pd.Series,
):
    header("Customer Prioritization", "개별 고객 확인과 무라벨 제공 test 고객의 배치 우선순위를 한 곳에서 검토합니다.")
    scenario_ids = scenarios["scenario_id"].tolist()
    selected_id = st.selectbox(
        "고객 우선순위 판단 기준",
        scenario_ids,
        index=scenario_ids.index(active_scenario["scenario_id"]) if active_scenario["scenario_id"] in scenario_ids else 0,
        format_func=lambda scenario_id: scenario_label(get_scenario(scenarios, scenario_id)),
        help="이 화면에서만 적용되는 위험 등급·대상 판단 기준입니다.",
    )
    active_scenario = get_scenario(scenarios, selected_id)
    st.caption("위험 확률은 저장된 Pipeline이 계산합니다. 여기서 고르는 기준은 위험 등급과 검토 대상 표시만 바꾸며, 모델이나 확률 자체를 다시 계산하지 않습니다.")
    scoring_tab, batch_tab = st.tabs(["개별 고객 점수", "배치 우선순위"])
    with scoring_tab:
        prediction_page(train, model, metadata, scenarios, active_scenario, show_header=False)
    with batch_tab:
        batch_page(predictions, scenarios, active_scenario, show_header=False)


def simulator_page(test: pd.DataFrame, predictions: pd.DataFrame, metadata: dict, active_scenario: pd.Series):
    header("Campaign Simulator", "예산·인원·상위 % 중 하나를 기준으로 가정 기반의 캠페인 우선순위를 비교합니다.")
    st.markdown(
        f'<div class="note-card"><b>가정 기반 시뮬레이션</b><br>현재 운영 선택은 <b>{active_scenario["scenario_label"]}</b>입니다. 접촉 비용·고객 가치·유지 전환율은 모두 사용자가 입력하는 가정값이며, 화면은 계산 프레임만 제공합니다. '
        '모델 점수의 합은 확정 이탈자 수가 아니며, 제공 test에는 정답 라벨과 실제 캠페인 결과가 없습니다.</div>',
        unsafe_allow_html=True,
    )

    base = predictions.merge(test[["customer_id", "subscription_type"]], on="customer_id", how="inner")
    base = base.rename(columns={"churn_probability": "prob", "subscription_type": "plan"})[["customer_id", "prob", "plan"]]
    plan_list = base.groupby("plan")["prob"].mean().sort_values(ascending=False).index.tolist()

    ss = st.session_state
    ss.setdefault("sim_driver", "budget")
    ss.setdefault("sim_budget", 5_000_000)
    ss.setdefault("sim_count", 1666)
    ss.setdefault("sim_pct", 16.7)

    def _set_driver(name: str):
        st.session_state["sim_driver"] = name

    st.markdown('<div class="section-label">캠페인 규모</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-caption">셋 중 아무 값이나 수정하면 나머지 둘이 자동 환산됩니다.</div>', unsafe_allow_html=True)
    scale_box = st.container()

    def render_scale_inputs():
        with scale_box:
            a, b, c = st.columns(3)
            a.number_input("캠페인 예산 (원)", min_value=0, step=500_000, key="sim_budget", on_change=_set_driver, args=("budget",))
            b.number_input("연락 인원 (명)", min_value=0, step=50, key="sim_count", on_change=_set_driver, args=("count",))
            c.number_input("상위 위험군 (%)", min_value=0.0, max_value=100.0, step=1.0, key="sim_pct", on_change=_set_driver, args=("pct",))

    col_sr, col_sort = st.columns([1.4, 1])
    with col_sr:
        success_rate = st.slider("방어 성공률 (%)", 5, 50, 20, 5) / 100.0

    kpi_box = st.container()

    st.markdown('<div class="section-label">요금제별 설정</div>', unsafe_allow_html=True)
    st.markdown('<div class="small-caption">\'포함\'을 끄면 해당 요금제는 산식(배정·퍼널·곡선)에서 제외됩니다. 기본값 동일 = 전역 가정과 같은 동작.</div>', unsafe_allow_html=True)
    counts = base.groupby("plan")["prob"].agg(["size", "mean"])
    cfg = st.data_editor(
        pd.DataFrame(
            {
                "요금제": plan_list,
                "보유 고객": [int(counts.loc[p, "size"]) for p in plan_list],
                "평균 위험도": [f"{counts.loc[p, 'mean']:.1%}" for p in plan_list],
                "포함": True,
                "접촉 비용(원)": 3000,
                "고객 가치(원/년)": 120000,
            }
        ),
        hide_index=True,
        disabled=["요금제", "보유 고객", "평균 위험도"],
        column_config={
            "포함": st.column_config.CheckboxColumn("포함"),
            "접촉 비용(원)": st.column_config.NumberColumn("접촉 비용(원)", min_value=0, step=500),
            "고객 가치(원/년)": st.column_config.NumberColumn("고객 가치(원/년)", min_value=0, step=10000),
        },
        width="stretch",
    ).set_index("요금제")

    included = [p for p in plan_list if bool(cfg.loc[p, "포함"])]
    if not included:
        render_scale_inputs()
        st.warning("산식에 포함된 요금제가 없습니다. 하나 이상 선택해주세요.")
        return

    work = base[base["plan"].isin(included)].copy()
    work["cost"] = work["plan"].map(cfg["접촉 비용(원)"].astype(float))
    work["ltv"] = work["plan"].map(cfg["고객 가치(원/년)"].astype(float))
    work["ev"] = work["prob"] * success_rate * work["ltv"] - work["cost"]
    work = work.sort_values(["ev", "prob"], ascending=False).reset_index(drop=True)
    pool = len(work)
    cum_spend = work["cost"].cumsum().values

    driver = ss["sim_driver"]
    if driver == "budget":
        n_sel = int((cum_spend <= float(ss["sim_budget"])).sum())
    elif driver == "count":
        n_sel = int(min(int(ss["sim_count"]), pool))
    else:
        n_sel = int(round(float(ss["sim_pct"]) / 100.0 * pool))
    n_sel = max(0, min(n_sel, pool))

    sel = work.iloc[:n_sel]
    spend = float(sel["cost"].sum())
    caught = float(sel["prob"].sum())
    saved = caught * success_rate
    revenue = float((sel["prob"] * success_rate * sel["ltv"]).sum())
    net = revenue - spend

    if driver != "budget":
        ss["sim_budget"] = int(spend)
    if driver != "count":
        ss["sim_count"] = int(n_sel)
    if driver != "pct":
        ss["sim_pct"] = round(n_sel / pool * 100, 1) if pool else 0.0
    render_scale_inputs()

    counts_all = base.groupby("plan")["prob"].mean()
    plan_result = []
    for p in plan_list:
        if p in included:
            row_ev = float(counts_all.loc[p]) * success_rate * float(cfg.loc[p, "고객 가치(원/년)"]) - float(cfg.loc[p, "접촉 비용(원)"])
            plan_result.append([p, f"{row_ev:,.0f}원", f"{int((sel['plan'] == p).sum()):,}명"])
        else:
            plan_result.append([p, "제외", "-"])
    st.dataframe(pd.DataFrame(plan_result, columns=["요금제", "기대이익/인(평균 위험도 기준)", "연락 배정"]), hide_index=True, width="stretch")

    uniform = cfg["접촉 비용(원)"].nunique() == 1 and cfg["고객 가치(원/년)"].nunique() == 1
    sort_label = "이탈 확률 순 (요금제별 값 동일 → 기대이익 순과 일치)" if uniform else "기대이익 순 (요금제별 값 차등 반영)"
    with col_sort:
        st.markdown(f'<div class="note-card" style="margin-top:28px"><b>정렬 기준</b><br>{sort_label}</div>', unsafe_allow_html=True)

    with kpi_box:
        st.markdown(f'<div class="small-caption">대상 풀 {pool:,}명 · 기대 이탈자 {work["prob"].sum():,.0f}명</div>', unsafe_allow_html=True)
        k1, k2, k3, k4 = st.columns(4)
        with k1: metric_card("집행액", f"{spend:,.0f}원", f"{n_sel:,}명 연락")
        with k2: metric_card("확률 합산 위험도", f"{caught:,.0f}", f"가정상 유지 전환 {saved:,.0f} (전환율 {success_rate:.0%})")
        with k3: metric_card("가정 기반 편익", f"{revenue:,.0f}원", "고객가치·전환율 가정 반영")
        with k4: metric_card("가정 기반 순편익", f"{net:,.0f}원", f"가정상 ROI {net / spend:.0%}" if spend > 0 else "-")

    st.markdown('<div class="section-label">캠페인 퍼널</div>', unsafe_allow_html=True)
    miss = max(n_sel - caught, 0.0)
    stages = ["가정상 유지 전환", "확률 합산 위험도", "연락 대상", "대상 고객(선택 요금제)"]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=stages, x=[0, 0, 0, pool], orientation="h", name="대상 고객", marker_color="#b4b2a9", showlegend=False))
    fig.add_trace(go.Bar(y=stages, x=[0, caught, caught, 0], orientation="h", name="확률 합산 위험도", marker_color="#d03b3b"))
    fig.add_trace(go.Bar(y=stages, x=[0, 0, miss, 0], orientation="h", name="비위험 추정", marker_color="#eda100"))
    fig.add_trace(go.Bar(y=stages, x=[saved, 0, 0, 0], orientation="h", name="가정상 유지 전환", marker_color="#008300", showlegend=False))
    fig.update_layout(barmode="stack", height=280, xaxis_range=[0, pool * 1.05], xaxis_title=None, yaxis_title=None)
    st.plotly_chart(plotly_theme(fig), width="stretch")
    if n_sel:
        st.markdown(f'<div class="small-caption">연락 대상 {n_sel:,}명 중 확률 합산 위험도 {caught:,.0f} · 비위험 추정 {n_sel - caught:,.0f} (평균 점수 {caught / n_sel:.1%})</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">연락 범위별 순이익 곡선</div>', unsafe_allow_html=True)
    curve_mode = st.radio("곡선 모드", ["현재 설정 기준", "단가 시나리오 비교"], horizontal=True)
    scen_costs = []
    if curve_mode == "단가 시나리오 비교":
        s1, s2, s3 = st.columns(3)
        scen_costs = [
            s1.number_input("시나리오 단가 1 (원)", min_value=0, value=3000, step=500),
            s2.number_input("시나리오 단가 2 (원)", min_value=0, value=6000, step=500),
            s3.number_input("시나리오 단가 3 (원)", min_value=0, value=9000, step=500),
        ]
    scen_style = [("#2a78d6", None), ("#1baf7a", "dash"), ("#eb6834", "dot")]

    def curve_arrays(sorted_df):
        rev_c = (sorted_df["prob"].values * success_rate * sorted_df["ltv"].values).cumsum()
        net_c = (rev_c - sorted_df["cost"].values.cumsum()) / 1e6
        x = np.arange(1, len(sorted_df) + 1) / len(sorted_df) * 100
        step = max(1, len(sorted_df) // 300)
        idx = np.unique(np.concatenate([np.arange(0, len(sorted_df), step), [len(sorted_df) - 1]]))
        return x[idx], net_c[idx], net_c

    def group_figure(sub_df, group_pool, marker_n):
        fig = go.Figure()
        if scen_costs:
            for c, (color, dash) in zip(scen_costs, scen_style):
                tmp = sub_df.copy()
                tmp["cost"] = float(c)
                tmp["ev"] = tmp["prob"] * success_rate * tmp["ltv"] - tmp["cost"]
                tmp = tmp.sort_values(["ev", "prob"], ascending=False)
                x, y, _ = curve_arrays(tmp)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"단가 {c:,.0f}원", line=dict(color=color, dash=dash, width=3)))
        else:
            x, y, full_net = curve_arrays(sub_df)
            fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="순이익", line=dict(color="#2a78d6", width=3)))
            if marker_n and marker_n > 0:
                fig.add_trace(go.Scatter(x=[marker_n / group_pool * 100], y=[full_net[marker_n - 1]], mode="markers", name="현재 배정 위치", marker=dict(size=11, color="#2a78d6")))
        fig.update_layout(height=340, xaxis_title="연락 범위 (그룹 내 상위 %)", yaxis_title="순이익 (백만원)")
        return plotly_theme(fig)

    tabs = st.tabs(["전체(선택)"] + plan_list)
    with tabs[0]:
        st.plotly_chart(group_figure(work, pool, n_sel), width="stretch")
        st.markdown('<div class="small-caption">산식에 포함된 요금제 합계 기준 — 기대이익 순 배정.</div>', unsafe_allow_html=True)
    for i, p in enumerate(plan_list, start=1):
        with tabs[i]:
            sub = base[base["plan"] == p].copy()
            sub["cost"] = float(cfg.loc[p, "접촉 비용(원)"])
            sub["ltv"] = float(cfg.loc[p, "고객 가치(원/년)"])
            sub = sub.sort_values("prob", ascending=False).reset_index(drop=True)
            marker = int((sel["plan"] == p).sum()) if p in included else 0
            st.plotly_chart(group_figure(sub, len(sub), marker), width="stretch")
            note = f"{p} — 대상 {len(sub):,}명, 기대 이탈자 {sub['prob'].sum():,.0f}명, 현재 배정 {marker:,}명."
            if p not in included:
                note += " (현재 산식에서 제외됨 — 곡선은 참고용)"
            st.markdown(f'<div class="small-caption">{note}</div>', unsafe_allow_html=True)


def main():
    (train, test), model, (comparison, threshold, metadata, predictions, importance, targeting, deciles, scenarios) = safe_load()
    candidate, progression, decision_matrix, insights, fair_comparison, preprocessing_register = load_presentation_evidence()
    default_scenario_id = metadata.get("app_default_scenario_id", "balanced_f1")
    active_scenario = get_scenario(scenarios, default_scenario_id)
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">♫</span><span class="brand-name">PlaylistPro</span></div>', unsafe_allow_html=True)
        st.caption("분석 워크스페이스")
        page = st.radio(
            "Workspace",
            ["프로젝트 요약", "고객 인사이트", "모델 개선 과정", "모델 비교·선정", "운영 시나리오", "고객 우선순위"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown(f"**데모 모델**  `{metadata['model']}`")
        if candidate:
            st.markdown(f"**후보 모델**  `{str(candidate.get('candidate_model', 'unknown')).upper()}`")
            candidate_status = str(candidate.get("status", "UNKNOWN"))
            sidebar_status = "후보 저장 완료 · 검토 대기" if candidate_status == "MODEL_CANDIDATE_SAVED_AWAITING_REVIEW" else candidate_status
            st.caption(sidebar_status)
        st.markdown("<div class='small-caption'>데이터셋 라벨 기준 점수<br>30일 예측 기간 미검증<br>캠페인 효과 미검증</div>", unsafe_allow_html=True)
        with st.expander("참고 분석"):
            show_simulator = st.checkbox(
                "가정 기반 캠페인 시뮬레이터 열기",
                help="실제 Uplift나 ROI가 아닌 계획 민감도 분석 도구입니다.",
            )

    if page == "프로젝트 요약":
        project_summary_v2(train, test, metadata, candidate, targeting)
    elif page == "고객 인사이트":
        customer_insights_v2(train, insights)
    elif page == "모델 개선 과정":
        improvement_journey_v2(preprocessing_register, fair_comparison, candidate)
    elif page == "모델 비교·선정":
        model_comparison_v2(fair_comparison, candidate)
    elif page == "운영 시나리오":
        operations_v2(targeting, deciles, scenarios)
    else:
        prioritization_page(train, model, metadata, predictions, scenarios, active_scenario)

    if show_simulator:
        st.divider()
        st.caption("Reference analysis — assumption-based only")
        simulator_page(test, predictions, metadata, active_scenario)


if __name__ == "__main__":
    main()
