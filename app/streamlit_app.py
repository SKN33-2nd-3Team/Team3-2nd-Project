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
    [data-testid="stSidebar"] * { color: #e8eef5 !important; }
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
def load_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    comparison = pd.read_csv(ARTIFACT_DIR / "model_comparison.csv")
    threshold = pd.read_csv(ARTIFACT_DIR / "threshold_sweep_validation.csv")
    importance_path = ARTIFACT_DIR / "feature_importance.csv"
    importance = pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame(columns=["feature", "importance", "rank"])
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_csv(ARTIFACT_DIR / "test_predictions.csv")
    return comparison, threshold, metadata, predictions, importance


def safe_load():
    required = [DATA_DIR / "train.csv", DATA_DIR / "test.csv", MODEL_PATH, METADATA_PATH]
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


def risk_level(probability: float, threshold: float) -> tuple[str, str]:
    if probability >= threshold:
        return "고위험", "risk-high"
    if probability >= threshold * 0.65:
        return "관찰", "risk-mid"
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


def overview_page(train: pd.DataFrame, test: pd.DataFrame, comparison: pd.DataFrame, metadata: dict, predictions: pd.DataFrame):
    st.markdown(
        '<div class="hero"><div class="eyebrow" style="color:#ffb3a4">MUSIC RETENTION INTELLIGENCE</div><h1>이탈 신호를 읽고, 다음 행동을 설계하세요.</h1><p>구독·결제·청취 행동을 한 화면에서 탐색하고, 데이터셋 라벨 기준 이탈 위험을 고객 단위로 확인하는 PlaylistPro 분석 workspace입니다.</p><span class="tag">Light mode · Model inference only · FN 우선</span></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="info-card"><b>해석 범위 안내</b><br>현재 파일에는 관측 기준일·해지일·30일 라벨 생성 규칙이 없습니다. 따라서 아래 확률은 <b>데이터셋의 churned 라벨 기준</b>이며, 실제 향후 30일 해지 확률로 확정 표시하지 않습니다.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">오늘의 데이터 snapshot</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("학습 고객", f"{len(train):,}", "target 포함")
    with c2:
        metric_card("제공 test", f"{len(test):,}", "정답 없는 추론용")
    with c3:
        metric_card("관찰 이탈률", f"{train['churned'].mean():.1%}", "churned=1")
    with c4:
        metric_card("검증 후보", f"{len(comparison):,}개", "provisional 비교")

    st.markdown('<div class="section-label">핵심 신호</div>', unsafe_allow_html=True)
    a, b, c = st.columns(3)
    monthly = train.loc[train.payment_plan == "monthly", "churned"].mean()
    annual = train.loc[train.payment_plan == "annual", "churned"].mean()
    student = train.loc[train.subscription_type == "student", "churned"].mean()
    family = train.loc[train.subscription_type == "family premium", "churned"].mean()
    with a:
        metric_card("월간 vs 연간", f"{monthly - annual:+.1%}", "payment_plan 이탈률 차이")
    with b:
        metric_card("학생 요금제", f"{student:.1%}", "세그먼트 관찰 이탈률")
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
        bins = pd.cut(predictions["churn_probability"], bins=[-0.01, 0.2, 0.37, 0.6, 1.0], labels=["Low", "Watch", "High", "Very high"])
        risk_counts = bins.value_counts().sort_index().reset_index()
        risk_counts.columns = ["risk_band", "customers"]
        fig = px.bar(risk_counts, x="risk_band", y="customers", color="risk_band", color_discrete_sequence=["#8ac5a6", "#f2cb74", "#f19b78", "#d94a3a"])
        fig.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Customers")
        st.plotly_chart(plotly_theme(fig), width="stretch")
        st.markdown('<div class="small-caption">정답이 없는 test이므로 이 분포는 예측 결과 분포이며 실제 이탈률이 아닙니다.</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">운영자가 먼저 볼 인사이트</div>', unsafe_allow_html=True)
    st.dataframe(
        pd.DataFrame(
            [
                ["결제 주기", "monthly", f"{monthly:.1%}", "월간 결제 고객의 이탈률 차이를 리텐션 실험 후보로 확인"],
                ["구독 유형", "student", f"{student:.1%}", "학생 고객군의 가격·혜택·사용 패턴을 별도 세그먼트로 검토"],
                ["구독 유형", "family premium", f"{family:.1%}", "가족형 고객군의 상대적으로 낮은 관찰 이탈률 확인"],
                ["결제 컬럼 품질", "num_subscription_pauses", "주의", "값이 annual/monthly라 컬럼명과 의미 불일치; 데이터 담당 확인 필요"],
            ],
            columns=["관점", "세그먼트/필드", "관찰값", "다음 확인"],
        ),
        hide_index=True,
        width="stretch",
    )


def eda_page(train: pd.DataFrame):
    header("Data Explorer", "각 입력 변수와 churned 라벨의 관계를 분포·이탈률·표본 수로 확인합니다.")
    feature_options = [c for c in train.columns if c not in ["customer_id", "churned"]]
    selected = st.selectbox("분석할 컬럼", feature_options, index=feature_options.index("payment_plan"))
    st.markdown('<div class="small-caption">모든 컬럼을 선택해 개별 target 관계를 확인할 수 있습니다. 그래프는 연관관계이며 인과관계가 아닙니다.</div>', unsafe_allow_html=True)
    left, right = st.columns([1.45, 1])
    with left:
        if selected == "signup_date":
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
        if selected == "signup_date":
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
        if column == "signup_date":
            series = pd.to_datetime(train[column])
            summary_rows.append([column, "date", f"{series.min():%Y-%m-%d} ~ {series.max():%Y-%m-%d}", "cohort only", "30일 라벨 미검증"])
        elif pd.api.types.is_numeric_dtype(train[column]):
            corr = train[[column, "churned"]].corr().iloc[0, 1]
            summary_rows.append([column, "numeric", f"corr {corr:+.4f}", f"median {train[column].median():,.2f}", "association"])
        else:
            rates = train.groupby(column)["churned"].mean()
            summary_rows.append([column, "categorical", f"max {rates.max():.1%}", f"min {rates.min():.1%}", "association"])
    st.dataframe(pd.DataFrame(summary_rows, columns=["컬럼", "타입", "핵심 신호", "분포/범위", "해석"]), hide_index=True, width="stretch")


def model_page(comparison: pd.DataFrame, threshold: pd.DataFrame, metadata: dict, importance: pd.DataFrame):
    header("Model Lab", "후보 모델 성능과 FN 우선 threshold trade-off를 확인합니다. 현재 추천은 provisional입니다.")
    best = comparison.iloc[0]
    st.markdown(f'<div class="note-card"><b>현재 기술 추천: {best["model"]}</b><br>Validation expected cost 기준으로 선정된 임시 후보입니다. 비용비는 FN:FP={metadata.get("false_negative_cost", 3):g}:1이며, 실제 운영 threshold와 모델 채택은 승인 전입니다.</div>', unsafe_allow_html=True)
    m1, m2, m3, m4 = st.columns(4)
    with m1: metric_card("Test PR-AUC", f"{best['test_pr_auc']:.3f}", "ranking quality")
    with m2: metric_card("Test Recall", f"{best['test_operating_recall']:.1%}", "provisional threshold")
    with m3: metric_card("Test Precision", f"{best['test_operating_precision']:.1%}", "provisional threshold")
    with m4: metric_card("Threshold", f"{best['validation_operating_threshold']:.2f}", "Validation 선택값")

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
    st.markdown('<div class="small-caption">Threshold가 낮아지면 Recall은 보통 올라가지만 FP도 늘어납니다. 현재 비용비 3:1은 provisional이며, 운영 수용량과 실제 FN/FP 비용으로 재설정해야 합니다.</div>', unsafe_allow_html=True)

    with st.expander("모델 한계와 데이터 리스크"):
        st.markdown("""
        - 현재 Test는 내부 분할 Test이며, 제공 `data/test.csv`는 정답이 없어 실제 성능을 검증하지 못합니다.
        - `churned`가 향후 30일 내 해지를 뜻한다는 생성 규칙이 확인되지 않았습니다.
        - 성능이 낮거나 특정 세그먼트에서 달라질 수 있으므로 자동 할인·차단에 사용하지 않습니다.
        - 변수 중요도는 곧바로 해지 원인이나 캠페인 효과를 의미하지 않습니다.
        """)


def prediction_page(train: pd.DataFrame, model, metadata: dict):
    header("Customer Scoring", "고객 정보를 입력하면 저장된 Pipeline이 이탈 위험 점수와 리텐션 확인 포인트를 반환합니다.")
    threshold = float(metadata["validation_threshold"])
    st.markdown(f'<div class="note-card"><b>모델 실행 방식</b><br>저장된 `{metadata["model"]}` Pipeline만 로드합니다. 현재 threshold는 `{threshold:.2f}`이며 FN:FP 비용비 `{metadata.get("false_negative_cost", 3):g}:1` 기준의 provisional 값입니다.</div>', unsafe_allow_html=True)

    with st.form("customer_form"):
        st.markdown('<div class="section-label">Subscription & profile</div>', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=18, max_value=100, value=int(train.age.median()), step=1)
            location = st.selectbox("Location", sorted(train.location.unique()), index=sorted(train.location.unique()).index("California") if "California" in train.location.unique() else 0)
            subscription_type = st.selectbox("Subscription type", sorted(train.subscription_type.unique()))
        with c2:
            payment_plan = st.selectbox("Payment plan", sorted(train.payment_plan.unique()), index=1 if "monthly" in train.payment_plan.unique() else 0)
            pause_value = st.selectbox("num_subscription_pauses (실제 값)", sorted(train.num_subscription_pauses.unique()))
            payment_method = st.selectbox("Payment method", sorted(train.payment_method.unique()))
        with c3:
            customer_service = st.selectbox("Customer service inquiries", ["none", "few", "some", "many"], index=1)
            signup_default = pd.to_datetime(train.signup_date).median().date()
            signup_date = st.date_input("Signup date", value=signup_default, min_value=pd.Timestamp("2013-01-01").date(), max_value=pd.Timestamp("2022-02-15").date())
            st.caption("가입일은 cohort feature로만 사용되며, 30일 관측 창은 데이터에 없습니다.")

        st.markdown('<div class="section-label">Listening behavior</div>', unsafe_allow_html=True)
        numeric_cols = ["weekly_hours", "average_session_length", "song_skip_rate", "weekly_songs_played", "weekly_unique_songs", "num_favorite_artists", "num_platform_friends", "num_playlists_created", "num_shared_playlists", "notifications_clicked"]
        defaults = {c: float(train[c].median()) for c in numeric_cols}
        row1 = st.columns(5)
        weekly_hours = row1[0].number_input("Weekly hours", min_value=0.0, max_value=100.0, value=defaults["weekly_hours"], step=0.1)
        avg_session = row1[1].number_input("Avg session length", min_value=0.0, max_value=20.0, value=defaults["average_session_length"], step=0.05)
        skip_rate = row1[2].number_input("Song skip rate", min_value=0.0, max_value=1.0, value=defaults["song_skip_rate"], step=0.01)
        songs = row1[3].number_input("Weekly songs played", min_value=0, max_value=2000, value=int(defaults["weekly_songs_played"]), step=1)
        unique_songs = row1[4].number_input("Weekly unique songs", min_value=0, max_value=2000, value=int(defaults["weekly_unique_songs"]), step=1)
        row2 = st.columns(5)
        favorites = row2[0].number_input("Favorite artists", min_value=0, max_value=500, value=int(defaults["num_favorite_artists"]), step=1)
        friends = row2[1].number_input("Platform friends", min_value=0, max_value=500, value=int(defaults["num_platform_friends"]), step=1)
        playlists = row2[2].number_input("Playlists created", min_value=0, max_value=500, value=int(defaults["num_playlists_created"]), step=1)
        shared = row2[3].number_input("Shared playlists", min_value=0, max_value=500, value=int(defaults["num_shared_playlists"]), step=1)
        notifications = row2[4].number_input("Notifications clicked", min_value=0, max_value=2000, value=int(defaults["notifications_clicked"]), step=1)
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
        probability = float(model.predict_proba(row)[:, 1][0])
        level, css = risk_level(probability, threshold)
        st.markdown('<div class="section-label">Scoring result</div>', unsafe_allow_html=True)
        r1, r2, r3 = st.columns(3)
        with r1: metric_card("Dataset-label churn probability", f"{probability:.1%}", "model score")
        with r2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">Risk level</div><div class="metric-value"><span class="{css}">{level}</span></div><div class="metric-delta">threshold {threshold:.2f}</div></div>', unsafe_allow_html=True)
        with r3: metric_card("Decision", "우선 검토" if probability >= threshold else "일반 모니터링", "자동 조치 아님")
        st.markdown('<div class="section-label">리텐션 검토 포인트</div>', unsafe_allow_html=True)
        actions = []
        if payment_plan == "monthly": actions.append("월간 결제 고객: 연간 전환·가격/혜택 실험 후보로 분리")
        if subscription_type in {"student", "premium"}: actions.append(f"{subscription_type} 고객: 요금제 혜택과 사용 맥락을 별도 세그먼트로 확인")
        if skip_rate >= train.song_skip_rate.quantile(.75): actions.append("높은 skip rate: 개인화 추천·탐색 경험의 추가 진단 후보")
        if weekly_hours <= train.weekly_hours.quantile(.25): actions.append("낮은 주간 청취: 재활성화 메시지·콘텐츠 추천 실험 후보")
        if not actions: actions.append("현재 입력만으로는 강한 단일 신호가 없으므로 전체 세그먼트 맥락과 함께 검토")
        for action in actions:
            st.markdown(f"- {action}")
        st.caption("위 내용은 예측 신호에 기반한 실험 후보이며, 해지 원인·캠페인 효과를 의미하지 않습니다.")


def batch_page(predictions: pd.DataFrame, metadata: dict):
    header("Batch Prioritization", "정답이 없는 제공 test 고객의 예측 분포와 우선 검토 대상을 확인합니다.")
    threshold = float(metadata["validation_threshold"])
    min_probability = st.slider("최소 이탈 확률", 0.0, 1.0, threshold, 0.01)
    filtered = predictions[predictions["churn_probability"] >= min_probability].sort_values("churn_probability", ascending=False)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("전체 test", f"{len(predictions):,}", "정답 없음")
    with c2: metric_card("현재 대상", f"{len(filtered):,}", f"probability ≥ {min_probability:.2f}")
    with c3: metric_card("비율", f"{len(filtered) / len(predictions):.1%}", "예측 대상 비중")
    st.markdown('<div class="info-card"><b>주의</b><br>제공 test에는 정답이 없으므로 고위험 고객 목록의 정확도나 실제 유지 효과는 여기서 검증할 수 없습니다. 운영 전에는 라벨과 캠페인 결과를 연결해야 합니다.</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-label">우선 검토 고객</div>', unsafe_allow_html=True)
    view = filtered.head(500).copy()
    view["risk"] = np.where(view["churn_probability"] >= threshold, "High", "Watch")
    st.dataframe(view, hide_index=True, width="stretch")
    st.download_button("현재 필터 결과 CSV 다운로드", data=filtered.to_csv(index=False).encode("utf-8-sig"), file_name="playlistpro_churn_priorities.csv", mime="text/csv", width="stretch")


def main():
    (train, test), model, (comparison, threshold, metadata, predictions, importance) = safe_load()
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">♫</span><span class="brand-name">PlaylistPro</span></div>', unsafe_allow_html=True)
        st.caption("Insight workspace")
        page = st.radio("Workspace", ["Overview", "Data Explorer", "Model Lab", "Customer Scoring", "Batch Prioritization"], label_visibility="collapsed")
        st.divider()
        st.markdown(f"**Model**  `{metadata['model']}`")
        st.markdown(f"**Threshold**  `{metadata['validation_threshold']:.2f}`")
        st.markdown("<div class='small-caption'>Dataset-label score<br>30-day horizon unverified</div>", unsafe_allow_html=True)

    if page == "Overview":
        overview_page(train, test, comparison, metadata, predictions)
    elif page == "Data Explorer":
        eda_page(train)
    elif page == "Model Lab":
        model_page(comparison, threshold, metadata, importance)
    elif page == "Customer Scoring":
        prediction_page(train, model, metadata)
    else:
        batch_page(predictions, metadata)


if __name__ == "__main__":
    main()
