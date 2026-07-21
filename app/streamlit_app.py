"""PlaylistPro 고객 이탈 위험 의사결정 지원 대시보드."""
from __future__ import annotations

import hashlib
import html
import importlib
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

from src import retention_strategy as _retention_strategy  # noqa: E402

# Streamlit reruns this file inside a long-lived Python process. Reload when
# either the public planning function or the current economics defaults are
# missing from the cached helper module, so source updates appear immediately.
if (
    not hasattr(_retention_strategy, "plan_economic_assumptions")
    or getattr(_retention_strategy, "PLAN_ASSUMPTION_VERSION", 0) < 2
):
    _retention_strategy = importlib.reload(_retention_strategy)

build_strategy_queue = _retention_strategy.build_strategy_queue
derive_strategy_thresholds = _retention_strategy.derive_strategy_thresholds
plan_economic_assumptions = _retention_strategy.plan_economic_assumptions
recommend_retention_strategy = _retention_strategy.recommend_retention_strategy

DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model" / "music_churn_pipeline.joblib"
METADATA_PATH = ARTIFACT_DIR / "model" / "metadata.json"
TABLE_PATHS = {
    "predictions": ARTIFACT_DIR / "test_predictions.csv",
    "importance": ARTIFACT_DIR / "feature_importance.csv",
    "targeting": ARTIFACT_DIR / "topk_lift_oof_catboost.csv",
    "deciles": ARTIFACT_DIR / "risk_decile_oof_catboost.csv",
    "scenarios": ARTIFACT_DIR / "operating_scenarios_oof_catboost.csv",
    "threshold_sweep": ARTIFACT_DIR / "threshold_sweep_oof_catboost.csv",
    "comparison": ARTIFACT_DIR / "model_comparison_fair.csv",
    "fine_tuning": ARTIFACT_DIR / "top3_fine_tuning_summary.csv",
    "preprocessing": ARTIFACT_DIR / "insight_preprocessing_register.csv",
    "preprocessing_results": ARTIFACT_DIR / "preprocessing_experiment_results.csv",
    "random_search": ARTIFACT_DIR / "random_search_summary.csv",
    "insights": ARTIFACT_DIR / "current_insight_inventory.csv",
    "bootstrap": ARTIFACT_DIR / "bootstrap_confidence_intervals.csv",
    "calibration": ARTIFACT_DIR / "calibration_summary.csv",
    "equal_contact": ARTIFACT_DIR / "equal_contact_comparison.csv",
    "equal_recall": ARTIFACT_DIR / "equal_recall_comparison.csv",
}
SEED_STABILITY_PATH = ARTIFACT_DIR / "seed_stability_summary.csv"

COLORS = {
    "navy": "#18324a",
    "blue": "#315d78",
    "orange": "#e4573d",
    "amber": "#e8a23a",
    "green": "#2f7d62",
    "muted": "#64748b",
    "line": "#e5e7eb",
}
MODEL_KO = {
    "dummy": "Dummy 기준",
    "logistic": "Logistic 회귀",
    "decision_tree": "Decision Tree",
    "random_forest": "Random Forest",
    "gradient_boosting": "Gradient Boosting",
    "xgboost": "XGBoost",
    "lightgbm": "LightGBM",
    "catboost": "CatBoost",
}

st.set_page_config(
    page_title="PlaylistPro Insight",
    page_icon="♫",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+KR:wght@400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap');
    :root { --ink:#111827; --muted:#64748b; --line:#e5e7eb; --accent:#e4573d; --navy:#18324a; }
    html, body, [class*="css"] { font-family:'Noto Sans KR','DM Sans',sans-serif; }
    h1, h2, h3, h4 { font-family:'Noto Sans KR','Space Grotesk',sans-serif; letter-spacing:-.035em; color:var(--ink); }
    .stApp { background:linear-gradient(180deg,#f5f7f9 0,#fbfcfd 420px); color:var(--ink); }
    .block-container { max-width:1440px; padding-top:2rem; padding-bottom:4rem; }
    [data-testid="stSidebar"] { background:#102235; border-right:0; }
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] *,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"],
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] *,
    [data-testid="stSidebar"] label, [data-testid="stSidebar"] label * { color:#e8eef5 !important; }
    [data-testid="stSidebar"] [data-baseweb="select"] > div,
    [data-testid="stSidebar"] [data-baseweb="select"] > div * { color:#111827 !important; }
    [data-testid="stSidebar"] .stRadio label { padding:.58rem .7rem; border-radius:.65rem; }
    [data-testid="stSidebar"] .stRadio label:hover { background:rgba(255,255,255,.08); }
    .brand { padding:.3rem 0 1.25rem; }
    .brand-mark { display:inline-flex; align-items:center; justify-content:center; width:40px; height:40px; background:#e4573d; color:white; border-radius:13px; font-size:23px; font-weight:700; margin-right:10px; box-shadow:0 8px 20px rgba(228,87,61,.28); }
    .brand-name { font-family:'Space Grotesk'; font-weight:700; font-size:21px; color:white; vertical-align:middle; }
    .sidebar-kicker { color:#94a9ba; font-size:11px; letter-spacing:.1em; text-transform:uppercase; margin:3px 0 9px; }
    .eyebrow { color:#e4573d; text-transform:uppercase; letter-spacing:.13em; font-size:11px; font-weight:700; margin-bottom:.45rem; }
    .hero { position:relative; overflow:hidden; background:linear-gradient(120deg,#18324a 0%,#214765 62%,#315d78 100%); color:white; border-radius:22px; padding:32px 34px; margin-bottom:22px; box-shadow:0 16px 42px rgba(24,50,74,.16); }
    .hero:after { content:''; position:absolute; width:260px; height:260px; border:55px solid rgba(255,255,255,.055); border-radius:50%; right:-65px; top:-105px; }
    .hero h1 { color:white; margin:0; max-width:880px; font-size:clamp(28px,3vw,40px); line-height:1.22; }
    .hero p { color:#d9e4ed; max-width:850px; margin:.75rem 0 0; font-size:15px; line-height:1.7; }
    .hero .tag { display:inline-block; background:rgba(255,255,255,.13); color:#f4f8fb; border:1px solid rgba(255,255,255,.12); border-radius:999px; padding:6px 11px; margin-top:15px; margin-right:6px; font-size:12px; }
    .metric-card { background:white; border:1px solid var(--line); border-radius:17px; padding:18px 19px; min-height:116px; box-shadow:0 6px 20px rgba(15,23,42,.04); }
    .metric-label { color:#64748b; font-size:12px; font-weight:600; }
    .metric-value { color:#111827; font-family:'Space Grotesk','Noto Sans KR'; font-size:29px; font-weight:700; margin-top:7px; }
    .metric-delta { color:#e4573d; font-size:12px; margin-top:4px; line-height:1.4; }
    .section-head { margin-top:8px; margin-bottom:16px; }
    .section-head h1 { margin:.15rem 0 .35rem; font-size:31px; }
    .section-head p { color:#64748b; margin:0; line-height:1.6; }
    .section-label { margin-top:27px; margin-bottom:10px; font-size:19px; font-weight:700; letter-spacing:-.025em; }
    .info-card, .note-card, .white-card { border-radius:15px; padding:16px 18px; line-height:1.6; height:100%; }
    .info-card { border:1px solid #fed6cc; background:#fff8f5; color:#63352d; }
    .note-card { border:1px solid #dbe5ef; background:#f3f7fb; color:#29445d; }
    .white-card { border:1px solid #e5e7eb; background:white; color:#334155; box-shadow:0 5px 18px rgba(15,23,42,.025); }
    .card-title { color:#111827; font-weight:700; margin-bottom:6px; }
    .card-kicker { color:#e4573d; font-size:11px; font-weight:700; letter-spacing:.08em; text-transform:uppercase; }
    .flow { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; margin:8px 0 18px; }
    .flow-step { background:white; border:1px solid #e5e7eb; border-radius:14px; padding:14px 12px; min-height:92px; }
    .flow-num { color:#e4573d; font-family:'Space Grotesk'; font-weight:700; font-size:12px; }
    .flow-title { color:#18324a; font-weight:700; margin-top:7px; font-size:13px; }
    .flow-copy { color:#64748b; font-size:11px; margin-top:4px; line-height:1.4; }
    .risk-high { color:#b93827; background:#fff0ec; border:1px solid #ffc7ba; border-radius:999px; padding:5px 10px; font-weight:700; display:inline-block; }
    .risk-mid { color:#9a6413; background:#fff8e9; border:1px solid #f7dc9b; border-radius:999px; padding:5px 10px; font-weight:700; display:inline-block; }
    .risk-low { color:#267152; background:#edf9f3; border:1px solid #bce6d0; border-radius:999px; padding:5px 10px; font-weight:700; display:inline-block; }
    .action-card { background:white; border:1px solid #e5e7eb; border-top:4px solid #e4573d; border-radius:15px; padding:17px 18px; min-height:205px; box-shadow:0 6px 20px rgba(15,23,42,.035); }
    .action-card.alt { border-top-color:#315d78; }
    .action-rank { color:#e4573d; font-size:11px; font-weight:700; letter-spacing:.09em; }
    .action-title { color:#18324a; font-weight:700; font-size:17px; margin:7px 0 9px; line-height:1.4; }
    .action-row { color:#475569; font-size:12px; line-height:1.6; margin-top:6px; }
    .strategy-strip { display:grid; grid-template-columns:repeat(4,1fr); gap:10px; margin:10px 0 18px; }
    .strategy-cell { background:#f8fafc; border:1px solid #e5e7eb; border-radius:13px; padding:13px 14px; }
    .strategy-cell span { display:block; color:#64748b; font-size:10px; margin-bottom:5px; }
    .strategy-cell b { color:#18324a; font-size:13px; line-height:1.45; }
    div[data-testid="stDataFrame"] { border-radius:13px; overflow:hidden; border:1px solid #edf0f2; }
    div[data-testid="stPlotlyChart"] { background:white; border:1px solid #edf0f2; border-radius:16px; padding:4px; box-shadow:0 5px 18px rgba(15,23,42,.025); }
    [data-testid="stMetric"] { background:white; border:1px solid #e5e7eb; padding:14px 16px; border-radius:15px; }
    .boundary { background:#fffdf8; border:1px solid #f2dfb4; border-left:4px solid #e8a23a; border-radius:13px; padding:14px 16px; line-height:1.6; color:#62491d; margin-top:18px; }
    .small { color:#64748b; font-size:12px; line-height:1.55; }
    @media (max-width:900px) {
        .flow { grid-template-columns:repeat(2,1fr); }
        .strategy-strip { grid-template-columns:repeat(2,1fr); }
        .hero { padding:25px 23px; }
        .block-container { padding-left:1rem; padding-right:1rem; }
        .metric-card { padding:13px 11px; min-height:104px; }
        .metric-label { font-size:10px; }
        .metric-value { font-size:18px; white-space:nowrap; }
        .metric-delta { font-size:9px; }
    }
    @media (max-width:620px) {
        .flow { grid-template-columns:1fr; }
        .strategy-strip { grid-template-columns:1fr; }
    }
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
def load_tables() -> dict:
    tables = {"train": pd.read_csv(DATA_DIR / "train.csv"), "test": pd.read_csv(DATA_DIR / "test.csv")}
    tables["metadata"] = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    tables.update({name: pd.read_csv(path) for name, path in TABLE_PATHS.items()})
    seed = pd.read_csv(SEED_STABILITY_PATH, header=[0, 1], index_col=0)
    seed.columns = [f"{first}_{second}" for first, second in seed.columns]
    tables["seed_stability"] = seed.reset_index(names="model")
    tables["strategy_thresholds"] = derive_strategy_thresholds(tables["train"])
    tables["strategy_queue"] = build_strategy_queue(
        tables["test"],
        tables["predictions"],
        tables["strategy_thresholds"],
        low_risk_threshold=float(tables["scenarios"]["threshold"].min()),
        high_risk_threshold=float(tables["scenarios"]["threshold"].max()),
    )
    return tables


@st.cache_resource(show_spinner=False)
def load_model(expected_hash: str):
    model_path = MODEL_PATH.resolve()
    if not model_path.is_relative_to(ROOT.resolve()):
        raise RuntimeError("허용된 프로젝트 폴더 밖의 모델은 로드할 수 없습니다.")
    if not expected_hash or sha256(model_path) != expected_hash:
        raise RuntimeError("모델 SHA-256이 메타데이터와 다릅니다. 파일 무결성을 확인해 주세요.")
    # joblib/pickle은 코드 실행이 가능한 형식이므로 저장소 내부에서 검증된 파일만 로드한다.
    return joblib.load(model_path)


def load_workspace() -> tuple[dict, object]:
    required = [DATA_DIR / "train.csv", DATA_DIR / "test.csv", MODEL_PATH, METADATA_PATH, SEED_STABILITY_PATH, *TABLE_PATHS.values()]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        st.error("필수 파일이 없습니다: " + ", ".join(missing))
        st.stop()
    tables = load_tables()
    return tables, load_model(str(tables["metadata"].get("artifact_sha256", "")))


def plot_style(fig: go.Figure, height: int = 390) -> go.Figure:
    fig.update_layout(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Noto Sans KR, DM Sans", color="#334155"),
        colorway=[COLORS["navy"], COLORS["orange"], COLORS["green"], COLORS["amber"], COLORS["blue"]],
        margin=dict(l=18, r=18, t=58, b=20),
        height=height,
        hoverlabel=dict(bgcolor="white"),
    )
    return fig


def metric_card(label: str, value: str, sub: str = "") -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{html.escape(label)}</div>'
        f'<div class="metric-value">{html.escape(value)}</div><div class="metric-delta">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, description: str) -> None:
    st.markdown(
        f'<div class="section-head"><div class="eyebrow">{html.escape(kicker)}</div>'
        f'<h1>{html.escape(title)}</h1><p>{html.escape(description)}</p></div>',
        unsafe_allow_html=True,
    )


def card(title: str, body: str, kind: str = "white-card", kicker: str = "") -> None:
    kicker_html = f'<div class="card-kicker">{html.escape(kicker)}</div>' if kicker else ""
    st.markdown(
        f'<div class="{kind}">{kicker_html}<div class="card-title">{html.escape(title)}</div>{body}</div>',
        unsafe_allow_html=True,
    )


def boundary_note() -> None:
    st.markdown(
        '<div class="boundary"><b>근거의 범위</b><br>'
        '성능·Threshold·Top-K·Lift·Decile은 저장된 5-Fold OOF 예측 근거입니다. 독립 외부 Holdout, '
        '미래 이탈 기간, 캠페인 Uplift·ROI·인과효과를 검증한 결과가 아니며 실제 CRM 발송이나 고객 접촉을 실행하지 않습니다.</div>',
        unsafe_allow_html=True,
    )


def selected_scenario(scenarios: pd.DataFrame, scenario_id: str) -> pd.Series:
    found = scenarios.loc[scenarios["scenario_id"].eq(scenario_id)]
    return found.iloc[0] if not found.empty else scenarios.iloc[0]


def model_label(value: str) -> str:
    return MODEL_KO.get(str(value), str(value))


def project_summary_page(tables: dict, scenario: pd.Series) -> None:
    train, test, metadata, targeting = tables["train"], tables["test"], tables["metadata"], tables["targeting"]
    metrics = metadata["metrics"]["five_fold_oof"]
    churn_rate = float(train["churned"].mean())
    st.markdown(
        '<div class="hero"><div class="eyebrow" style="color:#ffab97">RETENTION DECISION SYSTEM</div>'
        '<h1>누구에게 어떤 유지 활동을 먼저 검토할 것인가?</h1>'
        '<p>PlaylistPro의 고객 행동 인사이트와 최종 CatBoost 위험 점수를 연결해, 검토 대상 범위와 고객별 유지 활동 후보를 함께 선택하는 의사결정 화면입니다.</p>'
        '<span class="tag">관측 타깃: churned</span><span class="tag">최종 모델: CatBoost</span><span class="tag">검증: 5-Fold OOF</span></div>',
        unsafe_allow_html=True,
    )
    columns = st.columns(4)
    values = [
        ("학습 고객", f"{len(train):,}명", f"관측 이탈률 {churn_rate:.1%}"),
        ("점수 산출 고객", f"{len(test):,}명", "정답 라벨이 없는 운영 후보"),
        ("OOF PR-AUC", f"{metrics['pr_auc']:.4f}", "불균형 이탈 탐지 1차 선정 지표"),
        ("현재 운영안", str(scenario["scenario_label"]), f"Threshold {float(scenario['threshold']):.2f}"),
    ]
    for column, value in zip(columns, values):
        with column:
            metric_card(*value)

    st.markdown('<div class="section-label">결정까지의 한 줄 흐름</div>', unsafe_allow_html=True)
    steps = [
        ("01", "고객 인사이트", "청취·스킵·구독·문의"),
        ("02", "Feature 검증", "log_numeric 채택"),
        ("03", "8개 모델 비교", "동일 Fold·동일 조건"),
        ("04", "정밀 검증", "OOF·Seed·Bootstrap"),
        ("05", "운영 기준", "Recall·Precision·용량"),
        ("06", "고객 전략", "우선순위·행동 후보·KPI"),
    ]
    flow = "".join(
        f'<div class="flow-step"><div class="flow-num">{number}</div><div class="flow-title">{title}</div><div class="flow-copy">{copy}</div></div>'
        for number, title, copy in steps
    )
    st.markdown(f'<div class="flow">{flow}</div>', unsafe_allow_html=True)

    top30 = targeting.loc[targeting["top_percent"].eq(30)].iloc[0]
    left, right = st.columns([1.15, 1])
    with left:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(top30["capture_rate"]) * 100,
            number={"suffix": "%", "font": {"color": COLORS["navy"]}},
            title={"text": "위험 점수 상위 30%의 관측 이탈 포착률"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": COLORS["orange"]}, "bgcolor": "#eef2f5"},
        ))
        st.plotly_chart(plot_style(fig, 310), width="stretch")
    with right:
        st.markdown('<div class="section-label" style="margin-top:4px">현재 화면이 지원하는 결정</div>', unsafe_allow_html=True)
        card(
            "사용자가 직접 정하는 것",
            f"검토 용량과 누락 비용을 보고 <b>{html.escape(str(scenario['scenario_label']))}</b> 등 Threshold 시나리오를 선택합니다. 현재 선택은 OOF 기준 약 <b>{float(scenario['oof_target_rate']):.1%}</b>를 검토합니다.",
            "info-card",
            "HUMAN DECISION",
        )
        st.write("")
        card(
            "시스템이 제공하는 것",
            "검증된 모델 로드, 위험 점수 재계산, 고객 순위·행동 후보·KPI·전략 CSV를 제공합니다. 캠페인 실행과 할인 적용은 포함하지 않습니다.",
            "note-card",
            "SYSTEM SUPPORT",
        )
    boundary_note()


def customer_insights_page(tables: dict) -> None:
    train = tables["train"]
    section_header("CUSTOMER INSIGHT", "고객 행동에서 무엇을 관찰했는가", "연관성을 서비스 개선 가설로 바꾸되, 인과효과로 과장하지 않습니다.")

    missing = int(train.isna().sum().sum())
    duplicate_rows = int(train.duplicated().sum())
    duplicate_ids = int(train["customer_id"].duplicated().sum())
    id_overlap = len(set(train["customer_id"]).intersection(set(tables["test"]["customer_id"])))
    for column, item in zip(st.columns(4), [
        ("관측 이탈 고객", f"{int(train['churned'].sum()):,}명", f"전체의 {train['churned'].mean():.1%}"),
        ("결측 셀", f"{missing:,}개", "입력 데이터 품질 점검"),
        ("중복 행 / ID", f"{duplicate_rows:,} / {duplicate_ids:,}", "학습 표본 중복 확인"),
        ("Train-Test ID 중첩", f"{id_overlap:,}명", "고객 단위 겹침 확인"),
    ]):
        with column:
            metric_card(*item)

    feature_options = {
        "주간 청취시간": ("weekly_hours", "numeric"),
        "스킵률": ("song_skip_rate", "numeric"),
        "구독 유형": ("subscription_type", "categorical"),
        "고객 문의 수준": ("customer_service_inquiries", "categorical"),
        "구독 일시정지 횟수": ("num_subscription_pauses", "numeric"),
    }
    left, right = st.columns([1.15, 1])
    with left:
        choice = st.selectbox("살펴볼 고객 행동", list(feature_options))
        feature, kind = feature_options[choice]
        if kind == "numeric":
            grouped = train.assign(구간=pd.qcut(train[feature], 4, duplicates="drop")).groupby("구간", observed=True).agg(
                고객수=("churned", "size"), 관측이탈률=("churned", "mean")
            ).reset_index()
            grouped["구간"] = grouped["구간"].astype(str)
        else:
            grouped = train.groupby(feature, observed=True).agg(고객수=("churned", "size"), 관측이탈률=("churned", "mean")).reset_index()
            grouped = grouped.rename(columns={feature: "구간"}).sort_values("관측이탈률", ascending=False)
        fig = px.bar(grouped, x="구간", y="관측이탈률", text=grouped["관측이탈률"].map(lambda x: f"{x:.1%}"), custom_data=["고객수"])
        fig.update_traces(marker_color=COLORS["orange"], hovertemplate="%{x}<br>관측 이탈률 %{y:.1%}<br>고객 %{customdata[0]:,}명<extra></extra>")
        fig.update_layout(title=f"{choice}별 관측 이탈률", yaxis_tickformat=".0%", xaxis_title=None, yaxis_title="관측 이탈률")
        st.plotly_chart(plot_style(fig), width="stretch")
    with right:
        importance = tables["importance"].head(12).sort_values("importance")
        fig = px.bar(importance, x="importance", y="feature", orientation="h", text=importance["importance"].map(lambda x: f"{x:.1f}"))
        fig.update_traces(marker_color=COLORS["navy"])
        fig.update_layout(title="CatBoost 전역 Feature 중요도", xaxis_title="중요도", yaxis_title=None)
        st.plotly_chart(plot_style(fig), width="stretch")

    st.markdown('<div class="section-label">관찰을 실행 가능한 가설로 번역</div>', unsafe_allow_html=True)
    insight_cards = [
        ("낮은 청취·높은 스킵", "청취시간과 스킵률 구간에서 관측 이탈률 차이가 나타났습니다.", "추천 적합도와 콘텐츠 발견 경험을 진단합니다.", "시간 순서가 없어 원인으로 단정할 수 없습니다."),
        ("Free 구독 유형", "Free 고객군의 관측 이탈률이 상대적으로 높습니다.", "혜택 인지와 전환 제안을 별도 실험합니다.", "할인·오퍼 효과는 아직 검증되지 않았습니다."),
        ("고객 문의 수준", "문의 수준은 모델의 주요 분류 신호입니다.", "해결 속도와 반복 문의 원인을 점검합니다.", "문의가 이탈을 만든다는 인과 근거는 아닙니다."),
    ]
    for column, (title, observation, action, limit) in zip(st.columns(3), insight_cards):
        with column:
            card(title, f"<b>관찰</b> · {observation}<br><b>활용 가설</b> · {action}<br><span class='small'><b>한계</b> · {limit}</span>", "white-card", "INSIGHT")
    boundary_note()


def improvement_page(tables: dict) -> None:
    section_header("EXPERIMENT STORY", "무엇을 바꾸었고, 성능은 어떻게 달라졌는가", "채택한 실험뿐 아니라 제외한 실험도 동일한 평가 근거로 보여줍니다.")
    prep = tables["preprocessing_results"].copy()
    raw_pr = float(prep.loc[prep["variant"].eq("raw"), "pr_auc"].iloc[0])
    prep["PR-AUC 변화"] = prep["pr_auc"] - raw_pr
    prep["결정"] = np.where(prep["variant"].eq("log_numeric"), "채택", "제외")
    prep_labels = {
        "raw": "Raw 기준",
        "plus1_ratios": "+1 비율",
        "zero_aware_ratios": "0 인지 비율",
        "log_numeric": "수치 로그 변환",
        "interaction": "상호작용",
        "signal_pruned": "신호 축소",
    }
    prep["실험"] = prep["variant"].map(prep_labels)
    left, right = st.columns([1.15, 1])
    with left:
        fig = px.bar(prep, x="실험", y="pr_auc", color="결정", text=prep["pr_auc"].map(lambda x: f"{x:.4f}"),
                     color_discrete_map={"채택": COLORS["orange"], "제외": "#a8b4bf"})
        fig.update_layout(title="1차 Feature 가공 검증 · Logistic OOF", yaxis_range=[0.895, 0.91], yaxis_title="PR-AUC", xaxis_title=None)
        st.plotly_chart(plot_style(fig), width="stretch")
        st.caption("Logistic은 최종 후보가 아니라 변환 효과를 같은 조건에서 빠르게 비교한 고정 검증기입니다. CatBoost 재확인에서도 log_numeric이 가장 높았지만 Raw와 차이는 매우 작았습니다. 공통 Feature 결정 근거이지 전체 성능 향상의 주된 원인으로 과장하지 않습니다.")
    with right:
        card("문제 관찰", "횟수·시간 변수의 긴 꼬리가 선형 기준 모델에서 신호를 압축할 수 있었습니다.", "note-card", "01 OBSERVE")
        st.write("")
        card("가설과 결과", f"로그 변환이 순서 신호를 보존하며 분포를 완화할 것으로 가정했습니다. PR-AUC는 <b>{raw_pr:.4f} → {prep['pr_auc'].max():.4f}</b>로 개선됐습니다.", "info-card", "02 TEST")
        st.write("")
        card("결정", "log_numeric만 공통 Feature로 채택하고, 개선이 없던 비율·상호작용·신호 축소안은 제외했습니다.", "white-card", "03 DECIDE")

    st.markdown('<div class="section-label">동일 조건 8개 모델과 탐색 전후</div>', unsafe_allow_html=True)
    comparison = tables["comparison"].copy()
    comparison["모델"] = comparison["model"].map(model_label)
    comparison = comparison.sort_values("pr_auc", ascending=False)
    random_search = tables["random_search"][["model", "pr_auc"]].rename(columns={"pr_auc": "탐색 후"})
    before_after = comparison[["model", "pr_auc"]].rename(columns={"pr_auc": "기본 비교"}).merge(random_search, on="model", how="inner")
    before_after["모델"] = before_after["model"].map(model_label)
    a, b = st.columns([1.15, 1])
    with a:
        fig = px.bar(comparison, x="모델", y="pr_auc", color=np.where(comparison["model"].eq("catboost"), "최종 선정", "비교 후보"),
                     text=comparison["pr_auc"].map(lambda x: f"{x:.4f}"), color_discrete_map={"최종 선정": COLORS["orange"], "비교 후보": COLORS["navy"]})
        fig.update_layout(title="동일 Feature · 동일 5-Fold OOF 모델 비교", yaxis_range=[0.5, 0.97], yaxis_title="PR-AUC", xaxis_title=None, showlegend=False)
        st.plotly_chart(plot_style(fig), width="stretch")
    with b:
        long = before_after.melt(id_vars="모델", value_vars=["기본 비교", "탐색 후"], var_name="단계", value_name="PR-AUC")
        fig = px.line(long, x="단계", y="PR-AUC", color="모델", markers=True)
        fig.update_layout(title="Random Search 전후", yaxis_range=[0.92, 0.952], legend_title=None)
        st.plotly_chart(plot_style(fig), width="stretch")

    display = prep[["실험", "pr_auc", "PR-AUC 변화", "recall", "precision", "fn", "fp", "결정"]].rename(columns={
        "pr_auc": "PR-AUC", "recall": "Recall", "precision": "Precision", "fn": "FN", "fp": "FP"
    })
    with st.expander("채택·제외 실험 전체 표"):
        st.dataframe(display.style.format({"PR-AUC": "{:.6f}", "PR-AUC 변화": "{:+.6f}", "Recall": "{:.1%}", "Precision": "{:.1%}"}), hide_index=True, width="stretch")
    boundary_note()


def model_comparison_page(tables: dict) -> None:
    section_header("MODEL DECISION", "왜 최종 모델은 CatBoost인가", "단일 점수만 보지 않고 PR-AUC, 오류 균형, Seed 안정성, Bootstrap 불확실성을 함께 판단했습니다.")
    fine = tables["fine_tuning"].sort_values("best_cv_pr_auc", ascending=False).copy()
    fine["모델"] = fine["model"].map(model_label)
    winner = fine.iloc[0]
    runner_up = fine.iloc[1]
    gap = float(winner["best_cv_pr_auc"] - runner_up["best_cv_pr_auc"])
    for column, item in zip(st.columns(4), [
        ("최종 선정", "CatBoost", "상위 3개 정밀 탐색 1위"),
        ("Fine CV PR-AUC", f"{float(winner['best_cv_pr_auc']):.6f}", f"차선 대비 +{gap:.6f}"),
        ("5-Seed 평균", f"{float(tables['seed_stability'].iloc[0]['pr_auc_mean']):.6f}", f"표준편차 {float(tables['seed_stability'].iloc[0]['pr_auc_std']):.6f}"),
        ("OOF PR-AUC", f"{float(winner['pr_auc']):.6f}", "저장된 전체 OOF 예측"),
    ]):
        with column:
            metric_card(*item)

    left, right = st.columns([1.15, 1])
    with left:
        metric_long = fine.melt(id_vars=["모델"], value_vars=["recall", "precision", "f1"], var_name="지표", value_name="값")
        metric_long["지표"] = metric_long["지표"].map({"recall": "Recall", "precision": "Precision", "f1": "F1"})
        fig = px.bar(metric_long, x="모델", y="값", color="지표", barmode="group", text=metric_long["값"].map(lambda x: f"{x:.3f}"))
        fig.update_layout(title="상위 3개 모델의 기본 Threshold 오류 균형", yaxis_range=[0.78, 0.9], yaxis_tickformat=".0%", xaxis_title=None)
        st.plotly_chart(plot_style(fig), width="stretch")
    with right:
        boot = tables["bootstrap"].query("metric == 'pr_auc'").copy()
        boot["모델"] = boot["model"].map(model_label)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=boot["estimate"], y=boot["모델"], mode="markers", marker={"size": 11, "color": COLORS["orange"]},
            error_x={"type": "data", "symmetric": False, "array": boot["ci_high"] - boot["estimate"], "arrayminus": boot["estimate"] - boot["ci_low"]},
        ))
        fig.update_layout(title="Bootstrap PR-AUC 95% 신뢰구간", xaxis_title="PR-AUC", yaxis_title=None)
        st.plotly_chart(plot_style(fig), width="stretch")

    st.markdown('<div class="section-label">선정 논리와 트레이드오프</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        card("1. 불균형 타깃에 맞는 1차 기준", "Accuracy가 아니라 <b>PR-AUC</b>를 1차 선정 지표로 사용했습니다. 이탈 고객 탐지 성능을 Recall·FN과 함께 봅니다.", "white-card", "METRIC")
    with c2:
        card("2. 점수 차이는 작다", f"CatBoost와 차선 후보의 Fine CV PR-AUC 차이는 <b>{gap:.6f}</b>입니다. 압도적 우위가 아니라 정해진 규칙에서의 근소한 1위입니다.", "info-card", "HONESTY")
    with c3:
        card("3. 운영 기준은 별도", "LightGBM은 기본 Threshold에서 Recall·F1이 더 높을 수 있습니다. 그러나 모델 선정과 Threshold 선택을 분리해 CatBoost 확률에 운영 기준을 적용합니다.", "note-card", "TRADE-OFF")

    comparison = tables["comparison"].copy()
    comparison["모델"] = comparison["model"].map(model_label)
    with st.expander("8개 모델 동일 조건 상세 지표"):
        cols = ["모델", "pr_auc", "roc_auc", "f1", "recall", "precision", "fn", "fp", "seconds"]
        st.dataframe(comparison[cols].sort_values("pr_auc", ascending=False).style.format({
            "pr_auc": "{:.6f}", "roc_auc": "{:.6f}", "f1": "{:.1%}", "recall": "{:.1%}", "precision": "{:.1%}", "seconds": "{:.1f}초"
        }), hide_index=True, width="stretch")
    with st.expander("Calibration 진단"):
        calibration = tables["calibration"].copy()
        calibration["모델"] = calibration["model"].map(model_label)
        st.dataframe(calibration[["모델", "brier", "mean_abs_calibration_gap"]].style.format({"brier": "{:.6f}", "mean_abs_calibration_gap": "{:.6f}"}), hide_index=True, width="stretch")
    boundary_note()


def operations_page(tables: dict, scenario: pd.Series) -> None:
    section_header("OPERATING POLICY", "누락과 접촉 비용 사이의 기준을 선택하세요", "모델은 고객 순위를 제공하고, 사용자는 검토 용량과 FN 비용에 맞춰 Threshold를 선택합니다.")
    for column, item in zip(st.columns(5), [
        ("선택 시나리오", str(scenario["scenario_label"]), f"Threshold {float(scenario['threshold']):.2f}"),
        ("검토 대상", f"{int(scenario['oof_target_customers']):,}명", f"OOF 표본의 {float(scenario['oof_target_rate']):.1%}"),
        ("Recall", f"{float(scenario['oof_recall']):.1%}", f"FN {int(scenario['oof_fn']):,}명"),
        ("Precision", f"{float(scenario['oof_precision']):.1%}", f"FP {int(scenario['oof_fp']):,}명"),
        ("F1", f"{float(scenario['oof_f1']):.1%}", str(scenario["selection_rule"])),
    ]):
        with column:
            metric_card(*item)

    sweep = tables["threshold_sweep"].copy()
    left, right = st.columns([1.25, 1])
    with left:
        fig = go.Figure()
        for column, label, color in [("oof_recall", "Recall", COLORS["orange"]), ("oof_precision", "Precision", COLORS["navy"]), ("oof_f1", "F1", COLORS["green"])]:
            fig.add_trace(go.Scatter(x=sweep["threshold"], y=sweep[column], name=label, mode="lines", line={"width": 3, "color": color}))
        fig.add_vline(x=float(scenario["threshold"]), line_dash="dash", line_color=COLORS["amber"], annotation_text="현재 선택")
        fig.update_layout(title="Threshold에 따른 OOF 지표 변화", xaxis_title="Threshold", yaxis_title="지표", yaxis_tickformat=".0%", legend_orientation="h")
        st.plotly_chart(plot_style(fig), width="stretch")
    with right:
        scenarios = tables["scenarios"].copy()
        fig = px.scatter(scenarios, x="oof_target_rate", y="oof_recall", size="oof_fp", color="scenario_label", text="scenario_label", hover_data=["oof_precision", "oof_fn"])
        fig.update_traces(textposition="top center")
        fig.update_layout(title="검토 범위와 이탈 포착의 균형", xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="검토 대상 비율", yaxis_title="Recall", showlegend=False)
        st.plotly_chart(plot_style(fig), width="stretch")

    st.markdown('<div class="section-label">용량 기반 우선순위 진단</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        topk = tables["targeting"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=topk["top_percent"], y=topk["capture_rate"], name="포착률", marker_color=COLORS["orange"], text=topk["capture_rate"].map(lambda x: f"{x:.1%}")))
        fig.add_trace(go.Scatter(x=topk["top_percent"], y=topk["lift"], name="Lift", yaxis="y2", mode="lines+markers", line={"color": COLORS["navy"], "width": 3}))
        fig.update_layout(title="OOF Top-K 포착률과 Lift", xaxis_title="위험 점수 상위 비율(%)", yaxis={"title": "포착률", "tickformat": ".0%"}, yaxis2={"title": "Lift", "overlaying": "y", "side": "right"}, legend_orientation="h")
        st.plotly_chart(plot_style(fig), width="stretch")
    with b:
        deciles = tables["deciles"]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=deciles["risk_decile"], y=deciles["actual_churn_rate"], name="관측 이탈률", marker_color=COLORS["navy"]))
        fig.add_trace(go.Scatter(x=deciles["risk_decile"], y=deciles["mean_predicted_probability"], name="평균 예측 확률", mode="lines+markers", line={"color": COLORS["orange"], "width": 3}))
        fig.update_layout(title="OOF Risk Decile 진단", xaxis_title="위험 Decile (10=최고)", yaxis_tickformat=".0%", legend_orientation="h")
        st.plotly_chart(plot_style(fig), width="stretch")

    scenarios_display = tables["scenarios"][["scenario_label", "threshold", "oof_target_rate", "oof_recall", "oof_precision", "oof_f1", "oof_fn", "oof_fp"]].copy()
    scenarios_display.columns = ["시나리오", "Threshold", "대상 비율", "Recall", "Precision", "F1", "FN", "FP"]
    st.dataframe(scenarios_display.style.format({"Threshold": "{:.2f}", "대상 비율": "{:.1%}", "Recall": "{:.1%}", "Precision": "{:.1%}", "F1": "{:.1%}"}), hide_index=True, width="stretch")
    st.markdown('<div class="info-card"><b>사용자가 결정할 질문</b><br>이탈 고객 한 명을 놓치는 비용(FN)은 불필요한 접촉 한 건(FP)보다 얼마나 큰가? 그리고 한 회차에 실제로 검토할 수 있는 고객은 몇 명인가?</div>', unsafe_allow_html=True)
    boundary_note()


def risk_level(probability: float, scenarios: pd.DataFrame) -> tuple[str, str]:
    low = float(scenarios["threshold"].min())
    high = float(scenarios["threshold"].max())
    if probability >= high:
        return "높은 우선순위", "risk-high"
    if probability >= low:
        return "검토 후보", "risk-mid"
    return "낮은 우선순위", "risk-low"


def input_context(record: dict, train: pd.DataFrame) -> list[str]:
    contexts = []
    if str(record["subscription_type"]) == "Free":
        contexts.append("Free 구독 유형")
    if str(record["customer_service_inquiries"]) == "High":
        contexts.append("높은 고객 문의 수준")
    if float(record["weekly_hours"]) < float(train["weekly_hours"].median()):
        contexts.append("중앙값보다 낮은 주간 청취시간")
    if float(record["song_skip_rate"]) > float(train["song_skip_rate"].median()):
        contexts.append("중앙값보다 높은 스킵률")
    if int(record["num_subscription_pauses"]) >= int(train["num_subscription_pauses"].quantile(.75)):
        contexts.append("상위 구간의 구독 일시정지 횟수")
    return contexts or ["선택한 핵심 행동 변수에서 뚜렷한 진단 플래그 없음"]


def strategy_review_panel(record: dict, probability: float, tables: dict, scenario: pd.Series) -> None:
    strategy = recommend_retention_strategy(
        record,
        probability,
        tables["strategy_thresholds"],
        low_risk_threshold=float(tables["scenarios"]["threshold"].min()),
        high_risk_threshold=float(tables["scenarios"]["threshold"].max()),
    )
    st.markdown('<div class="section-label">유지 전략 검토</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="strategy-strip">'
        f'<div class="strategy-cell"><span>운영 단계</span><b>{html.escape(str(strategy["campaign_tier"]))}</b></div>'
        f'<div class="strategy-cell"><span>고객 세그먼트</span><b>{html.escape(str(strategy["strategy_segment"]))}</b></div>'
        f'<div class="strategy-cell"><span>행동 가능 신호</span><b>{int(strategy["risk_signal_count"])}개</b></div>'
        f'<div class="strategy-cell"><span>검토 상태</span><b>담당자 승인 필요</b></div>'
        '</div>',
        unsafe_allow_html=True,
    )
    primary, alternative = st.columns(2)
    with primary:
        st.markdown(
            '<div class="action-card"><div class="action-rank">1순위 행동 후보</div>'
            f'<div class="action-title">{html.escape(str(strategy["primary_action"]))}</div>'
            f'<div class="action-row"><b>왜</b> · {html.escape(str(strategy["action_rationale"]))}</div>'
            f'<div class="action-row"><b>언제</b> · {html.escape(str(strategy["action_timing"]))}</div>'
            f'<div class="action-row"><b>KPI</b> · {html.escape(str(strategy["validation_kpi"]))}</div></div>',
            unsafe_allow_html=True,
        )
    with alternative:
        st.markdown(
            '<div class="action-card alt"><div class="action-rank" style="color:#315d78">대안 행동 후보</div>'
            f'<div class="action-title">{html.escape(str(strategy["alternative_action"]))}</div>'
            f'<div class="action-row"><b>관찰 신호</b> · {html.escape(str(strategy["observed_signals"]))}</div>'
            '<div class="action-row"><b>사용하지 않은 근거</b> · 나이·지역·고객 ID</div>'
            f'<div class="action-row"><b>근거 수준</b> · {html.escape(str(strategy["evidence_level"]))}</div></div>',
            unsafe_allow_html=True,
        )

    decision_options = [
        str(strategy["primary_action"]),
        str(strategy["alternative_action"]),
        "담당자 직접 검토",
        "이번 회차 보류",
    ]
    decision = st.radio(
        "담당자 결정",
        decision_options,
        horizontal=True,
        key=f"action_decision_{int(record['customer_id'])}",
        help="선택은 이 화면과 다운로드 기록에만 반영되며 외부 CRM 조치를 실행하지 않습니다.",
    )
    note = st.text_input(
        "검토 메모(선택)",
        placeholder="예: 최근 문의 해결 여부를 먼저 확인",
        key=f"action_note_{int(record['customer_id'])}",
    )
    review_record = pd.DataFrame([{
        "customer_id": int(record["customer_id"]),
        "churn_probability": probability,
        "operating_scenario": str(scenario["scenario_label"]),
        "campaign_tier": strategy["campaign_tier"],
        "strategy_segment": strategy["strategy_segment"],
        "observed_signals": strategy["observed_signals"],
        "suggested_action": strategy["primary_action"],
        "alternative_action": strategy["alternative_action"],
        "selected_action": decision,
        "validation_kpi": strategy["validation_kpi"],
        "review_note": note,
        "evidence_level": strategy["evidence_level"],
        "external_action_executed": False,
    }])
    st.download_button(
        "담당자 검토 기록 CSV 다운로드",
        review_record.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"customer_{int(record['customer_id'])}_retention_review.csv",
        mime="text/csv",
    )
    st.caption("행동 후보는 EDA 연관성과 투명한 규칙에서 생성됩니다. 선택해도 CRM 발송·할인·고객 접촉은 실행되지 않습니다.")


def campaign_planning_panel(test: pd.DataFrame, predictions: pd.DataFrame, scenario: pd.Series) -> None:
    st.markdown(
        '<div class="info-card"><b>가정 기반 캠페인 계획</b><br>'
        f'현재 운영 선택은 <b>{html.escape(str(scenario["scenario_label"]))}</b>입니다. 예산·캠페인 변동비·연간 매출 대용치·추가 유지 성공률은 사용자 가정이며, '
        '위험 점수 합계는 확정 이탈자 수가 아닙니다. 실제 Uplift·ROI로 해석하지 마세요.</div>',
        unsafe_allow_html=True,
    )
    base = predictions[["customer_id", "churn_probability"]].merge(
        test[["customer_id", "subscription_type"]], on="customer_id", how="inner", validate="one_to_one"
    ).rename(columns={"churn_probability": "probability", "subscription_type": "plan"})
    plan_order = base.groupby("plan")["probability"].mean().sort_values(ascending=False).index.tolist()

    state = st.session_state
    state.setdefault("campaign_driver", "budget")
    state.setdefault("campaign_budget", 5_000_000)
    state.setdefault("campaign_count", 1666)
    state.setdefault("campaign_percent", 2.2)

    def set_campaign_driver(name: str) -> None:
        st.session_state["campaign_driver"] = name

    st.markdown('<div class="section-label">계획 가정과 대상 범위</div>', unsafe_allow_html=True)
    input_box = st.container()
    assumption_left, assumption_right = st.columns([1.35, 1])
    with assumption_left:
        success_rate = st.slider("접촉 고객의 추가 유지 성공률 가정", 0.0, 0.5, 0.10, 0.01, key="campaign_success_rate")
    with assumption_right:
        st.markdown(
            '<div class="note-card"><b>연동 입력</b><br>예산·인원·상위 비율 중 마지막으로 수정한 값을 기준으로 나머지를 자동 환산합니다.</div>',
            unsafe_allow_html=True,
        )

    counts = base.groupby("plan")["probability"].agg(["size", "mean"])
    economic_defaults = plan_economic_assumptions(plan_order).set_index("plan")
    config = st.data_editor(
        pd.DataFrame({
            "구독 유형": plan_order,
            "보유 고객": [int(counts.loc[plan, "size"]) for plan in plan_order],
            "평균 위험 점수": [float(counts.loc[plan, "mean"]) for plan in plan_order],
            "기본 접촉 채널": [str(economic_defaults.loc[plan, "default_channel"]) for plan in plan_order],
            "포함": True,
            "1인 캠페인 변동비(원)": [int(economic_defaults.loc[plan, "contact_cost"]) for plan in plan_order],
            "연간 매출 대용치(원)": [int(economic_defaults.loc[plan, "customer_value"]) for plan in plan_order],
        }),
        hide_index=True,
        disabled=["구독 유형", "보유 고객", "평균 위험 점수", "기본 접촉 채널"],
        column_config={
            "평균 위험 점수": st.column_config.NumberColumn("평균 위험 점수", format="percent"),
            "포함": st.column_config.CheckboxColumn("포함"),
            "1인 캠페인 변동비(원)": st.column_config.NumberColumn("1인 캠페인 변동비(원)", help="발송료뿐 아니라 해당 채널의 자동화·상담 운영비를 포함한 가정입니다.", min_value=0, step=100),
            "연간 매출 대용치(원)": st.column_config.NumberColumn("연간 매출 대용치(원)", help="LTV나 기여이익이 아닌 외부 공개 요금 기반 참고값입니다.", min_value=0, step=1000),
        },
        width="stretch",
        key="campaign_plan_config_v2",
    ).set_index("구독 유형")
    with st.expander("구독 유형별 기본 가정의 추론 근거"):
        assumption_view = economic_defaults.reset_index().rename(columns={
            "plan": "구독 유형", "contact_cost": "1인 캠페인 변동비", "customer_value": "연간 매출 대용치",
            "default_channel": "기본 접촉 채널", "basis": "추론 근거",
        })
        st.dataframe(assumption_view, hide_index=True, width="stretch")
        st.caption("PlaylistPro의 가격·마진·채널 원가가 없어 [외부 공개 음악 구독 요금](https://www.spotify.com/kr-ko/premium/)을 부가세 제외 연간 매출로 환산했습니다(2026-07-22 기준). Free는 근거가 없어 0원이며, 모든 값은 편집 가능한 계획 가정입니다. LTV·기여이익이 아닙니다.")

    included = [plan for plan in plan_order if bool(config.loc[plan, "포함"])]
    if not included:
        st.warning("계획에 포함할 구독 유형을 하나 이상 선택해 주세요.")
        return
    work = base.loc[base["plan"].isin(included)].copy()
    work["cost"] = work["plan"].map(config["1인 캠페인 변동비(원)"].astype(float))
    work["customer_value"] = work["plan"].map(config["연간 매출 대용치(원)"].astype(float))
    work["assumed_net_value"] = work["probability"] * success_rate * work["customer_value"] - work["cost"]
    uniform_economics = config.loc[included, "1인 캠페인 변동비(원)"].nunique() == 1 and config.loc[included, "연간 매출 대용치(원)"].nunique() == 1
    sort_columns = ["probability"] if uniform_economics else ["assumed_net_value", "probability"]
    work = work.sort_values(sort_columns, ascending=False).reset_index(drop=True)
    pool = len(work)
    cumulative_spend = work["cost"].cumsum().to_numpy()

    driver = state["campaign_driver"]
    if driver == "budget":
        selected_count = int((cumulative_spend <= float(state["campaign_budget"])).sum())
    elif driver == "count":
        selected_count = min(int(state["campaign_count"]), pool)
    else:
        selected_count = int(round(float(state["campaign_percent"]) / 100 * pool))
    selected_count = max(0, min(selected_count, pool))
    selected = work.iloc[:selected_count]
    spend = float(selected["cost"].sum())
    risk_score_sum = float(selected["probability"].sum())
    assumed_retained = risk_score_sum * success_rate
    assumed_benefit = float((selected["probability"] * success_rate * selected["customer_value"]).sum())
    assumed_net = assumed_benefit - spend

    if driver != "budget":
        state["campaign_budget"] = int(spend)
    if driver != "count":
        state["campaign_count"] = int(selected_count)
    if driver != "percent":
        state["campaign_percent"] = round(selected_count / pool * 100, 1) if pool else 0.0
    with input_box:
        a, b, c = st.columns(3)
        a.number_input("캠페인 예산(원)", min_value=0, step=500_000, key="campaign_budget", on_change=set_campaign_driver, args=("budget",))
        b.number_input("연락 인원(명)", min_value=0, max_value=pool, step=50, key="campaign_count", on_change=set_campaign_driver, args=("count",))
        c.number_input("상위 위험군(%)", min_value=0.0, max_value=100.0, step=0.5, key="campaign_percent", on_change=set_campaign_driver, args=("percent",))

    sort_label = "위험 점수 순" if uniform_economics else "사용자 가정 매출효과-비용 순"
    st.caption(f"대상 풀 {pool:,}명 · 현재 정렬 기준: {sort_label} · 제공 test.csv에는 정답 라벨이 없습니다.")
    for column, item in zip(st.columns(4), [
        ("가정 집행액", f"{spend:,.0f}원", f"{selected_count:,}명 검토"),
        ("위험 점수 합계", f"{risk_score_sum:,.0f}", f"평균 {risk_score_sum / selected_count:.1%}" if selected_count else "-"),
        ("가정 추가 유지", f"{assumed_retained:,.1f}명", f"성공률 가정 {success_rate:.0%}"),
        ("가정 매출효과-비용", f"{assumed_net:,.0f}원", f"매출효과/비용 {assumed_benefit / spend:.1f}배" if spend else "-"),
    ]):
        with column:
            metric_card(*item)

    st.markdown('<div class="section-label">캠페인 퍼널과 구독 유형별 배정</div>', unsafe_allow_html=True)
    funnel_col, allocation_col = st.columns([1.2, 1])
    with funnel_col:
        funnel = go.Figure(go.Funnel(
            y=["선택 구독 유형 고객", "연락 대상", "위험 점수 합계", "가정 유지 전환"],
            x=[pool, selected_count, risk_score_sum, assumed_retained],
            textinfo="value+percent initial",
            marker={"color": ["#315d78", "#e4573d", "#e8a23a", "#2f7d62"]},
            connector={"line": {"color": "#dbe5ef"}},
        ))
        funnel.update_layout(title="가정 기반 캠페인 퍼널")
        st.plotly_chart(plot_style(funnel, 360), width="stretch")
    with allocation_col:
        allocation = selected.groupby("plan", observed=True).agg(
            연락배정=("customer_id", "size"), 위험점수합계=("probability", "sum"), 가정집행액=("cost", "sum")
        ).reindex(plan_order, fill_value=0).reset_index().rename(columns={"plan": "구독 유형"})
        fig = px.bar(allocation, x="구독 유형", y="연락배정", color="위험점수합계", text="연락배정", color_continuous_scale=["#dce8ef", "#e4573d"])
        fig.update_layout(title="구독 유형별 연락 배정", xaxis_title=None, yaxis_title="연락 인원", coloraxis_colorbar_title="위험 점수 합계")
        st.plotly_chart(plot_style(fig, 360), width="stretch")

    st.markdown('<div class="section-label">연락 범위별 가정 매출효과-비용 곡선</div>', unsafe_allow_html=True)
    curve_mode = st.radio("곡선 비교 방식", ["현재 요금제별 설정", "접촉 단가 시나리오 비교"], horizontal=True, key="campaign_curve_mode")
    scenario_costs: list[float] = []
    if curve_mode == "접촉 단가 시나리오 비교":
        c1, c2, c3 = st.columns(3)
        scenario_costs = [
            float(c1.number_input("낮은 단가(원)", min_value=0, value=100, step=100)),
            float(c2.number_input("중간 단가(원)", min_value=0, value=1000, step=100)),
            float(c3.number_input("높은 단가(원)", min_value=0, value=5000, step=100)),
        ]

    def net_curve(frame: pd.DataFrame, fixed_cost: float | None = None) -> tuple[np.ndarray, np.ndarray]:
        ordered = frame.copy()
        if fixed_cost is not None:
            ordered["cost"] = fixed_cost
            ordered["assumed_net_value"] = ordered["probability"] * success_rate * ordered["customer_value"] - fixed_cost
            ordered = ordered.sort_values(["assumed_net_value", "probability"], ascending=False)
        cumulative_benefit = (ordered["probability"] * success_rate * ordered["customer_value"]).cumsum().to_numpy()
        cumulative_net = (cumulative_benefit - ordered["cost"].cumsum().to_numpy()) / 1_000_000
        x = np.arange(1, len(ordered) + 1) / len(ordered) * 100
        step = max(1, len(ordered) // 350)
        indexes = np.unique(np.concatenate([np.arange(0, len(ordered), step), [len(ordered) - 1]]))
        return x[indexes], cumulative_net[indexes]

    curve_tabs = st.tabs(["전체"] + plan_order)
    frames = [("전체", work)] + [(plan, work.loc[work["plan"].eq(plan)].reset_index(drop=True)) for plan in plan_order]
    for tab, (name, frame) in zip(curve_tabs, frames):
        with tab:
            fig = go.Figure()
            if scenario_costs:
                for cost, color, dash in zip(scenario_costs, [COLORS["green"], COLORS["blue"], COLORS["orange"]], ["solid", "dash", "dot"]):
                    x, y = net_curve(frame, cost)
                    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=f"접촉 단가 {cost:,.0f}원", line={"color": color, "dash": dash, "width": 3}))
            else:
                x, y = net_curve(frame)
                fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name="가정 매출효과-비용", fill="tozeroy", line={"color": COLORS["orange"], "width": 3}))
                if name == "전체" and selected_count:
                    marker_index = int(np.argmin(np.abs(x - selected_count / pool * 100)))
                    fig.add_trace(go.Scatter(x=[x[marker_index]], y=[y[marker_index]], mode="markers", name="현재 계획", marker={"size": 12, "color": COLORS["navy"]}))
            fig.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            fig.update_layout(title=f"{name} · 연락 범위별 가정 매출효과-비용", xaxis_title="그룹 내 상위 위험 고객 연락 범위", xaxis_ticksuffix="%", yaxis_title="가정 매출효과-비용(백만원)", legend_orientation="h")
            st.plotly_chart(plot_style(fig, 390), width="stretch")
    st.caption("모든 편익·ROI·유지 전환 값은 사용자 입력 가정에 따른 민감도 분석입니다. 실제 성과는 대조군을 둔 캠페인 실험과 미래 라벨로 검증해야 합니다.")


def prioritization_page(tables: dict, model, scenario: pd.Series) -> None:
    section_header("CUSTOMER PRIORITY", "고객 위험을 확인하고 유지 활동을 선택하세요", "저장된 CatBoost가 위험 점수를 재계산하고, 별도의 투명한 전략 규칙이 담당자 검토용 행동 후보와 KPI를 제시합니다.")
    train, test, predictions = tables["train"], tables["test"], tables["predictions"]
    single_tab, batch_tab, planning_tab = st.tabs(["단건 시뮬레이션", "배치 우선순위", "가정 기반 계획"])
    with single_tab:
        base_id = st.selectbox("기준 고객 ID", test["customer_id"].head(3000).tolist(), help="test.csv의 기존 고객을 불러온 뒤 입력값을 수정할 수 있습니다.")
        base = test.loc[test["customer_id"].eq(base_id)].iloc[0]
        categorical = ["location", "subscription_type", "payment_plan", "payment_method", "customer_service_inquiries"]
        options = {column: sorted(pd.concat([train[column], test[column]]).dropna().astype(str).unique().tolist()) for column in categorical}
        key_prefix = f"score_{int(base_id)}"
        with st.form("customer_score_form"):
            st.markdown('<div class="section-label">구독·고객 맥락</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            record = {"customer_id": int(base["customer_id"])}
            with c1:
                record["age"] = st.number_input("나이", int(train.age.min()), int(train.age.max()), int(base.age), key=f"{key_prefix}_age")
                record["location"] = st.selectbox("지역", options["location"], index=options["location"].index(str(base.location)), key=f"{key_prefix}_location")
                record["subscription_type"] = st.selectbox("구독 유형", options["subscription_type"], index=options["subscription_type"].index(str(base.subscription_type)), key=f"{key_prefix}_subscription")
            with c2:
                record["payment_plan"] = st.selectbox("결제 주기", options["payment_plan"], index=options["payment_plan"].index(str(base.payment_plan)), key=f"{key_prefix}_plan")
                record["payment_method"] = st.selectbox("결제 수단", options["payment_method"], index=options["payment_method"].index(str(base.payment_method)), key=f"{key_prefix}_method")
                record["customer_service_inquiries"] = st.selectbox("고객 문의 수준", options["customer_service_inquiries"], index=options["customer_service_inquiries"].index(str(base.customer_service_inquiries)), key=f"{key_prefix}_inquiry")
            with c3:
                record["num_subscription_pauses"] = st.number_input("구독 일시정지 횟수", 0, int(train.num_subscription_pauses.max()), int(base.num_subscription_pauses), key=f"{key_prefix}_pause")
                record["signup_date"] = st.number_input("가입일 상대값", int(train.signup_date.min()), int(train.signup_date.max()), int(base.signup_date), key=f"{key_prefix}_signup")
                record["notifications_clicked"] = st.number_input("알림 클릭 수", 0, int(train.notifications_clicked.max()), int(base.notifications_clicked), key=f"{key_prefix}_notification")

            st.markdown('<div class="section-label">이용 행동</div>', unsafe_allow_html=True)
            c1, c2, c3 = st.columns(3)
            with c1:
                record["weekly_hours"] = st.number_input("주간 청취시간", 0.0, float(train.weekly_hours.max()), float(base.weekly_hours), .5, key=f"{key_prefix}_hours")
                record["average_session_length"] = st.number_input("평균 세션 길이", 0.0, float(train.average_session_length.max()), float(base.average_session_length), 1.0, key=f"{key_prefix}_session")
                record["song_skip_rate"] = st.slider("스킵률", 0.0, 1.0, float(base.song_skip_rate), .01, key=f"{key_prefix}_skip")
            with c2:
                record["weekly_songs_played"] = st.number_input("주간 재생곡 수", 0, int(train.weekly_songs_played.max()), int(base.weekly_songs_played), key=f"{key_prefix}_songs")
                record["weekly_unique_songs"] = st.number_input("주간 고유곡 수", 0, int(train.weekly_unique_songs.max()), int(base.weekly_unique_songs), key=f"{key_prefix}_unique")
                record["num_favorite_artists"] = st.number_input("선호 아티스트 수", 0, int(train.num_favorite_artists.max()), int(base.num_favorite_artists), key=f"{key_prefix}_artists")
            with c3:
                record["num_platform_friends"] = st.number_input("플랫폼 친구 수", 0, int(train.num_platform_friends.max()), int(base.num_platform_friends), key=f"{key_prefix}_friends")
                record["num_playlists_created"] = st.number_input("생성 플레이리스트 수", 0, int(train.num_playlists_created.max()), int(base.num_playlists_created), key=f"{key_prefix}_playlists")
                record["num_shared_playlists"] = st.number_input("공유 플레이리스트 수", 0, int(train.num_shared_playlists.max()), int(base.num_shared_playlists), key=f"{key_prefix}_shared")
            submitted = st.form_submit_button("위험 점수 다시 계산", type="primary", width="stretch")

        row = pd.DataFrame([record], columns=tables["metadata"]["input_columns"])
        probability = float(model.predict_proba(row)[:, 1][0])
        label, css_class = risk_level(probability, tables["scenarios"])
        selected = probability >= float(scenario["threshold"])
        st.markdown('<div class="section-label">재계산 결과</div>', unsafe_allow_html=True)
        r1, r2, r3, r4 = st.columns(4)
        with r1:
            metric_card("이탈 위험 점수", f"{probability:.1%}", "저장된 CatBoost 재추론")
        with r2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">위험 구간</div><div style="margin-top:16px"><span class="{css_class}">{label}</span></div></div>', unsafe_allow_html=True)
        with r3:
            metric_card("현재 시나리오", str(scenario["scenario_label"]), f"Threshold {float(scenario['threshold']):.2f}")
        with r4:
            metric_card("검토 대상 포함", "예" if selected else "아니오", "자동 접촉이 아닌 목록 포함 여부")
        contexts = input_context(record, train)
        st.markdown(
            '<div class="note-card"><b>진단 맥락</b><br>' + " · ".join(html.escape(item) for item in contexts) +
            '<br><span class="small">이는 개인 단위 원인 설명이나 SHAP 값이 아니라, EDA와 전역 중요도에 근거한 검토 맥락입니다.</span></div>',
            unsafe_allow_html=True,
        )
        strategy_review_panel(record, probability, tables, scenario)
        if submitted:
            st.success("수정한 입력값으로 위험 점수를 다시 계산했습니다.")

    with batch_tab:
        threshold = st.slider("배치 최소 위험 점수", 0.0, 1.0, float(scenario["threshold"]), .01)
        queue = tables["strategy_queue"]
        filtered = queue.loc[queue["churn_probability"].ge(threshold)].copy()
        filtered["selected_action"] = "담당자 미검토"
        filtered["operating_scenario"] = str(scenario["scenario_label"])
        filtered["external_action_executed"] = False
        c1, c2, c3 = st.columns(3)
        c1.metric("검토 고객", f"{len(filtered):,}명")
        c2.metric("전체 대비", f"{len(filtered) / len(queue):.1%}")
        c3.metric("평균 위험 점수", f"{filtered['churn_probability'].mean():.1%}" if len(filtered) else "-")
        if not filtered.empty:
            segment_col, tier_col = st.columns(2)
            with segment_col:
                segment_counts = filtered["strategy_segment"].value_counts().rename_axis("전략 세그먼트").reset_index(name="고객 수")
                fig = px.bar(segment_counts, x="고객 수", y="전략 세그먼트", orientation="h", text="고객 수")
                fig.update_traces(marker_color=COLORS["orange"])
                fig.update_layout(title="검토 큐의 전략 세그먼트", xaxis_title="고객 수", yaxis_title=None)
                st.plotly_chart(plot_style(fig, 340), width="stretch")
            with tier_col:
                tier_counts = filtered["campaign_tier"].value_counts().rename_axis("운영 단계").reset_index(name="고객 수")
                fig = px.pie(tier_counts, names="운영 단계", values="고객 수", hole=.58, color="운영 단계",
                             color_discrete_map={"집중 관리": COLORS["orange"], "자동화 검토": COLORS["blue"], "관찰 유지": COLORS["green"]})
                fig.update_layout(title="운영 단계 구성", showlegend=True)
                st.plotly_chart(plot_style(fig, 340), width="stretch")
        st.markdown('<div class="section-label">고객별 전략 검토 표</div>', unsafe_allow_html=True)
        control_left, control_right = st.columns([1.2, 1])
        with control_left:
            queue_view_mode = st.radio(
                "표시 방식",
                ["운영 단계·세그먼트 대표 보기", "위험 점수 순"],
                horizontal=True,
                help="위험 점수 순은 최상위 고객 특성이 유사해 여러 컬럼이 반복될 수 있습니다.",
            )
        with control_right:
            available_segments = filtered["strategy_segment"].drop_duplicates().tolist()
            selected_segments = st.multiselect("전략 세그먼트 필터", available_segments, default=available_segments)
        view_source = filtered.loc[filtered["strategy_segment"].isin(selected_segments)].copy()
        if queue_view_mode == "운영 단계·세그먼트 대표 보기":
            rows_per_segment = st.slider("운영 단계·세그먼트 조합별 표시 고객 수", 5, 100, 25, 5)
            view_source["_representative_rank"] = view_source.groupby(
                ["campaign_tier", "strategy_segment"], sort=False
            ).cumcount()
            table_rows = view_source.loc[view_source["_representative_rank"].lt(rows_per_segment)].sort_values(
                ["_representative_rank", "campaign_tier", "strategy_segment"]
            )
        else:
            table_rows = view_source.head(1000)

        display_columns = [
            "customer_id", "churn_probability", "campaign_tier", "strategy_segment",
            "risk_signal_count", "primary_signal", "primary_action", "alternative_action",
            "validation_kpi", "evidence_level",
        ]
        display = table_rows[display_columns].rename(columns={
            "customer_id": "고객 ID", "churn_probability": "위험 점수", "campaign_tier": "운영 단계",
            "strategy_segment": "전략 세그먼트", "risk_signal_count": "신호 수", "primary_signal": "1차 신호",
            "primary_action": "1순위 행동 후보", "alternative_action": "대안 행동", "validation_kpi": "검증 KPI",
            "evidence_level": "근거 수준",
        })
        st.caption(
            f"현재 표 {len(display):,}명 · 운영 단계 {display['운영 단계'].nunique()}개 · "
            f"전략 세그먼트 {display['전략 세그먼트'].nunique()}개 · 1차 신호 {display['1차 신호'].nunique()}개 · "
            f"1순위 행동 {display['1순위 행동 후보'].nunique()}개"
        )
        st.dataframe(display, hide_index=True, width="stretch")
        st.download_button("전략 포함 검토 큐 CSV 다운로드", filtered.to_csv(index=False).encode("utf-8-sig"), "catboost_retention_strategy_queue.csv", "text/csv")

        st.caption("행동 후보는 자동 실행되지 않으며 모든 행의 selected_action은 담당자 미검토 상태로 내려받습니다. test 데이터에는 정답 라벨이 없습니다.")

    with planning_tab:
        campaign_planning_panel(test, predictions, scenario)

    boundary_note()


def main() -> None:
    tables, model = load_workspace()
    metadata = tables["metadata"]
    scenarios = tables["scenarios"]
    with st.sidebar:
        st.markdown('<div class="brand"><span class="brand-mark">♫</span><span class="brand-name">PlaylistPro</span></div>', unsafe_allow_html=True)
        st.markdown('<div class="sidebar-kicker">Decision workspace</div>', unsafe_allow_html=True)
        page = st.radio(
            "메뉴",
            ["프로젝트 요약", "데이터·고객 인사이트", "개선 실험", "모델 선정", "운영 시나리오", "고객 우선순위"],
            label_visibility="collapsed",
        )
        st.divider()
        st.markdown('<div class="sidebar-kicker">Operating policy</div>', unsafe_allow_html=True)
        scenario_ids = scenarios["scenario_id"].tolist()
        default_id = metadata.get("app_default_scenario_id", "balanced_f1")
        scenario_id = st.selectbox(
            "검토 시나리오",
            scenario_ids,
            index=scenario_ids.index(default_id) if default_id in scenario_ids else 0,
            format_func=lambda value: str(selected_scenario(scenarios, value)["scenario_label"]),
        )
        chosen = selected_scenario(scenarios, scenario_id)
        st.caption(f"Threshold {float(chosen['threshold']):.2f} · OOF 대상 {float(chosen['oof_target_rate']):.1%}")
        st.divider()
        st.markdown("**현재 모델** · CatBoost")
        st.caption("SHA-256 무결성 검증 후 로드\n\n화면에서 재학습하지 않음")

    scenario = selected_scenario(scenarios, scenario_id)
    if page == "프로젝트 요약":
        project_summary_page(tables, scenario)
    elif page == "데이터·고객 인사이트":
        customer_insights_page(tables)
    elif page == "개선 실험":
        improvement_page(tables)
    elif page == "모델 선정":
        model_comparison_page(tables)
    elif page == "운영 시나리오":
        operations_page(tables, scenario)
    else:
        prioritization_page(tables, model, scenario)


if __name__ == "__main__":
    main()
