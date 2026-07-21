"""PlaylistPro 고객 이탈 위험 의사결정 지원 대시보드."""
from __future__ import annotations

import hashlib
import html
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
    div[data-testid="stDataFrame"] { border-radius:13px; overflow:hidden; border:1px solid #edf0f2; }
    div[data-testid="stPlotlyChart"] { background:white; border:1px solid #edf0f2; border-radius:16px; padding:4px; box-shadow:0 5px 18px rgba(15,23,42,.025); }
    [data-testid="stMetric"] { background:white; border:1px solid #e5e7eb; padding:14px 16px; border-radius:15px; }
    .boundary { background:#fffdf8; border:1px solid #f2dfb4; border-left:4px solid #e8a23a; border-radius:13px; padding:14px 16px; line-height:1.6; color:#62491d; margin-top:18px; }
    .small { color:#64748b; font-size:12px; line-height:1.55; }
    @media (max-width:900px) {
        .flow { grid-template-columns:repeat(2,1fr); }
        .hero { padding:25px 23px; }
        .block-container { padding-left:1rem; padding-right:1rem; }
        .metric-card { padding:13px 11px; min-height:104px; }
        .metric-label { font-size:10px; }
        .metric-value { font-size:18px; white-space:nowrap; }
        .metric-delta { font-size:9px; }
    }
    @media (max-width:620px) {
        .flow { grid-template-columns:1fr; }
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
        '<h1>누구에게 먼저 리텐션 자원을 배정할 것인가?</h1>'
        '<p>PlaylistPro의 고객 행동 인사이트와 최종 CatBoost 위험 점수를 연결해, 이탈 고객 누락과 불필요한 접촉 사이의 운영 기준을 선택하는 의사결정 화면입니다.</p>'
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
        ("06", "고객 우선순위", "점수 기반 검토 목록"),
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
            "검증된 모델 로드, 위험 점수 재계산, 고객 순위·Top-K·오류 수 비교, CSV 목록 생성을 수행합니다. 캠페인 실행과 할인 결정은 포함하지 않습니다.",
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
        fig.update_layout(title="인사이트 기반 전처리 Logistic OOF 비교", yaxis_range=[0.895, 0.91], yaxis_title="PR-AUC", xaxis_title=None)
        st.plotly_chart(plot_style(fig), width="stretch")
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


def prioritization_page(tables: dict, model, scenario: pd.Series) -> None:
    section_header("CUSTOMER PRIORITY", "고객 입력을 바꾸고 위험 점수를 다시 계산하세요", "저장된 CatBoost가 입력값을 실제 재추론하며, 결과는 자동 조치가 아닌 검토 우선순위로 사용합니다.")
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
        if submitted:
            st.success("수정한 입력값으로 위험 점수를 다시 계산했습니다.")

    with batch_tab:
        threshold = st.slider("배치 최소 위험 점수", 0.0, 1.0, float(scenario["threshold"]), .01)
        filtered = predictions.loc[predictions["churn_probability"].ge(threshold)].sort_values("churn_probability", ascending=False)
        c1, c2, c3 = st.columns(3)
        c1.metric("검토 고객", f"{len(filtered):,}명")
        c2.metric("전체 대비", f"{len(filtered) / len(predictions):.1%}")
        c3.metric("평균 위험 점수", f"{filtered['churn_probability'].mean():.1%}" if len(filtered) else "-")
        st.dataframe(filtered.head(1000), hide_index=True, width="stretch")
        st.download_button("검토 목록 CSV 다운로드", filtered.to_csv(index=False).encode("utf-8-sig"), "catboost_customer_priority.csv", "text/csv")

        st.caption("표에는 저장된 test.csv 배치 점수만 표시됩니다. test 데이터에는 정답 라벨이 없습니다.")

    with planning_tab:
        st.markdown(
            '<div class="info-card"><b>가정 기반 계획 도구</b><br>'
            '아래 값은 실제 Uplift·ROI가 아니라 예산과 접촉 용량을 비교하기 위한 사용자 가정입니다.</div>',
            unsafe_allow_html=True,
        )
        c1, c2 = st.columns(2)
        with c1:
            contact_cost = st.number_input("고객 1명당 접촉 비용(원)", min_value=0, value=5000, step=500)
            customer_value = st.number_input("유지 고객 1명 가치(원)", min_value=0, value=120000, step=5000)
        with c2:
            success_rate = st.slider("접촉 후 유지 전환율 가정", 0.0, 1.0, .10, .01)
            capacity = st.number_input("최대 접촉 인원", min_value=1, max_value=len(predictions), value=min(10000, len(predictions)), step=500)
        audience = predictions.sort_values("churn_probability", ascending=False).head(int(capacity))
        assumed_saves = float(audience["churn_probability"].sum() * success_rate)
        assumed_cost = len(audience) * contact_cost
        assumed_value = assumed_saves * customer_value
        assumed_net = assumed_value - assumed_cost
        p1, p2, p3 = st.columns(3)
        p1.metric("가정 유지 전환", f"{assumed_saves:,.1f}명")
        p2.metric("가정 총비용", f"{assumed_cost:,.0f}원")
        p3.metric("가정 순편익", f"{assumed_net:,.0f}원")
        st.caption("예측 확률 합계에 사용자가 입력한 전환율·가치를 곱한 단순 시나리오이며, 캠페인 실험 결과가 아닙니다.")

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
