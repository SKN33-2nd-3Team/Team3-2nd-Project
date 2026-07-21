"""Explainable retention-action hypotheses layered on top of churn ranking.

The predictive model ranks review priority.  This module deliberately keeps
action suggestions separate: rules use only service/usage/subscription signals,
never age, location, or customer identifiers, and every suggestion requires a
human decision and a controlled experiment before its effect can be claimed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import pandas as pd


EVIDENCE_LEVEL = "관찰 연관성 기반 가설 · 캠페인 효과 미검증"

# Presentation-planning assumptions, not observed PlaylistPro revenue or costs.
# Revenue proxies use a public Korean music-plan benchmark excluding VAT;
# variable costs include delivery plus the operating effort implied by the channel.
PLAN_ECONOMIC_ASSUMPTIONS: dict[str, dict[str, object]] = {
    "Free": {
        "contact_cost": 100,
        "customer_value": 0,
        "default_channel": "앱 푸시·자동 메시지",
        "basis": "발송·자동화 운영비 100원 가정 · 광고·유료 전환 근거가 없어 구독 매출 대용치는 0원",
    },
    "Student": {
        "contact_cost": 200,
        "customer_value": 72_000,
        "default_channel": "자동화 혜택 안내",
        "basis": "자동 발송·운영비 200원 가정 · 외부 공개 요금 월 6,000원(부가세 제외) × 12개월",
    },
    "Premium": {
        "contact_cost": 500,
        "customer_value": 130_800,
        "default_channel": "개인화 디지털 메시지",
        "basis": "개인화 발송·운영비 500원 가정 · 외부 공개 요금 월 10,900원(부가세 제외) × 12개월",
    },
    "Family": {
        "contact_cost": 4_500,
        "customer_value": 196_200,
        "default_channel": "상담 우선·혜택 안내",
        "basis": "상담 인력·발송 운영비 4,500원 가정 · 공유형 외부 요금 월 16,350원(부가세 제외) × 12개월 대용",
    },
}


@dataclass(frozen=True)
class StrategyThresholds:
    """Training-distribution cut points used only for transparent signal flags."""

    low_weekly_hours: float
    high_skip_rate: float
    high_subscription_pauses: float


def plan_economic_assumptions(plans: list[str]) -> pd.DataFrame:
    """Return editable plan defaults with their explicit assumption basis."""

    rows = []
    for plan in plans:
        assumption = PLAN_ECONOMIC_ASSUMPTIONS.get(plan, {
            "contact_cost": 500,
            "customer_value": 130_800,
            "default_channel": "담당자 지정 필요",
            "basis": "알 수 없는 유형이므로 Premium 외부 벤치마크를 임시 적용",
        })
        rows.append({"plan": plan, **assumption})
    return pd.DataFrame(rows)


def derive_strategy_thresholds(train: pd.DataFrame) -> StrategyThresholds:
    """Derive fixed quartile cut points from the labeled training snapshot."""

    required = {"weekly_hours", "song_skip_rate", "num_subscription_pauses"}
    missing = sorted(required - set(train.columns))
    if missing:
        raise ValueError(f"전략 임계값 계산에 필요한 컬럼이 없습니다: {missing}")
    return StrategyThresholds(
        low_weekly_hours=float(train["weekly_hours"].quantile(0.25)),
        high_skip_rate=float(train["song_skip_rate"].quantile(0.75)),
        high_subscription_pauses=float(train["num_subscription_pauses"].quantile(0.75)),
    )


def _signal_catalog(record: Mapping[str, object], thresholds: StrategyThresholds) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    if str(record.get("customer_service_inquiries", "")) == "High":
        signals.append({
            "signal": "높은 고객 문의 수준",
            "segment": "서비스 회복 필요형",
            "primary": "미해결 문의 확인 및 상담 우선 배정",
            "alternative": "지원 채널·FAQ 안내 후 해결 여부 확인",
            "timing": "점수 산출 후 다음 접촉 회차 전에 상담 이력을 검토",
            "kpi": "문의 해결시간 · 반복 문의율 · 실험군 유지율",
            "rationale": "문의 수준 High가 관측 이탈과 모델의 주요 분류 신호로 확인됨",
        })
    if float(record.get("num_subscription_pauses", 0)) >= thresholds.high_subscription_pauses:
        signals.append({
            "signal": "상위 구간의 구독 일시정지 횟수",
            "segment": "구독 휴지형",
            "primary": "복귀 장애 요인 확인 및 구독 옵션 안내",
            "alternative": "일시정지·해지 흐름 사용 지원",
            "timing": "다음 정기 접촉 회차에서 복귀 의사와 장애 요인을 확인",
            "kpi": "재활성화율 · 재일시정지율 · 실험군 유지율",
            "rationale": "구독 일시정지 횟수가 학습 데이터 상위 구간에 해당함",
        })
    if float(record.get("weekly_hours", float("inf"))) <= thresholds.low_weekly_hours:
        signals.append({
            "signal": "하위 구간의 주간 청취시간",
            "segment": "이용 저하형",
            "primary": "개인화 재활성화 콘텐츠 안내 실험",
            "alternative": "선호 아티스트 기반 플레이리스트 제안 실험",
            "timing": "다음 캠페인 회차에 저비용 채널 후보로 검토",
            "kpi": "재방문율 · 주간 청취시간 변화 · 실험군 유지율",
            "rationale": "주간 청취시간이 학습 데이터 하위 25% 구간에 해당함",
        })
    if float(record.get("song_skip_rate", 0)) >= thresholds.high_skip_rate:
        signals.append({
            "signal": "상위 구간의 스킵률",
            "segment": "콘텐츠 탐색 피로형",
            "primary": "추천 만족도 피드백 요청 실험",
            "alternative": "다양성 중심 탐색 플레이리스트 제안 실험",
            "timing": "다음 추천 경험 접점에서 실험 후보로 검토",
            "kpi": "스킵률 변화 · 추천 클릭률 · 세션당 재생곡 수",
            "rationale": "스킵률이 학습 데이터 상위 25% 구간에 해당함",
        })
    if str(record.get("subscription_type", "")) == "Free":
        signals.append({
            "signal": "Free 구독 유형",
            "segment": "Free 혜택 탐색형",
            "primary": "유료 혜택 인지·체험 제안 실험",
            "alternative": "현재 이용 기능·혜택 가이드 제공",
            "timing": "다음 전환 캠페인 회차의 실험 후보로 검토",
            "kpi": "체험 참여율 · 유료 전환율 · 실험군 유지율",
            "rationale": "Free 고객군에서 높은 관측 이탈률 차이가 확인됨",
        })
    return signals


def recommend_retention_strategy(
    record: Mapping[str, object],
    probability: float,
    thresholds: StrategyThresholds,
    low_risk_threshold: float,
    high_risk_threshold: float,
) -> dict[str, object]:
    """Return a human-reviewable segment and two action hypotheses."""

    probability = float(probability)
    if not 0 <= probability <= 1:
        raise ValueError("probability는 0과 1 사이여야 합니다.")
    if low_risk_threshold >= high_risk_threshold:
        raise ValueError("low_risk_threshold는 high_risk_threshold보다 작아야 합니다.")

    signals = _signal_catalog(record, thresholds)
    signal_count = len(signals)
    if probability >= high_risk_threshold:
        campaign_tier = "집중 관리"
    elif probability >= low_risk_threshold:
        campaign_tier = "자동화 검토"
    else:
        campaign_tier = "관찰 유지"

    if not signals:
        result = {
            "strategy_segment": "일반 모니터링형",
            "primary_signal": "뚜렷한 행동 가능 신호 없음",
            "primary_action": "현재 캠페인 제외·관찰 유지",
            "alternative_action": "다음 정기 점수 갱신 시 재평가",
            "action_timing": "다음 정기 점수 갱신 시",
            "validation_kpi": "위험 점수 이동 · 관측 이탈률",
            "action_rationale": "선택한 행동 가능 변수에서 사전 정의된 신호가 확인되지 않음",
        }
    else:
        lead = signals[0]
        result = {
            "strategy_segment": lead["segment"],
            "primary_signal": lead["signal"],
            "primary_action": lead["primary"],
            "alternative_action": lead["alternative"],
            "action_timing": lead["timing"],
            "validation_kpi": lead["kpi"],
            "action_rationale": lead["rationale"],
        }
        if signal_count >= 2 and campaign_tier == "집중 관리":
            secondary_action = signals[1]["primary"] if signal_count >= 2 else lead["alternative"]
            result.update({
                "strategy_segment": "복합 고위험형",
                "primary_action": lead["primary"],
                "alternative_action": secondary_action,
                "action_timing": "점수 산출 후 다음 접촉 회차 전에 담당자 검토",
                "validation_kpi": f"{lead['kpi']} · 검토 완료율",
                "action_rationale": f"행동 가능 위험 신호 {signal_count}개 중 '{lead['signal']}'을 1차 검토 신호로 적용",
            })
        elif campaign_tier == "관찰 유지":
            result.update({
                "primary_action": "현재 캠페인 제외·관찰 유지",
                "alternative_action": lead["primary"],
                "action_timing": "다음 정기 점수 갱신 시 우선순위 재평가",
                "validation_kpi": "위험 점수 이동 · 관측 이탈률",
                "action_rationale": "행동 신호는 있으나 현재 위험 점수가 집중·자동화 검토 구간보다 낮음",
            })

    return {
        "campaign_tier": campaign_tier,
        "risk_signal_count": signal_count,
        "observed_signals": " · ".join(signal["signal"] for signal in signals) or "없음",
        **result,
        "evidence_level": EVIDENCE_LEVEL,
        "human_review_required": True,
        "excluded_from_action_basis": "age · location · customer_id",
    }


def build_strategy_queue(
    customers: pd.DataFrame,
    predictions: pd.DataFrame,
    thresholds: StrategyThresholds,
    low_risk_threshold: float,
    high_risk_threshold: float,
) -> pd.DataFrame:
    """Attach transparent strategy hypotheses to a label-free scored queue."""

    required_customer_columns = [
        "customer_id", "subscription_type", "customer_service_inquiries",
        "num_subscription_pauses", "weekly_hours", "song_skip_rate",
    ]
    missing_customer = sorted(set(required_customer_columns) - set(customers.columns))
    if missing_customer:
        raise ValueError(f"전략 큐 생성에 필요한 고객 컬럼이 없습니다: {missing_customer}")
    if not {"customer_id", "churn_probability"}.issubset(predictions.columns):
        raise ValueError("예측 결과에는 customer_id와 churn_probability가 필요합니다.")

    base = predictions[["customer_id", "churn_probability"]].merge(
        customers[required_customer_columns], on="customer_id", how="inner", validate="one_to_one"
    )
    strategies = [
        recommend_retention_strategy(
            row,
            probability=float(row["churn_probability"]),
            thresholds=thresholds,
            low_risk_threshold=low_risk_threshold,
            high_risk_threshold=high_risk_threshold,
        )
        for row in base.to_dict(orient="records")
    ]
    strategy_frame = pd.DataFrame(strategies, index=base.index)
    return pd.concat([base[["customer_id", "churn_probability"]], strategy_frame], axis=1).sort_values(
        "churn_probability", ascending=False
    ).reset_index(drop=True)
