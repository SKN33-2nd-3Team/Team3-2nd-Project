import numpy as np
import pandas as pd

from app.streamlit_app import _campaign_net_curve, _prepare_campaign_plan


def _planning_inputs():
    queue = pd.DataFrame([
        {"customer_id": 1, "churn_probability": 0.90, "risk_signal_count": 3, "campaign_tier": "집중 관리", "primary_action": "문의 해결", "strategy_segment": "복합 고위험형"},
        {"customer_id": 2, "churn_probability": 0.85, "risk_signal_count": 2, "campaign_tier": "집중 관리", "primary_action": "문의 해결", "strategy_segment": "복합 고위험형"},
        {"customer_id": 3, "churn_probability": 0.80, "risk_signal_count": 2, "campaign_tier": "집중 관리", "primary_action": "복귀 안내", "strategy_segment": "복합 고위험형"},
        {"customer_id": 4, "churn_probability": 0.75, "risk_signal_count": 2, "campaign_tier": "자동화 검토", "primary_action": "콘텐츠 안내", "strategy_segment": "이용 저하형"},
        {"customer_id": 5, "churn_probability": 0.70, "risk_signal_count": 1, "campaign_tier": "자동화 검토", "primary_action": "혜택 안내", "strategy_segment": "Free 혜택 탐색형"},
        {"customer_id": 6, "churn_probability": 0.30, "risk_signal_count": 3, "campaign_tier": "자동화 검토", "primary_action": "추천 진단", "strategy_segment": "콘텐츠 탐색 피로형"},
    ])
    plans = pd.DataFrame({
        "customer_id": [1, 2, 3, 4, 5, 6],
        "subscription_type": ["Premium", "Premium", "Family", "Student", "Free", "Family"],
    })
    economics = pd.DataFrame([
        {"plan": "Free", "cost": 100, "customer_value": 0},
        {"plan": "Student", "cost": 200, "customer_value": 72_000},
        {"plan": "Premium", "cost": 500, "customer_value": 130_800},
        {"plan": "Family", "cost": 4_500, "customer_value": 196_200},
    ])
    return queue, plans, economics


def test_campaign_plan_applies_scenario_and_action_signal_gate():
    queue, plans, economics = _planning_inputs()
    result = _prepare_campaign_plan(queue, plans, economics, 0.35, 2, 0.10, "risk_priority")
    assert result["customer_id"].tolist() == [1, 2, 3, 4]
    assert result["probability"].min() >= 0.35
    assert result["risk_signal_count"].min() >= 2


def test_action_portfolio_spreads_first_round_across_action_routes():
    queue, plans, economics = _planning_inputs()
    result = _prepare_campaign_plan(queue, plans, economics, 0.35, 2, 0.10, "action_portfolio")
    first_round = result.loc[result["_allocation_round"].eq(0)]
    assert len(first_round) == 3
    assert first_round["primary_action"].nunique() == 3
    assert result.iloc[-1]["customer_id"] == 2


def test_campaign_net_curve_handles_an_empty_plan():
    empty = pd.DataFrame(columns=["probability", "customer_value", "cost"])

    x, y = _campaign_net_curve(empty, success_rate=0.10)

    assert isinstance(x, np.ndarray)
    assert isinstance(y, np.ndarray)
    assert x.size == 0
    assert y.size == 0
