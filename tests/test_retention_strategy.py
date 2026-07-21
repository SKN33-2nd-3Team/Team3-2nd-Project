import pandas as pd

from src.retention_strategy import (
    StrategyThresholds,
    build_strategy_queue,
    recommend_retention_strategy,
)


THRESHOLDS = StrategyThresholds(
    low_weekly_hours=10.0,
    high_skip_rate=0.75,
    high_subscription_pauses=3.0,
)


def test_sensitive_profile_fields_do_not_change_action_assignment():
    base = {
        "customer_service_inquiries": "High",
        "num_subscription_pauses": 4,
        "weekly_hours": 5.0,
        "song_skip_rate": 0.9,
        "subscription_type": "Free",
        "age": 20,
        "location": "A",
    }
    changed_sensitive = {**base, "age": 79, "location": "B"}
    first = recommend_retention_strategy(base, 0.9, THRESHOLDS, 0.29, 0.74)
    second = recommend_retention_strategy(changed_sensitive, 0.9, THRESHOLDS, 0.29, 0.74)
    assert first == second
    assert first["strategy_segment"] == "복합 고위험형"
    assert first["risk_signal_count"] == 5


def test_low_risk_customer_is_kept_in_observation_lane():
    record = {
        "customer_service_inquiries": "Low",
        "num_subscription_pauses": 0,
        "weekly_hours": 20.0,
        "song_skip_rate": 0.2,
        "subscription_type": "Premium",
    }
    strategy = recommend_retention_strategy(record, 0.1, THRESHOLDS, 0.29, 0.74)
    assert strategy["campaign_tier"] == "관찰 유지"
    assert strategy["primary_action"] == "현재 캠페인 제외·관찰 유지"
    assert strategy["human_review_required"] is True


def test_batch_queue_contains_decision_fields_and_preserves_rows():
    customers = pd.DataFrame([
        {"customer_id": 1, "subscription_type": "Free", "customer_service_inquiries": "High", "num_subscription_pauses": 4, "weekly_hours": 5.0, "song_skip_rate": 0.9},
        {"customer_id": 2, "subscription_type": "Premium", "customer_service_inquiries": "Low", "num_subscription_pauses": 0, "weekly_hours": 20.0, "song_skip_rate": 0.2},
    ])
    predictions = pd.DataFrame({"customer_id": [1, 2], "churn_probability": [0.9, 0.1]})
    queue = build_strategy_queue(customers, predictions, THRESHOLDS, 0.29, 0.74)
    assert len(queue) == 2
    assert {"strategy_segment", "primary_action", "validation_kpi", "evidence_level"}.issubset(queue.columns)
    assert queue.iloc[0]["customer_id"] == 1
