# Model Run Report

- Run time (UTC): `2026-07-21T13:41:15.719497+00:00`
- Python: `3.13.14`
- Final selected model: `gradient_boosting`
- Threshold rule: `lowest Validation expected cost with FN:FP=3:1, then PR-AUC, then Recall; final model selected`
- Validation operating threshold: `0.30`

## Data and split

- Local train: `125,000` rows with target; local test: `75,000` rows without target
- Internal split: `75,000` / `25,000` / `25,000`
- Split: stratified random split with random_state=42; switch to chronological split if event timestamps become available
- `customer_id` excluded from model input; raw `signup_date` excluded and date-derived/ratio features are generated inside the Pipeline

## Comparison

| Model | Val PR-AUC | Val Recall@operating | Val Precision@operating | Val cost/customer | Test PR-AUC | Test Recall@operating | Test Precision@operating | Test cost/customer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gradient_boosting | 0.9446 | 0.9621 | 0.7662 | 0.2090 | 0.9473 | 0.9617 | 0.7679 | 0.2082 |
| random_forest | 0.9376 | 0.9455 | 0.7688 | 0.2299 | 0.9409 | 0.9440 | 0.7719 | 0.2295 |
| decision_tree | 0.9259 | 0.9496 | 0.7276 | 0.2601 | 0.9314 | 0.9522 | 0.7337 | 0.2510 |
| extra_trees | 0.9119 | 0.9469 | 0.7148 | 0.2758 | 0.9162 | 0.9475 | 0.7172 | 0.2727 |
| logistic_engineered | 0.8968 | 0.9423 | 0.6962 | 0.3000 | 0.9013 | 0.9440 | 0.6987 | 0.2953 |
| logistic_raw | 0.8968 | 0.9423 | 0.6960 | 0.3002 | 0.9014 | 0.9437 | 0.6987 | 0.2956 |
| dummy_prior | 0.5134 | 1.0000 | 0.5134 | 0.4866 | 0.5134 | 1.0000 | 0.5134 | 0.4866 |

## Operating scenarios (selected on Validation, evaluated on internal Test)

| Scenario | Threshold | Test target customers | Test precision | Test recall | Test TP | Test FN | Test FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 공격적 대응 | 0.30 | 16,073 | 0.7679 | 0.9617 | 12,343 | 492 | 3,730 |
| 균형 대응 | 0.41 | 14,856 | 0.8022 | 0.9286 | 11,918 | 917 | 2,938 |
| 정밀 대응 | 0.71 | 9,313 | 0.9553 | 0.6932 | 8,897 | 3,938 | 416 |

## Held-out targeting metrics

| Top-risk customers | Target customers | Precision | Churner capture | Lift |
|---:|---:|---:|---:|---:|
| 5% | 1,250 | 1.0000 | 0.0974 | 1.9478 |
| 10% | 2,500 | 1.0000 | 0.1948 | 1.9478 |
| 20% | 5,000 | 0.9998 | 0.3895 | 1.9474 |
| 30% | 7,500 | 0.9713 | 0.5676 | 1.8920 |
| 50% | 12,500 | 0.8632 | 0.8407 | 1.6813 |

## Probability calibration diagnostic

- Mean absolute decile gap: `0.0627`; maximum decile gap: `0.1047`.
- Campaign Simulator sums model probabilities for an assumption-based planning estimate. It is not an observed campaign outcome or an uplift estimate.

## Interpretation guardrails

- Gradient Boosting is the final selected model based on the documented Validation selection rule.
- Metrics are for the dataset-provided `churned` label. They do not validate a future 30-day churn horizon.
- The operating threshold uses the team assumption FN:FP=3:1; a high Recall operating point can still increase false positives, so campaign capacity and the cost ratio require business review.
- Feature importance and EDA relationships are associations, not causal churn drivers; see `artifacts/feature_importance.csv`.
