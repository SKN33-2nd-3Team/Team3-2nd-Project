# Model Run Report

- Run time (UTC): `2026-07-21T08:47:51.867678+00:00`
- Python: `3.12.13`
- Recommended candidate (provisional): `gradient_boosting`
- Threshold rule: `lowest Validation expected cost with FN:FP=3:1, then PR-AUC, then Recall; provisional and awaiting user approval`
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

## Interpretation guardrails

- The recommended candidate is not final approval; it is a technical provisional recommendation.
- Metrics are for the dataset-provided `churned` label. They do not validate a future 30-day churn horizon.
- The provisional threshold uses FN:FP=3:1; a high Recall operating point can still increase false positives, so the threshold and campaign capacity require business approval.
- Feature importance and EDA relationships are associations, not causal churn drivers; see `artifacts/feature_importance.csv`.
