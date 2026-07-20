# Model Run Report

- Run time (UTC): `2026-07-19T14:19:19.171693+00:00`
- Python: `3.13.13`
- Recommended candidate (provisional): `gradient_boosting`
- Threshold rule: `lowest Validation expected cost with FN:FP=3:1, then PR-AUC, then Recall; provisional and awaiting user approval`
- Validation operating threshold: `0.37`

## Data and split

- Local train: `41,000` rows with target; local test: `9,999` rows without target
- Internal split: `24,600` / `8,200` / `8,200`
- Split: stratified random split with random_state=42; switch to chronological split if event timestamps become available
- `customer_id` excluded from model input; raw `signup_date` excluded and date-derived/ratio features are generated inside the Pipeline

## Comparison

| Model | Val PR-AUC | Val Recall@operating | Val Precision@operating | Val cost/customer | Test PR-AUC | Test Recall@operating | Test Precision@operating | Test cost/customer |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gradient_boosting | 0.3616 | 0.3817 | 0.4908 | 0.4804 | 0.3631 | 0.3819 | 0.4985 | 0.4774 |
| random_forest | 0.3608 | 0.4074 | 0.4465 | 0.4872 | 0.3670 | 0.4111 | 0.4436 | 0.4868 |
| decision_tree | 0.3530 | 0.3720 | 0.4728 | 0.4906 | 0.3642 | 0.3751 | 0.4870 | 0.4841 |
| extra_trees | 0.3501 | 0.4137 | 0.4261 | 0.4943 | 0.3562 | 0.4122 | 0.4199 | 0.4976 |
| logistic_engineered | 0.3409 | 0.4509 | 0.3806 | 0.5082 | 0.3463 | 0.4437 | 0.3740 | 0.5144 |
| logistic_raw | 0.3271 | 0.4651 | 0.3647 | 0.5154 | 0.3330 | 0.4597 | 0.3577 | 0.5218 |
| dummy_prior | 0.2134 | 0.0000 | 0.0000 | 0.6402 | 0.2133 | 0.0000 | 0.0000 | 0.6399 |

## Interpretation guardrails

- The recommended candidate is not final approval; it is a technical provisional recommendation.
- Metrics are for the dataset-provided `churned` label. They do not validate a future 30-day churn horizon.
- The provisional threshold uses FN:FP=3:1; a high Recall operating point can still increase false positives, so the threshold and campaign capacity require business approval.
- Feature importance and EDA relationships are associations, not causal churn drivers; see `artifacts/feature_importance.csv`.
