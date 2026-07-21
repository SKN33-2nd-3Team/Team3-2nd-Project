# Presentation Evidence Pack v3

## Full fair comparison outcome

- Common Feature set: `log_numeric`
- Same validation: saved 5-fold stratified assignments
- Seven tuned models: 20/20/24/24/30/30/24 randomized trials; Random Forest and GradientBoosting used user-approved lightweight parameter ranges while retaining their trial/fold counts.
- Actual top three: CatBoost, XGBoost, LightGBM.
- Selected local model: CatBoost; it is now the local Streamlit inference artifact. The replaced GradientBoosting artifacts are preserved under `artifacts/model/archive/20260722_pre_full_fair_catboost/`.

## Evidence map

| Claim | Source |
| --- | --- |
| Feature process and deltas | `artifacts/preprocessing_experiment_results.csv` |
| Eight-model fair baseline | `artifacts/model_comparison_fair.csv` |
| Search and fine-tuning | `artifacts/random_search_summary.csv`, `artifacts/top3_fine_tuning_summary.csv` |
| Stability and uncertainty | `artifacts/seed_stability_summary.csv`, `artifacts/bootstrap_confidence_intervals.csv` |
| Operating choices | `artifacts/threshold_operating_scenarios_v2.csv`, `artifacts/equal_contact_comparison.csv`, `artifacts/equal_recall_comparison.csv` |
| Ranking/calibration/segments | `artifacts/topk_lift_v2.csv`, `artifacts/risk_decile_v2.csv`, `artifacts/calibration_summary.csv`, `artifacts/segment_error_analysis.csv` |

No external holdout, campaign uplift, causal effect, or prediction horizon is claimed. The app and local operational tables explicitly label their metrics as five-fold OOF evidence.
