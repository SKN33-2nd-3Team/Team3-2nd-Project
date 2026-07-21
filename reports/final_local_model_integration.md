# Final Local Model Integration

## Active local pipeline

The local Streamlit inference artifact is now the saved **CatBoost** pipeline from `20260721_full_fair_v1`. It uses the selected `log_numeric` feature variant and scores the unlabeled `data/test.csv` without refitting.

| Decision | Evidence |
| --- | --- |
| Selected model | CatBoost |
| Primary selection rule | Highest fine-tuned five-fold CV PR-AUC among CatBoost, XGBoost, and LightGBM |
| Fine-tuned CV PR-AUC | 0.947913 |
| Five-seed mean PR-AUC | 0.947880 |
| Saved OOF PR-AUC / ROC-AUC | 0.947897 / 0.941959 |
| Reload check | Passed |

## Replaced local artifacts

- Active model: `artifacts/model/music_churn_pipeline.joblib`
- Active metadata: `artifacts/model/metadata.json`
- Unlabeled test scores: `artifacts/test_predictions.csv`
- Model interpretation: `artifacts/feature_importance.csv`
- OOF operating evidence: `artifacts/operating_scenarios_oof_catboost.csv`, `artifacts/topk_lift_oof_catboost.csv`, and `artifacts/risk_decile_oof_catboost.csv`

The prior GradientBoosting model, metadata, test scores, and feature-importance file were copied to `artifacts/model/archive/20260722_pre_full_fair_catboost/` before replacement.

## Insight integration

The final pipeline uses deterministic, prediction-time-available features only: raw behavior and subscription signals, four smoothed ratios, and log transforms of skewed behavioral counts. The common feature choice is recorded in `artifacts/insight_preprocessing_register.csv`; it is a predictive design choice, not a causal explanation of churn.

## Operating options (five-fold OOF only)

| Scenario | Threshold | Target rate | Recall | Precision | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall-first | 0.29 | 62.16% | 95.18% | 78.61% | 3,091 | 16,617 |
| Balanced F1 (app default) | 0.35 | 60.91% | 94.44% | 79.59% | 3,569 | 15,538 |
| Precision-first | 0.74 | 38.77% | 71.80% | 95.07% | 18,099 | 2,388 |

## Boundaries

- No untouched external/final holdout was available; all operational quality values above are saved five-fold OOF evidence.
- The model does not establish a future prediction horizon, causal drivers, campaign uplift, ROI, or intervention effectiveness.
- CRM sending, discounts, customer contact, and final business/legal approval remain external decisions.
