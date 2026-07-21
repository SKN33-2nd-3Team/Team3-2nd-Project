# Streamlit Presentation Improvement Audit

> **과거 통합 전 감사 기록:** 현재 앱은 최종 CatBoost 모델과 OOF 운영 표로 교체됐습니다. 아래 내용은 변경 이력을 보존한 기록입니다.

## Audit scope

This audit reviewed the Streamlit page flow, the saved inference artifact, the isolated LightGBM candidate metadata, saved P0 evidence CSVs, and the in-progress full-fair-comparison manifest. No model fitting, artifact replacement, or threshold reselection was performed.

| Problem | Presentation impact | Improvement applied | Priority | Evidence / boundary |
| --- | --- | --- | --- | --- |
| The app inference pipeline and the technical candidate could be mistaken for the same model. | A reviewer could believe an unapproved candidate is deployed. | The dashboard now labels the Gradient Boosting demo pipeline and the LightGBM technical candidate separately. | P0 | `artifacts/model/metadata.json`; `models/candidates/20260721_submission_bounded_v1/metadata.json` |
| The experiment story was buried inside general model content. | The baseline → change → metric → adopt/exclude reasoning was hard to present. | Added **Improvement Journey** using `performance_progression.csv` and the saved selection matrix. | P0 | Historical saved evidence only; no retraining in the app. |
| Data Explorer exposed many fields before the main business signals. | The audience had to infer the key insights themselves. | Added four association-spread cards before the drill-down explorer. | P1 | Associations are not causal claims. |
| Model performance and operations were mixed without an explicit artifact boundary. | Historical operational metrics might be attributed to LightGBM. | Added a boundary note and candidate-status expander to **Model & Operations**. | P0 | Threshold, Top-K, Lift, and decile charts remain tied to the saved Gradient Boosting demo artifacts. |
| Customer scoring and batch targeting were separate navigation destinations. | The prioritization workflow was fragmented. | Kept the earlier integrated **Customer Prioritization** tabs. | P1 | Individual scoring continues to use the saved pipeline's `predict_proba`. |
| Campaign simulation could look like observed uplift. | Planning assumptions could be presented as business outcomes. | Kept it under **Reference analysis** and preserved the assumption-only warning. | P0 | Campaign uplift, ROI, and causal effect remain unverified. |

## Final page flow

1. **Project Summary** — business question, data scope, model state, and headline evidence.
2. **Customer Insights** — observed customer patterns and drill-down exploration.
3. **Improvement Journey** — baseline, feature/model experiments, metric movement, adopted and excluded choices.
4. **Model & Operations** — saved model comparison, threshold scenarios, Top-K/Lift, risk deciles, and limits.
5. **Customer Prioritization** — actual individual scoring and label-free batch prioritization.

## Known evidence limits

- The LightGBM artifact is `MODEL_CANDIDATE_SAVED_AWAITING_REVIEW`; it does not replace the dashboard inference pipeline.
- The full fair-comparison run is still marked `RUNNING`; incomplete tuning, seed, bootstrap, calibration, and promotion results are not displayed as completed evidence.
- The supplied data does not document a future prediction horizon. “30-day churn” is unverified.
- Ranking metrics and internal holdout results do not demonstrate campaign uplift, causal effect, or ROI.

## Verification performed

- `python -m compileall -q app/streamlit_app.py`
- Streamlit `AppTest` render check for the five primary navigation pages.
- Manual local light-theme check for sidebar control contrast.
