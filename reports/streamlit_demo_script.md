# Streamlit Demo Script

> **과거 데모 기록:** 현재 앱은 최종 CatBoost Pipeline을 사용합니다. 발표에는 `presentation_evidence_pack_v3.md`를 사용합니다.

## Demonstration order

1. **Project Summary**
   - Say: “This project ranks customers for review against the dataset's `churned` label. The supplied files do not verify a 30-day horizon.”
   - Open **Model state and candidate evidence**.
   - Say: “CatBoost powers the live demo scorer through the saved final pipeline; the score is a review aid, not an automatic campaign decision.”

2. **Customer Insights**
   - Point to the four spread cards.
   - Say: “These are observed differences in label rate. They become hypotheses for feature design or service investigation, not causal campaign rules.”
   - Use the explorer for one variable only if asked.

3. **Improvement Journey**
   - Say: “We keep the raw logistic baseline visible, show the metric movement after changes, and retain at least one excluded experiment.”
   - Point to the saved-candidate table.
   - Say: “The broader fair-comparison run is complete under the same OOF conditions; CatBoost is a conditional final candidate because the top scores are close.”

4. **Model & Operations**
   - Select each operating scenario in the sidebar.
   - Say: “Lower thresholds cover more labeled churners but create more false positives and contacts. The business owner must choose capacity and costs.”
   - Point to Top-K capture, Lift, risk decile, and the confusion matrix.

5. **Customer Prioritization**
   - Open **Customer scoring**, change one input, and submit.
   - Say: “The score is calculated by the saved pipeline at this moment; it is not a fixed JSON example.”
   - Open **Batch prioritization**.
   - Say: “`test.csv` has no label, so this is a review queue rather than measured churn performance.”

## Likely questions and answers

| Question | Answer | Evidence |
| --- | --- | --- |
| Why not call a feature a churn cause? | The data shows association, not an intervention or randomized experiment. | Customer Insights wording; feature-importance limitation. |
| Why not deploy LightGBM now? | LightGBM is competitive, but CatBoost has a small OOF PR-AUC and seed-stability edge; overlapping uncertainty means this is a conditional technical choice, not an absolute win. | Candidate metadata; full-run manifest. |
| Which threshold is best? | There is no universal best threshold without capacity and FN/FP cost. The page shows consequences of three saved OOF scenarios. | `artifacts/operating_scenarios_oof_catboost.csv`. |
| Does a high score prove a campaign will retain the customer? | No. Uplift and ROI require a controlled campaign experiment and future outcomes. | Dashboard limitation text. |
| Why use label-free test data in the batch view? | It supports ranking and operational review only; it cannot measure performance without labels. | `data/test.csv` schema and batch-view warning. |

## Demo recovery

- If an evidence CSV is unavailable, show the dashboard warning rather than entering a manual metric.
- If a page fails to load, return to **Project Summary** and use the saved presentation figures under `figures/presentation_v3/`.
- Do not rerun training during a presentation. The app loads saved artifacts only.
