# PlaylistPro Presentation Storyline

> **과거 발표 흐름:** 현재 발표는 최종 CatBoost 기준의 `presentation_evidence_pack_v3.md`를 사용합니다. 아래 내용은 통합 전 기록입니다.

## One-line story

Use observed customer behavior to rank churn-risk review candidates, then choose contact coverage only after the business owner sets capacity and false-negative/false-positive trade-offs.

| Page | Audience question | Evidence-backed conclusion | Speaker transition |
| --- | --- | --- | --- |
| Project Summary | What problem are we solving, and what is actually ready? | The dashboard scores the dataset's `churned` label using the saved Gradient Boosting pipeline. LightGBM is a separate validated candidate awaiting review. | “Before selecting a model, we checked which observable customer patterns are associated with the label.” |
| Customer Insights | What patterns are worth investigating? | Support contacts, subscription type, payment plan, and listening-volume groups show different observed label rates. They form hypotheses, not causal explanations. | “Those hypotheses informed feature and model experiments rather than a claim that any one feature causes churn.” |
| Improvement Journey | How did the approach improve? | Saved evidence tracks raw logistic baseline, feature experiment, and model-screen metric movement; each row is marked adopted or excluded. | “After establishing ranking quality, we compare operational consequences instead of selecting on one score alone.” |
| Model & Operations | Why this model and threshold behavior? | PR-AUC, Recall, Precision, FN/FP, Top-K capture, Lift, and decile behavior expose the coverage-versus-contact trade-off. A threshold is a business decision, not an automatic recommendation. | “The final workflow turns that evidence into an analyst review queue.” |
| Customer Prioritization | What does an operator do next? | Individual input calls the saved pipeline's real `predict_proba`; batch prioritization ranks label-free `test.csv` customers without claiming their outcomes are known. | “This is decision support. A campaign test is required to measure real retention impact.” |

## Suggested 5-minute flow

1. Open **Project Summary** and state the target, dataset-label boundary, and demo-versus-candidate model status.
2. Show two or three cards in **Customer Insights** and explicitly call them associations.
3. Open **Improvement Journey**; explain one adopted and one excluded experiment with PR-AUC movement.
4. Open **Model & Operations**; compare the three threshold scenarios, then show Top-K capture and deciles.
5. Finish in **Customer Prioritization** with one individual score and the batch tab.

## Non-negotiable wording

- Say “observed association,” not “cause of churn.”
- Say “internal validation/holdout evidence,” not “proven business impact.”
- Say “technical candidate awaiting review,” not “deployed LightGBM model.”
- Say “contact-priority scenario,” not “the one correct threshold.”
