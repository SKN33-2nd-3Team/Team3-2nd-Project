# Presentation Evidence Pack v2

> **과거 P0 기록:** 현재 최종 모델과 발표 근거는 `presentation_evidence_pack_v3.md`의 CatBoost 결과를 사용합니다. 아래 LightGBM 내용은 최종 선정 이전 기록입니다.

## Recommended-for-review candidate

**LightGBM** is the technical candidate because it had the highest saved validation PR-AUC among the eight required models (0.9457), retained that score after the bounded 3-trial/3-fold search, and generalized to an internal-holdout PR-AUC of 0.9486. The top single-score model and selected model are the same.

## Operating decision, not a claim of business impact

Choose a threshold only after the campaign owner sets contact capacity and the false-negative/false-positive trade-off. The saved scenarios show the consequences: recall-first (0.30) targets 61.9% and achieves 95.1% recall; balanced-F1 (0.35) targets 61.0%, with 94.5% recall, 79.5% precision, FN 705 and FP 3,131; precision-first (0.77) targets 38.6% and achieves 95.5% precision.

## Presentation file map

| Slide claim | Graph | Source CSV |
| --- | --- | --- |
| Baseline to candidate progression | `01_performance_progression.png` | `artifacts/performance_progression.csv` |
| Eight-model screen | `02_eight_model_comparison.png` | `artifacts/model_selection_decision_matrix.csv` |
| Threshold choices | `03_threshold_operating_scenarios.png` | `artifacts/threshold_operating_scenarios.csv` |
| Contact capacity / capture | `04_topk_capture.png` | `artifacts/topk_lift.csv` |
| Targeting lift | `05_topk_lift.png` | `artifacts/topk_lift.csv` |
| Risk ranking and calibration | `06_risk_decile.png` | `artifacts/risk_decile.csv` |
| Selection rationale | `07_lightgbm_selection_evidence.png` | `artifacts/model_selection_decision_matrix.csv` |

## Limits

Metrics are internal validation/holdout evidence, not campaign uplift, causal impact, or a verified prediction horizon. Candidate status remains `MODEL_CANDIDATE_SAVED_AWAITING_REVIEW`; no current app model was replaced.
