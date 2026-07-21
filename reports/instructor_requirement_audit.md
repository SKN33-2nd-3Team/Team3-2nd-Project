# Instructor Requirement Audit

> **과거 P0 감사 기록:** 현재 최종 감사 결과는 `instructor_requirement_audit_v2.md`를 사용합니다. 아래 LightGBM 결론은 최종 선정 이전 기록입니다.

## Audit scope

This audit inspected the saved candidate metadata, P0 comparison, final evidence pack, every run CSV, manifest, Project Register, and the non-training artifact-contract test. No model fitting or threshold reselection was performed.

| 강사 요구사항 | 관련 산출물 | 실제 근거 | 상태 | 부족한 내용 | 보완 필요 여부 |
| --- | --- | --- | --- | --- | --- |
| 성능 개선 과정·논리 | `feature_experiment_screening.csv`, `performance_progression.csv` | Raw logistic → deterministic features와 모델군 비교가 저장됨 | 부분 충족 | 6개 feature 실험 중 실제 수치가 있는 것은 FE-00/01뿐 | 예 — v2 표·결론으로 보완, 추가 학습은 보류 |
| 조치별 성능 변화 | `performance_progression.csv`, `01_performance_progression.png` | ROC-AUC·PR-AUC·F1·Recall·Precision, 직전/누적 Δ를 기록 | 충족 (v2) | 외부 3개 모델의 FN/FP 행수는 저장되지 않음 | 아니오 — 검증 수치와 범위를 명시 |
| 8개 모델 비교 | `model_comparison_p0.csv`, `model_selection_decision_matrix.csv`, `02_eight_model_comparison.png` | 8개 필수 모델의 Validation PR-AUC 비교 | 부분 충족 | Raw logistic만 의도적으로 비엔지니어드 feature라 완전 동일 feature 조건은 아님 | 예 — 다음 run에서 전 모델 동일 feature 재비교 |
| 모델별 튜닝 전후 | candidate metadata, decision matrix | LightGBM default와 3-trial/3-fold 탐색 후보만 확인 | 부분 충족 | 다른 7개 모델의 tuning before/after는 없음 | 예 — 필요 시 validation-only 실험으로 수행 |
| Threshold 운영 시나리오 | `threshold_operating_scenarios.csv`, `03_threshold_operating_scenarios.png` | Validation에서 threshold 선택 후 internal holdout 1회 평가 | 충족 (v2) | 실제 캠페인 비용·용량은 없음 | 예 — 사업자가 시나리오 선택 |
| Top-K / Lift / Risk Decile | `topk_lift.csv`, `risk_decile.csv`, `04`~`06` 그래프 | final LightGBM internal holdout 결과 재사용 | 충족 (v2) | 외부 실제 캠페인 효과는 없음 | 아니오 — Uplift와 구분 표기 |
| LightGBM 최종 선정 근거 | decision matrix, `07_lightgbm_selection_evidence.png` | 8개 중 Validation PR-AUC 최고(0.9457), 3-fold CV PR-AUC 0.9477, holdout PR-AUC 0.9486 | 부분 충족 | Recall/FN trade-off의 사업 우선순위는 미승인 | 예 — 운영 시나리오를 사람이 선택 |
| 최고 단일 점수와 최종 선정의 차이 | decision matrix | 동일 모델 LightGBM이 최고 Validation PR-AUC이자 최종 후보 | 충족 | 기술 점수 외 사업 승인 없음 | 예 — 배포 전 승인 |
| Validation–Holdout 일반화 | candidate metadata, progression | Validation 0.9457 → holdout 0.9486 PR-AUC | 충족 | 시간 기반 미래 검증·prediction horizon 없음 | 예 — 새 데이터 수집 후 검증 |
| 발표용 그래프와 원본 CSV | `figures/presentation_v2/`, `artifacts/*.csv` | 7 PNG와 대응 CSV를 생성 | 충족 (v2) | 원본 run에는 그래프 없음 | 아니오 |

## Missing evidence and retraining decision

No additional model training is needed to create the v2 evidence pack. Bootstrap confidence intervals require saved row-level candidate predictions (not present); seed stability and segment error analysis require new scoring or refits. These are deferred so the original holdout remains untouched.

## Guardrails

- **VERIFIED:** listed validation/internal-holdout metrics and reload check.
- **UNVERIFIED:** prediction horizon, campaign uplift, causal effect, and production impact.
- The historical GradientBoosting cost ratio is not used in the LightGBM selection.
