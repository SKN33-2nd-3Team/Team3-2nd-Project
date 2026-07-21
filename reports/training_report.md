# 최종 모델 학습·운영 결과

## 최종 선정

- 모델: **CatBoost**
- Run: `20260721_full_fair_v1`
- 공통 Feature: `log_numeric`
- 선정 기준: 실제 상위 3개 모델 중 Fine-tuned 5-Fold CV PR-AUC 최고
- 재학습 없는 최종 통합: 저장된 후보 Pipeline과 OOF 예측 재사용

| 근거 | CatBoost | XGBoost | LightGBM |
| --- | ---: | ---: | ---: |
| Fine-tuned CV PR-AUC | **0.947913** | 0.947793 | 0.947806 |
| OOF PR-AUC | **0.947897** | 0.947765 | 0.947790 |
| 5개 Seed 평균 PR-AUC | **0.947880** | 0.947726 | 0.947721 |
| 5개 Seed 표준편차 | 0.000061 | 0.000061 | 0.000095 |

세 모델의 차이는 작습니다. CatBoost는 PR-AUC와 Seed 평균을 중심으로 기술 후보로 선정했으며, 특정 Threshold에서 Recall 또는 FN을 더 중시하면 LightGBM이나 다른 Threshold가 더 적합할 수 있습니다.

## 기준 Threshold 0.50의 OOF 성능

| PR-AUC | ROC-AUC | F1 | Recall | Precision | FN | FP |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.947897 | 0.941959 | 0.849118 | 0.836382 | 0.862247 | 10,500 | 8,575 |

## 운영 시나리오

| 시나리오 | Threshold | 대상 고객 | 대상 비율 | Recall | Precision | F1 | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 재현율 우선 | 0.29 | 77,700 | 62.16% | 95.18% | 78.61% | 0.8611 | 3,091 | 16,617 |
| 균형형 F1 | 0.35 | 76,143 | 60.91% | 94.44% | 79.59% | 0.8638 | 3,569 | 15,538 |
| 정밀도 우선 | 0.74 | 48,463 | 38.77% | 71.80% | 95.07% | 0.8181 | 18,099 | 2,388 |

앱 기본값은 균형형 F1입니다. 이는 사업 확정값이 아니며 접촉 인원과 오류 비용에 따라 사용자가 바꿀 수 있습니다.

## 모델 산출물

- 현재 모델: `artifacts/model/music_churn_pipeline.joblib`
- 현재 메타데이터: `artifacts/model/metadata.json`
- 무라벨 테스트 점수: `artifacts/test_predictions.csv`
- Feature 중요도: `artifacts/feature_importance.csv`
- 운영 시나리오: `artifacts/operating_scenarios_oof_catboost.csv`
- Top-K / Lift: `artifacts/topk_lift_oof_catboost.csv`
- 위험 Decile: `artifacts/risk_decile_oof_catboost.csv`

## 검증 경계

- 독립된 외부 라벨 Holdout이 없습니다.
- OOF 결과를 본 뒤 외부 Holdout에 맞춰 모델·Feature·파라미터를 재선택한 과정은 없습니다.
- 제공 `test.csv`는 라벨이 없어 성능 평가가 아닌 추론에만 사용했습니다.
- 예측 기간, 캠페인 Uplift, 인과효과, ROI는 검증되지 않았습니다.
- 실제 CRM 실행과 최종 사업·법적 승인은 저장소 밖의 권한과 결과 데이터가 필요합니다.
