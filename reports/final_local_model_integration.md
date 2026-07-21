# 최종 로컬 모델 통합 결과

## 현재 운영 Pipeline

`20260721_full_fair_v1`의 최종 **CatBoost** 후보를 로컬 Streamlit 추론 모델로 통합했습니다. `log_numeric` Feature를 사용하며 모델 재학습 없이 라벨 없는 `data/test.csv` 75,000명의 위험 점수를 다시 생성했습니다.

| 결정 항목 | 근거 |
| --- | --- |
| 최종 모델 | CatBoost |
| 1차 선정 기준 | 상위 3개 중 Fine-tuned 5-Fold CV PR-AUC 최고 |
| Fine-tuned CV PR-AUC | 0.947913 |
| 5개 Seed 평균 PR-AUC | 0.947880 |
| OOF PR-AUC / ROC-AUC | 0.947897 / 0.941959 |
| 새 프로세스 재로딩 | 통과 |
| 모델 무결성 | 메타데이터 SHA-256과 일치 |

## 통합 산출물

- 현재 모델: `artifacts/model/music_churn_pipeline.joblib`
- 현재 메타데이터: `artifacts/model/metadata.json`
- 테스트 고객 점수: `artifacts/test_predictions.csv`
- CatBoost Feature 중요도: `artifacts/feature_importance.csv`
- OOF 운영 근거: `artifacts/operating_scenarios_oof_catboost.csv`
- OOF Top-K/Lift: `artifacts/topk_lift_oof_catboost.csv`
- OOF 위험 Decile: `artifacts/risk_decile_oof_catboost.csv`

기존 Gradient Boosting 모델과 이전 CatBoost 통합본은 `artifacts/model/archive/` 아래에 서로 다른 폴더로 보존했습니다.

## 인사이트 통합

최종 Pipeline은 예측 시점에 계산 가능한 원천 행동·구독 신호, 네 가지 완화 비율, 치우친 행동 수치의 Log 변환을 사용합니다. 전처리 선택 근거는 `artifacts/insight_preprocessing_register.csv`에 있습니다. 이는 예측을 위한 Feature 설계이며 이탈 원인에 대한 인과 설명이 아닙니다.

## 안전성 보완

- 후보 메타데이터의 SHA-256과 실제 후보 파일을 승격 전에 비교합니다.
- 저장소 외부 경로의 모델은 거부합니다.
- 현재 모델과 후보가 다를 때만 고유한 시각 기반 폴더에 백업합니다.
- 임시 파일의 SHA-256을 확인한 뒤 원자적으로 현재 모델을 교체합니다.
- 앱도 모델 로드 직전에 현재 메타데이터와 모델 SHA-256을 다시 비교합니다.
- joblib/pickle은 코드 실행이 가능한 형식이므로 저장소 내부에서 검증된 모델만 신뢰합니다.

## 검증 경계

- 독립된 외부 라벨 Holdout이 없어 모든 운영 품질 수치는 5-Fold OOF 근거입니다.
- 미래 예측 기간, 인과관계, 캠페인 Uplift·ROI는 검증되지 않았습니다.
- CRM 발송·할인·고객 접촉과 최종 사업·법적 승인은 외부 결정입니다.
