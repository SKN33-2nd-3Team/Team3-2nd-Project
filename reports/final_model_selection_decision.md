# 최종 모델 선정 결정

## 결정

최종 로컬 기술 모델은 `CatBoost`입니다. 실제 상위 3개 중 Fine-tuned 5-Fold CV PR-AUC가 0.947913으로 가장 높고, 5개 Seed 평균 PR-AUC도 0.947880으로 가장 높았습니다.

## 단일 점수만으로 결정하지 않은 이유

CatBoost·XGBoost·LightGBM의 차이는 작습니다. Fine-tuned CV와 OOF PR-AUC, 5개 Seed 안정성, Bootstrap 신뢰구간, 확률 보정, Threshold와 접촉 용량 표를 함께 검토했습니다. Recall·FN·접촉 가능 인원은 사업 정책에 따라 달라지므로 최종 운영 Threshold는 사용자가 선택합니다.

## 한계

독립된 외부 라벨 Holdout이 없습니다. 최종 근거는 5-Fold OOF·Seed 안정성·Bootstrap이며, 예측 기간·캠페인 Uplift·인과효과를 검증하지 않습니다.
