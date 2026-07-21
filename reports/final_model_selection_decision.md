# 최종 모델 선정 의사결정

## 추천 후보

현재 목적의 검토 후보는 `CatBoost`다. 이는 운영 승인 모델이라는 뜻이 아니다.

| 모델 | Fine-tuned CV PR-AUC | OOF PR-AUC | 5-Seed 평균 PR-AUC | Threshold 0.50 Recall | FN |
| --- | ---: | ---: | ---: | ---: | ---: |
| CatBoost | 0.947913 | 0.947897 | 0.947880 | 0.8364 | 10,500 |
| XGBoost | 0.947793 | 0.947765 | 0.947726 | 0.8443 | 9,993 |
| LightGBM | 0.947806 | 0.947790 | 0.947721 | 0.8598 | 8,994 |

CatBoost는 Fine-tuned CV와 5-Seed 평균 PR-AUC가 가장 높고 Seed 표준편차도 낮아, 현재 사전 정의된 1차 지표에 가장 잘 맞는다. 다만 상위 3개 모델의 1,000회 Bootstrap 신뢰구간이 겹치므로 점수 차이를 압도적 우위라고 주장하지 않는다.

## LightGBM과의 선택 차이

LightGBM은 기본 Threshold 0.50에서 CatBoost보다 Recall이 높고 FN이 `1,506건` 적다. 반면 CatBoost는 PR-AUC 기반 위험 순위 품질과 Seed 평균이 근소하게 높다.

- 위험 순위와 제한된 접촉 대상 선정을 우선하면 CatBoost가 현재 추천 후보이다.
- 기본 Threshold에서 FN 최소화를 최우선으로 하면 LightGBM이 합리적인 대안이다.
- CatBoost의 Threshold를 낮춘 Recall-first 시나리오도 가능하지만 접촉 고객과 FP가 늘어난다.

따라서 모델 후보는 CatBoost로 저장하되, 최종 운영 Threshold와 FN·접촉비용의 우선순위는 사용자가 선택하도록 분리했다.

## 검증 경계

- 근거: 동일 5-fold OOF, 5개 Seed, 1,000회 Bootstrap, Calibration, Threshold, Top-K/Lift, Risk Decile, Segment error.
- 독립 라벨 Holdout이 없어 외부 일반화는 확인 불가다.
- 캠페인 uplift, 인과효과, 예측 기간은 검증하지 않았다.
- Holdout을 본 뒤 모델·Feature·파라미터·Threshold를 다시 고르지 않았다.
