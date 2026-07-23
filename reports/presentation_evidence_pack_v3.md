# PlaylistPro 발표 근거 팩 v3

## 결론

- 동일한 `log_numeric` 특성과 저장된 5-fold 분할로 8개 모델을 공정 비교했다.
- 7개 학습 모델의 제한된 RandomizedSearch와 실제 상위 3개 모델의 15회 정밀 탐색을 완료했다.
- 최종 검토 후보는 CatBoost다. Fine-tuned CV PR-AUC `0.94791`, OOF PR-AUC `0.94790`, 5개 Seed 평균 PR-AUC `0.94788`이다.
- CatBoost·XGBoost·LightGBM의 Bootstrap 구간이 겹치므로 단일 점수의 압도적 우위가 아니라, 평균 성능·Seed 안정성·운영 지표를 함께 사용한 추천이다.
- 외부 라벨 Holdout, 캠페인 Uplift, 인과효과, 예측 기간은 검증하지 않았으며 주장하지 않는다.

## 발표용 그래프와 원본 데이터

모든 그래프는 `figures/presentation_v3/`에 있으며, 대응 원본 CSV는 `artifacts/presentation_v3/`에 있다. 전체 연결표는 `figures/presentation_v3/figure_inventory.csv`다.

| 번호 | 발표 질문 | 그래프 | 원본 CSV |
| --- | --- | --- | --- |
| 01 | Baseline에서 최종 후보까지 얼마나 개선됐는가? | `01_performance_progression.png` | `01_performance_progression.csv` |
| 02 | 어떤 전처리가 실제 성능을 높였는가? | `02_preprocessing_experiments.png` | `02_preprocessing_experiments.csv` |
| 03 | 어떤 실험을 채택·제외했는가? | `03_adopted_rejected_experiments.png` | `03_adopted_rejected_experiments.csv` |
| 04 | 8개 모델을 동일 조건에서 비교했는가? | `04_eight_model_fair_comparison.png` | `04_eight_model_fair_comparison.csv` |
| 05 | 7개 모델의 탐색 전후 변화는 무엇인가? | `05_seven_model_search_before_after.png` | `05_seven_model_search_before_after.csv` |
| 06 | 실제 Top 3 정밀 튜닝 결과는 무엇인가? | `06_top3_fine_tuning_before_after.png` | `06_top3_fine_tuning_before_after.csv` |
| 07 | Seed가 달라져도 결과가 안정적인가? | `07_seed_stability.png` | `07_seed_stability.csv` |
| 08 | 점수 불확실성은 어느 정도인가? | `08_bootstrap_confidence_intervals.png` | `08_bootstrap_confidence_intervals.csv` |
| 09 | 확률값이 관측 이탈률과 맞는가? | `09_calibration_curve.png` | `09_calibration_curve.csv` |
| 10 | Threshold에 따른 Precision·Recall·F1은? | `10_threshold_precision_recall_f1.png` | `10_11_threshold_sweep.csv` |
| 11 | Threshold에 따른 대상 고객·FN·FP는? | `11_threshold_volume_fn_fp.png` | `10_11_threshold_sweep.csv` |
| 12 | 같은 고객 수에 접촉하면 어느 모델이 더 포착하는가? | `12_equal_contact_comparison.png` | `12_equal_contact_comparison.csv` |
| 13 | 같은 Recall을 얻으려면 몇 명에게 접촉해야 하는가? | `13_equal_recall_comparison.png` | `13_equal_recall_comparison.csv` |
| 14 | Top-K Capture와 Lift는 어떤가? | `14_topk_capture_lift.png` | `14_topk_capture_lift.csv` |
| 15 | 위험도 구간이 실제 이탈률을 분리하는가? | `15_risk_decile.png` | `15_risk_decile.csv` |
| 16 | 기본 Threshold에서 FN·FP는 몇 건인가? | `16_final_confusion_matrix.png` | `16_final_confusion_matrix.csv` |
| 17 | 최종 후보의 주요 입력 신호는 무엇인가? | `17_final_feature_importance.png` | `17_final_feature_importance.csv` |
| 18 | 고객군별 오류 차이가 있는가? | `18_segment_error_analysis.png` | `18_segment_error_analysis.csv` |
| 19 | 수치형 변수와 이탈 라벨의 상관관계는? | `19_numeric_target_correlations.png` | `19_numeric_target_correlations.csv` |
| 20 | 주요 범주별 관측 이탈률 차이는? | `20_categorical_churn_associations.png` | `20_categorical_churn_associations.csv` |

## LightGBM이 아닌 CatBoost를 선택한 이유

이전 P0에서는 LightGBM이 제한된 비교의 후보였다. 이번 전체 공정 비교에서는 실제 Top 3가 CatBoost, XGBoost, LightGBM으로 결정됐고 CatBoost가 다음 근거에서 최종 검토 후보가 됐다.

| 근거 | CatBoost | XGBoost | LightGBM |
| --- | ---: | ---: | ---: |
| Fine-tuned CV PR-AUC | 0.947913 | 0.947793 | 0.947806 |
| OOF PR-AUC | 0.947897 | 0.947765 | 0.947790 |
| 5-Seed 평균 PR-AUC | 0.947880 | 0.947726 | 0.947721 |
| 5-Seed 표준편차 | 0.000061 | 0.000061 | 0.000095 |

LightGBM은 Threshold 0.50에서 Recall이 더 높고 FN이 더 적다. 따라서 접촉 정책이 FN 최소화를 최우선으로 정하면 LightGBM 또는 더 낮은 CatBoost Threshold도 합리적인 선택이다. 현재 CatBoost 추천은 PR-AUC 기반 순위 품질과 안정성을 우선한 기술적 후보이며, Threshold는 별도의 사업 결정이다.

## 발표 핵심 메시지 5개

1. Raw Logistic PR-AUC `0.89950`에서 최종 CatBoost OOF PR-AUC `0.94790`까지 개선했고, 각 단계의 변화와 제외 실험을 기록했다.
2. 8개 모델을 동일 특성·동일 5-fold 조건으로 비교했으며, RandomizedSearch와 Top 3 정밀 튜닝을 거쳐 후보를 결정했다.
3. CatBoost는 Fine-tuned CV와 5-Seed 평균 PR-AUC가 가장 높았지만 상위 3개 모델 차이는 작다. 절대적 승리로 표현하지 않는다.
4. Threshold `0.35`의 OOF 균형 시나리오는 Recall `0.9444`, Precision `0.7959`, FN `3,569`, 대상 고객 `76,143명`이다. 실제 접촉 규모는 사업 담당자가 선택해야 한다.
5. `weekly_hours`, 구독 유형, 문의 수준, 중지 횟수, skip rate 등이 강한 연관 신호지만 인과 원인으로 해석하지 않는다.

## 검증 경계

- 전체 성능은 저장된 5-fold OOF, 5개 Seed, 1,000회 Bootstrap 근거다.
- 독립적인 라벨 Holdout이 없어 외부 일반화는 확인 불가다.
- Holdout 결과를 본 뒤 모델·Feature·파라미터·Threshold를 다시 선택하는 행위는 수행하지 않았다.
- 모델 재학습 없이 저장된 결과로 이 발표 팩과 그래프를 생성했다.
