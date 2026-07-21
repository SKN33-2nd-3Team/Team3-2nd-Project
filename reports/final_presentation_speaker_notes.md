# PlaylistPro 최종 발표자 대본

## Slide 1. 모든 고객을 관리할 수 없다면, 먼저 검토할 고객을 정해야 한다

“이 프로젝트는 누가 이탈할지를 맞히는 데서 끝나지 않습니다. 리텐션 조직이 모든 고객에게 같은 수준으로 대응할 수 없다는 문제에서 출발했습니다. 고객의 구독·이용·문의 신호로 위험 순위를 만들고, 제한된 인원 안에서 누구부터 검토할지 선택할 수 있는 구조를 만드는 것이 목표였습니다. 최종적으로 CatBoost 위험 점수와 Threshold·Top-K 분석을 Streamlit 검토 흐름에 연결했습니다.”

전환: “먼저 좋은 예측이 어떤 운영 결정을 지원해야 하는지 정의하겠습니다.”

## Slide 2. 예측의 목적은 이탈 판정이 아니라 검토 자원의 배분이다

“떠날 고객을 놓치는 FN을 줄이려면 더 많은 고객을 검토해야 하고, 그만큼 남을 고객까지 검토하는 FP가 늘어납니다. 반대로 적중률을 높이면 검토량은 줄지만 놓치는 이탈 고객이 많아집니다. 그래서 이 프로젝트는 하나의 Threshold를 정답으로 제시하지 않고, Recall 우선·균형·Precision 우선 시나리오를 비교합니다. PR-AUC도 데이터 불균형 때문이 아니라, 관심 클래스의 순위 품질과 Recall–Precision 균형을 보기 위해 사용했습니다.”

전환: “이 의사결정을 해석하려면 데이터와 검증 범위를 먼저 명확히 해야 합니다.”

## Slide 3. 12.5만 고객의 `churned`를 학습했지만 시간 지평과 외부 일반화는 아직 모른다

“학습 데이터는 고객 12만 5천 명, 점수 산출용 test는 7만 5천 명입니다. 고객 한 명이 한 행이며, 제공된 `churned` 라벨을 예측했습니다. 다만 관측 기준일과 결과 기간이 없어 ‘향후 30일 이탈’처럼 말할 수 없습니다. 데이터는 변수 균등성과 계단식 이탈률 등으로 규칙 기반 합성으로 판정됐고, test에는 라벨이 없어 독립 외부 Holdout 평가도 불가능합니다. 따라서 뒤의 수치는 모델링 절차와 내부 순위 품질의 근거입니다.”

전환: “이 경계 안에서 어떤 신호가 가장 강하게 라벨을 분리하는지 확인했습니다.”

## Slide 4. 낮은 청취, Free 구독, 높은 문의가 가장 강한 관측 신호였다

“핵심 인사이트는 세 축으로 압축했습니다. 첫째, 주간 청취시간이 낮을수록 관측 이탈률이 높았고 5·10·40시간에서 계단식 변화가 보였습니다. 둘째, Free 고객의 관측 이탈률은 79.4%로 Premium의 33.9%보다 높았습니다. 셋째, 문의 수준 High는 74.3%, Low는 28.9%였습니다. 이 결과는 이용 저하, 구독 맥락, 서비스 마찰이라는 행동 가설을 만들지만 원인을 증명하지는 않습니다.”

전환: “이 관찰이 실제 모델 성능에도 도움이 되는지 Feature 실험으로 바꿨습니다.”

## Slide 5. 로그 변환은 채택됐고 비율·상호작용·신호 축소는 제외됐다

“Feature 효과만 비교하기 위해 Logistic을 고정 검증기로 사용했습니다. Raw PR-AUC 0.8995에서 로그 수치 Feature는 0.9063으로 0.0068 상승했습니다. 반면 행동 구성 비율, 상호작용, 신호 축소는 개선이 거의 없거나 오히려 감소해 제외했습니다. 이후 CatBoost에서도 로그 Feature가 가장 높았지만 Raw와의 차이는 약 0.00009로 매우 작았습니다. 따라서 로그 변환은 공통 Feature 선정 근거이지 최종 성능 향상의 대부분을 설명하지는 않습니다.”

전환: “Feature를 고정한 다음에는 모델만 바꿔 차이를 공정하게 측정했습니다.”

## Slide 6. 동일 Fold와 Pipeline으로 변화의 원인을 분리했다

“모든 모델은 동일한 `log_numeric` Feature와 동일한 Stratified 5-Fold를 사용했습니다. 전처리는 각 학습 Fold 안에서만 Fit했고 Target 기반 인코딩은 사용하지 않았습니다. Dummy PR-AUC 0.513, Raw Logistic 0.8995, 로그 Logistic 0.9063, CatBoost baseline 0.9476으로 진행됐습니다. 가장 큰 상승은 전처리보다 선형 모델에서 비선형 부스팅 모델로 전환할 때 발생했습니다. Recall·Precision·F1·FN·FP·Brier도 함께 기록했습니다.”

전환: “이제 같은 조건에서 비교한 8개 모델 중 실제 경쟁 후보를 보겠습니다.”

## Slide 7. 8개 모델 비교 후 CatBoost·XGBoost·LightGBM만 정밀 검증에 남았다

“Dummy를 포함한 8개 모델을 비교했고, 7개 학습 모델에 제한된 RandomizedSearch를 적용했습니다. 그 결과 CatBoost·XGBoost·LightGBM이 실제 Top 3로 남았고 각 15회 정밀 탐색을 수행했습니다. Fine-tuned CV PR-AUC는 CatBoost 0.947913, LightGBM 0.947806, XGBoost 0.947793으로 매우 가까웠습니다. Gradient Boosting은 탐색 후 오히려 감소했습니다. 즉, 더 많은 탐색이 자동으로 더 좋은 결과를 만든 것은 아닙니다.”

전환: “격차가 이렇게 작다면 최고 한 번의 점수가 아니라 반복 안정성과 불확실성이 중요합니다.”

## Slide 8. CatBoost는 근소한 순위 품질과 Seed 안정성으로 선택됐지만 압도적 승자는 아니다

“CatBoost는 Fine-tuned CV와 5개 Seed 평균 PR-AUC가 가장 높았고 Seed 표준편차도 0.000061로 작았습니다. 하지만 1,000회 Bootstrap 신뢰구간은 세 모델이 겹칩니다. LightGBM은 Threshold 0.50 Recall과 Brier에서 더 좋았습니다. 그래서 CatBoost를 통계적으로 압도적인 승자라고 말하지 않습니다. 현재 선정 규칙이 순위 품질과 반복 안정성을 우선했기 때문에 CatBoost를 기술 후보로 선택했고, FN 최소화 정책이라면 LightGBM이나 더 낮은 CatBoost Threshold도 합리적입니다.”

전환: “모델을 선택한 뒤에도 실제 대상 규모는 Threshold에 따라 달라집니다.”

## Slide 9. Threshold를 낮추면 누락은 줄고 검토 대상은 늘어난다

“Recall 우선 0.29에서는 62.16%를 검토해 Recall 95.18%, FN 3,091건을 얻습니다. Precision 우선 0.74에서는 대상이 38.77%로 줄고 Precision이 95.07%까지 오르지만 FN은 18,099건으로 증가합니다. 기본 0.35는 OOF F1 최고점일 뿐 현업의 정답이 아닙니다. 접촉 가능 인원과 FN·FP 비용이 주어져야 최종 운영 기준을 선택할 수 있습니다.”

전환: “현장에서는 Threshold보다 이번 회차에 볼 수 있는 인원이 먼저 정해지기도 합니다.”

## Slide 10. Top-K는 제한된 인력에서 가장 직접적인 검토 기준이다

“위험 순위 상위 10%는 관측 이탈 고객의 19.48%, 상위 20%는 38.95%, 상위 30%는 56.81%를 포함했습니다. 무작위 선정과 비교한 Lift는 약 1.9입니다. 이 결과는 제한된 인력에서 Top-K가 직접적인 운영 기준이 될 수 있음을 보여줍니다. 다만 Lift 1.9는 유지 효과가 1.9배라는 뜻이 아니라, 관측 이탈 라벨을 더 집중해서 포함했다는 뜻입니다. 극단적인 Decile 분리는 합성 규칙의 영향도 큽니다.”

전환: “이 순위와 시나리오를 사용자가 실제로 검토할 수 있도록 앱에 연결했습니다.”

## Slide 11. Streamlit은 점수·위험 신호·행동 가설을 한 검토 흐름으로 연결한다

“Streamlit은 저장된 CatBoost의 해시를 확인한 뒤 실제 `predict_proba`를 호출합니다. 사용자는 운영 시나리오를 비교하고, 단건 또는 7만 5천 명 배치에서 위험 순위를 확인할 수 있습니다. 별도의 투명한 규칙이 문의·청취·스킵·정지·구독 신호를 행동 후보로 바꾸지만, 모델이 학습한 최적 처방은 아닙니다. 담당자는 1순위 행동·대안·직접 검토·보류 중 하나를 선택합니다. 앱은 CRM 발송이나 할인을 실행하지 않습니다.”

전환: “따라서 현재 가능한 결정과 다음 검증 질문을 분리해 마무리하겠습니다.”

## Slide 12. 현재는 검토 우선순위 도구이며, 다음 검증은 시간 기반 외부 평가와 A/B 테스트다

“현재까지 확인한 것은 5-Fold OOF 순위 품질, Seed 안정성, Bootstrap 불확실성, 저장·재로딩, 앱 추론입니다. 아직 관측 기준일과 예측 기간, 독립 미래 라벨 Holdout, 실제 캠페인 Uplift와 ROI는 없습니다. 다음 순서는 예측 지평을 정의하고, 미래 라벨로 외부 평가를 수행한 뒤, 운영 Threshold를 승인하고, A/B 테스트로 행동 효과를 측정하는 것입니다. 이 프로젝트는 ‘어떤 행동이 이탈을 줄이는가’가 아니라 ‘누구부터 검토할 것인가’를 해결했습니다.”

마지막 문장: “따라서 CatBoost 위험 점수는 현재 고객 검토와 후속 유지 실험 대상 선정의 근거로 사용할 수 있습니다.”

---

# 예상 질문과 답변

## Q1. 왜 Accuracy가 아니라 PR-AUC를 주요 지표로 사용했나요?

데이터 불균형 때문이 아닙니다. 이탈 비율은 51.34%로 거의 균형입니다. 관심 클래스인 `churned=1`의 순위 품질과 Threshold 전 구간의 Precision–Recall Trade-off를 비교하기 위해 PR-AUC를 1차 지표로 사용했고 ROC-AUC·F1·Recall·Precision·FN·FP·Brier를 보조로 확인했습니다.

근거: `docs/requirements.md`, `docs/validation_plan.md`, `artifacts/model_comparison_fair.csv`

## Q2. 왜 CatBoost를 선택했나요?

Top 3 중 Fine-tuned CV PR-AUC 0.947913과 5-Seed 평균 0.947880이 가장 높았고 Seed 변동도 작았습니다. 다만 Bootstrap 구간이 겹치므로 압도적 우위는 아닙니다. 현재의 기술적 선정 기준이 순위 품질과 반복 안정성을 우선했기 때문에 선택했습니다.

근거: `artifacts/final_model_selection_matrix.csv`, `artifacts/bootstrap_confidence_intervals.csv`

## Q3. 최고 성능 모델과 최종 선정 모델이 다른가요?

최종 선정 기준인 Fine-tuned CV PR-AUC에서는 CatBoost가 1위입니다. 다만 Random search 단계의 CatBoost OOF 0.948004는 최종 후보 OOF 0.947897보다 약간 높았습니다. 단일 OOF 최고값을 다시 선택 기준으로 사용하지 않고, 사전에 정한 Fine CV·Seed·Bootstrap 검증을 따른 결과입니다.

근거: `artifacts/presentation_v3/01_performance_progression.csv`, `artifacts/top3_fine_tuning_summary.csv`

## Q4. 데이터 누수는 어떻게 막았나요?

고객 식별자인 `customer_id`를 제외했고, 고정 5-Fold를 모든 후보에 공유했습니다. 전처리와 인코딩은 Pipeline 내부에서 학습 Fold에만 Fit했고 Target 기반 인코딩을 사용하지 않았습니다. train과 label-free test의 고객 ID 교집합도 0건입니다.

근거: `docs/data_card.md`, `docs/validation_plan.md`

## Q5. Holdout은 언제 사용했나요?

독립 외부 라벨 Holdout은 사용하지 못했습니다. 제공된 test 75,000명에는 라벨이 없어 추론에만 사용했습니다. 모든 성능·Threshold·Top-K 수치는 5-Fold OOF 근거입니다.

근거: `experiments/submission_full_comparison/20260721_full_fair_v1/run_manifest.json`, `artifacts/model/metadata.json`

## Q6. Feature Importance를 이탈 원인으로 볼 수 있나요?

볼 수 없습니다. 중요도는 모델이 예측에 사용한 신호의 기여를 나타낼 뿐 원인이나 행동 효과를 증명하지 않습니다. 이 프로젝트에서는 나이·지역·ID를 행동 배정 근거에서 제외하고, 서비스·이용·구독 신호만 행동 가설에 사용했습니다.

근거: `reports/current_insight_inventory.md`, `src/retention_strategy.py`

## Q7. Recall을 높이면 Precision이 낮아지는 문제는 어떻게 처리했나요?

하나의 Threshold를 자동 확정하지 않고 세 시나리오를 제공합니다. 0.29는 Recall 95.18%, 0.74는 Precision 95.07%입니다. 검토 용량과 FN·FP 비용을 기준으로 담당자가 선택합니다.

근거: `artifacts/threshold_operating_scenarios_v2.csv`

## Q8. 전체 고객의 몇 %를 관리해야 하나요?

현재 데이터만으로 하나의 비율을 확정할 수 없습니다. Threshold 시나리오는 38.77%~62.16% 범위를 보여주고, 용량이 먼저 정해진 경우 Top 10%·20%·30%의 Capture를 비교할 수 있습니다. 실제 비율은 인력·예산·고객 피로도를 반영해 결정해야 합니다.

근거: `artifacts/model/metadata.json`, `artifacts/presentation_v3/14_topk_capture_lift.csv`

## Q9. 모델을 사용하면 실제 이탈이 줄어드나요?

아직 알 수 없습니다. 모델은 위험 순위를 검증했지만 캠페인 처치 효과를 검증하지 않았습니다. 실제 이탈 감소는 대조군이 있는 A/B 테스트 또는 Uplift 실험으로 측정해야 합니다.

근거: `docs/validation_plan.md`, `reports/retention_action_layer_plan.md`

## Q10. 예측기간은 정말 30일인가요?

아닙니다. 관측 기준일과 해지 발생 시점이 없어 예측 기간은 미정의입니다. 현재 표현은 “스냅샷 고객 프로필로 제공된 `churned` 라벨 판별”입니다.

근거: `docs/requirements.md`, `artifacts/model/metadata.json`

## Q11. 새로운 고객 데이터에서도 같은 성능이 나오나요?

확인할 수 없습니다. 독립 미래 라벨 Holdout이 없고 데이터가 합성 규칙 기반으로 판정됐기 때문입니다. 새 시점의 실제 고객 데이터로 시간 기반 외부 검증이 필요합니다.

근거: `docs/data_card.md`, `docs/validation_plan.md`

## Q12. Streamlit의 예측값은 실제 저장 모델 결과인가요?

맞습니다. 앱은 `artifacts/model/music_churn_pipeline.joblib`을 로드하고 메타데이터 SHA-256과 비교합니다. 새 프로세스 재로딩, 표본 재추론, 6개 페이지 실행 검사를 통과했습니다. 화면에서 재학습하지 않습니다.

근거: `artifacts/model/metadata.json`, `reports/final_local_model_integration.md`

## Q13. 가장 의미 있었던 실패 실험은 무엇인가요?

행동량의 구성 비율이 더 나은 신호일 것이라는 가설입니다. +1 비율과 0 인지 비율 모두 Raw Logistic보다 약간 낮아 제외했습니다. 직관적으로 그럴듯한 Feature라도 동일 OOF 결과가 개선되지 않으면 채택하지 않았다는 점이 의미 있습니다.

근거: `artifacts/preprocessing_experiment_results.csv`

## Q14. 다음 단계에서 가장 먼저 검증해야 할 것은 무엇인가요?

먼저 관측 기준일과 예측 기간을 정의한 미래 라벨 데이터를 확보해야 합니다. 그 데이터로 시간 기반 외부 Holdout을 평가한 뒤 Threshold를 승인하고, 마지막으로 A/B 테스트로 유지 행동의 Uplift와 비용 효과를 검증해야 합니다.

근거: `docs/validation_plan.md`

## Q15. 성능과 Top-K Precision이 지나치게 높은 이유는 무엇인가요?

데이터가 규칙 기반 합성으로 판정됐고 이탈률이 특정 문턱에서 계단식으로 변하기 때문입니다. 모델이 심어진 생성 규칙을 상당 부분 복원해 내부 분리도가 높습니다. 따라서 이 수치를 실제 스트리밍 서비스의 기대 성능으로 사용하면 안 됩니다.

근거: `docs/data_card.md`, `artifacts/eda_insight/INSIGHT_REPORT.md`

## Q16. LightGBM이 Recall과 Calibration에서 더 좋은데 왜 바꾸지 않았나요?

모델 선정 지표와 운영 Threshold를 분리했기 때문입니다. CatBoost는 Fine CV와 Seed 평균 PR-AUC가 근소하게 높았고, LightGBM은 Threshold 0.50 Recall과 Brier가 더 좋았습니다. FN 최소화가 절대 우선이면 LightGBM이나 낮은 CatBoost Threshold를 다시 선택할 수 있으며, 이는 사업 담당자의 정책 결정입니다.

근거: `reports/training_report.md`, `artifacts/calibration_summary.csv`

## Q17. 가정 기반 계획의 금액은 실제인가요?

아닙니다. 외부 공개 요금을 바탕으로 만든 편집 가능한 연간 매출 대용치와 채널 운영비 가정입니다. 실제 PlaylistPro의 LTV·기여이익·캠페인 비용·ROI가 아니므로 발표 본문에서는 재무 성과로 사용하지 않습니다.

근거: `reports/retention_action_layer_plan.md`, `README.md`
