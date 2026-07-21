# PlaylistPro 최종 발표 예상 Q&A

> 답변 원칙: 결론을 먼저 말하고, 근거 수치 하나와 한계를 붙인다. 각 답변은 약 20~40초 분량이다.

## 1. 왜 Accuracy가 아니라 PR-AUC를 주 지표로 사용했나요?

Accuracy는 특정 Threshold에서 전체 정답 비율을 보지만, 이 프로젝트는 제한된 인원 안에서 위험 고객을 얼마나 잘 위로 정렬하는지가 중요합니다. 그래서 Precision과 Recall의 관계를 Threshold 전반에서 보는 PR-AUC를 주 지표로 사용했습니다. 이탈 비율은 51.34%라서 ‘심한 클래스 불균형 때문’이라고 설명하지 않습니다. ROC-AUC, F1, Recall, Precision, FN, FP도 함께 확인했습니다.

- 근거: `artifacts/model/metadata.json`, 최종 Run 비교 CSV

## 2. 왜 최종 기술 후보가 CatBoost인가요?

같은 고정 5-Fold OOF 조건에서 CatBoost의 PR-AUC가 0.947897로 LightGBM 0.947790, XGBoost 0.947765보다 근소하게 높았고 Seed 평균도 가장 높았습니다. 다만 Bootstrap 구간이 겹치므로 압도적 승자라고 하지는 않습니다. 순위 품질과 안정성을 우선한 조건부 선택이며, Recall이나 Calibration을 더 중시하면 LightGBM을 선택할 수도 있습니다.

- 근거: `reports/final_model_selection_decision.md`, `figures/presentation_v3/08_top3_bootstrap_ci.png`

## 3. 최고 단일 점수와 최종 선정 모델이 다른가요?

탐색 단계의 CatBoost 점수 0.948004가 최종 fine OOF 0.947897보다 약간 높습니다. 하지만 탐색 최고값은 여러 후보 중 선택된 낙관적 값일 수 있어, 최종 비교는 사전에 고정한 상위 후보의 정밀 OOF·Seed 안정성·Bootstrap·운영 지표를 기준으로 했습니다. 따라서 최고 숫자 하나보다 재현 가능한 비교 절차를 우선했습니다.

- 근거: `experiments/submission_full_comparison/20260721_full_fair_v1/run_manifest.json`

## 4. 데이터 누수는 어떻게 방지했나요?

모든 모델에 동일한 고정 5-Fold 분할을 적용하고, 로그 변환·인코딩 등 전처리는 각 Fold의 학습 부분에서만 적합했습니다. 검증 Fold의 정보가 전처리 학습에 들어가지 않도록 Pipeline 단위로 평가했고 OOF 예측을 저장했습니다. 다만 시간 기반 외부 검증 데이터가 없기 때문에 현실 배포 수준의 누수·드리프트 검증까지 완료됐다고 주장하지는 않습니다.

- 근거: 최종 Run manifest, 저장된 fold assignment와 OOF 파일

## 5. Holdout 성능은 얼마인가요?

현 저장소의 7만 5천 명 테스트 데이터에는 `churned` 라벨이 없어 Holdout 성능을 계산할 수 없습니다. 따라서 발표 수치는 고정 5-Fold OOF 내부 검증 결과이고, 테스트는 최종 모델로 점수만 산출한 대상입니다. 이를 Holdout 검증이라고 부르지 않는 것이 핵심이며, 다음 단계에서 미래 시점의 실제 라벨 Holdout이 필요합니다.

- 근거: `docs/data_card.md`, `reports/source_repository_audit.md`

## 6. Feature importance가 높으면 이탈의 원인인가요?

아닙니다. 중요도와 그룹별 이탈률은 모델 안에서의 예측 연관성을 뜻할 뿐 인과효과를 증명하지 않습니다. 예를 들어 Free 고객의 이탈률이 높아도 할인하면 이탈이 줄어든다고 바로 결론 낼 수 없습니다. 그래서 해당 변수는 행동 가설을 세우는 신호로만 사용하고, 실제 행동 효과는 A/B 테스트로 검증해야 합니다.

- 근거: `reports/current_insight_inventory.md`, `docs/streamlit_insight_redesign_prompt.md`

## 7. Recall과 Precision 중 무엇이 더 중요한가요?

사업 비용에 따라 달라집니다. 놓친 이탈 고객의 비용이 크면 Recall과 낮은 FN을 우선하고, 상담 자원이 부족하거나 불필요한 접촉 비용이 크면 Precision과 낮은 FP를 우선합니다. 그래서 저희는 하나의 고정 Threshold를 정답으로 제시하지 않고 0.29, 0.35, 0.74 같은 운영 시나리오와 Top-K를 함께 제공합니다.

- 근거: `artifacts/threshold_operating_scenarios_v2.csv` 및 `artifacts/operating_scenarios_oof_catboost.csv`

## 8. 실제로 몇 퍼센트의 고객을 관리해야 하나요?

현재 데이터만으로 한 비율을 정답이라고 정할 수는 없습니다. 예를 들어 Threshold 0.29는 62.16%를 검토해 Recall 95.18%를 확보하고, Top 10%는 1만 2,500명으로 전체 이탈자의 19.48%를 포착합니다. 실제 선택은 월간 상담 가능 인원, 접촉 비용, 놓침 비용을 입력한 뒤 해야 합니다.

- 근거: `artifacts/threshold_operating_scenarios_v2.csv`, `artifacts/topk_lift_oof_catboost.csv`

## 9. 이 모델로 이탈률이 실제로 얼마나 줄어드나요?

아직 답할 수 없습니다. 모델은 위험 순위를 평가했지만, 추천 행동을 실행한 실험 데이터가 없으므로 이탈 감소율이나 uplift는 증명되지 않았습니다. Lift 1.95도 캠페인 효과가 아니라 상위 고객군에 실제 이탈자가 무작위보다 1.95배 농축됐다는 의미입니다. 실제 감소율은 캠페인 대조군을 둔 A/B 테스트가 필요합니다.

- 근거: `artifacts/topk_lift_oof_catboost.csv`, `reports/presentation_evidence_pack_v3.md`

## 10. ‘30일 이탈 예측’ 모델인가요?

아닙니다. 원천 데이터에 예측 기준일과 미래 관찰 기간이 정의되어 있지 않아 30일 또는 90일 모델이라고 부를 수 없습니다. 현재는 제공된 `churned` 라벨의 분류·순위 모델입니다. 다음 실제 운영 버전에서는 기준일을 고정하고 ‘향후 30일 내 이탈’처럼 타깃 기간을 명확히 정의해야 합니다.

- 근거: `docs/data_card.md`, `docs/requirements.md`

## 11. 새로운 고객 데이터가 들어오면 바로 사용할 수 있나요?

동일한 스키마와 전처리 조건이면 저장된 Pipeline으로 점수는 산출할 수 있습니다. 하지만 배포 전에는 컬럼·범주·분포 드리프트 검사와 점수 Calibration 점검이 필요합니다. 특히 현재 데이터가 합성 규칙 성격을 보이므로, 실제 고객 데이터에서는 먼저 Shadow 평가를 하고 Threshold를 다시 운영 검증해야 합니다.

- 근거: `artifacts/model/metadata.json`, `reports/source_repository_audit.md`

## 12. Streamlit은 실제 최종 모델을 사용하나요?

네. 현재 앱은 `artifacts/model`의 저장된 최종 Pipeline과 메타데이터를 읽어 고객 위험 점수를 보여주도록 통합했습니다. 다만 행동 추천은 모델이 직접 생성한 처방이 아니라 `src/retention_strategy.py`의 투명한 규칙입니다. 즉 예측 점수와 행동 가설의 출처를 분리해 사용자가 각각 검토할 수 있습니다.

- 근거: `app/streamlit_app.py`, `src/retention_strategy.py`, `artifacts/model/metadata.json`

## 13. 실패한 실험 중 가장 의미 있었던 것은 무엇인가요?

Gradient Boosting 탐색이 Baseline 대비 PR-AUC 0.000793 하락한 사례와, 여러 비율·상호작용 전처리가 거의 개선되지 않은 사례입니다. 이를 통해 복잡한 파생변수나 탐색이 자동으로 성능 향상을 만들지 않는다는 점을 확인했습니다. 개선되지 않은 실험을 제외한 근거까지 남긴 것이 재현성과 과적합 통제 측면에서 의미 있습니다.

- 근거: `artifacts/performance_progression.csv`, `reports/insight_driven_preprocessing_review.md`

## 14. 다음에 가장 먼저 검증할 것은 무엇인가요?

가장 먼저 예측 기준일과 미래 기간이 정의된 실제 Holdout을 확보하겠습니다. 그다음 시간 순서 검증으로 성능과 Calibration을 확인하고, 고객 세그먼트별 FN·FP를 점검합니다. 모델 검증이 끝난 뒤에야 유지 행동별 A/B 테스트로 실제 uplift와 비용 대비 효과를 확인하는 순서가 맞습니다.

- 근거: `reports/final_model_selection_decision.md`, `docs/data_card.md`

## 15. 데이터가 합성 데이터라면 이 프로젝트의 의미가 있나요?

현실 성능을 입증했다는 의미는 제한적이지만, 분석–모델 검증–운영 기준–행동 가설을 연결하는 재현 가능한 의사결정 구조를 만든 의미는 있습니다. 오히려 합성 데이터 가능성을 공개하고 인과·ROI·외부 일반화를 주장하지 않은 것이 중요합니다. 실제 데이터가 들어오면 같은 구조로 재검증할 수 있습니다.

- 근거: `reports/source_repository_audit.md`, `reports/final_presentation_consistency_audit.md`

## 16. LightGBM이 Recall과 Brier가 더 좋은데 왜 선택하지 않았나요?

0.5 기준 LightGBM의 Recall은 0.859850으로 CatBoost 0.836382보다 높고, Brier도 0.095049로 CatBoost 0.095215보다 조금 좋습니다. 반면 CatBoost는 OOF·Seed 평균 PR-AUC가 근소하게 앞섰습니다. 따라서 현재는 위험 순위 품질을 기준으로 CatBoost를 기술 후보로 두었지만, 사업이 높은 Recall이나 확률 정확성을 최우선으로 정하면 LightGBM으로 바꾸는 것이 합리적입니다.

- 근거: `reports/final_model_selection_decision.md`

## 17. 추천 행동은 어떤 근거로 만들었나요?

행동은 관찰된 신호를 운영자가 검토할 수 있는 가설로 번역했습니다. 높은 문의는 미해결 문의 확인, 낮은 청취는 재활성화, 높은 Skip은 콘텐츠 만족도, 높은 Pause는 복귀 장벽, Free는 혜택 인지 검토로 연결했습니다. 민감하거나 실행성이 낮은 Age·Location과 식별자인 ID는 행동 근거에서 제외했습니다. 효과가 증명된 처방이 아니라 검토 후보입니다.

- 근거: `src/retention_strategy.py`, `docs/streamlit_insight_redesign_prompt.md`

## 18. 화면의 접촉비용과 고객가치는 실제 값인가요?

아닙니다. 구독 유형과 접촉 방식에 따라 차등화한 편집 가능한 계획 가정입니다. 사용자가 시나리오를 비교하도록 돕기 위한 값이며 실제 LTV, ROI, 회계 기준으로 해석하면 안 됩니다. 실무 적용 시에는 재무·CRM 담당자가 실제 원가와 유지가치를 입력하고 검증해야 합니다.

- 근거: `src/retention_strategy.py`, Streamlit 고객 우선순위 화면

## 19. 왜 Age와 Location을 행동 추천에서 제외했나요?

고객이 바꾸기 어려운 특성이고, 차별적 운영으로 오해될 가능성이 있으며, 관찰된 연관성이 있어도 실행 가능한 유지 행동으로 직접 연결하기 어렵기 때문입니다. 예측 모델 입력과 고객 접촉 근거는 별개로 관리해야 합니다. 현재 행동 규칙은 사용·문의·구독 상태처럼 담당자가 서비스 개선으로 대응할 수 있는 신호에 한정했습니다.

- 근거: `src/retention_strategy.py`, `docs/streamlit_insight_redesign_prompt.md`

## 20. 이 프로젝트를 한 문장으로 설명하면 무엇인가요?

“PlaylistPro는 고객 이탈 위험을 예측하는 데서 멈추지 않고, 제한된 유지 자원 안에서 누구를 먼저 검토하고 어떤 행동 가설을 확인할지 근거와 함께 제시하는 의사결정 지원 도구입니다.” 모델 성능은 이 흐름의 신뢰성을 뒷받침하는 수단이고, 최종 가치는 사람이 검토 가능한 운영 순서에 있습니다.

- 근거: `README.md`, `docs/requirements.md`, 최종 발표 콘셉트
