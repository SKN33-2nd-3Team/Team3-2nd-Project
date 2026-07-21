# PlaylistPro 최종 발표자 노트

## 1. 표지

이 프로젝트의 주인공은 모델이 아니라 리텐션·CRM 담당자입니다. 목표는 모든 고객에게 같은 비용을 쓰는 것이 아니라 먼저 검토할 고객과 유지 활동 후보를 정하는 것입니다.

## 2. Business Problem

구독·이용·문의 프로필과 제공된 `churned` 라벨을 바탕으로 위험 고객을 정렬하고, 위험 신호별 행동 가설을 담당자에게 제안합니다.

## 3. Data

Train 125,000명, 점수 산출 대상 75,000명, 학습 이탈률 51.34%입니다. 결측·중복·ID 겹침은 없지만, 합성 데이터·예측 기간 미정의·외부 Holdout 부재로 현실 효과를 확정할 수 없습니다.

## 4. Customer Insight

낮은 `weekly_hours`, Free 구독, 높은 문의, Skip·Pause가 이탈 라벨과 강하게 연결됐습니다. 이는 원인이 아니라 서비스 회복·재활성화·혜택 안내를 검토할 신호입니다.

## 5. Preprocessing

동일 OOF 조건에서 Logistic의 `log_numeric`만 PR-AUC를 0.899495에서 0.906293으로 개선해 채택했습니다. interaction·ratio·pruning은 개선이 작거나 없어 제외했고, CatBoost의 raw→log 차이가 +0.000089임도 함께 밝혔습니다.

## 6. Validation

모든 후보에 동일 Feature, 동일 5-Fold, Fold 내부 Fit, OOF 평가를 적용했습니다. 따라서 전처리와 모델 변경에 따른 차이를 비교할 수 있고 누수 위험을 통제했습니다.

## 7. Model Comparison

8개 모델 동일 조건 비교에서 CatBoost 0.947622, LightGBM 0.947534, XGBoost 0.947052가 상위였습니다. PR-AUC를 중심으로 보되 Recall·Precision·F1·FN·FP도 함께 확인했습니다.

## 8. Experiment Journey

Baseline에서 Search, Fine까지 성능 변화와 채택·제외 실험을 기록했습니다. 탐색 상한을 지켰고, 최고 단일 점수만으로 결론 내리지 않았습니다.

## 9. Final Model

CatBoost는 Fine OOF PR-AUC와 5개 seed 안정성에서 근소하게 앞서 조건부 최종 후보로 선정했습니다. LightGBM은 Recall·Calibration 측면의 장점이 있어 절대적 우승 모델로 과장하지 않습니다.

## 10. Model Persistence

고정 Pipeline을 joblib으로 저장하고 SHA-256, 새 프로세스 재로딩, 앱의 동일 모델 호출을 확인했습니다. 이 단계는 학습 결과가 실제 서비스 화면까지 전달되는지 검증하는 단계입니다.

## 11. Streamlit Demo

6개 화면에서 프로젝트 요약, 인사이트, 실험, 모델 선정, 고객 우선순위, 캠페인 계획을 연결합니다. 앱은 행동 후보를 보여주며 CRM 발송·할인 집행을 자동으로 수행하지 않습니다.

## 12. Operating Threshold

Recall 우선, 균형 F1, Precision 우선 시나리오를 대상 고객 수·Recall·Precision·FN·FP와 함께 제시합니다. Threshold는 모델의 정답이 아니라 접촉 비용과 허용 FN에 따른 담당자 선택입니다.

## 13. Prioritization

Top-K Capture·Lift·Risk Decile은 제한된 접촉 자원을 어디에 집중할지 보여줍니다. Top 10%의 Lift 1.95는 OOF 라벨 집중도이며 실제 캠페인 uplift나 ROI가 아닙니다.

## 14. Customer Workflow

고객 우선순위 화면에서 위험 점수, 운영 단계, 1차 신호, 다음 행동 후보를 확인하고 가정 기반 계획에서 비용·가치를 조정합니다. 실행과 승인은 사람의 책임입니다.

## 15. Retention Strategy

같은 고위험 고객이라도 문의·청취시간·Skip·Free 등 신호가 다르면 제안 행동이 달라야 합니다. 서비스 회복, 개인화 재활성화, 콘텐츠 개선, 혜택 안내는 검증할 가설입니다.

## 16. Final Takeaway

PlaylistPro는 데이터에서 위험 맥락을 찾고, 동일 조건 모델 검증으로 CatBoost를 조건부 선정한 뒤, Threshold·Top-K·행동 후보를 담당자 의사결정 흐름으로 연결했습니다. 실제 유지 효과는 미래 라벨과 A/B 테스트로 검증해야 합니다.
