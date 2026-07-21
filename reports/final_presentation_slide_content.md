# PlaylistPro 최종 발표 슬라이드 콘텐츠

## Slide 1. 모든 고객을 관리할 수 없다면, 먼저 검토할 고객을 정해야 한다

### 이 슬라이드가 답하는 질문

이 프로젝트가 해결한 문제와 최종적으로 만든 가치는 무엇인가?

### 화면에 표시할 핵심 문장

> 고객 행동 신호와 검증된 위험 점수를 연결해, 제한된 고객관리 자원에서 누구부터 검토할지 결정한다.

보조 문구: `고객 행동 이해 → 공정한 모델 검증 → Threshold·Top-K 우선순위 → 후속 유지 실험`

### 사용할 수치

- 학습 고객 125,000명
- 점수 산출 고객 75,000명
- 최종 기술 후보 CatBoost

### 사용할 표 또는 그래프

숫자 중심 차트는 사용하지 않는다. 제목과 한 문장 Thesis, `고객 신호 → 위험 순위 → 담당자 검토`의 단순한 3단 흐름만 배치한다.

### 그래프 해석

해당 없음. 첫 장의 역할은 모델 점수가 아니라 의사결정 문제를 기억시키는 것이다.

### 발표자가 강조할 내용

이 프로젝트의 목표는 모든 고객에게 캠페인을 보내는 것이 아니라, 위험 고객을 먼저 검토할 수 있는 근거를 만드는 것이다.

### 다음 슬라이드 연결

“그렇다면 좋은 예측이란 단순히 맞히는 것이 아니라, 어떤 오류와 자원 제약을 줄여야 하는지부터 정의해야 합니다.”

### 주의하거나 말하지 말아야 할 내용

- 첫 장에서 PR-AUC·파라미터·모델 비교표를 나열하지 않는다.
- “실제 이탈을 줄였다” 또는 “운영 배포 완료”라고 말하지 않는다.

### 근거 파일

- `README.md`
- `artifacts/model/metadata.json`
- `reports/final_local_model_integration.md`

---

## Slide 2. 예측의 목적은 이탈 판정이 아니라 검토 자원의 배분이다

### 이 슬라이드가 답하는 질문

고객 이탈 예측이 어떤 의사결정을 지원하는가?

### 화면에 표시할 핵심 문장

> FN을 줄이면 더 많은 이탈 고객을 찾지만 검토 대상과 FP가 늘어난다. 따라서 모델은 정답 하나가 아니라 선택 가능한 운영 시나리오를 제공해야 한다.

### 사용할 수치

- FN: 떠날 고객을 검토 대상에서 놓침
- FP: 남을 고객을 불필요하게 검토함
- 최종 운영 Threshold: 미확정

### 사용할 표 또는 그래프

가로 균형축 하나를 사용한다.

`누락 최소화·높은 Recall ← 검토 용량과 비용 → 높은 적중률·높은 Precision`

아래에 사용자와 결정 항목을 짧게 표시한다.

- 사용자: 리텐션 담당자
- 결정: 검토 대상 규모, 우선순위, 행동 가설 선택

### 그래프 해석

모델의 확률 순위와 사업 담당자의 용량·오류 비용 판단이 결합돼야 실제 검토 기준이 된다.

### 발표자가 강조할 내용

Accuracy 단독 최적화가 목적이 아니다. `churned=1` 고객의 순위를 얼마나 잘 정하고, Threshold를 바꿀 때 Recall·Precision·대상 고객 수가 어떻게 달라지는지가 핵심이다.

### 다음 슬라이드 연결

“이 의사결정을 측정하려면 먼저 데이터가 무엇을 의미하고 어디까지 검증할 수 있는지 경계를 정해야 합니다.”

### 주의하거나 말하지 말아야 할 내용

- 데이터가 불균형해서 PR-AUC를 사용했다고 말하지 않는다. 양성 비율은 51.34%다.
- F1 최고점을 현업 최적안으로 단정하지 않는다.

### 근거 파일

- `docs/requirements.md`
- `docs/validation_plan.md`
- `artifacts/model/metadata.json`

---

## Slide 3. 12.5만 고객의 `churned`를 학습했지만 시간 지평과 외부 일반화는 아직 모른다

### 이 슬라이드가 답하는 질문

무엇을 이용해 무엇을 예측했고, 검증 범위는 어디까지인가?

### 화면에 표시할 핵심 문장

> 고객 1행의 구독·이용·문의 프로필로 제공된 `churned` 라벨을 판별했다. 데이터는 규칙 기반 합성으로 판정됐고 예측 기간과 독립 외부 Holdout은 없다.

### 사용할 수치

- Train: 125,000명 × 20열, `churned` 포함
- Test: 75,000명 × 19열, 정답 라벨 없음
- Train 이탈 비율: 51.34%
- Train–Test 고객 ID 교집합: 0건
- 결측값·중복행: 0건

### 사용할 표 또는 그래프

두 영역으로 구성한다.

1. 왼쪽: `고객 1명 = 1행`과 주요 Feature 범주(구독·이용·문의)
2. 오른쪽: 검증 경계 3개
   - 규칙 기반 합성 판정
   - 예측 기간 미정의
   - 외부 라벨 Holdout 없음

### 그래프 해석

내부 성능은 모델링 절차를 검증하는 근거지만 실제 서비스의 미래 성능으로 그대로 이전할 수 없다.

### 발표자가 강조할 내용

합성 데이터라는 사실은 프로젝트를 무효화하지 않는다. 대신 이 프로젝트의 가치는 실제 고객 효과를 증명하는 데 있지 않고, 문제 정의·누수 방지·공정 비교·운영 연결 절차를 재현 가능하게 만든 데 있다.

### 다음 슬라이드 연결

“이 경계 안에서, 모델링과 운영 가설에 사용할 만한 강한 신호를 먼저 찾았습니다.”

### 주의하거나 말하지 말아야 할 내용

- “향후 30일 이탈”이라고 표현하지 않는다.
- 75,000명 test를 Holdout 성능 평가에 사용했다고 말하지 않는다.
- 합성 판정 근거를 숨기지 않는다.

### 근거 파일

- `docs/data_card.md`
- `docs/data_dictionary.md`
- `docs/validation_plan.md`
- `experiments/submission_full_comparison/20260721_full_fair_v1/run_manifest.json`

---

## Slide 4. 낮은 청취, Free 구독, 높은 문의가 가장 강한 관측 신호였다

### 이 슬라이드가 답하는 질문

데이터에서 어떤 고객 행동과 구독 맥락이 `churned`와 연관됐는가?

### 화면에 표시할 핵심 문장

> 이용 저하, 구독 맥락, 서비스 마찰이라는 서로 다른 세 축에서 이탈률 차이가 관찰됐다.

### 사용할 수치

- `weekly_hours`와 Target 상관계수: −0.302
- Free 관측 이탈률: 79.41% / Premium: 33.91%
- 문의 High 관측 이탈률: 74.33% / Low: 28.92%
- 보조 신호: 일시정지 횟수 상관 0.183, 스킵률 상관 0.160

### 사용할 표 또는 그래프

- 주 그래프: `artifacts/eda_insight/02_weekly_hours_steps.png`
- 보조 그래프: `figures/presentation_v3/20_categorical_churn_associations.png`
- 세 개의 인사이트 라벨만 표시한다: 이용 저하 / 구독 맥락 / 서비스 마찰

### 그래프 해석

낮은 청취 활동은 가장 강한 수치 신호였고, Free 구독과 높은 문의 수준은 범주별 이탈률을 크게 분리했다. 다만 이 차이는 합성 생성 규칙과 연관된 관찰이며 인과관계가 아니다.

### 발표자가 강조할 내용

세 신호는 서로 다른 대응 가설로 연결된다. 저사용은 콘텐츠 경험 진단, 높은 문의는 서비스 회복, Free는 혜택·전환 실험 후보가 된다. 실제 효과는 아직 검증되지 않았다.

### 다음 슬라이드 연결

“관찰을 설명으로 끝내지 않고, 실제 성능이 달라지는 Feature 가설로 바꿔 검증했습니다.”

### 주의하거나 말하지 말아야 할 내용

- “낮은 청취가 이탈을 유발했다”라고 말하지 않는다.
- Feature Importance를 이 슬라이드의 관찰 근거로 대체하지 않는다.
- 나이·지역은 행동 배정 근거로 사용하지 않는다.

### 근거 파일

- `artifacts/presentation_v3/19_numeric_target_correlations.csv`
- `artifacts/presentation_v3/20_categorical_churn_associations.csv`
- `reports/current_insight_inventory.md`
- `docs/data_card.md`

---

## Slide 5. 로그 변환은 채택됐고 비율·상호작용·신호 축소는 제외됐다

### 이 슬라이드가 답하는 질문

관찰된 인사이트를 어떤 Feature 가설로 바꾸었고, 무엇을 채택했는가?

### 화면에 표시할 핵심 문장

> 치우친 수치 분포의 로그 변환만 Logistic 1차 검증에서 의미 있는 개선을 보였다. 실패한 비율·상호작용·신호 축소안도 같은 기준으로 제외했다.

### 사용할 수치

- Raw Logistic PR-AUC: 0.899495
- Log-feature Logistic PR-AUC: 0.906293, 변화 +0.006798
- +1 비율: −0.000009
- 0 인지 비율: −0.000010
- 상호작용: +0.000142
- 신호 축소: −0.000002
- CatBoost Raw 대비 log_numeric 변화: 약 +0.000089

### 사용할 표 또는 그래프

- 주 그래프: `figures/presentation_v3/03_adopted_rejected_experiments.png`
- 오른쪽 작은 결정표: 채택 `log_numeric`, 대표 제외 `plus1_ratios`
- 하단 주석: “Logistic은 Feature 효과를 비교한 고정 검증기이며 최종 모델이 아님”
- `02_preprocessing_experiments.png`는 음수 값 레이블 겹침이 있어 원본 CSV로 다시 그릴 때만 사용한다.

### 그래프 해석

로그 변환은 선형 검증기에서 가장 분명한 개선을 보였지만 트리 모델에서의 추가 이득은 매우 작았다. 따라서 공통 Feature 선정 근거로 사용하되 최종 성능 향상의 주원인으로 과장하지 않는다.

### 발표자가 강조할 내용

가장 의미 있는 실패는 직관적으로 그럴듯한 비율 Feature가 성능을 높이지 못한 것이다. 가설의 설득력보다 동일 OOF 평가 결과를 우선했다.

### 다음 슬라이드 연결

“이제 Feature를 고정하고, 동일한 분할과 Pipeline에서 모델만 바꿔 성능 차이를 측정했습니다.”

### 주의하거나 말하지 말아야 할 내용

- Logistic을 최종 후보처럼 표현하지 않는다.
- Raw Logistic에서 CatBoost까지의 전체 개선을 전처리 효과로 묶지 않는다.
- 실패 실험을 숨기지 않는다.

### 근거 파일

- `artifacts/preprocessing_experiment_results.csv`
- `experiments/submission_full_comparison/20260721_full_fair_v1/tree_feature_validation.csv`
- `reports/insight_driven_preprocessing_review.md`

---

## Slide 6. 동일 Fold와 Pipeline으로 변화의 원인을 분리했다

### 이 슬라이드가 답하는 질문

성능 개선을 어떻게 공정하고 재현 가능하게 측정했는가?

### 화면에 표시할 핵심 문장

> 모든 모델에 동일한 `log_numeric` Feature와 고정 Stratified 5-Fold를 적용하고, 전처리는 각 학습 Fold 안에서만 Fit했다.

### 사용할 수치

- 고정 Stratified 5-Fold, `random_state=42`
- Dummy PR-AUC: 0.513386
- Raw Logistic: 0.899495
- Log-feature Logistic: 0.906293
- CatBoost baseline: 0.947622
- 최종 CatBoost OOF: 0.947897

### 사용할 표 또는 그래프

- 주 그래프: `figures/presentation_v3/01_performance_progression.png`
- 하단 검증 원칙 3개: 동일 Fold / Pipeline 내부 Fit / Target 기반 인코딩 없음

### 그래프 해석

Feature 가공 효과와 모델 가족의 효과를 단계적으로 분리했다. 가장 큰 성능 증가는 로그 변환보다 Logistic에서 비선형 부스팅 모델로 전환할 때 발생했다.

### 발표자가 강조할 내용

PR-AUC는 데이터 불균형 때문이 아니라, 관심 클래스인 `churned=1`의 순위 품질과 Recall–Precision Trade-off를 보기 위한 1차 지표다. ROC-AUC·F1·Recall·Precision·FN·FP·Brier도 함께 확인했다.

### 다음 슬라이드 연결

“공통 조건이 고정된 뒤에야 여러 모델의 경쟁력을 같은 표에서 비교할 수 있었습니다.”

### 주의하거나 말하지 말아야 할 내용

- Random search OOF 최고값이 최종 선정 규칙과 같다고 말하지 않는다.
- Accuracy만으로 개선을 설명하지 않는다.

### 근거 파일

- `docs/validation_plan.md`
- `artifacts/presentation_v3/01_performance_progression.csv`
- `artifacts/model_comparison_fair.csv`

---

## Slide 7. 8개 모델 비교 후 CatBoost·XGBoost·LightGBM만 정밀 검증에 남았다

### 이 슬라이드가 답하는 질문

여러 모델 중 어떤 후보가 실제로 경쟁력을 보였고, 탐색은 어떻게 진행했는가?

### 화면에 표시할 핵심 문장

> Dummy 포함 8개 모델을 같은 조건에서 비교하고, 7개 학습 모델을 제한 탐색한 뒤 실제 상위 3개만 각 15회 정밀 탐색했다.

### 사용할 수치

- 비교 모델: 8개
- RandomizedSearch 대상: 7개 학습 모델
- Top 3: CatBoost·XGBoost·LightGBM
- Fine-tuned CV PR-AUC: CatBoost 0.947913 / LightGBM 0.947806 / XGBoost 0.947793
- Random search 대표 실패: Gradient Boosting −0.000793

### 사용할 표 또는 그래프

- 주 그래프: `figures/presentation_v3/04_eight_model_fair_comparison.png`
- 보조 미니 그래프: `figures/presentation_v3/06_top3_fine_tuning_before_after.png`
- 전체 7개 탐색 전후는 부록 A2로 이동한다.

### 그래프 해석

상위 부스팅 모델 세 개의 차이는 매우 작고, 탐색을 많이 했다고 항상 성능이 오르지는 않았다. 후보 축소는 같은 Feature·Fold에서 얻은 결과만 사용했다.

### 발표자가 강조할 내용

최종 후보를 고르기 전에 모델 비교 조건을 먼저 고정했다. 과거 P0 LightGBM 결과와 이번 최종 공정 비교를 섞지 않았다.

### 다음 슬라이드 연결

“점수 차이가 작은 상황에서는 한 번의 최고값보다 반복 안정성과 불확실성을 함께 봐야 합니다.”

### 주의하거나 말하지 말아야 할 내용

- CatBoost가 8개 모든 지표에서 1위라고 말하지 않는다.
- 과거 LightGBM 선정 그래프를 사용하지 않는다.
- Fine Tuning을 무제한 탐색으로 표현하지 않는다.

### 근거 파일

- `artifacts/model_comparison_fair.csv`
- `artifacts/presentation_v3/05_seven_model_search_before_after.csv`
- `artifacts/top3_fine_tuning_summary.csv`
- `experiments/submission_full_comparison/20260721_full_fair_v1/run_manifest.json`

---

## Slide 8. CatBoost는 근소한 순위 품질과 Seed 안정성으로 선택됐지만 압도적 승자는 아니다

### 이 슬라이드가 답하는 질문

상위 세 모델의 점수가 비슷한데 왜 CatBoost를 최종 기술 후보로 선택했는가?

### 화면에 표시할 핵심 문장

> CatBoost는 Fine-tuned CV와 5-Seed 평균 PR-AUC가 가장 높았다. 그러나 Bootstrap 구간이 겹치므로 우위는 근소하며 운영 기준에 따라 LightGBM도 합리적이다.

### 사용할 수치

- CatBoost Fine-tuned CV PR-AUC: 0.947913
- CatBoost OOF PR-AUC: 0.947897
- CatBoost 5-Seed 평균 ± 표준편차: 0.947880 ± 0.000061
- CatBoost Bootstrap 95% CI: 0.946781–0.949086
- LightGBM Threshold 0.50 Recall: 0.859850 / CatBoost: 0.836382
- Brier: LightGBM 0.095049 / CatBoost 0.095215

### 사용할 표 또는 그래프

- 주 그래프: `figures/presentation_v3/08_bootstrap_confidence_intervals.png`
- 오른쪽 결정표: 순위 품질 / Seed 안정성 / Recall / Calibration
- Seed 전체 그래프와 Calibration은 부록 A4·A5로 이동한다.

### 그래프 해석

세 모델의 신뢰구간이 겹치므로 통계적으로 압도적인 단일 승자를 주장할 수 없다. 현재 선정 규칙은 순위 품질과 반복 안정성을 우선해 CatBoost를 선택했으며, Recall·보정을 최우선으로 하면 다른 선택도 가능하다.

### 발표자가 강조할 내용

모델 선택과 Threshold 선택은 별개다. CatBoost를 선택했다고 해서 Threshold 0.50을 고정하거나 FN이 자동으로 최소화되는 것은 아니다.

### 다음 슬라이드 연결

“같은 CatBoost라도 Threshold를 바꾸면 검토 대상과 오류 구조가 크게 달라집니다.”

### 주의하거나 말하지 말아야 할 내용

- “CatBoost가 통계적으로 유의하게 우수하다”고 말하지 않는다.
- Calibration이 최상이라고 말하지 않는다.
- 최고 단일 OOF 값만으로 선정했다고 말하지 않는다.

### 근거 파일

- `artifacts/final_model_selection_matrix.csv`
- `artifacts/presentation_v3/07_seed_stability.csv`
- `artifacts/presentation_v3/08_bootstrap_confidence_intervals.csv`
- `artifacts/calibration_summary.csv`
- `reports/final_model_selection_decision.md`

---

## Slide 9. Threshold를 낮추면 누락은 줄고 검토 대상은 늘어난다

### 이 슬라이드가 답하는 질문

같은 모델을 어떤 운영 기준으로 사용할 수 있는가?

### 화면에 표시할 핵심 문장

> 하나의 정답 Threshold는 없다. 누락 비용과 검토 용량에 따라 Recall 우선·균형·Precision 우선 시나리오를 선택해야 한다.

### 사용할 수치

| 시나리오 | Threshold | 대상 고객 | 대상 비율 | Recall | Precision | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Recall 우선 | 0.29 | 77,700 | 62.16% | 95.18% | 78.61% | 3,091 | 16,617 |
| 균형형 F1 | 0.35 | 76,143 | 60.91% | 94.44% | 79.59% | 3,569 | 15,538 |
| Precision 우선 | 0.74 | 48,463 | 38.77% | 71.80% | 95.07% | 18,099 | 2,388 |

### 사용할 표 또는 그래프

- 주 그래프: `figures/presentation_v3/10_threshold_precision_recall_f1.png`
- 오른쪽에 위 3개 시나리오의 대상 비율과 FN만 강조한 작은 표
- 전체 대상 수·FN·FP 곡선은 부록 A6에 배치한다.

### 그래프 해석

Threshold 0.29는 FN을 줄이는 대신 62.16%를 검토해야 한다. Threshold 0.74는 대상 적중률을 높이지만 18,099명의 이탈 고객을 놓친다. 기본 0.35는 OOF F1 최고점일 뿐 현업 최적안이 아니다.

### 발표자가 강조할 내용

운영 인력과 FN·FP 비용이 주어지지 않았기 때문에 복수 시나리오를 제공했다. 최종 선택은 리텐션 담당자의 책임이다.

### 다음 슬라이드 연결

“실제 현장에서는 Threshold보다 ‘이번 회차에 몇 명을 볼 수 있는가’가 먼저 정해지는 경우가 많습니다.”

### 주의하거나 말하지 말아야 할 내용

- 0.35를 최종 운영 Threshold라고 단정하지 않는다.
- OOF 시나리오를 외부 Holdout 성능처럼 표현하지 않는다.

### 근거 파일

- `artifacts/model/metadata.json`
- `artifacts/threshold_operating_scenarios_v2.csv`
- `artifacts/presentation_v3/10_11_threshold_sweep.csv`

---

## Slide 10. Top-K는 제한된 인력에서 가장 직접적인 검토 기준이다

### 이 슬라이드가 답하는 질문

검토 가능 인원이 정해져 있다면 위험 고객을 얼마나 포착할 수 있는가?

### 화면에 표시할 핵심 문장

> CatBoost 위험 순위 상위 10%는 관측 이탈 고객의 19.48%, 상위 30%는 56.81%를 포함했다. 이는 무작위보다 약 1.9배 높은 내부 포착 효율이다.

### 사용할 수치

- Top 5%: Capture 9.74%, Lift 1.95
- Top 10%: Capture 19.48%, Lift 1.95
- Top 20%: Capture 38.95%, Lift 1.95
- Top 30%: Capture 56.81%, Lift 1.89
- Top 40%: Capture 73.48%, Lift 1.84

### 사용할 표 또는 그래프

- 주 그래프: `figures/presentation_v3/14_topk_capture_lift.png`
- 보조 작은 그래프: `figures/presentation_v3/15_risk_decile.png`
- 10%·20%·30% 지점만 강조한다.

### 그래프 해석

Top-K는 검토 인원을 먼저 정한 뒤 위험 순서대로 대상을 선택할 수 있게 한다. 위험 Decile이 관측 이탈률을 강하게 분리하지만 극단적인 분리는 합성 생성 규칙의 영향이므로 실제 서비스 성능으로 일반화할 수 없다.

### 발표자가 강조할 내용

Lift는 캠페인으로 유지율이 1.9배 개선됐다는 뜻이 아니다. 무작위 선정 대비 관측 이탈 라벨을 얼마나 더 집중적으로 포함했는지를 나타낸다.

### 다음 슬라이드 연결

“이 순위와 운영 선택지를 실제 사용자가 검토할 수 있도록 Streamlit에 연결했습니다.”

### 주의하거나 말하지 말아야 할 내용

- Capture를 실제 유지 성공률로 표현하지 않는다.
- Top 10% Precision 100%를 실제 고객 환경에서도 보장한다고 말하지 않는다.
- 상위 비율을 자동 캠페인 대상으로 확정하지 않는다.

### 근거 파일

- `artifacts/presentation_v3/14_topk_capture_lift.csv`
- `artifacts/presentation_v3/15_risk_decile.csv`
- `docs/data_card.md`

---

## Slide 11. Streamlit은 점수·위험 신호·행동 가설을 한 검토 흐름으로 연결한다

### 이 슬라이드가 답하는 질문

분석 결과와 저장 모델을 사용자가 어떻게 활용하는가?

### 화면에 표시할 핵심 문장

> 저장된 CatBoost로 위험을 재계산하고, Threshold·Top-K·위험 신호·행동 후보를 함께 보여주되 최종 행동은 담당자가 선택한다.

### 사용할 수치

- 6개 페이지
- 저장 모델 SHA-256 검증 및 새 프로세스 재로딩 통과
- label-free test 고객 75,000명 배치 우선순위
- 전략 세그먼트 7개, 모든 행동은 담당자 검토 필요

### 사용할 표 또는 그래프

대표 화면 2개만 사용한다.

1. 운영 시나리오: Threshold·Top-K 선택 화면
2. 고객 우선순위: 단건 위험 점수 + 1순위·대안 행동 + 담당자 선택

두 캡처 사이에 `저장 모델 → 위험 순위 → 규칙 기반 행동 가설 → 담당자 결정` 흐름을 표시한다.

### 그래프 해석

모델은 위험 점수를 만들고, 별도의 투명한 전략 규칙이 이용·문의·구독 신호를 행동 후보로 번역한다. 앱은 CRM 발송·할인·고객 접촉을 실행하지 않는다.

### 발표자가 강조할 내용

행동 제안은 모델이 학습한 최적 처방이 아니다. 가정 기반 계획의 비용·연간 매출 대용치·추가 유지 성공률도 편집 가능한 민감도 분석이며 실제 LTV·ROI가 아니다.

### 다음 슬라이드 연결

“따라서 현재 앱이 지원하는 결정과 실제 운영 전에 더 검증해야 할 영역을 구분해야 합니다.”

### 주의하거나 말하지 말아야 할 내용

- 앱이 실제 캠페인을 실행한다고 말하지 않는다.
- 전략 규칙을 Uplift 모델 또는 인과 처방으로 표현하지 않는다.
- 가정 기반 금액을 실제 재무 효과로 발표하지 않는다.

### 근거 파일

- `app/streamlit_app.py`
- `src/retention_strategy.py`
- `artifacts/model/metadata.json`
- `reports/final_local_model_integration.md`
- `reports/retention_action_layer_plan.md`

---

## Slide 12. 현재는 검토 우선순위 도구이며, 다음 검증은 시간 기반 외부 평가와 A/B 테스트다

### 이 슬라이드가 답하는 질문

현재 결과에서 무엇을 사용할 수 있고, 무엇을 다음에 검증해야 하는가?

### 화면에 표시할 핵심 문장

> 지금 사용할 수 있는 것은 고객 검토 순위와 후속 실험 대상 선정이다. 실제 이탈 감소를 주장하려면 예측 지평, 미래 라벨, 외부 Holdout, A/B 테스트가 필요하다.

### 사용할 수치

현재 확인된 근거:

- 5-Fold OOF PR-AUC 0.947897
- 5개 Seed 평균 0.947880
- 1,000회 Bootstrap 완료
- 저장·재로딩·Streamlit 추론 검증 통과

아직 없는 근거:

- 관측 기준일과 예측 기간
- 독립 외부 라벨 Holdout
- 캠페인 Uplift·인과효과·실제 ROI

### 사용할 표 또는 그래프

네 단계의 다음 검증 Gate를 사용한다.

`1. 시간 지평 정의 → 2. 미래 라벨 Holdout → 3. Threshold·용량 승인 → 4. A/B 테스트와 Uplift·ROI 측정`

하단 결론:

> CatBoost 확률을 위험 고객 검토와 유지 실험 대상 선정의 근거로 사용한다.

### 그래프 해석

내부 검증과 제품 통합은 완료됐지만 사업 효과 검증은 시작 전이다. 프로젝트의 현재 상태와 다음 단계를 분리하는 것이 책임 있는 결론이다.

### 발표자가 강조할 내용

이 프로젝트는 “누구에게 먼저 질문할 것인가”를 해결했다. “어떤 행동이 실제 이탈을 줄이는가”는 다음 실험의 질문이다.

### 다음 슬라이드 연결

본문 종료. 질문이 들어오면 해당 근거 부록으로 이동한다.

### 주의하거나 말하지 말아야 할 내용

- 높은 내부 성능을 실제 서비스 일반화 성능으로 해석하지 않는다.
- 한계를 사과처럼 말하지 않고 다음 검증 계획으로 제시한다.
- “모델 완성”과 “사업 검증 완료”를 혼동하지 않는다.

### 근거 파일

- `docs/validation_plan.md`
- `experiments/submission_full_comparison/20260721_full_fair_v1/run_manifest.json`
- `artifacts/model/metadata.json`
- `reports/training_report.md`

---

# 부록 슬라이드 제작 사양

| 부록 | 결론형 제목 | 사용할 시각자료 | 발표 시 사용할 상황 |
| --- | --- | --- | --- |
| A1 | 비선형 부스팅 모델이 선형·단일 트리보다 높은 순위 품질을 보였다 | `04_eight_model_fair_comparison.png`와 전체 지표표 | “8개 모델을 모두 비교했나?” |
| A2 | 탐색은 모델별로 효과가 달랐고 Gradient Boosting은 오히려 하락했다 | `05_seven_model_search_before_after.png` | “튜닝하면 항상 좋아졌나?” |
| A3 | Top 3의 정밀 탐색 후에도 차이는 0.0002 이내였다 | `06_top3_fine_tuning_before_after.png` | “상위 후보 격차는?” |
| A4 | CatBoost의 Seed 변동은 작지만 Bootstrap 구간은 세 모델이 겹친다 | `07_seed_stability.png`, `08_bootstrap_confidence_intervals.png` | “안정성과 유의성은?” |
| A5 | LightGBM은 Brier와 평균 보정 오차에서 CatBoost보다 소폭 우수했다 | `09_calibration_curve.png`, `calibration_summary.csv` | “확률 보정도 CatBoost가 최고인가?” |
| A6 | Threshold 전체 구간에서 FN·FP와 검토 규모가 함께 변한다 | `11_threshold_volume_fn_fp.png`, `16_final_confusion_matrix.png` | “0.35 외 다른 기준은?” |
| A7 | 같은 Recall을 만들 때 모델별 필요 접촉량은 유사하다 | `12_equal_contact_comparison.png`, `13_equal_recall_comparison.png` | “실제 운영 효율 차이는?” |
| A8 | 성능은 구독 유형과 문의 수준에 따라 다르다 | `18_segment_error_analysis.png` | “세그먼트별 오류는?” |
| A9 | 중요 Feature는 생성 규칙과 일치하지만 원인·처방 근거는 아니다 | `17_final_feature_importance.png` | “왜 이 고객이 위험한가?” |
| A10 | 극단적인 분리 성능은 규칙 기반 합성 데이터의 특성과 연결된다 | `data_card.md`의 합성 판정 표 | “성능이 왜 이렇게 높은가?” |
| A11 | 저장 모델은 해시·재로딩·스키마·표본 재추론 검사를 통과했다 | `metadata.json` 핵심 필드 표 | “앱이 실제 모델을 쓰나?” |
| A12 | 앱은 예측·검토를 지원하지만 외부 CRM 상태를 변경하지 않는다 | Streamlit 6페이지 구조와 전략 경계 | “앱에서 무엇까지 가능한가?” |
