# 검증 계획서 (Validation Plan)

> 가이드 §④(데이터 분리와 전처리 Pipeline)·§⑦(성능 평가와 임계값 결정) 대응.
> 구현: [`src/run_pipeline.py`](../src/run_pipeline.py) · 최종 갱신 2026-07-21.

## 1. 데이터 분할

| 세트 | 비율 | 행 수 | 역할 |
|---|---:|---:|---|
| Train | 60% | 75,000 | 모델·전처리기 학습 |
| Validation | 20% | 25,000 | 모델 선택, threshold 결정 |
| Test | 20% | 25,000 | **최종 1회 평가만** |

- 방식: `train_test_split(..., stratify=y, random_state=42)` 2단계 (60/40 → 40을 20/20으로)
- 근거: **고객 1명 = 1행**이므로 Group split이 불필요하다. 동일 고객 반복 기록이 없음을 `customer_id` 고유값 125,000 = 행 수로 확인했다.
- 시간 분할을 쓰지 않는 이유: 이벤트 타임스탬프가 없다. `signup_date`는 가입 시점일 뿐 관측·이탈 시점이 아니다. **타임스탬프가 확보되면 즉시 시간 분할로 전환**한다.
- 제공 `data/test.csv`는 라벨이 없으므로 **성능 평가용이 아니라 추론 대상**이다. 성능 수치는 전부 위 내부 분할에서 나온다.

## 2. 누수 방지 규칙

| 규칙 | 구현 |
|---|---|
| 분할을 **먼저** 하고 전처리한다 | `fit_and_compare()`에서 split 후 Pipeline 학습 |
| 전처리기는 Train에만 `fit` | 결측 대치·인코딩·스케일링이 전부 `Pipeline` 내부 단계 |
| Validation/Test에는 `transform`만 | `pipe.predict_proba(X_val)` 호출로 자동 보장 |
| 식별자 제외 | `customer_id` 드롭 (`MusicFeatureEngineer`) |
| 사후 정보 제외 | 이탈 사유·해지일·점수 컬럼이 데이터에 **없음**을 확인 |
| Test로 모델·threshold 선택 금지 | threshold는 `choose_operating_threshold(y_val, val_prob)`로 **Validation에서만** 결정 |

- SMOTE·class_weight 오버샘플링은 **사용하지 않는다**. 이탈률 51.3%로 균형 데이터이기 때문이다.
- ⚠️ 잔여 위험: `weekly_hours` 등 이용 지표가 이탈 **직전** 값이라면 사후 피처일 수 있다. 스냅샷 정의가 없어 검증 불가하며, 이를 §4의 한계로 명시한다.

## 3. 평가 지표

| 지표 | 우선순위 | 이유 |
|---|---|---|
| **Recall (이탈)** | 1순위 | FN = 떠날 고객을 놓침 = LTV 전액 손실 |
| **PR-AUC** | 2순위 | 임계값 무관한 확률 품질 |
| Precision / F1 | 보조 | 캠페인 낭비 통제 |
| Brier | 보조 | 확률 보정 품질 |
| Confusion Matrix | 필수 첨부 | 놓친 이탈자·헛경고 실수를 절대수로 확인 |
| Accuracy | **단독 사용 금지** | 가이드 §7 금지항목 |

기준 모델(`DummyClassifier(strategy="prior")`)을 반드시 비교표에 포함한다. 현재 Dummy PR-AUC = 0.5134.

## 4. 임계값 결정

1. Validation 확률에 대해 threshold를 0.05 ~ 0.95 (91개 격자)로 스윕한다.
2. 각 지점의 기대비용 `(3 × FN + 1 × FP) / N`을 계산한다. **FN:FP = 3:1**
3. 기대비용 최소 → 동률 시 Recall → Precision 순으로 선택한다.
4. 선택된 threshold를 **Test에 고정 적용**한다. Test를 보고 재조정하지 않는다.

현재 운영 threshold = **0.30**. 전체 스윕 결과는 [`artifacts/threshold_sweep_validation.csv`](../artifacts/threshold_sweep_validation.csv)에 있다.

⚠️ FN:FP = 3:1은 **팀 가정값**이며 사업부 승인 전이다. 비용비가 바뀌면 threshold도 재계산해야 한다.

### 캠페인 관점 보조 지표

이탈률이 51.3%라 Lift 상한이 구조적으로 **1.95배**(=1/0.513)로 고정된다. 따라서 상위 K% 타겟팅 효율을 함께 본다.

| 상위 | 10% | 20% | 30% | 50% |
|---|---|---|---|---|
| 이탈자 포착률 | 19.5% | 39.0% | 57.1% | 84.2% |

## 5. 모델 후보와 확장 중단 근거

후보 7종을 **동일 split·동일 지표**로 비교한다: Dummy → Logistic(raw/engineered) → DecisionTree → RandomForest → ExtraTrees → GradientBoosting.

**딥러닝은 의도적으로 제외한다.** 가이드 §⑥은 "데이터에 맞는 이유가 있을 때" 추가하라고 명시하는데, 근거는 다음과 같다.

| 모델 | Test PR-AUC |
|---|---:|
| 7버킷 가법 로지스틱 (파라미터 15개) | 0.9468 |
| GradientBoosting (신호 7개) | 0.9464 |
| GradientBoosting (전체 18개, 현행) | 0.9473 |
| 포화 셀 모델 (이론 상한) | 0.9491 |

파라미터 15개짜리 로지스틱이 GB와 **0.0005 차이**다. 데이터가 가법 로짓 규칙으로 생성되어 이론 상한 정확도가 **86.1%**인데 현행 모델이 이미 84.8%에 도달했다. 남은 1.3%p는 Bernoulli 난수라 **어떤 알고리즘으로도 회수 불가능**하다.

→ MLP·앙상블·추가 튜닝은 기대 이득이 0이므로 착수하지 않는다. 상세 근거: [../artifacts/eda_insight/INSIGHT_REPORT.md](../artifacts/eda_insight/INSIGHT_REPORT.md) §2 차트 6, §5 한계 ①.

## 6. 재현 절차

```bash
python -m src.run_pipeline          # EDA·모델 비교·artifact 전량 재생성
python scripts/eda_insight.py       # 인사이트 차트 7종 + 생성규칙 역산
python scripts/business_levers.py   # 레버 효과·세그먼트·손익분기
```

`random_state=42` 고정. 입력 CSV의 SHA-256을 [`artifacts/model/metadata.json`](../artifacts/model/metadata.json)의 `source_files`에 기록하므로 데이터 버전 불일치를 감지할 수 있다.
