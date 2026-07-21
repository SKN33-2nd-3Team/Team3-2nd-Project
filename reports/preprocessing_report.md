# 데이터 전처리 결과서

> 작성일: 2026-07-21  
> 실행 코드: [`src/run_pipeline.py`](../src/run_pipeline.py), [`src/features.py`](../src/features.py)

## 1. 데이터 소개

| 항목 | 내용 |
|---|---|
| 데이터 이름 | Streaming Subscription Churn Model |
| 출처 | [Kaggle Competition](https://www.kaggle.com/competitions/streaming-subscription-churn-model/data) |
| 라이선스 | Kaggle Metadata의 MIT 표기. 다운로드에는 대회 규칙 동의 필요 |
| 다운로드 날짜 | 2026-07-21 |
| 실제·합성 여부 | 대회 명시 없음. 팀 분석 결과 규칙 기반 합성으로 판정 |
| 학습 데이터 | `data/train.csv`, 125,000행 × 20열 |
| 추론 데이터 | `data/test.csv`, 75,000행 × 19열 |
| 분석 단위 | 고객 1명 = 1행 |
| Target | `churned`: 0 = 유지, 1 = 이탈 |

파일 해시, 합성 판정 근거, 개인정보 검토는 [Data Card](../docs/data_card.md)에 기록했습니다.

## 2. 분석 기준

- 학습 Feature: Target과 식별자를 제외한 고객 구독·이용·상담 프로필
- 제공 `test.csv`: 정답이 없어 성능 평가가 아닌 최종 추론에만 사용
- 내부 분할: Train 75,000 / Validation 25,000 / Test 25,000
- Target 비율: 이탈 64,174명(51.34%), 유지 60,826명(48.66%)
- 예측 시점·관찰 기간·결과 기간: 원천 데이터에 기준일·해지일이 없어 정의 불가

## 3. 품질 점검

| 검사 | Train | Test | 처리·판단 |
|---|---:|---:|---|
| 결측값 | 0건 | 0건 | Pipeline 대치기는 신규 입력 안전장치로 유지 |
| 완전 중복행 | 0건 | 0건 | 제거 없음 |
| 고객 ID 중복 | 0건 | 0건 | 고객 1명 = 1행 확인 |
| train/test ID 교집합 | 0건 | — | 고객 중복 누수 없음 |
| `weekly_unique_songs > weekly_songs_played` | 36,996건(29.6%) | 별도 점검 | 물리적 불일치, 해석 제한 |
| `num_shared_playlists > num_playlists_created` | 30,778건(24.6%) | 별도 점검 | 물리적 불일치, 해석 제한 |

공식 컬럼 설명과 실제 값이 다른 항목은 다음과 같습니다.

| 컬럼 | 문서 설명 | 실제 값 | 처리 |
|---|---|---|---|
| `num_subscription_pauses` | 최대 2회 | 0~4회 | 실제 범위를 허용 |
| `signup_date` | 날짜 | −2,922~-1 정수 | `signup_days_ago = -signup_date`로 변환 후 원본 제거 |
| `average_session_length` | 시간 | 1~120, 단위 불명 | 수치로 입력하되 해석하지 않음 |

## 4. EDA 결과

### Target 분포

![Target 분포](../artifacts/eda/target_distribution.png)

이탈률은 51.34%로 클래스 불균형이 크지 않습니다. 따라서 SMOTE를 적용하지 않고 동일한 클래스 비율을 유지하도록 층화 분할했습니다.

### 주요 Feature

![신호와 노이즈 비교](../artifacts/eda_insight/03_signal_vs_noise.png)

18개 원천 Feature를 10분위 또는 범주 수준별 이탈률 스프레드로 비교한 결과, 0.10을 경계로 강한 신호 7개와 약한 신호 11개가 구분되었습니다.

| 강한 신호 | 이탈률 스프레드 | 관찰 결과와 처리 결정 |
|---|---:|---|
| `weekly_hours` | 0.620 | 5·10·40시간 부근의 계단형 관계. 비선형 모델 비교 |
| `subscription_type` | 0.455 | Free·Student에서 높은 이탈률. 원-핫 인코딩 |
| `customer_service_inquiries` | 0.454 | High 등급에서 높은 이탈률. 원-핫 인코딩 |
| `num_subscription_pauses` | 0.228 | 3회 이상에서 증가. 수치형 유지 |
| `song_skip_rate` | 0.226 | 0.7 초과에서 증가. 수치형 유지 |
| `age` | 0.225 | U자형 관계. 트리 모델과 선형 기준선 비교 |
| `notifications_clicked` | 0.149 | 5회 부근 문턱. 수치형 유지 |

관찰된 관계는 합성 생성 규칙의 연관성일 수 있으므로 인과관계로 해석하지 않습니다.

### 핵심 시각화

- [EDA 자동 생성 이미지 21종과 모델 해석 이미지 1종](../artifacts/eda/README.md)
- [위험 깃발 누적](../artifacts/eda_insight/01_risk_flag_ladder.png)
- [청취시간 계단형 관계](../artifacts/eda_insight/02_weekly_hours_steps.png)
- [요금제×상담문의](../artifacts/eda_insight/04_plan_x_inquiry_heatmap.png)
- [연령 U자형 관계](../artifacts/eda_insight/05_age_u_curve.png)
- [모델 성능 상한](../artifacts/eda_insight/06_model_ceiling.png)
- [상위 위험군 포착률](../artifacts/eda_insight/07_capture_curve.png)

## 5. 데이터 분할과 누수 방지

1. 원본 `train.csv`를 먼저 Train 60% / Validation 20% / Test 20%로 분리했습니다.
2. `train_test_split(..., stratify=y, random_state=42)`를 두 번 적용했습니다.
3. 피처 생성·결측 대치·인코딩·스케일링은 Pipeline 내부에서 Train에만 `fit`했습니다.
4. Validation은 모델·임계값 선택에만, Test는 최종 평가에 한 번만 사용했습니다.
5. Target, `customer_id`, 검증할 수 없는 원본 `signup_date`는 모델 입력에서 제외했습니다.
6. 이탈 사유·해지일·사후 점수 컬럼이 원천 데이터에 없음을 확인했습니다.

상세 규칙은 [검증 계획서](../docs/validation_plan.md)를 참조합니다.

## 6. 전처리 Pipeline

```text
입력 DataFrame
→ MusicFeatureEngineer
→ ColumnTransformer
   ├─ 수치형: median 대치, Logistic Regression만 StandardScaler
   └─ 범주형: 최빈값 대치, OneHotEncoder(handle_unknown="ignore")
→ 분류 모델
```

### 제거·생성 Feature

| 구분 | Feature | 근거 |
|---|---|---|
| 제거 | `customer_id` | 식별자 |
| 제거 | 원본 `signup_date` | 예측 스냅샷 미정의 |
| 생성 | `signup_days_ago` | 음수 상대일을 양수 경과일로 변환 |
| 생성 | `unique_song_ratio` | `weekly_unique_songs / (weekly_songs_played + 1)` |
| 생성 | `shared_playlist_ratio` | `num_shared_playlists / (num_playlists_created + 1)` |
| 생성 | `hours_per_song` | `weekly_hours / (weekly_songs_played + 1)` |
| 생성 | `friends_per_playlist` | `num_platform_friends / (num_playlists_created + 1)` |

최종 변환 차원은 수치형 17개와 범주형 더미 32개를 합친 49개입니다. 전처리기와 모델은 하나의 Pipeline으로 저장해 학습과 Streamlit 추론의 Feature 이름·순서·자료형을 일치시켰습니다.

## 7. 전처리 결과

| 항목 | 결과 |
|---|---|
| 학습 가능 행 | 125,000행 전량 |
| 제거 행 | 0행 |
| 모델 원천 입력 | `customer_id` 포함 19개 입력 컬럼; Pipeline 내부에서 ID·원본 날짜 제거 |
| 변환 후 Feature | 49개 |
| 불균형 처리 | 미적용, 양성 비율 51.34% |
| 저장 방식 | 전처리기+모델 통합 joblib Pipeline |

생성 산출물은 `artifacts/eda_summary.json`, `artifacts/eda/`, `artifacts/eda_insight/`에 저장됩니다.

## 8. 한계와 개선 방향

1. 규칙 기반 합성 데이터이므로 EDA 관계와 높은 성능을 실제 서비스에 일반화할 수 없습니다.
2. 시간 기준이 없어 미래 정보 누수를 완전히 검증할 수 없습니다.
3. 논리 위반 컬럼도 현행 Pipeline 입력에 포함됩니다. 중요도가 거의 없으므로 다음 버전에서는 제거 실험 후 입력 스키마를 7개 핵심 Feature 중심으로 축소할 수 있습니다.
4. 파생 Feature 5개 중 일부는 노이즈 원천 컬럼 조합이며, 제거 시 성능 차이가 거의 없었습니다.
5. 데이터 수집 단계에서 이벤트 타임스탬프와 실제 해지일을 확보하면 시간 분할로 재검증해야 합니다.
