# PlaylistPro Insight

> 음악 스트리밍 가입 고객의 이탈 가능성을 예측하고, 리텐션 담당자의 고객 유지 활동 우선순위 결정을 지원하는 머신러닝 프로젝트

[![Python](https://img.shields.io/badge/Python-3.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)

문서 기준일은 **2026-07-21**이며, 모든 수치와 산출물은 `data/train.csv` 125,000행을 기준으로 작성했습니다.

## 1. 프로젝트 개요

### 문제 정의

> **리텐션·CRM 담당자**가 **어떤 고객에게 어떤 유지 활동을 배정할지** 결정할 수 있도록, 고객별 구독·이용·상담 프로필로 **이탈 여부(`churned`)**를 예측하고 위험 등급별 유지 활동을 제안합니다.

- Target: `churned` — `0`은 유지, `1`은 이탈
- 분석 단위: 고객 1명 = 1행
- 우선 지표: 이탈 Recall, PR-AUC
- 오류 비용: FN:FP = `3:1` 가정
- 운영 임계값: Validation 기대비용 최소점인 `0.30`
- 활용 범위: 캠페인 검토 우선순위 지원. 자동 실행·고객 차단은 범위 밖

데이터에 관측 기준일과 해지일이 없어 예측 시점·관찰 기간·결과 기간은 정의할 수 없습니다. 따라서 이 프로젝트는 **스냅샷 프로필로 제공 라벨을 판별하는 검증 프로젝트**이며, “향후 30일 내 이탈”처럼 시간 범위를 확장해 해석하지 않습니다.

```mermaid
flowchart LR
    A["비즈니스 문제 정의"] --> B["데이터 점검·EDA"]
    B --> C["분할 후 Pipeline 전처리"]
    C --> D["동일 조건 모델 비교"]
    D --> E["Validation 임계값 결정"]
    E --> F["Test 최종 평가"]
    F --> G["Pipeline 저장·재로딩"]
    G --> H["Streamlit 예측·유지 활동"]
```

## 2. 주요 기능

| 화면 | 제공 기능 |
|---|---|
| Overview | 데이터 계약, 클래스 비율, 핵심 관찰점, 위험 고객 현황 |
| Data Explorer | 주요 변수 분포와 Target별 차이, 데이터 품질 점검 |
| Model Lab | 후보 모델 성능, PR-AUC, 임계값 trade-off, Feature Importance |
| Customer Scoring | 신규 고객 1명의 실제 이탈 확률·위험 등급·유지 활동 |
| Batch Prioritization | `test.csv` 75,000명 일괄 예측, 고위험 고객 CSV 다운로드 |
| Campaign Simulator | 단가·성공률·대상 비율에 따른 캠페인 손익 비교 |

Streamlit은 실행 시 모델을 다시 학습하지 않고 저장된 `artifacts/model/music_churn_pipeline.joblib`을 불러옵니다. 입력값을 바꾸면 같은 Pipeline의 `predict_proba` 결과가 변경됩니다.

## 3. 빠른 실행

프로젝트 루트에서 다음 명령만 실행합니다.

```bash
pip install -r requirements.txt
python -m src.run_pipeline
streamlit run app/streamlit_app.py
```

브라우저에서 `http://localhost:8501`을 엽니다. 이미 생성된 모델과 산출물을 사용할 때는 두 번째 명령을 생략할 수 있습니다.

## 4. 프로젝트 구조

```text
.
├── README.md
├── requirements.txt
├── app/
│   └── streamlit_app.py
├── data/
│   ├── train.csv
│   └── test.csv
├── docs/
│   ├── README.md
│   ├── requirements.md
│   ├── data_card.md
│   ├── data_dictionary.md
│   ├── validation_plan.md
│   └── submission_checklist.md
├── reports/
│   ├── preprocessing_report.md
│   └── training_report.md
├── src/
│   ├── features.py
│   └── run_pipeline.py
├── scripts/
│   ├── eda_insight.py
│   └── business_levers.py
└── artifacts/
    ├── eda/                       # 전처리·EDA 이미지
    ├── eda_insight/               # 핵심 인사이트 이미지와 보고서
    ├── model/
    │   ├── music_churn_pipeline.joblib
    │   └── metadata.json
    ├── model_comparison.csv
    ├── threshold_sweep_validation.csv
    ├── feature_importance.csv
    └── test_predictions.csv
```

공지의 권장 구조와 이름이 다른 실행 파일은 위 실제 경로를 기준으로 사용합니다. 원본 CSV는 수정하지 않으며, 모든 코드에서 프로젝트 루트 기준 상대경로를 사용합니다.

## 5. 데이터 및 전처리

| 항목 | 내용 |
|---|---|
| 출처 | [Kaggle — Streaming Subscription Churn Model](https://www.kaggle.com/competitions/streaming-subscription-churn-model/data) |
| 라이선스·다운로드 | MIT 표기, 2026-07-21 다운로드 |
| 학습 데이터 | `data/train.csv`, 125,000행 × 20열 |
| 추론 데이터 | `data/test.csv`, 75,000행 × 19열, 정답 없음 |
| Target 분포 | 이탈 64,174명(51.34%), 유지 60,826명(48.66%) |
| 품질 점검 | 결측 0건, 중복 0건, train/test 고객 ID 교집합 0건 |
| 분할 | Train 60% / Validation 20% / Test 20%, `stratify=y`, `random_state=42` |

전처리는 `MusicFeatureEngineer → ColumnTransformer → 모델` 순서의 단일 scikit-learn Pipeline으로 구성했습니다.

- `customer_id`와 원본 `signup_date` 제거
- 수치형: median 대치, 로지스틱 회귀에만 표준화
- 범주형: 최빈값 대치 후 `OneHotEncoder(handle_unknown="ignore")`
- 파생 Feature: `signup_days_ago`, `unique_song_ratio`, `shared_playlist_ratio`, `hours_per_song`, `friends_per_playlist`
- 분할을 먼저 수행하고 전처리기는 Train에만 `fit`
- SMOTE는 사용하지 않았습니다. 최종 Gradient Boosting에는 Class Weight를 적용하지 않았지만 일부 비교 후보에는 balanced 계열 Class Weight를 적용했습니다.

자세한 근거와 결과는 [전처리 결과서](reports/preprocessing_report.md), [Data Card](docs/data_card.md), [데이터 사전](docs/data_dictionary.md)에서 확인할 수 있습니다.

## 6. 모델 학습 및 평가

모든 후보를 동일한 split과 지표로 비교했습니다. 각 후보의 임계값은 Validation에서 결정했으며, 고정된 임계값으로 각 모델을 Test에서 한 번씩 평가했습니다. Test 결과는 모델이나 임계값 선택에 사용하지 않았습니다.

| 후보 | Val PR-AUC | Test PR-AUC | Test Recall* | Test Precision* |
|---|---:|---:|---:|---:|
| **Gradient Boosting (최종)** | **0.9446** | **0.9473** | **96.2%** | **76.8%** |
| Random Forest | 0.9376 | 0.9409 | 94.4% | 77.2% |
| Decision Tree | 0.9259 | 0.9314 | 95.2% | 73.4% |
| Extra Trees | 0.9119 | 0.9162 | 94.7% | 71.7% |
| Logistic Regression | 0.8968 | 0.9013 | 94.4% | 69.9% |
| Dummy prior | 0.5134 | 0.5134 | 100.0% | 51.3% |

<sub>*각 모델의 Validation 비용 최소 임계값을 Test에 고정 적용한 결과입니다.</sub>

최종 모델의 운영 임계값은 `0.30`이며 Test 혼동행렬은 TN 8,435 / FP 3,730 / FN 492 / TP 12,343입니다. 모델 선정 근거, 하이퍼파라미터, 현재 오류 현황은 [모델 학습 결과서](reports/training_report.md)와 [검증 계획서](docs/validation_plan.md)에 정리했습니다. FP·FN 사례 분석은 팀원 협의 후 추가할 예정입니다.

딥러닝은 의도적으로 제외했습니다. 15개 파라미터의 가법 로지스틱 모델이 Test PR-AUC 0.9468로 최종 모델 0.9473과 사실상 동률이고, 합성 생성 규칙의 이론 상한에 이미 근접해 추가 복잡도의 근거가 없기 때문입니다.

## 7. 핵심 인사이트와 유지 활동

- 18개 원천 Feature 중 강한 신호는 7개이며 `weekly_hours`, `subscription_type`, `customer_service_inquiries`의 영향이 가장 큽니다.
- `payment_plan`은 월납·연납 간 이탈률 차이가 0.001로, 연납 전환을 유지 전략으로 제안할 근거가 없습니다.
- 상담 경험 개선, 유료 요금제 전환, 일시정지 고객 케어, 높은 스킵률 고객의 추천 개선을 우선 실험 대상으로 제안합니다.
- Feature Importance는 인과관계가 아닙니다. 유지 활동 효과는 실제 A/B 테스트로 검증해야 합니다.

전체 분석은 [인사이트 보고서](artifacts/eda_insight/INSIGHT_REPORT.md)에서 확인할 수 있습니다.

## 8. 제한사항

1. **규칙 기반 합성 데이터로 판정했습니다.** Test PR-AUC 0.9473은 생성 규칙 복원 성능이며 실제 서비스 일반화 성능이 아닙니다.
2. 관측 기준일·해지일이 없어 예측 시점과 결과 기간을 검증할 수 없습니다.
3. `weekly_unique_songs > weekly_songs_played`가 29.6%, `num_shared_playlists > num_playlists_created`가 24.6%입니다. 현행 Pipeline은 이 컬럼들을 입력으로 사용하지만 중요도가 거의 없으므로, 향후 제거 실험과 입력 화면 축소가 필요합니다.
4. FN:FP `3:1`과 요금제별 고객 가치·접촉 비용은 **전부 팀 가정**입니다. 데이터에 금액 컬럼이 하나도 없어 국내 시세를 참조했습니다. 산출 근거와 민감도는 [docs/campaign_assumptions.md](docs/campaign_assumptions.md)에 있으며, 실제 사업 단가로 다시 승인·산정해야 합니다.
5. 모델 관계는 연관성만 보여주며 캠페인 효과를 보장하지 않습니다.

## 9. 제출 산출물

| 공지 필수 산출물 | 저장소 파일 |
|---|---|
| 데이터 전처리 결과서 | [reports/preprocessing_report.md](reports/preprocessing_report.md) |
| 인공지능 모델 학습 결과서 | [reports/training_report.md](reports/training_report.md) |
| 학습된 최종 모델 | `artifacts/model/music_churn_pipeline.joblib` |
| 모델 보조 파일 | [artifacts/model/metadata.json](artifacts/model/metadata.json), [artifacts/model_comparison.csv](artifacts/model_comparison.csv) |
| Streamlit 시연 | `app/streamlit_app.py` |
| 요구사항·데이터·검증 문서 | [docs/README.md](docs/README.md) |
| 제출 전 점검표 | [docs/submission_checklist.md](docs/submission_checklist.md) |

Google Drive 제출용 PDF와 프로젝트 폴더 ZIP은 저장소 밖에서 별도로 생성합니다. 저장소에는 원본 Markdown과 재현 코드·이미지·모델을 보관합니다.

## 10. 재현 및 협업

```bash
python -m src.run_pipeline
python scripts/eda_insight.py
python scripts/business_levers.py
```

- [협업 규칙](CONTRIBUTING.md)
- [Git·GitHub 설정 가이드](docs/GIT_SETUP.md)

`main`에 직접 push하지 않고 기능 브랜치에서 작업한 뒤 Pull Request로 병합합니다.
