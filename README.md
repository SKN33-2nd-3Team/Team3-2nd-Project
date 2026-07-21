# music — PlaylistPro Insight

음악 스트리밍 서비스의 고객 이탈 위험을 분석하고, 유지 활동 대상 고객을 식별하는 머신러닝 프로젝트입니다.

> **문서 기준일 2026-07-21.** 모든 수치는 신규 데이터셋(train 125,000행) 기준이며,
> `python -m src.run_pipeline` 재실행으로 재현됩니다.

현재 상태는 **분석·모델링 완료 / 사업부 승인 대기**입니다. 종료 전 다음 항목에 승인이 필요합니다.

- FN:FP 비용비 `3:1`과 운영 threshold `0.30`을 캠페인 기준으로 사용할지
- 캠페인 손익 계산에 쓴 LTV 가정(월 ARPU 10,000원 × 12개월)을 실제 사업 수치로 대체할지

## 1. 빠른 실행

```bash
pip install -r requirements.txt

# 재현 가능한 EDA·모델 비교·artifact 생성
python -m src.run_pipeline

# 저장된 Pipeline을 로드하는 사용자 화면
python -m streamlit run app/streamlit_app.py --server.port 8501
```

브라우저에서 `http://localhost:8501`을 엽니다. Streamlit은 재학습하지 않고 `artifacts/model/music_churn_pipeline.joblib`만 로드합니다.

## 2. 제품 화면

- **Overview**: 데이터 계약, 클래스 비율, 현재 기술 추천과 주요 관찰점
- **Data Explorer**: 모델 입력 컬럼별 target 관계 그래프와 요약표
- **Model Lab**: 후보 모델 비교, PR-AUC, Recall/Precision/비용 threshold trade-off, feature importance
- **Customer Scoring**: 단일 고객 입력에 대한 이탈 확률과 위험 등급
- **Batch Prioritization**: 라벨 없는 `test.csv` 고객의 우선순위 목록과 CSV 다운로드
- **Campaign Simulator**: 단가·성공률 시나리오별 캠페인 손익 곡선

## 3. 데이터 계약

| 항목 | 내용 |
|---|---|
| 학습 데이터 | `data/train.csv`, **125,000행 × 20열**, `churned` 포함 |
| 추론 데이터 | `data/test.csv`, **75,000행 × 19열**, `churned` 없음 |
| 단위 | 고객 1행 (`customer_id` 고유값 = 행 수) |
| target | `churned` 0 = 유지 / 1 = 이탈 |
| 양성 비율 | 64,174 / 125,000 = **51.34%** (균형 데이터) |
| ID 처리 | `customer_id`는 모델에서 제외 |
| 누수 점검 | train/test `customer_id` 교집합 **0건**, 결측 0건, 중복 0건 |
| 원본 보존 | 원본 CSV를 수정하지 않음 |
| 출처 | [Streaming Subscription Churn Model — Kaggle Competition](https://www.kaggle.com/competitions/streaming-subscription-churn-model/data) · 라이선스 MIT · 2026-07-21 다운로드 |

제공 `test.csv`는 정답이 없으므로 최종 평가용이 아니라 **추론 대상**입니다. 성능 수치는 `train.csv`를 60/20/20으로 나눈 내부 train/validation/test에서 계산했습니다.

상세: [docs/data_dictionary.md](docs/data_dictionary.md) · [docs/data_card.md](docs/data_card.md)

## 4. 모델링 결과

FN 비용을 FP보다 높게 보는 요구사항을 반영해 `FN:FP = 3:1` 비용함수를 사용하고, validation expected cost가 최소인 threshold를 선택했습니다. 선정 모델은 **gradient_boosting**, 운영 threshold는 **0.30**입니다.

| 후보 | Val PR-AUC | Test PR-AUC | Test Recall* | Test Precision* |
|---|---:|---:|---:|---:|
| **gradient_boosting (선정)** | **0.9446** | **0.9473** | 96.2% | 76.8% |
| random_forest | 0.9376 | 0.9409 | 94.4% | 77.2% |
| decision_tree | 0.9259 | 0.9314 | 95.2% | 73.4% |
| extra_trees | 0.9119 | 0.9162 | 94.7% | 71.7% |
| logistic_engineered | 0.8968 | 0.9013 | 94.4% | 69.9% |
| dummy_prior (기준선) | 0.5134 | 0.5134 | — | — |

<sub>*운영 threshold 0.30 기준. 전체 지표는 [artifacts/model_comparison.csv](artifacts/model_comparison.csv)</sub>

> ⚠️ **이 점수를 실제 서비스 성능으로 읽으면 안 됩니다.** §6 제한사항 1·2번 참조.

## 5. 핵심 인사이트

### 데이터 차원

- **18개 피처 중 7개만 신호이고 11개는 완전 무신호**입니다. 이탈률 스프레드 0.10을 경계로 중간지대 없이 갈립니다.
- 신호 7개: `weekly_hours`(0.620) · `subscription_type`(0.455) · `customer_service_inquiries`(0.454) · `num_subscription_pauses`(0.228) · `song_skip_rate`(0.226) · `age`(0.225) · `notifications_clicked`(0.149)
- **`payment_plan`(월납/연납)은 스프레드 0.001로 완전 무신호**입니다. "연납 전환으로 락인"이라는 통념은 이 데이터에서 근거가 없습니다.
- 파라미터 15개짜리 로지스틱 회귀가 Gradient Boosting과 **PR-AUC 0.0005 차이**입니다. 복잡한 모델이 이득을 주지 않습니다.

### 비즈니스 차원

- **가장 강한 "신호"와 가장 큰 "레버"는 다릅니다.** 중요도 1위는 청취시간이지만, 총 이탈방지 효과는 상담 경험 개선(18,758명) > 유료 전환(14,058명) > … > 청취 회복(2,089명) 순입니다. 대상 인구 규모가 다르기 때문입니다.
- **최고위험군을 공략하면 안 됩니다.** 동일 레버가 위험도 85~95% 구간에서 81.4%p를 낮추지만 95%+ 구간에서는 32.2%p에 그칩니다(시그모이드 포화). 1인당 효율이 **2.5배** 차이납니다.
- 이탈률 51.3%라 **Lift 상한이 구조적으로 1.95배**입니다. 상위 10%는 정밀도 100%지만 이탈자의 19.5%만 포착합니다. 소수 정예 타겟팅이 통하지 않습니다.

전체 분석: [artifacts/eda_insight/INSIGHT_REPORT.md](artifacts/eda_insight/INSIGHT_REPORT.md)

> 위 관계는 모두 **연관성**입니다. Feature importance를 이탈의 원인으로 단정하지 않으며, 레버 효과는 A/B 테스트 전까지 상한선 추정치입니다.

## 6. 중요한 제한사항

1. 🔴 **규칙 기반 합성 데이터입니다.** 범주 균등 분포, 피처 간 최대 상관 0.007, 정수 문턱값(5/10/40h·0.7·3회·25/35/61세), 그리고 로지스틱 계수가 0.5 단위로 복원되는 점이 근거입니다. **PR-AUC 0.947은 심어진 생성 공식을 되찾은 값이며 실제 이탈 예측 성능이 아닙니다** (실무 벤치마크는 통상 0.3~0.6대).
2. **이론 성능 상한이 존재합니다.** Bernoulli 난수 때문에 최대 달성 가능 정확도가 86.1%이고 현행 모델이 84.8%입니다. 남은 1.3%p는 어떤 알고리즘으로도 회수할 수 없어 추가 튜닝·딥러닝을 착수하지 않았습니다.
3. **예측 시점이 정의되지 않았습니다.** 관측 기준일·해지일이 없어 "향후 30일 내 해지" target을 검증하지 못했습니다. 화면·문서에서 시간 지평을 주장하지 않습니다.
4. **물리적으로 불가능한 레코드가 25~30% 있습니다.** 고유곡 > 총재생곡 29.6%, 공유 > 생성 플레이리스트 24.6%. 해당 컬럼은 모두 노이즈로 판정되어 모델에서 제외되지만, 이 데이터로 청취 패턴 2차 분석을 해서는 안 됩니다.
5. **대회 컬럼 설명과 실제 값이 3곳 어긋납니다.** `num_subscription_pauses`(문서 max 2 / 실제 0~4), `signup_date`(문서 date / 실제 음수 정수), `average_session_length`(문서 시간 / 실제 단위 미상). 컬럼 의미를 단정하지 않고 실측값 기준으로 처리했습니다. [docs/data_card.md](docs/data_card.md) 참조
6. 외부 캠페인 자동 실행·고객 차단·자동 해지 통지는 범위 밖입니다. 화면 결과는 검토 우선순위 지원용입니다.

## 7. 산출물 위치

| 산출물 | 경로 |
|---|---|
| 요구사항 정의 | [docs/requirements.md](docs/requirements.md) |
| 데이터 사전 | [docs/data_dictionary.md](docs/data_dictionary.md) |
| Data Card | [docs/data_card.md](docs/data_card.md) |
| 검증 계획 | [docs/validation_plan.md](docs/validation_plan.md) |
| 전처리·학습 결과서 | [전처리_모델링_상세결과지_20260721.md](전처리_모델링_상세결과지_20260721.md) |
| 인사이트 결과서 | [artifacts/eda_insight/INSIGHT_REPORT.md](artifacts/eda_insight/INSIGHT_REPORT.md) |
| 모델 비교 | [artifacts/model_comparison.csv](artifacts/model_comparison.csv) |
| 모델 보고서 | [artifacts/model_report.md](artifacts/model_report.md) |
| 모델 해석 | [artifacts/feature_importance.csv](artifacts/feature_importance.csv), `artifacts/eda/feature_importance.png` |
| 저장 Pipeline | `artifacts/model/music_churn_pipeline.joblib` |
| 모델 메타데이터 | [artifacts/model/metadata.json](artifacts/model/metadata.json) |
| test.csv 추론 결과 | [artifacts/test_predictions.csv](artifacts/test_predictions.csv) |
| EDA 차트 (자동 생성) | `artifacts/eda/` |
| 인사이트 차트 7종 | `artifacts/eda_insight/` |

### 재현 스크립트

```bash
python -m src.run_pipeline          # EDA·모델 비교·artifact 전량
python scripts/eda_insight.py       # 인사이트 차트 7종 + 생성규칙 역산
python scripts/business_levers.py   # 레버 효과·세그먼트·손익분기
```

## 8. 협업 및 Git 설정

- [GitHub 협업 규칙](CONTRIBUTING.md): 브랜치, 커밋, Issue, PR, 리뷰 규칙
- [Git·GitHub 최초 설정 가이드](docs/GIT_SETUP.md): Clone, 사용자 정보, 작업 브랜치, Push, 저장소 관리자 설정
- Pull Request와 Issue 작성 시 `.github/`의 템플릿을 사용합니다.

기본 원칙은 `main`에 직접 push하지 않고 `feature/*`, `fix/*`, `docs/*` 작업 브랜치에서 변경한 뒤 Pull Request로 병합하는 것입니다.
