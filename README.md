# music — PlaylistPro Insight

음악 스트리밍 서비스의 고객 이탈 위험을 분석하고, 우선 검토 고객을 식별하는 Light-mode 머신러닝 프로젝트입니다.

현재 상태는 **1차 구현 완료 / 최종 사용자 승인 대기**입니다. 저장된 모델과 Streamlit은 동작하지만, 최종 종료 전 다음 항목에 대한 사용자 승인이 필요합니다.

- `churned`가 실제로 “현재 시점 기준 향후 30일 내 해지”를 의미하는지
- FN:FP 비용비 `3:1`과 threshold `0.37`을 운영 기준으로 사용할지
- 데이터 버전·출처 불일치와 `num_subscription_pauses`의 의미를 허용할지

## 1. 빠른 실행

# 재현 가능한 EDA·모델 비교·artifact 생성
python -m src.run_pipeline

# 저장된 Pipeline을 로드하는 사용자 화면
python -m streamlit run app/streamlit_app.py --server.port 8501
```

브라우저에서 `http://localhost:8501`을 열면 됩니다. Streamlit은 재학습하지 않고 `artifacts/model/music_churn_pipeline.joblib`만 로드합니다.

## 2. 제품 화면

- **Overview**: 데이터 계약, 클래스 비율, 현재 기술 추천과 주요 관찰점
- **Data Explorer**: 모든 모델 입력 컬럼별 target 관계 그래프와 요약표
- **Model Lab**: 후보 모델 비교, PR-AUC, Recall/Precision/비용 threshold trade-off, feature importance
- **Customer Scoring**: 단일 고객 입력에 대한 이탈 확률과 우선 검토 판단
- **Batch Prioritization**: 라벨이 없는 제공 `test.csv` 고객의 우선순위 목록과 CSV 다운로드

## 3. 현재 데이터 계약

| 항목 | 내용 |
|---|---|
| 학습 데이터 | `data/train.csv`, 41,000행, `churned` 포함 |
| 추론 데이터 | `data/test.csv`, 9,999행, `churned` 없음 |
| 단위 | 고객 1행 |
| target | `churned` 0/1 |
| 양성 비율 | 8,747 / 41,000 = 21.33% |
| ID 처리 | `customer_id`는 모델에서 제외 |
| 원본 보존 | 원본 CSV를 수정하지 않음 |

제공 `test.csv`는 정답이 없으므로 최종 성능 평가용 test가 아니라 inference 대상입니다. 성능 수치는 `train.csv`를 60/20/20으로 나눈 내부 train/validation/test에서 계산했습니다.

## 4. 모델링 결과 — provisional

FN 비용을 FP보다 높게 보는 요구사항을 반영해 임시 `FN:FP = 3:1` 비용함수를 사용하고, validation expected cost가 가장 낮은 threshold를 선택했습니다.

| 후보 | Test PR-AUC | Test Recall | Test Precision | Test cost/customer |
|---|---:|---:|---:|---:|
| gradient_boosting | 0.3631 | 0.3819 | 0.4985 | 0.4774 |
| random_forest | 0.3670 | 0.4111 | 0.4436 | 0.4868 |
| decision_tree | 0.3642 | 0.3751 | 0.4870 | 0.4841 |
| logistic_engineered | 0.3463 | 0.4437 | 0.3740 | 0.5144 |

기술적 provisional 추천은 `gradient_boosting`, validation threshold는 `0.37`입니다. 다만 `random_forest`의 test PR-AUC가 더 높으므로, 모델 선택은 단일 지표가 아니라 FN 비용·캠페인 처리량·운영 threshold를 함께 승인해야 합니다.

## 5. 현재까지의 데이터 인사이트

- `payment_plan`은 가장 뚜렷한 관찰 신호입니다. monthly 고객의 churn rate는 약 36.9%, annual은 약 15.4%였습니다.
- `subscription_type`에서는 student와 premium 그룹의 관찰 churn rate가 상대적으로 높았습니다.
- 수치형 컬럼의 단순 Pearson 상관은 전반적으로 매우 낮았습니다. 선형 상관이 낮다고 해서 비선형·범주형·상호작용 신호가 없다는 뜻은 아닙니다.
- 이 결과는 연관성 분석이며 “어떤 요인이 해지를 일으킨다”는 인과 결론이나 캠페인 효과를 의미하지 않습니다.

## 6. 중요한 제한사항

1. 현재 파일의 `churned` 생성 규칙과 관측 시점이 확인되지 않아, 요구된 “현재 시점 기준 향후 30일 내 해지” target을 검증하지 못했습니다.
2. Kaggle 페이지의 설명과 로컬 파일 행 수가 다릅니다. 출처 페이지는 40,000명 데이터셋으로 설명하지만 로컬 파일은 train 41,000행/test 9,999행입니다. 로컬 파일을 분석 기준으로 사용하고 해시를 `artifacts/run_metadata.json`에 기록했습니다.
3. Kaggle 출처 페이지에서 데이터가 합성인지 여부를 확정할 수 없어 Data Card에는 `미확인`으로 기록했습니다. 확인 전에는 실제 고객 일반화 성능을 주장하지 않습니다.
4. `num_subscription_pauses`라는 컬럼명이지만 실제 값은 `annual`/`monthly`입니다. 의미 확인 전까지 범주형으로만 처리합니다.
5. 외부 캠페인 자동 실행·고객 차단·자동 해지 예측 통지는 범위에 포함하지 않았습니다. 화면 결과는 검토 우선순위 지원용입니다.

## 7. 산출물 위치

- 모델 비교: `artifacts/model_comparison.csv`
- 모델 보고서: `artifacts/model_report.md`
- 모델 해석: `artifacts/feature_importance.csv`, `artifacts/eda/feature_importance.png`
- 저장 Pipeline: `artifacts/model/music_churn_pipeline.joblib`
- 제공 test 추론 결과: `artifacts/test_predictions.csv`

원본 출처: [Music Streaming Customer Churn Dataset — Kaggle](https://www.kaggle.com/datasets/daliado98/music-streaming-customer-churn-dataset)

## 8. 협업 및 Git 설정

- [GitHub 협업 규칙](CONTRIBUTING.md): 브랜치, 커밋, Issue, PR, 리뷰 규칙
- [Git·GitHub 최초 설정 가이드](docs/GIT_SETUP.md): Clone, 사용자 정보, 작업 브랜치, Push, 저장소 관리자 설정
- Pull Request와 Issue 작성 시 `.github/`의 템플릿을 사용합니다.

기본 원칙은 `main`에 직접 push하지 않고 `feature/*`, `fix/*`, `docs/*` 작업 브랜치에서 변경한 뒤 Pull Request로 병합하는 것입니다.
