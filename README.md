# PlaylistPro Insight

음악 스트리밍 고객의 구독·이용·문의 정보를 바탕으로 관측 이탈 위험을 예측하고, 리텐션 담당자가 **누구를 먼저 검토하고 어떤 유지 활동을 실험할지** 결정하도록 돕는 머신러닝 의사결정 지원 프로젝트입니다.

> 최종 기술 모델은 CatBoost입니다. 독립 외부 라벨 Holdout과 실제 캠페인 결과가 없으므로, 내부 5-Fold OOF 성능을 미래 일반화 성능이나 캠페인 효과로 표현하지 않습니다.

## 목차

1. [팀 소개](#1-팀-소개)
2. [프로젝트 개요](#2-프로젝트-개요)
3. [요구사항](#3-요구사항)
4. [데이터](#4-데이터)
5. [데이터 분석 및 전처리](#5-데이터-분석-및-전처리)
6. [모델링](#6-모델링)
7. [주요 기능](#7-주요-기능)
8. [프로젝트 구조](#8-프로젝트-구조)
9. [기술 스택](#9-기술-스택)
10. [설치 및 실행](#10-설치-및-실행)
11. [수행 화면과 제출 문서](#11-수행-화면과-제출-문서)
12. [한계 및 개선 방향](#12-한계-및-개선-방향)
13. [회고와 제출 전 확인](#13-회고와-제출-전-확인)

## 1. 팀 소개

### 팀명

SKN 33기 2차 프로젝트 3팀

### 팀원 및 역할 분담

| 팀원 | 주역할 | 담당 산출물 | 교차검토 |
|---|---|---|---|
| 최경돈 | 데이터 분석·전처리 | 데이터 품질·분포 점검, Feature Engineering, 데이터 점검·EDA Notebook, Data Card, 전처리 결과서·시각화 | 모델링 과정의 데이터 누수 여부와 발표자료의 데이터 수치 검토 |
| 김일환 | 모델링·성능 검증 | 8개 모델 비교, CatBoost 튜닝, OOF·Seed·Bootstrap 검증, Threshold 분석, 최종 모델·학습 결과서 | Feature 재현성과 Streamlit 추론 결과 검토 |
| 이준희B | Streamlit·서비스 구현 | 저장 CatBoost 연동, 단건·배치 예측, 고객 우선순위·운영 시나리오, 앱 테스트·시연 화면 | 모델·메타데이터 연결과 실행 안내 검토 |
| 이준희A | 프로젝트 통합·문서·발표·QA | README, 요구사항 문서, 발표자료·발표자 노트, 제출 체크리스트·Manifest, 링크·경로·실행 검증 | 분석 수치·결론 표현과 앱 시연 흐름 검토 |

각 담당자는 주 산출물을 작성하고, 교차검토 담당자는 수치·경로·재현성·표현의 일관성을 확인합니다. 최종 모델 선정과 발표 결론은 전원이 공동 검토했습니다. 세부 확인 항목은 [docs/human_confirmation_required.md](docs/human_confirmation_required.md)에 정리했습니다.

### 프로젝트 일정

| 단계 | 기준 일자 | 주요 작업 | 산출물 |
|---|---|---|---|
| 요구사항·데이터 감사 | 2026-07-21 | Target, 출처, 품질·합성 여부 점검 | Data Card, 요구사항 정의서 |
| EDA·전처리 | 2026-07-21 | 신호·노이즈, 데이터 계약 위반, Feature 실험 | 전처리 결과서, Notebook |
| 모델링·검증 | 2026-07-21~22 | 8개 모델 비교, 탐색, 안정성·운영 분석 | 학습 결과서, 저장 모델 |
| 서비스·통합 | 2026-07-22 | 저장 모델 연결, Streamlit·행동 후보 | Streamlit 시연 화면 |
| 제출 패키지 | 2026-07-22 | 문서·PDF·발표자료·ZIP 검증 | 현재 폴더 |

## 2. 프로젝트 개요

### 문제 정의

> 음악 스트리밍 서비스의 **리텐션·CRM 담당자**가 고객 관리 우선순위와 유지 활동 후보를 결정할 수 있도록, 고객별 구독·이용·문의 프로필로 제공된 `churned` 라벨을 예측합니다.

| 항목 | 정의 |
|---|---|
| 사용자 | 리텐션·CRM 담당자 |
| 예측 단위 | 고객 1명 = 1행 |
| Target | `churned`: 0=유지, 1=이탈 |
| 예측 시점 | 데이터에 기준일·결과 기간이 없어 미정의 |
| 1차 모델 지표 | PR-AUC |
| 운영 중요 지표 | Recall, Precision, F1, FN, FP, 대상 고객 수 |
| 중요 오류 | FN은 이탈 고객을 놓치고, FP는 불필요한 접촉 비용을 발생시킴 |
| 활용 | 위험 고객 순위화, 운영 Threshold 비교, 담당자 검토용 유지 활동 후보 |

### 최종 상태

| 항목 | 결과 |
|---|---:|
| 학습 고객 | 125,000명 |
| 무라벨 점수 산출 고객 | 75,000명 |
| 최종 기술 모델 | CatBoost |
| 공통 Feature | `log_numeric` |
| 평가 | 고정 Stratified 5-Fold OOF |
| Fine-tuned CV PR-AUC | 0.947913 |
| OOF PR-AUC / ROC-AUC | 0.947897 / 0.941959 |
| 5-Seed 평균 PR-AUC | 0.947880 |
| 기본 화면 Threshold | 0.35(균형형 F1, 사업 확정값 아님) |

## 3. 요구사항

| ID | 기능 | 구현 위치 | 상태 |
|---|---|---|---|
| FR-01 | 고객·이탈 현황과 EDA | Streamlit `데이터·고객 인사이트` | 완료 |
| FR-02 | 모델 성능 비교·선정 근거 | `개선 실험`, `모델 선정` | 완료 |
| FR-03 | 저장 모델 기반 개별 예측 | `고객 우선순위 > 단건 시뮬레이션` | 완료 |
| FR-04 | 위험 등급·Threshold | `운영 시나리오`, `고객 우선순위` | 완료 |
| FR-05 | 위험 요인·유지 활동 후보 | `고객 우선순위` | 완료 |
| FR-06 | 배치 우선순위·CSV | `고객 우선순위 > 배치 우선순위` | 선택 기능 완료 |
| FR-07 | 가정 기반 계획 | `고객 우선순위 > 가정 기반 계획` | 선택 기능 완료 |
| FR-08 | DL·개별 SHAP | - | 의도적 제외 |

상세 완료 기준과 범위는 [docs/requirements.md](docs/requirements.md), 구조 차이는 [docs/project_structure_audit.md](docs/project_structure_audit.md)에서 확인합니다.

## 4. 데이터

| 항목 | 내용 |
|---|---|
| 데이터셋 | Streaming Subscription Churn Model |
| 공식 페이지 | [Kaggle Competition](https://www.kaggle.com/competitions/streaming-subscription-churn-model/data) |
| Train | 125,000행 × 20열 |
| Test | 75,000행 × 19열, 정답 라벨 없음 |
| 키 | `customer_id`, Train/Test 교집합 0건 |
| 관측 Target 비율 | 64,174 / 125,000 = 51.34% |
| 라이선스 | 프로젝트 조사 당시 Kaggle metadata에 MIT 표기; 대회 Rules 동의 및 최종 적법성 확인 필요 |
| 실제·합성 | 공식 표기는 확인되지 않았으나 분포·문턱값·계수 복원 결과 규칙 기반 합성으로 판정 |

파일 출처·SHA-256·공식 설명과 실제 값의 불일치는 [docs/data_source.md](docs/data_source.md)와 [docs/data_card.md](docs/data_card.md)에 기록했습니다.

## 5. 데이터 분석 및 전처리

### 품질 점검

| 점검 | 결과 | 처리·근거 |
|---|---:|---|
| 결측 | 0 | 새 입력 대비 Pipeline 대치는 유지 |
| 완전 중복 | 0 | 삭제 없음 |
| 고객 ID 중복 | 0 | 식별자는 모델에서 제외 |
| `weekly_unique_songs > weekly_songs_played` | 36,996건(29.6%) | 수집 규칙 경고, 원본 임의 수정 금지 |
| `num_shared_playlists > num_playlists_created` | 30,778건(24.6%) | 수집 규칙 경고, 원본 임의 수정 금지 |

### Pipeline

1. `customer_id`를 모델 Feature에서 제외합니다.
2. 음수 정수 `signup_date`를 `signup_days_ago`로 변환합니다.
3. 분모 0을 고려한 비율 Feature와 `log1p` 수치 Feature를 생성합니다.
4. 수치형은 median, 범주형은 최빈값 대치 후 One-Hot 인코딩합니다.
5. 모든 변환을 학습 Fold 안에서만 Fit합니다.
6. 최종 변환 Feature는 55개입니다.

실행 결과가 저장된 분석 파일은 [01_data_check.ipynb](notebooks/01_data_check.ipynb)와 [02_eda.ipynb](notebooks/02_eda.ipynb), 전체 문서는 [전처리 결과서](reports/preprocessing_report.md)입니다.

## 6. 모델링

### 비교 절차

`Dummy → Logistic → Decision Tree → Random Forest → Gradient Boosting → XGBoost → LightGBM → CatBoost`를 동일 `log_numeric` Feature와 동일 5-Fold로 비교했습니다. Dummy를 제외한 7개 모델은 제한된 RandomizedSearch를 수행했고, 실제 상위 3개는 각각 15회 Fine Tuning을 수행했습니다.

| 모델 | Fine CV PR-AUC | OOF PR-AUC | OOF F1 | OOF Recall | OOF Precision |
|---|---:|---:|---:|---:|---:|
| **CatBoost** | **0.947913** | **0.947897** | 0.849118 | 0.836382 | 0.862247 |
| LightGBM | 0.947806 | 0.947790 | **0.851892** | **0.859850** | 0.844079 |
| XGBoost | 0.947793 | 0.947765 | 0.850378 | 0.844283 | 0.856562 |

CatBoost는 사전에 정한 Fine CV PR-AUC와 Seed 평균에서 근소하게 1위여서 선택했습니다. 상위 3개의 Bootstrap 구간이 겹치므로 압도적 우위로 표현하지 않습니다. 기본 Threshold에서 Recall·F1을 더 중시하면 LightGBM도 합리적인 대안입니다.

### 운영 시나리오

| 시나리오 | Threshold | 대상 | Recall | Precision | F1 | FN | FP |
|---|---:|---:|---:|---:|---:|---:|---:|
| 재현율 우선 | 0.29 | 77,700명 | 95.18% | 78.61% | 0.8611 | 3,091 | 16,617 |
| 균형형 F1 | 0.35 | 76,143명 | 94.44% | 79.59% | 0.8638 | 3,569 | 15,538 |
| 정밀도 우선 | 0.74 | 48,463명 | 71.80% | 95.07% | 0.8181 | 18,099 | 2,388 |

자세한 과정은 [03_model_experiments.ipynb](notebooks/03_model_experiments.ipynb), [모델 학습 결과서](reports/training_report.md), [artifacts/metrics.csv](artifacts/metrics.csv)에 있습니다.

## 7. 주요 기능

- 프로젝트·데이터·모델 핵심 결과 요약
- 신호와 노이즈, Feature 중요도, 세그먼트 인사이트
- 전처리·모델 개선 실험과 채택·제외 근거
- 8개 모델·Top 3·Seed·Bootstrap 비교
- Threshold, Top-K Capture, Lift, Risk Decile
- 저장 CatBoost를 이용한 고객 단건 재추론
- 행동 가능 신호 기반 담당자 검토용 유지 활동 후보
- 배치 우선순위 CSV와 가정 기반 캠페인 계획

유지 활동은 자동 실행되지 않습니다. CRM 발송, 할인 제공, 고객 접촉 및 실제 효과 검증은 이 프로젝트 범위 밖입니다.

## 8. 프로젝트 구조

```text
Team3-2nd-Project/
├── README.md
├── requirements.txt
├── app/streamlit_app.py
├── data/{train.csv,test.csv}
├── notebooks/{01_data_check,02_eda,03_model_experiments}.ipynb
├── src/                         # 전처리·추론·유지 전략 코드
├── scripts/                     # 최종 CatBoost 비교·검증·승격 코드
├── models/churn_pipeline.joblib # 제출용 최종 모델 alias
├── artifacts/                   # 운영 모델·metadata·metrics·실험 CSV
├── figures/                     # EDA·모델링·발표 시각화
├── reports/                     # Markdown 및 PDF 결과서
├── presentation/                # 신규 발표자료 반영 전 작업 영역(현재 제출 검증 제외)
├── assets/screenshots/          # Streamlit 시연 근거
├── docs/                        # 데이터·요구사항·구조·제출 문서
├── tests/
├── tools/
└── submission_manifest.json
```

`src/legacy/run_holdout_pipeline.py`는 초기 Holdout·Gradient Boosting 실험을 보존한 레거시 코드입니다. 최종 CatBoost 학습 계보는 `scripts/run_full_fair_comparison.py` → `scripts/finalize_full_fair_candidate.py` → `scripts/promote_full_fair_candidate.py`입니다.

## 9. 기술 스택

| 구분 | 기술 |
|---|---|
| Language | Python 3.13 |
| Data | pandas 2.3.3, NumPy 2.3.4 |
| ML | scikit-learn 1.9.0, CatBoost 1.2.10, LightGBM 4.7.0, XGBoost 3.3.0 |
| Visualization | Matplotlib 3.9.2, Plotly 6.4.0 |
| Service | Streamlit 1.51.0 |
| Serialization | joblib 1.5.3 + SHA-256 검증 |

## 10. 설치 및 실행

```bash
python -m pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

검증은 다음 명령으로 실행합니다.

```bash
python -m pytest -q
python tools/validate_submission.py
```

앱은 모델을 다시 학습하지 않습니다. 운영 모델 `artifacts/model/music_churn_pipeline.joblib`과 `artifacts/model/metadata.json`의 SHA-256이 다르면 로드를 중단합니다. `models/churn_pipeline.joblib`은 동일 SHA-256을 가진 제출용 alias입니다.

## 11. 수행 화면과 제출 문서

| 제출물 | Markdown·원본 | PDF·실행 파일 |
|---|---|---|
| 데이터 전처리 결과서 | [reports/preprocessing_report.md](reports/preprocessing_report.md) | `reports/preprocessing_report.pdf` |
| 인공지능 모델 학습 결과서 | [reports/training_report.md](reports/training_report.md) | `reports/training_report.pdf` |
| 최종 모델 | [artifacts/model_metadata.json](artifacts/model_metadata.json) | `models/churn_pipeline.joblib` |
| 분석 파일 | `notebooks/*.ipynb`, `src/*.py` | 실행 출력 포함 |
| Streamlit 시연 | `app/streamlit_app.py` | `assets/screenshots/*.png` |
| 발표자료 | 신규본 반영 예정 | 현재 `submission_manifest.json`과 제출 검증 대상에서 제외 |
| 전체 패키지 | `submission_manifest.json` | 상위 폴더의 ZIP |

### 대표 실행 화면

| 프로젝트 요약 | 모델 선정 | 고객 우선순위 | 캠페인 계획 |
|---|---|---|---|
| ![프로젝트 요약](assets/screenshots/01_project_summary.png) | ![모델 선정](assets/screenshots/02_model_selection.png) | ![고객 우선순위](assets/screenshots/03_customer_priority.png) | ![캠페인 계획](assets/screenshots/04_campaign_planning.png) |

발표자료는 신규본으로 교체한 뒤 별도로 검증하고 제출 Manifest에 추가합니다.

## 12. 한계 및 개선 방향

| 현재 한계 | 영향 | 다음 검증 |
|---|---|---|
| 규칙 기반 합성 데이터 | 점수가 실제 서비스보다 과대평가될 수 있음 | 실제 고객 데이터로 재검증 |
| 예측 기준일·결과 기간 없음 | 미래 N일 이탈로 해석 불가 | 관측·결과 창 정의 |
| 독립 외부 라벨 Holdout 없음 | 외부 일반화 확인 불가 | 시간 이후 Holdout 1회 평가 |
| Feature 중요도는 연관성 | 이탈 원인·개입 효과로 단정 불가 | 도메인 검토와 통제 실험 |
| 유지 활동 효과 미검증 | Uplift·ROI 주장 불가 | 행동별 A/B 테스트 |
| 라이선스·업무 적합성 최종 승인 미완료 | 외부 배포 판단 필요 | 담당 조직·강사 최종 확인 |

## 13. 회고와 제출 전 확인

기술적으로 확인 가능한 항목은 [docs/final_submission_checklist.md](docs/final_submission_checklist.md)에 기록했습니다. 팀원별 역할·회고, 최종 제출 계정, 데이터 이용 승인처럼 사람이 확인해야 하는 항목은 [docs/human_confirmation_required.md](docs/human_confirmation_required.md)에 분리했습니다.

이 프로젝트의 예측과 행동 제안은 담당자의 의사결정을 보조하며, 고객에 대한 자동 조치나 실제 캠페인 효과를 보장하지 않습니다.
