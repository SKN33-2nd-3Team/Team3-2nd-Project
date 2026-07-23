# 프로젝트 구조·제출 요구사항 감사

기준 문서: 저장소 루트의 `프로젝트 산출물 요구사항.md`, `프로젝트 요구사항.md`, `프로젝트 데이터 요구사항.md`  
감사 기준일: 2026-07-22

## 1. 현재 원본 저장소와 권장 구조의 차이

| 요구·권장 구조 | 원본 저장소 상태 | 영향 | 제출 패키지 조치 | 판정 |
|---|---|---|---|---|
| `notebooks/01_data_check.ipynb` | `eda 예시.ipynb`만 존재 | 실제 데이터 분석 Notebook 누락 | 실행 출력이 포함된 Notebook 신규 생성 | 보완 완료 |
| `notebooks/02_eda.ipynb` | 없음 | EDA 재현 파일 누락 | 예시 순서와 실제 그래프로 신규 생성 | 보완 완료 |
| `notebooks/03_model_experiments.ipynb` | 없음 | 모델 비교 설명 파일 누락 | 저장 실험 CSV 감사·재로딩 Notebook 생성 | 보완 완료 |
| `reports/preprocessing_report.md` | 존재하나 요약형 | 필수 항목과 그래프 해석이 부족 | 상세 결과서로 재작성 | 보완 완료 |
| `reports/training_report.md` | 존재하나 요약형 | Baseline·탐색·오류 분석 밀도 부족 | 상세 결과서로 재작성 | 보완 완료 |
| 제출용 PDF 2종 | 없음 | Google Drive 제출 형식 미충족 | PDF 생성·페이지 렌더 검증 | 보완 완료 |
| `models/churn_pipeline.joblib` | `artifacts/model/music_churn_pipeline.joblib` | 기능은 되지만 요구 경로와 다름 | 동일 SHA-256 파일을 권장 이름으로 복사 | 보완 완료 |
| `artifacts/feature_schema.json` | 메타데이터 내부에만 존재 | 입력 계약을 찾기 어려움 | 독립 schema JSON 생성 | 보완 완료 |
| `artifacts/model_metadata.json` | `artifacts/model/metadata.json` | 경로·이름이 예시와 다름 | 제출용 alias 생성 | 보완 완료 |
| `artifacts/metrics.csv` | 없음 | 대표 성능 표준 파일 누락 | 최종 OOF 기준 행 생성 | 보완 완료 |
| `data/` | `train.csv`, `test.csv`가 앱·Notebook 공통 입력 | 실행 경로 단일화 필요 | 루트 `data/`를 canonical 경로로 확정 | 보완 완료 |
| `streamlit_app/app.py` | `app/streamlit_app.py` | 권장 예시와 경로 차이 | 실제 실행 경로를 README에 명시 | 허용 가능한 차이 |
| `src/preprocessing`, `src/training` | 역할별 단일 모듈·scripts 구조 | 예시와 폴더명이 다름 | 실제 모듈을 유지하고 Notebook·보고서에서 연결 | 허용 가능한 차이 |
| `DATA_SOURCE.md` | `docs/data_card.md`만 존재 | README 인접 출처 파일 누락 | 요약 `DATA_SOURCE.md` 추가 | 보완 완료 |
| README 13개 섹션 | 기존 README는 운영 요약 중심 | 팀·구조·화면·회고 항목 부족 | 요구 템플릿 순서로 제출용 README 작성 | 보완 완료 |
| 발표자료 PDF | 신규본 반영 예정 | 현재 PPT는 교체 대상 | 신규 발표자료 반영 전까지 Manifest·검증에서 제외 | 보류 |
| 프로젝트 ZIP | 신규 발표자료 반영 후 생성 | 현 단계 생성 시 교체본 누락 | 최종 PPT 반영 뒤 생성·검증 | 보류 |

`streamlit_app/app.py`와 `src/preprocessing/` 같은 경로는 문서의 **권장 예시**입니다. 현재 `app/streamlit_app.py`가 실제 저장 모델을 로드하고 두 명령으로 실행되므로 기능을 깨뜨리는 강제 이동보다 실제 경로를 명확히 문서화했습니다.

## 2. 필수 산출물 추적

| 필수 산출물 | 요구 형식 | 패키지 파일 | 검증 |
|---|---|---|---|
| 데이터 전처리 결과서 | MD, PDF | `reports/preprocessing_report.md`, `.pdf` | PDF 렌더·본문 추출 |
| 전처리 분석 파일 | Notebook 또는 py, 이미지 | `notebooks/01_data_check.ipynb`, `02_eda.ipynb`, `figures/preprocessing/` | 실행 출력·오류 셀 0 |
| 최종 데이터 | 프로젝트 내부 | `data/train.csv`, `data/test.csv` | 행 수·SHA-256 |
| 모델 학습 결과서 | MD, PDF | `reports/training_report.md`, `.pdf` | PDF 렌더·수치 대조 |
| 모델 분석 파일 | Notebook 또는 py, 이미지 | `notebooks/03_model_experiments.ipynb`, `figures/training/` | 저장 결과 재현·오류 셀 0 |
| 학습된 최종 모델 | joblib | `models/churn_pipeline.joblib` | 새 프로세스 재로딩·1행 예측 |
| 모델 보조 파일 | schema, metadata, metrics | `artifacts/feature_schema.json`, `model_metadata.json`, `metrics.csv` | 스키마·해시 대조 |
| Streamlit | 저장 모델 실제 예측 | `app/streamlit_app.py`, `assets/screenshots/` | 6개 화면 회귀·시각 확인 |
| README | 설치·실행·전체 프로젝트 설명 | `README.md` | 2개 실행 명령·상대 링크 검사 |
| 발표자료 | PPTX 또는 PDF | 신규본 반영 예정 | 현재 Manifest·검증에서 제외 |
| ZIP | 프로젝트 폴더 압축 | 신규 발표자료 반영 후 생성 | 생성 뒤 압축 무결성 검사 |

## 3. 요구사항과 다른 부분 중 남아 있는 사항

### 기술적으로 의도된 차이

- 변환된 55개 Feature의 전체 행렬을 CSV로 저장하지 않습니다. Fold 외부에서 전처리 행렬을 만들면 누수 가능성이 있어, 고객 단위 입력 테이블과 저장 Pipeline을 최종 데이터 계약으로 사용합니다.
- 제공 Test는 정답이 없어 Test 성능 표를 만들지 않습니다. 이를 Holdout으로 오표기하지 않고 5-Fold OOF로 명시합니다.
- DL·SHAP은 선택 기능이며, 정형 합성 데이터와 제출 기한을 고려해 의도적으로 제외했습니다.

### 사람이 최종 확인해야 하는 차이

- 팀원별 역할과 회고의 최종 검토
- 실제 사업 Threshold와 유지 활동 승인
- Google Drive 업로드와 GitHub Push

위 항목은 `docs/human_confirmation_required.md`에 별도로 관리합니다.

## 4. 최종 판정

발표자료와 최종 ZIP을 제외한 기술 산출물은 저장소 루트의 권장 구조로 구성했습니다. 원본 저장소의 경로 차이는 기능 결함과 권장 구조 차이를 구분해 처리했습니다. 외부 Holdout·실제 캠페인 효과·법적 승인처럼 저장소에서 확정할 수 없는 내용은 완료로 위장하지 않았습니다.
