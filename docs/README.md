# 프로젝트 문서 안내

공지의 필수 작성 항목과 저장소 문서를 연결합니다.

| 문서 | 공지 대응 항목 | 주요 내용 |
|---|---|---|
| [requirements.md](requirements.md) | 비즈니스 문제 정의 | 사용자, Target, 예측 시점, 주요 지표, 화면, 완료 기준 |
| [data_card.md](data_card.md) | 데이터 수집·구조 설계 | 출처, 라이선스, 단위, 키, 실제·합성, 개인정보, 규모 |
| [data_dictionary.md](data_dictionary.md) | 데이터 전처리 결과서 | 컬럼, 자료형, 단위, 범위, 모델 사용 여부 |
| [validation_plan.md](validation_plan.md) | 분할·평가·임계값 | 누수 방지, Train/Validation/Test, 모델 비교, threshold |
| [submission_checklist.md](submission_checklist.md) | 필수 산출물 완료 기준 | 제출 파일, 실행 명령, 점검 상태 |
| [../reports/preprocessing_report.md](../reports/preprocessing_report.md) | 데이터 전처리 결과서 | 품질 점검, EDA, 전처리, Feature, 한계 |
| [../reports/training_report.md](../reports/training_report.md) | 인공지능 모델 학습 결과서 | 후보 모델, 비교표, 최종 Test, 오류 현황, 모델 저장 |

문서 수치의 기준 파일은 `artifacts/model/metadata.json`, `artifacts/model_comparison_fair.csv`, `artifacts/metrics.csv`입니다.
