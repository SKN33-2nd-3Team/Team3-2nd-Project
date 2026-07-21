# 데이터 폴더

`processed/train.csv`와 `processed/test.csv`는 고객 1명당 1행인 모델 입력 테이블입니다. 원본 값을 임의 수정하지 않았으며, 결측 처리·범주형 인코딩·수치 변환은 저장된 Pipeline이 학습 Fold 내부에서 수행합니다.

- `train.csv`: 125,000행, `churned` 포함
- `test.csv`: 75,000행, 정답 라벨 없음
- 출처·라이선스·해시: `../DATA_SOURCE.md`
- 컬럼 정의: `../docs/data_dictionary.md`

고차원 One-Hot 변환 행렬을 별도 CSV로 저장하지 않은 이유는 전처리 누수를 방지하고 학습·추론에 동일한 Pipeline을 강제하기 위해서입니다.
