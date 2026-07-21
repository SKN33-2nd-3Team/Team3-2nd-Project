# 데이터·전처리 결과

최종 갱신일: 2026-07-22. 근거: `notebooks/01_data_check.ipynb`, `notebooks/02_eda.ipynb`, `figures/presentation_v3/` 및 저장된 최종 Run manifest.

## 데이터 범위

| 항목 | Train | 제공 Test |
|---|---:|---:|
| 행 수 | 125,000 | 75,000 |
| 열 수 | 20 | 19 |
| `churned` 라벨 | 있음 | 없음 |
| 결측치 | 0 | 0 |
| 중복 행 | 0 | 0 |
| `customer_id` 교집합 | \- | 0 |

Train의 관찰 라벨 이탈률은 51.34%(64,174/125,000)입니다. `churned`의 관측 시점과 결과 기간은 제공되지 않아 미래 기간의 이탈로 확정 해석하지 않습니다.

## 확인한 데이터 품질 이슈

- `weekly_unique_songs > weekly_songs_played`: 36,996건(29.6%)
- `num_shared_playlists > num_playlists_created`: 30,778건(24.6%)
- 수치형 특성끼리의 최대 절대 상관: 0.0071

두 비율형 파생변수는 분모 0을 안전하게 처리합니다. 원본 값을 임의로 고치거나 행을 삭제하지 않았습니다. 위 불가능 조합은 실제 서비스 도입 전 수집 규칙을 확인해야 할 데이터 계약 이슈입니다.

## Pipeline 처리

1. `customer_id`를 모델 입력에서 제외합니다.
2. `signup_date`로부터 `signup_days_ago`를 만듭니다.
3. `unique_song_ratio`, `shared_playlist_ratio`, `hours_per_song`, `friends_per_playlist`을 파생합니다.
4. 수치형은 median 대치, 범주형은 최빈값 대치와 `OneHotEncoder(handle_unknown="ignore")`를 적용합니다.
5. 모든 변환기는 Train 분할에만 fit하고 Validation·Test에는 transform만 적용합니다.

파생변수와 인코딩을 포함한 최종 모델 입력은 49개입니다. 저장된 Pipeline이 전처리와 추론을 함께 수행하므로 앱의 단건 점수와 배치 점수가 같은 경로를 사용합니다.

## 재현성

- 데이터 SHA-256, 환경, 입력 열 순서는 [artifacts/model/metadata.json](../artifacts/model/metadata.json)에 기록됩니다.
- 현재 고정 환경: Python 3.13.14, pandas 2.3.3, numpy 2.3.4, scikit-learn 1.9.0, joblib 1.5.3
- 원본 `data/train.csv`, `data/test.csv`는 수정하지 않습니다.

세부 EDA 그림은 [artifacts/eda/](../artifacts/eda/)와 [artifacts/eda_insight/](../artifacts/eda_insight/)에 생성됩니다.
