# 데이터 사전 (Data Dictionary)

> 대상: `data/train.csv` (125,000행 × 20열), `data/test.csv` (75,000행 × 19열, `churned` 제외)
> 분석 단위: **고객 1명 = 1행**. 결측값 0건, 중복행 0건.
> 출처·라이선스·합성 여부는 [data_card.md](data_card.md) 참조.

## 컬럼 정의

| 컬럼 | 자료형 | 고유값 | 범위 / 수준 | 단위·의미 | 모델 사용 |
|---|---|---:|---|---|---|
| `customer_id` | int64 | 125,000 | 1 ~ 125,000 (test는 200,000 ~ 274,999) | 고객 식별자 | **제외** (식별자) |
| `age` | int64 | 62 | 18 ~ 79 | 세 | **신호** (U자형: 25/35/61 경계) |
| `location` | str | 19 | Alabama, California, … | 미국 주(州) | 노이즈 (스프레드 0.016) |
| `subscription_type` | str | 4 | Free, Student, Family, Premium | 요금제 등급 | **신호** (스프레드 0.455) |
| `payment_plan` | str | 2 | Monthly, Yearly | 결제 주기 | 노이즈 (**스프레드 0.001**) |
| `num_subscription_pauses` | int64 | 5 | 0 ~ 4 | 구독 일시정지 횟수. ⚠️ 대회 문서는 "max 2"라 하나 실제는 4까지 | **신호** (문턱 3회) |
| `payment_method` | str | 4 | Apple Pay, Credit Card, Debit Card, Paypal | 결제 수단 | 노이즈 (0.011) |
| `customer_service_inquiries` | str | 3 | Low, Medium, High | 고객센터 문의 빈도 등급 | **신호** (스프레드 0.454) |
| `signup_date` | int64 | 2,922 | −2,922 ~ −1 | **기준일 대비 일수** (음수 = 과거). ⚠️ 대회 문서는 "date"라고 하나 실제는 정수 | 노이즈 (0.013) |
| `weekly_hours` | float64 | 125,000 | 0.00007 ~ 50 | 주간 청취시간(시간) | **신호 1위** (문턱 5/10/40h) |
| `average_session_length` | float64 | 124,996 | 1.0 ~ 120.0 | 평균 세션 길이. ⚠️ **단위 미상** (대회 문서는 "시간"이라 하나 120시간 세션은 비현실적) | 노이즈 (0.015) |
| `song_skip_rate` | float64 | 124,995 | 0.0 ~ 1.0 | 곡 스킵 비율 | **신호** (문턱 0.7) |
| `weekly_songs_played` | int64 | 497 | 3 ~ 499 | 주간 재생 곡 수 | 노이즈 (0.017) |
| `weekly_unique_songs` | int64 | 297 | 3 ~ 299 | 주간 고유 재생 곡 수 | 노이즈 (0.029) |
| `num_favorite_artists` | int64 | 50 | 0 ~ 49 | 즐겨찾기 아티스트 수 | 노이즈 (0.013) |
| `num_platform_friends` | int64 | 200 | 0 ~ 199 | 플랫폼 친구 수 | 노이즈 (0.015) |
| `num_playlists_created` | int64 | 100 | 0 ~ 99 | 생성 플레이리스트 수 | 노이즈 (0.021) |
| `num_shared_playlists` | int64 | 50 | 0 ~ 49 | 공유 플레이리스트 수 | 노이즈 (0.021) |
| `notifications_clicked` | int64 | 50 | 0 ~ 49 | 클릭한 알림 수 | **신호** (문턱 5회) |
| `churned` | int64 | 2 | 0, 1 | **Target.** 0 = 유지, 1 = 이탈 | Target (train 전용) |

"신호/노이즈"는 해당 컬럼을 10분위(범주형은 수준별)로 나눈 **이탈률 스프레드(max−min)** 기준이며, 0.10을 경계로 7개와 11개가 중간지대 없이 갈린다. 근거: [../artifacts/eda_insight/INSIGHT_REPORT.md](../artifacts/eda_insight/INSIGHT_REPORT.md) §2 차트 3.

## 파생 피처 (`src/features.py` · `MusicFeatureEngineer`)

Pipeline 내부에서 생성되므로 학습·추론에 동일하게 적용된다.

| 파생 피처 | 정의 | 비고 |
|---|---|---|
| `signup_days_ago` | `-signup_date` | 원본이 수치형일 때만 생성 |
| `unique_song_ratio` | `weekly_unique_songs / (weekly_songs_played + 1)` | 원천 2개가 모두 노이즈 |
| `shared_playlist_ratio` | `num_shared_playlists / (num_playlists_created + 1)` | 원천 2개가 모두 노이즈 |
| `hours_per_song` | `weekly_hours / (weekly_songs_played + 1)` | — |
| `friends_per_playlist` | `num_platform_friends / (num_playlists_created + 1)` | 원천 2개가 모두 노이즈 |

⚠️ 파생 4종 중 3종이 노이즈 컬럼 조합이다. 제거 시 Test PR-AUC 변화는 −0.0009로 무의미하다. [INSIGHT_REPORT.md](../artifacts/eda_insight/INSIGHT_REPORT.md) §4 결정 2 참조.

## 데이터 무결성 경고

| 검사 | 위반 건수 | 의미 |
|---|---:|---|
| `weekly_unique_songs > weekly_songs_played` | 36,996 (29.6%) | 고유 곡이 총 재생 곡보다 많을 수 없다 |
| `num_shared_playlists > num_playlists_created` | 30,778 (24.6%) | 만들지 않은 플레이리스트를 공유할 수 없다 |
| 피처 간 최대 \|상관\| | 0.007 | 모든 컬럼이 독립 난수로 생성됨 |

→ 두 위반 컬럼은 모두 노이즈로 판정되어 모델에서 제외되므로 성능 영향은 없다. 다만 **이 데이터로 청취 패턴 2차 분석을 수행해서는 안 된다**.

## 제거 대상 컬럼

- `customer_id` — 식별자. 모델 입력에서 제외한다.
- `signup_date` 원본 — 예측 시점 스냅샷이 검증되지 않아 원본은 드롭하고 파생 필드만 남긴다.
- 개인정보(이름·연락처·이메일·정밀 위치)에 해당하는 컬럼은 **없다**. `location`은 주(州) 단위라 식별 위험이 낮다.
