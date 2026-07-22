# 데이터 출처와 이용 경계

## 출처

- 데이터셋: **Streaming Subscription Churn Model**
- 공식 페이지: [Kaggle Competition Data](https://www.kaggle.com/competitions/streaming-subscription-churn-model/data)
- 다운로드 기록: 2026-07-21 약 15:00 KST
- 파일: `train.csv`, `test.csv`
- 프로젝트 내부 경로: `data/train.csv`, `data/test.csv`

## 무결성

| 파일 | 행 수 | SHA-256 |
|---|---:|---|
| `train.csv` | 125,000 | `e967211a589021bda2e024c28f31ca84662f53f8c0f2985b565f336bb040b9b5` |
| `test.csv` | 75,000 | `b9081ab39f51a65eb39240e9d8c2dc90011efdb554d8b62896d4fa021252a71f` |

## 라이선스와 재배포

프로젝트 조사 당시 Kaggle metadata에는 MIT가 표시된 것으로 기록되어 있습니다. 다만 대회 데이터 다운로드에는 Competition Rules 동의가 요구될 수 있으며, 이 저장소 분석만으로 법적 재배포 가능성을 최종 승인할 수는 없습니다.

따라서 제출 전 담당자는 다음을 직접 확인해야 합니다.

1. 로그인한 Kaggle 계정에서 현재 Rules와 License 표시를 재확인합니다.
2. 교육기관 Google Drive와 GitHub에 원본 CSV를 포함할 수 있는지 확인합니다.
3. 허용되지 않으면 CSV를 제외하고 다운로드 방법·해시·스키마만 제출합니다.

## 실제·합성 판정

공식 페이지에서 합성 여부가 명시된 근거는 확보하지 못했습니다. 그러나 팀 분석에서 범주 균등성, 피처 독립성, 논리 위반, 계단형 Target 문턱값 및 로지스틱 계수 복원이 확인되어 **규칙 기반 합성으로 판정**했습니다.

이 판정은 데이터 분석 결과이며 공식 제작자 진술을 대신하지 않습니다. 상세 근거는 `docs/data_card.md`에 있습니다.
