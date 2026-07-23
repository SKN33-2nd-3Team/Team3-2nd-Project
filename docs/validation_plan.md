# 최종 검증 계획과 판정 범위

## 평가 설계

| 항목 | 설정 |
| --- | --- |
| 예측 단위 | 고객 1명 |
| Target | `churned` |
| 공통 Feature | `log_numeric` |
| 비교 Fold | 고정 Stratified 5-Fold, `random_state=42` |
| 1차 지표 | PR-AUC |
| 보조 지표 | ROC-AUC, F1, Recall, Precision, FN, FP, Brier |
| 추가 안정성 | 5개 Seed, 1,000회 Bootstrap |

모든 8개 모델은 같은 Feature와 같은 Fold에서 비교했습니다. 전처리는 Pipeline 내부에서 각 학습 Fold에만 Fit되며 Target 기반 인코딩은 사용하지 않았습니다.

## 모델 선정 절차

1. Dummy·Logistic·Decision Tree·Random Forest·Gradient Boosting·XGBoost·LightGBM·CatBoost를 동일 조건으로 비교했습니다.
2. 지정된 상한 안에서 7개 학습 모델의 Randomized Search를 수행했습니다.
3. 실제 상위 3개 CatBoost·XGBoost·LightGBM을 각각 15회 정밀 탐색했습니다.
4. Fine-tuned CV PR-AUC, OOF 지표, 5개 Seed 안정성, Bootstrap, 보정과 운영 표를 함께 검토했습니다.
5. CatBoost를 로컬 앱의 최종 기술 모델로 통합했습니다.

## Threshold 검증 원칙

Threshold는 모델 자체와 별도의 운영 결정입니다. OOF 예측에서 다음 세 가지 선택지를 제공하지만 자동 승인하지 않습니다.

- 재현율 우선: Threshold 0.29
- 균형형 F1: Threshold 0.35
- 정밀도 우선: Threshold 0.74

OOF에서 Threshold를 고르고 같은 OOF에서 표시한 수치는 용량·오류 Trade-off 참고값입니다. 독립 외부 Holdout 성능으로 해석하지 않습니다.

## 스키마·산출물 검증

- 새 Python 프로세스에서 후보 모델 재로딩
- 후보와 현재 앱 모델의 SHA-256 일치 확인
- 정상 스키마·열 순서 변경·미등록 범주 허용
- 필수 열 누락·잘못된 수치 타입 거부
- 저장된 75,000개 예측의 표본 재추론 일치 확인
- Streamlit 6개 메뉴 실행 검사

## 아직 필요한 외부 검증

- 관측 시점과 결과 기간을 정의한 시간 기반 검증
- 독립된 미래 라벨 Holdout 평가
- 실제 캠페인 A/B 테스트와 Uplift·ROI 측정
- 세그먼트별 공정성과 운영 오류 모니터링
- 개인정보·업무 적합성 최종 승인(데이터 라이선스 고지는 `data/LICENSE.md`에 기록)
