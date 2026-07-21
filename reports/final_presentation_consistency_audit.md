# PlaylistPro 최종 발표 일관성·모순 감사

## 감사 기준

- 최종 권위 Run: `20260721_full_fair_v1`
- 최종 모델 메타데이터: `artifacts/model/metadata.json`
- 최종 발표 수치: `artifacts/presentation_v3/*.csv`
- 발표 그래프: `figures/presentation_v3/*.png`
- 데이터·검증 경계: `docs/data_card.md`, `docs/validation_plan.md`, `docs/requirements.md`

## 발견하고 수정한 모순

| ID | 발견된 모순 또는 위험 | 권위 근거 | 최종 처리 | 상태 |
| --- | --- | --- | --- | --- |
| A01 | 과거 자료는 LightGBM 또는 Gradient Boosting을 최종 모델처럼 표현 | 최신 Run과 모델 Metadata는 CatBoost | 본문 전체를 CatBoost 최종 기술 후보로 통일. LightGBM은 Recall·Calibration Trade-off 비교 후보로만 사용 | 수정 완료 |
| A02 | PR-AUC를 “양성 클래스가 적어서” 사용한다는 설명 | 이탈 비율 51.34% | 관심 클래스 순위 품질과 Precision–Recall Trade-off를 보기 위한 지표로 수정 | 수정 완료 |
| A03 | 내부 holdout 또는 외부 일반화 검증이 완료된 것처럼 읽히는 표현 | `external_labeled_holdout=NOT_AVAILABLE` | 모든 성능·Threshold·Top-K를 5-Fold OOF로 표기하고 외부 Holdout 없음 명시 | 수정 완료 |
| A04 | “향후 30일 이탈” 같은 시간 지평 | 관측 기준일·해지일 없음 | 스냅샷 고객 프로필로 제공된 `churned` 라벨 판별로 통일 | 수정 완료 |
| A05 | 로그 변환이 최종 CatBoost 성능을 크게 높였다는 서사 | Logistic +0.006798, CatBoost Raw 대비 약 +0.000089 | Logistic 1차 개선과 CatBoost 소폭 확인을 분리. 전체 성능 상승의 주원인은 모델 전환으로 설명 | 수정 완료 |
| A06 | Random search 최고 OOF를 최종 후보로 다시 선택한 것처럼 보임 | 최종 선정 지표는 Fine-tuned CV PR-AUC | 단일 OOF 최고값보다 Fine CV·Seed·Bootstrap 선정 절차를 따랐다고 명시 | 수정 완료 |
| A07 | CatBoost가 모든 지표에서 압도적 1위라는 표현 | Bootstrap 구간 중첩, LightGBM Recall·Brier 우위 | CatBoost는 근소한 순위 품질·Seed 안정성 기준의 기술 후보로 제한 | 수정 완료 |
| A08 | 0.35를 최종 운영 Threshold로 단정 | Metadata 상태는 사업 Threshold 결정 대기 | 0.29·0.35·0.74 복수 시나리오와 사용자 결정으로 표현 | 수정 완료 |
| A09 | Lift를 실제 유지 효과로 해석 | Lift는 OOF 라벨 집중도 | 무작위 선정 대비 관측 이탈 포착 효율로만 표현 | 수정 완료 |
| A10 | Feature Importance를 이탈 원인으로 해석 | 인과·시간 순서 미검증 | 관측 연관성과 예측 신호로만 표현. 나이·지역은 행동 배정 근거에서 제외 | 수정 완료 |
| A11 | Streamlit 행동 제안을 모델이 학습한 최적 처방처럼 표현 | 행동은 `retention_strategy.py`의 투명 규칙 | 예측 모델과 전략 규칙을 분리하고 담당자 검토 필요 명시 | 수정 완료 |
| A12 | 가정 기반 계획을 실제 LTV·Uplift·ROI로 표현 | 공개 요금·사용자 가정 기반 민감도 분석 | 본문 재무 성과에서 제외하고 Q&A·부록에서 가정임을 설명 | 수정 완료 |
| A13 | 높은 PR-AUC·Top-K 결과를 실제 서비스 기대치로 해석 | 규칙 기반 합성 데이터 판정 | Slide 3과 12에서 합성성·외부 일반화 부재를 공개 | 수정 완료 |
| A14 | 구 `streamlit_presentation_storyline.md`가 Gradient Boosting·LightGBM·holdout을 혼합 | 파일 자체가 통합 전 기록으로 표기 | 최종 발표 소스에서 제외 | 제외 |
| A15 | `figures/presentation_v2/07_lightgbm_selection_evidence.png`가 최신 그래프와 공존 | v3 발표 팩과 Run이 최신 | v2 전체를 본문·부록에서 제외 | 제외 |
| A16 | `source_repository_audit.md` 본문이 P0 시작 시점 상태를 기록 | 헤더가 현재 CatBoost 상태와 과거 스냅샷을 구분 | 최초 조사 이력으로만 보존하고 최종 수치는 Run manifest·Metadata 사용 | 범위 제한 |

## 데이터와 문제 정의 감사

| 점검 항목 | 최종 판정 | 발표 처리 |
| --- | --- | --- |
| 분석 단위 | VERIFIED | 고객 1명 = 1행 |
| Train/Test 규모 | VERIFIED | 125,000 / 75,000 |
| Target | VERIFIED | `churned`, 0=유지·1=이탈 |
| 이탈 비율 | VERIFIED | 51.34%, 불균형 주장 금지 |
| 결측·중복 | VERIFIED | 0건 |
| Train–Test ID 중첩 | VERIFIED | 0건 |
| 데이터 합성성 | PARTIALLY_VERIFIED | 공식 명시가 아니라 팀 분석으로 “규칙 기반 합성으로 판정” 표현 |
| 예측 기간 | UNVERIFIED | “30일 이탈” 제외 |
| 외부 일반화 | UNVERIFIED | 외부 라벨 Holdout 부재 명시 |

## 모델링 감사

| 점검 항목 | 판정 | 근거 |
| --- | --- | --- |
| Raw Logistic과 Log Logistic 구분 | 충족 | `preprocessing_experiment_results.csv` |
| 동일 Feature·동일 Fold 비교 | 충족 | `model_comparison_fair.csv`, `cv_fold_assignments.csv` |
| 8개 모델 비교 | 충족 | Dummy 포함 8개 |
| 7개 모델 RandomizedSearch | 충족 | Run manifest Trial 수 기록 |
| 실제 Top 3 정밀 탐색 | 충족 | CatBoost·XGBoost·LightGBM 각 15회 |
| Seed 안정성 | 충족 | 5개 Seed |
| Bootstrap | 충족 | 1,000회, 상위 3개 CI 중첩 |
| Calibration | 충족 | Brier·평균 절대 보정 오차 저장 |
| 저장·재로딩 | 충족 | SHA-256·새 프로세스 재로딩 통과 |
| 외부 라벨 Holdout | 미충족 | NOT_AVAILABLE, 발표 한계로 처리 |

## 인사이트 감사

| 요구사항 | 반영 위치 | 판정 |
| --- | --- | --- |
| 핵심 고객 행동 인사이트 | Slide 4 | 충족 |
| 인사이트와 Feature 가설 연결 | Slide 5 | 충족 |
| 행동 가능한 신호와 진단 신호 구분 | Slide 4·11·A9 | 충족 |
| 연관성과 인과 구분 | Slide 3·4·12 | 충족 |
| 실패 Feature 실험 포함 | Slide 5 | 충족 |
| 실제 대응 행동 제안 | Slide 11 | 부분 충족 — 행동 후보이며 효과 미검증 |

## 강사 요구사항 감사

| 강사 요구사항 | 발표 반영 | 판정 |
| --- | --- | --- |
| 성능을 높이기 위해 무엇을 시도했는가 | Slide 5~7 | 충족 |
| 조치별 성능 변화 그래프 | Slide 5~7, `figures/presentation_v3` | 충족 |
| Recall·FN·PR-AUC·대상 수를 반영한 선정 설명 | Slide 8~10 | 충족 |
| Baseline부터 최종 후보까지 변화 | Slide 6 | 충족 |
| 채택·제외 실험 | Slide 5 | 충족 |
| 8개 모델 동일 조건 비교 | Slide 7 | 충족 |
| 튜닝 전후 | Slide 7·A2·A3 | 충족 |
| Threshold 시나리오 | Slide 9 | 충족 |
| Top-K·Lift·Risk Decile | Slide 10 | 충족 |
| 최고 단일 점수와 최종 선정 차이 | Slide 7~8·Q3 | 충족 |
| 일반화 검증 | Slide 12 | 확인 불가를 정직하게 명시 |
| 발표 그래프와 원본 CSV 연결 | Visual Asset Map | 충족 |

## 프로젝트 요구사항 감사

| 프로젝트 요구사항 | 발표 반영 | 판정 |
| --- | --- | --- |
| 데이터 구조·Target·누수 방지 | Slide 3·6 | 충족 |
| EDA와 고객 인사이트 | Slide 4 | 충족 |
| 인사이트 기반 개선 | Slide 5 | 충족 |
| 모델 비교·선정 | Slide 7·8 | 충족 |
| 운영 Threshold·우선순위 | Slide 9·10 | 충족 |
| 저장 모델 예측과 Streamlit | Slide 11 | 충족 |
| 고객별 행동 후보 | Slide 11 | 충족, 규칙 기반·담당자 검토 명시 |
| 실제 캠페인 실행 | 범위 밖 | 앱과 발표에서 실행하지 않음 명시 |
| 실제 유지 효과 검증 | 미완료 | Slide 12의 A/B 테스트 계획으로 이동 |

## 최종 판정

- 최신 CatBoost Run과 v3 시각자료 기준으로 본문 12장과 부록 사양이 일관되게 정리됐다.
- 데이터 합성성·예측 기간·외부 Holdout·Uplift 한계를 초반과 결론에 배치해 과장을 차단했다.
- 강사 요구사항의 모델링 Journey와 프로젝트 요구사항의 행동 제안이 하나의 의사결정 흐름으로 연결됐다.
- 남은 작업은 실제 PowerPoint 제작 단계의 화면 배치·폰트·겹침·가독성 렌더링 QA다.
