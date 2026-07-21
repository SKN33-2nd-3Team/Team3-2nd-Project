# PlaylistPro 최종 발표 콘셉트

## 커뮤니케이션 목표

발표가 끝날 때 강사·평가자와 리텐션 실무 관점의 청중은 PlaylistPro가 단순히 높은 점수의 분류기를 만든 프로젝트가 아니라, **합성 고객 데이터의 한계를 통제하면서 고객 행동 신호를 검증하고, 모델 확률을 제한된 고객관리 자원의 검토 우선순위로 전환한 의사결정 지원 프로젝트**임을 이해해야 한다.

## 최종 콘셉트

### 이탈을 맞히는 모델을 넘어, 누구부터 검토할지 결정하는 근거로

발표는 다음 질문으로 시작하고 끝난다.

> 모든 고객에게 같은 수준으로 대응할 수 없다면, 어떤 고객부터 검토해야 하는가?

모델 성능은 이 질문에 답하기 위한 중간 근거다. 최종 산출물은 CatBoost 확률 자체가 아니라, Threshold·Top-K·위험 구간과 행동 가설을 함께 보여주는 검토 체계다.

## 전체 Thesis

> 규칙 기반 합성 고객 데이터에서 낮은 청취 활동, Free 구독, 높은 고객 문의가 `churned`와 강하게 연관됨을 확인했다. 이 관찰을 수치 로그 변환 등 Feature 가설로 검증하고, 동일 Feature·동일 5-Fold 조건에서 8개 모델을 비교해 CatBoost를 최종 기술 후보로 선정했다. 최종적으로 모델 확률을 Threshold·Top-K 기반 고객 검토 우선순위와 후속 유지 실험 후보로 전환했다.

## 발표가 증명해야 할 세 가지

1. **분석 결론** — 고객 행동과 구독·서비스 마찰 신호가 제공된 `churned` 라벨과 연관됐다.
2. **모델 결론** — CatBoost가 Fine-tuned CV PR-AUC와 Seed 평균에서 근소하게 가장 높았지만 상위 세 모델의 불확실성 구간은 겹친다.
3. **비즈니스 제안** — 모델은 검토 대상의 순서를 정할 수 있지만, 실제 유지 효과·인과효과·ROI는 미래 라벨과 A/B 테스트가 있어야 확인할 수 있다.

## 청중에게 남길 의사결정 문장

> 현재 CatBoost는 운영 확정 모델이 아니라 로컬 앱에 통합된 최종 기술 후보다. 제한된 자원에서 위험 고객을 선별하고 후속 유지 실험 대상을 정하는 근거로 사용할 수 있으며, 실제 운영 전에는 시간 기반 외부 검증과 캠페인 실험이 필요하다.

## 본문 범위

- 본문 12장, 약 9~12분 발표를 기본으로 한다.
- 한 장에는 하나의 주장과 하나의 주 시각자료만 둔다.
- 전체 하이퍼파라미터, Calibration, Confusion Matrix, 세그먼트 오류, Feature Importance, 재로딩 검증은 부록으로 이동한다.
- 모델명·수치로 시작하지 않고 비즈니스 질문으로 시작한다.
- 마지막 장은 한계 나열이 아니라 다음 검증 순서와 현재 사용할 수 있는 범위를 확정한다.

## 비협상 표현 원칙

| 구분 | 사용할 표현 | 금지할 표현 |
| --- | --- | --- |
| 데이터 | 규칙 기반 합성으로 판정된 데이터 | 실제 PlaylistPro 고객 행동을 증명했다 |
| Target | 제공된 `churned` 라벨 판별 | 향후 30일 이탈 예측 |
| 인사이트 | 관측 연관성·위험 신호 | 이탈 원인 |
| Feature 실험 | Logistic 1차 검증에서 로그 변환 개선, CatBoost에서는 소폭 우위 | 로그 변환이 최종 성능을 크게 만들었다 |
| 최종 모델 | 최종 기술 후보·로컬 앱 통합 모델 | 운영 배포 완료 모델 |
| 검증 | 5-Fold OOF·5-Seed·Bootstrap 내부 근거 | 외부 Holdout으로 일반화 검증 완료 |
| 운영 | Threshold·Top-K 검토 시나리오 | 하나의 정답 Threshold |
| 행동 | 담당자가 검토할 유지 활동 가설 | 모델이 증명한 최적 캠페인 |
| 가치 | 편집 가능한 매출 대용치 기반 민감도 분석 | 실제 LTV·Uplift·ROI |

## 권위 기준

1. `experiments/submission_full_comparison/20260721_full_fair_v1/run_manifest.json`
2. `artifacts/model/metadata.json`
3. `reports/final_model_selection_decision.md`
4. `reports/presentation_evidence_pack_v3.md`
5. `artifacts/presentation_v3/*.csv`와 `figures/presentation_v3/*.png`
6. `docs/data_card.md`, `docs/validation_plan.md`, `docs/requirements.md`

이전 P0·bounded Run, `figures/presentation_v2/07_lightgbm_selection_evidence.png`, 과거 `reports/streamlit_presentation_storyline.md`의 모델·Holdout 표현은 최종 발표 근거로 사용하지 않는다.
