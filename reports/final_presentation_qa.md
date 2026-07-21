# PlaylistPro 최종 발표 콘텐츠 QA

## 최종 상태

| 영역 | 상태 | 비고 |
| --- | --- | --- |
| 최신 Run 고정 | PASS | `20260721_full_fair_v1` |
| 최종 모델 통일 | PASS | CatBoost |
| 본문 Storyline | PASS | 12장, 문제→근거→선정→운영→다음 검증 |
| 슬라이드별 콘텐츠 | PASS | 질문·결론·수치·시각자료·주의 문구 포함 |
| 발표자 대본 | PASS | 각 장 30~60초 수준 |
| 예상 질문 | PASS | 17개 답변과 근거 연결 |
| Claim Source Map | PASS | VERIFIED·PARTIALLY_VERIFIED·UNVERIFIED·EXCLUDE 사용 |
| Visual Asset Map | PASS | v3 그래프·원본 CSV·사용 위치 연결 |
| 수치 일관성 | PASS | Metadata·Run·v3 CSV 교차 확인 |
| 인과·Uplift 경계 | PASS | 원인·효과·ROI 과장 금지 |
| 실제 PPT 렌더링 | NOT_APPLICABLE | 이번 범위는 콘텐츠·제작 사양이며 PowerPoint 미생성 |

## 슬라이드별 QA

| Slide | 하나의 주장 | 근거 수치 | 시각자료 | 한계 문구 | 전환 | 상태 |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | 예 | 데이터·모델 상태 | 단순 흐름 | 운영 완료 과장 금지 | 예 | PASS |
| 2 | 예 | FN·FP·정책 상태 | 균형축 | 불균형 주장 금지 | 예 | PASS |
| 3 | 예 | 규모·비율·ID·경계 | 데이터 범위 | 합성·기간·Holdout | 예 | PASS |
| 4 | 예 | 청취·구독·문의 | EDA·범주 연관 | 인과 금지 | 예 | PASS |
| 5 | 예 | 채택·제외 delta | Feature 실험 | CatBoost 효과 과장 금지 | 예 | PASS |
| 6 | 예 | Baseline→후보 | 성능 여정 | 지표 선택 근거 | 예 | PASS |
| 7 | 예 | 8개·7개·Top 3 | 모델 비교 | 과거 Run 혼합 금지 | 예 | PASS |
| 8 | 예 | CV·Seed·CI·Recall·Brier | Bootstrap | 압도적 승자 주장 금지 | 예 | PASS |
| 9 | 예 | 3개 운영 시나리오 | Threshold 곡선 | 0.35 자동 승인 금지 | 예 | PASS |
| 10 | 예 | Top-K Capture·Lift | Top-K·Decile | Uplift 해석 금지 | 예 | PASS |
| 11 | 예 | 6페이지·75,000명·재로딩 | 앱 캡처 사양 | 행동·비용 가정 구분 | 예 | PASS |
| 12 | 예 | 내부 검증·미검증 항목 | 4개 Gate | 외부 검증 부재 | 종료 | PASS |

## 시각자료 QA

### 본문 직접 사용 권장

- `01_performance_progression.png` — 단계별 PR-AUC와 F1 흐름이 명확하다.
- `03_adopted_rejected_experiments.png` — 채택·제외 Feature 실험을 한눈에 보여준다.
- `04_eight_model_fair_comparison.png` — 8개 모델 동일 조건 비교를 직접 증명한다.
- `06_top3_fine_tuning_before_after.png` — Top 3 격차가 작다는 메시지에 적합하다.
- `08_bootstrap_confidence_intervals.png` — 상위 세 모델 구간 중첩을 명확하게 전달한다.
- `10_threshold_precision_recall_f1.png` — Threshold Trade-off를 설명하기 적합하다.
- `14_topk_capture_lift.png` — 용량 기반 우선순위 메시지에 적합하다.
- `15_risk_decile.png` — 순위 분리와 합성 데이터 경고를 함께 설명할 수 있다.
- `20_categorical_churn_associations.png` — Free와 문의 수준 인사이트를 동시에 보여준다.

### 수정 또는 재구성 후 사용

- `02_preprocessing_experiments.png` — 0 근처 음수 값의 레이블이 y축 레이블과 겹친다. 원본 CSV를 사용해 PPT에서 다시 그리거나 `03_adopted_rejected_experiments.png`로 대체한다.
- 모든 v3 그래프의 제목과 축은 영어다. 한국어 발표에서는 원본 CSV로 한국어 재작성하거나, 영어 차트를 유지할 경우 슬라이드 제목·한글 콜아웃으로 의미를 보완한다.
- `01_performance_progression.png`의 CatBoost search가 fine-tuned보다 소폭 높은 점을 숨기지 않고 선정 규칙 차이로 설명한다.
- `15_risk_decile.png`의 상·하위 극단값은 합성 생성 규칙의 영향이라는 경고를 같은 장에 둔다.

### 부록 사용 권장

- `05_seven_model_search_before_after.png`
- `07_seed_stability.png`
- `09_calibration_curve.png`
- `11_threshold_volume_fn_fp.png`
- `12_equal_contact_comparison.png`
- `13_equal_recall_comparison.png`
- `16_final_confusion_matrix.png`
- `17_final_feature_importance.png`
- `18_segment_error_analysis.png`
- `19_numeric_target_correlations.png`

## 슬라이드 제작 규격

- 제목은 결론형 한 문장으로 작성하고 한 줄을 넘기지 않는다.
- 제목 35pt 이상, 중간 제목 24pt 이상, 본문 16pt 이상을 유지한다.
- 한 슬라이드에 주 그래프는 한 개를 원칙으로 하고 보조 그래프는 메시지가 겹치지 않을 때만 추가한다.
- 본문 수치는 최대 3~5개만 강조하고 전체 표는 부록으로 이동한다.
- 그래프 아래에 `5-Fold OOF`, 표본 수, 주요 한계를 짧게 표기한다.
- 화면 캡처는 Streamlit 전체 화면을 나열하지 않고 운영 시나리오와 고객 우선순위 2개만 사용한다.
- 발표자 대본·시간·내부 제작 지시는 화면에 노출하지 않는다.

## 최종 제작 전 체크리스트

- [ ] 모든 그래프 제목과 축의 언어를 통일했다.
- [ ] Slide 11의 최신 Streamlit 캡처 2개를 새로 촬영했다.
- [ ] 각 그래프의 원본 CSV와 표본 범위를 발표자 노트에 연결했다.
- [ ] v2 LightGBM 그래프가 포함되지 않았다.
- [ ] “향후 30일”, “외부 Holdout 검증 완료”, “실제 ROI” 문구가 없다.
- [ ] 실제 PPT를 전체 렌더링해 겹침·잘림·한 줄 제목을 확인했다.
- [ ] 발표 시간에 맞춰 대본을 실제로 1회 이상 리허설했다.

## QA 결론

콘텐츠·수치·스토리라인·근거 파일 연결은 발표 제작 준비 상태다. 실제 PowerPoint 제작 후에는 별도로 전체 슬라이드 렌더링과 시각 QA를 수행해야 한다.
