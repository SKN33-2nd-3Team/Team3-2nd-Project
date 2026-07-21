# 강사 요구사항 감사 v2

| 강사 요구사항 | 관련 산출물 | 실제 근거 | 상태 | 부족한 내용 | 보완 필요 여부 |
| --- | --- | --- | --- | --- | --- |
| 성능 향상을 위한 구체적 과정과 논리 | `preprocessing_experiment_results.csv`, `random_search_summary.csv`, `top3_fine_tuning_summary.csv`, `presentation_evidence_pack_v3.md` | Raw baseline→특성 실험→8개 모델→7개 탐색→Top 3 정밀 튜닝의 순서와 변화량 기록 | 충족 | 없음 | 아니오 |
| 조치별 성능 변화 시각화 | `figures/presentation_v3/01~08`, `figure_inventory.csv` | 단계별 PR-AUC, 채택·제외 실험, 8개 모델, 탐색 전후, Seed, Bootstrap 그래프와 원본 CSV 존재 | 충족 | 없음 | 아니오 |
| 이탈 도메인의 Recall·FN·PR-AUC·대상 고객 수 반영 | `threshold_operating_scenarios_v2.csv`, `equal_contact_comparison.csv`, `equal_recall_comparison.csv`, `topk_lift_v2.csv` | Threshold별 Recall·Precision·F1·FN·FP·대상 고객 수와 동일 Contact/Recall 비교 | 충족 | 최종 운영 Threshold는 사업 결정 | 사용자 결정 |
| 최소 Baseline부터 최종 후보까지 성능 변화표 | `artifacts/presentation_v3/01_performance_progression.csv` | Raw Logistic, log feature, CatBoost baseline/search/fine-tuned 단계 기록 | 충족 | 없음 | 아니오 |
| 실험별 문제 관찰·가설·채택·제외 | `insight_preprocessing_register.csv`, `current_insight_inventory.csv` | 6개 특성 변형과 채택/제외 이유, 비인과적 Insight 한계 기록 | 충족 | 없음 | 아니오 |
| 변경 전후 ROC-AUC·PR-AUC·F1·Recall·Precision·FN·FP | `preprocessing_experiment_results.csv`, `model_comparison_fair.csv`, `random_search_summary.csv`, `top3_fine_tuning_summary.csv` | 각 주요 실험 단계에 공통 지표 저장 | 충족 | 없음 | 아니오 |
| 직전 실험 대비·최초 Baseline 대비 변화 | `01_performance_progression.csv`, `02_preprocessing_experiments.csv`, `05_seven_model_search_before_after.csv` | 순차 변화와 Raw 대비 delta 기록 | 충족 | 없음 | 아니오 |
| 제외 실험 최소 1개 | `03_adopted_rejected_experiments.csv` | log_numeric 외 raw, ratio, interaction, signal-pruned 실험 제외 사유 기록 | 충족 | 없음 | 아니오 |
| 8개 모델 동일 조건 비교 | `model_comparison_fair.csv`, 저장된 `cv_fold_assignments.csv` | 동일 `log_numeric` 특성과 동일 5-fold로 Dummy 포함 8개 비교 | 충족 | 없음 | 아니오 |
| 모델별 튜닝 전후 비교 | `05_seven_model_search_before_after.csv` | 7개 학습 모델의 Baseline과 RandomizedSearch 결과 연결 | 충족 | 없음 | 아니오 |
| 실제 Top 3 정밀 튜닝 | `top3_selection_matrix.csv`, `top3_fine_tuning_summary.csv` | CatBoost·XGBoost·LightGBM 각 15회 정밀 탐색 | 충족 | 없음 | 아니오 |
| Threshold 운영 시나리오 | `threshold_operating_scenarios_v2.csv`, 발표 그래프 10·11 | Recall-first, balanced F1, precision-first와 전체 sweep | 충족 | 운영 정책 미선택 | 사용자 결정 |
| Top-K Capture·Lift | `topk_lift_v2.csv`, 발표 그래프 14 | 접촉 상위 5·10·20·30·40%의 Capture·Precision·Lift | 충족 | 실제 캠페인 uplift 아님 | 아니오 |
| Risk Decile | `risk_decile_v2.csv`, 발표 그래프 15 | 10개 위험구간의 예측확률과 관측 이탈률 | 충족 | 외부 Holdout 보정 아님 | 아니오 |
| 최종 모델 선정 의사결정표 | `final_model_selection_matrix.csv`, `final_model_selection_decision.md` | Fine CV, OOF, Seed 안정성, Bootstrap, 운영표로 CatBoost 후보 추천 | 충족 | 최종 승인·Threshold는 사용자 결정 | 사용자 결정 |
| 최고 단일 점수 모델과 최종 선정 모델 차이 | `presentation_evidence_pack_v3.md` | CatBoost는 순위/안정성 후보, LightGBM은 0.50 Recall·FN에서 장점이 있음을 병기 | 충족 | 없음 | 아니오 |
| Validation과 Holdout 일반화 | `run_config.json`, `bootstrap_confidence_intervals.csv`, `seed_stability_summary.csv` | 5-fold OOF·5 Seed·1,000 Bootstrap 완료 | 부분 충족 | 독립 라벨 Holdout 자체가 제공되지 않음 | 새 라벨 데이터 확보 시 필요 |
| 발표용 그래프와 원본 CSV | `figures/presentation_v3/figure_inventory.csv` | PNG 20개, 대응 원본 CSV 19개(Threshold 그래프 2개가 동일 sweep CSV 공유) | 충족 | 없음 | 아니오 |
| 상관관계·범주별 연관성 | 발표 그래프 19·20, `eda_summary.json` | 수치형 Pearson 상관과 주요 범주별 관측 이탈률 저장 | 충족 | 인과관계 증거 아님 | 아니오 |
| 후보 모델 저장·독립 로드·앱 스키마 | `candidate_pipeline.joblib`, `app_schema_smoke_test.csv`, `streamlit_candidate_compatibility.md` | 새 Python 프로세스 로드, 순서 변경·미지 범주 허용, 누락·잘못된 수치 거부 | 충족 | 동일 Run CatBoost의 로컬 앱 승격은 기존 커밋 `00c031f`에서 완료; 이번 보완에서는 추가 교체 없음 | 운영·배포 승인은 사용자 결정 |

## 종합 판정

강사님의 핵심 세 요구사항인 **개선 과정**, **조치별 발표 그래프**, **이탈 도메인 운영 기준을 반영한 모델 선택 논리**는 모두 충족했다. 독립 라벨 Holdout은 원천 데이터가 없어 `부분 충족`이며, 이를 OOF 성능이나 기존 P0 Holdout으로 가장하지 않았다.

재학습은 수행하지 않았다. 기존 CSV, 저장된 OOF 예측, 후보 Pipeline을 재사용해 시각화·상관관계 원본표·스키마 테스트를 보완했다.
