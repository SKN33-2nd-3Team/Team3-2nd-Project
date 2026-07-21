# PlaylistPro Insight

고객별 구독·이용·문의 정보를 바탕으로 `churned` 라벨 위험을 순위화하고, 리텐션 담당자가 검토 대상 범위를 선택하도록 돕는 머신러닝 의사결정 지원 프로젝트입니다.

## 현재 최종 상태

| 항목 | 결과 |
| --- | --- |
| 최종 모델 | CatBoost |
| 공통 Feature | `log_numeric` |
| 평가 방식 | 고정된 5-Fold OOF |
| Fine-tuned CV PR-AUC | 0.947913 |
| OOF PR-AUC / ROC-AUC | 0.947897 / 0.941959 |
| 5개 Seed 평균 PR-AUC | 0.947880 |
| 학습 데이터 | 125,000명, `churned` 포함 |
| 점수 산출 데이터 | 75,000명, 정답 라벨 없음 |

CatBoost는 동일 Feature·동일 Fold로 비교한 8개 모델과 실제 상위 3개 정밀 탐색 결과를 근거로 선정했습니다. 독립된 외부 라벨 Holdout은 없으므로 성능 수치를 미래 일반화 성능이나 캠페인 효과로 표현하지 않습니다.

## 앱에서 할 수 있는 일

- 고객 인사이트와 Feature 중요도 확인
- 8개 모델 동일 조건 비교와 CatBoost 선정 과정 확인
- 재현율 우선·균형형 F1·정밀도 우선 시나리오 비교
- Top-K 포착률과 위험 Decile 진단 확인
- `test.csv` 고객 단건 재추론과 행동 가능 위험 신호 확인
- 고객 세그먼트별 1순위·대안 유지 활동, 실행 조건, 검증 KPI 제안
- 담당자의 행동 선택·직접 검토·보류와 검토 기록 CSV 생성
- 전략 세그먼트·행동 후보가 포함된 배치 우선순위 CSV 생성
- 구독 유형별 비용·가치 설정, 캠페인 퍼널, 연락 범위별 순편익 곡선을 포함한 가정 기반 계획

가정 기반 계획의 초기값은 Premium 연간 가치 120,000원을 팀 가정 기준점으로 삼아 Free·Student·Premium·Family의 기본 채널, 접촉 비용, 고객 가치를 상대 추정한 값입니다. 실제 가격·ARPU·마진 데이터가 아니며 화면에서 수정할 수 있습니다.

유지 활동은 관찰 연관성에 기반한 실험 후보이며 자동 조치가 아닙니다. 앱은 CRM 발송, 할인 제공, 고객 접촉, 계정 조치를 실행하지 않습니다.

## 기본 운영 시나리오

| 시나리오 | Threshold | 대상 비율 | Recall | Precision | FN | FP |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 재현율 우선 | 0.29 | 62.16% | 95.18% | 78.61% | 3,091 | 16,617 |
| 균형형 F1 | 0.35 | 60.91% | 94.44% | 79.59% | 3,569 | 15,538 |
| 정밀도 우선 | 0.74 | 38.77% | 71.80% | 95.07% | 18,099 | 2,388 |

위 값은 모두 저장된 5-Fold OOF 예측에서 계산한 의사결정 참고값입니다. 어느 시나리오를 실제로 사용할지는 접촉 가능 인원과 FN·FP 비용을 바탕으로 사용자가 결정합니다.

## 실행

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

앱은 모델을 다시 학습하지 않습니다. `artifacts/model/music_churn_pipeline.joblib`의 SHA-256을 메타데이터와 비교한 뒤 검증된 파일만 로드합니다.

## 핵심 산출물

| 목적 | 파일 |
| --- | --- |
| 현재 모델 메타데이터 | [artifacts/model/metadata.json](artifacts/model/metadata.json) |
| 최종 통합 결과 | [reports/final_local_model_integration.md](reports/final_local_model_integration.md) |
| 모델 선정 근거 | [reports/final_model_selection_decision.md](reports/final_model_selection_decision.md) |
| 학습·운영 결과 | [reports/training_report.md](reports/training_report.md) |
| 검증 계획과 한계 | [docs/validation_plan.md](docs/validation_plan.md) |
| 발표 근거 | [reports/presentation_evidence_pack_v3.md](reports/presentation_evidence_pack_v3.md) |
| 현재 Streamlit 앱 | [app/streamlit_app.py](app/streamlit_app.py) |

## 해석 한계

- `churned`의 관측 기준일과 결과 기간이 없어 “향후 30일 이탈”처럼 해석할 수 없습니다.
- OOF·Seed·Bootstrap은 내부 안정성 근거이며 독립 외부 Holdout을 대신하지 않습니다.
- Feature 중요도와 세그먼트 차이는 연관성이지 인과관계가 아닙니다.
- 유지 전략 규칙은 행동 가능한 서비스·이용·구독 신호만 사용하며 나이·지역·고객 ID를 행동 배정 근거에서 제외합니다.
- 제안된 유지 활동의 KPI는 향후 캠페인에서 검증할 측정 계획이지 현재 확인된 효과가 아닙니다.
- 실제 캠페인 Uplift·ROI는 A/B 테스트와 미래 결과 라벨로 별도 검증해야 합니다.
- 데이터 라이선스·법적·조직적 적합성의 최종 승인은 담당 조직의 판단이 필요합니다.

원본 `data/*.csv`는 수정하지 않습니다. GitHub에는 자동 Push하지 않으며 현재 작업은 로컬 브랜치에서 관리합니다.
