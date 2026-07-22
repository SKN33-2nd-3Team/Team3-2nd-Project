# 제출 점검표

> 기준 공지: 고객 이탈 예측 단위 프로젝트 가이드  
> 점검일: 2026-07-21

## 필수 산출물

| 산출물 | GitHub 제출 형식 | 현재 파일 | 상태 |
|---|---|---|---|
| 데이터 전처리 결과서 | Markdown | [`reports/preprocessing_report.md`](../reports/preprocessing_report.md) | 완료 |
| 인공지능 모델 학습 결과서 | Markdown | [`reports/training_report.md`](../reports/training_report.md) | 완료 |
| 학습된 최종 모델 | joblib | `artifacts/model/music_churn_pipeline.joblib` | 완료 |
| 모델 보조 파일 | JSON·CSV | `artifacts/model/metadata.json`, `artifacts/model_comparison_fair.csv` | 완료 |
| Streamlit 시연 | Python | `app/streamlit_app.py` | 완료 |
| README | Markdown | [`README.md`](../README.md) | 완료 |
| 발표자료 | PDF | 저장소에 없음 | 제출 전 추가 필요 |
| 프로젝트 전체 압축 | ZIP | Google Drive 제출용 | 제출 전 생성 필요 |
| 결과서 PDF | PDF | Google Drive 제출용 | 제출 전 변환 필요 |

## 전처리 결과서 포함 항목

- [x] 데이터 출처·라이선스·실제/합성 여부·규모
- [x] 분석 단위와 Target 0/1 정의
- [x] 결측·중복·자료형·이상값·클래스 비율
- [x] 주요 EDA와 시각화 링크
- [x] 데이터 정제와 Feature 생성 근거
- [x] 인코딩·스케일링·불균형 처리 여부
- [x] Train에만 전처리기를 fit하는 누수 방지 원칙
- [x] 최종 Feature 차원과 산출물 위치
- [x] 데이터 한계와 개선 방향

## 모델 학습 결과서 포함 항목

- [x] 이진 분류 문제와 Target 정의
- [x] Train·Validation·Test 비율과 분할 근거
- [x] Dummy 기준 모델
- [x] 동일 조건 후보 모델 비교
- [x] Precision·Recall·F1·ROC-AUC·PR-AUC
- [x] Validation에서 임계값 결정
- [x] 후보 모델별 Test 1회 평가와 최종 모델 Confusion Matrix
- [ ] FP·FN 사례 분석 — 팀원 협의 후 추가 예정
- [x] Feature Importance 해석 주의
- [x] 최종 모델 선정 근거
- [x] 딥러닝 제외 근거
- [x] 한계와 개선 방향

## 실행 점검

```bash
pip install -r requirements.txt
python scripts/run_full_fair_comparison.py
python scripts/finalize_full_fair_candidate.py
python scripts/promote_full_fair_candidate.py
streamlit run app/streamlit_app.py
```

- [x] 전처리기와 모델이 단일 Pipeline으로 저장됨
- [x] 저장 모델에 Feature 순서와 threshold 메타데이터가 함께 제공됨
- [x] Streamlit이 저장 모델을 로드하며 실행 시 재학습하지 않음
- [x] 신규 고객 1명 예측 화면 제공
- [x] 입력값 변경 시 실제 예측 확률 변경
- [x] 배치 예측과 CSV 다운로드 제공
- [ ] 다른 PC의 새 가상환경에서 설치·실행 최종 확인
- [ ] 발표용 화면 녹화 또는 스크린샷 준비

## Google Drive 제출 전

- [ ] `reports/preprocessing_report.md`를 PDF로 변환
- [ ] `reports/training_report.md`를 PDF로 변환
- [ ] 발표자료 PDF 추가
- [ ] 비밀키·가상환경·캐시·불필요한 대용량 파일 제외
- [ ] 프로젝트 전체 ZIP 생성 후 압축 해제·실행 확인
- [ ] README 수치와 `artifacts/model/metadata.json` 최종 대조
