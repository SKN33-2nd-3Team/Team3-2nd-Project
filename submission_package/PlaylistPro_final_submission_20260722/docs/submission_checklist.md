# 제출 체크리스트

최종 갱신일: 2026-07-21. 로컬 브랜치 `codex/decision-ready-retention` 기준입니다.

## 핵심 산출물

| 산출물 | 위치 | 상태 |
|---|---|---|
| 데이터·전처리 보고서 | [reports/preprocessing_report.md](../reports/preprocessing_report.md) | 완료 |
| 모델·운영 보고서 | [reports/training_report.md](../reports/training_report.md) | 완료 |
| 최종 Pipeline | `artifacts/model/music_churn_pipeline.joblib` | 완료 |
| 모델 메타데이터 | [artifacts/model/metadata.json](../artifacts/model/metadata.json) | 완료 |
| 운영 시나리오 | [artifacts/operating_scenarios.csv](../artifacts/operating_scenarios.csv) | 완료 |
| Streamlit 의사결정 지원 앱 | [app/streamlit_app.py](../app/streamlit_app.py) | 완료 |
| 발표자료 PPTX | [outputs/PlaylistPro_retention_decision_deck.pptx](../outputs/PlaylistPro_retention_decision_deck.pptx) | 완료 |
| 발표자료 PDF | [outputs/PlaylistPro_retention_decision_deck.pdf](../outputs/PlaylistPro_retention_decision_deck.pdf) | 완료 |

## 로컬 검증 완료

- [x] `python -m src.run_pipeline`으로 모델·지표·시나리오 산출물 재생성
- [x] `python scripts/eda_insight.py`, `python scripts/business_levers.py` 실행
- [x] `python -m unittest discover -s tests -p "test_*.py" -v` 통과 (3건)
- [x] Streamlit 6개 화면과 단건 점수 AppTest 실행 오류 0건
- [x] PPTX 슬라이드 오버플로 검사 통과
- [x] PDF 10페이지 렌더링·대표 페이지 시각 검토

## 제출 전 사람이 최종 확인할 항목

- [ ] 파일명·제출 경로가 운영 주체의 최신 제출 규칙과 일치하는지 확인
- [ ] 데이터 라이선스, 개인정보, 외부 배포 권한에 대한 최종 승인
- [ ] 실제 고객 접촉을 한다면 캠페인 승인·대조군·측정 기간 확정
- [ ] 외부 PC에서 `pip install -r requirements.txt` 후 앱 실행 확인

실제 CRM 실행, 캠페인 uplift, 법적·조직적 최종 승인은 이 저장소의 자동화 범위 밖입니다.
