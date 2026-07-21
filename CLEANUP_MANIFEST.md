# 최종 정리 Manifest

정리일: 2026-07-22  
브랜치: `codex/final-submission-cleanup`

## 최종 보존 기준

- `app/`, `src/`, `data/`, `models/`: 실행 앱·최종 데이터·최종 CatBoost Pipeline과 공정 비교 후보
- `artifacts/`: 최종 모델 메타데이터, OOF 운영 지표, 발표용 원본 CSV
- `figures/presentation_v3/`: 최종 발표 그래프와 그래프 원본 inventory
- `outputs/PlaylistPro_final_presentation.pptx/.pdf`: 단일 최종 발표자료
- `submission_package/PlaylistPro_final_submission_20260722/`: 제출용 폴더·보고서·노트북·앱·모델·발표자료
- `reports/`, `docs/`, `PROJECT_REGISTER.md`: 최종 설명·감사·요구사항 추적과 재현 문서

## 프로젝트 폴더에서 제거한 중간 산출물

- project-centered·revised·submission-ready 이전 발표자료 3종
- `figures/presentation_v2/` 이전 발표 그래프
- `models/candidates/20260721_submission_bounded_v1/` 이전 bounded 후보
- `experiments/submission_bounded/` 이전 P0/bounded 실행 결과
- `artifacts/model/archive/` 이전 Pipeline 백업
- Full-fair 실행의 원시 OOF `.npy`, Fold `.npz`, 탐색용 `.joblib`, 중간 로그·부분 Trial
- `tmp/`, `.pytest_cache/`, `catboost_info/` 로컬 캐시와 렌더 임시파일
- 최종본으로 승격되지 않은 발표자료 생성 스크립트

삭제 대신 복구 가능성을 위해 위 중간 파일은 프로젝트 밖의
`C:\Users\TEST OS\Desktop\PlaylistPro_cleanup_archive_20260722\`에 이동했습니다.
프로젝트 폴더에는 최종본과 최종 근거만 남겼습니다.

## 검증

- 최종 발표자료: 16장 렌더링·overflow 검사 통과
- 제출 패키지: 모델 해시·노트북·PDF·PPTX·Manifest 검증 통과
- Streamlit: 6개 화면 예외 없음
- 테스트: 17개 통과
