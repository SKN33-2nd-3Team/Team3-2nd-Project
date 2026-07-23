# 최종 제출 체크리스트

## 자동·기술 검증

- [x] Train 125,000행, Test 75,000행 확인
- [x] Target 0/1 의미와 예측 기간 미정의 기록
- [x] 데이터 출처 URL·다운로드 기록·해시 작성
- [x] 실제·합성 판정 근거와 일반화 한계 작성
- [x] 결측·중복·이상·논리 위반 기록
- [x] EDA 그래프마다 관찰·다음 결정·한계 작성
- [x] Train Fold 내부 Pipeline과 누수 방지 기록
- [x] 채택·제외 Feature 실험 기록
- [x] Dummy 포함 8개 모델 동일 조건 비교
- [x] 7개 모델 탐색과 실제 Trial 수 기록
- [x] Top 3 정밀 탐색·Seed·Bootstrap 기록
- [x] Threshold·Recall·Precision·F1·FN·FP·대상 고객 수 기록
- [x] Top-K Capture·Lift·Risk Decile 기록
- [x] 저장 모델 새 프로세스 재로딩·신규 1행 예측
- [x] Feature schema·metadata·metrics 생성
- [x] Streamlit이 재학습 없이 저장 모델을 호출
- [x] Streamlit 6개 화면 예외 없음
- [x] README 실행 명령과 실제 경로 일치
- [x] Notebook 3종 실행 출력 포함·오류 없음
- [x] 전처리·모델 결과서 MD·PDF 생성
- [ ] 신규 발표자료 PPTX·PDF·발표자 노트 반영 및 검증
- [x] 제출 폴더 파일 Manifest·SHA-256 생성
- [ ] 신규 발표자료 반영 후 제출 ZIP 생성·무결성 검사
- [x] GitHub PR 브랜치 Push 및 CI 검증

## 제출 담당자가 직접 확인

- [x] 팀명·팀원별 역할·회고 반영
- [x] Kaggle Data 페이지의 MIT 표시 확인 및 `data/LICENSE.md` 고지 포함
- [ ] 강사에게 제출할 Google Drive 폴더와 파일명 확인
- [ ] 실제 사업 Threshold를 기술 권고와 구분해 발표
- [ ] 캠페인 효과·ROI를 검증 결과로 오해할 표현이 없는지 최종 리허설
- [ ] ZIP을 Google Drive에 직접 업로드

외부 제출과 조직적 결정은 자동화하지 않았습니다. 세부 내용은 `docs/human_confirmation_required.md`를 확인합니다.
