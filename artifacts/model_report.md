# 최종 모델 보고서 참조

현재 운영 Pipeline은 `20260721_full_fair_v1`의 CatBoost 후보입니다.

- Fine 5-Fold OOF PR-AUC: `0.9479127421`
- OOF PR-AUC / ROC-AUC: `0.9478973395` / `0.9419585884`
- 5-seed 평균 PR-AUC: `0.9478803956`
- 기본 운영 시나리오: balanced F1, threshold `0.35`

전체 과정과 지표는 [reports/training_report.md](../reports/training_report.md)와
[submission_package/PlaylistPro_final_submission_20260722/reports/training_report.md](../submission_package/PlaylistPro_final_submission_20260722/reports/training_report.md)에 기록되어 있습니다.

이 문서는 과거 Gradient Boosting 실행 결과를 최종 모델로 해석하지 않도록 만든 참조 파일입니다. 제공 test에는 라벨이 없고 외부 Holdout·캠페인 uplift·ROI는 검증되지 않았습니다.
