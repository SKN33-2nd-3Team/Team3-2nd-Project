# Source Repository Audit

> **최초 조사 기록:** 현재 운영 모델은 CatBoost이며 최종 상태는 `final_local_model_integration.md`를 기준으로 합니다. 아래 내용은 P0 시작 시점의 저장소 상태입니다.

Audit time: 2026-07-21 (local repository snapshot)

## VERIFIED findings

| Area | Evidence | Finding |
|---|---|---|
| Data | `data/train.csv`, `data/test.csv` | Train has 125,000 rows and the `churned` target; test has 75,000 rows and no target. |
| Current application model | `app/streamlit_app.py:23,84` | The app loads `artifacts/model/music_churn_pipeline.joblib` through `joblib.load`. |
| Current model family | `src/run_pipeline.py:492` | The existing train script uses `sklearn.ensemble.GradientBoostingClassifier`. |
| Current preprocessing | `src/features.py`, `src/run_pipeline.py:257-275` | Feature engineering and preprocessing run inside a scikit-learn `Pipeline`. |
| Current split | `src/run_pipeline.py:478-481` | Existing training uses two stratified splits for Train/Validation/Internal Test. |
| Existing runtime | `requirements.txt` | Python package pins include scikit-learn 1.9.0, pandas 2.3.3, and numpy 2.3.4. |
| External boosting libraries | import check after explicit user approval on 2026-07-21 | Pinned `xgboost` 3.3.0, `lightgbm` 4.7.0, and `catboost` 1.2.10 are installed for the isolated candidate run only. |

## Scope boundary for the bounded modeling run

- The existing Streamlit code and the current operational artifact under `artifacts/model/` are implementation references only; this bounded run must not replace either one.
- Final candidate artifacts are retained under `models/candidates/20260721_full_fair_v1/` and mirrored in the submission package.
- The earlier bounded/P0 experiment directory was an intermediate checkpoint and was removed from the cleaned project tree after its evidence was copied into the final package.
- The supplied data does not document an observation date or outcome horizon. A 30-day prediction interpretation is therefore **UNVERIFIED**.

## Direct-comparison status

The current app model can be inspected as an implementation reference, but it is not automatically comparable to the bounded run until target, data version, feature schema, preprocessing, split, and threshold are all identical. The bounded run will record this explicitly rather than claiming equivalence.

## Next checkpoint

`repository_audit_complete` — run the isolated candidate workflow, preserving the current app and existing model artifacts.

## Final presentation authority update — 2026-07-22

최종 발표 콘텐츠 작성에서는 위 P0 시작 스냅샷을 현재 모델 근거로 재사용하지 않고 다음 순서로 권위 소스를 고정했다.

1. 완료 Run `experiments/submission_full_comparison/20260721_full_fair_v1/run_manifest.json`
2. 현재 모델 `artifacts/model/metadata.json`
3. 최종 선정 결정 `reports/final_model_selection_decision.md`
4. 발표 근거 `reports/presentation_evidence_pack_v3.md`
5. 원본 수치 `artifacts/presentation_v3/*.csv`
6. 발표 그래프 `figures/presentation_v3/*.png`
7. 데이터·검증 경계 `docs/data_card.md`, `docs/validation_plan.md`, `docs/requirements.md`

현재 최종 기술 후보는 CatBoost다. `figures/presentation_v2/`, 이전 bounded/P0 모델 결론, 통합 전 `reports/streamlit_presentation_storyline.md`의 Gradient Boosting·LightGBM·holdout 표현은 최종 발표 근거에서 제외한다. 독립 외부 라벨 Holdout은 없으며, 제공 test 75,000명은 추론에만 사용한다.

이번 조사 결과의 재사용 파일은 다음과 같다.

- `reports/final_presentation_consistency_audit.md`
- `artifacts/final_presentation_claim_source_map.csv`
- `artifacts/final_presentation_visual_asset_map.csv`
- `artifacts/final_presentation_slide_map.csv`
