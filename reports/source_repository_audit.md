# Source Repository Audit

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
- Candidate artifacts will be written under `models/candidates/20260721_submission_bounded_v1/`.
- Experiment checkpoints, metrics, figures, and a run manifest will be written under `experiments/submission_bounded/20260721_submission_bounded_v1/`.
- The supplied data does not document an observation date or outcome horizon. A 30-day prediction interpretation is therefore **UNVERIFIED**.

## Direct-comparison status

The current app model can be inspected as an implementation reference, but it is not automatically comparable to the bounded run until target, data version, feature schema, preprocessing, split, and threshold are all identical. The bounded run will record this explicitly rather than claiming equivalence.

## Next checkpoint

`repository_audit_complete` — run the isolated candidate workflow, preserving the current app and existing model artifacts.
