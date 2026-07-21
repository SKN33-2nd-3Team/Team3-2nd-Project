# PROJECT REGISTER

## Current bounded-run status

- Run ID: `20260721_submission_bounded_v1`
- Checkpoint: `candidate_saved_and_reloaded`
- Status: `MODEL_CANDIDATE_SAVED_AWAITING_REVIEW`
- Existing app model replacement: `DEFERRED`
- Streamlit integration: `DEFERRED`
- Production promotion: `DEFERRED`

## Completed without rerun

- Source repository audit: `reports/source_repository_audit.md`
- Existing committed model comparison, threshold scenarios, Top-K/Lift, and risk-decile evidence materialized under `experiments/submission_bounded/20260721_submission_bounded_v1/`
- Feature and model screening tables materialized with `VERIFIED`, `INFERRED`, and `UNVERIFIED` status labels.
- Pinned XGBoost 3.3.0, LightGBM 4.7.0, and CatBoost 1.2.10 installed with explicit user approval.
- Isolated candidate comparison, bounded 3-trial/3-fold search, candidate save, and prediction-equivalence reload check completed.

## Review boundary

- Candidate: `models/candidates/20260721_submission_bounded_v1/candidate_pipeline.joblib`
- Candidate promotion, current-app replacement, Streamlit integration, prediction horizon, campaign uplift, and causal effect remain deferred or **UNVERIFIED**.

## Resume command

```bash
python scripts/run_submission_bounded.py
```

Next P0 work must preserve `data/`, `app/`, and `artifacts/model/` and write only to the bounded run and candidate paths.
