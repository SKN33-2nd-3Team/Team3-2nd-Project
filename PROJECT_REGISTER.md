# PROJECT REGISTER

## Current full-fair run

- Run ID: `20260721_full_fair_v1`
- Status: `COMPLETED_CANDIDATE_AWAITING_BUSINESS_REVIEW`
- Candidate: `CatBoost`
- Candidate artifact: `models/candidates/20260721_full_fair_v1/candidate_pipeline.joblib`
- Validation: saved 5-fold OOF, 5 seeds, 1,000 bootstrap resamples
- Presentation evidence: `figures/presentation_v3/` with source CSV mapping
- Fresh-process reload and schema smoke test: `PASSED`
- Retraining during final evidence completion: `NO`
- Existing operational Streamlit model: `CATBOOST ALREADY PROMOTED IN 00c031f`
- Additional app replacement during this evidence completion: `NOT_PERFORMED`
- GitHub push: final-cleanup branch pending this task

## Selection summary

- Same common feature variant (`log_numeric`) and the same saved five folds were used for the eight-model comparison.
- Seven non-dummy models completed their bounded RandomizedSearch trial counts.
- The actual top three, CatBoost, XGBoost, and LightGBM, completed 15 fine-tuning trials each.
- CatBoost is the review candidate because it has the highest fine-tuned CV PR-AUC and five-seed mean PR-AUC. Top-three bootstrap intervals overlap, so this is a recommendation for the current objective, not proof of absolute superiority.

## Evidence boundaries

- No untouched external labeled holdout is available.
- No prediction horizon or churn event timestamp was supplied.
- Top-K/Lift are OOF ranking diagnostics, not campaign uplift.
- Feature importance and churn-rate differences are associations, not causal effects.
- Operating threshold, campaign action, candidate promotion, legal/data-rights approval, and production deployment remain user or organization decisions.

## Cleanup status

The earlier bounded P0 run, presentation-v2 assets, raw OOF arrays, and search-only model files were removed from the project tree on the cleanup branch. The final submission package and the full-fair run manifest remain.
