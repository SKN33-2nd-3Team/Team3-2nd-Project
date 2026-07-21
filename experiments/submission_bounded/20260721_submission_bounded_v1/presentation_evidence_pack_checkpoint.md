# Presentation Evidence Pack — checkpoint 1

## Verified evidence reused without rerun

- Current committed comparison includes Dummy, Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.
- Current internal Test Gradient Boosting PR-AUC is recorded in `presentation_metrics.csv`.
- Threshold scenarios, Top-K/Lift, and risk deciles were copied with their original internal-holdout provenance.

## Bounded-run gaps

- XGBoost, LightGBM, and CatBoost are **UNAVAILABLE** because their packages are absent and no dependency was added.
- No separate bounded candidate has yet been trained, saved, and reloaded; candidate status remains **DEFERRED**.
- The supplied data does not independently verify a 30-day prediction horizon or campaign uplift.

## Next P0 work

Run only the isolated bounded candidate comparison needed to create a separate candidate artifact, reload validation, and a final evidence pack. Do not modify the current app or operational model.
