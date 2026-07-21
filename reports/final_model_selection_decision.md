# Final Model Selection Decision

## Decision

`CatBoost` is the candidate for review, not an approved production model. It has the highest fine-tuned five-fold CV PR-AUC (0.94791) and highest five-seed mean PR-AUC (0.94788) among the actual top three.

## Why not select by a single point score only

CatBoost, XGBoost, and LightGBM are close. The decision uses fine-tuned CV PR-AUC, OOF PR-AUC, five-seed stability, Bootstrap confidence intervals, calibration, and threshold/capacity tables in `artifacts/`. The campaign owner must still choose an operating threshold because Recall, FN, and customer-contact capacity are policy choices.

## Limits

No untouched external or final holdout was available for this full comparison Run. Final evidence is five-fold OOF, stability, and bootstrap evidence. It does not verify a prediction horizon, campaign uplift, or causal effect.
