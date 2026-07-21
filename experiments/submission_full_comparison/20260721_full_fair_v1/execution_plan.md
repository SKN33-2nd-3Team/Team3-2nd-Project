# Submission Full Fair Comparison Run — Execution Plan

## Data and validation boundary

- Source: `data/train.csv` (125,000 labeled rows); `data/test.csv` has no target.
- Independent labeled final holdout: **NOT AVAILABLE** after repository audit.
- The P0 internal holdout, its LightGBM, and all P0 artifacts are preserved as reference only and are not used to select this Run's model, features, parameters, or threshold.
- This Run uses saved, shared 5-fold stratified assignments, out-of-fold (OOF) predictions, five-seed stability, and 1,000 bootstrap samples.

## Execution order and resume behavior

1. Persist fold assignments, current-insight inventory, and preprocessing register.
2. Run raw/engineered Logistic feature experiments, then verify candidate transformations with the strongest default tree family.
3. Lock one common feature set and compare all eight required models under the same input, folds, and OOF metric computation.
4. Run the full 20/20/24/24/30/30/24-trial five-fold RandomizedSearch plan; retain every `cv_results_` row, including failures.
5. Select the actual top three from search results, fine-tune all three for 15 trials, and run seed, bootstrap, calibration, operational, and segment diagnostics.
6. Save/reload only the selected candidate, run an isolated schema smoke test, then write v3 evidence and instructor audit.

No stage is marked passed without its required output file. If stopped, the manifest identifies the last complete checkpoint and the same command resumes from completed artifacts.

## User-approved Random Forest lightweight adjustment

On 2026-07-22 the user approved a lighter Random Forest search because the original tree-count/depth space made each 5-fold trial excessively slow on CPU. Trial count remains 24 and CV remains 5-fold. The final search space is `n_estimators={10,20,30}`, `max_depth={4,6,8}`, `min_samples_leaf={20,40,80}`, and `max_features={sqrt,0.4}`. The single completed pre-adjustment trial is retained separately and excluded from the post-adjustment ranking.


## Data/feature guardrails

- Raw source CSVs and the current app/P0 candidate are never modified.
- Transforms are deterministic and prediction-time computable; target encoding is excluded.
- `age` is retained only as an existing context feature and flagged for fairness review; no causal or campaign-impact claim is made.
