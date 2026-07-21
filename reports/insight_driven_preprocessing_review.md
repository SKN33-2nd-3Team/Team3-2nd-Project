# Insight-Driven Preprocessing Review

```text
          variant    feature_change  prediction_time_available                        leakage_risk  logistic_pr_auc          decision                      reason
              raw               raw                       True none; deterministic raw inputs only         0.899495           exclude   lower logistic OOF PR-AUC
     plus1_ratios      plus1_ratios                       True none; deterministic raw inputs only         0.899486           exclude   lower logistic OOF PR-AUC
zero_aware_ratios zero_aware_ratios                       True none; deterministic raw inputs only         0.899485           exclude   lower logistic OOF PR-AUC
      log_numeric       log_numeric                       True none; deterministic raw inputs only         0.906293 provisional_adopt highest logistic OOF PR-AUC
      interaction       interaction                       True none; deterministic raw inputs only         0.899637           exclude   lower logistic OOF PR-AUC
    signal_pruned     signal_pruned                       True none; deterministic raw inputs only         0.899493           exclude   lower logistic OOF PR-AUC
```

Ratio variants compare +1 smoothing, zero-aware indicators, and transformed alternatives; no target-derived encoding is used.
