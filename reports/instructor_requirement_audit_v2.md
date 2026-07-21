# Instructor Requirement Audit v2

| Instructor requirement | Evidence | Status |
| --- | --- | --- |
| Concrete performance-improvement process | Feature register, preprocessing experiments, baseline, full search, fine tuning | 충족 |
| Visual performance evidence | CSV evidence listed in `presentation_evidence_pack_v3.md`; figure generation remains pending | 부분 충족 |
| Churn operating criteria | OOF Recall, FN, Precision, contact capacity, Top-K/Lift, decile, calibration, segment errors | 충족 |
| Fair eight-model comparison | Same common features and saved 5-fold assignments | 충족 |
| Full tuning protocol | Seven searches and three 15-trial fine-tunes recorded | 충족 |
| Stability / uncertainty | Five seeds and 1,000 bootstrap resamples | 충족 |

The final candidate is locally integrated into the app as CatBoost. Its evidence scope remains five-fold OOF; the integration does not claim an external holdout, campaign uplift, or production-business approval.
