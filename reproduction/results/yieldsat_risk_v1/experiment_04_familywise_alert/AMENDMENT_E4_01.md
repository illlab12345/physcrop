# Amendment E4-01: null-conditional conformal p-values

Attempt 1 is retained as `report_attempt1.json` by the amended runner.  Regression
upper bounds controlled false alerts at zero but also produced zero true alerts,
failing both sensitivity and precision gates.  Final-audit labels remain unread.

Attempt 2 changes the formal test to match the decision boundary.  For each look,
the score is `(q20 - prediction) / training_group_yield_std`; larger values mean
stronger evidence of low yield.  Among held-out calibration fields whose observed
yield is above q20, the one-sided null-conformal p-value is

`p=(1 + count(null_score >= test_score))/(n_null + 1)`.

Under null exchangeability this is super-uniform.  Applying the already frozen
alpha-spending vector and a union bound therefore controls field-season false
alerts directly, without demanding 98% coverage of the entire continuous yield.

Because Experiment 2 used fixed hyperparameters and did not fit on either role,
the previously unused farm-disjoint `early_stop` and `conformal` roles are pooled
for null calibration (279 fields before look eligibility).  Neither role is used
to refit the predictor.  Score, pooling, alpha vectors, q20 thresholds, metrics,
and original pass gates are frozen before attempt 2.  The regression-UCB result
remains an explicit conservative ablation.

