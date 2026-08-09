# Experiment 1 audit

Status: **PASS** under the frozen Experiment 1 gates.

- Development fields evaluated: 218 across the five country/crop tasks shown in
  the public YieldSAT benchmark.
- LSTM RMSE / R2 / rho: 1.180 / 0.819 / 0.853.
- Country/crop training-mean RMSE / R2: 1.319 / 0.774.
- Stratified bootstrap 95% CI for RMSE(LSTM)-RMSE(mean):
  [-0.209, -0.075].
- Finite prediction and eligible-field coverage: 100% / 100%.
- Deterministic repeat maximum absolute prediction difference: 0.
- `audit_validation` target reads: 0.

Important limitation: pooled R2 is inflated by between-task yield differences.
Within-task R2 is negative for Brazil-corn and Germany-rapeseed and near zero for
Germany-wheat.  The macro within-task R2 is 0.164.  This baseline therefore
passes as a faithful public-tutorial baseline, not as the proposed strong model.
Experiment 5 must report and improve the within-task/worst-group results.

