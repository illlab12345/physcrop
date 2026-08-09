# PhysCrop-Risk frozen master protocol (v1)

Frozen before model fitting on 2026-08-01 (Asia/Shanghai).

## Scientific question

Can partial-season Sentinel-2, weather, soil, and terrain observations issue a
calibrated warning that a field's final yield will fall below a training-defined
low-yield threshold?  This is a yield-risk warning task.  It is not disease,
stress-cause, or agronomic diagnosis.

## Data roles and blindness

- `adapter_reference`: development pool.  It may be subdivided, by deterministic
  farm group, into model-fit, early-stop, and conformal-calibration partitions.
- `locked_external`: development-only challenge pool because its outcomes were
  inspected in earlier work.
- `audit_validation`: one-shot final field-disjoint audit.  Its yield magnitudes
  must not be read by Experiments 1--8.  It is not farm-disjoint: the legacy
  split was made by field hash and 57/82 farms span roles.  No claim of unseen-
  farm independence may be made from this audit.
- Farm-disjoint (LOFO) and year-disjoint (LOYO) robustness are evaluated only
  inside development data and reported explicitly as development estimates.

## Causal observation rule

At a prediction look, only acquisitions and weather dated on or before that look
may be used.  Early looks are absolute, crop-specific GDD cutoffs estimated from
the development training partition.  Harvest dates, final-season GDD, future
cloud patterns, and test-field outcome statistics may not define a look.

## Primary endpoint

Field-level low-yield warning.  The low-yield cutoff is the 20th percentile of
yield in the model-fit partition, computed within a predeclared hierarchy:
country/crop when support is adequate, otherwise crop, otherwise global.
The system warns only when the one-sided conformal upper bound is below the
applicable cutoff.  Three sequential looks share a family-wise alpha of 0.05
using Bonferroni alpha/3 unless Experiment 4's already-frozen alternative is
selected on development data.

## Metrics

- Prediction: field-level RMSE, MAE, R2, and Spearman rho, with field bootstrap
  confidence intervals.
- Risk ranking: AUROC and AUPRC; AUPRC is always paired with prevalence.
- Calibration: one-sided coverage, interval width, and group-wise coverage.
- Alerting: field-season FWER/FPR among non-low-yield fields, sensitivity,
  precision, specificity, and median lead time.
- Robustness: macro country/crop scores and worst supported group, never only a
  pooled micro average.

## Adaptation and final-use rule

Architecture, preprocessing, GDD looks, hierarchy, minimum group sizes,
hyperparameters, calibration method, alert persistence, and all baselines are
chosen using `adapter_reference` and `locked_external` only.  After the model
card and hashes are frozen, `audit_validation` is revealed and run once.  A
failed final audit is reported as failed; it does not trigger tuning.

## Spatial claim boundary

Experiment 7 may visualize 10 m relative risk and evaluate field-clustered
spatial metrics.  Pixels from a field never cross a split.  Formal conformal
coverage and bootstrap units remain fields; pixels are not treated as iid.

## Passing rule

"Pass" means the predeclared statistical and integrity gates for that experiment
are met.  It does not mean guaranteed WACV acceptance.  Failed gates remain in
the record.  Optimizing a design after seeing development results requires a new
numbered development amendment; changing a final-audit design is forbidden.

