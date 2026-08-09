# Frozen one-shot YieldSAT final audit protocol

Frozen after Experiments 1--9 and before any `audit_validation` yield magnitude
was read by the new yield-risk pipeline.

## Population and claim

The audit contains 322 feature-eligible fields from the legacy
`audit_validation` role.  It is field-disjoint from development fields but not
farm-disjoint: the legacy role was assigned by field hash.  This audit therefore
tests new fields, not guaranteed new farms.  One structurally ineligible field
is excluded without reading its outcome.

## Immutable system

- three crop-specific absolute-GDD looks and model-fit q20 thresholds from
  Experiment 2;
- early-stop-selected dual-tree predictor (HistGB weights 0/0/0.1 by look);
- 90/95% calibration policy: global raw at looks 1/3, hierarchical power-0.5 at
  look 2;
- null-conformal alpha spending (0.010/0.015/0.025); single hit is a watch and
  two consecutive hits are the formal action alert;
- every normalizer, imputer, calibration record, model, and threshold is frozen.

## One-shot endpoints and gates

Integrity:

- all frozen hashes match; exactly one execution; >=300 finite outcomes; no
  duplicate or development-overlapping field IDs.

Prediction/ranking:

- late-look RMSE beats country/crop climatology with stratified field-bootstrap
  upper 95% difference <0;
- late low-yield AUROC >=0.70 and AUPRC exceeds prevalence;
- macro supported country/crop R2 >0 and worst supported group R2 >0;
- at least two looks have AUROC >=0.65 and late RMSE <= early RMSE.

Calibration:

- selected one-sided coverage is >=0.87 at nominal 0.90 and >=0.92 at nominal
  0.95 at every look.

Action alert:

- field-season FPR <=0.05 and one-sided Clopper-Pearson upper95 <=0.08;
- >=3 true positives, sensitivity >=0.05, precision >=0.50, and worst supported
  group FPR <=0.15.

All gates are evaluated once.  Failure is final and cannot trigger tuning.

