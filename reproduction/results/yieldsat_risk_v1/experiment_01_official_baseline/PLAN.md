# Experiment 1 plan: public YieldSAT LSTM baseline

Frozen on 2026-08-01 before training.

## Purpose

Establish a traceable baseline from the public YieldSAT ML tutorial and verify
that our flexible-format adapter, grouped split, training loop, and field-level
evaluation are sound.  The public tutorial, not an unreleased benchmark codebase,
is the reproducible reference.

## Public reference and faithful elements

- 12 Sentinel-2 bands in the tutorial order.
- 24 uniformly sampled time steps for this full-season reproduction only.
- one-layer LSTM, hidden size 64, last-state linear regression head.
- MSE, Adam at 1e-3, 15 epochs.
- grouped splitting so a field's pixels cannot cross train/validation.
- pixel and field RMSE/R2 definitions when pixel predictions are produced.

The downloaded dataset is the flexible raw format, not the tutorial's
preprocessed NetCDF.  We therefore report a clearly named **field-mean raw-format
adaptation**.  It is not presented as an exact numerical reproduction of the
CVPR table, which used additional unpublished/unsupplied benchmark machinery.

## Execution

1. Build a cached acquisition table from raw Sentinel-2/SCL, weather, soil, DEM,
   metadata, and development yields.  Do not read `audit_validation` yields.
2. Use SCL classes 4/5 and nonzero image support; calculate per-field band means.
3. Resample the full season to 24 uniform time positions for this baseline.
4. Standardize predictors and target using model-fit statistics only.
5. Train five deterministic seeds and retain epoch by early-stop loss; evaluate
   once on `locked_external` development fields.
6. Save per-field predictions, seed metrics, ensemble metrics, provenance, and
   hashes.

## Integrity gates

- zero `audit_validation` outcome reads;
- disjoint field IDs between fitting and development evaluation;
- all normalizer statistics learned from fitting data only;
- deterministic rerun of a fixed seed within numerical tolerance;
- finite predictions for at least 95% of eligible development fields.

## Scientific gates

- ensemble RMSE beats the crop/country training-mean predictor;
- R2 is reported per supported country/crop and macro-averaged, including
  negative values rather than suppressing them;
- 95% field-bootstrap CI is reported for the RMSE difference;
- if the LSTM does not beat the simple baseline, Experiment 1 fails honestly and
  Experiment 5 supplies stronger comparators; it is not tuned on final data.

