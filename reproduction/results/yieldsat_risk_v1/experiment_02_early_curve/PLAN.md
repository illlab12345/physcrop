# Experiment 2 plan: causal GDD early-prediction curve

Frozen on 2026-08-01 before Experiment 2 model fitting.

## Question

How early can final field yield and training-defined low-yield risk be predicted
without using any future-season information?

## Roles

Only `adapter_reference` is used to fit or calibrate.  Its farm groups are
assigned by SHA-256 to model-fit (<70), early-stop (70--84), and conformal
(85--99); this produces 1,254 / 154 / 125 fields from 57 / 11 / 11 farm groups.
`locked_external` is the development challenge set.  Legacy cross-role farm
overlap is disclosed.  `audit_validation` labels remain unread.

## Frozen looks

For each crop, compute the median maximum observed GDD using model-fit fields
only, without yields.  The three absolute looks are 0.35, 0.50, and 0.65 times
that training median.  Once written to `milestones.json`, test-field harvest
date, final GDD, and future acquisitions cannot alter them.  Expected structural
values from the development cache are approximately:

- corn: 936 / 1338 / 1739 GDD;
- rapeseed: 626 / 895 / 1163 GDD;
- soybean: 955 / 1364 / 1773 GDD;
- wheat: 569 / 812 / 1056 GDD.

Only observations whose cumulative GDD is at or below the look are features.
A field needs at least three usable acquisitions by the look.

## Predictor

The initial frozen predictor is a pooled phenology-conditioned ExtraTrees model
on target-normalized yield.  It uses six GDD-aligned causal samples of 12 bands,
NDVI/NDMI/NDRE/EVI, cumulative precipitation, valid fraction, static soil/DEM,
and country/crop one-hot indicators.  Normalizers, imputers, yield means/stds,
and low-yield q20 cutoffs are learned from model-fit fields only.  Separate
models are fit for the three looks; hyperparameters are fixed at 600 trees,
minimum leaf 3, maximum features 0.7, bootstrap false, and seed 2201.

This model is an early-curve probe, not yet the final claimed architecture.
Experiment 5 compares strong alternatives; Experiment 6 tests the contribution
of phenology alignment and modalities.

## Endpoints and gates

- Primary: macro country/crop RMSE and low-yield AUROC at each look on
  `locked_external`.
- Baseline: model-fit country/crop mean, applied without development outcomes.
- Integrity: zero future observations, zero final-label reads, farm-disjoint
  model-fit/early-stop/conformal roles, and >=80% eligible development coverage
  at every look.
- Scientific pass: the late-look RMSE beats climatology with the upper bound of
  a country/crop-stratified 95% field bootstrap below zero; late RMSE does not
  exceed early RMSE; and AUROC is >=0.65 at at least two looks.
- All eight available country/crop groups and the worst supported group are
  reported.  Pooled R2 alone is not a passing endpoint.

