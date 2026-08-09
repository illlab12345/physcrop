# Post-audit P0/P1 correction protocol

Status: frozen before running the analyses below.

This protocol does not alter, replace, or re-label the pre-reveal final audit. All
outputs under this directory are post-audit diagnostics or post-audit protocol
corrections. The original audit artifacts and their hashes remain immutable.

## P0-A: complete audit reporting

1. Report two-sided exact 95% Clopper--Pearson intervals for false-positive
   rate, sensitivity, and precision for both watch (any-hit) and action
   (two-consecutive-hit) policies.
2. Recompute Look 1--3 prediction metrics on the intersection of field IDs
   available at every look. Report sample size and prevalence at every look.
3. Measure pre-harvest lead time as harvest date minus the last acquisition used
   by each look. Report quantiles, crop-stratified medians, and the count of
   non-positive lead times.
4. Audit overlap with adapter-development farms. If an unseen-farm audit stratum
   is too small for inference, state this rather than presenting an unstable
   estimate.

## P0-B: clean null, fair baselines, and farm-level uncertainty

1. Rebuild empirical null distributions using only the held-out conformal
   subrole. Early-stopping and model-fitting fields are forbidden from null
   estimation. Low-yield fields are excluded from each country--crop null.
2. The score, hierarchy, candidate alpha grid, and persistence rules remain
   unchanged from Experiment 4 attempt 2. Alpha selection uses development data
   only; no audit outcomes may select a threshold.
3. Compare climatology, the fixed NDVI+NDMI+rain Ridge, Extra Trees, Histogram
   Gradient Boosting, and the dual ensemble under the same clean null,
   calibration hierarchy, alpha-selection rule, and persistence policy.
   Extra Trees and HGB predictions are extracted from the pre-reveal frozen
   Look 1--3 model bundles. Any baseline trained after reveal is labeled as a
   post-audit fixed-protocol comparison and is never called a frozen audit.
4. Compute field-level exact intervals and farm-cluster bootstrap intervals.
   Resample farms with replacement and include all their fields; use a fixed RNG
   seed and 10,000 replicates.
5. Run grouped out-of-fold prediction across the full development cohort with
   farms as indivisible groups. Report field count, farm count, fold support, and
   aggregate/macro performance. The audit's seen/unseen composition is reported
   separately and is not overstated.

## P1: timing alignment and completeness

1. Compare calendar-day, acquisition-rank, normalized observed-season fraction,
   and growing-degree-day alignment using identical model family, feature budget,
   training roles, and development challenge fields. No audit result is used for
   configuration selection.
2. Report RMSE, R2, AUROC, AUPRC, normalized RMSE, climatology skill, country--crop
   macro R2, and worst-group R2 with support counts.
3. Clarify the relationship to multimodal crop-yield forecasting, early-season
   forecasting, uncertainty calibration, and operational alerting. Claims are
   limited to what the experiments identify.
4. Update figures only when a new panel materially improves traceability; do not
   visually promote a post-audit correction as pre-registered evidence.

## Integrity gates

The following gates, rather than performance-dependent thresholds, determine
whether an analysis is eligible for the paper:

- audit field IDs exactly match the immutable final predictions;
- no audit label influences training, calibration, alpha selection, or model
  selection in the corrected comparison;
- the clean null contains conformal-role non-low-yield fields only;
- all denominators and support counts are saved with the metrics;
- all confidence intervals are finite or explicitly marked undefined for zero
  denominator;
- every acquisition used precedes harvest; exceptions, if any, are enumerated;
- scripts, configuration, predictions, and machine-readable reports are saved;
- adverse or non-monotone results are reported rather than tuned away.

