# Experiment 7 plan: auxiliary within-field low-yield maps

Frozen on 2026-08-01 before pixel-residual model fitting.

## Claim boundary

The formal product remains a field-level risk alert.  This experiment asks only
whether an early look can provide a useful **relative within-field** map.  It
does not claim pixel-iid conformal coverage.  Splits, bootstrap units, and main
sample sizes are fields.

## Frozen construction

- Use the last clear Sentinel-2 acquisition at or before the crop's third
  absolute-GDD look; no later image is read as a feature.
- For each model-fit field, target is pixel yield minus that field's observed
  mean.  Features are within-field standardized 12 bands, four vegetation
  indices, eight soil summaries, and five terrain variables.
- Deterministically sample at most 256 valid pixels per model-fit field so large
  fields cannot dominate.
- Fit a HistGradientBoosting residual model (300 iterations, leaf 50, L2=1) and
  an NDVI Ridge residual baseline.
- On each challenge field, add the predicted, mean-centered residual map to the
  frozen field-level dual-tree prediction.  Centering must preserve the field
  prediction exactly.  Compare with a spatially uniform field prediction.

## Evaluation

- Per-field pixel RMSE and within-field R2/rho;
- per-field AUROC for identifying the observed bottom-yield quintile;
- field-bootstrap CI for mean RMSE difference versus uniform and NDVI Ridge;
- deterministic qualitative fields chosen by SHA-256 before map outcomes, saved
  with predictions and targets.

## Passing gates

- >=250 challenge fields and >=100,000 challenge pixels;
- mean-centered map preserves each field prediction within 1e-5 t/ha;
- residual model mean per-field RMSE beats uniform with upper paired field-
  bootstrap 95% bound below zero;
- median within-field Spearman rho >=0.20 and bottom-quintile AUROC >=0.65;
- no pixel-level confidence interval or p-value treats pixels as independent.

