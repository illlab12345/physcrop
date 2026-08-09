# Experiment 3 plan: one-sided conformal calibration ladder

Frozen on 2026-08-01 before calibration outcomes are evaluated.

## Objective

Evaluate whether upper prediction bounds used for low-yield warnings attain
their nominal field-level coverage while remaining useful under heterogeneous
country/crop errors.

## Data and predictions

- Train models and preprocessing are the frozen Experiment 2 artifacts.
- The 125 farm-disjoint `conformal` fields are the only source of residual
  quantiles.  Model-fit and early-stop residuals are prohibited.
- `locked_external` is the development evaluation set; `audit_validation`
  remains unread.
- Every look is calibrated separately.  Only observations available by that
  look enter its frozen Experiment 2 predictor.

## Ladder

1. `global_raw`: one global quantile of `y - prediction`.
2. `crop_raw`: crop quantile for calibration n>=20, otherwise global.
3. `hierarchy_raw`: country/crop for n>=20, then crop n>=30, then global.
4. `hierarchy_scaled` (proposed): the same hierarchy applied to
   `(y-prediction)/s`, where `s` is the across-tree standard deviation in t/ha
   floored at 0.25 t/ha; the bound is `prediction + q*s`.

Quantiles use the finite-sample split-conformal rank
`ceil((n+1)*(1-alpha))`, clipped to the largest calibration score.  No
interpolation is used.  Alpha values 0.10 and 0.05 are evaluated here; the
alpha/3 sequential bound is reserved for Experiment 4.

## Metrics and gates

- Pooled one-sided coverage and mean/median upper slack (`upper-prediction`).
- Coverage for every development country/crop group with n>=10; macro and worst
  group are reported.
- Integrity: exact calibration/evaluation field disjointness, calibration farms
  disjoint from model-fit farms, causal look match, zero final-label reads.
- Proposed method coverage must be at least nominal-0.03 at both 90% and 95%
  for every look.
- Worst supported group coverage must be at least nominal-0.15.
- Proposed mean width averaged over looks may not exceed `global_raw` by more
  than 5% at either nominal level.

Coverage on the legacy development challenge is an empirical shift test, not a
distribution-free guarantee when exchangeability fails.

