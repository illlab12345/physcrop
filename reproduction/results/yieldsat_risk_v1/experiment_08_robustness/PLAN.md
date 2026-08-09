# Experiment 8 plan: temporal, farm, country, and worst-group robustness

Frozen on 2026-08-01 before robustness fitting.

## Evaluations

1. **LOYO**: for each year 2016--2024, exclude every field from that year;
   recompute crop GDD looks, target normalization, q20, and imputers from
   remaining model-fit fields; choose dual-tree weight only on remaining
   early-stop fields; evaluate all development fields in the held year.
2. **Unseen-farm challenge subset**: use the frozen main model on the 59
   `locked_external` fields from 16 farms absent from model-fit.  This is the
   cleanest available farm-generalization estimate; it is small and reported
   with uncertainty.
3. **Leave-one-country-out**: train on other countries using crop-level target
   normalization and evaluate only crops represented in both source and held
   country.  Germany rapeseed is structurally excluded because no source-country
   rapeseed exists.
4. Report every supported country/crop and year, macro averages, and the worst
   group; never hide negative R2.

All folds use the third causal absolute-GDD look and the frozen dual-tree model
family.  No audit-validation labels are read.

## Gates

- LOYO covers >=1,500 fields, has positive macro-year R2, and beats its training-
  group climatology with an upper field-bootstrap RMSE-difference bound <0;
- unseen-farm support >=50, R2>0, and low-yield AUROC>=0.65;
- leave-one-country-out evaluates at least three countries and beats crop
  climatology in at least two held countries;
- the frozen challenge model's worst supported country/crop R2 remains >0;
- all exclusions and legacy farm overlaps are explicit.

These are development robustness estimates, not the untouched final audit.

