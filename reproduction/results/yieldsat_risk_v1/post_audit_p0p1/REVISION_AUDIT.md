# P0/P1 revision audit

## Outcome

All protocol-integrity gates in `PLAN.md` passed. The immutable final audit was
not overwritten. New decision results are explicitly labeled post-audit
corrections or post-audit uniform-policy comparisons.

## P0-A

- Added two-sided exact 95% intervals for audit FPR, sensitivity, and precision.
- Recomputed all looks on the common 300-field cohort. Look 3 versus Look 1 has
  paired RMSE, AUROC, and AUPRC intervals of [-0.2527, -0.0537],
  [0.0446, 0.1417], and [0.0652, 0.2371].
- Verified actual pre-harvest timing from metadata and used acquisitions. Median
  lead is 107, 86, and 62.5 days; no evaluated prefix is post harvest.
- Counted 321 seen-farm and one unseen-farm audit field. No unstable unseen-farm
  estimate is reported.

## P0-B

- Rebuilt the empirical null from the conformal role only. Early-stop and
  model-fit fields are excluded; null support is 96/97/97 non-low-yield fields.
- Applied identical alpha spending and persistence to climatology, a fixed
  NDVI+NDMI+rain Ridge, frozen ExtraTrees, frozen HGB, and the frozen dual-tree.
- The clean-null dual-tree watch has FPR/sensitivity/precision
  0.0484/0.3108/0.6571. Action has 0.0161/0.1081/0.6667.
- Added 10,000-replicate farm-cluster bootstrap intervals.
- Added five-fold farm-grouped OOF over 1,848 fields and 81 farms: RMSE 1.2224,
  R2 0.8120, AUROC 0.8116, AUPRC 0.5435, with zero farm overlap in every fold.

## P1

- Compared calendar day-of-season, acquisition rank, normalized observed-season
  fraction, and absolute GDD on the same 316 fields and causal prefix.
- GDD has the best RMSE (1.0361), NRMSE (0.3698), RMSE skill (0.2921), AUROC
  (0.8112), AUPRC (0.5735), and macro-group R2 (0.4477).
- GDD significantly improves over calendar time. Its differences from rank and
  normalized fraction cross zero and are described only as point-estimate gains.
- Added normalized, climatology-relative, macro, and worst-group reporting.
- Expanded related work to early-stage forecasting, field/subfield multimodal
  fusion, and within-field yield mapping.

## Manuscript checks

- `paper.md` and `table.md` distinguish frozen evidence from post-audit evidence.
- Figure 3 shows the original frozen action and clean-null watch/action points.
- Numeric-token consistency and the presence of all four manuscript figures were
  checked automatically.
- All new scripts compile successfully and save predictions plus JSON reports.

## Residual review risk

The revision removes the avoidable calibration, baseline, interval, cohort, and
timing weaknesses. It cannot manufacture a second independent audit: only one
audit field is from an unseen farm, the conformal-only correction was performed
after outcome reveal, and action sensitivity remains 10.8%. These boundaries are
now explicit. The evidence is substantially closer to a defensible WACV
Applications weak-accept case, but acceptance cannot be guaranteed.

