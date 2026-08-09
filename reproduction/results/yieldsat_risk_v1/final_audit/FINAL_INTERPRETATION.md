# Final one-shot interpretation

Status: **FAIL_FINAL**.  No retuning or repeat is permitted.

The final audit passes integrity, pooled prediction/ranking, early curve,
calibration, and action-alert gates.  At look 3, 322 fields yield RMSE 1.058
versus country/crop climatology 1.448; the stratified bootstrap difference is
[-0.516,-0.269].  R2 is 0.836, low-yield AUROC 0.833, and AUPRC 0.625 at
prevalence 0.230.  Selected one-sided coverage is 0.953/0.988 at 90%/95%.

The persistent action alert has 8 TP and 5 FP among 74 low-yield and 248
non-low-yield fields: FPR 0.0202 (one-sided upper95 0.0419), sensitivity 0.108,
precision 0.615, and worst supported group FPR 0.040.

The sole failed frozen gate is strict worst-group R2 >0.  Eight of nine look-3
country/crop groups have positive R2; Brazil-wheat has n=14 and R2=-0.031
(RMSE 0.683).  Macro group R2 is 0.409.  The result must be reported as a narrow
worst-group failure, not rounded to a pass.

The audit is field-disjoint but not farm-disjoint because the legacy split
hashed fields.  Separate development unseen-farm analysis contains 59 fields
and is not a substitute for an untouched farm-disjoint final audit.

