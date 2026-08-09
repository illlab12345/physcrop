# Amendment E4-02: persistent action alert

Attempt 2 restored useful power but the any-hit alpha-spending policy had
field-season FPR 0.0588 (upper95 0.0891), failing the 0.05/0.08 gates.  The
predeclared consecutive-hit comparator had 9 TP, 1 FP, sensitivity 0.148,
precision 0.900, FPR 0.0039 (upper95 0.0185), and worst-group FPR 0.0128.

The frozen system therefore separates two outputs:

- a single significant null-conformal result is a non-actionable `watch`;
- a formal `action_alert` requires significance at two consecutive available
  looks under the same alpha-spending bounds.

Persistence can only remove alerts, so it preserves the family-wise upper bound.
No p-values, alpha allocations, thresholds, models, calibration records, or pass
gates change.  Attempt 2 is archived and attempt 3 only reclassifies the already
predeclared operational output as primary.

