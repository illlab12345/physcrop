# Amendment E3-01: tempered uncertainty scaling

Filed after development attempt 1 failed, before attempt 2.  Final-audit labels
remain unread.

Attempt 1's `hierarchy_scaled` attained or exceeded all pooled and worst-group
coverage gates, but its mean upper adjustment was 1.120x / 1.082x global raw at
90% / 95%, failing the frozen <=1.05 efficiency gates.  The complete failed
report is retained as `report_attempt1.json` when attempt 2 executes.

The failure is consistent with noisy across-tree standard deviation entering
linearly.  Attempt 2 adds `hierarchy_power50`: calibration scores are

`(y - prediction) / sqrt(max(tree_sd, 0.25))`

and the prediction-time adjustment is the selected quantile times the same
square-root scale.  Cell hierarchy, support thresholds, calibration fields,
nominal levels, models, endpoints, and all pass gates are unchanged.  The
linear-scale result remains reported as an ablation.  No search over exponents
is performed; 0.5 is the sole amended value.

