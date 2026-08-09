# Amendment E3-02: development-selected stage calibration

Filed after attempt 2 failed only the efficiency gates.  No final-audit outcome
has been read.  Attempt 2 reduced the 90%/95% mean-adjustment ratios to
1.074/1.057, but the frozen ceiling is 1.05.

The per-look development audit shows that `global_raw` already passes pooled and
worst-group coverage at looks 1 and 3.  At look 2, its worst-group 90% coverage
is 0.72, below the 0.75 gate, while `hierarchy_power50` raises it to 0.80.
Therefore the frozen deployment ladder is:

- look 1: `global_raw`;
- look 2: `hierarchy_power50`;
- look 3: `global_raw`.

The same choice is used for both nominal levels.  It is explicitly a development-
selected policy, not a theorem-derived optimum.  All individual ladder methods
remain reported.  Calibration samples, cells, finite-sample ranks, models, and
pass thresholds are unchanged.  This is the last Experiment 3 amendment.

