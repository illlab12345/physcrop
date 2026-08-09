# Experiment 5 audit

Final status: **PASS on attempt 2**.

HistGB had the lowest RMSE (1.022).  The selected dual-tree predictor used an
early-stop-selected HistGB weight of 0.1 and achieved RMSE 1.036, AUROC 0.811,
macro group R2 0.448, and worst supported group R2 0.149.  It is 1.35% from the
best RMSE and its improvement over climatology has bootstrap CI
[-0.544, -0.297].  All selected-predictor gates pass.

Fit-only winsorization repaired the initial Ridge/MLP cross-domain numerical
failure.  The public-context Transformer RMSE is 1.149; its worst-group R2 0.199
is useful but it does not beat the tree methods overall.  After selection, the
dual-tree predictor was propagated through Experiments 2--4; all three retained
PASS status.  Final labels remain unread.

