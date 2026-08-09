# Amendment E5-01: fit-only robustness and dual-tree predictor

Attempt 1 failed because frozen ExtraTrees RMSE 1.0459 was 2.30% above HistGB
RMSE 1.0223, narrowly outside the <=2% gate.  It also exposed a genuine baseline
bug: a nearly constant static curvature feature in model-fit had a development
standardized magnitude of 2.66e9, causing unbounded Ridge/MLP extrapolation.
Final-audit labels remain unread.

Attempt 2 makes two development-only corrections:

1. Ridge and MLP inputs are clipped to model-fit 0.5/99.5 percentiles before
   standardization.  Tree models are unchanged.
2. A `phenology_dual_tree` prediction convexly combines the frozen ExtraTrees
   and HistGB outputs.  The HistGB weight is selected from {0,0.1,...,1} using
   only the farm-disjoint early-stop role.  The selected weight is then frozen.

On early-stop, the unique selected weight is 0.1 (z-RMSE 0.74825 versus 0.74850
for ExtraTrees and 0.78505 for HistGB).  No challenge outcome enters this weight
selection.  Experiment 5's same <=2%, AUROC, all-group, and bootstrap gates are
applied to the dual-tree predictor.  If it passes, Experiments 2--4 must be
rerun with this predictor before any final audit.

