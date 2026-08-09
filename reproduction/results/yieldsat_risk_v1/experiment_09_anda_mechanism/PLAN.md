# Experiment 9 plan: retrospective Anda controlled-mechanism synthesis

Recorded on 2026-08-01 after OOD-v4 results were already known.  This is a
read-only retrospective synthesis, not a new preregistered confirmation.

## Scope boundary

The Anda/Sentinel-2/ERA5 controlled experiment was produced by the earlier
PhysCrop anomaly detector, not the new YieldSAT yield-risk predictor.  It can
support the general mechanisms of GDD-conditioned reference calibration,
causal past-only operation, persistence, and nuisance stress testing.  It cannot
support external yield prediction, crop-loss causation, or current-model
generalization claims.

## Read-only evidence

- the stage-rescue report: GDD worst-stage FPR reduction, causal retention/FPR,
  and contamination sensitivity;
- the immutable OOD-v4 corrected report and integrity hashes;
- sensitivity/pAUC versus strongest non-PhysCrop baseline, clean FPR, positive-
  family TPR, corrected hard-normal episode alerts, and two-consecutive recall.

No score, threshold, event, feature, or label is regenerated or retuned.

## Mechanism-support gates

- all existing v4 integrity checks pass;
- stage-rescue gate passes, worst-stage FPR reduction >=20%, causal FPR<10%, and
  5% contamination TPR loss <=3 points;
- v4 TPR gain >=2 points, normalized pAUC delta >=-1 point, clean FPR<10%, gain
  in >=3/4 positive families, and every corrected hard-normal family <10%.

The old additional `hard_no_worse_than_baseline` gate remains failed and is
reported separately.  Thus Experiment 9 may pass only as **mechanism support
with a nuisance non-dominance limitation**, never as strict OOD dominance.

