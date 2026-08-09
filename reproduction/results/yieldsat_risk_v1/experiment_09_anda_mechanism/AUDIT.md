# Experiment 9 audit

Status: **PASS_MECHANISM_ONLY** (retrospective, not new confirmation).

Stage-conditioned calibration reduces worst-stage FPR by 27.65%; past-only FPR
is 6.40%; 5% reference contamination reduces TPR by 2.10 points.  Corrected
OOD-v4 improves TPR by 5.24 points and normalized pAUC by 0.84 point versus
NDVI+NDMI, improves all four positive families, and has hard-normal episode rates
0/6.67/4.0/4.17%, all below 10%.

The old strict OOD gate remains failed because uniform-radiometry false alerts
are worse than the strongest baseline.  The Anda result supports mechanisms of
stage calibration and persistence only.  It does not evaluate the current
YieldSAT yield predictor.

