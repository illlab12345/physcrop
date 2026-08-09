# Experiments 1--9 and final audit: executive summary

## Completed development experiments

- E1 public YieldSAT LSTM adaptation: PASS; RMSE 1.180 vs mean 1.319.
- E2 causal early curve: PASS; look-3 RMSE 1.036, AUROC 0.811.
- E3 conformal ladder: PASS after two retained development amendments.
- E4 family-wise alerting: PASS; persistent action 9 TP/1 FP on development.
- E5 strong baselines: PASS; selected dual-tree within 1.35% of best HistGB.
- E6 ablations: PASS; weather, multimodality, domain conditioning, and
  persistence have measurable contributions; GDD-vs-rank CI crosses zero.
- E7 spatial maps: PASS; 316 fields, field-clustered improvement and rho 0.402.
- E8 robustness: PASS under frozen gates; unseen-farm R2 0.709/AUROC 0.754,
  but leave-country-out absolute R2 is negative in three countries.
- E9 Anda: PASS_MECHANISM_ONLY; the old strict OOD dominance gate remains failed.

## Final audit

`FAIL_FINAL` on one strict gate: Brazil-wheat worst-group R2=-0.031 (n=14).
All pooled prediction, ranking, calibration, and action-alert gates pass.  The
audit is immutable and cannot be rerun.

## Defensible paper claims

1. Causal partial-season multimodal prediction at absolute training-derived GDD
   looks improves over country/crop climatology.
2. Held-out conformal null p-values plus alpha spending and persistence provide
   low empirical field-season false-alert rates with nonzero power.
3. A dual-tree predictor is competitive with strong ML and temporal Transformer
   baselines; no claim of universally best backbone is warranted.
4. Auxiliary within-field maps improve field-clustered spatial metrics without
   treating pixels as iid.
5. Robustness varies materially by crop/country; strict universal and
   cross-country transfer claims are not supported.

## Prohibited claims

- guaranteed WACV acceptance or a perfect all-gate result;
- exact reproduction of the unpublished YieldSAT benchmark training stack;
- farm-independent final validation;
- successful absolute cross-country transfer;
- disease/stress diagnosis from low-yield risk;
- Anda as external validation of the new YieldSAT predictor;
- pixel-level iid confidence or formal coverage.

