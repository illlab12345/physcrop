# Experiment 5 plan: strong and fair predictor baselines

Frozen on 2026-08-01 before baseline fitting.

## Fair comparison

All models use the Experiment 2 third absolute-GDD look, identical eligible
fields, model-fit targets, group-normalized target, imputation learned on
model-fit, and the untouched `locked_external` development challenge.  No model
may use final-season values, calibration labels, or final-audit labels.

## Frozen baselines

1. country/crop model-fit mean;
2. NDVI+NDMI+cumulative-rain Ridge with categorical indicators;
3. full multimodal Ridge;
4. Random Forest (600 trees, min leaf 3, max features 0.7);
5. histogram gradient boosting (300 iterations, leaf 15, L2=1);
6. MLP (128/64, ReLU, L2=1e-4, early stopping);
7. causal temporal Transformer: six GDD-aligned tokens, 64 dimensions, two
   encoder layers, four heads, static/categorical fusion, fit-only normalization,
   AdamW, and early stopping on the separate early-stop farm role;
8. the frozen Experiment 2 phenology-conditioned ExtraTrees predictor.

The public full-season LSTM result from Experiment 1 is included as context but
not treated as a paired early-look comparison.

## Metrics and gates

- raw-yield RMSE/MAE/R2/rho, low-yield AUROC/AUPRC, macro country/crop RMSE/R2,
  and worst supported group R2 (n>=10);
- field-stratified paired bootstrap RMSE differences relative to ExtraTrees;
- parameter count and CPU fit time for operational comparison;
- all challenge fields receive finite predictions;
- the selected ExtraTrees predictor must be within 2% RMSE of the best strong
  baseline, significantly beat climatology (upper 95% CI <0), attain AUROC>=0.75,
  and have positive R2 in every supported group.

If another frozen baseline is more than 2% better, it replaces the predictor only
through a recorded development amendment followed by re-running Experiments
2--4; silent model switching is forbidden.

