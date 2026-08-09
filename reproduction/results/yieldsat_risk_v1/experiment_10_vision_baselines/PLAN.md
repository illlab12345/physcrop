# Experiment 10: frozen YieldSAT vision baselines

Frozen before feature extraction and model fitting on 2026-08-01.

## Objective

Re-run AgriFM and ConvLSTM as genuine YieldSAT yield-risk baselines. Historical
Anda controlled-anomaly scores are not inputs to this experiment.

## Shared comparison protocol

- Endpoint: field yield at the third crop-specific absolute-GDD look.
- Roles: the exact Experiment 5 `model_fit`, `early_stop`, and 316-field
  `locked_external` challenge roles.
- Causality: six GDD-aligned Sentinel-2 frames at or before the frozen third
  look; no later imagery, weather, harvest map, calibration label, or audit
  target is read.
- Sentinel-2 bands: B02, B03, B04, B05, B06, B07, B08, B8A, B11, and B12.
- Image adapter: cloud-valid pixels (SCL classes 4/5), per-acquisition valid
  median filling, aspect-preserving square padding, and 32 x 32 resizing.
- Context supplied to both models: causal valid-fraction and cumulative-rain
  trajectories, static soil/terrain/geometry, and country/crop indicators.
- Target: the same model-fit country-crop standardized yield used by Experiment
  5. All reported predictions are converted back to raw t/ha.
- Metrics: the exact Experiment 5 RMSE, MAE, pooled R2, Spearman rho, AUROC,
  AUPRC, macro-group R2, and worst supported group R2 implementation.

## AgriFM adapter

- Official local `AgriFM.pth` encoder, frozen without fine-tuning.
- Six-frame 10-band tensor, global spatial pooling of the released encoder's
  final 1,024-dimensional representation.
- Fit-only PCA to 64 components, concatenated with shared context.
- Standardized Ridge head; alpha in {0.1, 1, 10, 100} selected only by
  early-stop normalized RMSE.

## ConvLSTM adapter

- The local ndrplz reference ConvLSTM implementation.
- Two recurrent layers with 16 and 32 channels, 3 x 3 kernels; final hidden
  state globally pooled and fused with a 32-dimensional context branch.
- AdamW, learning rate 1e-3, weight decay 1e-4, maximum 80 epochs, patience 10.
- Three deterministic seeds: 5611, 5612, and 5613. Predictions are averaged
  before metric computation.

## Admission checks

A row enters the manuscript only if all checks pass:

1. exactly the same 316 challenge field IDs and targets as Experiment 5;
2. every selected acquisition is at or before the crop-specific look;
3. all predictions and metrics are finite;
4. metrics recomputed from the written prediction CSV match the report;
5. no final-audit file is opened by the experiment script.

No minimum performance gate is imposed on a baseline. Weak results are reported
honestly; protocol or implementation failures are fixed before admission.
