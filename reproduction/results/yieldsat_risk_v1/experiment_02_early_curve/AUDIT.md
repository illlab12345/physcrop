# Experiment 2 audit

Status: **PASS** under all frozen gates.

| Look | Eligible challenge fields | RMSE | Climatology RMSE | Low-yield AUROC | AUPRC / prevalence |
|---|---:|---:|---:|---:|---:|
| 1 | 291 (92.1%) | 1.138 | 1.495 | 0.764 | 0.471 / 0.172 |
| 2 | 309 (97.8%) | 1.130 | 1.472 | 0.777 | 0.514 / 0.181 |
| 3 | 316 (100%) | 1.046 | 1.464 | 0.811 | 0.572 / 0.193 |

Late-look RMSE improvement over climatology has stratified bootstrap 95% CI
[-0.545, -0.290].  All nine supported country/crop groups have positive late-look
R2; the worst is Germany-rapeseed at 0.139.  No future observations or final
labels were used.  The internal fit/early-stop/calibration farm groups are
disjoint.  Thirty-three legacy challenge farms overlap model-fit farms and this
limits any unseen-farm claim; Experiment 8 addresses LOFO separately.

