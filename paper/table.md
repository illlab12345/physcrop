# Tables for the WACV Applications Track Manuscript

Values are transcribed from the immutable audit, development reports, and the explicitly labeled post-audit P0/P1 analyses. Values are rounded only for presentation. Downward and upward arrows indicate whether lower or higher values are better, respectively.

## Table 1. Dataset scope and frozen evaluation protocol

**Suggested caption.** *YieldSAT composition and the field-level protocol used in this work. The dataset covers four countries, four crops, and nine growing seasons. Of 2,173 available fields, 2,171 passed the causal feature-availability requirements. The development pool combines the adapter-reference and previously inspected challenge roles, whereas the 322-field audit set was revealed only after the model, calibration, alerting rule, and evaluation gates had been frozen.*

### (a) Geographic and crop composition

| Country | Years | Corn | Rapeseed | Soybean | Wheat | Total fields |
|:--|:--:|--:|--:|--:|--:|--:|
| Argentina | 2017–2024 | 185 | — | 440 | 126 | 751 |
| Brazil | 2017–2024 | 118 | — | 293 | 140 | 551 |
| Germany | 2016–2022 | — | 111 | — | 188 | 299 |
| Uruguay | 2018–2022 | — | — | 572 | — | 572 |
| **Total** | **2016–2024** | **303** | **111** | **1,305** | **454** | **2,173** |

### (b) Data roles and task definition

| Item | Value |
|:--|:--|
| Countries / crops / growing seasons | 4 / 4 / 9 |
| Available / causally usable fields | 2,173 / 2,171 |
| Development pool | 1,849 fields |
| One-shot frozen audit | 322 fields |
| Structurally excluded fields | 2 fields with fewer than four usable acquisitions |
| Input modalities | Sentinel-2, weather, soil, terrain, field geometry, and domain metadata |
| Prediction schedule | Three crop-specific absolute GDD looks |
| Primary prediction endpoint | Final field yield |
| Primary decision endpoint | Training-defined low-yield risk (approximately the lowest 20%) |
| Calibration | One-sided hierarchical conformal upper bounds; alert null uses conformal-role non-low-yield fields only |
| Sequential decision | Alpha spending across three looks; two consecutive hits trigger an action alert |
| Statistical unit | Field; pixels are not treated as independent observations |
| Audit independence boundary | Field-disjoint one-shot audit; the legacy split is not farm-disjoint |

---

## Table 2. Comparative assessment on the locked development challenge

**Suggested caption.** *Comparison at the final causal look on the same 316-field development challenge (low-yield prevalence: 19.3%). All methods obey the same crop-specific GDD cutoff. AgriFM and ConvLSTM consume six causal 10-band Sentinel-2 frames together with the shared weather, static, and domain context; AgriFM uses the official frozen encoder and ConvLSTM averages three independently initialized models. PhysCrop-Risk uses the frozen phenology-aligned dual-tree predictor selected on the early-stop partition. It is within 1.35% of the lowest-RMSE model while attaining the strongest AUROC and AUPRC. Macro and worst-group scores are computed over country–crop groups; the worst-group statistic includes groups with at least ten fields.*

| Method | RMSE ↓ | Pooled R² ↑ | AUROC ↑ | AUPRC ↑ | Macro-group R² ↑ | Worst-group R² ↑ |
|:--|--:|--:|--:|--:|--:|--:|
| Country–crop climatology | 1.4637 | 0.7270 | 0.5524 | 0.2298 | −0.0553 | −0.2216 |
| NDVI+NDMI+rain ridge | 1.3254 | 0.7762 | 0.7409 | 0.3783 | 0.0948 | −0.8275 |
| AgriFM frozen encoder + context | 1.3268 | 0.7757 | 0.6622 | 0.2913 | 0.0302 | −0.3689 |
| Full-feature ridge | 1.1845 | 0.8212 | 0.7352 | 0.4073 | 0.2515 | −0.5310 |
| MLP | 1.2058 | 0.8147 | 0.7788 | 0.4535 | 0.2947 | 0.0600 |
| ConvLSTM + context | 1.1443 | 0.8332 | 0.7688 | 0.4998 | 0.2703 | −0.2134 |
| Temporal Transformer | 1.1493 | 0.8317 | 0.7841 | 0.4906 | 0.3758 | **0.1992** |
| Random Forest | 1.0702 | 0.8540 | 0.7950 | 0.5718 | 0.3807 | 0.0695 |
| HistGradientBoosting | **1.0223** | **0.8668** | 0.7921 | 0.5627 | **0.4703** | 0.1763 |
| Phenology-aligned ExtraTrees | 1.0459 | 0.8606 | 0.8110 | 0.5717 | 0.4362 | 0.1388 |
| **PhysCrop-Risk dual-tree** | 1.0361 | 0.8632 | **0.8112** | **0.5735** | 0.4477 | 0.1490 |

*The AgriFM adapter applies fit-only PCA and an early-stop-selected Ridge head to the official frozen representation. The ConvLSTM has 73,793 trainable parameters and is early-stopped independently for each of three seeds. Fit-only 0.5%–99.5% context winsorization is shared by both vision baselines. Their paired field-stratified RMSE differences relative to PhysCrop-Risk are [0.1732, 0.3969] for AgriFM and [0.0531, 0.1657] for ConvLSTM (95% bootstrap CIs; positive values favor PhysCrop-Risk). The dual-tree predictor combines 90% ExtraTrees and 10% HistGradientBoosting at the final look; the mixture weight was selected without accessing the final audit outcomes. Bold values denote the best result in each column, not merely the proposed method.*

---

## Table 3. One-shot audit and post-audit protocol correction

**Suggested caption.** *Prediction and coverage in (a) are from the immutable one-shot audit. Panel (b) removes the changing-cohort explanation by evaluating all looks on the same 300 fields. Panels (c–d) are an explicitly post-audit correction requested during review: they retain the pre-reveal frozen predictors, cutoffs, alpha allocation, and persistence rule, but rebuild every empirical null using only conformal-role non-low-yield fields. Thus they are uniform-policy sensitivity analyses, not a second frozen audit.*

### (a) Yield prediction and low-yield ranking

| Look | Fields | RMSE ↓ | Climatology RMSE ↓ | Paired RMSE difference, 95% CI | Pooled R² ↑ | AUROC ↑ | AUPRC ↑ | Prevalence |
|:--:|--:|--:|--:|:--:|--:|--:|--:|--:|
| 1 | 300 | 1.2407 | 1.4830 | [−0.3685, −0.1200] | 0.7851 | 0.7370 | 0.4412 | 0.2233 |
| 2 | 310 | 1.1902 | 1.4629 | [−0.4026, −0.1384] | 0.7963 | 0.7539 | 0.4649 | 0.2258 |
| **3** | **322** | **1.0582** | **1.4481** | **[−0.5158, −0.2694]** | **0.8360** | **0.8331** | **0.6245** | **0.2298** |

*Frozen one-sided coverage at Looks 1/2/3 is 0.9300/0.9290/0.9534 at 90% nominal and 0.9800/0.9903/0.9876 at 95% nominal. Look-3 macro-group and worst-group R² are 0.4093 and −0.0313 (8/9 positive groups).*

### (b) Same-field early curve and verified pre-harvest timing

| Look | Common fields | RMSE ↓ | R² ↑ | AUROC ↑ | AUPRC ↑ | Median lead to harvest [5th, 95th] days | Post-harvest uses |
|:--:|--:|--:|--:|--:|--:|:--:|--:|
| 1 | 300 | 1.2407 | 0.7851 | 0.7370 | 0.4412 | 107 [72, 170] | 0 |
| 2 | 300 | 1.2084 | 0.7962 | 0.7534 | 0.4510 | 86 [50, 130] | 0 |
| **3** | **300** | **1.0887** | **0.8346** | **0.8272** | **0.5942** | **62.5 [30, 102]** | **0** |

*On paired country–crop-stratified resampling, Look 3 minus Look 1 has RMSE CI [−0.2527, −0.0537], AUROC CI [0.0446, 0.1417], and AUPRC CI [0.0652, 0.2371]. The minimum lead is 36, 11, and 1 day at Looks 1–3, respectively.*

### (c) Strong audit baselines under identical clean-null post-processing

| Predictor | Look-3 RMSE ↓ | AUROC ↑ | AUPRC ↑ | Action TP / FP | Action FPR ↓ | Sensitivity ↑ | Precision ↑ |
|:--|--:|--:|--:|:--:|--:|--:|--:|
| Country–crop climatology | 1.4481 | 0.5780 | 0.2913 | 5 / 18 | 0.0726 | 0.0676 | 0.2174 |
| NDVI+NDMI+rain Ridge† | 1.2767 | 0.7472 | 0.4684 | 1 / 0 | **0.0000** | 0.0135 | **1.0000** |
| Frozen ExtraTrees | 1.0640 | 0.8316 | 0.6198 | 8 / 4 | 0.0161 | 0.1081 | 0.6667 |
| Frozen HGB | 1.0879 | 0.8223 | 0.5992 | **10 / 2** | 0.0081 | **0.1351** | 0.8333 |
| **Frozen dual-tree** | **1.0582** | **0.8331** | **0.6245** | 8 / 4 | 0.0161 | 0.1081 | 0.6667 |

*All five predictors use the same conformal-only null (96/97/97 non-low-yield fields at Looks 1/2/3), alpha schedule (0.010/0.015/0.025), and two-hit rule. †The Ridge is a fixed-protocol post-audit fit and is not described as pre-reveal frozen; the tree components are extracted from pre-reveal model bundles.*

### (d) Clean-null dual-tree operating points with two-sided exact intervals

| Tier | TP / FP | FPR [95% CI] | Sensitivity [95% CI] | Precision [95% CI] | Farm-clustered FPR / sensitivity / precision 95% intervals |
|:--|:--:|:--:|:--:|:--:|:--:|
| Watch (any hit) | 23 / 12 | 0.0484 [0.0252, 0.0830] | 0.3108 [0.2083, 0.4290] | 0.6571 [0.4779, 0.8087] | [0.0202, 0.0811] / [0.1918, 0.4085] / [0.5277, 0.8000] |
| **Action (two hits)** | **8 / 4** | **0.0161 [0.0044, 0.0408]** | 0.1081 [0.0478, 0.2020] | **0.6667 [0.3489, 0.9008]** | [0.0000, 0.0325] / [0.0156, 0.2090] / [0.3333, 1.0000] |

*The original frozen-null action result was TP=8, FP=5, FPR=0.0202, sensitivity=0.1081, and precision=0.6154. The clean correction changes one false positive and leaves sensitivity unchanged. Audit farm composition is 321 seen-farm fields and only one unseen-farm field (52 seen farms, one unseen farm), so no unstable seen/unseen performance split is claimed.*

---

## Table 4. Alignment, components, and group-respecting robustness

**Suggested caption.** *Development-only analyses with audit outcomes unused. Panel (a) varies only the interpolation coordinate while holding the causal 65%-GDD prefix, model family, feature budget, and 316 fields fixed. GDD is significantly better than calendar time; its point estimates are better than rank and normalized season fraction, but those intervals cross zero. Panels (b–c) test modality dependence and temporal, farm, and spatial robustness.*

### (a) Four-level alignment ladder on the same 316 fields

| Alignment coordinate | RMSE ↓ | R² ↑ | NRMSE ↓ | RMSE skill ↑ | AUROC ↑ | AUPRC ↑ | Macro / worst-group R² | GDD − alternative RMSE, 95% CI |
|:--|--:|--:|--:|--:|--:|--:|:--:|:--:|
| Calendar day-of-season | 1.0961 | 0.8469 | 0.3913 | 0.2511 | 0.8045 | 0.5571 | 0.4196 / **0.2788** | **[−0.1328, −0.0001]** |
| Acquisition rank | 1.0579 | 0.8574 | 0.3776 | 0.2772 | 0.7992 | 0.5361 | 0.4223 / 0.1762 | [−0.0582, 0.0125] |
| Normalized observed-season fraction | 1.0553 | 0.8581 | 0.3767 | 0.2790 | 0.8019 | 0.5712 | 0.4397 / 0.1965 | [−0.0482, 0.0093] |
| **Absolute GDD** | **1.0361** | **0.8632** | **0.3698** | **0.2921** | **0.8112** | **0.5735** | **0.4477** / 0.1490 | — |

*NRMSE divides RMSE by the challenge-set target standard deviation; RMSE skill is $1-\mathrm{RMSE}/\mathrm{RMSE}_{clim}$. “Calendar” uses elapsed days from the first usable in-season acquisition and a model-fit country–crop reference span. Negative intervals favor GDD.*

### (b) Critical component ablations

| Variant | RMSE ↓ | AUROC ↑ | AUPRC ↑ | Macro-group R² ↑ | Full − ablation RMSE, 95% CI |
|:--|--:|--:|--:|--:|:--:|
| **Full dual-tree** | **1.0361** | **0.8112** | **0.5735** | **0.4477** | — |
| ExtraTrees only | 1.0459 | 0.8110 | 0.5717 | 0.4362 | [−0.0163, −0.0038] |
| Without weather | 1.0936 | 0.7835 | 0.5117 | 0.3934 | [−0.0876, −0.0324] |
| Sentinel-2 only | 1.1419 | 0.7760 | 0.4702 | 0.3505 | [−0.1489, −0.0618] |
| Without domain information | 1.3411 | 0.7691 | 0.4188 | 0.1970 | [−0.4246, −0.1895] |

### (c) Temporal, farm, and spatial robustness

| Evaluation | Support | Proposed result | Reference result | Additional evidence |
|:--|:--|:--|:--|:--|
| Leave-one-year-out (LOYO) | 1,849 fields; 9 years | RMSE 1.3251; R² 0.7790; macro-year R² 0.7223 | Climatology RMSE 1.6390 | Paired RMSE difference 95% CI [−0.3567, −0.2712] |
| Five-fold farm-grouped OOF | 1,848 fields; 81 farms | RMSE 1.2224; R² 0.8120; AUROC 0.8116; AUPRC 0.5435 | Climatology RMSE 1.5545 | NRMSE 0.4336; RMSE skill 0.2136; macro/worst-group R² 0.2491/−0.0282; zero farm overlap in every fold |
| Unseen-farm development subset | 59 fields; 16 farms | RMSE 1.0593; R² 0.7086; AUROC 0.7544 | — | Secondary frozen-model subset |
| Field-clustered spatial mapping | 316 fields; 1,530,144 pixels | Mean field RMSE 1.5234 | Uniform 1.6053; NDVI 1.5671 | Spatial−uniform CI [−0.0980, −0.0663]; spatial−NDVI CI [−0.0528, −0.0348] |
| Within-field spatial ranking | 316 fields | Median Spearman ρ 0.4024 | — | Median bottom-20% AUROC 0.7316 |

---

## Provenance

- Table 1: `YieldSAT_raw_data/`, `results/yieldsat_risk_v1/MASTER_PROTOCOL.md`, and the two cache reports.
- Table 2: `results/yieldsat_risk_v1/experiment_05_strong_baselines/report.json` and `results/yieldsat_risk_v1/experiment_10_vision_baselines/report.json` (independently checked by `validation.json`).
- Table 3: `results/yieldsat_risk_v1/final_audit/final_report.json` and `freeze_manifest.json`.
- Table 3 post-audit panels: `results/yieldsat_risk_v1/post_audit_p0p1/p0a_report.json` and `p0b_report.json`.
- Table 4(a): `results/yieldsat_risk_v1/post_audit_p0p1/p1_report.json`; Table 4(b): `experiment_06_ablations/report.json`.
- Table 4(c): `post_audit_p0p1/p0b_report.json`, `experiment_07_spatial_maps/report.json`, and `experiment_08_robustness/report.json`.
