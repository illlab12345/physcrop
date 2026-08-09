# PhysCrop-Risk: Turning Partial-Season Earth Observation into Calibrated Low-Yield Alerts

## Abstract

Crop-yield prediction is most useful before harvest, but repeatedly thresholding partial-season forecasts can produce unreliable alerts. We present **PhysCrop-Risk**, a multimodal Earth-observation system that converts early yield prediction into calibrated, persistent low-yield warnings. At three crop-specific growing-degree-day (GDD) looks, it uses only currently available Sentinel-2, weather, soil, terrain, geometry, and domain information. A phenology-aligned dual-tree predictor estimates final field yield; a strictly held-out conformal partition supplies non-low-yield reference scores; an alpha budget of 0.05 is distributed across looks; and two consecutive crossings produce a conservative action alert. We evaluate 2,171 usable YieldSAT fields spanning four countries, four crops, and nine seasons. On a 322-field one-shot audit, the final partial-season look attains RMSE 1.0582 t ha$^{-1}$ versus 1.4481 for country--crop climatology, $R^2=0.8360$, AUROC 0.8331, and AUPRC 0.6245. Under a post-audit integrity correction that retains the frozen predictors and decisions but excludes the early-stop partition from the empirical null, persistent alerts yield 1.61% FPR, 10.81% sensitivity, and 66.67% precision; the two-sided exact FPR interval is [0.44%, 4.08%]. A common 300-field cohort verifies significant Look-1-to-3 gains, all looks precede harvest, and five-fold farm-grouped prediction gives AUROC 0.8116 on 1,848 fields. Mean-preserving 10 m residual maps further improve field-clustered RMSE over uniform and NDVI-only maps. PhysCrop-Risk thus connects partial-season visual prediction to an auditable reliability--coverage trade-off.

## 1. Introduction

Crop-yield models are commonly evaluated as end-of-season regressors, yet many agricultural decisions must be made while the season is still unfolding. A useful warning system must answer a harder question than *what will the final yield be?*: with only the imagery and weather available today, is there sufficient evidence that a field will finish below an operationally meaningful yield level? This distinction matters. A predictor may achieve low average error while producing poorly calibrated tail decisions, and repeatedly querying it through the season can accumulate false alarms. In practice, excessive alerts consume limited scouting capacity and rapidly erode trust.

Earth observation offers the temporal coverage needed for early warning, but it also creates three coupled difficulties. First, crop appearance changes rapidly with phenology, so observations taken on the same calendar date need not represent comparable developmental states across crops or climates. Second, yield distributions differ markedly across country--crop domains and years. Third, a field-level warning should be accompanied by spatial evidence that identifies where the predicted shortfall is expressed, without treating millions of within-field pixels as independent validation samples. Existing remote-sensing yield work has made substantial progress in multimodal regression and spatiotemporal representation learning [You et al., 2017](https://doi.org/10.1609/aaai.v31i1.11172); [Lin et al., 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_MMST-ViT_Climate_Change-aware_Crop_Yield_Prediction_via_Multi-Modal_Spatial-Temporal_Vision_ICCV_2023_paper.html), while recent datasets enable evaluation at field and subfield scales [Sani et al., 2024](https://openaccess.thecvf.com/content/WACV2024/html/Sani_SICKLE_A_Multi-Sensor_Satellite_Imagery_Dataset_Annotated_With_Multiple_Key_WACV_2024_paper.html); [Miranda et al., 2026](https://arxiv.org/abs/2604.00940). However, early prediction, calibrated repeated testing, and actionable spatial evidence are usually evaluated separately.

We introduce **PhysCrop-Risk**, an end-to-end formulation that converts partial-season multimodal observations into two levels of low-yield evidence: a sensitive *watch* signal and a conservative *action* alert. At three crop-specific accumulated-GDD looks, the system uses only Sentinel-2 acquisitions and weather observed by that look. It aligns irregular sequences in thermal time, combines them with soil, terrain, field geometry, and domain metadata, and predicts final field yield with an early-stop-selected ensemble of randomized and boosted trees. A training-defined country--crop hierarchy converts the prediction into a low-yield risk score. A separate conformal-role partition, never used for fitting or mixture selection, provides the non-low-yield empirical null. We allocate a family-wise error budget across the three looks and require two consecutive crossings before issuing action. This separation turns a ranking model into an operational decision rule whose reliability--coverage profile is measured once per field-season.

The evaluation follows this deployment sequence. From 2,173 YieldSAT fields across four countries, four crops, and nine growing seasons, 2,171 satisfy the partial-season availability requirements. We use 1,849 fields for development and keep 322 fields in a one-shot audit whose outcomes are unavailable until the features, models, coverage calibration, original alert logic, and pass criteria have been frozen. The audit establishes final-look RMSE 1.0582 t ha$^{-1}$, $R^2=0.8360$, AUROC 0.8331, and AUPRC 0.6245 at prevalence 0.2298. A review-driven integrity analysis then removes early-stop fields from the alert reference library without altering any predictor, cutoff, alpha, or persistence rule. Under this stricter null, watch and action attain FPR/sensitivity of 4.84%/31.08% and 1.61%/10.81%, respectively. Five-fold farm-grouped out-of-fold evaluation and farm-clustered audit intervals address shared-farm dependence; actual acquisition and harvest dates verify median leads of 107, 86, and 62.5 days at the three looks.

Figure 1 summarizes the intended use: observations available at each thermal-time look accumulate into calibrated evidence; an isolated excursion remains a watch, whereas persistent evidence triggers action before the harvest outcome is known. Here and throughout, *time-causal* means that no acquisition, weather observation, final-season statistic, or harvest outcome occurring after a look is used. It does not denote causal-effect estimation.

![Figure 1. A frozen-audit true-positive example. Only observations available by each crop-specific GDD look enter the predictor. Null-calibrated evidence is compared with the preallocated look-wise level; two consecutive hits trigger an action alert, and the harvest yield is used only for retrospective evaluation. The example is illustrative, while all reported audit metrics aggregate the complete 322-field audit.](figure1_heldout_alert_overview.png)

Our contributions are threefold:

1. We formulate partial-season crop monitoring as **calibrated sequential low-yield warning**, rather than end-of-season regression alone, coupling risk ranking with an explicit field-season false-alarm budget.
2. We develop a **phenology-aligned multimodal system** that combines time-causal Sentinel-2 and weather histories with static context, one-sided calibration, persistent decisions, and mean-preserving within-field evidence.
3. We provide a **multi-axis field-level assessment** spanning strong visual and tabular--temporal baselines, a common-field early curve, clean-null audit comparisons, leave-one-year-out and farm-grouped evaluation, exact and farm-clustered uncertainty, and field-clustered spatial analysis.

## 2. Related Work

### 2.1. Multimodal crop-yield prediction

Remote sensing has long supported pre-harvest yield estimation by linking spectral trajectories to canopy development. You et al. combined representation learning and Gaussian processes for county-level soybean prediction [You et al., 2017](https://doi.org/10.1609/aaai.v31i1.11172). Multimodal work increasingly joins imagery with environmental context: MMST-ViT models satellite, meteorological, spatial, and long-range temporal structure at county scale [Lin et al., 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Lin_MMST-ViT_Climate_Change-aware_Crop_Yield_Prediction_via_Multi-Modal_Spatial-Temporal_Vision_ICCV_2023_paper.html), and field/subfield early fusion combines Sentinel-2, weather, soil, and elevation [Pathak et al., 2023](https://arxiv.org/abs/2308.08948). SICKLE supplies multi-sensor annotations for crop type, phenology, and yield [Sani et al., 2024](https://openaccess.thecvf.com/content/WACV2024/html/Sani_SICKLE_A_Multi-Sensor_Satellite_Imagery_Dataset_Annotated_With_Multiple_Key_WACV_2024_paper.html), while SegMango demonstrates the application value of early-stage image-and-weather yield prediction [Ven et al., 2026](https://openaccess.thecvf.com/content/WACV2026/html/Ven_SegMango_Early_Deep_Mango_Yield_Prediction_based_on_Flower_Segmentation_WACV_2026_paper.html). These studies primarily evaluate forecast error; our target is the additional decision layer required when forecasts are queried repeatedly.

YieldSAT moves this setting to high-resolution multimodal prediction across 2,173 fields, four countries, four crops, and multiple years, pairing Sentinel-2 imagery with weather, soil, topography, and combine-harvester yield maps [Miranda et al., 2026](https://arxiv.org/abs/2604.00940). Its shifts across countries, crops, regions, and years motivate domain-aware evaluation. Pixelwise yield-density estimation provides useful within-field structure [Baghdasaryan et al., 2022](https://openaccess.thecvf.com/content/CVPR2022W/WiCV/html/Baghdasaryan_Deep_Density_Estimation_Based_on_Multi-Spectral_Remote_Sensing_Data_for_CVPRW_2022_paper.html); in our setting, however, maps explain a separately evaluated field alert and pixels never inflate the decision sample size. PhysCrop-Risk therefore asks how much actionable low-yield evidence is available *before* season completion, with field-season warnings as the primary endpoint.

Multispectral foundation models offer another route to agricultural representation learning. SatMAE adapts masked autoencoding to temporal and multispectral satellite imagery through temporal embeddings and spectral positional structure [Cong et al., 2022](https://papers.neurips.cc/paper_files/paper/2022/hash/01c561df365429f33fcd7a7faa44c985-Abstract-Conference.html). ConvLSTM represents spatial and temporal dynamics through convolutional recurrent transitions [Shi et al., 2015](https://papers.nips.cc/paper/2015/hash/07563a3fe3bbe7e3ba84431ad9d055af-Abstract.html). Rather than assuming that a larger visual backbone is always preferable for heterogeneous field-level tabular--temporal fusion, we compare a frozen AgriFM encoder, ConvLSTM, a temporal Transformer, and strong tree ensembles under the same field split and final GDD cutoff. This comparison supports a system-level choice: a compact heterogeneous predictor supplies competitive forecasts, while reliability is introduced explicitly after prediction.

### 2.2. Phenology alignment, uncertainty, and repeated decisions

Calendar time is an imperfect coordinate for crop development because temperature accumulation changes the rate at which crops progress through phenological stages. Satellite phenology studies consequently use vegetation-index trajectories and thermal-time information for within-season monitoring [Gao and Zhang, 2021](https://spj.science.org/doi/10.34133/2021/8379391); accumulated GDD has also been integrated directly into remote-sensing phenology models [Liao et al., 2022](https://doi.org/10.3390/rs14215337). PhysCrop-Risk uses GDD more narrowly and operationally: it defines crop-specific early looks from the model-fit partition and interpolates all dynamic variables onto a common thermal-time grid. The result is a fixed information boundary shared by every compared method.

Uncertainty quantification is especially important when a continuous prediction is thresholded into an intervention. Split conformal inference constructs finite-sample prediction sets under exchangeability without committing to a parametric error distribution [Lei et al., 2018](https://arxiv.org/abs/1604.04173), and conformal risk control extends this perspective to application-specific losses [Angelopoulos et al., 2022](https://arxiv.org/abs/2208.02814). Our objective differs from generic interval prediction in two respects. First, the operational null is *non-low yield*: a warning should be rare for fields whose final yield exceeds the training-defined cutoff. Second, three looks create a multiple-testing problem at the field-season level. We therefore combine one-sided calibration diagnostics with empirical null p-values, allocate a total level of 0.05 across looks, and apply a two-hit persistence filter. This produces a transparent sensitivity--false-alarm ladder rather than a single unconstrained score threshold.

## 3. Method

![Figure 2. PhysCrop-Risk has five stages. It observes only information available by each look, aligns irregular sequences in crop-specific thermal time, predicts field yield with a dual-tree ensemble, calibrates one-sided low-yield evidence on held-out reference fields, and converts persistent crossings into action alerts. A separate mean-preserving residual model provides 10 m within-field evidence.](figure2_physcrop_risk_framework.png)

### 3.1. Task formulation and time-available observations

For field $i$, let $y_i$ denote final yield, $d_i$ its country--crop domain, and $\{(\mathbf{x}_{it},g_{it})\}_{t=1}^{T_i}$ an irregular sequence of observations indexed by accumulated GDD $g_{it}$. The dynamic vector contains 12 Sentinel-2 surface-reflectance bands, NDVI, NDMI, NDRE, EVI, valid-pixel fraction, and cumulative precipitation. Static context contains eight soil properties, five terrain variables, field area, circularity, and one-hot country and crop indicators. The task is to estimate final yield and warn when it falls below a cutoff learned exclusively from model-fit fields.

We define three crop-specific looks without using any evaluation-field outcome or harvest date. For crop $c$,

$$
L_{c,k}=\rho_k\,\operatorname{median}_{i\in\mathcal{D}_{\mathrm{fit}},c_i=c}
\left(\max_t g_{it}\right), \qquad
(\rho_1,\rho_2,\rho_3)=(0.35,0.50,0.65).
$$

At look $k$, all records with $g_{it}>L_{c_i,k}$ are removed. Each of the 18 dynamic channels is linearly interpolated at six thermal-time positions $\{0,0.2L_{c_i,k},\ldots,L_{c_i,k}\}$; a field is eligible when at least three acquisitions are available. Missing feature values are replaced by model-fit medians. This representation fixes the information set across irregular acquisition schedules while retaining the partial trajectory. A controlled ladder later compares calendar day-of-season, acquisition rank, normalized observed-season fraction, and absolute GDD while holding this causal prefix and feature budget fixed. GDD gives the look a crop-dependent physical interpretation, but is not a proxy for season completion: final-season GDD, future weather, future cloud availability, and harvest timing are excluded.

Low yield is defined relative to the training distribution because absolute yield scales differ across crops and countries. We estimate a cutoff $\tau_d$ as the 20th percentile of model-fit yield within a country--crop cell when at least 30 fields are available, falling back to the crop distribution when it has at least 50 fields and then to the global distribution. Thus the binary endpoint is $z_i=\mathbb{1}[y_i\leq\tau_{d_i}]$. The same precomputed hierarchy is used for development comparisons and the frozen audit.

### 3.2. Phenology-aligned field prediction

We standardize the regression target within each observed country--crop group,

$$
\tilde y_i=\frac{y_i-\mu_{d_i}}{\sigma_{d_i}},
$$

where $\mu_d$ and $\sigma_d$ are computed on model-fit fields and $\sigma_d$ is lower-bounded by 0.25 t ha$^{-1}$. Two complementary regressors are fitted independently at every look: an ExtraTrees ensemble with 600 trees, minimum leaf size 3, and 0.7 feature subsampling, and a histogram gradient-boosted regressor with 300 iterations, minimum leaf size 15, and $\ell_2$ regularization 1. Their normalized predictions are combined as

$$
\hat{\tilde y}_{i,k}=(1-\lambda_k)f^{\mathrm{ET}}_k(\mathbf{h}_{i,k})+
\lambda_k f^{\mathrm{HGB}}_k(\mathbf{h}_{i,k}),
$$

where $\mathbf{h}_{i,k}$ is the aligned multimodal feature vector and $\lambda_k\in\{0,0.1,\ldots,1\}$ minimizes RMSE on a farm-disjoint early-stop partition. At the final look, the selected mixture is 90% ExtraTrees and 10% HGB. Transforming back to physical units gives $\hat y_{i,k}=\mu_{d_i}+\sigma_{d_i}\hat{\tilde y}_{i,k}$. The corresponding ranking score

$$
r_{i,k}=\tau_{d_i}-\hat y_{i,k}
$$

is larger when the prediction lies farther below the domain-appropriate low-yield cutoff. This design separates the source of risk ranking from the subsequent reliability layer: any regressor could replace the dual-tree backbone, while the warning semantics remain fixed.

### 3.3. One-sided calibration, persistent alerts, and spatial evidence

We use held-out reference fields in two complementary calibrations. First, one-sided yield bounds quantify predictive coverage. For calibration residual $e_{j,k}=y_j-\hat y_{j,k}$ and nominal coverage $1-\alpha$, the upper bound is

$$
U_{i,k}^{(1-\alpha)}=\hat y_{i,k}+Q_{\lceil(m+1)(1-\alpha)\rceil}
\left(\{e_{j,k}\}_{j=1}^{m}\right),
$$

with the finite-sample conformal rank clipped to the calibration-set size. A development calibration ladder considers global, crop, country--crop hierarchical, and tree-dispersion-scaled scores. The frozen choices are global raw residuals at Looks 1 and 3 and hierarchical residuals scaled by the square root of tree dispersion at Look 2. We report empirical 90% and 95% coverage at every look; their distribution-free interpretation requires exchangeability between calibration and evaluation fields.

Second, action decisions use a class-conditional empirical null. Both coverage and alert calibration use the 125-field conformal partition, whose farms are disjoint from model fitting and early-stop mixture selection. At each look, the alert library retains only eligible conformal-role fields whose yield exceeds the frozen cutoff; the resulting null sizes are 96, 97, and 97. For such a non-low-yield field, define

$$
s_{j,k}=\frac{\tau_{d_j}-\hat y_{j,k}}{\sigma_{d_j}}.
$$

Let $\mathcal{N}_k$ be the held-out non-low-yield reference scores at look $k$. The one-sided p-value for an evaluation field is

$$
p_{i,k}=\frac{1+\sum_{s\in\mathcal{N}_k}\mathbb{1}[s\geq s_{i,k}]}
{1+|\mathcal{N}_k|}.
$$

We allocate $(\alpha_1,\alpha_2,\alpha_3)=(0.010,0.015,0.025)$, whose sum is 0.05. A *watch* occurs when $p_{i,k}\leq\alpha_k$. Under a frozen score and class-conditional null exchangeability, super-uniform p-values and the union bound control the probability of at least one false watch in a field-season by $\sum_k\alpha_k$. The operational *action* is more selective:

$$
A_{i,k}=\mathbb{1}[p_{i,k}\leq\alpha_k]\,
\mathbb{1}[p_{i,k-1}\leq\alpha_{k-1}], \qquad k\in\{2,3\}.
$$

Persistence can only remove single-look alerts, so it cannot increase the family-wise false-positive rate. It instead exposes an application-controlled trade-off: watch maximizes coverage for scouting, while action prioritizes precision for scarce intervention capacity.

Finally, we attach spatial evidence to the final-look field prediction. From the latest valid Sentinel-2 acquisition no later than $L_{c,3}$, a pixel residual model consumes 12 bands, four indices, eight soil layers, and five terrain layers. Features are standardized within field. An HGB model is trained on at most 256 sampled pixels per model-fit field to predict deviations from each field's observed mean yield. At inference, its residuals are centered to zero before being added to $\hat y_{i,3}$:

$$
\hat y_{i,p}^{\mathrm{spatial}}=\hat y_{i,3}+\hat\delta_{i,p}
-\frac{1}{|P_i|}\sum_{q\in P_i}\hat\delta_{i,q}.
$$

The map therefore preserves the frozen field-level prediction exactly while ranking heterogeneous zones within the field. Statistical comparisons resample fields, never pixels; the spatial output is evidence for localization, not a second set of independent field outcomes.

## 4. Experiments

### 4.1. Dataset, splits, and evaluation

We evaluate on the flexible release of YieldSAT [Miranda et al., 2026](https://arxiv.org/abs/2604.00940). It contains field boundaries, irregular Sentinel-2 sequences, daily weather, soil and terrain rasters, metadata, field yield, and spatial combine-harvester yield maps. The 2,173 fields span Argentina, Brazil, Germany, and Uruguay; corn, rapeseed, soybean, and wheat; and nine growing seasons from 2016 to 2024. Two fields have fewer than four usable acquisitions, leaving 2,171 fields. Table 1 summarizes the geographic coverage and protocol.

**Table 1. YieldSAT composition and frozen evaluation protocol.** The development pool contains a 1,533-field adapter reference and a 316-field locked challenge; the former is divided by a deterministic farm hash into 1,254 model-fit, 154 early-stop, and 125 conformal fields. The final audit is field-disjoint and is executed once after freezing the models, calibration, alert logic, and gates.

| Country | Years | Corn | Rapeseed | Soybean | Wheat | Total |
|:--|:--:|--:|--:|--:|--:|--:|
| Argentina | 2017--2024 | 185 | -- | 440 | 126 | 751 |
| Brazil | 2017--2024 | 118 | -- | 293 | 140 | 551 |
| Germany | 2016--2022 | -- | 111 | -- | 188 | 299 |
| Uruguay | 2018--2022 | -- | -- | 572 | -- | 572 |
| **Total** | **2016--2024** | **303** | **111** | **1,305** | **454** | **2,173** |

| Protocol item | Frozen value |
|:--|:--|
| Available / time-eligible fields | 2,173 / 2,171 |
| Development / one-shot audit | 1,849 / 322 fields |
| Input modalities | Sentinel-2, weather, soil, terrain, field geometry, domain metadata |
| Prediction schedule | Three crop-specific absolute-GDD looks |
| Prediction / decision endpoint | Final field yield / training-defined bottom-20% risk |
| Calibration | One-sided split calibration and class-conditional empirical null |
| Sequential action | Alpha spending over three looks; two consecutive hits |
| Statistical unit | Field; pixels are not independent observations |
| Audit boundary | Field-disjoint one-shot audit; not farm-disjoint under the legacy field hash |

All predictors are selected on development data. The 316-field challenge is used for comparative assessment, ablation, and spatial evaluation; its low-yield prevalence is 19.3%. Before final evaluation, a feature cache for 323 audit-role fields is constructed without outcomes, one structurally ineligible field is removed, and hashes of the remaining 322 field identifiers, features, models, protocol, and evaluation script are frozen. The audit script then reveals yields once and refuses repeated execution.

We report field-level RMSE, $R^2$, normalized RMSE (RMSE divided by target standard deviation), climatology-relative RMSE skill, AUROC, and AUPRC; AUPRC is paired with prevalence. Country--crop robustness is summarized by macro $R^2$ and the worst supported group. Prediction contrasts use paired country--crop-stratified field bootstraps. FPR, sensitivity, and precision are computed once per field-season with two-sided exact Clopper--Pearson intervals; a 10,000-replicate farm-cluster bootstrap resamples farms and retains all their fields. Spatial contrasts likewise bootstrap fields, never pixels.

### 4.2. Comparative assessment on locked development data

Table 2 compares methods at the same final GDD look and on the same 316 fields. Climatology predicts the model-fit country--crop mean. Two Ridge models represent compact spectral--weather and full-feature linear fusion. We further evaluate MLP, Random Forest, HGB, ExtraTrees, and a three-seed temporal Transformer. For image-native comparisons, [AgriFM](https://github.com/CAU-COE-VEICLab/AgriFM) uses its official frozen encoder on six time-available 10-band frames, followed by fit-only PCA and an early-stop-selected Ridge head; ConvLSTM consumes the same image sequence and shared context and averages three seeds. Both use model-fit-only 0.5%--99.5% context winsorization.

**Table 2. Comparative assessment at the final partial-season look.** Bold denotes the best value in each column. PhysCrop-Risk is within 1.35% of the lowest RMSE while providing the strongest low-yield ranking.

| Method | RMSE $\downarrow$ | Pooled $R^2\uparrow$ | AUROC $\uparrow$ | AUPRC $\uparrow$ | Macro-group $R^2\uparrow$ | Worst-group $R^2\uparrow$ |
|:--|--:|--:|--:|--:|--:|--:|
| Country--crop climatology | 1.4637 | 0.7270 | 0.5524 | 0.2298 | -0.0553 | -0.2216 |
| NDVI+NDMI+rain Ridge | 1.3254 | 0.7762 | 0.7409 | 0.3783 | 0.0948 | -0.8275 |
| AgriFM frozen encoder + context | 1.3268 | 0.7757 | 0.6622 | 0.2913 | 0.0302 | -0.3689 |
| Full-feature Ridge | 1.1845 | 0.8212 | 0.7352 | 0.4073 | 0.2515 | -0.5310 |
| MLP | 1.2058 | 0.8147 | 0.7788 | 0.4535 | 0.2947 | 0.0600 |
| ConvLSTM + context | 1.1443 | 0.8332 | 0.7688 | 0.4998 | 0.2703 | -0.2134 |
| Temporal Transformer | 1.1493 | 0.8317 | 0.7841 | 0.4906 | 0.3758 | **0.1992** |
| Random Forest | 1.0702 | 0.8540 | 0.7950 | 0.5718 | 0.3807 | 0.0695 |
| HistGradientBoosting | **1.0223** | **0.8668** | 0.7921 | 0.5627 | **0.4703** | 0.1763 |
| Phenology-aligned ExtraTrees | 1.0459 | 0.8606 | 0.8110 | 0.5717 | 0.4362 | 0.1388 |
| **PhysCrop-Risk dual-tree** | 1.0361 | 0.8632 | **0.8112** | **0.5735** | 0.4477 | 0.1490 |

HGB attains the lowest RMSE (1.0223), whereas PhysCrop-Risk attains RMSE 1.0361, AUROC 0.8112, and AUPRC 0.5735. Its ranking is effectively tied with phenology-aligned ExtraTrees, and the 10% HGB component yields a small paired-bootstrap-supported RMSE reduction: the 95% interval for RMSE(full)$-$RMSE(ExtraTrees) is $[-0.0163,-0.0038]$. Baseline-minus-PhysCrop RMSE intervals are $[0.1732,0.3969]$ for AgriFM and $[0.0531,0.1657]$ for ConvLSTM. Thus the dual-tree is a compact predictor for heterogeneous field records, while the complete system's distinguishing role is converting its output into calibrated repeated decisions.

### 4.3. Audit: early prediction and clean-null decisions

Figure 3 visualizes the audit trajectory. The fully eligible cohort grows from 300 to 322 fields; on the common 300 fields, RMSE still improves from 1.2407 to 1.0887, AUROC from 0.7370 to 0.8272, and AUPRC from 0.4412 to 0.5942. Paired Look-3-minus-Look-1 intervals are $[-0.2527,-0.0537]$, $[0.0446,0.1417]$, and $[0.0652,0.2371]$, respectively. Thus the early curve is not a changing-cohort artifact. Using metadata harvest dates and the last acquisition actually entering each prefix, median leads are 107, 86, and 62.5 days; all 932 evaluated look--field pairs precede harvest.

The original frozen alert audit pooled early-stop and conformal reference fields. Because early stop selected mixture weights, we preserve that result but correct the exchangeability argument post audit: the empirical null is rebuilt from conformal-role non-low-yield fields only, with no change to predictors, cutoffs, alpha allocation, or persistence. Figure 3d distinguishes the original frozen action from the stricter clean-null watch and action points.

![Figure 3. Early prediction and sequential reliability. (a) Audit RMSE versus country--crop climatology. (b) low-yield ranking across GDD looks; the dashed line is final-look prevalence. (c) empirical coverage of frozen one-sided bounds. (d) development operating points, the original frozen action, and post-audit clean-null watch/action.](figure3_early_prediction_and_alerting.png)

**Table 3. One-shot audit and uniform post-audit correction.** Panel (a) is the immutable prediction audit. Panels (b--c) use pre-reveal frozen tree components under the same conformal-only null and decision policy. Ridge is a fixed-protocol post-audit comparator.

| Look | Fields | RMSE $\downarrow$ | Climatology RMSE | Paired RMSE difference, 95% CI | $R^2\uparrow$ | AUROC $\uparrow$ | AUPRC $\uparrow$ | Prevalence |
|:--:|--:|--:|--:|:--:|--:|--:|--:|--:|
| 1 | 300 | 1.2407 | 1.4830 | $[-0.3685,-0.1200]$ | 0.7851 | 0.7370 | 0.4412 | 0.2233 |
| 2 | 310 | 1.1902 | 1.4629 | $[-0.4026,-0.1384]$ | 0.7963 | 0.7539 | 0.4649 | 0.2258 |
| **3** | **322** | **1.0582** | **1.4481** | **$[-0.5158,-0.2694]$** | **0.8360** | **0.8331** | **0.6245** | **0.2298** |

| Predictor | Look-3 RMSE | AUROC | AUPRC | Action TP / FP | FPR | Sensitivity | Precision |
|:--|--:|--:|--:|:--:|--:|--:|--:|
| Climatology | 1.4481 | 0.5780 | 0.2913 | 5 / 18 | 0.0726 | 0.0676 | 0.2174 |
| NDVI+NDMI+rain Ridge | 1.2767 | 0.7472 | 0.4684 | 1 / 0 | **0.0000** | 0.0135 | **1.0000** |
| Frozen ExtraTrees | 1.0640 | 0.8316 | 0.6198 | 8 / 4 | 0.0161 | 0.1081 | 0.6667 |
| Frozen HGB | 1.0879 | 0.8223 | 0.5992 | **10 / 2** | 0.0081 | **0.1351** | 0.8333 |
| **Frozen dual-tree** | **1.0582** | **0.8331** | **0.6245** | 8 / 4 | 0.0161 | 0.1081 | 0.6667 |

| Dual-tree tier | TP / FP | FPR [exact 95% CI] | Sensitivity [exact 95% CI] | Precision [exact 95% CI] |
|:--|:--:|:--:|:--:|:--:|
| Watch | 23 / 12 | 0.0484 [0.0252, 0.0830] | **0.3108 [0.2083, 0.4290]** | 0.6571 [0.4779, 0.8087] |
| **Persistent action** | **8 / 4** | **0.0161 [0.0044, 0.0408]** | 0.1081 [0.0478, 0.2020] | **0.6667 [0.3489, 0.9008]** |

At Look 3, the dual predictor reduces RMSE by 0.390 t ha$^{-1}$ relative to climatology; empirical 90%/95% coverages are 0.9534/0.9876. Under identical clean-null post-processing, the dual model provides the best ranking, while HGB supplies the most sensitive action tier. For the dual model, farm-clustered intervals are [0.0202, 0.0811], [0.1918, 0.4085], and [0.5277, 0.8000] for watch FPR, sensitivity, and precision; action intervals are [0, 0.0325], [0.0156, 0.2090], and [0.3333, 1]. The original null gave TP=8 and FP=5; the clean correction removes one false positive without changing true positives.

### 4.4. Components, robustness, and spatial evidence

Table 4 first isolates the alignment coordinate. Calendar day-of-season, acquisition rank, normalized observed-season fraction, and GDD use the same 316 fields, causal prefix, feature budget, and dual-tree family. GDD has the best RMSE, NRMSE, climatology skill, AUROC, AUPRC, and macro-group $R^2$. Its improvement over calendar time is supported by the paired interval; differences from rank and normalized fraction remain point-estimate advantages because their intervals cross zero. Separately, removing weather increases RMSE from 1.0361 to 1.0936, Sentinel-2 alone reaches 1.1419, and removing domain information reaches 1.3411; the respective paired intervals $[-0.0876,-0.0324]$, $[-0.1489,-0.0618]$, and $[-0.4246,-0.1895]$ exclude zero.

**Table 4. Alignment and group-respecting robustness on development data.** NRMSE divides by target standard deviation; RMSE skill is $1-\mathrm{RMSE}/\mathrm{RMSE}_{clim}$. Alignment intervals report RMSE(GDD)$-$RMSE(alternative).

| Alignment | RMSE | NRMSE | RMSE skill | AUROC | AUPRC | Macro / worst $R^2$ | Paired RMSE 95% CI |
|:--|--:|--:|--:|--:|--:|:--:|:--:|
| Calendar day-of-season | 1.0961 | 0.3913 | 0.2511 | 0.8045 | 0.5571 | 0.4196 / **0.2788** | **$[-0.1328,-0.0001]$** |
| Acquisition rank | 1.0579 | 0.3776 | 0.2772 | 0.7992 | 0.5361 | 0.4223 / 0.1762 | $[-0.0582,0.0125]$ |
| Normalized season fraction | 1.0553 | 0.3767 | 0.2790 | 0.8019 | 0.5712 | 0.4397 / 0.1965 | $[-0.0482,0.0093]$ |
| **Absolute GDD** | **1.0361** | **0.3698** | **0.2921** | **0.8112** | **0.5735** | **0.4477** / 0.1490 | -- |

| Robustness evaluation | Support | Result | Reference / additional evidence |
|:--|:--|:--|:--|
| Leave-one-year-out | 1,849 fields; 9 years | RMSE 1.3251; $R^2$ 0.7790; macro-year $R^2$ 0.7223 | Climatology RMSE 1.6390; difference CI $[-0.3567,-0.2712]$ |
| Five-fold farm-grouped OOF | 1,848 fields; 81 farms | RMSE 1.2224; $R^2$ 0.8120; AUROC 0.8116; AUPRC 0.5435 | NRMSE 0.4336; skill 0.2136; macro/worst $R^2$ 0.2491/$-0.0282$; zero train--test farm overlap |
| Field-clustered spatial mapping | 316 fields; 1,530,144 pixels | Mean field RMSE 1.5234 | Uniform 1.6053, CI $[-0.0980,-0.0663]$; NDVI 1.5671, CI $[-0.0528,-0.0348]$ |
| Within-field spatial ranking | 316 fields | Median Spearman $\rho=0.4024$ | Median bottom-20% AUROC 0.7316 |

The farm analysis is necessary because the audit itself is almost entirely seen-farm: 321/322 fields and 52/53 farms overlap the adapter-reference farm set, leaving only one unseen field. We therefore do not report an unstable audit seen/unseen contrast. Instead, five GroupKFold splits treat farms as indivisible and cover 1,848/1,849 development fields; every fold has zero farm overlap. This complements leave-one-year-out retraining and quantifies the remaining hardest group rather than hiding it.

Figure 4 evaluates the spatial output. On 316 challenge fields and 1,530,144 pixels, the mean field RMSE of the spatial model is 1.5234, compared with 1.6053 for broadcasting the field prediction uniformly and 1.5671 for an NDVI-only residual model. Both paired field-bootstrap intervals exclude zero. Across fields, the median within-field Spearman correlation is 0.4024 and the median AUROC for locating the observed bottom-yield quintile is 0.7316. The qualitative examples illustrate that the residual model recovers heterogeneous structures that a uniform prediction cannot express, while the zero-centering constraint preserves the independently evaluated field-level mean.

![Figure 4. Field-clustered spatial evidence. (a) Three illustrative development-challenge fields showing the latest time-available RGB image, observed within-field anomaly, PhysCrop-Risk map, and NDVI baseline. (b) Paired field RMSE differences relative to uniform and NDVI maps; negative values favor the spatial model. (c) Across-field distributions of within-field Spearman correlation and bottom-20% AUROC.](figure4_spatial_evidence.png)

## 5. Conclusion

PhysCrop-Risk reframes partial-season yield prediction as a calibrated decision problem. The system aligns irregular multimodal observations in crop-specific thermal time, predicts final field yield from information available at three early looks, calibrates low-yield evidence against held-out non-low-yield references, and requires temporal persistence before issuing a conservative action alert. This architecture connects four quantities that are often reported separately: regression accuracy, tail-risk ranking, uncertainty calibration, and the field-season false-positive rate. Mean-preserving residual maps add within-field evidence without changing the field prediction or inflating the statistical sample size.

The frozen prediction audit supports this systems-level formulation: on 322 fields, final-look RMSE is 1.0582 t ha$^{-1}$ versus 1.4481 for country--crop climatology, AUROC is 0.8331, and AUPRC is 0.6245. A stricter conformal-only null preserves the frozen predictors and yields watch/action FPR of 4.84%/1.61% at sensitivity 31.08%/10.81%. Exact and farm-clustered intervals expose the uncertainty around both tiers. A common-field analysis confirms significant Look-1-to-3 gains, metadata verify that every look precedes harvest, and five-fold farm-grouped OOF remains predictive. Development experiments further show contributions from weather, multimodal context, domain conditioning, and thermal-time alignment, while spatial maps improve over uniform and NDVI-only alternatives. The useful output is therefore not only an end-of-season number, but a staged warning with a measurable reliability--coverage profile.

Several boundaries define the claim. The audit is field-disjoint but not farm-disjoint; only one audit field comes from an unseen farm, so the farm-grouped development estimate cannot replace a strictly independent deployment audit. The original frozen empirical null reused early-stop fields; the conformal-only result is an explicitly post-audit integrity correction and not a second pre-registered reveal. The strict positive-$R^2$ requirement is narrowly missed by Brazil--wheat in the audit ($n=14$, $R^2=-0.0313$) and by Germany--rapeseed in full farm-OOF ($R^2=-0.0282$). Action prioritizes reliability and detects 10.8% of audit low-yield fields; watch is the higher-coverage interface. Finally, the evidence establishes low-yield risk warning on YieldSAT, not causal attribution, guaranteed cross-country transfer, or pixel-level conformal coverage. Future work should execute the clean frozen protocol on farm-independent seasons and prospective agronomic outcomes.
