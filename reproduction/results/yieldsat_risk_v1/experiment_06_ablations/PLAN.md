# Experiment 6 plan: causal, modality, and decision ablations

Frozen on 2026-08-01 before ablation fitting.

## Prediction ablations at look 3

Every variant uses identical model-fit/early-stop/challenge fields and the same
ExtraTrees+HistGB family with its mixture weight selected on early-stop only.

- `full`: frozen six-bin absolute-GDD, all dynamic/static/domain features.
- `rank_aligned`: six equally spaced acquisitions from the causal prefix instead
  of absolute-GDD interpolation; no future acquisition is used.
- `no_weather`: remove cumulative precipitation.
- `no_static`: remove soil, terrain, area, and circularity.
- `no_indices`: remove NDVI, NDMI, NDRE, and EVI while retaining raw bands.
- `s2_only`: raw Sentinel-2 bins plus country/crop indicators.
- `no_domain`: remove country/crop indicators and predict globally standardized
  raw yield rather than country/crop-normalized yield.
- `extratrees_only`: full features with HistGB weight fixed to zero.

## Decision ablations

Use frozen Experiment 4 outputs to compare continuous-yield UCB, any-hit null
conformal watch, and consecutive-hit action alert.  This avoids refitting or
reusing development outcomes.

## Metrics and gates

- RMSE, AUROC, macro/worst group R2, early-stop-selected weight, and paired
  country/crop-stratified bootstrap delta versus `full`;
- `full` must be within 2% of the best ablation and significantly beat `s2_only`
  in RMSE (upper 95% delta <0);
- absolute-GDD alignment must not be >2% worse than rank alignment, and its point
  RMSE must be lower;
- at least two modality/domain removals must worsen point RMSE;
- persistence must lower watch FPR by at least 50%, retain at least 25% of watch
  true positives, and keep precision >=0.80.

Negative or null ablations remain reportable; no feature is claimed essential
unless supported by its paired interval.

