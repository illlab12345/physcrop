# PhysCrop-Risk Reproduction

This folder reproduces the paper:
*PhysCrop-Risk: Turning Partial-Season Earth Observation into Calibrated Low-Yield Alerts*.

## What is included

- `scripts/` — all pipeline scripts used to produce the paper results;
- `results/yieldsat_risk_v1/` — the frozen authoritative outputs:
  - `MASTER_PROTOCOL.md`, `final_audit/` (freeze manifest, final predictions,
    final report, interpretation),
  - `post_audit_p0p1/` (P0-A, P0-B clean-null comparison, P1 alignment analysis),
  - all experiment `report.json` / `predictions.csv` outputs,
  - frozen feature caches (`cache/*.npz`),
  - frozen look-1/2/3 models (`experiment_02_early_curve/model_look*.joblib`)
    and the spatial residual model (`experiment_07_spatial_maps/spatial_residual_models.joblib`).

Heavy regenerable artifacts are excluded (vision baseline feature caches,
optional baseline model weights, logs). They can be regenerated with the scripts.

## Raw data requirement

The scripts read the raw YieldSAT archive. Expected layout (relative to this
folder, same as the original workspace):

```text
YieldSAT_raw_data/
  <Country>/
    <field_id>/
      metadata-<field_id>.json
      s2_images/S2_L2A_<date>.tif
      scl_masks/S2_L2A_SCL_<date>.tif
      weather/<field_id>.csv
      soil/*.tif
      dem/*.tif
      yield_masks/*.tif
```

Place the raw archive at `../YieldSAT_raw_data` relative to this README's parent
(i.e., `submission_physcrop_risk/YieldSAT_raw_data/`) so `ROOT` resolution in
the scripts matches `results/yieldsat_risk_v1/`.

## Reproduction order

Run from this `reproduction/` directory (all paths are relative to it).

### 0. Raw data audit and eligibility (optional, informational)

```bash
python scripts/audit_yieldsat_raw.py
python scripts/audit_yieldsat_chip_geometry.py
python scripts/finalize_yieldsat_eligibility.py
```

### 1. Feature caches (from raw data)

```bash
python scripts/build_yieldsat_risk_cache.py --roles adapter_reference,locked_external
python scripts/build_yieldsat_risk_cache.py \
  --roles audit_validation \
  --output results/yieldsat_risk_v1/cache/audit_features_blind.npz
```

The audit cache must be built **without** outcomes (`--include-target` is not
used), matching `cache/audit_features_blind.report.json`.

### 2. Experiments 1–9 (development analysis)

```bash
python scripts/run_yieldsat_experiment1.py
python scripts/run_yieldsat_experiment2.py
python scripts/run_yieldsat_experiment3.py
python scripts/run_yieldsat_experiment4.py
python scripts/run_yieldsat_experiment5.py
python scripts/run_yieldsat_experiment6.py
python scripts/run_yieldsat_experiment7.py
python scripts/run_yieldsat_experiment8.py
python scripts/run_yieldsat_experiment9.py
```

### 3. Vision baselines (optional; Table 2 AgriFM / ConvLSTM)

Requires `torch` and, for the frozen AgriFM encoder, `einops`, `mmengine`,
`timm`, plus the external AgriFM weights:

```bash
python scripts/run_yieldsat_vision_baselines.py --skip-agrifm   # ConvLSTM only, or
python scripts/run_yieldsat_vision_baselines.py --smoke         # smoke check
python scripts/validate_yieldsat_vision_baselines.py
```

### 4. Freeze and one-shot final audit

The frozen artifacts are already included. **Do not re-freeze or re-run the
final audit to change results.** The included
`results/yieldsat_risk_v1/final_audit/final_report.json` is authoritative.

If you need to verify the machinery on a fresh environment:

```bash
python scripts/freeze_yieldsat_final_audit.py          # only if not already frozen
python scripts/run_yieldsat_final_audit.py --preflight
python scripts/run_yieldsat_final_audit.py --execute-once
```

The script refuses to run twice and verifies all frozen hashes first.

### 5. Post-audit integrity analyses

```bash
python scripts/run_yieldsat_postaudit_p0a.py
python scripts/run_yieldsat_postaudit_p0b.py
python scripts/run_yieldsat_postaudit_p1.py
```

### 6. Figures

```bash
python scripts/generate_yieldsat_paper_figures.py
python scripts/generate_yieldsat_system_figures.py
```

## Integrity rules

- `final_report.json` is immutable; its status `FAIL_FINAL` is reported honestly.
- `post_audit_p0p1/` results are post-audit corrections/comparisons, not a
  second pre-registered reveal; never present them as such.
- No script in this folder retunes the frozen predictor, calibration, alpha
  spending, or persistence after the audit.
