# Amendment 01: fit-only context winsorization

The first full run was rejected before manuscript entry. One locked challenge
field (`Argentina_DUP1_farm45_field596_corn_2020`) contains a DEM-curvature
value of 69,830,576, while the model-fit range for that feature is approximately
[-0.174, 0.207]. Standardization alone converted it to a 1.93e9-sigma context
value and made the linear AgriFM head extrapolate to 6.25e7 t/ha.

This is a structural numeric outlier rather than a legitimate model comparison.
The shared context adapter is therefore amended to clip every context feature to
its model-fit 0.5th and 99.5th percentiles before fit-only standardization. This
matches the robust handling already used by the full-feature Ridge/MLP baselines
in Experiment 5. The same transform is applied to both AgriFM and ConvLSTM, and
both models are rerun from scratch. No challenge target or final-audit value is
used to estimate a clipping bound.

The integrity flag is also renamed from `audit_files_read: false` to
`no_audit_files_read: true`; the former was semantically correct but incorrectly
failed an `all(checks.values())` assertion.
