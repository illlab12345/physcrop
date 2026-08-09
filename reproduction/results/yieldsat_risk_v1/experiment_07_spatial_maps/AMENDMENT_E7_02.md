# Amendment E7-02: field-local static-layer imputation

Attempt 3 passed every quality gate but failed support (80 fields).  The adapter
required all 13 soil/terrain layers to be finite at the same pixel, although
these resampled rasters contain modality-specific edge nodata.  This is stricter
than the plan and unrelated to causal image eligibility.

Attempt 4 keeps the SCL/yield support mask and imputes each static layer's nodata
with that layer's median over valid pixels of the same field (zero only if an
entire layer is missing).  Features are then standardized within field as
already frozen.  No yield magnitude enters imputation, no future image is read,
and model, sampling, metrics, and gates are unchanged.

