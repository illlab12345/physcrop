# Experiment 7 audit

Final status: **PASS on attempt 4**.

All 1,254 model-fit and 316 challenge fields are usable.  Challenge evaluation
contains 1,530,144 pixels but inference remains field-clustered.  Mean per-field
RMSE is 1.523 versus 1.605 for a uniform field map and 1.567 for NDVI Ridge;
field-bootstrap CIs for the differences are [-0.098,-0.066] and
[-0.053,-0.035].  Median within-field rho is 0.402 and bottom-quintile AUROC is
0.732.  Mean-centering preserves the field prediction to 1.8e-15 t/ha.

Attempts 1--2 were adapter failures; attempt 3 passed quality but failed support
because static-layer nodata was intersected.  The final field-local imputation is
documented.  No pixel-iid uncertainty claim is made and final labels remain
unread.

