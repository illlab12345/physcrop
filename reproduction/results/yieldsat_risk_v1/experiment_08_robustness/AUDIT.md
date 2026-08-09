# Experiment 8 audit

Status: **PASS** under all frozen gates.

LOYO evaluates all 1,849 fields: RMSE 1.325 versus climatology 1.639,
bootstrap difference [-0.357,-0.271], macro-year R2 0.722.  The 59-field,
16-farm subset unseen by model-fit has R2 0.709 and low-yield AUROC 0.754.
All four leave-country-out models beat crop climatology, satisfying the frozen
gate, but absolute R2 is negative for Brazil (-0.536), Germany (-8.016), and
Uruguay (-0.106).  This is a major scale-transfer limitation.  The frozen
challenge worst country/crop R2 remains positive at 0.149.  Final labels remain
unread.

