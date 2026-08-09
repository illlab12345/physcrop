# Experiment 6 audit

Status: **PASS** under all frozen gates.

Full RMSE is 1.036.  Removing weather, using S2 only, and removing domain
conditioning worsens RMSE to 1.094, 1.142, and 1.341, with paired 95% intervals
for full-minus-ablation [-0.088,-0.032], [-0.149,-0.062], and
[-0.425,-0.190].  No-static and no-indices effects are smaller and their CIs
cross zero.  Rank alignment RMSE is 1.058 versus absolute-GDD 1.036, but its CI
[-0.056,0.011] crosses zero; GDD alignment is therefore promising, not proven
superior.  The persistence decision tradeoff also passes.  Final labels remain
unread.

