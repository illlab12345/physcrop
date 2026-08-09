# Experiment 3 audit

Final status: **PASS on attempt 3**.  Attempts 1 and 2 remain archived.

The selected development policy is global raw calibration at looks 1 and 3 and
hierarchical square-root scaling at look 2.  Its 90% coverage across looks is
0.911 / 0.926 / 0.940; 95% coverage is 0.959 / 0.981 / 0.972.  Worst supported
group coverage is 0.760 / 0.786 / 0.800 at 90% and 0.800 / 0.929 / 0.857 at
95%.  Mean adjustment ratios versus global raw, averaged over looks, are 1.023
and 1.016.

Attempt 1's linear scaling failed efficiency (1.120/1.082).  Attempt 2's square-
root scaling at every look also failed narrowly (1.074/1.057).  These failures
must be disclosed when describing development selection.  Calibration and fit
farms are disjoint, calibration and evaluation fields do not overlap, and final
labels remain unread.

