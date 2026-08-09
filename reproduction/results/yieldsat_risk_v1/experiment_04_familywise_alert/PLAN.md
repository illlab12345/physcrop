# Experiment 4 plan: field-season family-wise false alert control

Frozen on 2026-08-01 before sequential alert evaluation.

## Alert semantics

For field `i` at look `k`, issue a formal low-yield alert only when its one-sided
upper conformal bound `U_ik` is below the model-fit q20 threshold applicable to
its country/crop hierarchy.  A false alert for a truly non-low-yield field implies
one-sided noncoverage.  Therefore the union bound controls the probability of
one or more false alerts across the three looks, subject to split-conformal
exchangeability.

## Frozen policies

- `naive`: prediction below q20; diagnostic comparator without error control.
- `marginal_05`: alpha=0.05 independently at each look; diagnostic comparator.
- `bonferroni_equal`: alpha=(1/60, 1/60, 1/60), total 0.05.
- `alpha_spending` (primary): alpha=(0.010, 0.015, 0.025), total 0.05, allocating
  more power to the later and more accurate look.
- `alpha_spending_persistent`: same bounds, but requires alerts at two
  consecutive available looks; operational precision comparator.

Conformal cells and scaling follow the frozen Experiment 3 selected policy:
global raw at looks 1/3 and hierarchical power-0.5 at look 2.  Quantiles use the
same finite-sample ceiling rank.  The q20 reference is frozen from model-fit.

## Units and metrics

The inference unit is a field-season.  Main FPR/FWER is the fraction of truly
non-low-yield fields with at least one alert.  We report its one-sided 95%
Clopper-Pearson upper bound, sensitivity, precision, specificity, earliest alert
look/GDD, stage-wise FPR, and supported country/crop groups.  Pixel counts are
never used as sample size.

## Passing gates for primary alpha spending

- alpha sum <=0.05, causal bound construction, disjoint conformal/evaluation
  fields, and zero final-label reads;
- empirical field-season FPR <=0.05 and one-sided 95% upper bound <=0.08;
- sensitivity >=0.05 and at least three true-positive fields (a minimal
  non-degeneracy gate for a deliberately stringent formal alert);
- precision >=0.50;
- no supported country/crop group has FPR >0.15.

Failure of sensitivity is reported as lack of power; FWER control alone is not a
useful-system pass.

