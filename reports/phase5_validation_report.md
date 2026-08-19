# Phase 5 Validation Report — Holdout Evaluation (One-Shot)

The exam. `data/anchors/holdout_2025.csv` (36 players, class 2025) was
opened exactly once, scored exactly as `data/translator/chosen_model.md`
specified, with no adjustment made or considered at any point regardless
of any individual number. See `reports/phase5_worklog.md` for the full
step-by-step narrative, including two real bugs caught and fixed *before*
any holdout number was seen (Step 0/1) and one methodological correction
caught *during* a purely descriptive diagnostic (Step 3) that changed no
verdict.

## A. Pre-registration audit

All 14 checks pass, run entirely before the holdout was opened
(`data/translator/phase5_step0_audit.json`):

| # | Check | Result |
|---|---|---|
| 1 | Input column list/order (21 cols) matches fitted model | PASS |
| 1 | Fitted B's dimensionality (p_aug=22) matches registered cols+intercept | PASS |
| 2 | Standardization mu matches recomputed (tol 1e-9) | PASS (0.00e+00) |
| 2 | Standardization sd matches recomputed (tol 1e-9) | PASS (0.00e+00) |
| 2 | Shot-type imputation means match recomputed (tol 1e-9) | PASS (0.00e+00) |
| 2 | Frozen-recipe transform reproduces `variant_b(df,df)` exactly | PASS (0.00e+00) |
| 3 | B prior is Normal(0,1) in current code | PASS |
| 3 | phi prior is LogNormal(1,1) in current code | PASS |
| 3 | Reference-category zero-padding present | PASS |
| 3 | Manifest records converged fit (R-hat<1.01, ESS>=400) | PASS (1.0034, 1330) |
| 3 | Manifest records zero divergences | PASS |
| 3 | Chosen cell matches (T1, variant b) | PASS |
| 4 | All 5 core metric functions unchanged from Phase 4 | PASS |
| 4 | `jensen_shannon` signature unchanged | PASS |

**One real bug was caught by this audit process, before any holdout
scoring**: `chosen_model_preprocessing.json` never recorded the shot-type
columns' own post-imputation standardization mu/sd (only their imputation
means) — the deployed model itself was unaffected (it was trained on
`variant_b(df,df)`'s real, correctly-standardized output), but a
first-draft version of this audit's own "end-to-end" check was a
hardcoded `True` that would never have caught it. Both were fixed:
the transform bug (recomputed the missing statistics from
`train_2017_2024.csv` only, verified against the real fitted output to
1e-10) and the audit's own placeholder check (now genuinely executes the
transform). Full account in the worklog.

## B. Headline results — holdout vs. CV

| metric | CV (Phase 4) | Holdout (n=36) | 95% bootstrap CI |
|---|---|---|---|
| JSD (primary) | 0.1608 | **0.1688** | [0.139, 0.210] |
| L1 | 0.663 | 0.664 | [0.590, 0.758] |
| Top-1 hit rate | 0.515 | 0.528 | [0.361, 0.667] |
| Top-1-within-top-2 | 0.793 | 0.694 | — |

**Reading**: JSD degrades modestly from CV (0.161 → 0.169, +5% relative)
— the CV point estimate sits comfortably inside the holdout's bootstrap
CI, so this is the "normal, expected" direction and magnitude of
degradation the pre-registration anticipated, not a red flag. Top-1 hit
rate is essentially unchanged (52.8% vs. 51.5%). **Top-1-within-top-2
drops more noticeably** (79.3% → 69.4%) — the largest gap between CV and
holdout of any metric here; flagged plainly, not explained away. With
n=36, all of these estimates carry real width (see the bootstrap CIs) —
that width is part of the finding.

## C. Baseline comparison — deployment eligibility

| model | JSD | L1 | top-1 hit |
|---|---|---|---|
| **T1_b (chosen)** | **0.169** | **0.663** | **0.528** |
| T4 Hungarian | 0.428 | 1.181 | 0.194 |
| T5(i) global mean | 0.243 | 0.848 | 0.417 |
| T5(ii) height tercile | 0.223 | 0.798 | 0.361 |

Computed on the identical 36 holdout players, identical transform.
**Beats T4: yes. Beats T5(i): yes. Beats T5(ii): yes, on every metric.**

**Deployment eligibility verdict: ELIGIBLE.**

## D. Per-dimension MAE vs. baseline

| dim | model MAE | baseline MAE (train-mean) | better? |
|---|---|---|---|
| y0 | 0.121 | 0.157 | yes |
| y1 | 0.058 | 0.063 | yes |
| y2 | 0.065 | 0.102 | yes |
| y3 | 0.115 | 0.171 | yes |
| y4 | 0.070 | 0.083 | yes |
| y5 | 0.096 | 0.119 | yes |
| **y6** | **0.049** | **0.049** | **no (statistically a wash)** |
| y7 | 0.089 | 0.104 | yes |

Beats the naive per-dimension baseline on 7/8 archetypes. The one
exception (y6) is not a red flag on its own: archetype 6 has zero training
anchors as anyone's top pick (Phase 4's own preflight finding) — its true
values are uniformly small, so even a naive constant prediction is nearly
optimal there; a wash against that baseline is expected, not a model
weakness specific to this dimension.

## E. Calibration

| interval | nominal | CV empirical | Holdout empirical |
|---|---|---|---|
| 50% | 50% | ~33% | 35.1% |
| 90% | 90% | ~60% | 59.7% |

The miscalibration found in Phase 4's CV **reproduces on genuinely unseen
data**, essentially unchanged — not a CV-specific artifact.

**Attempted a correction, found it doesn't work — reported as a finding,
not glossed over.** Regenerated per-dimension CV posterior-predictive
samples (rerunning T1 on the same 8 CV folds, variant b, training data
only) and scanned percentile-interval widths up to nominal 99.9% (nearly
the full posterior-predictive sample range). **Empirical coverage
plateaus around 63%, never reaching the 90% target at any width tested.**
This rules out a simple "intervals are too narrow by a fixed factor"
story — no amount of widening the existing intervals gets there. It
points to something a scale correction can't fix: likely point-prediction
bias for a subset of players (see Section F's worst-3 cases) and/or
between-player variance that the model's single scalar `phi` cannot
represent (one precision parameter is shared by every player and every
archetype dimension).

**Recommendation for Phase 6: do not present these posterior intervals as
calibrated confidence statements.** If uncertainty must be communicated,
qualitative framing (e.g., citing the overall JSD/L1 range from Section B,
or naming known-hard cases like Section F's worst-3) is defensible; the
pick-bucket split (Section F) turned out to be a weak, outlier-driven
signal on the median and should **not** be leaned on as a confidence
proxy without a larger holdout confirming it first. None of this changes
the core recommendation: the model's own 50%/90% interval language cannot
be trusted even after attempting a correction.

## F. Diagnostics beyond the headline

**JSD by pick bucket** — weaker and more caveated than it first looks:

| bucket | mean JSD | median JSD | n |
|---|---|---|---|
| 1–14 | 0.148 | 0.133 | 12 |
| 15–30 | 0.161 | 0.155 | 10 |
| 31–60 | 0.192 | **0.137** | 14 |

The **mean** is monotonic and looks like a clean confirmation of "lottery
picks are more predictable." The **median** is not — the 31-60 bucket's
median (0.137) is nearly identical to the 1-14 bucket's (0.133). The
mean-level pattern is driven almost entirely by two outliers that both
happen to be second-round picks: Kalkbrenner (JSD=0.645) and Raynaud
(JSD=0.380), both discussed below. **Honest reading: with n=14 in the
worst-populated bucket and two extreme outliers doing most of the work,
this holdout does not cleanly confirm the lottery-more-predictable
hypothesis** — it's directionally consistent in the mean but not robust
to two data points, and the median comparison actively argues against a
strong version of the claim. Worth re-testing with a larger holdout in a
future draft class, not stated as settled here.

**JSD vs. rookie minutes**: Pearson r=0.087 — no meaningful relationship
in this holdout. The "low-minute rookies have noisier observed Y"
hypothesis is not supported here; reported as a null result, not
suppressed for not fitting the expected story.

**JSD by college archetype**: worst is archetype 6, "High-Usage Interior
Scorer" (0.208, n=14, the largest group in this holdout); best is
archetype 4, "Ball-Hawking Defensive Guard" (0.108, n=4). Big/interior-
scorer college profiles translate less reliably than guard profiles here.

**Three worst predictions**, with college recipe → predicted → actual:

- **Ryan Kalkbrenner** (Creighton, pick 34, 1479 min) — JSD=0.645, by far
  the single worst prediction in the holdout. College recipe: archetype 6
  (0.49) / archetype 2 (0.28) / archetype 3 (0.23) — a rim-protecting-big
  profile. Predicted rookie recipe was a moderate, spread-out mixture
  (top weight only 0.234 on archetype 2). **Actual rookie recipe: a pure
  1.0 on archetype 4** — his real rookie season was statistically so
  extreme in one direction that the simplex projection saturated to a
  single vertex, a known behavior for statistically extreme players (seen
  before with Zach Edey's *college* recipe, also a pure 1.00). The model
  predicted meaningful but insufficient weight on archetype 4 (0.174) —
  right direction, wrong magnitude, and beaten badly by the saturation.
- **Maxime Raynaud** (Stanford, pick 42, 1964 min) — JSD=0.380. Predicted
  heaviest on archetype 0 (0.424); actual is heaviest on archetype 1
  (0.513) and archetype 2 (0.230) — a genuine directional miss, not just
  a magnitude one.
- **Egor Demin** (BYU, pick 8, 1308 min) — JSD=0.263, yet **pred_argmax
  and true_argmax both equal 3** — a correct top-1 hit that still ranks
  among the worst by JSD, because the *rest* of the distribution is
  wrong (predicted spreads real weight onto archetypes 0 and 7 that carry
  none of the true weight, while underweighting the true secondary mass
  on archetype 0). Concrete illustration of why JSD and top-1 hit rate
  can disagree: JSD scores the whole distribution, top-1 only the mode.

**Three best**: VJ Edgecombe (0.074, Baylor, pick 3), Micah Peavy (0.080,
Georgetown, pick 40 — a good prediction well outside the lottery, evidence
the pick-bucket pattern in Section F is a tendency, not a hard rule),
Jeremiah Fears (0.082, Oklahoma, pick 7).

## G. What the translator cannot see

Written for a reader who will act on these predictions — cited from
`reports/college_archetypes.md`'s appendix, not restated at length:

- **16 of the NBA basis's 29 dimensions have no college counterpart at
  all** (Appendix A2) — 7 shot-location dimensions and 9 play-type
  dimensions. The model cannot distinguish a rim-protecting roll man from
  a perimeter-mobile big, a spot-up shooter from a movement shooter, or a
  pick-and-roll handler from an isolation scorer — all collapse to the
  same 12-feature signature. **This is a direct, plausible contributor to
  Ryan Kalkbrenner's miss above** (Section F): a college rim-protector's
  rookie role can bifurcate along exactly the axes this model can't see.
- **Three selection biases baked into who is even in the training data**
  (Appendix A6): elite prospects whose final college season was
  truncated by injury/suspension/opt-out are entirely absent from
  training (the model has never seen that profile — predictions for such
  a player would be extrapolation, not interpolation); every anchor
  earned a real rookie role by construction (≥300 minutes), so the model
  answers "what role will he play if he plays," never "will he play";
  gap-year paths are excluded, so players whose college-to-NBA transition
  wasn't a single clean draft cycle aren't represented.
- **Coaching, scheme, and opportunity are not in this model at all.** A
  rookie's role is jointly determined by his own skill and his team's
  system, injuries ahead of him on the depth chart, and coaching
  decisions — none of which any statistical feature here captures.
- **Sample size, stated plainly**: 237 training anchors, 36 holdout
  players. Every number in this report — including the bootstrap CIs in
  Section B — should be read with that denominator in mind.

## H. Obstacles

1. A real bug in `chosen_model_preprocessing.json` (missing shot-type
   standardization statistics) was caught by the hash-verified holdout
   read crashing with a `KeyError` — before any prediction or scoring
   happened. Traced, fixed (recomputed from training data only, verified
   to 1e-10 against the actual fitted model's real output), and exposed a
   second issue: the audit step meant to catch exactly this kind of
   mismatch had one placeholder check that never actually ran. Both fixed
   before re-attempting Step 1.
2. A first attempt at deriving an interval-widening multiplier
   (Section E) produced a self-contradictory result (implying a modest
   k=1.0 already fixed calibration) - caught by cross-checking it against
   the already-computed raw calibration curve before trusting it, traced
   to a flawed median+/-k*half-width construction on skewed, [0,1]-clipped
   Dirichlet marginals, and replaced with a direct percentile-curve
   inversion.
3. Regenerating the CV-fold posterior-predictive samples for Section E
   required refitting T1 eight more times (variant b, training data
   only) - a real but modest compute cost (~2 minutes total), not a
   Phase 5 "adjustment" to the deployed model, which was never touched.

## I. Verdict and open decisions for Phase 6

**Verdict: the model is eligible for deployment** (Section C) — it beats
the Hungarian identity baseline and both naive floors on every metric, on
genuinely held-out data, by a wide margin. Its point predictions (JSD,
top-1 hit rate) hold up close to CV expectations. **Its uncertainty
quantification does not hold up** — the posterior predictive intervals
are miscalibrated in a way that simple widening cannot repair (Section E)
— and Phase 6 must not present them as calibrated confidence statements.

**Open decisions for Phase 6:**

1. **Interval language**: given Section E's finding, what should Phase 6
   cards say about uncertainty instead of a nominal credible interval?
   The pick-bucket split is **not** a reliable candidate on this holdout
   (Section F - median-flat, mean driven by 2 outliers). More viable:
   qualitative "how well do we usually do for players like this" framing
   grounded in Section B's overall JSD/L1 range, or simply omitting
   interval claims and leaning on comps (T3's neighbor tables, Phase 4
   Section G) instead.
2. **The Ryan Kalkbrenner case** (Section F): worth a specific callout in
   Phase 6 about saturation risk — statistically extreme rookies can land
   on a simplex vertex that no smooth prediction anticipates.
3. **College archetype 6** (Section F): the least reliable college
   profile to translate in this holdout, and also the largest group
   (n=14/36). Worth flagging to Phase 6 as the profile needing the most
   hedged framing.
4. Per the phase's hard boundary: **no post-hoc recalibration of the
   deployed model happens here** — that decision, if made, belongs to
   whoever scopes Phase 6, informed by this report, not enacted by it.
