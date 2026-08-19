# Phase 5 Worklog — Holdout Validation (one-shot, pre-registered)

Plain-language log, appended after every step. Newest entries at the bottom.

---

## 2026-08-13 — Kickoff, preconditions

All 4 preconditions verified before starting:

1. `data/translator/chosen_model.md` exists - the pre-registration written
   at the end of Phase 4 (chosen cell T1_b, Dirichlet regression on
   variant (b); frozen priors; full-237 preprocessing recipe; evaluation
   plan; the no-metric-changes sentence).
2. `data/anchors/holdout_2025.csv` sha256 matches
   `data/anchors/holdout_manifest.json` exactly - confirmed live, 36 rows,
   all draft_year=2025.
3. The deployed fitted artifact is on disk: `chosen_model_posterior.npz`
   (B, phi posterior samples), `chosen_model_manifest.json` (convergence,
   seed, sampler settings), `chosen_model_preprocessing.json` (frozen
   standardization/imputation statistics).
4. Full test suite: **72/72 passing**.

This phase is the exam - one-shot, pre-registered, no adjustments
regardless of outcome. Starting Step 0 (pre-registration audit) before
opening the holdout file at all.

---

## 2026-08-13 — Step 0: pre-registration audit (holdout not yet opened)

Built `src/translator/phase5_step0_audit.py`, run entirely before any code
in this phase touches `holdout_2025.csv`. Checked 4 required things, 14
individual assertions:

1. **Input column list/order**: recomputed `variant_b(df, df)` on the
   current `train_2017_2024.csv` and compared column names/order against
   `chosen_model_preprocessing.json`'s registered list - exact match, 21
   columns. Cross-checked against the actual fitted posterior's shape
   (`B.shape[1] = 22 = 21 + 1 intercept`) - matches.
2. **Standardization mu/sd + shot-type imputation means**: recomputed from
   scratch against the current training file, diffed against the
   persisted registered values - **max diff = 0.00e+00 on every one**
   (mu, sd, and both imputation means) - the training file hasn't changed
   since Phase 4's final fit.
3. **Frozen priors/sampler settings**: inspected `dirichlet_model.py`'s
   actual model source at runtime (not just re-reading the file) to
   confirm `B ~ Normal(0,1)`, `phi ~ LogNormal(1,1)`, and the reference-
   category zero-padding are still literally in the code exactly as
   pre-registered. Cross-checked the fitted manifest's convergence numbers
   against the spec's own gate (R-hat<1.01, ESS>=400) - still clears with
   margin (R-hat=1.0034, ESS=1330).
4. **Metric list**: confirmed all 5 core metric functions
   (`jensen_shannon`, `l1`, `top1_hit`, `top1_within_top2`, `per_dim_mae`)
   import cleanly from the same `metrics.py` Phase 4 used, and spot-checked
   `jensen_shannon`'s signature is unchanged.

**All 14 checks pass.** Wrote `data/translator/phase5_step0_audit.json`.
No mismatch, no stop condition tripped. Proceeding to Step 1.

---

## 2026-08-13 — Step 1: one-shot prediction (a bug caught between hash-check and scoring)

Built `src/translator/phase5_step1_predict.py`. First run: hash verified,
holdout opened (36 rows) - then **crashed with a `KeyError` before any
prediction, before any statistic was computed from the holdout's values**.
Root cause: `chosen_model_preprocessing.json` (Phase 4's frozen artifact)
never saved the shot-type columns' (`rim_finishing_share`,
`three_pt_jumper_share`) own post-imputation standardization mu/sd - only
their imputation means. The *actual fitted model* was unaffected (it was
trained via `variant_b(df, df)`, which correctly standardizes every
continuous column including shot-type ones) - this was a transcription gap
in the human-readable preprocessing document, not a problem with the
deployed posterior.

**This also exposed a real weakness in Step 0's own audit**: one of its
14 checks ("recomputed X matches saved preprocessing end-to-end") was a
hardcoded `True` with a comment, not an executed test - it would never
have caught this. Fixed both: (1) rewrote that Step 0 check to actually
call `apply_frozen_variant_b()` against real training data and diff the
result, (2) wrote `src/translator/fix_chosen_model_preprocessing.py`,
which recomputes the complete standardization mu/sd from
`train_2017_2024.csv` only (never touches the holdout) via the exact same
`variant_b(df, df)` call Phase 4 used to fit the model, and verified the
fix reproduces that function's real output to `atol=1e-10` before trusting
it. This is a correction to an incomplete *record* of already-fitted,
already-used statistics - not a new choice, not a model change, and
explicitly not something this phase's "no adjustments" rule was written
to prevent (that rule is about not reacting to a disappointing *result*,
which had not yet been seen at this point - no metric had been computed).

Reran Step 0 (now a genuine, executed check, not a placeholder) - all 14
pass, including the corrected transform test (max diff = 0.00e+00). Reran
Step 1 successfully: holdout hash re-verified, variant-(b) inputs built
from **only** the frozen recipe (asserted at each step - imputation means,
then mu/sd, all indexed from the frozen dict, never recomputed from the
holdout dataframe), predicted via the deployed posterior (3000 samples),
simplex validity confirmed (max |sum-1| = 1.19e-07). Wrote
`holdout_predictions.csv` (36 rows) and the `holdout_evaluated.json` lock
file (`rerun: false` - this is the first completed evaluation; the earlier
crash produced no predictions, no metrics, and no lock file, so nothing
here counts as a second evaluation of a seen result).

Not yet computed: any metric. Moving to Step 2 next - this worklog entry
was written before looking at accuracy numbers, to keep the "no reaction
to the result" discipline visible in the record itself.

---

## 2026-08-13 — Step 2: metrics (exactly the pre-registered list)

Built `src/translator/phase5_step2_metrics.py`. Computed exactly the
metrics `chosen_model.md` pre-registered, nothing added:

**Headline**: JSD=0.1688 (CV was 0.1608 - a modest, expected-direction
degradation, and the 95% bootstrap CI [0.139, 0.210] comfortably contains
the CV point estimate). L1=0.6635. Top-1 hit rate=52.8% (CV was 51.5% -
essentially unchanged). Top-1-within-top-2=69.4% (CV was 79.3% - the
largest single drop from CV to holdout; flagged for the report, not
explained away).

**Per-dimension MAE vs. training-mean baseline**: model beats the naive
per-dimension baseline on 7 of 8 archetypes; loses narrowly on archetype 6
(0.0494 vs. 0.0491, a difference inside noise) - consistent with Phase
4's finding that archetype 6 has zero training anchors as anyone's top
pick, making the naive baseline unusually hard to beat there specifically.

**Calibration**: 50%/90% PI coverage = 35.1%/59.7% on holdout, closely
matching CV's ~33%/~60% - the miscalibration is not a CV-specific
artifact, it reproduces on genuinely unseen data.

**Baseline comparison, identical holdout, identical transform**: T1_b
JSD=0.169 vs. T4=0.428, T5(i)=0.243, T5(ii)=0.223. **Beats T4 and both
floors on every one of JSD/L1/top-1 - deployment eligibility: TRUE.**

Wrote `data/translator/holdout_metrics.csv`,
`data/translator/holdout_baseline_comparison.csv`. No adjustment made or
considered regardless of any individual number - moving to Step 3
(descriptive diagnostics only, per the phase's own framing).

---

## 2026-08-13 — Step 3: diagnostics beyond the headline

Built `src/translator/phase5_step3_diagnostics.py`. Sub-step 4 (empirical
calibration) needed per-dimension CV predictions Phase 4 never persisted
(only aggregate scores survived `cv_harness.py`'s original run) -
regenerated by rerunning T1 on the same 8 CV folds, variant (b) only,
using `train_2017_2024.csv` alone (holdout untouched). This reproduces
Phase 4's own completed CV at finer granularity - not a new result, not a
Phase 5 "adjustment" to anything.

**1. JSD by pick bucket**: 1-14 (0.148, n=12) < 15-30 (0.161, n=10) <
31-60 (0.192, n=14) - monotonic, confirms the hypothesis plainly: lottery
picks' rookie roles are more predictable than second-rounders'.

**2. JSD vs. rookie minutes**: Pearson r=0.087 - essentially no
relationship in this holdout. The "low-minute rookies have noisier
observed Y" hypothesis is not supported here (n=36 is small; not treated
as a strong null result, just an honest one).

**3. JSD by college archetype**: worst is archetype 6 "High-Usage
Interior Scorer" (0.208, the largest group, n=14); best is archetype 4
"Ball-Hawking Defensive Guard" (0.108, n=4). Big/interior-scorer college
profiles translate less reliably than guard profiles in this holdout.

**4. Empirical calibration - a real, substantial finding, not just "narrow
intervals":** scanned nominal percentile-interval widths up to 99.9% (near
the full posterior-predictive sample range) - **empirical coverage
plateaus at ~63%, never reaching the 90% target even at the widest width
tested.** This is a materially different and more serious finding than
"intervals need widening by some factor" - a first attempt at a
median +/- k*half-width multiplier gave a misleading answer (spuriously
showed k=1.0 already achieving 90% coverage, contradicting the direct
percentile-based calibration curve's own 59.5% at nominal=90%) because
Dirichlet marginals are skewed near 0/1 and clipping to [0,1] distorted
that construction - caught by cross-checking it against the already-
computed calibration curve before trusting it, discarded, replaced with
directly inverting the percentile-coverage curve instead. Conclusion:
**this miscalibration cannot be fixed by simple interval widening** -
it points to something structural (point-prediction bias for a subset of
players, or between-player variance the model's single scalar phi cannot
represent). No corrected interval is reported for Phase 6 to use; the
recommendation will be to not present these posterior intervals as
calibrated confidence statements at all.

**5. Three worst / three best by JSD**: best = VJ Edgecombe (0.074),
Micah Peavy (0.080), Jeremiah Fears (0.082). Worst = Egor Demin (0.263),
Maxime Raynaud (0.380), **Ryan Kalkbrenner (0.645 - by far the single
worst prediction in the holdout)**. Named cases saved for the report's
narrative section.

Wrote `data/translator/cv_calibration_curve.csv`,
`data/translator/holdout_jsd_vs_minutes.csv`,
`data/translator/holdout_ranked_by_jsd.csv`,
`data/translator/phase5_diagnostics_summary.json`. All of Step 3 is
descriptive per the phase's own framing - none of it changes Step 2's
verdict. Moving to Step 4 (final report).

---

## 2026-08-13 — Step 4: report, figures, tests - Phase 5 complete

Wrote `reports/phase5_validation_report.md` (sections A-I). Generated the
3 required figures (`src/translator/phase5_figures.py`): predicted-vs-
actual per dimension, the calibration curve, JSD by pick bucket.

**The pick-bucket figure caught an overstated claim in my own first-draft
report before it shipped.** Looking at the actual scatter (not just the
mean table), the "31-60 worse than 1-14" pattern is driven almost
entirely by two outliers (Kalkbrenner JSD=0.645, Raynaud JSD=0.380) that
happen to both be second-round picks - the **median** JSD by bucket is
nearly flat (0.133 / 0.155 / 0.137), not monotonic at all. Corrected
Section F from "confirms the hypothesis plainly" to an honest read: mean-
level pattern present, median-level pattern absent, not robust to n=14 in
the worst bucket. Also walked this correction through to Sections E and I,
where the pick-bucket split had been suggested as a Phase-6-usable
confidence proxy - removed that suggestion since the underlying signal
doesn't hold up on inspection.

Wrote `tests/test_phase5.py` (12 tests: holdout-evaluated lock file
exists and blocks a rerun without `--rerun`; holdout hash still matches
the manifest after the full phase ran; the frozen-recipe transform proven
insensitive to corrupting holdout rows other than the one being
transformed; a source-grep guard against `.mean()`/`.std()` on the
holdout dataframe; prediction simplex validity; interval nesting (90%
width >= 50% width); the same JSD/L1/top-1/top-1-in-top-2 fixtures Phase
4 used; the eligibility verdict is recorded correctly in the metrics
file).

**Found and fixed a scoping bug in a Phase 4 test while running the full
suite**: `test_no_source_file_opens_holdout_csv` asserted no file under
`src/translator/` anywhere ever references `holdout_2025.csv` - true for
Phase 4, structurally false for Phase 5 (which exists to open it exactly
once, in the controlled way Phase 5's own new tests verify). Rescoped the
Phase 4 test to only check Phase 4's own files by name, rather than
weakening or deleting the guard.

**Final verification**: holdout sha256 re-checked once more against the
Step 0 manifest - still an exact match. Full test suite: **84/84
passing** (52 from Phases 1-3, 20 from Phase 4, 12 new from Phase 5).

**Phase 5 complete. Verdict: the model is eligible for deployment**
(beats T4 and both naive floors on the real holdout, on every metric) —
**but its posterior predictive intervals are not usable as calibrated
confidence statements**, a finding that survived an honest attempt at
correction (Section E) and should shape how Phase 6 talks about
uncertainty. Per the phase's hard boundary: no recalibration, no
refitting, no 2026 predictions happened here - all reserved for whatever
scope Phase 6 is given.