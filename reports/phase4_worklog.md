# Phase 4 Worklog — Translator: Fit, Ablation, Cross-Validation, Pre-Registration

Plain-language log, appended after every step. Newest entries at the bottom.

---

## 2026-08-13 — Kickoff, preconditions

Checked all 3 preconditions before starting:

1. **Phase 3 accepted**: `data/anchors/anchors.csv` confirmed 273 rows, 66 columns,
   draft years 2017-2025 (9 classes), matching Phase 3's final deliverable exactly.
2. **Phase 2 frozen college basis + recipes.csv**: both present
   (`data/college/model/k8_frozen_basis.npz`, `data/college/recipes.csv`).
3. **Owner decisions on record**: confirmed against the appendix the owner
   appended to `reports/college_archetypes.md` in the prior turn - conf_tier
   map approved as proposed (A7); Morant `years_in_college=2` kept as
   computed, not hand-corrected (A4); position excluded from v1 covariates
   (A4); shot-type mix approved for variant (b) with training-fold-mean
   imputation + missing indicator (A7); archetype labels drafted but not
   required to build the translator (A7 - required before Phase 6 only).

**Environment check**: `numpyro`/`jax` not installed in the `.nets` venv
(pymc also absent). Spec prefers numpyro NUTS, pip-installable - installing
now, in background (large download, jax has real weight). `scipy` 1.11.4
and `sklearn` 1.7.2 already present (needed for T2 ridge, T3 k-NN, T4
Hungarian's `linear_sum_assignment`).

Starting Step 0 (quarantine & preflight) while the numpyro install runs in
the background - Step 0 needs neither package.

---

## 2026-08-13 — Step 0: quarantine & preflight

Built `src/translator/step0_quarantine.py`. Split `anchors.csv` (273 rows)
into `data/anchors/train_2017_2024.csv` (237 rows, classes 2017-2024) and
`data/anchors/holdout_2025.csv` (36 rows) - both counts matched the spec's
expected values exactly. Wrote `data/anchors/holdout_manifest.json` with
the holdout file's sha256
(`34c5cec5c5de2fdbe68007a021839d942bbed3ce3dd8fbdae8c1ca6e616116ed`) for the
phase-end unchanged-hash test.

**Leakage boundary, recorded verbatim per spec:** intentionally shared
across train/holdout/deployment are the *unsupervised coordinate systems*
- the college basis, per-season college z-scores, the NBA basis and
per-season NBA standardization. They contain no rookie-outcome information
and are identical at deployment time (a 2026 rookie's college season sits
in the basis pool too). Everything *supervised or data-dependent in
fitting* - input standardization statistics, imputation means,
zero-compression N, λ, k, any prior tuning - is computed inside training
folds only.

**Preflight scans** (train only, n=237):
- **Y argmax distribution**: archetype 3 dominant (85/237, 35.9%) - this
  is the naive top-1 floor every later hit rate must be read against.
  Distribution: {0:50, 1:11, 2:22, 3:85, 4:15, 5:27, 7:27}.
  **Archetype 6 has ZERO training anchors with that argmax** - top-1 hit
  rate for that class is structurally unmeasurable in CV. Caught a real
  near-miss here: my first draft of this check also compared against the
  full 273-row table (including the 2025 holdout) to see if the gap
  persists there - that's a leakage violation even as a pure diagnostic
  (reading 2025-class labels before Phase 5), caught and removed before
  running. The question of whether archetype 6 is empty in the holdout too
  stays genuinely open until Phase 5.
- **College-recipe (X-side) argmax distribution**: archetype 6 dominant
  (94/237, 39.7%). Distribution: {1:3, 2:26, 3:13, 4:23, 5:57, 6:94, 7:21}
  (archetype 0 also empty on the X side, in training).
- **age_at_draft × years_in_college implausibility scan**: 1 row - Semi
  Ojeleye (2017, age 22.55, years_in_college=1, matched via
  `name_year_team` not `direct_athlete_id`). Flagged, not edited - same
  treatment as the Morant case in Phase 3, possibly the same class of
  undercount (a transfer or juco season CBBD's roster data doesn't fully
  capture), but not confirmed.
- **Missingness**: shot-type mix (rim-finishing share, 3PT-jumper share)
  missing 7.2% of training rows - close to the spec's ~6% expectation, the
  small difference is just the train/full-table split changing the
  denominator slightly. All other input columns 0% missing.
- **Simplex validity**: both `c_alpha` and `y` pass (non-negative, sum to 1
  within 1e-6) on all 237 training rows.

Step 0 complete. Starting Step 1 (input variants) - does not need
numpyro/jax, which is still installing in the background for Step 2's T1.

**numpyro/jax install note**: the first install attempt hung for 25+
minutes with a `CLOSE_WAIT` TCP connection and zero output (confirmed via
`sample` profiling and `lsof`) - killed it and retried with
`--progress-bar off`, which completed normally in under a minute. The
install **silently upgraded scipy 1.11.4 -> 1.15.3**, which conflicts with
the `archetypes` package's own pin (`scipy<1.12`, per pip's own printed
warning). Verified this doesn't actually break anything before proceeding:
`import archetypes` still succeeds, and the full existing test suite
(52 tests, Phases 1-3) still passes unchanged. One real, harmless
side-effect noticed: `archetypes` now prints "Using Jax backend" instead
of "Using NumPy backend" at import time (it auto-detects jax's presence) -
this does not affect any already-frozen basis/artifact, since those are
loaded from disk, not refit, anywhere in Phase 4.

---

## 2026-08-13 — Step 1: input variants (pre-registered, no additions later)

Built `src/translator/compositional.py` (Smithson-Verkuilen zero-compression
+ ILR transform) and `src/translator/input_variants.py` (variants a/b/c).

**ILR chosen over drop-one/ALR** for variant (a)'s 8->7 dim reduction, fixed
and documented per spec's requirement to pick one: ALR's log-ratio against
an arbitrary reference component isn't isometric and the reference choice
is arbitrary; ILR's orthonormal basis means Euclidean operations in the
transformed space (ridge, k-NN distances) correspond to a real geometry on
the simplex. Implemented the standard Helmert-type orthonormal basis
construction; self-check confirms the basis is truly orthonormal
(`V.T @ V = I`) and the full round-trip (compress -> ILR -> inverse-ILR)
recovers the original composition to 3.3e-16 max error.

**Variant shapes confirmed exactly as expected**: (a) 7 dims, (b) 21 dims,
(c) 28 dims (matches the spec's "≈20-21" and "≈27-28" ranges precisely).
Variant (b)'s shot-type-mix missingness (7.2% per Step 0's preflight) is
imputed with the **training-fold mean only** plus one shared missing-
indicator column (1 if either shot-type feature is missing) - both fit
inside `variant_b(train_df, test_df)`, which only ever sees the two
DataFrames passed to it, so there is no path for a test row to leak into
its own imputation mean.

All continuous columns in variant (b) standardized on training-fold
mean/SD (`conf_tier` and the missing-indicator are left as 0/1, not
standardized). The `y` target's zero-compression (needed for T1's
Dirichlet likelihood) is a separate function, `build_targets()`, using the
same Smithson-Verkuilen formula with n = training-fold size, applied
identically to train and test rows so both live in the same coordinate
system the model was fit in.

Self-check run on a real single-fold split (train != 2017, test == 2017):
all three variants produce NaN-free matrices of the expected shape.

Step 1 complete. Starting Step 2 (models T2-T5 first, since they don't
need numpyro; T1 next once the jax/numpyro install - see above - is
confirmed not to have broken anything).

---

## 2026-08-13 — Step 2: models T1-T5

Built `src/translator/metrics.py` (JSD/L1/top-1/top-1-in-top-2/per-dim MAE,
self-tested against hand-computable fixtures) and `src/translator/models.py`
(T2-T5), then `src/translator/dirichlet_model.py` (T1).

**T2 (ridge)**: per-dimension `Ridge` on raw y, λ selected via a nested
4-fold `KFold` built from the OUTER training fold's rows only (never
another leave-class-out split - the inner selection just needs an
unbiased λ estimate, not another class-generalization test), clip-at-zero
+ renormalize.

**T3 (k-NN)**: always runs in variant-(c) space regardless of which outer
variant loop the cell belongs to, per spec (variant is a T1/T2 design-
matrix concept; T3's neighbor space is fixed). k selected the same nested
way. Neighbor tables emitted on request for named row indices - real
output for a 2020-holdout smoke test surfaced sensible neighbors for that
season's rookies (Paolo Banchero, Johnny Davis, Josh Christopher, ... for
one test player).

**T4 (Hungarian)**: extracted the college basis's (8,12) archetype
z-profiles directly from `k8_frozen_basis.npz` and the NBA basis's
matching 12 columns from `basis_2025_26/basis.npz` (asserted all 12 shared
names exist in the 29-feature order first, per spec). Cosine similarity
8x8, `linear_sum_assignment` on the negated matrix (Hungarian minimizes;
cosine similarity should be maximized). Prediction = college recipe
permuted through the assignment - a pure permutation preserves the simplex
automatically, no clip/renorm needed.

**T5**: (i) training-fold global mean recipe: (ii) height-tercile mean,
cutpoints from training-fold quantiles.

**T1 (Dirichlet regression, the primary model)**: `numpyro` NUTS, 4 chains,
reference-category identification (archetype 7 fixed at logit 0 - its
coefficient row is never sampled). Priors fixed per spec: `B ~ Normal(0,1)`
(including an intercept row - the spec's prior section doesn't split one
out, but Step 0's preflight found wildly unequal archetype base rates
[archetype 3 at 36% of training rookies, archetype 6 at 0%], so an
intercept-free model would badly misfit the marginal immediately; added
one and treated it under the same Normal(0,1) prior as everything else,
the simplest reading consistent with "no tuning"), `phi ~ LogNormal(1,1)`.
Divergences handled via a target_accept ladder (0.8 -> 0.9 -> 0.95), never
silenced; a fold that still fails R-hat<1.01 / ESS>=400 after the full
ladder raises rather than returning a bad fit.

**Smoke test** (single fold, 2017 held out, variant a): converged at the
first rung (target_accept=0.8), max R-hat=1.0044, min ESS=874 (well above
the 400 floor), 0 divergent transitions out of 3000 total samples. Enabled
`numpyro.set_host_device_count(4)` so the 4 chains run as true parallel CPU
devices - fit time dropped from ~13s to ~3.5s for this fold/variant. At
that per-fit speed, the full 24-fit T1 grid (8 folds x 3 variants) should
take low single-digit minutes, not the spec's own "minutes each, expect
tens of minutes to an hour or so" ballpark.

All 5 model families produce valid simplex predictions on a real smoke
test (2020 held out): T2(b) JSD=0.122/top1=0.52 was the best single-fold
reading, **T4 (Hungarian) was the worst of everything including both naive
floors** (JSD=0.369, top1=0.08, vs. T5(i)'s 0.231/0.40 and T5(ii)'s
0.161/0.48) - a real, substantive early signal that a simple identity
permutation between the two independently-fit archetype spaces performs
badly, exactly the kind of result the spec frames T4 as needing to be
beaten by. One fold only - not a CV result, not reported as one.

Step 2 complete. Starting Step 3 (the full leave-one-draft-class-out CV
harness, 8 folds x 12 cells).

---

## 2026-08-13 — Step 3: CV harness, a real spec ambiguity caught mid-run

Built `src/translator/cv_harness.py`. First run (8 folds, all 12 nominal
cells) was **stopped partway through** on catching a genuine internal
inconsistency in the spec: T3's own description says "k-NN in the
standardized variant-(c) space" (read literally: T3 is fixed to variant
(c), never varies), but Step 3's own grid formula - "Grid = 3 variants x
{T1, T2, T3} + T4 + T5(i) + T5(ii) = 12 cells" - only resolves to 12
arithmetically if T3 ALSO runs across all 3 variants like T1/T2
(3+3+3+1+1+1=12; the fixed-T3 reading gives 3+3+1+1+1+1=10, not 12). My
first implementation followed T3's own bullet literally (fixed to variant
c) and produced identical T3_a/T3_b/T3_c results in fold 1 - a visible
tell that something was off, not just a spec technicality.

**Resolution**: the explicit "= 12 cells" arithmetic is much harder to
misread than a single prose sentence, and it's internally load-bearing (it
sets the shape of `cv_results.csv` and the report's grid table) - went
with **T3 varies across all 3 variants**, passing each variant's own
`variant_fn` into `fit_t3_knn` (previously hardcoded to `variant_c`).
Verified the fix didn't break anything (`tests/test_translator.py`, 20/20
still passing) before restarting the full CV run. Not treated as a
stop-and-ask condition - the correct reading was unambiguous once both
passages were checked against each other, and re-running cost only a few
minutes of compute, not a decision only the owner could make.

Also recording the deliberate resolution of a smaller, related ambiguity
this raised: what does the nested hyperparameter selection for T2/T3 use
as its inner folds? The spec says "nested inside training folds" without
specifying inner-fold structure. Used a plain row-level 4-fold `KFold` on
the outer training fold's own rows (not another leave-class-out split) -
the inner selection only needs an unbiased lambda/k estimate, not another
class-generalization test, and a second layer of class-holdout would
shrink the already-small per-class inner training data further for little
benefit.

Re-running the full 8-fold x 12-cell grid now.

Full run complete: 96 cell-fits (12 cells x 8 folds), wall time 217s. T1
converged cleanly on all 24 fits (max R-hat 1.0094, min ESS 768, 0
divergences). Pooled results (primary metric = mean JSD, 237 players):
T3_b best (0.1484), T2_b essentially tied (0.1492), T1_b third (0.1608,
but pre-registered as primary, and its gap over T3_b sits inside the
fold-level spread), T4 dramatically worst (0.4383, beaten by both naive
floors). Ablation: (b) beats (a) by ~18-19% JSD across all three
variant-sweeping models; (c) is mildly worse than (b) alone in all three -
variant (b) is the sweet spot, concatenation doesn't help.

T1 calibration came back poor: 50%/90% posterior predictive intervals
cover only ~33%/~60% empirically (should be ~50%/~90%) - the model is
overconfident across all 3 variants. Real, reported plainly (Section F of
the CV report), not smoothed over - a genuine limitation for any Phase 6
interval-based card language if T1 is the chosen cell.

T4's failure has direct player-level evidence, not just an aggregate
number: Trae Young's college recipe (0.91 on archetype 5, the cleanest
archetype in the whole college model) gets assigned by T4 to NBA archetype
6, but his real rookie-season recipe (independently exact-matched against
the app's stored 2025-26... no, his was an earlier class, matched via the
Phase 3 Y-side computation) actually loads on NBA archetype 3 - even the
single most confident, most famous college archetype loading in the
dataset gets mapped to the wrong NBA archetype by cosine similarity on the
12 shared coordinates alone.

Step 4 complete: wrote `reports/phase4_cv_report.md` (sections A-I).
**CHECKPOINT reached** - presenting the report and a recommendation (T1_b,
reasoned from the CV numbers, with T3_b flagged as a legitimate alternative
the owner may prefer). Waiting for confirmation before writing
`data/translator/chosen_model.md` and stopping - no further step touches
the holdout regardless of which cell is chosen.

**Owner confirmed T1_b** (Dirichlet regression, variant b) at the
CHECKPOINT.

---

## 2026-08-13 — Post-CHECKPOINT: final fit + pre-registration

Built `src/translator/fit_final_model.py`: refit T1 on variant (b) using
**all 237 training rows** (no CV split - this is the production fit Phase
5 will actually evaluate), reusing `variant_b(df, df)` (passing the full
set as both "train" and "test" args) to get the full-data standardization/
imputation statistics through the same tested code path rather than
duplicating the logic. Converged cleanly: max R-hat=1.0034, min ESS=1330,
0 divergent transitions - comfortably inside the same gate every CV fold
was held to.

Saved `data/translator/chosen_model_posterior.npz` (B: (3000,22,7), phi:
(3000,)), `chosen_model_preprocessing.json` (the frozen standardization
mu/sd and shot-type imputation means, computed on the full 237 - explicit
per-column values, not just a description, so Phase 5 has no ambiguity
about what to apply to the holdout), and `chosen_model_manifest.json`.

Wrote `data/translator/chosen_model.md` - the pre-registration document:
chosen cell, every frozen hyperparameter (priors, reference-category
identification, sampler settings), the exact preprocessing recipe with
column order, the frozen Phase 5 evaluation plan (identical metric set to
Step 3, the holdout's sha256 it must be checked against first, the
consistency re-asserts), and the required closing sentence that no metric
may be added, removed, or reweighted after this document.

**Final verification before declaring the phase complete**: holdout
sha256 re-checked against the Step 0 manifest - still an exact match,
confirming nothing in Steps 1-4 (including this final full-data refit)
ever touched `holdout_2025.csv`. Full test suite re-run:
`tests/test_translator.py` (20 tests) + all of Phases 1-3's existing
tests = **72/72 passing**.

**Phase 4 complete.** Per the phase's hard boundary, this is the end -
no Phase 4 code opens the holdout or predicts on it; Phase 5 begins only
when explicitly started as its own phase.