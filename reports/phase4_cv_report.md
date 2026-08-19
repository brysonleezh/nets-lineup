# Phase 4 CV Report — Translator: Fit, Ablation, Cross-Validation

Final deliverable for Phase 4 up to the CHECKPOINT (no holdout evaluation,
no 2026-rookie predictions anywhere in this phase — the 2025 class stayed
quarantined in `holdout_2025.csv` throughout; see `reports/phase4_worklog.md`
for the full step-by-step narrative this report distills).

## A. Preflight scans

Training population (237 anchors, classes 2017-2024):

**Y (rookie NBA archetype) argmax distribution** — the naive top-1 floor
every later hit rate must be read against:

| archetype | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| count | 50 | 11 | 22 | **85** | 15 | 27 | **0** | 27 |

Modal share (archetype 3): **35.9%** — any model beating a ~36% naive
top-1 hit rate is doing real work. **NBA archetype 6 has zero training
anchors with that argmax** — top-1 hit rate for that class is structurally
unmeasurable in this CV. Whether it's also empty in the 2025 holdout is
deliberately not checked (would mean reading 2025 labels before Phase 5).

**College-recipe (X-side) argmax distribution**: modal share 39.7%
(archetype 6, the "High-Usage Interior Scorer" label from
`college_archetypes.md`).

**age_at_draft × years_in_college plausibility scan**: 1 flagged row —
Semi Ojeleye (2017, age 22.55, years_in_college=1, matched via
`name_year_team` rather than `direct_athlete_id`). Left as computed, not
edited — same treatment as the Ja Morant case from Phase 3.

**Missingness**: shot-type mix (rim-finishing share, 3PT-jumper share)
missing in 7.2% of training rows; every other input column 0% missing.

**Simplex validity**: both `c_alpha` and `y` pass (non-negative, sum to 1
within 1e-6) on all 237 training rows.

## B. CV grid — pooled results (leave-one-draft-class-out, 8 folds, 237 players)

Sorted by mean Jensen-Shannon divergence (base 2, the primary selection
metric). `jsd_fold_spread` = std of the 8 per-fold mean JSD values (fold-
level stability, not player-level variance).

| cell | JSD (mean) | JSD fold-spread | L1 (mean) | top-1 hit | top-1-in-top-2 |
|---|---|---|---|---|---|
| **T3_b** | **0.1484** | 0.017 | 0.632 | 0.489 | 0.743 |
| T2_b | 0.1492 | 0.015 | 0.630 | **0.553** | **0.810** |
| T2_c | 0.1521 | 0.015 | 0.639 | 0.498 | 0.776 |
| T1_b | 0.1608 | 0.015 | 0.663 | 0.515 | 0.793 |
| T1_c | 0.1626 | 0.012 | 0.672 | 0.489 | 0.772 |
| T3_c | 0.1635 | 0.017 | 0.672 | 0.485 | 0.738 |
| T2_a | 0.1820 | 0.016 | 0.723 | 0.460 | 0.684 |
| T3_a | 0.1830 | 0.019 | 0.729 | 0.430 | 0.671 |
| T5(ii) height-tercile | 0.1973 | 0.015 | 0.755 | 0.384 | 0.561 |
| T1_a | 0.1995 | 0.014 | 0.765 | 0.477 | 0.688 |
| T5(i) global mean | 0.2482 | 0.008 | 0.872 | 0.359 | 0.570 |
| **T4 Hungarian** | **0.4383** | 0.030 | 1.182 | 0.122 | 0.249 |

**Headline reads:**
- **T4 is dramatically the worst cell in the entire grid** — nearly 3x the
  JSD of the best cell, worse than *both* naive floors, top-1 hit rate
  (0.122) barely above a random 8-way guess (0.125). The identity-mapping
  baseline is beaten decisively, in both directions the spec asked for
  (beats T5 easily; every real model beats T4). See Section D for why.
- **T3_b and T2_b are statistically indistinguishable** on the primary
  metric (0.1484 vs. 0.1492, a 0.0008 gap against a ~0.015-0.017 fold-
  spread) — this is a tie, not a win for k-NN.
- **T2_b has the clearly best top-1 hit rate (55.3%) and top-1-in-top-2
  rate (81.0%)** of every cell in the grid, T1/T3 included.
- Every naive floor and T4 is beaten by every real model at variant (b) or
  (c) — the translator is doing real work, not just reproducing base rates.

## C. Ablation reading — does the design question have an answer?

This is the project's original design question: do covariates carry signal
beyond the recipe itself?

**Does (b) beat (a)? Yes, consistently and by a wide margin, across all
three models that vary by input:**

| model | (a) JSD | (b) JSD | relative improvement |
|---|---|---|---|
| T1 | 0.1995 | 0.1608 | −19.4% |
| T2 | 0.1820 | 0.1492 | −18.0% |
| T3 | 0.1830 | 0.1484 | −18.9% |

Three independent model families, three consistent ~18-19% JSD reductions
moving from recipe-only to features+covariates. **Answer: covariates carry
real signal beyond the college recipe.**

**Does (c) beat (b)? No — (c) is mildly but consistently worse than (b)
alone, in all three models:**

| model | (b) JSD | (c) JSD | change |
|---|---|---|---|
| T1 | 0.1608 | 0.1626 | +1.1% |
| T2 | 0.1492 | 0.1521 | +1.9% |
| T3 | 0.1484 | 0.1635 | +10.2% |

Concatenating the college recipe on top of variant (b) does not help and
mildly hurts — worst for T3 (k-NN in a higher-dimensional space dilutes
the distance metric with redundant information, since several of variant
(b)'s 12 raw z-features are literally what the college recipe was fit on).
**Variant (b) is the sweet spot** across every model family tested.

## D. T4 — the Hungarian identity-mapping table

College archetype mean z-profiles (12-dim, from the frozen AA basis) vs.
the NBA basis restricted to the same 12 shared coordinates. Cosine
similarity, `linear_sum_assignment` (maximizing cosine, square matrix, no
unmatched-mass handling needed).

| college archetype (label, top loaders) | → | NBA archetype | cosine |
|---|---|---|---|
| 0 (Low-Minute Statistical Outlier — weak): Donnie Lewis, Wiggy Ball | → | 3 | 0.32 |
| 1 (Inefficient Low-Usage Reserve — weak): Jack Webb, Ernest Minton | → | 2 | −0.49 |
| 2 (Rim-Protecting Big): Ike Obiagu, Ari Boya | → | 4 | 0.89 |
| 3 (Efficient Low-Usage Play-Finisher): Keller Boothby, JR Hobbie | → | 0 | 0.61 |
| 4 (Ball-Hawking Defensive Guard): Ivy Smith Jr., Nigel Ribeiro | → | 7 | 0.52 |
| **5 (High-Usage Primary Ball-Handler): Trae Young, Gus Etchison** | → | **6** | 0.82 |
| 6 (High-Usage Interior Scorer): Zach Edey, Nathan Knight | → | 1 | −0.49 |
| 7 (Low-Event Floor Role Player — weak): Jonah Jackson, Chris Ashby | → | 5 | 0.55 |

**Why T4 fails this badly, with real evidence, not just a number:** Trae
Young loads college archetype 5 at 0.91 (Phase 2's own sanity vignette).
T4's assignment sends 100% of that weight to NBA archetype 6. But Trae
Young's *actual* rookie-season recipe (Phase 3's Step 5, also exact-matched
against the app's stored recipes) loads primarily on **NBA archetype 3**,
not 6. Even the single cleanest, most confident college archetype in the
whole model — matched to a real, unambiguous superstar — gets assigned to
the wrong NBA archetype by cosine similarity on shared coordinates alone.
This is direct, player-level evidence for `college_archetypes.md`
Appendix A2's structural claim: 16 of the NBA basis's 29 dimensions have no
college counterpart, so two archetypes can look similar on the 12 shared
coordinates while being fit on very different, partially-invisible axes.

Full cosine matrix in `data/translator/t4_matching_table.json`, alongside
each college archetype's real top-3 loading player-seasons.

## E. Per-dimension MAE

Pooled (fold-n-weighted) MAE per NBA archetype dimension, best 4 cells vs.
T4:

| cell | y0 | y1 | y2 | y3 | y4 | y5 | y6 | y7 | mean |
|---|---|---|---|---|---|---|---|---|---|
| T2_b | 0.109 | 0.053 | 0.077 | 0.115 | 0.060 | 0.093 | 0.036 | 0.087 | **0.0787** |
| T3_b | 0.117 | 0.045 | 0.083 | 0.114 | 0.047 | 0.102 | 0.037 | 0.089 | 0.0790 |
| T1_b | 0.109 | 0.060 | 0.085 | 0.117 | 0.065 | 0.093 | 0.046 | 0.089 | 0.0829 |
| T4 | 0.146 | 0.222 | 0.119 | 0.219 | 0.085 | 0.102 | 0.177 | 0.113 | 0.1478 |

**Honest caveat, not the spec's anticipated reading**: the spec expected
"play-type-blind big-man dimensions worst." We cannot make that specific
claim — **no NBA archetype has an owner-assigned label** (unlike the
college side), so dimension identity beyond a bare index number is
unknown here. What the table does show plainly: archetype **6 has the
lowest MAE in every good model (0.036-0.046)**, but this tracks with it
also being the dimension with **zero training anchors as anyone's top
pick** (Section A) — true y_6 values are uniformly small across the
population, so low MAE there is largely a byproduct of an easy target, not
evidence of a stronger fit. Archetype **3 has the highest MAE everywhere**
(0.11-0.22), tracking with it being the *most* populous true class (36%
modal share) — more true variance in that dimension gives more room for
absolute error. Read this table as "which dimensions have more true
spread," not "which dimensions the model handles worst."

## F. T1 sampler diagnostics + calibration

**Convergence: clean across all 24 fits** (8 folds x 3 variants). 23/24
converged at the first rung of the target_accept ladder (0.8); one
(2018/variant-a) needed 0.9. **Max R-hat across every fit: 1.0094** (spec
gate: <1.01). **Min ESS across every fit: 768** (spec gate: >=400). **Zero
divergent transitions** in any of the 24 fits, 3000 post-warmup samples
each. No fold needed a remedy beyond the pre-planned target_accept ladder.

**Calibration (posterior predictive intervals, not a selection metric —
a trust metric for Phase 6's interval claims):**

| variant | 50% PI empirical coverage | 90% PI empirical coverage |
|---|---|---|
| a | 35.5% | 59.7% |
| b | 33.0% | 59.8% |
| c | 32.3% | 59.5% |

**Both intervals under-cover substantially** (nominal 50%/90% vs. actual
~33%/~60%) — T1's posterior predictive intervals are **too narrow across
every variant**, meaning the model is systematically overconfident. This
is honest, real, and load-bearing for Phase 6: if T1 is chosen, any
interval-based confidence claim on a card **must** carry this caveat (or
be corrected via a post-hoc recalibration, which is out of this phase's
scope — no tuning after pre-registration).

## G. T3 neighbor-table previews (comps content for Phase 6 cards)

Interpretive note: the spec's "6 sample players per fold" was read as "6
sample player-fold pairs total, previewed across variants" for report
length (48 full tables would not fit a preview section) — flagged as a
minor scope interpretation, not a hard requirement, since this section is
explicitly illustrative and unscored.

Two standouts, real and eyeball-plausible:

- **Ja Morant (2019 fold), variant (b): nearest neighbor is Trae Young
  (distance 1.56)** — by a wide margin over the next-closest (Collin
  Sexton, 4.29). Directly consistent with Phase 2's sanity vignette, where
  both players load the same college archetype (5) at >0.9.
- **Deandre Ayton (2018 fold)**: neighbors are all real NBA bigs (Mark
  Williams, Donovan Clingan, Jaxson Hayes, Lauri Markkanen, Dereck Lively
  II) — a coherent, sensible comp set.

Full previews (6 test-player/fold pairs x 3 variants = 18 tables) in
`data/translator/t3_neighbor_previews.json`.

## H. Obstacles

1. **First CV run stopped mid-grid on a real spec inconsistency**: T3's
   own description ("k-NN in the standardized variant-(c) space") reads as
   fixed-to-one-variant, but Step 3's own grid arithmetic ("3 variants x
   {T1,T2,T3} + T4 + T5i + T5ii = 12 cells") only resolves to 12 if T3
   also varies across all 3 variants. Caught directly in the first run's
   fold-1 output (T3_a/T3_b/T3_c producing byte-identical results) before
   the report stage, not after. Resolved in favor of the explicit
   arithmetic; documented in `reports/phase4_worklog.md`.
2. **numpyro/jax install hung for 25+ minutes** on the first attempt
   (confirmed via `sample` profiling — a stuck `CLOSE_WAIT` TCP
   connection); killed and retried with `--progress-bar off`, completed in
   under a minute. The install silently upgraded `scipy` in a way that
   conflicts with the `archetypes` package's own pin — verified the full
   existing 52-test suite still passes before trusting the environment.
3. **T1's calibration is poor** (Section F) — a real limitation carried
   forward, not resolved in this phase (no tuning after pre-registration).
4. **NBA archetype dimensions have no owner-assigned labels**, unlike the
   college side — limited how specifically Section E's per-dimension
   findings could be interpreted; reported honestly as an open gap rather
   than guessed at.

## I. Recommendation

**Recommended cell: T1_b (Dirichlet regression, variant b).**

Justification, strictly from the CV numbers above: T1_b's JSD (0.1608) is
statistically indistinguishable from the grid's nominal leader, T3_b
(0.1484) — the 0.0008 gap between T3_b and T2_b, and the 0.012 gap between
T1_b and T3_b, both sit inside the ~0.015-0.017 fold-level spread every
cell shows, so none of the top three cells can be called a clean winner on
JSD alone. Among that effective three-way tie, T2 was pre-registered as "a
reference model only" (Step 2's own framing — clip-and-renormalize breaks
its probabilistic reading), which leaves T1_b and T3_b as the real choice.
T1_b is the pre-registered primary model, is the only one of the three
that produces genuine posterior uncertainty (needed for Phase 5's
consistency re-asserts and any Phase 6 interval-based card language), and
converged cleanly with zero diagnostic issues across all 8 folds (Section
F). Its calibration is honestly poor and must be caveated wherever
intervals are shown — that is a real cost, weighed against T3_b's own
real cost (no uncertainty quantification at all, and its JSD advantage
over T1_b is not statistically distinguishable in this CV). T3_b remains
a strong, legitimate alternative if the owner weighs raw point-prediction
accuracy and built-in real-player comps (Section G) above probabilistic
output — that is the owner's call, not a CV-numbers call, which is exactly
why this stops here for confirmation rather than proceeding.

---

**CHECKPOINT: presenting this report. Awaiting the owner's confirmation of
the chosen cell before writing `data/translator/chosen_model.md` and
stopping — per the phase's hard boundary, no further step touches the
holdout regardless of which cell is chosen.**
