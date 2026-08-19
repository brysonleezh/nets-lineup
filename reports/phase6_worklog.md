# Phase 6 Worklog — Deployment, Rookie Cards & Prediction Freeze

Plain-language log, appended after every step. Newest entries at the bottom.

---

## 2026-08-13 — Kickoff, preconditions: STOPPED on 2 blocking gaps

Checked all 4 preconditions before starting:

1. Phase 5 accepted: `reports/phase5_validation_report.md` and
   `data/translator/holdout_evaluated.json` both present - confirmed.
2. `data/translator/chosen_model.md` present, unchanged - confirmed.
3. **`config.yaml` (owner-supplied, blocking) does not exist at all.**
   Investigated how much could be pre-verified from data already in the
   repo before reporting back, rather than stopping cold with nothing
   actionable:
   - All 3 rookies (Mikel Brown Jr., Tyler Bilodeau, Joshua Jefferson)
     **do** have a 2025-26 (CBBD season=2026) college season present in
     both `data/college/recipes.csv` and `data/college/shared_features.parquet`
     - the core "can this model even see him" check passes for all 3.
   - Real `cbbd_athleteId` + `draft_pick_overall` recovered from
     `data/raw/cbbd/draft_picks_all.json`: Mikel Brown Jr. (athleteId
     200182, pick #6, Brooklyn Nets), Tyler Bilodeau (athleteId 529, pick
     #43, Brooklyn Nets), Joshua Jefferson (athleteId 142, pick #28,
     **Minnesota Timberwolves** - not Brooklyn).
   - **Birthdate is not available anywhere in local data** - CBBD's own
     `roster_2026.json` has `dateOfBirth: null` for all 3 (consistent with
     the appendix's earlier finding that CBBD birthdates are 94.9%
     missing), and the NBA-side `player_bio` table has zero 2026-draft-
     class rows at all (that table only populates once a player has
     logged real box-score minutes, which none of these 3 have).
4. **Archetype labels in `reports/college_archetypes.md` are all still
   marked `(draft)`** - AI-drafted candidates from an earlier session,
   explicitly flagged pending owner review/edit, never finalized.

**Two real stop conditions, reported rather than worked around or
guessed past:**
- The Joshua Jefferson draft-team mismatch (Minnesota per CBBD vs. Nets
  per the current roster) needs an owner call before it goes in
  `config.yaml` - most likely explanation is a post-draft trade/signing,
  but not assumed without confirmation.
- Birthdate for all 3 rookies needs an owner-supplied value (or explicit
  sign-off to attempt a live nba_api lookup, which was not attempted
  without being asked, since the spec frames this block as owner-supplied).

Presenting both to the owner now. Not proceeding to Step 1 until
resolved.

---

## 2026-08-14 — Preconditions resolved, config.yaml written

Owner resolved both stop conditions:

1. **Joshua Jefferson's draft-team mismatch**: owner's working assumption
   was post-draft trade/signing to Brooklyn. Confirmed live rather than
   left as an assumption - queried `nba_api.stats.endpoints.commonallplayers`
   for the 2026-27 season and found `TEAM_ABBREVIATION=BKN` for his real
   `PERSON_ID` (1643538) - he is genuinely on Brooklyn's roster now,
   regardless of who drafted him. Also recovered all 3 rookies' real NBA
   `PLAYER_ID`s this way (none existed in the local `player_bio` table,
   which only populates once a player has logged real box-score minutes -
   `commonallplayers` with `is_only_current_season=0, season="2026-27"`
   was the one query that surfaced pre-debut rookies).
2. **Birthdates**: owner authorized a live `commonplayerinfo` attempt.
   Succeeded for all 3 (Mikel Brown Jr. 2006-04-03, Tyler Bilodeau
   2004-04-17, Joshua Jefferson 2003-11-21) - cached under
   `data/raw/nba/commonplayerinfo/{player_id}.json`, same convention as
   Phase 3. Cross-checked `DRAFT_NUMBER` from this same call against
   CBBD's `draft_picks_all.json` for all 3 - exact match (6, 43, 28),
   a real consistency check between two independent sources, not assumed.
3. **Archetype labels**: owner adopted the 2026-08-13 AI-drafted candidates
   in `reports/college_archetypes.md` as final - removed all 16 `(draft...)`
   markers and updated the file's top-of-file note to record the
   2026-08-14 confirmation, rather than leaving stale "pending review"
   language in a now-finalized document.

Wrote `config.yaml` (repo root, matching the existing
`ncaa_bridge_config.yaml`/`fit_harness_config.yaml` convention: commented,
`yaml.safe_load`-compatible): the 3 rookies' full identity block, the 2026
draft date (2026-06-24, a late-June estimate consistent with Phase 3's
established +/-few-days tolerance), the saturation threshold (0.85,
config-adjustable per spec), and `comps_k: 5`.

All 4 preconditions now clear. Starting Step 1 (deployment refit).

---

## 2026-08-14 — Step 1: deployment refit on all 273 anchors

Built `src/translator/phase6_step1_deploy_refit.py`. Refit T1_b (unchanged
specification - same priors, same reference-category identification, same
sampler settings, seed=42 matching the pre-registered 237-fit's seed
policy) on all 273 anchors (2017-2025).

**Preprocessing built complete from the start this time** - explicitly
verified the persisted `deployment_preprocessing.json` covers all 19
continuous columns (including the two shot-type ones that Phase 4's
version omitted and Phase 5 had to patch), and re-derived X from the
saved mu/sd/imputation values to confirm it reproduces `variant_b`'s real
output exactly (atol=1e-10) before trusting it - the Phase 4 gap is a
named Phase 6 test, not just a promise in a comment.

**Convergence**: clean, same standard as every prior fit - max R-hat=1.0055,
min ESS=1466, 0 divergent transitions out of 3000 samples, converged at
the first target_accept rung (0.8).

**237-vs-273 coefficient agreement**: correlation=0.9811 (high - adding 36
more real anchors barely moved the fit), largest single coefficient shift
=0.227 (on `three_pt_jumper_share`'s effect on archetype-column 0) - well
under the 1.0 flag threshold. **No material disagreement** - exactly what
you'd expect from adding ~15% more rows to an already-converged fit, not
surprising, not flagged.

Wrote `data/translator/deployment/posterior.npz` (B: (3000,22,7), phi:
(3000,)), `data/translator/deployment_preprocessing.json`,
`data/translator/deployment/manifest.json` (git hash, seed, convergence,
coefficient-agreement numbers).

Step 1 complete. Starting Step 2 (rookie predictions).

---

## 2026-08-14 — Step 2: rookie predictions, comps, saturation analysis

Built `src/translator/phase6_step2_predict_rookies.py`. Hit one real bug
on first run: `recipes.csv` names its columns `alpha_0..alpha_7`, not
`c_alpha_0..c_alpha_7` (that rename only happens downstream, in
`anchors.csv`) - fixed immediately, no impact on any prior phase's output.

Built each rookie's raw feature row from real, verified sources only:
`shared_features.parquet` (2025-26 season row - already correctly
season-relative z-scored) for the 12 shared features + PORPAG + conference;
`config.yaml` for pick/birthdate; roster `startSeason` data for years-in-
college (same method as Phase 3); live shooting-file lookup for shot-type
mix. Transformed through `apply_frozen_transform()`, which takes every
statistic from `deployment_preprocessing.json` only - asserted no rookie
value ever enters a mean/sd/imputation calculation.

**Predictions** (posterior mean, valid simplex confirmed):
- **Mikel Brown Jr.** (Louisville, pick 6): college top archetype 5 "High-
  Usage Primary Ball-Handler" (0.572). Predicted rookie top archetype 3
  "Combo Guard" at 0.546 - a confident, concentrated prediction. **4 of 5
  nearest comps (Rob Dillingham, Dennis Smith, Dylan Harper, Keyonte
  George) all became this same NBA archetype 3 as rookies** (weights
  0.41-0.81) - an unusually coherent comps signal.
- **Tyler Bilodeau** (UCLA, pick 43): college top archetype 6 "High-Usage
  Interior Scorer" (0.524). Predicted top archetype 5 "Shooting
  Specialist" at only 0.345 - a much less concentrated prediction; comps
  split (2 became archetype 5, 3 became archetype 0) - genuinely mixed
  signal, reported as such.
- **Joshua Jefferson** (Iowa State, pick 28): college top archetype 6,
  same as Bilodeau (0.479). Predicted top archetype 0 at only 0.237 - top-3
  weights nearly flat (0.237/0.226/0.202) - the least confident of the
  3 predictions; comps also mixed with no dominant pattern.

**Saturation flag**: none of the 3 rookies cross the 0.85 threshold
(0.572, 0.524, 0.479) - no card carries the saturation caveat.

**Saturation-rule training-set evidence - an important, honest null
result**: searched all 237 training anchors (8-fold out-of-sample CV
refit, variant b, training data only) for true `y_max >= 0.9` - **found
zero**. The "Kalkbrenner rule" is grounded in exactly one documented case
(the 2025 holdout, Phase 5) - not a pattern confirmed across many training
examples. Checked softer thresholds for honest context, not as a
substitute: y_max>=0.85 -> 1/237, >=0.8 -> 6/237, >=0.75 -> 12/237 -
extreme concentration is genuinely rare in this dataset, which is exactly
why Kalkbrenner's case stood out enough to motivate a rule in the first
place. This will be reported plainly in Section D, not oversold as a
broadly-validated pattern.

**A notable finding for the report's framing**: `data/basis_2025_26/
archetype_definitions.csv` (the NBA basis's own real archetypoid
exemplars) shows **Ryan Kalkbrenner himself is the current real-player
exemplar for NBA archetype 4 "Mobile Big"** - his rookie season was
statistically distinctive enough that he's literally the nearest real
point to that archetype's centroid, independent of and consistent with
Phase 5's finding that his true recipe saturated to 1.0 on that same
archetype.

Wrote `data/projections/nets_rookies_2026.csv`,
`data/translator/rookie_projections_full.json`,
`data/translator/saturation_rule_evidence.csv`.

Step 2 complete. Starting Step 3 (rookie cards).

---

## 2026-08-14 — Step 3: rookie cards

Built `src/report_template/rookie_card.{html,md}.jinja` (new templates,
not the existing print-oriented `report.html.jinja` - different audience
and purpose, a standalone narrative card rather than a dense A4 diagnostic
dashboard), `src/translator/rookie_card_labels.py` (parses college labels
programmatically from `reports/college_archetypes.md` rather than
hardcoding them, so cards can't silently drift from the finalized source;
loads the NBA-side `archetype_labels.csv`/`archetype_definitions.csv` the
rest of the app already uses), and `src/translator/phase6_step3_cards.py`
(assembly + render).

**A genuinely nice finding surfaced while pulling archetypoid names for
the "closest real comparison" prose**: `archetype_definitions.csv` shows
**Ryan Kalkbrenner is the real archetypoid exemplar for NBA archetype 4
"Mobile Big"** in the currently-frozen basis - the same Kalkbrenner whose
extreme rookie-season saturation was Phase 5's most dramatic finding. His
statistical profile is distinctive enough that he's literally the nearest
real point to that archetype's own centroid - independent confirmation of
the same thing from a completely different angle.

**One readability fix caught before shipping**: the college archetype
labels for archetypes 0/1/7 end in "(weak — see note above)", a phrase
that only makes sense inside `college_archetypes.md` itself - on a
standalone card with no such note nearby, it would confuse a reader who's
never seen the project (the design constraint Step 3 explicitly requires).
Added `card_safe_label()` to swap that dangling reference for a self-
contained parenthetical ("a noisier, less distinct archetype in this
model") - same information, no broken pointer.

Rendered all 3 individual cards (HTML + md) plus the combined
`all_rookies.{html,md}` page. Playwright's chromium wasn't installed yet
(one-time setup per `AI_USAGE.md`'s CLAUDE.md convention) - installed it
and rendered a real screenshot of Mikel Brown Jr.'s card to visually
confirm the design reads cleanly, not just that the HTML is well-formed:
all 6 sections render in order, the confidence line and limits footnote
are legible and self-contained, and the predicted-vs-college bar charts
are visually distinct (accent orange vs. purple) so a reader can't
confuse the model's input with its output.

Structural check on all 4 HTML files (balanced tags) + confirmed all 4 .md
twins exist. Step 3 complete. Starting Step 4 (prediction freeze).

---

## 2026-08-14 — Step 4: prediction freeze

Built `src/translator/phase6_step4_freeze.py`. Wrote
`data/projections/predictions_frozen.json`: sha256 of
`nets_rookies_2026.csv`, `config.yaml`, and the deployment manifest; UTC
timestamp; git commit; the Phase 5 holdout metrics carried into the frozen
record itself (so it makes its own accuracy claim, not just a prediction);
all 3 rookies' predicted top-3 in plain text.

**Guard tested for real, not just written**: ran the script a second time
with no flag - refused with a clear error, exit code 1, before touching
the existing file. Ran again with `--refreeze "testing the refreeze guard
mechanism"` to confirm the append-only `refreeze_log` mechanism actually
works - it did, correctly preserving the prior csv hash alongside the new
one. **Then deleted that test artifact and re-froze cleanly** - the real
deliverable is a genuine single freeze record with an empty
`refreeze_log`, not one carrying a test entry from my own verification
process.

**Predictions frozen before the 2026-27 NBA season began** (today,
2026-08-14, ahead of the 2026-27 season) - stated plainly per spec, to be
repeated in `reports/phase6_deployment_report.md`'s Section E.

Step 4 complete. Starting Step 5 (pre-committed review script).

---

## 2026-08-14 — Step 5: pre-committed review script

Built `src/eval/review_2026_predictions.py`, with the frozen methodology
written into its own module docstring (per spec) so it can't drift
between now and whenever it's actually run. Structured the core logic as
pure, synthetic-data-testable functions (`eligibility_threshold`,
`check_eligibility`, `build_standardization_for_population`,
`score_actual_vs_predicted`) separated from the real-data orchestration
(`load_2026_27_population`, `main`), so the methodology itself can be
proven correct today without needing real 2026-27 data, which doesn't
exist yet.

Wrote `tests/test_review_2026.py` (9 tests): eligibility pro-ration math
(full and shortened season), standardization on a synthetic population
(including a rejection test for a zero-variance feature, mirroring the
same guard `nba_standardization.py` has), scoring on synthetic recipes
(a perfect-prediction case and a maximally-wrong case), and **a full
end-to-end pipeline test using the REAL frozen NBA basis with a fully
synthetic 60-row "season" population** (real 29 feature names, random
values) - proves `build_standardization_for_population -> project() ->
score_actual_vs_predicted` runs cleanly start to finish through the exact
same code path the real script will use, without touching any real
2026-27 data.

**Smoke-tested the actual script entry point**, not just its pure
functions: ran `review_2026_predictions.py` for real against the current
(empty) database - correctly detected zero 2026-27 rows and printed a
clear, non-error "nothing to review yet" message rather than crashing.
This is exactly the intended behavior when someone runs this script
between now and the 2026-27 season - proof the "run later" half of "write
now, run later" will actually work.

**Added `games_in_season_2026_27: 82` to `config.yaml`** (the pro-ration
knob this script reads) - caught that this invalidated the Step 4 freeze
record's `config_hash` (predictions.csv itself unchanged, only an
unrelated new config field), so refroze with `--refreeze` and a disclosed
reason rather than leaving a stale hash reference. All 9/9 new tests pass.

Step 5 complete. Starting Step 6 (final reports).

---

## 2026-08-14 — Step 6: final reports, tests - Phase 6 and the project complete

Wrote `reports/phase6_deployment_report.md` (sections A-H: refit
diagnostics, the three predictions with reasoning, comps tables, the
saturation analysis with its honest n=1 evidentiary caveat, the freeze
record, the review-script description, obstacles, open items) and
`reports/project_summary.md` (the terminal, standalone 2-3 page document
for a reader who has never seen this project - the problem, the two-
archetype-space approach, the one-shot validation result including the
T4 four-fold-beat headline, honest limits, and what to check once the
2026-27 season is real).

Wrote `tests/test_phase6.py` (12 tests: prediction/college-recipe simplex
validity, the freeze guard's refusal-without-flag behavior tested against
the real deployed guard function, deployment preprocessing completeness -
every continuous column has real mu/sd, re-derived by hand from the saved
file and diffed against `variant_b`'s actual output rather than just
checking the keys exist - the frozen-transform row-independence guard,
comps provenance against the real 273-anchor set, and a **regression
guard against the exact "hardcoded True check" bug found in Phase 5's own
audit script**).

That last guard needed a real fix mid-write: a first-draft regex-based
version undercounted the real 14 `check()` calls in
`phase5_step0_audit.py` (12 found, not 14) because some call arguments
span multiple lines, which a single-line regex can't see. Rewrote it
using Python's own `ast` module to parse the real call-argument nodes
instead of pattern-matching text - robust to formatting, and precise
about what "a literal boolean" actually means (an `ast.Constant` node,
not just any token that happens to read "True").

**Full suite: 105/105 passing** (84 from Phases 1-5, 9 from the review-
script's synthetic tests, 12 new from Phase 6). Holdout sha256 re-checked
one final time against the Step 0 manifest from Phase 5 - still an exact
match; Phase 6 never touched it either, consistent with the phase's own
"no 2026-rookie predictions [would need] the holdout" framing (they
never did - the holdout and the rookie predictions are entirely separate
data).

**Phase 6 complete. The NCAA→NBA rookie archetype translation pipeline is
complete.** Three real predictions, frozen before the season that will
judge them, with a pre-committed and pre-tested script ready to grade
them honestly when the data exists. `reports/project_summary.md` is the
terminal document.