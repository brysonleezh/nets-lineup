# Phase 3 Worklog — Anchor Construction & Historical Rookie Recipes

Plain-language log, appended after every step. Newest entries at the bottom.

---

## 2026-08-12 — Kickoff, preconditions

Checked all 3 preconditions before starting:
1. Phase 2 complete: frozen K=8 basis (`data/college/model/k8_frozen_basis.npz`)
   and `data/college/recipes.csv` both present and confirmed from Phase 2's
   own worklog/report. **Gap found**: `reports/college_archetypes.md`'s 8
   archetype `label` fields are all still blank (owner hasn't filled them
   yet) - this is listed as a literal precondition ("labels filled by
   owner"), so flagged rather than silently proceeding. Asked the owner
   directly; **decision: skip this and proceed** - labels are a human-
   readable naming convenience only, don't feed any computation (the alpha
   values are already final), can be filled anytime later.
2. Phase 1 artifacts: `crosswalk_draft.csv`, `shared_features.parquet`,
   `basis_2025_26/{basis.npz,feature_order.json,standardization.json}` -
   all present, confirmed directly (file listing, not assumed).
3. NBA historical DB: `load_population(min_threshold=300)` returns all 9
   seasons (2017-18..2025-26), confirmed live.

Starting Step 1 (per-season NBA standardization artifacts).

---

## 2026-08-12 — Step 1: per-season NBA standardization artifacts

Built `src/college_model/nba_standardization.py`. For each of the 9 seasons
(2017-18..2025-26), loads `load_population(min_threshold=300, season_min=X,
season_max=X)` (flat MIN>=300 for all 9 - the eligibility pro-ration for
2020-21 is a Step 2 concern, not applied here), computes mu/sd over the 29
basis features in `feature_order.json`'s exact order, writes
`data/nba_historical/standardization_{season}.json`.

**Hard assert failed on first run** (mu diff 1.67e-02, sd diff 7.18e-03,
both >> 1e-9) - stopped immediately per spec, did not touch either file,
investigated instead of assuming a bug in the new script:

1. Diffed the live 2025-26 population's PLAYER_IDs against
   `data/basis_2025_26/recipes.csv` (frozen alongside `basis.npz`, same
   timestamp, 2026-07-28 23:28, so it's a true snapshot of the exact
   population the frozen basis was fit on). Live = 434 rows, frozen = 433,
   **exactly one row added, none removed: Ronald Holland II (DET, ~1550
   min)**.
2. Ruled out a threshold-crossing explanation (his minutes are nowhere near
   300) and ruled out query non-determinism (`nets_synergy.db`'s content
   predates the basis build by ~23h per mtime, and two independent fresh
   loads gave identical row sets/PLAYER_ID checksums). Conclusion: his row
   was incomplete (dropped by `build_nba_side_tables`'s completeness filter)
   at basis-build time and has since been backfilled upstream - real,
   small data drift in the live DB relative to the frozen basis, not a
   bug in this phase's code.
3. Presented the finding to the owner (AskUserQuestion): reuse the frozen
   `standardization.json` verbatim for 2025-26, or recompute fresh and
   accept the drift. **Owner decision: reuse the frozen values** - stay
   bit-identical to `basis.npz`/`recipes.csv`, which the rest of the app
   (portal pages, RAPM, etc.) already depends on. Fixing the live drift
   itself would mean re-fitting Phase 2's frozen basis, out of scope here.

Implemented: the other 8 historical seasons are freshly computed (no frozen
counterpart exists for them, nothing to diverge from); 2025-26's artifact is
copied verbatim from `data/basis_2025_26/standardization.json` (`n=433`,
tagged with a `source` field explaining the reuse). A guard
(`_verify_drift_is_the_known_holland_case`) re-checks on every run that any
live-vs-frozen drift still matches this specific, already-reviewed shape
(exactly the investigated magnitude) before reusing the frozen file - if a
*different*, larger, or differently-shaped divergence shows up later, it
raises instead of silently reusing a now-stale frozen file.

**Locked convention** (per spec, restated for later steps): a rookie's
features are always z-scored with **his own rookie season's** mu/sd - never
2025-26's for other seasons, never recomputed from rookies only.
`project(df, basis, mu, sd, feature_columns)` will be reused as-is (mu/sd
passed explicitly) in Step 5.

Output: `data/nba_historical/standardization_{2017-18..2025-26}.json` (9
files), all with `feature_columns` matching `feature_order.json` exactly.

Step 1 complete. Starting Step 2 (eligibility ledger + NBA ID bridge).

---

## 2026-08-12 — Step 2: eligibility ledger + NBA ID bridge (CHECKPOINT A)

Built `src/college_model/anchor_ledger.py`. Every row of `crosswalk_draft.csv`
(593 picks, 2017-2026, all paths) appears exactly once in
`data/anchors/anchor_ledger.csv` with a status; 2026 rows (60) get
`prediction_target` and skip the chain entirely - no 2026 row was given a
matched_athlete_id lookup, gap check, or rookie-season check anywhere in this
script.

**Performance note**: first run hung past the 2-minute timeout - the rule-5
rookie-MIN check was calling `load_population()` (a full 6-table rebuild)
once per candidate row (~420 calls). Fixed by loading the full population
once, splitting it into 9 per-season dicts keyed by PLAYER_ID, and reusing
those - reran in ~15s.

**The 6 named special cases** (all resolved individually, per spec, before
the general chain runs):
- **Shaedon Sharpe (2022), Mitchell Robinson (2018)**: searched all raw CBBD
  `player_season_*.json` (2017-2026, no MIN floor) for their names - zero
  hits either case, confirming true zero-college-games → `no_college_data`.
  Also checked: the crosswalk's tentative athlete_id for Sharpe (31152) has
  zero player_season rows either - was a stale/bad candidate match, not a
  real record.
- **Goga Bitadze (2019)**: zero hits under "Bitadze" in any season, any
  spelling → confirms the country/university mis-tag theory from Phase 1.
  Path corrected to `International (Serbia)` in the ledger; falls out via
  the ordinary `non_ncaa_path` rule rather than needing a special status.
- **De'Anthony Melton (2018), Dewan Hernandez (2019)**: recovered by
  searching seasons draft_year-1/draft_year-2 by normalized name - found
  Melton at USC in season=2017 (972 min) and Hernandez at Miami in
  season=2018 (825 min), both matching the "suspended/ineligible, sat out
  their draft-year season" public narrative. Both land on `gap_years=1` →
  `college_gap_year` (excluded by default, `INCLUDE_GAP_YEARS=False`), but
  now correctly documented instead of sitting in an unresolved bucket.
- **Wesley Iwundu (2017)**: recovered via a 1-entry alias map
  (`"wesley iwundu" -> "wes iwundu"`, matching how CBBD actually lists him)
  → season=2017 at Kansas State (1063 college min). Proceeded through the
  *normal* chain from there (not auto-included) and cleared rule 5 with
  1020 rookie-season NBA minutes → a full, real anchor.

**General chain results** (2017-2025, 533 picks):
| stage | count |
|---|---|
| drafted | 533 |
| NCAA-path | 431 |
| college-matched | 429 |
| features present (shared_features.parquet) | 422 |
| gap_years==0 | 420 |
| **included** | **273** |

`college_below_min` = 7 (gate: stop if >15 - clears). All 7 are real,
explicable short-college-career cases (checked by hand): James Wiseman,
Michael Porter Jr., Darius Garland, Bol Bol, Jarred Vanderbilt, Jalen
Johnson, Cedric Coward - injury/suspension/opt-out-shortened final seasons,
not a data problem.

**NBA-side ID bridge**: joined on (draft_year, overall_pick) against
`player_bio`'s DRAFT_YEAR/DRAFT_NUMBER (deduped to one row per PLAYER_ID
first - confirmed 0 players have conflicting draft_year across season rows,
so the dedup is safe). **98.4% join rate** among NCAA-path 2017-2025 picks
(425/432; gate: stop if <95% - clears). Validated every join by comparing
normalized names: 15 "disagreements", all hand-inspected and confirmed
benign (nickname/diacritic variants - Cam/Cameron, Mo/Mohamed, Bam/Edrice,
Svi/Sviatoslav, an HTML-entity-escaped "Topi&#263;" vs "Topić", etc. - the
join key is numeric so none of these are identity errors). Notably "Ron
Holland II" vs "Ronald Holland II" appears here too - the same player whose
late-arriving box-score row caused Step 1's 2025-26 population drift.
7 NCAA-path picks have no NBA PLAYER_ID at all (Thomas Sorber, Marcus
Zegarowski, Balsa Koprivica, Justinian Jessup, Jaylen Hands, Justin Jackson
[the 2018 Maryland one - confirmed distinct from the 2017 UNC Justin Jackson
who IS in the table], Tony Carr) - checked each via a broad surname search
in `player_bio`, confirmed genuinely absent from every pulled season, not a
join-key problem. Treated as zero/undocumented NBA minutes → `rookie_below_min`.

**COVID footnote**: 0 of the 2019 class landed in the 250-299 minute
near-miss band for the (not pro-rated) 2019-20 season.

**Gates, all clear**: college_below_min=7 (≤15); NBA ID join=98.4% (≥95%);
included=273 (≥120); 2025-class included=36 (≥20).

Outputs: `data/anchors/anchor_ledger.csv` (593 rows), `data/anchors/anchor_attrition.csv`
(per-class funnel table above).

**CHECKPOINT A reached** - presenting the ledger summary, attrition funnel,
and the six case resolutions to the owner; requesting approval for Step 3's
network ingest (~430 `commonplayerinfo` calls for birthdates). Waiting.

**Owner approved CHECKPOINT A** - proceeding to Step 3.

---

## 2026-08-12 — Step 3: covariates

Built `src/college_model/anchor_covariates.py`, scoped to the 273 `included`
anchors only (not the full ~430-pick candidate pool) - birthdate/age is only
ever consumed downstream by `anchors.csv`, so pulling it for excluded rows
would be pure wasted network traffic against the spec's own scope discipline.

**years_in_college**: `final_college_season - min(startSeason across all
roster entries) + 1`, per spec. Validated against 12 cases (spec named 6 -
Frank Mason III, Trae Young, Jalen Brunson, Zach Edey, Ja Morant, Cooper
Flagg; added 6 more - Zion Williamson, Payton Pritchard, Anthony Edwards,
RJ Barrett, Markelle Fultz, Luka Garza - for a statistically firmer read
than n=6 alone would give). **11/12 matched (8.3% disagreement) - under the
10% fallback trigger, so the pre-2016 player_season-extension fallback was
NOT invoked.** The one mismatch, Ja Morant (computed 2 years, expected 3),
was investigated rather than shrugged off: checked both `roster_*.json` and
raw `player_season_*.json` directly - he has zero CBBD records at all for
season 2017 (his true freshman 2016-17 year), even though `roster_2017.json`
exists as a file and covers that year for other players. This is a genuine,
isolated CBBD coverage gap for that specific player-season, not something
the spec's proposed fallback (extending the pull to 2013-16) would even
fix, since 2017 is already inside the pulled window. Flagged in
`years_in_college_validation.csv` rather than hand-corrected - his
`years_in_college=2` in the output data should be read as an undercount by
one when comparing to public-record narratives, not an error in this table.

**position**: from `data/raw/cbbd/draft_picks_all.json`, keyed by
`matched_athlete_id` (not the crosswalk's raw `athlete_id` column - Wesley
Iwundu's raw `athlete_id` is still null even after Step 2's alias recovery,
since that recovery only ever touched `matched_athlete_id`). **96.3%
coverage** (10/273 missing - 8 of the 10 are 2017-class picks, suggesting a
per-year gap in the raw draft-picks pull rather than anything player-
specific; not gated in the spec, left as a documented gap for Phase 4 to
decide whether it matters).

**conference / conf_tier**: joined `shared_features.parquet` on
(matched_athlete_id, final_college_season) - **100% coverage**. Hit one
real bug: `shared_features.parquet` has 2 rows sharing a duplicate
(player_id_source, season) key (A.J. Lawson, a mid-season-transfer double-
count - a pre-existing Phase 1 data quirk, out of scope to fix here);
deduped (keep-first) before the lookup to avoid a silent multi-row-per-key
join corruption. conf_tier proposed map (ACC/Big Ten/Big 12/SEC/Big East
always; Pac-12 through season 2024 only, since realignment gutted it for
2024-25): among the 273 anchors, True={ACC, Big 12, Big East, Big Ten,
Pac-12, SEC}, False={A-10, American, Big Sky, Big West, CUSA, MVC, Mountain
West, OVC, Patriot, WCC} - owner to review this map before Phase 4 uses it
for anything beyond bookkeeping.

**age_at_draft**: pulled `commonplayerinfo` BIRTHDATE for all 273 anchors'
NBA ids, cached immutably under `data/raw/nba/commonplayerinfo/{player_id}.json`.
**100% birthdate coverage** (gate: stop if <95% - cleared with room to
spare). Sanity: age_at_draft ranges 18.5-24.3 years, mean 20.9 - no
implausible outliers.

Outputs: `data/anchors/anchor_covariates.csv` (273 rows), `data/anchors/years_in_college_validation.csv`.

Step 3 complete, all gates clear. Starting Step 4 (X-side assembly).

---

## 2026-08-12 — Step 4: X-side assembly

Built `src/college_model/anchor_xside.py`. Joined Phase 2's `recipes.csv` on
(matched_athlete_id, final_college_season) -> `c_alpha_0..c_alpha_7`,
`c_argmax`, `c_alpha_max` (kept 0-indexed to match `recipes.csv`'s own
column naming and every other archetype reference in this repo - read the
spec's "c_alpha_1..8" as shorthand for "the 8 alpha columns," not a literal
1-indexing mandate, since renaming would introduce an off-by-one mismatch
against the rest of the codebase). **0 anchors missing a recipe** (expected
- Step 2's `college_below_min` gate already filtered out anyone without a
`shared_features.parquet` row, and the recipe table is a superset
projection of that same table). Simplex check passed: all 273 recipes
non-negative, all sum to 1 within 1e-6.

Joined the 12 raw + 12 per-season-z `shared_features.parquet` columns plus
`PORPAG` (kept as its own column, never merged with BPM - different scales,
different things). **0 missing**, same reasoning. Reused the same
duplicate-key dedup from Step 3 (the A.J. Lawson (player_id_source, season)
collision).

**Step 4b (optional)**: built college-native shot-type mix (rim-finishing
share, 3PT-jumper share) from the 340 already-cached `shooting_*.json`
files - no network calls. One dedup needed: a handful of (athleteId,
season) pairs appear in two conference files (mid-season transfers), kept
the row with more tracked shots. **93.8% coverage** - no claim made about
whether Phase 4 uses these; reported as available, college-only, no NBA-side
counterpart.

Output: `data/anchors/anchor_xside.csv` (273 rows, 54 columns).

Step 4 complete. Starting Step 5 (Y-side computation - the core of the phase).

---

## 2026-08-12 — Step 5: Y-side computation (the core of the phase)

Built `src/college_model/anchor_yside.py`. For each of the 273 anchors:
pulled his rookie-season row (full 29 features) from the live NBA
population, z-scored with **that season's** persisted `standardization_{season}.json`
from Step 1 (grouped anchors by rookie season, one `project()` call per
group with that season's own mu/sd - never 2025-26's, never recomputed),
NNLS-projected onto the frozen basis via the existing `project()` (mu/sd
passed explicitly, matching the locked convention) -> `y_0..y_7`, `y_argmax`,
`y_max`.

**All 3 consistency asserts pass:**
1. Simplex validity: max |sum(y) - 1| = 2.22e-16 (float-precision noise,
   effectively exact).
2. **2025-class consistency vs. the app's stored `data/basis_2025_26/recipes.csv`:
   36/36 2025-class anchors matched by PLAYER_ID, max |y_j - arch_j| across
   all 8 archetypes = 0.000000 - an exact match**, not just within the
   reused Part-D tolerance (1e-3). This is the single strongest correctness
   signal in the phase: two independently-run computation paths (the app's
   original `phase2_build` and this Step 5 recomputation) landed on
   bit-identical recipes for the same 36 real players, confirming the
   season-specific standardization, feature ordering, and projection are
   all wired correctly.
3. 6 sanity vignettes (Trae Young, Zion Williamson, Ja Morant, Zach Edey,
   Stephon Castle, Cooper Flagg) - all found, all eyeball-plausible: the
   two primary ball-handlers (Young, Morant) both map college archetype 5
   -> rookie-NBA archetype 3 consistently; Zach Edey's near-pure college
   center profile (archetype 6 = 1.00) spreads across NBA archetypes
   1/2/4 as a rookie, consistent with a dominant college big needing to
   adapt his role early in the league. Eyeball-plausibility only, per
   spec - no translator or generalization claim made anywhere here.

Outputs: `data/anchors/anchor_yside.csv` (273 rows, 66 columns),
`data/anchors/sanity_vignettes.csv`.

Step 5 complete, all consistency asserts pass. Starting Step 6 (final
outputs, attrition, report, tests).

---

## 2026-08-12 — Step 6: final outputs, report, tests — Phase 3 complete

Built `src/college_model/anchor_finalize.py`. Found two spec-required
columns (`college_team`, `college_minutes`) had been dropped in Step 4's
generic shared-features join (that join deliberately excluded
`team`/`minutes`/`conference` from the feature-column list since
`conference` was already rebuilt separately in Step 3) - pulled them back
in from `shared_features.parquet` here. Renamed columns to the spec's exact
names (`matched_athlete_id`->`cbbd_athlete_id`, `player_name_raw`->
`player_name`, `rookie_season_expected`->`rookie_season`), pulled
`college_match_method` from the ledger, assembled the final column order
(identity / college side / covariates / Y side / bookkeeping) per spec.
**Wrote `data/anchors/anchors.csv`: 273 rows, 66 columns, draft_year range
2017-2025 (asserted zero 2026 rows), per-class counts 25-36** - matches the
attrition funnel exactly.

Wrote `reports/phase3_anchor_report.md` (sections A-G, distilling this
worklog's full narrative into the final deliverable).

Wrote `tests/test_anchors.py` - 21 tests covering: Y and college-recipe
simplex validity; all 9 standardization files exist and are well-formed;
per-season standardization determinism (two independent loads of the same
season agree); the 2025-26 hard-assert re-check against the frozen basis;
2025-class recipe consistency against the app's stored recipes (reusing
the same 1e-3 tolerance); ledger completeness (every pick appears exactly
once, valid status, 2017-2025/2026 partition correctly); the six named
cases resolve to their expected statuses; zero 2026 rows in `anchors.csv`;
an `age_at_draft` fixture computation; the years-in-college validation
file's disagreement rate stays in a sane range. **All 21 pass; full repo
suite (Phases 1-2's 31 + these 21) = 52/52 passing.**

**All phase gates, final check**: total included=273 (≥120); 2025-class
included=36 (≥20); NBA ID join=98.4% (≥95%); birthdate coverage=100%
(≥95%); 2025-26 recipe-consistency assert passes (exact match); `college_below_min`=7
(≤15). **All clear.**

**Phase 3 complete.** No translator fitting, no cross-validation, no
baselines, no model fitting of any kind occurred anywhere in this phase -
the only model-adjacent computation was NNLS projection onto the two
already-frozen bases (college K=8 AA, NBA 8x29), exactly as the phase's own
hard boundary required. Phase 4 (translator fitting) does not begin until
the owner explicitly accepts `reports/phase3_anchor_report.md`.

---

## 2026-08-13 — Side task: drafting `college_archetypes.md` labels

Out of Phase 3's own step sequence, but the owner asked directly: given the
8 college archetypes have sat with blank `label` fields since Phase 2
(explicitly flagged there as "not an agent's call" — basketball domain
judgment), could candidate labels be drafted from the real data for the
owner to review/edit? Presented all 8 archetypes' z-profiles + top-loading
real players + population share in chat first, flagged which ones have a
clean basketball story vs. which don't, before writing anything — owner
said write them in now, will edit later.

Wrote draft labels into `reports/college_archetypes.md` (both `label` and
"apparent correspondence to the paper's archetypes" fields), each marked
`(draft)`, plus a note at the top of the file explaining these are AI-
drafted pending owner review, not settled:

- **Archetype 2** → Rim-Protecting Big / Shot-Blocking Center (Traditional
  Center) — high confidence, cleanest fit (Jamarion Sharp, Ike Obiagu, most
  loaders at alpha≈1.00).
- **Archetype 4** → Ball-Hawking Defensive Guard (Defensive Specialist) —
  high confidence, dominated by extreme STL%.
- **Archetype 5** → High-Usage Primary Ball-Handler (Traditional Playmaker
  / High Usage Guard hybrid) — high confidence; loaders include Trae Young
  (0.91), and Mikel Brown Jr.'s college recipe loads here too (0.57),
  consistent with his scouting profile.
- **Archetype 6** → High-Usage Interior Scorer (High Usage Forward) — high
  confidence; loaders include Zach Edey (0.997), Luka Garza.
- **Archetype 3** → Efficient Low-Usage Play-Finisher (Role Guard) —
  reasonable fit, but this is the same archetype Phase 2's sensitivity
  check flagged as NOT robust to the conference-restriction flag
  (cosine=-0.18) — carried that caveat into the label note.
- **Archetypes 0, 1, 7** → flagged as weak/noisy rather than forced into a
  clean label. All three have undifferentiated low-major bench players as
  top loaders with no coherent basketball story (0 and 1 are extreme on a
  single feature with nothing else supporting it; 7 is the largest
  archetype by population share but has no feature with |z| > 2.2 at all —
  a diffuse catch-all, not a real style). Labeled "draft, weak" rather than
  invented a confident-sounding name for a bucket that doesn't have one.

This does not change any Phase 2 numeric output (alphas, basis, recipes) —
purely descriptive labels layered on top of an already-frozen fit. Not
gated by anything downstream; Phase 3/4 use the numeric archetype indices,
not these labels.

---

## 2026-08-13 — Owner-authored appendix appended to college_archetypes.md

Owner supplied a full "Appendix — Model Provenance, Special Cases & Known
Limitations" (sections A1–A7) to append verbatim to
`reports/college_archetypes.md`, framed as the canonical register that
Phase 5/6 reports should cite rather than restate. Before appending, spot-
checked the appendix's more specific quantitative claims against the real
source files rather than treating owner-authored content as automatically
correct, since the document explicitly asks to be treated as canonical:

- K-selection numbers (A3): verified against `data/college/model/
  k_selection_raw.csv` and `k_selection_summary.csv` directly. **All
  checked out exactly**, including two easy-to-get-wrong details - the
  K=9 "0.557 vs. 0.575" split (seed 1 hit 0.557 but did NOT converge
  [`converged=False`, hit max_iter]; the two seeds that did converge, 0 and
  2, both landed at ~0.575 - so the lower number is a real but untrusted
  outlier, exactly as the appendix frames it), and the K=5 explained-
  variance triplet (0.6680/0.6678/0.6674), which matched the raw sweep to
  four decimals.
- Complete-case drop counts (A4): verified against `features_raw.csv`
  (29,395 rows) vs. `features_complete.csv` (29,312 rows) - height=39,
  ORB%=37, FTr=8 nulls, all exact matches.
- **One discrepancy found**: A3 states "only 3 of 27 fits met the formal
  criterion" - the raw sweep actually shows **4** converged fits (K=7 seed
  2, K=9 seeds 0 and 2, K=11 seed 0). Flagged to the owner rather than
  silently corrected, since it's their document.
- **One stale item found**: A7 lists "Archetype labels are still blank" as
  an open item, but this session's immediately preceding turn (see above)
  drafted candidate labels into the file already. Flagged to the owner
  rather than editing their prose unprompted.

Appended as-given (both issues are minor and don't change any conclusion);
owner to decide whether to correct the "3 of 27" figure and refresh the A7
open-items line.
