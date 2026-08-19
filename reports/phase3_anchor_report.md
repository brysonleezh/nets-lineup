# Phase 3 Anchor Report — Anchor Construction & Historical Rookie Recipes

Final deliverable for Phase 3 (anchor table construction only — no translator
fitting, no cross-validation, no baseline, no model fitting of any kind
anywhere in this phase; the only model-adjacent computation is NNLS
projection onto the two already-frozen bases). See `reports/phase3_worklog.md`
for the full step-by-step narrative this report distills.

## A. Standardization summary

Per-season NBA populations (`load_population(min_threshold=300)`, one season
isolated at a time):

| season | n |
|---|---|
| 2017-18 | 392 |
| 2018-19 | 403 |
| 2019-20 | 389 |
| 2020-21 | 404 |
| 2021-22 | 426 |
| 2022-23 | 407 |
| 2023-24 | 413 |
| 2024-25 | 430 |
| 2025-26 | 433 (reused verbatim from the frozen basis, not recomputed — see below) |

**2025-26 hard assert result: initially FAILED, root-caused, resolved by
owner decision — not silently patched.** A from-scratch recompute of the
2025-26 population returned 434 rows, one more than the 433 the frozen
`basis.npz`/`recipes.csv` were built on (max mu diff 1.67e-2, max sd diff
7.18e-3 — both far outside the 1e-9 tolerance). Diffed player IDs against
`recipes.csv` (frozen at the same timestamp as `basis.npz`, so a true
snapshot of the fit population) and found exactly one row added, none
removed: **Ronald Holland II** (DET, ~1550 minutes — not a threshold-
crossing case; his row must have been incomplete at fit time and was
later backfilled upstream). Ruled out query non-determinism first (two
independent fresh loads gave identical results; the underlying db's mtime
predates the basis build). Presented both resolution paths to the owner;
**decision: reuse the frozen `standardization.json` verbatim for 2025-26**
to stay bit-identical with `basis.npz`/`recipes.csv`, which the rest of the
app already depends on — recomputing and accepting the drift would have
broken that continuity for a 1-player, sub-2%-of-population difference. The
other 8 historical seasons have no frozen counterpart and were computed
fresh with no such issue. A guard re-checks this specific drift shape on
every future run and would raise (not silently reuse) if a larger or
differently-shaped divergence appears later.

## B. Attrition funnel

Per-class funnel (drafted → NCAA-path → college-matched → features present
→ gap_years==0 → included), classes 2017–2025:

| draft_year | drafted | ncaa_path | college_matched | features_present | gap_ok | included |
|---|---|---|---|---|---|---|
| 2017 | 60 | 50 | 50 | 50 | 50 | 31 |
| 2018 | 60 | 52 | 51 | 49 | 48 | 29 |
| 2019 | 60 | 52 | 52 | 50 | 49 | 29 |
| 2020 | 60 | 48 | 48 | 47 | 47 | 25 |
| 2021 | 60 | 50 | 50 | 49 | 49 | 36 |
| 2022 | 58 | 44 | 43 | 43 | 43 | 27 |
| 2023 | 58 | 46 | 46 | 46 | 46 | 27 |
| 2024 | 58 | 43 | 43 | 43 | 43 | 33 |
| 2025 | 59 | 46 | 46 | 45 | 45 | 36 |
| **Total** | **533** | **431** | **429** | **422** | **420** | **273** |

**All gates clear**: total included = 273 (≥120); 2025-class included = 36
(≥20); `college_below_min` count = 7 (≤15); NBA ID join rate = 98.4% (≥95%);
2025-26 recipe-consistency assert passes (see Section E).

The `college_below_min` = 7 cases (checked by hand, all basketball-
explicable, not a data problem): James Wiseman, Michael Porter Jr., Darius
Garland, Bol Bol, Jarred Vanderbilt, Jalen Johnson, Cedric Coward — all
injury/suspension/opt-out-shortened final college seasons.

COVID footnote: 0 of the 2019 class landed in the 250–299 minute near-miss
band for the (not pro-rated) 2019-20 season.

## C. The six named case resolutions

| player | draft | resolution | status |
|---|---|---|---|
| Shaedon Sharpe | 2022 | Confirmed zero CBBD records anywhere under his name (redshirted at Kentucky, never played, declared for the draft). The crosswalk's tentative athlete_id (31152) also has zero records — a stale/bad candidate match, not a real record. | `no_college_data` |
| Mitchell Robinson | 2018 | Confirmed zero CBBD records anywhere (committed to Western Kentucky, never enrolled/played). | `no_college_data` |
| Goga Bitadze | 2019 | Confirmed zero CBBD records under "Bitadze" in any season — verifies the crosswalk's `path=NCAA`/`Georgia` tag was a source-data mis-tag (Georgia the country, where he actually played for Mega Bemax/Serbia, conflated with Georgia the university). Path corrected to `International (Serbia)`. | `non_ncaa_path` |
| De'Anthony Melton | 2018 | Recovered via a search of seasons draft_year−1/−2: found at USC in season 2017 (972 min). He sat out the 2017-18 season → `gap_years=1`. | `college_gap_year` |
| Dewan Hernandez | 2019 | Recovered the same way: found at Miami in season 2018 (825 min). He sat out 2018-19 → `gap_years=1`. | `college_gap_year` |
| Wesley Iwundu | 2017 | Recovered via a 1-entry alias map (CBBD lists him as "Wes Iwundu"): season 2017 at Kansas State (1063 college min). `gap_years=0`, proceeded through the normal chain and cleared the rookie-MIN gate with 1020 rookie minutes. | **`included`** — a full anchor |

## D. Covariate coverage

- **Birthdate / age_at_draft**: 100% coverage (273/273), gate is ≥95%.
  `age_at_draft` ranges 18.5–24.3 years, mean 20.9 — no implausible outliers.
- **Years in college**: validated against 12 known cases (spec named 6; 6
  more added for a firmer read):

  | name | computed | expected | match |
  |---|---|---|---|
  | Frank Mason III | 4 | 4 | ✓ |
  | Trae Young | 1 | 1 | ✓ |
  | Jalen Brunson | 3 | 3 | ✓ |
  | Zach Edey | 4 | 4 | ✓ |
  | Ja Morant | 2 | 3 | ✗ |
  | Cooper Flagg | 1 | 1 | ✓ |
  | Zion Williamson | 1 | 1 | ✓ |
  | Payton Pritchard | 4 | 4 | ✓ |
  | Anthony Edwards | 1 | 1 | ✓ |
  | RJ Barrett | 1 | 1 | ✓ |
  | Markelle Fultz | 1 | 1 | ✓ |
  | Luka Garza | 4 | 4 | ✓ |

  11/12 match (8.3% disagreement, under the 10% fallback trigger — the
  pre-2016 player_season-extension fallback was **not** invoked). The one
  mismatch, Ja Morant, was investigated directly rather than dismissed:
  both `roster_*.json` and raw `player_season_*.json` have zero records for
  him in season 2017 (his true freshman 2016-17 year), even though the
  2017 files exist and cover other players — a genuine, isolated CBBD
  coverage gap, not something the spec's proposed fallback would even fix
  (the gap is inside the pulled window, not before it). His
  `years_in_college=2` in the data should be read as a known undercount by
  one, not corrected by hand.
- **Position**: 96.3% coverage (10/273 missing; 8 of the 10 are 2017-class
  picks, suggesting a per-year gap in the raw draft-picks pull rather than
  anything player-specific). Not gated in the spec.
- **Conference / conf_tier**: 100% coverage. Proposed `conf_tier` map (owner
  to review before Phase 4 uses it beyond bookkeeping): high-major always =
  {ACC, Big Ten, Big 12, SEC, Big East}; Pac-12 counted high-major only
  through season 2024 (2023-24), since realignment gutted the conference
  for 2024-25. Among the 273 anchors: True = {ACC, Big 12, Big East, Big
  Ten, Pac-12, SEC}; False = {A-10, American, Big Sky, Big West, CUSA, MVC,
  Mountain West, OVC, Patriot, WCC}.
- **Shot-type mix (Step 4b, optional)**: 93.8% coverage. College-only
  features, no NBA-side counterpart; reported as available with no claim
  about downstream use.

## E. Sanity vignettes

Six named players, college recipe (top-3) → rookie NBA recipe (top-3),
eyeball-plausibility only — no translator or generalization claim:

| player | college top-3 (archetype: weight) | rookie NBA top-3 |
|---|---|---|
| Trae Young ('18) | 5: 0.91, 0: 0.07, 6: 0.02 | 3: 0.77, 6: 0.17, 7: 0.04 |
| Zion Williamson ('19) | 6: 0.53, 4: 0.30, 3: 0.13 | 1: 0.46, 4: 0.15, 6: 0.13 |
| Ja Morant ('19) | 5: 0.78, 0: 0.12, 6: 0.10 | 3: 0.52, 6: 0.23, 7: 0.18 |
| Zach Edey ('24) | 6: 1.00, 2: 0.00, 0: 0.00 | 1: 0.41, 2: 0.36, 4: 0.17 |
| Stephon Castle ('24) | 6: 0.26, 5: 0.19, 3: 0.17 | 3: 0.55, 7: 0.15, 2: 0.11 |
| Cooper Flagg ('25) | 6: 0.55, 5: 0.21, 4: 0.21 | 3: 0.35, 6: 0.31, 2: 0.12 |

Both primary ball-handlers (Young, Morant) map college archetype 5 → rookie
archetype 3 consistently. Zach Edey's near-pure college center profile
(archetype 6 = 1.00) spreads across NBA archetypes 1/2/4 as a rookie —
consistent with a dominant college big needing to adapt his role early in
the league. **Also serving as a correctness proof, not just a vignette**:
the 2025-class subset of these computations (Cooper Flagg among them) is
independently confirmed exact against the app's own stored recipes — see
Section B's gate note and the worklog's Step 5 entry (max diff = 0.000000
across all 36 2025-class anchors, all 8 archetypes).

## F. Obstacles

1. The 2025-26 population drift (Section A) — a real, live-data staleness
   issue in the underlying NBA database relative to the frozen Phase 2
   basis, not a bug in this phase's code. Resolved by owner decision to
   preserve continuity with the frozen artifacts rather than chase the
   live DB.
2. Rule-5's rookie-MIN check initially called `load_population()` (a full
   6-table rebuild) once per candidate row (~420 calls) — timed out past 2
   minutes. Fixed by caching the population once, split by season.
3. The years-in-college formula's one real mismatch (Ja Morant) required
   direct investigation of raw CBBD files to confirm it was a genuine,
   isolated data gap rather than a formula bug — the spec's own proposed
   fallback mechanism would not have fixed this specific case.
4. `shared_features.parquet` has one duplicate (player_id_source, season)
   key (A.J. Lawson, a mid-season-transfer double-count — pre-existing
   Phase 1 data quirk, out of scope to fix here) that broke a naive
   dictionary-style lookup on first attempt; deduped before joining.
5. Sports-Reference/CBBD access and the totalMinutes/5 bug (Phase 1) and
   the K=8 basin-selection trap (Phase 2) remain the two largest
   obstacles across the whole pipeline to date — nothing comparably large
   surfaced in Phase 3.

## G. Open decisions for the owner

1. **conf_tier map** (Section D) — the proposed high-major boolean grouping
   is a reasonable default but a basketball-domain judgment call; review
   before Phase 4 uses it for anything beyond bookkeeping.
2. **Ja Morant's years_in_college=2** — accept as a known, investigated
   undercount, or hand-correct to 3 with a documented override? Currently
   left as computed, flagged, not overridden.
3. **Position coverage gap (96.3%, mostly 2017-class)** — worth pulling
   from an alternate CBBD source, or acceptable as-is since it's not gated?
4. **Shot-type mix (Step 4b) inclusion in Phase 4** — built and available
   (93.8% coverage) but no claim made here about whether the translator
   should use it; owner/Phase-4 spec's call.
5. **Archetype labels** (`reports/college_archetypes.md`) — still blank as
   of this report, per the Phase 3 kickoff decision to skip and defer; a
   Phase 4 precondition to revisit.

## Deliverables

- `data/nba_historical/standardization_{season}.json` × 9.
- `data/anchors/anchor_ledger.csv` (593 rows — every 2017-2026 pick, exactly
  once, with a status), `data/anchors/anchor_attrition.csv` (per-class
  funnel).
- `data/anchors/anchor_covariates.csv`, `data/anchors/anchor_xside.csv`,
  `data/anchors/anchor_yside.csv` (intermediate build artifacts),
  `data/anchors/years_in_college_validation.csv`,
  `data/anchors/sanity_vignettes.csv`.
- **`data/anchors/anchors.csv`** (273 rows, 66 columns — the Phase 4
  training table): identity, college side (recipe + shared features +
  PORPAG + optional shot-type mix), covariates, Y side (rookie recipe),
  bookkeeping.
- `data/raw/nba/commonplayerinfo/` (273 cached birthdate pulls).
- `src/college_model/`: `nba_standardization.py`, `anchor_ledger.py`,
  `anchor_covariates.py`, `anchor_xside.py`, `anchor_yside.py`,
  `anchor_finalize.py`.
- `reports/phase3_anchor_report.md` (this file), `reports/phase3_worklog.md`.
- Tests: `tests/test_anchors.py` (21 tests — Y/college-recipe simplex
  validity, standardization file well-formedness × 9 seasons, per-season
  determinism, the 2025-26 hard-assert re-check, 2025-class recipe
  consistency, ledger completeness/partition/six-named-cases, zero-2026
  rows, age_at_draft fixture, years_in_college validation-file check).
  **52/52 passing across the full repo test suite** (31 from Phases 1-2 +
  21 new).
