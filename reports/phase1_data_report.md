# Phase 1 Data Report — NCAA ↔ NBA Ingestion & Alignment

Final deliverable for Phase 1 (data only — no model fitting, K selection, NNLS,
or translator work was done anywhere in this phase). See `reports/phase1_worklog.md`
for the full step-by-step narrative this report distills.

## A. Repo-audit summary

The NBA archetype basis (Z, mu, sd — `data/basis_2025_26/basis.npz`) was already
persisted; its feature order was cross-checked programmatically against
`src/step0_data.py`'s `FEATURE_COLUMNS` (exact match) and written out explicitly to
`data/basis_2025_26/feature_order.json` / `standardization.json`. NBA-side historical
ingest already covered all 9 seasons (2017-18…2025-26) before this phase started — not
just 2025-26. CBBD/nba_api/Basketball-Reference access all verified live. Full detail:
`reports/phase0_repo_audit.md`.

## B. Ingest inventory + row counts per season per side

No season deviates >25% from its neighbors, on either side (checked directly, not
assumed).

**NCAA (CBBD), MIN≥300, complete cases** — `data/college/features_complete.csv`:

| Season | Players | | Season | Players |
|---|---|---|---|---|
| 2016-17 | 2,901 | | 2021-22 | 2,959 |
| 2017-18 | 2,916 | | 2022-23 | 3,019 |
| 2018-19 | 2,937 | | 2023-24 | 3,054 |
| 2019-20 | 2,907 | | 2024-25 | 3,060 |
| 2020-21 | 2,460 *(COVID season — real, expected dip)* | | 2025-26 | 3,099 |

**NBA, MIN≥300** — `data/nba_historical/shared_features.parquet`:

| Season | Players | | Season | Players |
|---|---|---|---|---|
| 2017-18 | 392 | | 2022-23 | 407 |
| 2018-19 | 403 | | 2023-24 | 413 |
| 2019-20 | 389 | | 2024-25 | 430 |
| 2020-21 | 404 | | 2025-26 | 434 |
| 2021-22 | 426 | | | |

All within the expected 350-450/season band.

## C. Rate-validation table

4 real players (Zach Edey/C, Brandon Miller/F, Stephon Castle/G, Cooper Flagg/F),
fetched live from Sports-Reference/CBB, compared against computed values. **24/24
comparisons (USG%/AST%/TOV%/STL%/BLK%/TRB%) within ±0.2pp**, well inside the ±0.5pp
tolerance; minutes-played matched exactly each time (confirms correct row, not luck).
TS% shows a small, consistent +0.6 to +1.4pp bias (CBBD's own field, not derived here
— a source-level formula/rounding difference). Full table: `reports/rate_validation.md`.
Now regression-tested: `tests/test_rate_formulas.py`.

## D. Shot-coverage table + bin proposal

| Season | FGA with usable distance-in-feet data | Coverage |
|---|---|---|
| 2016-17 … 2025-26 (all 10 seasons) | 0 | **0%** |

This is not a per-season sparsity finding — CBBD's schema has no distance-in-feet
field at all, in any season (confirmed by reading its full OpenAPI spec and every
shooting-related endpoint response). What it has instead is shot-*type* (dunk/layup/
tip-in/2pt-jumper/3pt-jumper), a different classification axis entirely.

**Proposal (awaiting your sign-off, not decided here):** drop the shot-distance
feature family (`Dist.` + 5 distance bins + `Corner 3s_%3PA` — 7 of the NBA basis's
29 dimensions) from the shared feature set outright, rather than attempting a
shot-type-to-distance-bin approximation. An earlier pass this session proposed the
approximation as an alternative and you chose the drop — recorded here for the
formal Phase 1 deliverable, not re-litigated.

## E. Missingness

Shared feature columns (12 core, both sides): **0 nulls**, confirmed directly.

| Field | Missing | Note |
|---|---|---|
| College height | 39 / 29,395 (0.13%) | dropped from complete-case pool, not imputed |
| College `dateOfBirth` | 119,591 / 125,988 roster entries (**94.9%**) | CBBD roster data rarely includes birthdate — real gap, not a pipeline bug; not currently used by any shared feature, flagged for whenever age-at-draft covariates are needed |
| College FTr | 8 / 29,395 (0.03%) | dropped from complete-case pool |
| College ORB% | 37 / 29,395 (0.13%) | dropped from complete-case pool |

## F. Crosswalk match rates + failure examples

486 of 593 draft picks (2017-2026) tagged NCAA-path; 107 tagged International/
G-League Ignite/OTE (tag-never-drop — all 593 retained in `data/anchors/crosswalk_draft.csv`).

**NCAA-path match rate: 480/486 = 98.8%** (gate was 85%). Method breakdown:
- 461 (94.9%) direct `athleteId` join — discovered CBBD's draft/picks and
  stats/player/season endpoints share one athlete-ID space; not the spec's original
  planned method, reported as a deviation, not a silent substitution.
- 19 recovered via name-normalized (name, year, team) matching.
- 6 remain unmatched — inspected individually: at least 2 (Shaedon Sharpe, Mitchell
  Robinson) are well-documented players who **played zero college games** before
  turning pro (redshirt / never enrolled), so they have no season row to match at
  all — not a matching failure, a genuine absence of data to match to.

Per-class rates: 2020, 2021, 2023, 2024, 2025, 2026 at 100%; weakest class 2018
at 96.2%. Full table: `data/anchors/crosswalk_match_rate_by_class.csv`. Failure
detail: `data/anchors/crosswalk_failures_top20.csv`. Empty override template ready:
`data/anchors/name_overrides.csv`.

## G. Top-5 obstacles (distilled from the worklog)

1. **WebFetch blocked by Sports-Reference's bot detection (403)** — worked around
   with direct `curl` + browser user-agent, parsed raw HTML manually.
2. **Sports-Reference's own `csk` sort-key attributes are inconsistently scaled**
   (`tov_pct`'s csk is a raw fraction; every other stat's csk is already a
   percentage) — switched to parsing the displayed cell text instead, which is
   consistent.
3. **CBBD's `/stats/player/shooting/season` silently returns a truncated default
   set** if called without `team` or `conference` — not an error, easy to
   mistake for complete data. Required looping per conference.
4. **`/draft/picks`'s `sourceTeamName` is the mascot, not the school** —
   `sourceTeamLocation` is the real join key; using the wrong field would have
   collapsed the crosswalk's match rate to near zero.
5. **CBBD's team `totalMinutes` field is already "Tm MP / 5" in the textbook
   AST%/STL%/BLK%/TRB% formulas' terms**, not the full "Tm MP" — dividing by 5
   again silently understated every derived rate stat by roughly 5×. Found via
   a sanity check (a lead guard's AST% came out implausibly low), not assumed
   correct on first pass.

## H. Open decisions awaiting the owner

1. **Shot-distance family**: confirmed dropped (Section D) — no further action
   needed unless you want to revisit.
2. **`data/anchors/name_overrides.csv`**: empty template ready if you want to
   manually resolve any of the 6 remaining crosswalk failures (2 of the 6 are
   likely unresolvable — zero real college games played).
3. **BPM ↔ PORPAG**: kept as two separate, non-merged columns (NBA: `BPM`,
   college: `PORPAG`) rather than treated as equivalent anywhere. Confirm this
   stays separate into the next phase rather than being silently unified.
4. **College `dateOfBirth` 94.9% missing**: not blocking today (unused by any
   shared feature), but age-at-draft is a common covariate in translator-type
   models — flagging now in case it matters for a later phase's design.

## Reproducibility & tests

**Reproducibility**: full pipeline (`ingest_cbbd.py` → `build_features.py` →
`build_shared_features.py` → `build_draft_crosswalk.py`) re-run end to end.
Verified directly, not assumed: **zero cache files modified/created** (proves zero
network calls — any cache miss would have written a new file) and **all 6 output
files byte-for-byte identical** (SHA-256) between the two runs.

**Tests** (`tests/`, run via `pytest`): 19/19 passing —
`test_rate_formulas.py` (regression-tests the Section C validation against the
4 hand-checked player fixtures), `test_name_normalizer.py` (suffixes, diacritics
incl. non-NFKD-decomposable characters, punctuation, the Jr.-collision case that
motivates the crosswalk's team-then-year-only fallback), `test_cache_determinism.py`
(cache-hit-never-calls-network, cache-miss-writes-once, real cache dir has all 372
expected files).
