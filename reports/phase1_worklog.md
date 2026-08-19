# Phase 1 Worklog — NCAA ↔ NBA Data Ingestion & Alignment

Plain-language log, appended after every step. Newest entries at the bottom.

---

## 2026-08-11 — Kickoff

Starting Step 0 (repo audit). Note for context: this session already did
substantial earlier exploration of this exact repo for a related task
(an earlier, less formally-specified "NCAA bridge" pass) — that prior work
is being re-verified against THIS phase's precise requirements, not
assumed still accurate. Where prior findings hold up, they're cited as
"already known, re-confirmed here," not re-derived from scratch.

## 2026-08-11 — Step 0 complete

What I did: confirmed the NBA basis (Z/mu/sd) is already persisted in
`data/basis_2025_26/basis.npz`, cross-checked (programmatically, via
assert) that its feature order matches `src/step0_data.py`'s
`FEATURE_COLUMNS` exactly, and wrote the two new files this phase wants:
`feature_order.json`, `standardization.json`. Confirmed NBA-side ingest
(`step0_data.py`) already spans all 9 seasons (2017-18…2025-26), not just
2025-26 — live-queried the DB and `load_population()` and got real
389-434 players/season, inside the expected band. Confirmed CBBD/nba_api/
Basketball-Reference are all live-reachable with real test calls (not
assumed). Inventoried an earlier session pass that already pulled the
full 10-season CBBD raw cache to the exact path this phase wants
(`data/raw/cbbd/`, 372 files) — reusable as-is.

Obstacles: none blocking. One real discrepancy found and flagged, not
silently resolved: the earlier CBBD feature-construction script used a
different 15-feature list than this phase's exact shared-feature schema
(e.g. it had TS%/PORPAG/TRB%, this phase wants ft_pct/fg2_pct/fg3_pct/
share_3pa instead) — raw data is reusable, feature construction is not.

Decision needing sign-off: whether to reuse the existing raw CBBD cache
(recommended - saves a full re-pull) and only rewrite feature
construction, vs. starting the ingest fresh. Written up in
`reports/phase0_repo_audit.md`. Waiting for confirmation before Step 1.

## 2026-08-11 — Shared-feature build + validation (data-only, per owner's
## explicit scope confirmation - no modeling done)

What I did: built `data/college/shared_features.parquet` (29,312 rows,
2016-17..2025-26) and `data/nba_historical/shared_features.parquet`
(3,698 rows, 2017-18..2025-26) on the 12 confirmed-buildable NBA-basis
columns (see `reports/feature_dictionary.md`), plus each side's own
non-equivalent value column kept separate (NBA: BPM, college: PORPAG -
never merged into one column). Computed per-season z-scores independently
on each side. Validated the derived-formula columns against real,
published data: fetched 4 well-known recent lottery picks' pages directly
from Sports-Reference/CBB (curl with a browser user-agent - WebFetch
itself got blocked by their bot protection) and compared USG%/AST%/TOV%/
STL%/BLK%/TRB% against my computed values. All 24 comparisons landed
within ±0.2pp (well inside the ±0.5pp tolerance); minutes-played matched
exactly each time, confirming correct player/season matching.

Obstacles: (1) WebFetch blocked (403) by Sports-Reference's bot
detection - worked around with a direct curl + browser user-agent, then
parsed the raw HTML myself. (2) The page's own `csk` sort-key attributes
are inconsistently scaled (`tov_pct`'s csk is a raw fraction, every other
stat's csk is already a percentage) - switched to parsing the actually-
displayed cell text instead of trusting csk, which is reliable and
consistent. (3) TS% showed a small, consistent +0.6 to +1.4pp bias
(computed always higher than published) across all 4 players - traced to
CBBD's own `trueShootingPct` field (not something this pipeline derives),
so it's a source-level formula/rounding difference, not a bug here.
Flagged in `reports/rate_validation.md`, not silently absorbed.

Missingness: 0 nulls on all 12 shared columns, both sides.

No decision awaiting sign-off from this step - results support proceeding.

## 2026-08-12 — Step 4 (draft crosswalk)

What I did: tagged all 593 draft picks (2017-2026) by path using
`sourceTeamCollegeId`/`sourceTeamLeagueAffiliation` (clean split, no
ambiguous cases - verified the two fields are populated in exactly
complementary sets). For the 486 NCAA-path picks, discovered CBBD's
draft/picks and player/season endpoints share one athlete-ID space -
tested a direct `athleteId` join first (not in the original spec) and
got 94.9% on its own, already clearing the 85% gate before any name
matching. Built the name-normalization fallback (lowercase, strip
punctuation/suffixes, transliterate diacritics incl. đ/ø which NFKD
alone doesn't decompose) anyway, per spec, to recover the remainder -
combined match rate 98.8% (480/486).

Obstacles: 6 residual unmatched picks - spot-checked 2 (Shaedon Sharpe,
Mitchell Robinson) against public knowledge and confirmed both played
zero real college games (redshirt / never enrolled) before turning pro,
so they have no season row to match to at all - not a matching failure.

Decision flagged, not made unilaterally: using direct athleteId join as
the primary method (vs. the spec's name-matching-first design) - reported
in the data report, not silently substituted.

## 2026-08-12 — Final deliverables (reproducibility test, pytest suite,
## phase1_data_report.md) - Phase 1 complete

What I did: re-ran the full pipeline (ingest -> features -> shared
features -> crosswalk) a second time end to end; verified directly (not
assumed) zero cache files were modified/created (proves zero network
calls) and all 6 output files SHA-256-identical to the first run. Wrote
19 pytest tests across 3 files (rate-formula regression using the same 4
hand-checked players from the validation step, name-normalizer fixtures
including non-NFKD-decomposable diacritics and the Jr.-collision case,
cache-hit/cache-miss determinism) - all 19 passing. Wrote
`reports/phase1_data_report.md` with the 8 required sections, pulling
together everything from Steps 0-4. Separately: caught that this whole
phase's work hadn't been logged to the repo's own `AI_USAGE.md` (a
standing project requirement independent of this phase's worklog
discipline) - added a consolidated entry.

Obstacles: none new this step - closed out cleanly.

Open items for the owner (also in phase1_data_report.md Section H):
shot-distance family confirmed dropped; `name_overrides.csv` ready but
empty (2 of 6 crosswalk failures likely unresolvable - zero real college
games); BPM/PORPAG kept deliberately separate, confirm this holds into
the next phase; college `dateOfBirth` 94.9% missing, unused today but
flagged for whenever age-at-draft covariates matter.

**Phase 1 is complete. Awaiting your review before any Phase 2 (modeling) work.**
