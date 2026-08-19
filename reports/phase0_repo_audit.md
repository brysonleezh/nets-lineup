# Phase 0 — Repo Audit

## 1. NBA basis (Z) — found, persisted, cross-validated

`data/basis_2025_26/basis.npz`: `basis` (Z, shape 8×29), `mu` (29,), `sd` (29,) — already
saved, not recomputed. Feature order confirmed **programmatically identical** between
`basis_meta.json`'s `feature_columns` and `src/step0_data.py`'s `FEATURE_COLUMNS` (asserted
in code, not assumed). Written to the two files this phase requires:
`data/basis_2025_26/feature_order.json`, `data/basis_2025_26/standardization.json`.

The 29-feature order (for reference): `PTS_PER_100, PLAYER_HEIGHT_INCHES, TS%, USG%, AST%,
TOV%, STL%, BLK%, TRB%, FTr, BPM, Dist., [5× shot-distance-by-feet %], Corner 3s_%3PA,
% of FG Ast'd_2P, % of FG Ast'd_3P, [9× PLAYTYPE_*]`.

## 2. NBA-side ingest — already parameterized, already covers the full window

`src/step0_data.py`'s `SEASONS` list already spans **2017-18 → 2025-26 (9 seasons)** —
not hardcoded to 2025-26. Sources: NBA.com box/advanced (via `nba_api`), Basketball-
Reference advanced/shooting (shot-distance bins + %-assisted), NBA.com Synergy play-type
shares. Live-queried `data/nets_synergy.db`: all 9 seasons have real, populated rows in
every relevant table. `load_population(min_threshold=300)` returns **389–434 players/season**
across all 9 seasons — inside your expected 350–450 band. **Historical NBA ingest (Step 2)
is functionally already done**, not just feasible.

One gap: only 3 of 9 seasons (2023-24/2024-25/2025-26) have been *projected into archetype
recipes* — the other 6 have raw features but no recipe CSV. Irrelevant to this phase (data
only), noted for later.

## 3. CBBD side — a real head start, but feature list must be reconciled

An earlier, less formally-specified pass this session already: pulled full raw CBBD data
for **all 10 seasons (2016-17…2025-26), cached under `data/raw/cbbd/`** (372 files, 175MB —
same path this spec asks for) — player-season stats, team-season stats (with a full
`opponentStats` mirror, needed for AST%/STL%/BLK%/TRB% denominators), rosters (height),
shooting-by-conference, full draft-pick history. Confirmed live and worth carrying
forward as known gotchas:
- `/stats/player/shooting/season` **requires** `team` or `conference` — omitting both
  silently returns a truncated default set, not an error.
- `/draft/picks`'s `sourceTeamName` is the **mascot** ("Blue Devils"), not the school —
  `sourceTeamLocation` ("Duke") is the real join key for the Step 4 crosswalk.
- CBBD has **no shot-distance-by-feet data at all**, in any season — only a shot-*type*
  breakdown (dunk/layup/tip-in/2pt-jumper/3pt-jumper). This isn't a coverage/sparsity
  question Step 3's audit format assumes — it's a schema absence. Recommend the shot-
  distance audit in Step 3 be reframed around this from the start rather than re-discovered.

**Reconciliation needed, not yet done:** that earlier pass built a 15-feature table using
different names/coverage (e.g. included `TS%`/`PORPAG`/`TRB%`/assisted-%, omitted
`ft_pct`/`fg2_pct`/`fg3_pct`/`share_3pa`) than this phase's exact 15-feature shared list.
The raw CBBD pull is fully reusable; the feature-construction script is not, as-is.

## 4. External access — all verified live, not assumed

`CBBD_API_KEY`: real call succeeded (10,019 rows). `nba_api`: real call succeeded (569
rows, `LeagueDashPlayerStats`). Basketball-Reference: HTTP 200.

## Proposed plan for Step 1+

Reuse the existing raw CBBD cache as-is (no re-pull needed). Rewrite feature construction
to match this phase's exact shared schema. NBA-side historical ingest is already complete —
Step 2 becomes "verify + reformat to `data/nba_historical/`," not a new pull. Estimated
volume: no new CBBD API calls; NBA-side reformatting is local/fast.

**Awaiting your confirmation before Step 1.**
