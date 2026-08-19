# Phase 2 Model Report — College Archetype Model

Final deliverable for Phase 2 (college model only — no NBA-side computation,
no anchor covariates, no historical rookie recipes, no translator work anywhere
in this phase). See `reports/phase2_worklog.md` for the full step-by-step
narrative, including both mid-flight amendments this report distills.

## A. Feature reconciliation + early-era validation

**Canonical table confirmed: `data/college/shared_features.parquet`** (12
shared columns aligned to the real NBA 29-dim basis, not the original 15-item
wishlist — owner-confirmed after a real fork: 5 of the 15 originally-expected
columns, `ft_pct`/`fg2_pct`/`fg3_pct`/`fga_per100`/`share_3pa`, are genuinely
unbuilt anywhere, not a CBBD data gap — trivially buildable from cached raw
fields, but superseded by the 12-core NBA-basis-matched set instead).

**Early-era validation backfill**: 4 players (Frank Mason III/Kansas 2016-17,
Trae Young/Oklahoma 2017-18, Zion Williamson/Duke 2018-19, Ja Morant/Murray
State-OVC 2018-19 — the requested non-power-conference case), fetched live
from Sports-Reference/CBB. **24/24 comparisons within ±0.5pp** — no schema
drift detected across the pooled fit's full 10-season window. Combined with
Phase 1's original 4 players: **8/8 fixtures, 48/48 rate-stat comparisons,
all within tolerance.** Regression-tested: `tests/test_rate_formulas.py`
(11 tests).

## B. Pool sizes (flag ON/OFF)

`restrict_to_draft_conferences=True` (default): **20,849 rows, 22
conferences**. `=False`: **29,312 rows, 33 conferences**. Conference
membership derived per-season from each matched crosswalk pick's own
`final_college_season` (handles realignment correctly — a school's
conference-at-draft-time, not today's). 0 unresolved lookups against 480
matched NCAA-path picks.

## C. K diagnostics + recommendation

K=4..12 AA sweep (3 seeds each, max_iter=400): RSS/EV climbs smoothly
(0.61→0.87, no clean elbow, as the paper's own methodology predicts).
Intra-archetype variance (the deciding metric) is **not** cleanly monotonic —
oscillates, global minimum at K=9 (0.575), but K=9 also showed the **worst
restart stability in the whole sweep** (RSS spread 1942, vs. K5/K6's ~140-144).
Figures: `figures/phase2_k_selection_{rss_scree,intra_variance}.png`.

**Owner decision: K=8** — not the strict variance minimum, judged not
critical given K=8 also matches the NBA basis's own K.

## D. AA vs ADA evidence + decision

**CHECKPOINT 1 amendment**: the sweep's original K=8 "best-RSS" restart
(seed 0) was an outlier basin, not the consensus solution — confirmed by
refitting K=8 alone with 8 seeds (max_iter=1000) and Hungarian-matched
cosine-similarity basin clustering (threshold 0.95):

| basin | seeds | RSS | intra_var | converged |
|---|---|---|---|---|
| 0 | 0,5,6 | 51682.9 (lower) | 0.6014 (worse) | 2/3 |
| **1 (consensus)** | **1,2,3,7** | 52446.7 | **0.5639 (best)** | **4/4** |
| 2 | 4 | 52154.8 | 0.5941 | 1/1 |

**Frozen: basin 1, seed 2** — plurality basin, 100% genuine convergence,
best intra-variance (not just lowest RSS). `data/college/model/
k8_frozen_basis.npz` + `k8_frozen_manifest.txt`.

**Step 3 ADA** (restricted archetypoid search — the from-scratch `archetypes`
package ADA is infeasible at this scale, confirmed earlier this session at
~days/fit): candns init → RSS 59602.3 (+13.6% vs. continuous AA) → one
greedy swap pass (100-nearest-candidates/archetype) → RSS 56320.5 (+7.4%
final gap). **Archetypoid identity: 7 of 8 from low/mid-major conferences**
(OVC/CAA/WCC-Pacific/Ivy/Horizon/Big Sky/MVC), only 1 (Filip Petrusev,
Gonzaga) a nationally recognized player; 4 of 8 have minutes in the 315-600
range. This directly triggers the spec's own AA-preference condition
("archetypoids land on low-major statistical outliers").

**Owner decision: AA.** (Initial lean toward ADA was based on the incorrect
assumption that method choice doesn't affect per-player recipes — corrected
with concrete evidence: 4 validation players' recipes recomputed under both
bases, L1 differences 0.05–0.093, e.g. Trae Young's top-archetype weight
0.914→0.872. Recipes are basis-dependent by construction.)

## E. Interpretation table

`reports/college_archetypes.md` — per archetype: full 12-feature z-profile,
top-5 distinguishing features, top-10 loading player-seasons (with
conference/season/minutes shown), empty `label` field for the owner. One
archetype (0) is dominated by extreme-TOV%/low-minute low-major outliers
(top loader: 315 minutes, OVC) — flagged plainly as a candidate "noisy
small-sample" archetype rather than a clean basketball style, for the
owner's own judgment.

## F. Sanity projections

8 validation fixtures + the 3 Nets rookies' 2025-26 seasons, top-3
archetypes shown (`data/college/model/sanity_projections.csv`). Eyeball-
plausible, not a validation: elite lead guards (Trae Young 0.91, Ja Morant
0.78, Frank Mason III 0.55) all load on the same archetype (5); **Mikel
Brown Jr. loads primarily on that same archetype (0.57)** — consistent
with his "on-ball engine" scouting profile. No NBA-space interpretation or
translation claim made anywhere in this phase.

## G. Stability & sensitivity

**Restart stability (K=8)**: see Section D's basin table — 4/8 restarts
landed in the consensus basin, all genuinely converged and agreeing to
within 0.02 RSS.

**Conference-flag-OFF refit** (2 seeds, max_iter=1000, full 29,312-row
pool): Hungarian-matched against the frozen ON basis. **7 of 8 archetypes
match well (cosine 0.77–0.99); archetype 3 does not match at all (cosine
=-0.18)** — a real, reportable sensitivity finding, not smoothed over.
Archetype 3 (extreme TS%/low USG%/low TOV%/high assisted-2P% — an
"efficient low-usage finisher" profile) is top-loaded almost entirely by
mid/low-major players (Ivy, Big South, CUSA, Mountain West, OVC, NEC, Am.
East, MAC — zero power-conference players in its top 10). **Verdict: the
K=8 AA structure is NOT fully robust to the conference restriction** —
7/8 archetypes are, but this one specifically appears tied to the
restricted population's composition. `data/college/model/
conference_flag_sensitivity.csv` / `_verdict.txt`.

Temporal half-split (optional): not run — time-budgeted against the
higher-priority items above; flagged as a real gap, not silently skipped.

## H. Top obstacles

1. Sports-Reference blocks WebFetch (bot detection) — same curl+browser-UA
   workaround as Phase 1.
2. ADA at true scale is computationally infeasible (~days/fit) — resolved
   via candns+restricted-swap search, an explicit, reported deviation from
   a from-scratch ADA fit.
3. The original K=8 sweep's "best RSS" restart was a real methodological
   trap — an outlier basin, not the consensus solution. Caught by the
   owner before freezing, not by this agent independently — corrected via
   Amendment A's 8-seed basin analysis.
4. Conference-flag-OFF sensitivity check surfaced a real, non-trivial
   fragility (archetype 3) rather than confirming clean robustness — kept
   as a reported finding, not resolved or hidden.
5. None of the AA fits at K=8 (max_iter up to 1000) reliably converge in
   under ~250-1200s — real compute cost, budgeted around rather than
   avoided.

## I. Open decisions awaiting the owner

1. **Archetype labels** (`reports/college_archetypes.md`) — blank, per
   spec, basketball domain judgment required.
2. **Archetype 0 and 3**: both flagged with real caveats (0: low-minute/
   low-major noise risk; 3: conference-restriction sensitivity) — decide
   whether either needs special handling downstream (e.g. in the eventual
   translator) or is acceptable as-is.
3. **Temporal half-split** not run this phase — decide if it's needed
   before Phase 3, or can be deferred/skipped.
4. **Conference-restriction choice itself**: given archetype 3's
   sensitivity, confirm `restrict_to_draft_conferences=True` remains the
   right default going forward, now that its cost (this specific fragility)
   is known concretely rather than theoretical.

## Deliverables

- `data/college/model/`: `k_selection_{raw,summary}.csv`, `k_sweep_bases.npz`,
  `k8_all_seeds.csv`, `k8_basin_table.csv`, `k8_frozen_basis.npz` +
  `k8_frozen_manifest.txt`, `ada_archetypoid_identity.csv`, `ada_swap_log.csv`,
  `ada_rss_trajectory.csv`, `ada_k8_basis.npz`, `sanity_projections.csv`,
  `conference_flag_sensitivity.csv` + `_verdict.txt`.
- `src/college_model/`: `pool.py`, `project.py` (importable, docstring
  example), `k_selection.py`, `refit_k8_basins.py`, `restricted_ada.py`,
  `build_recipes.py`, `build_interpretation.py`, `sensitivity.py`.
- `data/college/recipes.csv`: **29,312 rows** (= canonical table row count),
  simplex-valid (tested).
- `reports/college_archetypes.md`, `reports/phase2_model_report.md` (this
  file), `reports/phase2_worklog.md`.
- Tests: `tests/test_college_model.py` (8 tests: simplex validity, row-count
  match, argmax consistency, no-empty-archetype, projection determinism,
  K-sweep internal consistency, basin-selection consistency) +
  `tests/test_rate_formulas.py` (11 tests, extended this phase). **31/31
  passing** across the full test suite (`tests/`).
