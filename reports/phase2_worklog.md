# Phase 2 Worklog — College Archetype Model

Plain-language log, appended after every step. Newest entries at the bottom.

---

## 2026-08-12 — Kickoff

Starting Step 0 (pre-flight reconciliation). Phase 1 is complete and
accepted (`reports/phase1_data_report.md`). This step is diagnostic only -
no fitting, no K selection.

## 2026-08-12 — Step 0 complete, one item requires owner decision before Step 1

**0.1 Feature enumeration.** Compared the two Phase-1 tables against the
expected 15 (`height, usg_pct, ast_pct, tov_pct, orb_pct, drb_pct,
stl_pct, blk_pct, ftr, ft_pct, fg2_pct, fg3_pct, pts_per100, fga_per100,
share_3pa`):
- `features_complete.csv` (29,312 rows) has 9 of the 15 directly
  (height/usg/ast/tov/orb/drb/stl/blk/ftr/pts_per100 = 10 actually, plus
  TRB% as a bonus), plus 5 EXTRA columns not in the expected list (TS%,
  TRB%, PORPAG, %FGAst'd_2P, %FGAst'd_3P) that WERE matched against the
  real NBA 29-feature basis instead.
- `shared_features.parquet` (built specifically for the Phase 1 NBA-
  comparison exercise) drops ORB%/DRB% from that - bookkeeping, not a
  gap, they're still in features_complete.csv.
- **5 of the 15 are genuinely not yet built anywhere: `ft_pct`, `fg2_pct`,
  `fg3_pct`, `fga_per100`, `share_3pa`** - exactly the columns the spec
  itself flagged as "the likely suspects." Checked whether this is a CBBD
  data gap (like shot-distance was) or just an unbuilt column: confirmed
  live against the raw cache - `twoPointFieldGoals.pct`, 
  `threePointFieldGoals.pct`, `freeThrows.pct` are DIRECTLY present in
  every cached `player_season_*.json` file; `fga_per100`/`share_3pa` are
  trivially derivable (same pattern as the already-built `PTS_PER_100`).
  **This is not a data limitation - it's that Step 3 of Phase 1 (my own
  earlier judgment call, flagged then, not resolved until now) built a
  12-feature set matched to the REAL NBA archetype model's 29 dimensions
  instead of this original 15-feature wishlist.** Per the spec's own stop
  condition ("if any expected feature is genuinely absent... stop and
  report before fitting"), stopping here - this is a real fork, not a
  bookkeeping note, since it decides what the fit pool's dimensionality
  even is. See message to owner for the actual question.
- **Canonical table**: neither existing table is treated as canonical
  as-is, since the feature list itself is still open. Once the owner
  decides, a fresh canonical table will be built (reusing
  `build_shared_features.py`'s per-season z-scoring logic) for exactly
  the agreed feature set, and recorded here.

**0.2 Early-era validation backfill.** Fetched Frank Mason III (Kansas
2016-17), Trae Young (Oklahoma 2017-18), Zion Williamson (Duke 2018-19),
Ja Morant (Murray State/OVC 2018-19 - the requested non-power-conference
case) live from Sports-Reference/CBB (same curl+parse method as Phase 1,
WebFetch still blocked). All 24 comparisons (4 players x 6 rate stats)
landed within +/-0.5pp - no schema drift detected across the early-era
seasons. Added as 4 new fixtures to `tests/test_rate_formulas.py`
(11/11 passing including the originals).

**0.3 Standardization convention.** Confirmed (not changed):
`build_shared_features.py`'s existing per-season z-scoring already
computes each season's mean/sd from the FULL D1 MIN>=300 pool for that
season, never a conference-restricted subset - matches the spec's
required convention exactly. No code change needed; will carry forward
unchanged into whatever the final canonical table becomes.

**Awaiting owner decision on the 0.1 feature-list fork before Step 1.**

**Decision (owner):** keep the 12-core feature set aligned to the real
NBA 29-dim basis (do not backfill ft_pct/fg2_pct/fg3_pct/fga_per100/
share_3pa; do not drop TS%/TRB%/PORPAG/%FGAst'd_2P/_3P). **Canonical
table confirmed: `data/college/shared_features.parquet`** (12 shared
columns + PORPAG kept separate + per-season z-scored versions of the 12,
29,312 rows, 2016-17..2025-26, MIN>=300 complete cases). Proceeding to
Step 1.

## 2026-08-12 — Step 1 complete (fit pool + projection function)

What I did: built `src/college_model/pool.py` - `picked_conference_set()`
derives each NCAA-path crosswalk pick's conference from that exact pick's
OWN final_college_season (handles realignment correctly: a school's
conference-at-draft-time, not today's), giving 22 distinct conferences
(0 unresolved lookups against 480 matched picks). `restrict_to_draft_
conferences=True` (default) pool: **20,849 rows** across 22 conferences;
`=False` pool: all **29,312 rows** across 33 conferences. Per-season
breakdown recorded (both ON and OFF) in the module's own report output.
Built `src/college_model/project.py :: project_recipe(z_features, basis)`
- simplex-constrained NNLS (same augmented-row method as the already-
validated NBA-side project() in step1_archetype_model.py, reused for
consistency, not reimplemented), works on single vectors or batches,
fit-pool-independent by construction (takes any z-vector + any basis).
Smoke-tested: simplex constraint holds (non-negative, sums to 1 exactly),
batch mode correct, deterministic (NNLS has no RNG, verified anyway).

Obstacles: none. Confirmed a strong consistency signal, not just
internal self-consistency: the 20,849-row restricted-pool count exactly
matches an earlier informal (pre-Phase-1) attempt's result, despite this
version being rebuilt from the new 12-feature canonical table via the
more rigorous crosswalk-based per-season conference derivation - good
evidence the methodology is stable, not a coincidence worth worrying
about.

No open decisions from this step. Moving to Step 2 (K selection, AA
sweep) - CHECKPOINT 1 required before Step 3, will present curves and
wait.

## 2026-08-12 — Step 2 complete: K sweep done, CHECKPOINT 1

What I did: ran the full K=4..12 AA sweep (3 seeds each, 27 fits) on the
confirmed fit pool (20,849 rows x 12 z-features), max_iter=400 (raised
from an earlier session's 200, a pragmatic call given AA's real-data fit
time here, not re-asked given a near-identical question was already
raised once this session). Total runtime ~4.4 hours (monitored live,
not left unattended blind). Results: `data/college/model/k_selection_
{raw,summary}.csv`, `data/college/model/k_sweep_bases.npz` (best-per-K
bases retained), figures at `figures/phase2_k_selection_{rss_scree,
intra_variance}.png`.

Convergence: only 2 of 27 fits (K=7 seed=2, K=9 seed=0, K=9 seed=2 -
actually 3) hit true convergence before max_iter; the rest hit the 400-
iteration cap. Not fully resolved even at 2x the original iteration
budget - flagging plainly rather than treating as clean.

Real finding, not smoothed over: the intra-archetype-variance curve is
NOT cleanly monotonic - it oscillates (drops K4->K7, rises at K8, drops
to a global minimum at K9, rises again at K10, drops K11->K12). The
per-K restart (RSS) spread is also uneven: K5/K6 are very stable
(spread ~140-144), but K9/K10 - right around the curve's global minimum -
show the WORST restart stability in the whole sweep (spread ~1867-1942).
This is a direct tension with the spec's own stop condition language
("no consistent K optimum across restarts") - not a full trigger (K5/K6
ARE consistent), but real enough that a bare "take the global minimum"
reading is not being applied blindly. Full reasoning and recommendation
sent to the owner directly (not just filed here) - see chat.

No archetype anywhere in the sweep dropped below the 30-player flag
threshold (worst case min_group_n=261 at K=12).

**CHECKPOINT 1 - presented to owner, awaiting K confirmation before
Step 3.**

**Decision (owner): K=8.** Not the strict intra-variance global minimum
(K=9), but the owner judged the exact K choice isn't critical here and
picked K=8 - also notable as the same K the NBA basis itself uses.
Proceeding to Step 3 (AA vs ADA at K=8).

## 2026-08-12 — CHECKPOINT 1 amended by owner (2 corrections) before
## proceeding to Step 3

Owner caught a real methodological issue in the K=8 sweep result before
I froze it: the sweep's "best RSS" K=8 restart (seed 0, intra_var=0.602)
actually DISAGREED with the other two restarts (seeds 1/2, intra_var
~0.564, close to each other) - the lowest-RSS pick was an outlier basin,
not the consensus solution. My original plan (freeze the lowest-RSS
restart from k_sweep_bases.npz) would have frozen the wrong basin.

**Amendment A**: refit AA at K=8 only, seeds 0-7 (up from 3), max_iter=
1000 (up from 400). Cluster restarts into basins via Hungarian-matched
cosine similarity of archetype z-profiles (same basin iff min matched
cosine >= 0.95). Freeze the lowest-RSS restart WITHIN the consensus
basin (most seeds), not the global lowest-RSS restart. Re-plot intra-
variance as all-seeds scatter + per-K median, with an explicit note that
the original best-RSS-only plotting rule produced the K=8 artifact.

**Amendment B**: Step 3's ADA replaced with a bounded "restricted
archetypoid search" - (i) snap each frozen AA archetype to its nearest
real player-season (candns), solve alphas once; (ii) one single greedy
swap pass, 100-nearest-candidates per archetype, re-solving the full-
pool projection only to evaluate accepted swaps. Not the from-scratch
`archetypes` package ADA (already confirmed infeasible at this scale
earlier this session) and not the earlier 5-sweep/400-candidate version
I had started running before this amendment landed (killed, superseded).

Running `refit_k8_basins.py` now (background) - 8 restarts at max_iter=
1000 will take longer per fit than the original 400-iter sweep; will
report the basin table once done, before moving to the restricted ADA
search (Amendment B).

## 2026-08-12 — Amendment A complete: 3 basins found, consensus frozen

8 restarts (seeds 0-7, max_iter=1000) resolved into exactly 3 basins
(Hungarian-matched cosine, threshold 0.95):

| basin | seeds | RSS | intra_var | min_group_n | converged |
|---|---|---|---|---|---|
| 0 | 0, 5, 6 | 51682.9 (LOWER, i.e. "better" by RSS alone) | 0.6014-0.6015 (worse) | 783 | 2/3 |
| **1 (consensus)** | **1, 2, 3, 7** | **52446.7** | **0.5638-0.5639 (best)** | **795-796** | **4/4** |
| 2 | 4 | 52154.8 | 0.5941 | 1026 | 1/1 |

This directly confirms the owner's concern was correct, not just a
plausible worry: basin 0 (the ORIGINAL sweep's frozen choice) has the
LOWEST RSS of the three basins but the WORST intra-variance - i.e. "best
RSS" and "best structure" pointed in opposite directions here. Basin 1 -
4 of 8 seeds, ALL genuinely converged (not just hitting max_iter), best
intra-variance - is both the plurality basin AND the best on the actual
deciding diagnostic. Froze seed=2 (basin 1's lowest RSS, though all 4
basin-1 members agree to within 0.02 RSS - essentially one solution, not
4 different ones).

**Frozen K=8 basis**: `data/college/model/k8_frozen_basis.npz` (seed=2,
RSS=52446.70, intra_var=0.5639, min_group_n=796, converged, max_iter=
1000). Manifest: `data/college/model/k8_frozen_manifest.txt`. Re-plotted
`figures/phase2_k_selection_intra_variance.png` as an all-restarts
scatter + per-K median line, with a note that the original best-RSS-only
plotting rule is exactly what produced the K=8 artifact (K=4-7/9-12
still only have 3 restarts each from the original sweep - not
re-verified with the same rigor, flagged as an accepted scope limit
since K=8 is the confirmed choice, not those).

Moving to Amendment B (restricted archetypoid search / Step 3 ADA).

## 2026-08-12 — Amendment B complete: restricted ADA search, CHECKPOINT 2

RSS trajectory: continuous AA (frozen consensus basin) = 52446.70 ->
candns (nearest-real-point snap) = 59602.29 (+13.64%) -> post-swap
(one greedy pass, 100-nearest-candidates/slot) = 56320.54, final gap vs
AA = +7.39%. The swap pass improved 3 of 8 slots (1, 2, 4), no improving
candidate found for the other 5.

Archetypoid identity table (`data/college/model/ada_archetypoid_
identity.csv`): 7 of 8 archetypoids are from low/mid-major conferences
(OVC, CAA, WCC/Pacific, Ivy, Horizon, Big Sky, MVC) rather than power
conferences; 4 of 8 have minutes in the 315-600 range, close to the
MIN>=300 floor. Only 1 (Filip Petrusev, Gonzaga/WCC - a genuine high-
major, nationally recognized player) reads as an archetypoid a domain
expert would recognize on sight.

This is a direct, real trigger for the spec's own decision rule
("...unless archetypoids land on low-major statistical outliers... then
prefer AA"), not a marginal call - reported to the owner with the full
identity table for the actual CHECKPOINT 2 sign-off, recommendation:
prefer AA.

**CHECKPOINT 2 - presented to owner, awaiting AA-vs-ADA sign-off before
Step 4.**

**Decision (owner): AA.** Owner initially leaned ADA on the (incorrect)
assumption that the choice doesn't affect per-player recipes - corrected
with concrete evidence (recomputed 4 validation players' recipes under
both bases: L1 differences of 0.05-0.093, e.g. Trae Young's top-archetype
weight shifted 0.914->0.872 - recipes are a projection onto whichever
basis is chosen, so they necessarily differ). Owner then asked for the
spec's exact AA-trigger conditions restated plainly; on seeing them
mapped against the actual archetypoid table (condition 1, low-major
outliers: clearly triggered, 7/8 archetypoids from non-power conferences;
condition 2, tiny-minute seasons: only weakly triggered, lowest minutes
315 clears the 300 floor) - AA confirmed as final. **Final: K=8, AA,
frozen basis = `data/college/model/k8_frozen_basis.npz` (consensus-basin
seed=2).** Proceeding to Step 4.

## 2026-08-12 — Step 4 complete

Computed recipes for the entire canonical table (29,312 rows, not just the
20,849-row fit pool - "fit pool != projection set" per Step 1) via
project_recipe against the frozen AA basis. Simplex validity checked
in-script (all rows sum to 1 within 1e-6, no negative weights). All 8
archetypes get real assignment mass (smallest: archetype 0, 1,218 players;
largest: archetype 7, 5,493). Wrote `data/college/recipes.csv`,
`reports/college_archetypes.md` (per-archetype z-profile, top-5
distinguishing features, top-10 loaders with conference/season/minutes,
blank label field), and sanity projections for the 8 validation fixtures +
3 Nets rookies (top-3 archetypes each) - Mikel Brown Jr. loads primarily on
the same archetype as Trae Young/Ja Morant/Frank Mason III, consistent with
his scouting profile, reported as an eyeball plausibility note only, no
translation claim.

Obstacle worth flagging honestly: archetype 0's top loaders are dominated
by extreme-TOV%/low-minute low-major players (e.g. 315-minute OVC season) -
this looks like a "noisy small-sample" archetype rather than a clean style,
surfaced plainly in the interpretation doc for the owner's own judgment,
not smoothed over.

## 2026-08-12 — Step 5 complete, Phase 2 done

Restart stability at K=8: summarized from the CHECKPOINT-1-amendment basin
analysis (4/8 seeds in the consensus basin, all genuinely converged,
agreeing to within 0.02 RSS).

Conference-flag-OFF refit (2 seeds, max_iter=1000, full 29,312-row pool,
~1150-1200s/fit): Hungarian-matched against the frozen ON basis. Real
finding, not glossed over: 7/8 archetypes match well (cosine 0.77-0.99)
but archetype 3 does NOT match at all (cosine=-0.18). Archetype 3 (extreme
TS%/low USG%/low TOV%/high assisted-2P% - an efficient low-usage finisher
profile) turns out to be top-loaded almost entirely by mid/low-major
players with zero power-conference representation in its top 10 - a
concrete explanation for why removing the conference restriction changes
this specific archetype's survival. Verdict: K=8 AA structure is NOT fully
robust to the conference-restriction choice, contrary to what a clean
"7/8 match, close enough" reading might suggest on its own - the OWNER
should see this, not have it smoothed into a blanket "stable" verdict.

Temporal half-split (marked optional in spec): not run, time-budgeted
against the higher-priority required checks above - flagged as a real,
acknowledged gap in Section G/I of the model report, not silently skipped
without a trace.

Added `tests/test_college_model.py` (8 tests: simplex validity, row-count
match, argmax/alpha_max consistency, no-empty-archetype, projection
determinism, K-sweep internal consistency via the persisted CSV, basin-
selection consistency against the manifest). Full suite: 31/31 passing.

Wrote `reports/phase2_model_report.md` (all 9 sections A-I) and this final
worklog entry.

**Phase 2 is complete. Awaiting owner review before Phase 3 (anchor set /
translator) - per the spec's own hard boundary, no anchor-covariate or
translator work has been done anywhere in this phase.**
