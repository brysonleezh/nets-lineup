# Skill-weighted archetype-RAPM — `src/fit_rapm.py` / `src/score_lineup.py`

Fits a possession-level value model on top of `build_stints.py`'s directed
stint table: each archetype-pair's estimated offensive/defensive tilt and
synergy, controlling for the individual talent of the players providing it.
Entirely separate from this project's main archetype/synergy pipeline —
reads `data/stints/stints_2025_26.parquet`, `data/basis_2025_26/recipes.csv`,
and `data/model/players_skill.parquet` (see `STINTS_README.md` /
`BUILD_SKILL_README.md`); writes only to `data/model/`.

## Inputs

| Input | Source | Required columns |
|---|---|---|
| Stints | `build_stints.py` | see `STINTS_README.md` |
| Alpha (archetype recipe) | `data/basis_2025_26/recipes.csv` | `PLAYER_ID`, `arch_0..arch_{k-1}` (rows sum to 1) |
| Skill (exogenous) | `data/model/players_skill.parquet` | `player_id`, `skill_off`, `skill_def`, `source`, `vintage` — see `BUILD_SKILL_README.md` |

Alpha coverage is capped by Step 1's own **≥300-minute ADA filter** — a
pre-existing modeling decision (see CLAUDE.md), not something this file
patches. Skill coverage is 581/581 (100%) once `players_skill.parquet` is
built — the whole reason that file exists.

**Known, deliberately-not-fixed gap**: of the 148 alpha-missing players,
147 are genuinely under 300 minutes (the filter working as intended). One,
**Ronald Holland II (1,550 real 2025-26 minutes)**, is missing purely from a
name-join miss in Step 1 (`step0_data_collect_process.py` matches him via
Basketball-Reference's short form "Ron Holland" against NBA.com's "Ronald
Holland II" - already fixed in code via `_NAME_ALIASES`, but `recipes.csv`
predates that fix). Regenerating the basis to include him was tried and
reverted: refitting on 434 rows instead of 433 changed 4 of the 8 archetype
exemplars (Tim Hardaway Jr.→Klay Thompson, Shai Gilgeous-Alexander→Luka
Dončić, Bez Mbeng→Bryce McGowens, D'Angelo Russell→none), which would make
`step1_archetypes_model.py`'s hand-written `ARCHETYPE_TO_PAPER` labels stale
for those 4 rows and ripple into any already-reviewed downstream content
that names an old exemplar (e.g. `STINTS_README.md`'s "arch_7 = Mbeng-type"
reference). Given one bench player (~0.7% of total possessions) doesn't
justify re-reviewing 4 archetype labels and re-verifying already-built
downstream numbers, the fit was reverted to the original 433-row basis and
Holland stays excluded from `recipes.csv` - a known, accepted limitation,
not an oversight.

## Design matrix (k = 8 archetypes → 75 columns)

Per directed stint row:

```
[ Soff, Sdef, home_off,                     <- unpenalized talent/context baseline
  Eoff_1..k, Edef_1..k,                     <- skill-weighted archetype exposure
  offpair_ij (i<j), defpair_ij (i<j) ]      <- within-side archetype-pair interactions
```

`Eoff_a = sum over the 5 offensive players of (archetype-a share * skill_off)`
— archetype exposure weighted by how skilled the specific players providing
it are, not a plain share average (a scrub and a star of the same archetype
aren't treated identically). `Soff`/`Sdef` are plain sums of on-court skill.
Pairwise terms are literal products `Eoff_i * Eoff_j`, never crossing
offense with defense — this is what lets the model express "these two
archetypes together are worth more/less than either alone" (the synergy
term), not just a main-effects model.

## Fitting: differentially-penalized weighted ridge

Closed-form weighted ridge with a **per-column penalty**: `Soff`/`Sdef`/
`home_off` (3 columns) are never penalized; the 72 archetype-tilt +
synergy columns are penalized by a single `lambda`, chosen via `GroupKFold`
(grouped by `game_id`, weighted by `off_possessions`) — implemented by hand
rather than sklearn's `Ridge` (no per-column penalty argument) and
standardization is refit **inside each CV fold** on that fold's own training
data only (a real, easy-to-miss leakage point in cross-validated ridge).

**Leave-one-team-out**: the holdout model's training set explicitly excludes
every row where `offense_team` OR `defense_team` is Brooklyn — `Brooklyn is
out-of-sample by construction for its own evaluation`, asserted via
`assert_no_leakage`, not just intended.

## How much does archetype/synergy actually explain? (an honest finding)

The first full run's lambda sweep picked `lambda=1e4` — the **right edge**
of the originally-tested grid `[1e-1, 1e4]`, a classic sign the grid was too
narrow. Widening it to `[1e-1, 1e7]` found the real (very shallow) interior
minimum around `lambda≈1.8e5`. A baseline comparison at the same CV splits
shows why:

| Model | CV weighted RMSE |
|---|---|
| Null (mean of `y_off` only) | 114.05 |
| Talent-only (`Soff`/`Sdef`/`home_off`) | 113.97 |
| Full model (+ archetype tilt + synergy), optimal lambda | 113.95 |

All three are within ~0.1% of each other. **At the single-stint level,
almost none of `y_off`'s variance is explainable by anything** — talent
included — because a single stint is often just a handful of possessions,
and points-per-100-possessions over a tiny sample is inherently extreme
(the possession-count noise floor dominates). This is the expected, textbook
behavior of RAPM-style regressions: very low per-possession R², with the
real signal only emerging in the **fitted coefficients** once aggregated
across many stints — not a sign this model is broken, and not swept under
the rug here.

**The more meaningful check** is lineup-level, where possessions actually
aggregate — see the GATE 2 section below. An earlier draft of this README
(and of `AI_USAGE.md` Entry 043) reported this level as "r=0.95 against real
Brooklyn lineups" and treated it as reassuring. That number was **ORtg-only,
at n=5 — the single smallest, least reliable sample available** — and it
does not hold up once checked properly. See GATE 2.

## GATE 2 — out-of-sample validation at the lineup level: **FAILED**

**Do not use this model to drive roster-deployment decisions as it stands.**
This is a real, checked finding, not a formality.

For real Brooklyn 5-man lineups with ≥100 offensive possessions in 2025-26,
the (Brooklyn-holdout) model's predicted **Net** (ORtg vs. league-average
defense, minus that same 5 on defense vs. league-average offense) was
compared against **observed Net** (real ORtg minus real DRtg, from that
lineup's own actual offensive and defensive stint rows). At the exact
threshold below, this is the formal gate: correlation < 0.3 or an
implausibly large MAE means stop, don't proceed to lineup-ranking/deployment
output.

| min_poss | n lineups | ORtg corr | **Net corr** | Net MAE (pts/100 poss) |
|---|---|---|---|---|
| 30 | 50 | 0.26 | 0.13 | 22.4 |
| 50 | 23 | 0.61 | 0.17 | 22.2 |
| 75 | 8 | 0.90 | 0.50 | 19.0 |
| 100 | 5 | 0.95 | **0.12** | 8.5 |

**GATE 2 fails at every threshold checked, not just the specified one.**
The n=5/n=8 rows *look* better on ORtg, but that's exactly the small-sample
instability this table exists to catch — Net correlation stays weak (0.12–
0.17) at every n large enough to mean anything (23–50 lineups), and MAE
(19–22 points/100 possessions) is larger than the entire realistic range of
NBA net ratings. The earlier "r=0.95" was small-sample luck on the easier
(ORtg-only) half of the question, not a validated result — flagging that
plainly rather than letting it stand.

**Likely causes** (per-cause, not just "it failed"):
1. **Heavy ridge shrinkage.** The chosen `lambda≈1.8e5` (see above) already
   shrinks the archetype/synergy terms close to zero at the single-stint
   level. Predicted Net across all 5 lineups at the n=5 cut spans only
   about -1.4 to -5.1 — almost no variance to correlate against anything,
   independent of whether the underlying archetype-pair idea is right.
2. **Lineup-level defensive rating is intrinsically noisy.** 100–500
   possessions is a small sample for isolating one specific 5-man unit's
   defense; Net compounds two noisy numbers (ORtg and DRtg) instead of one.
3. **A real definitional mismatch.** Observed Net reflects whatever real
   opponents Brooklyn actually faced during those specific stints (which
   varies a lot game-to-game); predicted Net assumes a fixed
   league-average opponent throughout. That's not quite the same
   comparison, and adds noise unrelated to coefficient quality.

**Decision (given this finding, discussed directly)**: stop here. Phases 3
and 4 of the original plan (enumerate/score every Nets 5-man lineup,
archetype marginal-value "needs" ranking) were **not built** — building
roster-deployment output on a model that fails its own validation gate
would risk presenting numbers that could genuinely mislead a real personnel
decision, which is the opposite of this project's own stated bar for
honesty over overclaiming. Roster Construction's lineup-ranking capability
should instead build on `step4_roster_construction.py`'s WLS mixture model
(`outcome ~ sum_bpm + archetype shares + entropy`), which has its own real,
already-validated cross-validated R² and does not depend on this
possession-level Net comparison at all.

This does not retroactively invalidate the archetype-pair coefficients
themselves as a descriptive/diagnostic object (Boff/Bdef/Goff/Gdef still
describe real fitted tilts and interactions) — it specifically means their
**lineup-level Net predictions are not validated well enough to rank real
5-man combinations or drive a deployment recommendation**, which was the
whole point of Phases 3-4.

## Extended investigation: is GATE 2's failure fixable, or fundamental?

After GATE 2 failed, rather than stop at "the model doesn't work," each
plausible fixable cause was tested directly against real data, one at a
time. None of them moved the needle. This section documents that search —
what was tried, what was found, and why the conclusion below is a
genuine negative result rather than an unfixed bug.

**1. Is the metric too lenient/strict?** Pearson correlation only measures
whether predicted and observed track the same direction, not whether the
scale matches. Computed proper out-of-sample R² (`1 - SS_res/SS_tot`) on
the same 1317-lineup pooled sample: **R²≈0.04 for Net, R²≈0.01 for ORtg** —
worse than the correlation numbers suggested. Predicted Net's own standard
deviation across all 1317 real lineups was 5.4 vs. observed Net's 28.9 —
the model barely varies its answer regardless of who's on the floor.

**2. Is the model too complex (56 pairwise synergy terms overfitting)?**
Simplified to an 8-feature model (`Eoff_a - Edef_a` per archetype only, no
interactions, no talent, no home), re-evaluated on the same pooled 30-team
holdout task: **corr=0.131** — *worse* than the full 75-feature model's
0.212, not better. Complexity was not the bottleneck.

**3. Are individual stints too short (mean 93 seconds, ~3 possessions) to
carry signal, with the noise fixable by filtering?** Refit the full model
after dropping stint rows below a minimum possession count, sweeping
1/3/5/8/12 (`--min-possessions` style filter, training data shrinking from
75,587 to 833 rows at the strictest end): pooled Net corr stayed flat at
**0.20–0.22 across the entire range**, then became uncomputable past 12
(insufficient training data). Possession-weighting already handles this
noise in the loss function; filtering just discards data for no gain.

**4. Does dropping the skill axis entirely change anything?** Refit with
`skill_off=skill_def=1.0` for every player (`Eoff_a` becomes a pure
archetype-share sum, no talent weighting at all; `Soff`/`Sdef` dropped,
since they become a structural constant): **corr=0.199, R²=0.037** —
statistically indistinguishable from the skill-weighted version. Whether
skill enters the exposure term at all does not explain the gap.

**5. Is "Net rating" even a reliably measurable quantity for a specific
lineup at this data volume, independent of any model?** Split-half
reliability test: for every real lineup, randomly split its own stint rows
into two halves, compute observed Net for each half independently, and
correlate the two halves against each other — the empirical ceiling any
model could possibly reach.

| min_poss per half | n lineups | split-half Net corr | split-half ORtg corr |
|---|---|---|---|
| 15 | 705 | -0.098 | +0.045 |
| 30 | 262 | -0.363 | -0.021 |
| 50 | 138 | -0.429 | -0.140 |

**The ceiling is at or below zero, not the model's own ~0.2.** A specific
lineup's own Net rating, split against itself, doesn't even correlate with
itself at this sample size — meaning "specific-lineup Net rating in a
single season" is not a stably measurable target at all here, independent
of which model or features are used to predict it.

**6. Does more data help (1 season → 3 seasons)?** Pulled two additional
full seasons (2023-24, 2024-25) via `build_stints.py`, and extended the
pipeline to support them:
- **Archetype coverage**: `step1_archetypes_model.py`'s `project()`
  function (built for exactly this) projects each historical season's
  players onto the **existing, already-labeled 2025-26 basis** — never
  refit, avoiding a repeat of the archetype-exemplar-reshuffle problem from
  the Ronald Holland investigation. Saved to `data/rc_multiseason/
  recipes_multiseason.csv` (1276 player-seasons across 3 seasons) —
  the portal's own `data/basis_2025_26/recipes.csv` is untouched.
- **Skill coverage**: `build_skill.py`'s `build_skill_table()` was already
  parameterized by `(stint_season, skill_season)` - reused directly with
  `skill_season` shifted back one year per additional season (2023-24 uses
  2022-23 as prior; 2024-25 uses 2023-24), each independently leakage-checked.
- **A real bug was found and fixed along the way**: `combine_and_validate()`
  globbed the entire shared `data/stints/games/` cache directory instead of
  filtering to the requested season's own game_ids - since that directory
  holds every game ever pulled for any season, this silently mixed other
  seasons' games into whatever season's output file was being combined
  (`stints_2024_25.parquet` ended up 1230/1235 "games" actually being
  2025-26 games; a first, buggy 2023-24 pull showed a physically
  impossible ~190 possessions/game, exactly 2x real pace). **`stints_2025_26.parquet`
  itself was unaffected** - not because the bug didn't apply to it, but
  because it happened to be combined before any other season's games
  existed in the shared directory - confirmed by checking its game_id
  composition directly (1230/1230 all `00225`) before trusting any
  conclusion drawn from it earlier in this investigation. Fixed by
  filtering to `games_df["GAME_ID"]` (the season's own real schedule)
  before combining; both affected seasons were regenerated and reverified.
- **Result, pooled 2-season (2024-25+2025-26) lineup-level test**:
  corr=0.204 vs. the 1-season baseline's 0.212 - no improvement.
- **Result, team-season-level test (the most decisive check)**: rather
  than evaluate at the noisy lineup level, aggregate every prediction for
  an ENTIRE team's season (holdout-model-scored, possession-weighted
  across ~7,500-8,000 real possessions) and compare against that team's
  **real official NET_RATING** (`LeagueDashTeamStats` - a clean,
  independent ground truth, not derived from this project's own stint
  table). Run across all 30 teams x 3 seasons = 90 team-seasons:

  | Season | n | Full model corr | Talent-only corr |
  |---|---|---|---|
  | 2023-24 | 30 | 0.756 | 0.786 |
  | 2024-25 | 30 | 0.779 | 0.787 |
  | 2025-26 | 30 | 0.804 | 0.782 |
  | **Pooled** | **90** | **0.770** | **0.767** |

  At this, the lowest-noise aggregation tested (a full team-season, ~8,000
  possessions), the model finally shows strong real-world correlation
  (~0.77) - but a talent-only baseline (zeroing every archetype/synergy
  coefficient, keeping only `Soff`/`Sdef`/`home_off`) matches it almost
  exactly, and beats it outright in 2 of 3 seasons. **Team quality is
  predictable - archetype composition and synergy add no detectable
  increment on top of talent, even here.**

**Conclusion.** Six independent angles - metric choice, model complexity,
stint length, skill-weighting, the target's own split-half reliability,
and 3x the data at the most favorable possible aggregation level - all
converge on the same answer: **the archetype-composition/synergy signal
this file was built to estimate does not show a reliable, reproducible
marginal contribution beyond individual talent, at any granularity tested,
in this data.** This reads as a property of the modeling approach and the
available data volume, not a bug waiting to be fixed. It does not
invalidate archetype recipes as a *descriptive* tool (composition
diagnostics, style-similarity search - see Diagnostic Analysis and
Scouting) - it specifically means they don't yet predict lineup-level
*outcomes* better than talent alone.

## Outputs

- `data/model/coefficients_full.json` — all 30 teams.
- `data/model/coefficients_holdout_BKN.json` — Brooklyn excluded from
  training; used for the Brooklyn-facing sanity check and any Nets-facing
  lineup scoring, so Brooklyn's own possessions never leak into its own
  evaluation.
- Each JSON stores: intercept, `d_off`/`d_sdef`/`h_home` (talent/home
  coefficients), `Boff`/`Bdef` (per-archetype tilt), `Goff`/`Gdef`
  (per-archetype-pair synergy), the fitted scaler (`scaler_mean`/
  `scaler_std` — reused verbatim by `score_lineup.py`, never recomputed),
  the observed per-feature standardized range (`feature_min`/`feature_max`
  — added for a planned in-support/extrapolation flag on scored lineups;
  the flag itself was never built in `score_lineup.py` once GATE 2 stopped
  Phase 3, so these two fields currently sit unused, not wired to anything
  — left in place since they're harmless and cheap to regenerate), chosen
  `lambda`, and possession-weighted league-average `Eoff`/`Edef`/`Soff`/
  `Sdef` vectors (used to score a lineup "vs. league-average", also
  duplicated standalone in `data/model/league_average_opponent.json`).

## `score_lineup.py`

`score_lineup(off_ids, defense, model, alpha, skill, k)` → `{ORtg, DRtg, Net}`:
- `ORtg`: `off_ids` on offense vs. `defense` (5 ids, or the literal string
  `"league_average"`).
- `DRtg`: `off_ids` on defense vs. **league-average offense** — a fixed
  definition regardless of what `defense` was for `ORtg`, matching the
  original spec's wording literally.
- `Net = ORtg - DRtg`.

Rebuilds features via the exact same `build_features_single` code path
Stage 1 training uses, and applies the model's own stored scaler — training
and scoring can never silently drift apart.

## How to run

```bash
python3 src/build_skill.py          # if data/model/players_skill.parquet doesn't exist yet
python3 src/fit_rapm.py             # full-league + BKN-holdout models, full validation report
python3 src/score_lineup.py --model data/model/coefficients_holdout_BKN.json \
    --off <5 player_ids> --def league_average
```

`--skill-season <season>` (legacy) falls back to the old prior-season-BPM-
only lookup (drops any player with no prior season at all) for direct
comparison against the default `--skill-path` behavior — not the default.
