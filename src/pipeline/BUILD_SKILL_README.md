# Exogenous player skill — `src/pipeline/build_skill.py`

Produces `data/model/players_skill.parquet`: one row per player_id appearing
in the 2025-26 directed-stint table (`data/stints/stints_2025_26.parquet`),
with a real, sourced `skill_off`/`skill_def` value and zero drops. This is
the `skill_off`/`skill_def` input to the skill-weighted archetype-RAPM in
`fit_rapm.py` (see `RAPM_README.md`).

## Why this file exists

`fit_rapm.py`'s join-coverage report found that using only prior-season
(2024-25) Basketball-Reference OBPM/DBPM as the skill source dropped **68.3%**
of stint rows — almost entirely because ~118 players in the 2025-26 season
(mostly true rookies, plus a handful of draft-and-stash/returning players)
have **no 2024-25 row at all**, so they could never get a skill value from a
purely prior-season lookup, no matter which season was passed. This file
fills that gap with a real, three-tier source hierarchy instead of dropping
those players (and every stint row any of them appear in) from the model.

## Exogeneity — the one rule everything else follows

A skill value's **timestamp must predate 2025-26 tip-off**. Preseason or
prior-season values are fine; any in-season/current-season 2025-26 DARKO,
EPM, or BPM value is forbidden, because it would make the "skill" covariate
partially a function of the very 2025-26 outcomes being regressed — circular
by construction. Every player gets exactly **one fixed value for the whole
season** (no in-season updating), and every value's `vintage` string
documents where it came from.

**A real leakage trap caught before writing any code, not after**: this
project's own `player_advanced_bref` table already contains real, in-season
2025-26 rows for this season's own rookies (e.g. a 2025-26 draft pick already
has a populated 2025-26 OBPM/DBPM row by the time this pipeline runs, since
that table is a live pull, not a frozen historical archive). Using those
rows directly for 2025-26 rookies would be leakage for exactly the players
the draft-slot tier exists to predict. Fixed by excluding the *current* stint
season from all training/reference data (the rookie curve's training set and
the undrafted replacement-level average both explicitly drop it) and scoring
2025-26's own rookies via the fitted curve's **prediction**, never a lookup
of their own real in-season row.

## Source hierarchy (checked live before writing the fallback path)

DARKO and EPM (the prompt's preferred first choice) were checked directly,
not assumed available: both `darko.app` and `dunksandthrees.com`'s public EPM
pages only expose the current, live-updating 2025-26 season with no
preseason/historical archive or season selector without a login — i.e.
exactly the in-season data this file is required to refuse. So every player
here falls into one of these three tiers instead:

| Tier | `source` value | Who | How |
|---|---|---|---|
| 1 | `prior_season_epm`* | Has a real 2024-25 Basketball-Reference row | 2024-25 OBPM/DBPM, MIN-weighted shrinkage toward the league mean (see below) |
| 2 | `draft_prior` | No 2024-25 row, but drafted (any year) | Predicted from real draft number via a fitted historical rookie-season curve |
| 3 | `replacement_level` | No 2024-25 row, undrafted | Empirical mean of real historical undrafted rookies' debut seasons |

*named `prior_season_epm` (not `prior_season_darko`/`prior_season_bref`) to
match the prompt's own documented-alternative source label — the actual
values are Basketball-Reference OBPM/DBPM, the concrete substitute used once
DARKO/EPM were confirmed unobtainable as a clean prior-season snapshot; this
substitution is flagged here explicitly, not silently treated as identical.

## Tier 1 — veteran shrinkage (a bug caught by inspecting real output, not assumed clean)

Raw single-season Basketball-Reference OBPM/DBPM is a rate stat and gets
extremely noisy at low minutes: Alondes Williams' entire 2024-25 season was
4 minutes, and his raw OBPM is **+37.9** — not a real skill signal, just
noise from a tiny sample. 20 of the 463 veteran-tier players who actually
appear in the 2025-26 stint table (4.3%) have under 50 2024-25 minutes.
Feeding these raw values into the model as if they were stable would inject
exactly the kind of uncontrolled-input contamination this project's synergy
matrix already had to guard against for star players (see CLAUDE.md's
"Known pitfalls") — here it's noise contamination instead of star
contamination, same failure mode.

**Fix**: standard minutes-weighted shrinkage toward the league mean,
`skill = w * raw + (1-w) * league_mean` where `w = MP / (MP + 500)` — a
well-established technique (regression to the mean weighted by sample size),
not a bespoke curve. `k=500` MP is a round, defensible stabilization point
(roughly half of a full-season rotation share) chosen before looking at this
project's own results, not tuned to them. Post-shrinkage range: OBPM
[-4.1, +8.3] (vs. raw [-19.5, +37.9]) — a plausible single-season BPM range.
The league mean itself came out to ~0.00 for both OBPM and DBPM, consistent
with BPM's known zero-mean-by-construction property — an incidental sanity
check that the shrinkage target is being computed correctly.

## Tier 2 — draft-slot prior (fit on this project's own real historical data)

`player_bio.DRAFT_NUMBER` + `player_advanced_bref.OBPM/DBPM`, joined on a
player's true rookie season (the `player_bio` row where `DRAFT_YEAR` equals
that row's own season's start year — not just "first season in this DB",
which would misclassify anyone whose real debut predates this DB's 2017-18
start). Built a 471-row historical rookie-season dataset spanning 2017-18
through 2025-26 draft classes; **excluded the 2025-26 class from training**
(the leakage trap above) leaving 416 rows across 2017-18–2024-25.

Fit: minutes-weighted least squares, `OBPM ~ a + b*log(draft_number)`, same
form for DBPM (log-pick is the standard draft-curve shape — steep drop-off
at the very top, flattening by the second round; checked, not assumed, to
fit this data's own shape better than a raw-linear form).

```
OBPM ~ -0.371 - 0.588 * log(pick)   (weighted R^2 = 0.102)
DBPM ~ -0.747 + 0.099 * log(pick)   (weighted R^2 = 0.007)
```

Predicted values:

| Pick | skill_off | skill_def |
|---|---|---|
| 1 | -0.37 | -0.75 |
| 15 | -1.96 | -0.48 |
| 30 | -2.37 | -0.41 |
| 55 | -2.73 | -0.35 |

**Reported honestly, not hidden**: draft slot explains a real but modest
10.2% of rookie-season OBPM variance, and essentially none (0.7%) of DBPM
variance — a rookie's draft position says very little about his defensive
impact on its own. This is a genuine finding, in the same spirit as this
project's other "state the limitation plainly" results (e.g. the pace-gap
correlations in `STINTS_README.md`).

## Tier 3 — replacement level (undrafted, empirical not assumed)

Rather than extrapolate the fitted curve past pick 60 (extrapolation beyond
the observed data range), replacement level is the **MIN-weighted mean of
302 real historical undrafted players' own debut-season** OBPM/DBPM
(excluding 2017-18 — this DB's own window start, so a true debut can't be
confirmed — and 2025-26, the same leakage exclusion as Tier 2):

```
skill_off = -2.50   skill_def = -0.53   (n=302, MIN-weighted)
```

Chosen over the unweighted mean (OBPM -4.18, DBPM -0.93) because the
unweighted mean is dominated by tiny-sample noisy outliers (the same
small-sample BPM problem Tier 1's shrinkage exists to fix) — the
MIN-weighted mean is the more defensible empirical choice. It sits between
the curve's own pick-30 (-2.37) and pick-55 (-2.73) predictions, a sensible
place for "worse than any drafted player" to land — a real cross-check, not
identical by construction, but in the right neighborhood.

## Validation (from the pipeline's own printed report)

- **581/581 stint-table players** get a value: 463 via Tier 1, 64 via Tier 2,
  54 via Tier 3.
- **118 rookie-tier players** (Tier 2 + Tier 3 combined) — all 118 confirmed
  to have zero 2024-25 `player_base` row at all (true rookies), matching the
  ~118 figure `fit_rapm.py`'s join-coverage report originally flagged.
- **Leakage guard**: asserted 0/581 rows cite 2025-26 as a performance data
  source.
- Full distribution-by-source and draft-curve-shape numbers print on every
  run — rerun `python3 src/pipeline/build_skill.py` to reproduce them.

## How to run

```bash
python3 src/pipeline/build_skill.py
# --stint-season / --skill-season / --stints-path / --out-path all overridable
```

Reads only `data/nets_synergy.db` (existing tables) and
`data/stints/stints_2025_26.parquet` (existing file, read-only). Writes only
`data/model/players_skill.parquet` — does not touch any existing data.
