# NBA 2025-26 Directed Stint Data — `src/pipeline/build_stints.py`

A standalone pipeline that builds a regression-ready "directed stint" table
from real NBA play-by-play, for a skill-weighted archetype-RAPM model. It is
entirely separate from this project's main archetype/synergy pipeline —
**it reads and writes nothing in `data/nets_synergy.db`**; all output lives
under `data/stints/`, a new directory, specifically so it can't conflict
with any existing table.

## What a "stint" is here

A continuous span of the game in which the same 10 players (5 per team) are
on the court. A new stint starts whenever **any substitution occurs on
either team, or a period ends** — even if the incoming lineup happens to be
identical to the outgoing one (a period boundary always splits a stint).

Each stint produces **two directed rows**, one per team as "offense": each
row states that team's own points scored and possessions used *during that
stint*, together with which 5 players were on offense and which 5 on
defense. A stint where one team never actually got a possession (rare, e.g.
a stint that opens and closes again mid-possession) simply has no row for
that side — not a zero, an absence.

## Why `pbpstats` isn't used, despite being the initially preferred tool

`pbpstats` was tried first. Its `stats_nba` data provider depends on the
legacy `boxscoretraditionalv2` stats.nba.com endpoint, which is **no longer
serving data for the 2025-26 season** — confirmed live (nba_api itself
raises `BoxScoreTraditionalV2 is deprecated ... Data is no longer being
published for BoxScoreTraditionalV2 as of the 2025-26 NBA season`), and
`pbpstats`'s last release predates that deprecation by over a year. This
pipeline is built from scratch on nba_api's modern `BoxScoreTraditionalV3`
and `PlayByPlayV3` endpoints instead, which were verified live to return
complete, fast, real data for the full season.

## How to run

```bash
pip install -r requirements.txt

# Full season (resumable - safe to interrupt and rerun)
python3 src/pipeline/build_stints.py --season 2025-26

# Test on a handful of games first
python3 src/pipeline/build_stints.py --season 2025-26 --limit 10

# Just recombine + re-validate already-pulled per-game files, no new pulls
python3 src/pipeline/build_stints.py --season 2025-26 --combine-only
```

**Resumability**: every game's raw boxscore/play-by-play response is cached
to `data/stints/raw_cache/{game_id}_box.parquet` / `_pbp.parquet`, and every
successfully-parsed game's directed rows are written to
`data/stints/games/{game_id}.parquet`, tracked in `data/stints/manifest.json`.
Rerunning the same command skips every game that's already done and only
pulls/parses what's missing or previously failed — safe to interrupt (Ctrl-C)
and resume at any time.

**Rate limiting**: `--sleep` (default 1.5s) throttles between games; failed
requests retry with linear backoff (5s, 10s, 15s... capped at 90s, 6 attempts)
before that game is marked failed in the manifest and skipped (rerun to retry
failed games automatically — the manifest only skips games marked `"done"`).

## Output files

| File | Contents |
|---|---|
| `data/stints/stints_2025_26.parquet` | The full combined directed-stint table |
| `data/stints/stints_2025_26_sample.csv` | A random 500-row sample, for quick inspection |
| `data/stints/players.csv` | `PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION` reference (from this project's own `player_base` table — not re-pulled) |
| `data/stints/manifest.json` | Per-game_id status: `"done"` or `"failed: <reason>"` |
| `data/stints/games/{game_id}.parquet` | One game's directed rows (the resumability unit) |
| `data/stints/raw_cache/{game_id}_{box,pbp}.parquet` | Raw per-game API responses (the caching unit) |

## Output schema (one row = one directed stint)

| Column | Type | Meaning |
|---|---|---|
| `game_id` | str | NBA game id |
| `date` | str | Game date |
| `stint_id` | int | Sequential stint number within the game (1-indexed) |
| `offense_team` | int | This row's offense team_id |
| `defense_team` | int | This row's defense team_id (the other team in the stint) |
| `home_off` | 0/1 | 1 if the offense team was the home team |
| `off_p1`..`off_p5` | int | The 5 offense player_ids on court this stint |
| `def_p1`..`def_p5` | int | The 5 defense player_ids on court this stint |
| `off_possessions` | float | Offense team's possessions used during this stint (real event-based count, not the FGA+0.44·FTA estimate) |
| `off_points` | float | Offense team's points scored during this stint (from the box score's own running score, not re-summed from shot values) |
| `y_off` | float | `100 * off_points / off_possessions` — points per 100 possessions for this stint |

Player ids are NBA.com's own `PLAYER_ID` space — the same one already used
throughout this project's `data/nets_synergy.db` tables and `recipes.csv` —
so this table joins directly against them with no id-mapping step.

## How possessions and points are actually computed (real logic, not an estimate)

- **Points**: taken directly from the box score's own running `scoreHome`/
  `scoreAway` fields (the difference between the score entering and leaving
  the stint) — not re-derived from summing shot values, which sidesteps
  needing to handle and-ones/technical FTs/etc. as a scoring question (the
  official score already reflects them correctly).
- **Possessions**: a real, event-based count (per the assignment's own
  requirement not to use the FGA+0.44·FTA estimate), built from these rules,
  each checked against real play-by-play before being encoded:
  - A **made shot** ends the shooting team's possession immediately (even
    with an and-1 free throw to follow — the FT is scored as part of the
    possession that already ended, not a new one).
  - A **turnover** ends the possession immediately.
  - A **missed shot** does not end the possession by itself — the following
    **rebound** decides: a defensive rebound ends the shooter's possession;
    an offensive rebound continues the same possession (second-chance).
  - **Free throws**: only the *last* attempt of a personal-foul trip (via
    the play-by-play's own "N of M" count) can end a possession — a make
    ends it immediately, a miss defers to the following rebound event, same
    as a missed field goal. "Technical" free throws never end a possession
    (the fouled team keeps the ball afterward).
  - **Edge case found via the full-season score reconciliation below**:
    NBA rules allow a substitution *between* free throws of a multi-shot
    trip. A non-last free throw scores real points without ending a
    possession by the rule above — if a stint boundary lands inside that
    trip, the points already happened but no possession had "ended" yet for
    that side in this exact stint window, and early testing found ~3% of
    total points were being silently dropped this way (the row got skipped
    outright since it had 0 counted possessions to divide by). Fixed by
    flooring to 1 possession whenever a side scored real points in a stint
    with an otherwise-zero possession count, instead of dropping the row —
    a small, bounded possession over-count in this specific rare case,
    traded for eliminating a real point under-count. This is what closed
    the score-reconciliation gap to exact (see below).

## On-court lineup tracking (the hard part)

- **Period-1 starters**: from `BoxScoreTraditionalV3`'s `position` column,
  which is non-blank only for the 5 starters per team — a standard, checked
  NBA.com box-score convention.
- **Substitutions**: NBA's own play-by-play text ("SUB: A FOR B") does
  **not** have a fixed in/out word order — confirmed live that sometimes the
  first-named player enters and sometimes the second does, in the same
  game. Direction is resolved by CURRENT ON-COURT STATE instead: both names
  are matched to player ids (handling first-initial disambiguation like
  "F. Wagner" vs. "M. Wagner", and diacritic mismatches between the box
  score and play-by-play text), and whichever of the two is presently
  tracked as on the court is the one leaving.
- **Between-period changes are not always logged as substitutions.**
  Confirmed live (cross-checking every shot/rebound/turnover/foul's own
  personId against tracked state) that a player can simply reappear next
  period with zero substitution event recorded for his return. Every period
  boundary (2nd quarter onward) re-derives the true opening five from
  directly-observed action-attributed players early in that period,
  correcting the substitution-tracked guess when they disagree.

## Known limitations (real, measured — not hidden)

- **A residual ~6% of substitution events are genuinely ambiguous** even
  after the above (measured on a 10-game sample: 24 unresolved out of
  roughly 400 substitutions). These are skipped atomically (the tracked
  lineup is left unchanged rather than corrupted) and self-correct at the
  next period boundary rather than compounding — but the affected stint(s)
  in between can have a stale (not necessarily wrong, just unconfirmed)
  lineup for a short window. The root cause is not fully understood; my
  working theory is simultaneous multi-player substitutions at the same
  clock timestamp aren't always paired 1:1 in the text the way a single
  "A FOR B" row assumes.
- **A possession that straddles a stint boundary** (time expires, or a
  substitution occurs, mid-possession) is split across two stints rather
  than attributed whole to one — an accepted edge case, not smoothed over.
- **Possession counts run consistently below the real number, checked
  directly against nba_api's own official `LeagueDashTeamStats` PACE/POSS
  for all 30 teams (2025-26)** — not just an eyeballed "typical" range:
  every single team is undercounted, by 3.5-7.1 possessions/game (mean 5.4,
  ~5.4% relative). This is a systematic gap, not per-team noise. Checked
  what it correlates with rather than just asserting a cause:
  - **substitution frequency**: r ≈ -0.40 to -0.42 against the true gap —
    teams that substitute more do show a bigger gap, real (if moderate)
    evidence the ~6% residual substitution ambiguity above is a genuine
    contributor.
  - **team archetype composition** (this project's own K=8 recipes,
    MIN-weighted team share): the strongest single correlate is `arch_7`
    (Mbeng-type - low USG%/BPM, high STL%, PnR-adjacent) at r ≈ +0.63 —
    stronger than substitution frequency, though less mechanistically
    obvious (plausibly steal-heavy/scrambly playstyles create more
    broken-play substitution clusters; with only 30 teams this is a real
    measured correlation, not a confirmed causal explanation).
  Even the single best team (Boston) still has a +3.5-possession gap, so
  substitution ambiguity likely isn't the whole story — there's probably
  also a smaller, universal gap from a possession-ending scenario missing
  from the 5 rules entirely (backcourt violations, defensive 3-seconds, a
  jump-ball redo — none are currently handled). This affects the
  denominator of `y_off`, not `off_points` (which reconciles exactly - see
  below), so aggregate `y_off` runs a little optimistic rather than points
  being wrong. Full numbers documented in `build_stints.py`'s own module
  docstring.
- **Validated on the full, real 2025-26 season (1230/1230 games, not a
  sample)**:
  - **Score reconciliation: 1230/1230 games (100.0%) exactly match their
    real final score** (summed directed `off_points` per team vs. the box
    score's own final `scoreHome`/`scoreAway`) — this was not exact on the
    first full run (only ~5.7% of games matched exactly, off by a mean of
    ~6.75 combined points/game); tracked down to the free-throw-mid-trip-
    substitution edge case above and fixed, closing the gap to exact across
    every game in the season, not just a sample.
  - Every one of the 75,587 output rows passes the 5-distinct-offense +
    5-distinct-defense player-id check, with zero exceptions.
  - 5 of 1230 games initially failed outright on a separate bug (both
    teams' schedule rows carried an identical, ambiguous MATCHUP string,
    breaking home/away detection) - fixed by parsing the home team directly
    out of the "X @ Y" text instead of relying on a per-row perspective
    assumption; all 1230 games parse successfully now.
  See the validation report printed at the end of every run (also
  re-derivable any time via `--combine-only`) for the exact current numbers.
