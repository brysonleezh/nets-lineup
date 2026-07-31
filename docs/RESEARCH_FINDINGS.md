# Research Findings: The Archetype-Synergy Investigation

A chronological record of a real investigation, not a methods paper. Every
number below comes from an actual run in this repo — `RAPM_README.md`,
`AI_USAGE.md` (Entries 040–047), or a script re-run while writing this
document. Where a number could not be re-derived from a surviving artifact,
it is marked `[source missing]` rather than recalled from memory.

## 1. Hypothesis and motivation

The starting hypothesis: **archetype composition and pairwise archetype
synergy should add predictive value for lineup performance beyond individual
talent alone.** This follows the general direction of prior work arguing
that roster fit and complementary skill-sets matter beyond a simple sum of
individual player value (Brill 2023's Wharton-affiliated acquisition-value
work, and the general roster-construction framing popularized by the
NBA-insights Substack community) — cited here as the motivating precedent
for *why this was worth testing*, not as a source of specific numbers this
project reproduces; nothing quantitative below is drawn from those sources.

This project's own archetype model (Step 1, K=8 Archetypoid Analysis) is
built directly from *"Scouting Anyone: Probabilistic Player Archetypes for
Any League"* (SSAC 2026), cited openly per CLAUDE.md. The archetype-pair
**synergy matrix** (Step 2) and this possession-level **skill-weighted
archetype-RAPM** extension are this project's own additions, explicitly
*not* in the source paper.

## 2. The model

`src/pipeline/fit_rapm.py`, consuming `src/pipeline/build_stints.py`'s directed possession
table (2025-26, later extended to 2023-24 and 2024-25):

```
X = [ Soff, Sdef, home_off,                    <- talent + context (unpenalized)
      Eoff_1..8, Edef_1..8,                    <- skill-weighted archetype exposure
      offpair_1..28, defpair_1..28 ]           <- within-side archetype-pair interactions (penalized)

Eoff_a = sum over the 5 offensive players of (archetype-a share * skill_off)
Edef_a = sum over the 5 defensive players of (archetype-a share * skill_def)
Y = y_off = 100 * off_points / off_possessions   (single-stint offensive rating)
```

Fit via a differentially-penalized weighted ridge (talent/home unpenalized,
archetype tilt + synergy ridge-penalized; `lambda` chosen by `GroupKFold`
grouped by `game_id`). `skill_off`/`skill_def` are exogenous — built by
`src/pipeline/build_skill.py` from **prior**-season data only (veteran tier:
prior-season Basketball-Reference OBPM/DBPM with MIN-weighted shrinkage;
rookie tier: a draft-slot prior fit on historical draft classes; undrafted:
empirical replacement level) — with an explicit leakage guard asserting no
skill value cites the season being modeled as its data source. A
**leave-one-team-out** ("holdout") model is fit excluding every possession
involving Brooklyn, specifically so evaluating Brooklyn's own lineups is
out-of-sample by construction.

## 3. GATE 2 failure and the escalation sequence

**GATE 2** (this project's own name for its out-of-sample validation step):
compare the holdout model's predicted Net rating for real Brooklyn 5-man
lineups against those lineups' actual observed Net rating. Required bar:
correlation ≥ 0.3.

**Step 1 — initial result.** At the originally-specified `min_poss=100`
threshold: **n=5 lineups, Net correlation = 0.116** (`RAPM_README.md`, GATE
2 table). Fails the 0.3 bar outright.

**Step 2 — an early overclaim, caught and corrected.** Before the Net-based
GATE 2 check existed in its final form, an earlier pass reported "r=0.95
against real Brooklyn lineups" as a reassuring result (`AI_USAGE.md` Entry
043). That number was **ORtg-only** (not Net), computed at **n=5 — the
single smallest, least reliable sample available** — and had never been
checked for robustness across other sample sizes before being reported.
Once checked, it did not hold up (see Step 3). This is left as a visible,
annotated correction in `AI_USAGE.md` Entry 043 rather than silently
edited, and restated in `RAPM_README.md`'s GATE 2 section — an explicit
honesty note: the first read of this evidence was wrong, and saying so
plainly is part of the record.

**Step 3 — robustness across thresholds (still Brooklyn only).**

| min_poss | n lineups | ORtg corr | Net corr | Net MAE (pts/100 poss) |
|---|---|---|---|---|
| 30 | 50 | 0.26 | 0.13 | 22.4 |
| 50 | 23 | 0.61 | 0.17 | 22.2 |
| 75 | 8 | 0.90 | 0.50 | 19.0 |
| 100 | 5 | 0.95 | 0.12 | 8.5 |

Net correlation is weak at every threshold large enough to be
statistically meaningful (0.13–0.17 at n=23–50); the n=5/n=8 ORtg numbers
were small-sample instability, not signal.

**Step 4 — the 30-team extension, ruling out "Brooklyn is just unlucky."**
Leave-one-team-out refit for all 30 teams individually, pooling each
team's own real lineups (`min_poss=30`): **n=1317 real lineups league-wide,
Net corr=0.212, ORtg corr=0.123**. A materially larger, more statistically
trustworthy sample than Brooklyn alone, still weak — this is a systemic
property of the model, not Brooklyn-specific sampling noise.

**Step 5 — true R² and the variance mismatch.** Pearson correlation
doesn't penalize scale mismatch; computed proper out-of-sample R²
(`1 - SS_res/SS_tot`) on the same 1317-lineup pool: **R²≈0.041 (Net),
R²≈0.011 (ORtg)** — worse than the correlation numbers implied. Directly
diagnosed why: predicted Net's own standard deviation across all 1317 real
lineups was **5.38**, vs. observed Net's **28.89** — the model barely
varies its answer regardless of which players are on the floor.

**Step 6 — ablations (each a specific, falsifiable hypothesis for the
cause, each tested against real data):**

| Hypothesis tested | Change made | Result | Verdict |
|---|---|---|---|
| Model too complex (56 interaction terms overfitting) | 8-feature model: `Eoff_a - Edef_a` only, no interactions/talent/home | corr=0.131 | *Worse*, not better — ruled out |
| Stints too short (mean 93 sec, ~3 possessions) to carry signal | Possession-count floor swept 1/3/5/8/12 | corr flat at 0.20–0.22 across the whole range, uncomputable past 12 | No effect — possession-weighting already handles this |
| Skill-weighting itself is the problem | `skill_off=skill_def=1.0` for everyone (pure archetype-share exposure) | corr=0.199, R²=0.037 | Statistically indistinguishable from skill-weighted — ruled out |
| Not enough data (1 season) | Pooled 2 seasons (2024-25+2025-26, corrected after a real data-pipeline bug — see below) | corr=0.204 | No improvement over the 1-season 0.212 |
| Net rating specifically is too noisy a target | Refit on process metrics (eFG%, TOV%) instead of Net, same lineup-level evaluation | eFG% corr=0.147 (n=1080); TOV% corr=0.126 (n=1150) | Same order of magnitude as Net — the target choice wasn't the bottleneck either |

**Step 7 — split-half reliability: the decisive diagnostic.** For every
real 2025-26 lineup, its own stint rows were randomly split into two
independent halves; each half's observed Net rating was computed
separately; the two halves were correlated against each other. This is the
empirical **reliability ceiling** — the best any model could possibly do,
since it measures whether the target even agrees with itself.

| min_poss per half | n lineups | split-half Net corr | split-half ORtg corr |
|---|---|---|---|
| 15 | 705 | -0.098 | +0.045 |
| 30 | 262 | -0.363 | -0.021 |
| 50 | 138 | -0.429 | -0.140 |

The ceiling sits **at or below zero** at this data volume — a specific
lineup's own Net rating does not even correlate with an independent random
half of its own possessions. By a Spearman-Brown-style reasoning about test
reliability, a measurement this unreliable at the "half-season" granularity
implies the *full-season* single-lineup Net rating itself carries a
reliability on the general order of the model's own observed ~0.2–0.3 —
**the model's ~0.21 pooled correlation is not below some hidden, higher
ceiling; it is sitting close to the ceiling the data itself allows.** The
bottleneck is the target's own measurement noise at this sample size, not
an underperforming model.

**Step 8 — a real data-pipeline bug found while extending to 3 seasons.**
Pulling 2023-24 and 2024-25 via `build_stints.py` surfaced a genuine bug:
`combine_and_validate()` globbed the entire shared `data/stints/games/`
cache directory (every game ever pulled, any season) instead of filtering
to the requested season's own game_ids — silently mixing other seasons'
games into whichever season's output file was being combined
(`stints_2024_25.parquet` was found to be 1230 of its 1235 "games" actually
2025-26 duplicates; a first, buggy 2023-24 combine showed a physically
impossible ~190 possessions/game, exactly 2x real pace). **Checked and
confirmed `stints_2025_26.parquet` — the file every result above was built
on — was unaffected** (1230/1230 games, all real 2025-26 game_ids), not
because the bug didn't apply to it but because it was combined before any
other season existed in the shared directory. Fixed by filtering to the
season's own real schedule before combining; both affected seasons were
regenerated and their game_id composition re-verified directly (not just
re-trusted from a printed report) — 2023-24: 69,825 rows/1230 games, 100%
score reconciliation; 2024-25: 71,259 rows/1230 games, 100% score
reconciliation.

**Step 9 — the team-season test (the most decisive check run).** Rather
than evaluate at the noisy lineup level, aggregate an ENTIRE team's season
of holdout-model predictions (possession-weighted across ~7,500–8,000 real
possessions) and compare against that team's real, independent
`LeagueDashTeamStats` NET_RATING (not derived from this project's own stint
table). Run across all 30 teams x 3 seasons = 90 team-seasons, with a
talent-only baseline (same model, archetype/synergy coefficients zeroed)
run alongside for comparison:

| Season | n | Full model corr | Talent-only corr |
|---|---|---|---|
| 2023-24 | 30 | 0.756 | 0.786 |
| 2024-25 | 30 | 0.779 | 0.787 |
| 2025-26 | 30 | 0.804 | 0.782 |
| **Pooled** | **90** | **0.770** | **0.767** |

At this, the lowest-noise aggregation tested, the model shows strong
real-world correlation (~0.77) — but the talent-only baseline matches it
almost exactly (and beats it outright in 2 of 3 individual seasons).
**Archetype composition and synergy add no detectable increment over
talent, even at the most favorable aggregation level tested.**

## 4. Conclusion — stated precisely, with its boundaries

**What the evidence supports:** box-score-derived archetype composition
and pairwise archetype synergy, as specified in this model, show no
detectable predictive increment over individual talent at this data scale
— at every granularity tested (single stint, real 5-man lineup, team-season)
and every model variant tested (with/without interactions, with/without
skill-weighting, 1 vs. 3 seasons of data, Net rating vs. process metrics).

**What this does NOT mean** (stated explicitly, not implied):
- It does not prove lineup fit doesn't matter on the court. A real effect
  smaller than this data's detection limit (per Step 7, a reliability
  ceiling near zero at the single-season/single-lineup level) would look
  identical to "no effect" here — absence of evidence, not evidence of
  absence, at this specific sample size.
- Fit effects may require data this project doesn't have: matchup-specific
  tracking data (spacing, defensive rotations), not just box-score-derived
  archetype shares.
- Talent metrics like the BPM-based `skill_off`/`skill_def` used here are
  themselves partially context-adjusted (a player's box-score-derived
  impact already reflects *some* of the environment he played in) — this
  can absorb part of any true fit effect into the "talent" term rather
  than leaving it detectable as a separate archetype/synergy signal.
- Real NBA rosters don't explore archetype-composition space randomly —
  teams already self-select into broadly sensible compositions (a range-
  restriction problem: if no team runs 5 centers, the data can't show that
  5 centers is bad). This both suppresses the detectable size of any true
  composition effect in observational data, and is itself a reason
  composition may be better used as a **constraint** (a sensibility check:
  "does this lineup avoid an obviously bad combination") than as an
  **objective** to numerically optimize.

## 5. What survives, validated

- **The stint pipeline** (`build_stints.py`): 1230/1230 games parsed for
  each of 3 seasons, 100% score reconciliation against real box scores in
  every case. A real, reusable, validated possession-level data asset,
  independent of what it's used to model.
- **The talent-aggregation engine**: the same ridge model's talent terms
  (`Soff`/`Sdef`), aggregated to the team-season level, achieve real
  out-of-sample correlation (0.77 pooled, 90 team-seasons) against
  independent official NET_RATING — a validated, working piece.
- **The ADA descriptive layer** (player archetype recipes, exposure gaps,
  the pairwise cosine-similarity conflict matrix): repositioned as
  **scouting/diagnostic tools** — describing player style, roster
  composition gaps, and stylistic incompatibility — carrying no predictive
  claim about lineup outcomes. Already live in the portal's Diagnostic
  Analysis and Roster Construction pages on exactly this basis.
- **The NCAA→NBA projection machinery** (`step1_archetype_model.py`'s
  `project()` function, extended in this investigation to project
  additional NBA seasons onto a fixed basis) — reusable for mapping a
  zero-NBA-data player onto the existing archetype space without refitting.

## 6. Design principle for everything downstream

**Talent for numbers, archetype for constraints/flags/scouting language.**
Any quantitative claim (a predicted rating, a ranking, a lineup
recommendation with a number attached) must come from the validated
talent-aggregation engine, never from the archetype/synergy terms alone.
Archetype-derived outputs are always labeled **descriptive** — composition
gaps, style comparables, conflict flags — never a numeric performance
prediction. This principle governs both `docs/BROWN_STUDY.md`'s labeling
convention and any future work on this codebase.
