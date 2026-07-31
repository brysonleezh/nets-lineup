# Diagnostic Analysis — per-player report (Sections A-E + PDF export)

Documents the formula and honesty caveats behind every number on the
Diagnostic Analysis page's per-player report, in Streamlit at
`src/portal.py` (`render_player_report` and `render_section_a`,
`render_section_b`, `render_section_c`, `render_section_d`,
`render_section_e`), computed in `src/step2b_player_diagnostics.py`.
Layer 1 (the screening quadrant + table, and the pre-existing
mismatch/teammate-lift machinery reused here as D1/D2) is documented in
`step2_diagnostic_analysis.py` itself, not repeated here.

**Correction:** an earlier version of this doc described the report as
closing with an auto-generated "Scouting Summary" paragraph
(`render_scouting_summary`). That feature was removed from the live page
(not superseded-but-hidden — deleted outright, since it was judged
redundant with the five sections' own verdict text). The report now ends
at Section E. The PDF export (below) reflects this — it has no Scouting
Summary page either, for the same reason.

**Report structure is a three-act narrative**, not the order the
underlying Python functions are named in (function names still match
their *original* content, not their current display letter — a
deliberate choice made when the report was reordered, to avoid a risky
mass rename with no behavioral benefit; see the reordering note at the
bottom of this file):

- **Act 1 — who he is**: Section A (identity + recipe stability), Section B
  (his personal signature within that identity).
- **Act 2 — where he came from, where he plays**: Section C (role drift
  over time + development comps), Section D (how his environment shapes
  him).
- **Act 3 — the verdict**: Section E (is he used the way he produces),
  the report's final section.

**Every number below is descriptive**, per this project's own
talent-vs-archetype design principle (`docs/RESEARCH_FINDINGS.md` §6):
none of it is a validated predictor of wins. The only piece of this
project's modeling shown to have real out-of-sample predictive power is
the talent-aggregation engine (`build_skill.py`), which this report does
not use.

## Phase 0 — per-season recipes (2023-24, 2024-25)

**What**: project each historical season's population onto the FIXED
2025-26 basis (`data/basis_2025_26/basis.npz`) using `step1_archetypes_model.project()`
— never a second ADA fit. A player's role can differ season to season,
which is exactly what Section C measures; fitting a separate basis per
season would confound "his role changed" with "the basis itself moved."

**Gate 0a** (feature availability): every one of the basis's 29 feature
columns — including the 6 BBRef shot-location features and 9 Synergy
play-type shares — was checked directly against `build_nba_side_tables()`
output for both 2023-24 (n=413, MIN≥300) and 2024-25 (n=430): zero missing
values in either season. `build_historical_recipes()` raises loudly
(`RuntimeError`) rather than imputing if this ever fails for a future
season's data.

**Gate 0b** (projection sanity): printed the 3-season recipe of 10
high-minute players present in all 3 seasons (Bridges, Harden, DeRozan,
White, Gobert, Adebayo, Brunson, Jokić, Durant, Shai Gilgeous-Alexander).
All evolve smoothly — no wild archetype jumps — confirming mu/sd/basis
were correctly reused rather than recomputed per season (the specific bug
this gate is designed to catch).

Output: `data/basis_2025_26/recipes_2023_24.csv`, `recipes_2024_25.csv` —
same schema as the production `recipes.csv`.

## Section A — Who is he? (includes the merged recipe-stability check)

Profile card, archetype column chart, detailed-stats expander — plus
purity/entropy, plus (once run) a game-level bootstrap overlaid directly
on the same chart, not a separate section.

**Purity** = `max(alpha)` — his single largest archetype share.
**Entropy** = `-sum(a_i * log2(a_i)) / log2(K)`, normalized to [0,1] — 0 =
a pure single archetype, 1 = maximally spread across all K=8. They're two
views of the same trait from opposite ends (stated on-page): low purity =
high entropy = a hybrid, multi-role player.

Both are reported with a league percentile (computed once over every
2025-26 recipe player, `league_purity_entropy()`) and a tercile label
(bottom/middle/top third of that same real distribution — not a
hardcoded threshold: `q1, q2 = quantile(distribution, [1/3, 2/3])`).
The gauge bar's fill width is his percentile on that metric (0-100%); the
tick mark is a constant at 50% — the league median in percentile-space,
always exactly halfway by definition — not a second computation.

The caption sentence under each bar is direction-free plain language, not
a bare "X% percentile" number: purity's own percentile-rank already means
"more specialized than X% of the league" directly; "more hybrid than" is
its complement (`1 - pct`). Entropy's percentile-rank is the mirror image
— "more hybrid than X%" directly, "more specialized than" is `1 - pct`.
Both resolve through the same three-way template (hybrid / specialized /
typical), driven by the same tercile bucket already computed for the
label — the sentence differs, the bucket computation doesn't.

**Recipe stability (a game-level bootstrap, lazy-computed behind a "Run
stability bootstrap" button — originally its own Section F, merged into A
so the confidence band sits directly on the recipe it describes).** A
further, disclosed downgrade from full box-score-shape fidelity: fully
recomputing all box-score-shaped features per resampled game (`USG%`,
`AST%`, `TRB%`, `STL%`, `BLK%`, `BPM`) needs each game's TEAM and OPPONENT
totals (shared-minutes-weighted usage/rebound/assist-rate formulas) —
buildable in principle from the per-game box cache
(`data/stints/raw_cache/*_box.parquet`), but judged too heavy for one
subsection of the report.

**What's actually bootstrapped**: only the 4 features whose standard
formula needs just the player's OWN per-game counts, no team denominator:

- `TS% = PTS / (2*(FGA + 0.44*FTA))`
- `FTr = FTA / FGA`
- `TOV% = 100 * TOV / (FGA + 0.44*FTA + TOV)`
- `PTS_PER_100` — an analytic APPROXIMATION, not a re-derivation of true
  pace-adjusted scoring rate: his real season `PTS_PER_100` is scaled by
  the ratio of his resampled per-minute scoring rate to his real season
  per-minute scoring rate.

The other 25 features (including `USG%`/`AST%`/`TRB%`/`STL%`/`BLK%`/`BPM`,
all 9 play-type shares, all 6 shot-location features, height) are held
fixed at their real season value in every resample. Games are found via
the already-loaded stint table's own `off_p1..5`/`def_p1..5` roster
columns (handles a traded player's two-team season for free, since game
discovery isn't team-scoped), then read from `raw_cache/*_box.parquet`.
B=500 resamples, 5th-95th percentile band, re-projected through the same
fixed `project()`, then drawn as asymmetric error bars on the SAME
archetype column chart (`archetype_column_chart`'s `err_lo`/`err_hi`
params) rather than a second chart.

**This makes the reported band UNDERSTATE true uncertainty**, on top of
the general "season-level features held fixed" caveat — stated explicitly
on-page (directly under the chart, once the bootstrap has run), not just
here. A low-minute player's wide band is the intended signal, not a bug.
Cached per `(player_id, season, B)` behind the button so it never reruns
on an unrelated widget interaction elsewhere on the page. Before the
bootstrap has run, the purity/entropy row shows "Stability: not yet
checked"; after, it reports the top archetype's resampled range plus a
**computed verdict word**, based on the resampled band's width in
percentage points (`hi95 - lo5` for the top archetype, ×100):

| Band width | Verdict |
|---|---|
| ≤ 3pp | "tight" — the reported purity is reliable, not small-sample noise |
| ≤ 8pp | "moderate" — reasonably reliable, with some resampling noise |
| > 8pp | "wide" — treat the reported purity as a rough estimate, not precise |

These three thresholds are the only hardcoded numbers in this section —
picked as round, defensible cutoffs on a 0-100pp scale, not tuned to any
specific player's result.

## Section B — What makes him different from his role?

**Signature radar**: originally compared a player against the centroid of
everyone who SHARES his top archetype (a hard membership bucket). Replaced
after a real failure case: Danny Wolf is 27.9% Combo Guard / 23.6% 3&D
Wing / 20.5% Rim Protector — a near three-way tie (purity percentile
4.8%, more hybrid than 95% of the league) — so bucketing him with
"everyone whose TOP archetype is Combo Guard" compared him against players
who are mostly much more purely guards than he actually is.

**What it does now**: `similarity_weighted_feature_centroid()` computes a
continuous Jensen-Shannon-similarity-weighted centroid over every OTHER
player's full recipe vector — the same weighting convention
`similarity_weighted_benchmark()` (`step2_diagnostic_analysis.py`) already
uses and this project already validated (JS distance, power=4.0), just
applied to raw z-scored features instead of teammate archetype exposure.
A hybrid player is now compared against other hybrids with a similar
overall mixture, not against players who happen to share only his nominal
top archetype label. Effective sample size (ESS) is reported on-page so a
thin, unstable peer group is never silently treated as solid — deliberately
NOT a literal "top-10 nearest neighbors" average, which would have far
higher variance with only ~10 players.

Radar axes = the top ~10 features by absolute value in the peer centroid
(the features that most define the peer group as a whole), plus any
feature where `|his_z - centroid_z| > 1.0` SD (forced onto the chart even
if not top-10, since a large personal deviation is exactly what this
section is for). The top-3 largest deviations are listed in plain
language beside the chart.

**Value vs. role peers: out of scope, not attempted.** No salary/contract
table exists anywhere in this project (checked: no `data/salaries*.csv`,
no salary-shaped column in any of the 8 tables in `nets_synergy.db`), and
CLAUDE.md's own stated project scope explicitly excludes salary/contract
modeling (the same boundary already applied in `docs/BROWN_STUDY.md`'s
Phase B4/B5). No standing on-page note for this (removed per request) —
documented here instead.

## Section C — How has his role changed?

**C1. Role drift across seasons.** His recipe across up to 3 projected
seasons. "Major" archetypes = those exceeding **15%** in ANY season; the
rest are pooled into a single "Minor" line so the chart doesn't clutter
with archetypes he's never meaningfully carried. A player with <2
projected seasons (a rookie, or a veteran who was below the MIN≥300 floor
in earlier seasons — e.g. E.J. Liddell, 349 min in 2025-26 only) shows
"insufficient" honestly rather than a padded/interpolated trend.

**Presentation (revised — was two side-by-side stat cards, one per
transition):** raised as a "this reads as disconnected, and the page in
general is getting cluttered" observation, together with C2's dropdown
below. For each season-to-season transition he actually has (1 for a
2-season player, 2 for a 3-season player):

- A short label (`Δ{magnitude:.3f} · {verdict}`) is annotated **directly
  on the line chart**, centered under the specific segment it describes
  (paper-coordinate placement below the plot area, the same convention
  this app's other charts already use for axis-end labels — so it can't
  land on top of a data line regardless of a given player's archetype
  trajectory).
- A plain markdown line below the chart carries the fuller sentence:
  `**{s_old} → {s_new}**: {verdict} — bigger than {pct:.0%} of the
  league's transitions (n={n_league}){changed-teams tag}.` — the same
  sentence format the PDF export's Section C already used for this
  (Entry 079), now shared by both surfaces instead of the live page using
  cards and the PDF using text.

The underlying computation is completely unchanged from the card version:

- **Magnitude** = Jensen-Shannon distance (base-2) between his own
  archetype-mixture vectors in the two seasons — 0 = identical mix, 1 =
  fully disjoint.
- **League percentile** = where that magnitude falls against every league
  player who has a recipe in BOTH seasons of that specific transition
  (MIN≥300 both seasons) — computed once per transition and cached at
  that granularity (not per player), since the distribution itself
  doesn't depend on who's being viewed.
- **Verdict word** = a tercile label off that same real distribution:
  bottom third → "stable role", middle third → "moderate shift", top
  third → "major shift" (never a hardcoded magnitude cutoff).
- **Changed-teams tag** shows when `TEAM_ABBREVIATION` differs between
  the two seasons — a plausible confound (new scheme, not necessarily
  development) surfaced inline rather than silently baked into the
  number.

The section's own caveat, unchanged: a season-over-season change
conflates real development with one season's sample noise and, if he
changed teams, a new team's scheme — read as descriptive, not diagnostic.

**C2. What's driving the drift — now one feature-trajectory line chart,
mirroring C1's form.** Prompt: "in C2 I don't want to use gap chart...
we want to show their features change in 2023-2024, 2024-2025 right?"
(a diverging bar only shows the *delta*, not the actual season-to-season
values), then the full rework spec. Replaced the two per-transition
diverging-bar panels (small multiples, one column per transition) with
ONE line chart: x = the same seasons/order as C1's own chart (visually
aligned above/below it), y = the feature's z-score against the basis's
own **fixed** mu/sd (unchanged standardization rule — never re-fit per
season), one line per feature in the **union of both transitions' top
movers** (deduplicated, plotted across the player's FULL season history —
a feature flagged in only one transition still shows its other season's
point too, via the new `diag2b.feature_trajectory()`), direct-labeled at
each line's own right end instead of a shared legend. A 2-season player's
chart naturally shows 2 points per line; a rookie with no drift history
still shows nothing here (unchanged, section-level "insufficient" gate).
The underlying `dz` computation and top-5-per-transition feature
selection (`|dz| >= 0.3`, `drift_attribution()`) are **completely
unchanged** — this redesign is chart form and text presentation only.

**A real label-collision bug found by rendering the chart and looking at
it, not assumed away.** With up to 9 features in the union (confirmed for
real players — the two transitions' top-5 sets aren't identical), several
lines' natural end-points land close enough together that placing each
label at its exact data position produces overlapping, unreadable text.
Fixed with `_declutter_label_positions()` — a greedy vertical spacer
(walks features lowest-to-highest, pushing each label up just enough to
keep a minimum gap from the previous one; only the LABEL position moves,
the line/marker stays at its true data value) with `min_gap` set as a
fraction of the chart's own y-range (~9%, tuned by rendering several real
players' charts, not guessed once) so it scales sensibly regardless of
how spread out a given player's data happens to be. The y-axis range is
then widened to comfortably fit the adjusted label positions, not just
the raw data, so nothing renders clipped.

**A second real issue, same discipline:** the union can exceed the
8-color validated categorical palette (confirmed — 9 features is common
enough). Per this project's own dataviz convention ("a 9th series is
never a generated hue"), the 9th+ feature keeps its recycled color but
switches to a dashed line, so it stays visually distinct even though each
line already carries its own direct label regardless of color.

Below the chart, the text is **merged into one block** (previously two
side-by-side captions, one per transition column, each independently
repeating the same closing caveat): one sentence per transition — each
kept feature still tied to **that specific transition's own** top rising
archetype R and top falling archetype F (`argmax`/`argmin` of the
alpha-vector delta for the two selected seasons — deliberately NOT
`role_drift`'s first-to-last riser/faller, which can span a different
window) via whether the feature's movement points toward R's real
archetypoid z-profile (`fit["basis"][R]`) or away from F's, stating a
toward-R/away-from-F clause **only when that feature's own sign actually
supports it** — then the "one lens, not causation" caveat **once**, not
once per transition.

**BUGFIX — a real, visible percentage double-scaling error**, found in
production: the attribution text rendered "Usage rate (2020% → 3050%)"
instead of "20.2% → 30.5%". Root cause: `format_raw_feature_value()` had
only two buckets — `NON_FRACTION_FEATURES` (plain numbers: `PTS_PER_100`,
`BPM`, `Dist.`, `PLAYER_HEIGHT_INCHES`) and "everything else" (formatted
as a 0-1 fraction, `.0%`, i.e. multiplied by 100). Six features — `USG%`,
`AST%`, `TOV%`, `STL%`, `BLK%`, `TRB%` — are actually **already stored as
percentage points** (e.g. `USG%` mean ≈19, range 8.5–38.1 in the real
2025-26 population — checked directly, not assumed), so they fell into
the fraction bucket and got multiplied by 100 a second time. Fixed by
adding a third bucket, `ALREADY_PERCENT_FEATURES`, formatted as `.1f}%`
(no re-multiplication). This project has hit this exact class of bug
before — `render_player_stats_tab`'s own `adv_defs` in `portal.py` had
already classified these same 6 features correctly for a different table
on the page; this formatter just never matched it. Every one of the 29
basis features' real population values were checked directly (min/max/
mean/median) before finalizing the three buckets, rather than assuming
one rule fits all — `TS%`, `FTr`, the 5 shot-distance shares, the 2
assisted-shot shares, and the 9 play-type shares are all confirmed genuine
0-1 fractions (correctly multiplied by 100); `PTS_PER_100`/`BPM`/`Dist.`/
`PLAYER_HEIGHT_INCHES` are confirmed raw ratings/measurements (correctly
never treated as a percent at all). Fixed in the ONE shared formatter, so
both the chart's hover and the attribution sentences are corrected
together, not separately.

**C3. Development comps**: find players whose recipe, at the SAME age the
subject is now, in whichever of our 3 seasons they were that age, is most
similar (Jensen-Shannon distance) to the subject's CURRENT recipe — then
show what those players' recipes became the following season in our
window. With only 3 projected seasons total, the lookahead is at most 2
seasons past the matched one; the page states this limit explicitly
rather than implying a full career arc.

## Section D — How does his environment shape him?

The first two sub-parts are `step2_diagnostic_analysis.py`'s existing
`mismatch_score()` and `teammate_lift()`, unchanged, with the existing
B/C-agreement verdict (that "B"/"C" naming is `compute_bc_verdict()`'s own
evidence-type shorthand — benchmark gap vs. context/teammate lift — a
pre-existing convention independent of this report's A-E section
lettering, not renamed when the report was reordered).

**D3. Does his game change with who's on the floor? — a real individual
measurement, not a team-level proxy.** This used to be a disclosed
downgrade (a team-level ORtg proxy, since true individual usage
attribution needed event-level play-by-play boundaries this diagnostic
didn't build). That gap is now closed in two phases:

**Phase 1 — event-to-stint link (`build_stints.py`, additive-only).**
`build_game_stints()`'s existing per-event walk (the same loop that
already resolves each shot/turnover to a team for the stint-level
`off_fga`/`off_tov`/etc. counters) now ALSO emits one row per attributed
event — Made/Missed Shot, Turnover, Free Throw — tagged with the
`stint_id` it belongs to, `personId`, `teamId`, `shotDistance` (shot
events only), and an `assistPersonId` parsed from the play description's
free-text "(Name N AST)" suffix (PlayByPlayV3 has no structured assist
column — resolved to a real player id via the same name-matcher
`build_game_stints`'s own substitution parser already validates).
Written to `data/stints/events/{game_id}.parquet` (one file per game,
resumable independent of the existing `games/{game_id}.parquet`) and
combined into `data/stints/events_2025_26.parquet` (310,863 events across
all 1230 games). **Validated**: per-stint summed scoring-event points
against the existing table's own `off_points` — 99.12% of directed rows
match exactly; the residual 0.876% is a real, pre-existing anomaly in the
already-shipped stint table (every mismatch co-occurs with `_flush()`'s
own documented "floored to 1 possession" free-throw-across-a-substitution
edge case, and nets to zero over a full game — score reconciliation is
still 100.0% exact for all 1230 games). Not repaired here: doing so would
mean changing the existing stint table's own scoring logic, which this
phase's own "additive only" constraint rules out.

**Phase 2 — individual elasticity, computed from real events.** For each
of the 8 context archetypes, the same median-exposure split as before
(his own on-floor stints split into HIGH vs. LOW by his 4 teammates'
combined archetype-share, weighted throughout by real possession counts,
not stint counts) — but the metric on each side is now **his own**
behavior, pulled from the Phase 1 event table:

- **Usage proxy** = his `(FGA + TOV + 0.44×FTA)` ÷ his TEAM's same total
  while he's on the floor (the standard usage-rate numerator structure,
  computed as a real share of real team events, not estimated).
- **Assist rate** = his assists (on any teammate's make) ÷ his team's
  total made shots, over the same stints.
- **Rim share / 3PT share** = his own shots within 3 ft. of the rim (this
  project's own "0-3" shot-distance bucket boundary) / from 3, as a share
  of his own FGA — available now specifically because Phase 1 closed the
  shot-distance-to-stint gap; the old team-proxy version's shot-mix gate
  is superseded, not deleted (see below).

The profile chart's one bar per archetype is the **usage-proxy delta**
(HIGH minus LOW, in percentage points) — this replaces the old ORtg delta
entirely, since ORtg-by-teammate-context duplicated D2's own question
(does the team perform better around him) rather than answering D3's
real one (does HE change). Assist rate/rim share/3PT share move to the
drill-down. Same possession floor as before (500, checked against real
summed `off_possessions`, never stint count) with the same redundant
color+pattern-fill low-confidence flag; same fixed-threshold positive
"plug-and-play" verdict framing, now keyed to a **3.0 percentage-point**
usage-share threshold (a disclosed, fixed cutoff — a like-for-like
analogue of standard USG% swings considered meaningful, not derived from
a league distribution).

**Copy simplification (display only, no computation change).** The dense
method paragraph is no longer always-visible — the standing caption is
one plain line ("Usage = his share of the team's shots while he's on the
floor. Each bar: how that share changes when one type of teammate is
heavily on the floor with him."), with the full individual-measurement
description moved into a collapsed "How this works" expander (same text,
same convention as Section E's own expander). The chart's x-axis has no
technical title anymore — two plain-language paper-coordinate annotations
replace it ("← cedes the ball" / "takes over more of the offense →"); the
technical framing lives in the expander and in hover text instead.

**Elastic/rigid chip.** A one-word badge next to the D3 header, computed
from the **spread** of well-supported usage deltas — `best usage_delta −
worst usage_delta` among non-thin archetypes — against
`ELASTIC_SPREAD_THRESHOLD_PP` (**6.0pp**, a disclosed, fixed cutoff, double
the single-shift 3.0pp bar the auto sentence itself uses, since a
meaningful single riser or faller necessarily implies at least that much
range end-to-end). "Elastic" = scales his role to his teammates; "rigid"
= same role regardless, explicitly framed as a *positive* ("a plug-and-
play profile") for a low-usage player, not a deficiency. A single shared
function (`_elasticity_verdict()`) computes this ONCE; both the chip and
the auto sentence below the chart read from that one result, so they can
never disagree with each other.

**D1-comparison note, reworded to one fixed line** (no longer computed by
comparing D3's vs. D1's top archetype): "Note: D1 asks what environment
he GETS; this asks how he RESPONDS to it - the two can point at different
archetypes without contradiction." A general framing note, not a
per-player comparison — the `d1_diff` parameter the old comparison logic
needed was removed from `_render_role_elasticity()` entirely (confirmed
via grep it had no other caller) rather than left unused.

**Drill-down removed; its evidence moved into the main chart's hover.**
The "Drill down into one context archetype" expander is gone. Each bar's
hover now carries what it showed: both sides' raw usage-proxy values,
the assist-rate delta, and both sides' possession counts — e.g. "usage
29.8% → 24.1% · assist rate +0.4pp · 1,180 vs 1,240 poss" — Python-
rounded strings in Plotly customdata, this file's established hover
convention. Hatching/greying below the 500-possession floor is unchanged,
still driven by the same real possession counts (never stint counts).
Checked directly against the code before removing the expander: its own
metrics were already individual (usage_proxy/assist_rate/rim_share/
three_share) from the Phase 2 rebuild, not the old team-level ORtg/TOV%
stats — so no additional inconsistency note was needed.

**Sanity check** (run directly against the computation before any UI was
built): Luka Dončić (2025-26, 1407 on-floor stints) shows a **negative**
usage-proxy delta on both on-ball-creator-adjacent contexts — Offensive
Engine (−8.4pp) and Combo Guard (−8.4pp) — with both sides of both splits
comfortably above the 500-possession floor (1850–2954 possessions),
confirming a real ball-dominant guard's own usage share shrinks, as
expected, when paired with another primary shot-creator. A low-minute
player (E.J. Liddell, 349 min) correctly shows every archetype thin/
hatched rather than a false confident reading.

**Superseded, not deleted.** The old team-level `role_sensitivity_profile()`,
`sensitivity_profile_chart()`, and `_role_sensitivity_verdict()`
(step2b_player_diagnostics.py / portal.py) are untouched but no longer
called from the live page — this project's standard "kept but not wired
up" convention (see `SHOW_STABILITY_BOOTSTRAP`). The old
`shot_mix_gate_status()` now checks for the Phase 1 events file directly
and reports available once it exists, rather than re-deriving its own
now-outdated per-column check.

## Section E — Is he being used the way he produces?

Splits the basis's 29 features into two blocks:

- **Opportunity** (deployment — how he's used): `USG%`, `Dist.`, the 5
  `% of FGA by Distance_*` shares, `Corner 3s_%3PA`, the 2 `% of FG Ast'd_*`
  shares, and all 9 `PLAYTYPE_*` shares (19 features).
- **Outcome** (production — how well it works): `PTS_PER_100`, `TS%`,
  `AST%`, `TOV%`, `STL%`, `BLK%`, `TRB%`, `FTr`, `BPM` (9 features).
- **Neutral, excluded from both**: `PLAYER_HEIGHT_INCHES` (1 feature) —
  a fixed physical trait, neither a deployment choice nor a production
  outcome. This is a disclosed judgment call, not an oversight.
  `Dist.` (average shot distance) is treated as opportunity/shot-selection,
  not outcome — also a disclosed call, arguable either way.

For each block, a partial recipe is projected by taking the player's real
feature row and replacing every OTHER feature's raw value with the
basis's own population mean for that feature (which z-scores to exactly
0 — a neutral, non-separating contribution), then calling `project()`
unmodified. **This computation is unchanged from Day 1** — the redesign
below is presentation, naming, and auto-text only.

**Header.** The dense feature-split explanation above is collapsed into a
"How this works" expander (same text, unchanged); the always-visible line
is one plain-language framing: "Two portraits of the same player: how his
team USES him vs. what his PRODUCTION looks like — do they agree?"

**Three verdict cards, above the chart.** Computed, house-style tiles
(the same label/value visual language as every other stat card in this
report, without a percentile bar — these are a name+share pair, not a
percentile position): "Used as: {his highest-share archetype in
role-as-used} ({share})", "Produces like: {his highest-share archetype in
role-as-productive} ({share})", "Biggest untapped: {archetype with the
largest positive productive-minus-used gap} ({gap:+.1f}pp)". When the
used-top and productive-top archetypes are the SAME, the first two cards
merge into one "Used and produces as: {X}" card (showing both shares in
its subtitle) so the report never shows two cards naming the identical
archetype.

**Main chart — the dumbbell, not the gap chart.** Two connected points per
archetype (all 8, no cutoff, sorted by `|gap|` descending) — **role-as-
used** and **role-as-productive** as two dots on the same axis, joined by
a line, so both raw shares are visible at once, not just their delta.
`dumbbell_chart()` itself is unchanged from when it was first built; only
its position changed. This is a deliberate swap away from a 4th
horizontal-bar chart in the same report (D1/D2/D3 are already bars) —
raised as a "this page is getting visually monotonous" observation, and
this was the section picked to change because it's the one place a
two-endpoint chart form is a *better* fit for the underlying question
(two portraits of the same player), not just a different-looking one.

**Gap chart — kept, moved into a collapsed expander** ("See the
usage-vs-production gap chart") for readers who want the delta read as a
single sorted, signed bar rather than two dots. Unchanged from the
gap-first redesign: one diverging bar per archetype, value =
**role-as-productive minus role-as-used** (in percentage points), reusing
the exact same `diverging_bar()` component D1 and C2's drift-attribution
chart use. Green = positive (production exceeds usage — "untapped"),
coral = negative (usage exceeds production — "over-deployed"), sorted by
the signed value so the biggest untapped direction reads at the top and
the most over-deployed at the bottom. Axis-end annotations replace a
technical axis title with plain language: right = "untapped — production
points here more than usage", left = "over-deployed — usage exceeds what
production supports". Hover shows the real before/after values ("used 2%
→ produces 13%"), Python-rounded, alongside the signed gap.

**Filtered by gap size — not every archetype is drawn in this chart.**
Only archetypes where `|productive − used| >= 3pp` (`GAP_CHART_THRESHOLD_PP`,
a disclosed, fixed cutoff) get a bar. The omitted ones are summarized in
one caption line below the chart: "The other {n} archetypes are aligned
within ±3pp: {names}." **The filter is on gap size alone — never on
whether an archetype is the player's own top archetype in either
projection** — an archetype that's small in usage but large in production
(the textbook "untapped direction" this chart exists to surface) always
survives regardless of its absolute share. If fewer than 2 archetypes
clear the threshold, the top-3 by `|gap|` are shown anyway with an
explicit note that every gap is small — an "everything aligned" player is
a real, valid finding, not an empty chart. (Verified against 120 real
2025-26 players: this edge case did not occur in that sample — with 8
archetypes summing to 1.0 in both projections, "every gap under 3pp" is a
real but rare state — so its correctness was confirmed with synthetic gap
vectors directly, including a dedicated case where a non-top,
near-zero-usage archetype has a large production gap, to prove the filter
cannot drop it.)

**Miscast score → a 3-tier verdict** (rendered below the chart/expander;
replacing the old single score + one sentence):

1. **Overall verdict** — a real tercile split of the league-wide JS-
   distance distribution (`tercile_label`, the same utility every other
   tercile verdict in this report uses): bottom third → "well aligned",
   middle third → "typical alignment", top third → "notably miscast".
   **Card removed** (was "Usage-production alignment," headline = the
   tercile word, subtitle = league percentile + raw JS distance) per
   "重复了 保留第二个 miscast risk please" (this repeats [the "Miscast risk"
   card just below it], keep the second one) — once "Miscast risk"
   (Entry 094/095) moved to sit right after this card in the same
   section, both showing the same percentile under different labels read
   as a duplicate. The underlying `pct`/`tercile_word`/`verdict_color`
   computation is unchanged and still feeds "Miscast risk" and the text
   caption below directly - only this one display call was removed.
2. **Top-role flip** (only shown when it's actually true) — his single
   highest-share archetype differs between the two projections:
   "Deployed primarily as a/an {used top}, but produces most like a/an
   {productive top}." (Kept even though the verdict cards above already
   name both archetypes — the cards are a quick visual scan, this line is
   the fuller narrative read; intentionally redundant, not a duplicate
   bug.)
3. **Largest-gap line**, explicitly worded as subordinate to (1) ("Within
   {overall verdict}, the largest untapped direction is...") — the same
   `underused_idx`/`gap_pp` the old caption used, grounded with up to one
   real number per side, restricted to an **intuitive, offense-relevant
   whitelist** (`INTUITIVE_EVIDENCE_FEATURES` — `TS%`, `AST%`, `USG%`,
   `% of FGA by Distance_3P`, `PLAYTYPE_SPOTUP`, `PLAYTYPE_OFFSCREEN`,
   `% of FG Ast'd_2P`, `% of FG Ast'd_3P`). A fixed bug: the original
   version searched ALL of `OUTCOME_FEATURES` unrestricted, which could
   (and did) surface defensive/rebounding stats like steal rate as
   "evidence" for an offensive archetype resemblance — a non-sequitur a
   reader would rightly distrust. Each side independently needs its best
   whitelisted feature to clear `GROUNDING_CONTRIBUTION_THRESHOLD` (0.3 SD
   — the same disclosed cutoff `drift_attribution()`'s own dz threshold
   uses) before being cited at all; the clause renders two-sided,
   one-sided, or is omitted entirely depending on what actually clears the
   bar — never forcing an unintuitive feature into the sentence just to
   fill it.

Honesty note (stated on-page, moved to sit directly under the verdict
block rather than at the top): this is a stylistic-consistency
diagnostic, not proof a role change would improve results. This is the
report's final analytical section — everything above (identity,
signature, drift, environment) is context for this verdict (and, just
above this caveat, the "Miscast score" headline card — see "Bottom line"
below).

## Bottom line (two coach/front-office-facing headline cards)

Prompt: "As for D3 and E, i think a big difference is that I can
generate some ideas to coach or front office... D3: Elastic ability? E:
Miscast? 我的意思这里在结尾可以放两个卡片" (D3 and E can each produce a headline
number a coach or front office could act on — put two cards at the end).

**Placement revised**: originally both cards sat together in their own
standalone section after Section E; moved per "把这两个Bottom Line的两个卡片
Role Elasticity and Miscast Risk放在D3 and E的图下面" (put the two cards under
D3's and E's own charts instead) — each card now lives at the end of the
section it summarizes instead of both being grouped at the report's very
end. The standalone `render_bottom_line_cards` function was removed
entirely (not kept superseded — its exact content was relocated, not
replaced by something different, so keeping a dead duplicate around would
only have been confusing): **Role elasticity** now renders at the end of
D3 (`_render_role_elasticity`, right after D3's own verdict sentence, no
new computation - reuses that function's own `elasticity`/`arch_names`
already in scope); **Miscast score** now renders at the end of Section E
(`render_section_d`, right before its own closing caveat, reusing that
function's own `tercile_word`/`pct`/`verdict_color`/`underused` already
in scope). Each card's own content, computation, and the graceful "Not
enough data" fallback are otherwise byte-for-byte unchanged from the
original combined section described below.

- **Role elasticity (D3)**: value = D3's existing elastic/rigid word
  (`_elasticity_verdict`, unchanged); subtitle names the two archetypes
  driving the spread (`best`/`worst`) and the spread itself in pp. **New
  computation**: a real league percentile for the spread
  (`load_league_elasticity_spreads`) — added specifically so this card
  isn't the only one of the pair without one (E's miscast score already
  had a league distribution; showing D3's card without a percentile next
  to one that has one would have looked inconsistent, and every other
  verdict card on this page follows the same {word} + {real percentile}
  + {raw number} anatomy). Timed before committing to this — computing it
  for one player is cheap, but `individual_role_sensitivity_profile`
  touches season-wide event/stint files, not just the recipes table like
  this page's other league sweeps, so a full league pass measured ~52s.
  **Cached** (`@st.cache_data`) — paid once per season, not once per
  player view (confirmed live: first player view ~96s including the
  sweep, next player view ~2s).
- **Miscast score (E)**: same score/percentile/tercile Section E's own
  "Usage-production alignment" card already computes (zero new
  computation) — but **re-labeled** ("Low/Moderate/High" instead of
  repeating "well aligned"/"typical alignment"/"notably miscast"
  verbatim), so it reads as a distinct bottom-line takeaway rather than a
  duplicate of the card shown just above it in Section E. Same underlying
  tercile bucket, different framing for a different purpose.
  **Renamed from "Miscast risk"** ("我觉得...Risk是不是有点过" — is "Risk" a
  bit much) — the underlying number is a JS-distance between usage and
  production, purely descriptive (this page's own standing caveat already
  says "not proof a role change would improve results"), so "risk" implied
  a validated downside likelihood the metric doesn't establish. "Score" is
  neutral and keeps this project's own pre-existing "miscast" vocabulary
  rather than inventing new terminology - same value, same color-coding,
  label text only.
- Graceful degradation: if D3's elasticity isn't well-supported for a
  player (confirmed on E.J. Liddell — his individual profile computes
  fine, but has no well-supported spread), the elasticity card shows "Not
  enough data" with a plain explanatory line rather than a fabricated or
  zeroed-out percentile.
- Closing caveat, shared by both cards: "stylistic-consistency reads, not
  causal estimates" — matching this project's own standing convention of
  never letting a coach-facing summary imply more certainty than the
  underlying descriptive metric supports.

**Gauge bars: "Median"/"Player" labels on the bar, their real values
below.** Went through several iterations to land here, each driven by a
follow-up clarification or a real bug found in the rendered page:
1. "Please mark it mediam and their value for player and league median
   please" → a plain text row below the bar.
2. "我的意思是标注在轴上 不是单独放在下面" (I meant label it ON THE AXIS, not a
   separate row below) → "Median" moved to sit directly above the tick.
3. "I want their value mark on the bar plot... Median and Player are up
   at the bar, their vaule are below at the bar. And add axis for all
   those bar plots please" → both "Median" and "Player" labels above the
   bar, their values below, plus a 0%/50%/100% axis.
4. A pasted screenshot showed the raw HTML source (literal
   `<div style="...">Median</div>` text) rendering as visible text
   instead of being parsed as HTML — a real production bug, not a
   styling nit. Root cause, confirmed by reproducing it with `markdown_it`
   (the same markdown-parsing package already installed as a Streamlit
   dependency in this environment, not a random substitute): an earlier
   version built the label/value overlay as an indented, multi-line
   f-string spliced into an already-indented spot in the outer template —
   the mismatched-indentation/blank-line seam where the two joined is a
   known CommonMark trigger for "this is an indented code block," which
   prints escaped HTML instead of rendering it. Fixed by rebuilding the
   ENTIRE function's output as flat, single-line concatenated f-strings
   with zero embedded newlines anywhere — not just patching the one spot
   that visibly broke, since the same trap could hide in a future variant
   of this function. Re-verified through the same `markdown_it`
   reproduction that the fixed version parses to real tags, no escaping.
   (A caveat on this test's own reliability: `markdown_it` in Python is
   not a perfect stand-in for Streamlit's actual browser-side renderer —
   it separately flagged an unrelated, already-proven-safe-in-production
   card component as "broken" too, which real screenshots elsewhere in
   this project directly contradict. A PASS is a meaningful signal; a
   FAIL isn't automatically a real bug.)
5. "Role elasticity, their Player text mark at the wrong place of the
   bar plot" — a real screenshot showed "PLAYER" sitting at ~66% while
   the actual fill bar (a 52nd-percentile player) ended at ~52%. Root
   cause: an anti-collision rule deliberately pushed the "Player" label
   away from "Median" whenever they'd land within 16 percentage points
   of each other (52 is only 2pp from 50) — correct for avoiding text
   overlap, but the result reads as the label pointing at the wrong spot
   on the bar, which is worse. Removed the anti-collision push entirely —
   "Player" now always renders at his TRUE percentile; only an edge-safety
   clamp remains (`_clamp_gauge_label_x`, now just `max(6, min(94, ...))`,
   no more "push away from 50" branch), so the label can't render half
   off the card at the very edges. Same message also removed the
   0%/50%/100% axis entirely ("no axis for now").

`_stat_gauge_card()` (shared by Purity, Entropy, Role Elasticity, and
Miscast score) still takes the same two OPTIONAL parameters,
`player_value`/`median_value` — default `None`, so any caller that
doesn't pass them renders unaffected. Purity and Entropy pass their own
real values plus `np.median()` of the already-loaded league distributions
(`load_league_purity_entropy` — no new data pull). Role Elasticity passes
`elasticity["spread_pp"]` and the median of
`load_league_elasticity_spreads`'s distribution; Miscast score passes
`mc["score"]` (the raw JS distance) and the median of
`load_league_miscasting`'s distribution.

Current layout, top to bottom: label → big headline value → the bar
itself (fill + median tick, both always at their true, real positions) →
"Median"/"Player" text labels directly above the bar (Median fixed at
50%, Player at his own true percentile, color-matched to the fill) →
their real raw values directly below, at the same two x-positions → the
section's own verdict sentence. No axis. This HTML/CSS still can't be
visually re-verified in this environment (no browser tool; not a Plotly
figure kaleido can export to check) — every fix above was grounded in
either a concrete reproduction (`markdown_it`) or a direct check of the
generated HTML string's own position values, not assumed correct from
the code alone.

## Report (PDF export)

A per-player PDF "snapshot" of the live report, now generated
automatically at the bottom of the **Diagnostic Analysis** page for
whichever player is currently selected there (`render_report_section`,
`collect_report_data` in `portal.py`; PDF assembly in
`src/player_report.py`) — Section "3. PDF scouting report," right after
Section 2's live A-E write-up. This is the SECOND reversal of this
feature's placement: originally a section at the bottom of Diagnostic
Analysis, then hidden behind a `SHOW_REPORT_SECTION` flag pending a
design pass, then promoted to its own top-level "Player Report" page
(with an independent `st.selectbox`, disconnected from whichever player
Diagnostic Analysis itself had selected) once that pass was done.
Reverted back to embedded-in-Diagnostic-Analysis per "As for player
report, I don't have a new tab for him right now, since it is the result
of diagnostic analysis, you can put it at the end of diagnostic analysis
page for now" — the standalone page's own original reasoning (surfacing
the report without first requiring a Diagnostic Analysis visit) gave way
to a simpler point: the PDF is a condensation of THAT SAME player's
analysis, so an independently-selected player on a separate tab was one
more "which player am I even looking at" question than this needed.
`render_player_report_page` and its selectbox are untouched, kept
reachable by flipping `SHOW_PLAYER_REPORT_PAGE` (now `False`) back to
`True` — the same "kept but not wired up" convention this project uses
for every other hidden page/section (`SHOW_FUTURE_WORK_PAGE`,
`SHOW_C3_DEVELOPMENT_COMPS`). `SHOW_REPORT_SECTION` itself is a
different, now-historical flag — it doesn't exist under either name in
the current code; this new one is unrelated.

**Auto-generated on page load, not button-gated.** Per "when loading the
diagnostic analysis page, it will generate preview of pdf report as
well," `render_report_section` runs the generation logic inline — same
render pass, no button click or rerun needed — the first time a given
player+season has no cached PDF yet
(`st.session_state[f"report_pdf_{player_id}_{SEASON}"]`, unchanged
cache key, so re-selecting an already-viewed player is still free). The
"Regenerate" button remains for a manual rebuild. **Real, measured cost,
re-measured after the rendering-technology swap below, not assumed
carried over**: switching to a never-viewed player (season-wide sweeps
already warm) now costs ≈3.4s end-to-end, down from the old reportlab
pipeline's ≈13.2s — confirmed live via `AppTest`, both numbers measured
the same way. The full first-ever page load (every season-wide sweep
cold) is unchanged at ≈100-108s, since that cost lives in the diagnostic
sweeps, not the report itself.

## Rendering pipeline: Jinja2 + Playwright (replaces reportlab)

Prompt: "For PDF page, please refer to README.md and
reference-1b-briefing.html file I just sent to you, I want this format
please" — an externally-authored, pixel-precise 2-page A4 design spec
(colors/type/spacing/chart-geometry formulas all specified as final) plus
a working HTML/CSS/SVG reference file with real sample data (Michael
Porter Jr.). This replaced the previous reportlab-built one-page report
wholesale, not alongside it — confirmed via grep that the old
`build_one_page_pdf`/`build_pdf` (reportlab) had zero callers anywhere
else in `src/*.py` before deleting them; git history preserves the old
version if ever needed again.

**Why Jinja2+Playwright, not reportlab-drawn to spec.** The reference is
real HTML/CSS/SVG, not a drawing-primitive spec — reproducing it in
reportlab would mean hand-deriving every flexbox/grid layout decision
into manual coordinate math with no CSS engine to lean on. Before
committing to a rendering technology, verified directly in this
environment (not assumed): `playwright`'s Python package was already
installed AND its Chromium binary was already downloaded and launched
successfully; the *actual* `reference-1b-briefing.html` file, rendered
through `page.pdf(format="A4", print_background=True)`, produced a real
2-page PDF at 794.6×1123.8px/page (matches the spec's "794×1123px"
almost exactly — real A4 math) in 1.6s, and looked highly faithful when
converted to PNG and inspected directly. `jinja2` was also already
installed. Real font files (Lato 400/700/900) were fetched once via
`requests` (bundles its own CA bundle — bare `urllib.request` fails in
this specific environment with a local certificate-store issue, a red
herring, not a real network block) and vendored into
`src/report_template/fonts/` as self-hosted `.woff2` files, embedded as
base64 `@font-face` data URIs — no live network dependency at
PDF-generation time.

**Architecture — same one-directional dependency as before, same CORE
RULE.** `portal.py` imports `player_report.py`; `player_report.py` never
imports `portal.py`, still has no Streamlit dependency. `collect_report_data()`
(`portal.py`) builds the report's data contract — a plain nested dict
matching the spec's own JSON shape (`player`, `season`, `archetypeMix`,
`purity`, `entropy`, `shotMix`, `playTypes`, `neighbors`, `drift`,
`environment`, `miscast`, `boxScore`, `recommendations`, `reads`) — by
calling the *exact same* cached loaders every live A-E section already
uses (`load_mismatch`, `load_teammate_lift`,
`load_individual_role_sensitivity_cached`, `load_miscasting_cached`,
`load_signature_cached`, `load_role_drift_cached`,
`load_drift_attribution_cached`, `load_league_elasticity_spreads`, etc.)
— nothing in the PDF is computed a second way, same discipline as
before. New pieces beyond a straight reshape:
- `shotMix`/`playTypes` weren't in the old report at all — adapted from
  `render_player_stats_tab`'s own `dist_pairs`/`pt_top3` logic (same
  source columns, same `PLAYTYPE_LABELS` map, now hoisted to module
  level so both functions share it).
- `diagnosisLine`, `reads.styleProfile`, `reads.neighbors`, and the 3
  `recommendations` are new prose, but built the same way the old
  `report_summary` paragraph always was — deterministic templates over
  already-computed numbers, never free text. E.g. `reads.neighbors`
  reuses Section B's own "stands out most in X" sentence verbatim;
  `recommendations[1]` (the "worst overlap" lever) is a new function,
  `_build_recommendation_overlap`, that mirrors `compute_bc_verdict`'s
  own well-supported-mass + graceful-fallback style, extended to check
  D1×D2×D3 agreement instead of just D1×D2.
- The radar's axis set is **kept dynamic**, not forced to the reference's
  literal 11 named axes — confirmed directly that `player_signature()`'s
  `axis_idx` genuinely varies in length per player (default top-10 by
  loading strength, +more via a "forced_in" rule for any feature
  deviating >1.0 SD) - not a fixed 11. Forcing a fixed 11 would mean
  either a second, parallel feature-selection method just for the PDF
  (diverging from what Section B already shows this same player on the
  live page) or padding in features that aren't actually his most
  distinctive ones. Confirmed with the user before implementing (real
  players do render with axis counts other than 11 - e.g. Julius Randle
  gets 9, Michael Porter Jr. gets 11).

**`player_report.py`'s new job: template render + PDF export, nothing
else.** `render_report_html(data)` renders `report_template/report.html.jinja`
via Jinja2 (custom filters: `signed` for typographic-minus-signed numbers,
`ordinal`, `abbreviate_archetype`). `build_pdf(data)` launches Playwright
Chromium, `page.set_content()`s the rendered HTML, `emulate_media("print")`,
and calls `page.pdf(format="A4", print_background=True)`. Chart SVGs
(radar/diverging-bars/drift-lines/dumbbell) are built by the new sibling
module `src/report_svg_charts.py` — pure geometry functions, no Jinja2/
Playwright/Streamlit, independently testable. **Headshots got simpler,
not just different**: `hull_callout_chart.get_headshot_data_uri()`
already returns a ready `data:` URI for both the real-photo and
graceful-SVG-initials-fallback cases (confirmed - it never returns a bare
URL or `None`), so the template just needs one `<img>` tag — no PIL
circular-crop code, no reportlab `Circle`/`String` fallback drawing; that
whole code path is gone, not relocated.

**Chart geometry — every formula verified numerically against the
reference's own hardcoded SVG/CSS coordinates before being trusted, not
derived from the prose spec alone.** For Michael Porter Jr.'s real sample
values: computed radar polygon points against the reference's literal
`<polygon points="110,72.9 131.1,77.2 ...">` (max error 0.08px, pure
rounding); computed D1/D2/D3 diverging-bar widths/positions against the
reference's own inline `width:`/`left:` percentages (exact match); computed
the dumbbell chart's dot positions and the drift line chart's y-mapping
the same way (both exact). Two exceptions, disclosed in
`report_svg_charts.py`'s own docstring: axis-label placement on the radar
and end-of-line labels on the drift chart are hand-tuned per-label in the
reference (confirmed — no single formula reproduces them), so this module
uses a clean generic formula (fixed radius past the outer ring, anchor
chosen by which side of the circle the axis falls on) instead of chasing
an unformalizable target — this is also required by the radar's dynamic
axis count, which the reference's one fixed-11 sample never had to solve
for.

**Real bugs found by rendering real players and looking at the output,
not by reasoning about the code** (this project's own standing
discipline, applied here across 7 real players — Michael Porter Jr.,
Julius Randle, Keon Ellis, Moritz Wagner, E.J. Liddell, Chaney Johnson,
Tyson Etienne):
- The radar's axis labels initially used per-side `text-anchor`
  ("start"/"end"/"middle") based on which half of the circle the axis
  fell on — reasonable-looking code, but it clipped wide labels like
  "SCORING" against the SVG viewBox edge on the right side. The
  reference's own labels turned out to all use `text-anchor="middle"`
  uniformly (misread on first pass) - switched to match, clipping gone.
- The top-diffs bar width divided a possibly-negative raw SD value by the
  #1 entry's raw value with no `abs()` — invisible in the sample data
  (all 5 of MPJ's top diffs happen to be positive) but would render a
  negative, invalid CSS width for any player whose top deviations include
  a negative one. Fixed before it could actually bite a real player.
- Several prose-assembly f-strings used Python's native `{:+.1f}`
  formatting, which renders a plain hyphen for negative numbers, not the
  typographic minus (−, U+2212) the rest of this report's chart labels
  use — most visibly on `alpha_delta_F_pp` (the "away from" archetype's
  share delta), which is *always* negative by construction (it's an
  `argmin`), so it *always* rendered as a plain hyphen. Fixed by routing
  every signed display value through `report_svg_charts.signed()`.
- `miscasting_feature_grounding()`'s `opportunity_deficit` is a positive
  magnitude by construction (kept only when it clears a positive
  threshold) — an early draft displayed it as `+0.5 SD`, which reads
  backwards for something described as a "shortfall." Fixed to match the
  wording convention the live page's own Section D/C already uses for
  this same value ("X.X SD **below** a typical Y" — unsigned, with the
  direction carried by the word "below," not the sign).
- Play-Type Usage's "Other" bucket reused `render_player_stats_tab`'s own
  `other_pct` definition (1 − sum of all 9 tracked Synergy play types,
  i.e. "share outside this project's tracked taxonomy") — correct for
  that function's own pie chart, wrong here: the new stacked bar's
  "Other" needs to be 100% − top3 so the bar visually fills to 100%,
  matching the spec's own sample data (which sums to exactly 100.0
  across its 4 bars). Caught by rendering a real player and noticing the
  bar stopped short of the end.
- The D2 "wins/loses next to" caption was built from an independent
  shared-minutes-mass filter, separate from the top-3/bottom-3-by-raw-lift
  selection the D2 bars themselves render. For a thinner-minutes player
  (E.J. Liddell) these two selections diverged — the caption named
  archetypes ("Trad. PM", "Combo Gd.") that weren't even in the chart
  directly above it. Fixed by having the caption describe the same
  `lift_rows` the bars already show, so they can never disagree.
- `role_drift()`'s `available_seasons` is a list (e.g. `["2025-26"]`) —
  an early version interpolated it directly into an f-string, rendering
  the literal Python repr (`"only ['2025-26'] projected"`) on any
  short-history player's PDF. Caught by rendering E.J. Liddell and
  reading the actual page text.
- A caption-building helper unconditionally appended a trailing period,
  which produced a double period whenever the sentence's last word was
  already an abbreviation ending in one (e.g. "...loses next to Combo
  Gd..") — fixed to check first.

None of these would have been caught by unit-testing the formulas in
isolation; all were found by generating a real PDF for a real player and
reading the actual rendered text/geometry.

**UI behavior — unchanged.** "Regenerate" rebuilds the PDF (spinner) and
re-caches it in `st.session_state[f"report_pdf_{player_id}_{SEASON}"]`.
"Download PDF" (`st.download_button`) serves the cached bytes, filename
`{player_name}_diagnostic_report_{YYYY-MM-DD}.pdf`. If Playwright's
Chromium isn't available in a given environment (the one-time
`playwright install chromium` setup step wasn't run — see
requirements.txt), `build_pdf` raises `PDFExportUnavailable` with an
actionable message, caught by `_generate_report()` and surfaced through
the exact same `{"error": ...}` → `st.caption(...)` path this section
already had for the "no recipe found" case — no new UI code needed.

**Preview mechanism — went through 3 attempts before landing on one that
actually works in a real browser** (each earlier one looked fine from
this environment's own tools, which have no browser access — the actual
failures only surfaced when the user tried the real page):
1. `st.pdf` — tried first, always. **In this environment `hasattr(st,
   "pdf")` is True but the required `streamlit-pdf` extra is not
   installed**, so it raises from inside Streamlit's own delta_generator
   — caught by a broad `except Exception` (not `except AttributeError`,
   which would have missed this exact failure mode). `streamlit-pdf` was
   deliberately not added as a new dependency, since this fallback chain
   already anticipates its absence.
2. *(Removed)* A `data:` URI passed directly to `st.iframe`. Chrome
   blocks `data:` URIs used as an iframe `src` for PDF content outright —
   confirmed in the user's real browser ("This page has been blocked by
   Chrome").
3. *(Removed)* A client-side Blob-URL conversion (`components.html` +
   inline JS decoding the base64 into a `Blob`, pointing a nested iframe
   at the resulting `blob:` URL). Still blocked in the same real browser
   test. Both (2) and (3) succeeded at the Python level with zero
   exceptions either time — `components.html`/`st.iframe` return
   successfully regardless of what the browser does with the HTML/JS
   they're given — so the exception-based fallback chain could never
   actually detect either failure and step past it on its own.
4. **What's live now: page-by-page PNG images**, the fallback the
   feature's original spec itself anticipated ("if neither renders
   reliably, show page-by-page PNG previews instead"). At Generate-time,
   the built PDF is rasterized via `pdf2image.convert_from_bytes` (a thin
   wrapper around the `poppler` binaries, `pdftoppm`/`pdftocairo` — a
   system dependency, not just a pip package, noted in requirements.txt)
   at 150 DPI, and each page's PNG bytes are cached alongside the PDF
   bytes in the SAME session-state entry — measured at ~0.43s for a
   3-page report, non-trivial to re-pay on every Streamlit rerun if done
   inside the preview render itself rather than once at generation time.
   The preview then loops `st.image` over the cached pages — a plain
   `<img>` tag, not a PDF viewer, so there's no PDF-specific browser
   restriction left to trigger.

**Validated end-to-end, twice over.** First pass (unchanged from before
the rendering-technology swap): a full-data veteran (Julius Randle) and
the thinnest-data roster player with a recipe, E.J. Liddell — both
generated with zero exceptions, Liddell's PDF correctly showing the live
page's own graceful-degradation text (Section C: "Rookie season — no
drift history available"; D3: every bar hatched, thin-data caveat text)
rather than a blank area or a crash.

Second pass, after the Jinja2/Playwright rewrite: `build_pdf()` called
directly (bare script, real data, no AppTest) for 7 real players covering
every known edge case this project has already identified —
Michael Porter Jr. (baseline), Julius Randle / Keon Ellis / Moritz Wagner
(traded this offseason — real `team` shows MIN/CLE/ORL while `teamColor`
chrome stays fixed Nets black, per the design decision above), E.J.
Liddell (thin-data elasticity), and Chaney Johnson / Tyson Etienne
(drift-insufficient, <2 seasons). All 7 produced a real 2-page PDF with
zero exceptions; every PDF was converted to PNG (`pdf2image`) and visually
inspected via the Read tool's image support, not just checked for "did it
run" — this is what surfaced every bug listed above. Then the full
`streamlit.testing.v1.AppTest` page-navigation regression (no
browser-automation tool is available in this environment) was re-run
across every nav page with zero exceptions, including the Diagnostic
Analysis page where the report auto-generates.

One unrelated, pre-existing bug was found and fixed by this same full
regression pass: the "🎯 Building Around Rookie" nav page (renamed from
"🎯 Rookie Slot Query" at some earlier point) still had its player-selector
scoping check (`if page in ("🎯 Rookie Slot Query",):`) pointed at the OLD
label, so `selected_player` was never assigned and the page threw a
`NameError` on every visit — caught only because this rewrite's own
regression pass happened to exercise every nav page, not because it was
being looked for. Fixed as a one-line label correction; unrelated to the
PDF report itself, noted here only because this is where it was found.

## What was reused vs. newly written

Reused verbatim, no second implementation: `step1_archetypes_model.project()`,
`recipes_frame()`, `load_basis()`, `load_population()`;
`step2_diagnostic_analysis.macro_archetype_exposure()`,
`similarity_weighted_benchmark()`, `mismatch_score()`, `teammate_lift()`,
`screening_table()`, `load_player_pairs()`; portal.py's
`render_profile_card()`, `archetype_column_chart()`, `diverging_bar()`,
`render_player_stats_tab()`, `_build_sortable_table_html()`,
`lift_bar_chart()`, `render_bc_comparison()`/`compute_bc_verdict()`.

New in `step2b_player_diagnostics.py`: Phase 0's projection/gates, and all
section computation (role drift, development comps, role elasticity,
miscasting, signature, bootstrap). New in `portal.py`: the
`render_section_*` functions, `render_player_report()`,
`collect_report_data()`/`render_report_section()` (the PDF export, see
above), cached loader wrappers, and two new chart types
(`signature_radar_chart`, `bootstrap_band_chart` — the latter now also
reachable via `archetype_column_chart`'s own `err_lo`/`err_hi` params).
New standalone module `src/player_report.py`: pure PDF-building
(reportlab layout, kaleido export, PIL headshot cropping) — no analytics,
no Streamlit import.

**On the report's ordering vs. its function names**: the report was
reordered into a three-act narrative (Section A "Who is he" through
Section E "Is he being used the way he produces" — at the time, closing
with a Scouting Summary since removed, see the correction note at the
top of this file) after the sections were first built in a different
order (the
original build order was, roughly, identity → role drift → environment →
miscasting → signature → stability, matching a literal A→F letter
sequence). Python function names in `portal.py` (`render_section_a`
through `render_section_e`) still refer to each function's *original*
content, not its current display letter — e.g. `render_section_e` renders
what the page now labels "Section B." Only each function's own
`st.markdown("#### ...")` call (the actual on-page label) and the call
order in `render_player_report()` changed; every function's internal
computation, arguments, and `st.session_state`/cache keys are unchanged
from before the reorder.

## Rookie Slot Query — Recommended units around him

**Scope note**: unlike every section above, this documents a different
page — `render_rookie_slot_query()` in `portal.py` (the 🎯 Rookie Slot
Query page), not the Diagnostic Analysis per-player report. Grouped here
anyway because that's where the task asked for it, and this project
otherwise has no single "portal features" doc.

Prompt: "Add an execution-level 'Recommended units around him' block to
the Rookie Slot Query, replacing the gap chart as the primary output
(move the gap chart into a collapsed expander 'The environment math
behind these picks' - unchanged)" — full spec in `AI_USAGE.md`. The page
already computed a typical-environment target vector for the user's
hand-set style hypothesis (`similarity_weighted_benchmark`'s
`all_baseline`) and showed a chart of which archetypes that environment
is short on vs. the Nets' current roster. That chart answered "what
*kind* of teammate is missing" at the archetype level; this block answers
the more concrete follow-up — "which *actual four Nets players*, played
together, come closest to providing it" — without changing anything
about how the target itself is computed.

**Computation:**
- **Candidate units**: every 4-man complement from the 16 Nets players who
  clear both this season's MIN≥300 recipe floor and have a real BPM —
  `step4_roster_construction.enumerate_roster_combos(..., combo_size=4)`,
  the same enumerator the Lineup Rankings page's outcome-model path
  already uses (with `combo_size=5` there), reused as-is rather than
  writing a second combo enumerator. C(16,4) = 1,820 combinations,
  confirmed by direct execution (not just the closed-form count) each
  time this ran during development. Each combo's "combined environment"
  is the plain mean of its 4 players' archetype-mixture vectors
  (`arch_0..arch_{k-1}`), the same "mean of alphas" convention
  `compute_team_archetype_profile` already uses for the Nets' full
  roster.
- **Environment match** = 1 − Jensen–Shannon distance (base 2, so it's
  bounded [0, 1]) between a combo's combined environment and the current
  hypothesis's target vector — the primary sort key. Computed with
  `scipy.spatial.distance.jensenshannon` in a plain per-row loop (1,820
  rows in ~0.02s; no vectorized `axis=` form exists in the installed
  scipy version, checked directly rather than assumed, and a loop is fast
  enough here that it didn't matter).
- **Talent** = sum of the 4 players' BPM this season
  (`load_bpm_lookup`, already used by the Lineup Rankings page — reused,
  not reimplemented).
- **Three structural flags**, each a hard combined-exposure check on the
  4-man mean environment:
  - **Rim protection**: combined Rim Protector/Roll Man + Mobile Big
    share < **0.06** → "no rim protection."
  - **Spacing**: combined Shooting Specialist + 3&D Wing share < **0.25**
    → "spacing risk."
  - **Ball dominance**: combined Offensive Engine + Combo Guard share
    > **0.36** → "second on-ball engine — clashes with the hypothesis
    role."

  **None of these three numbers were picked from intuition.** Before
  writing any thresholds into code, the full scoring pipeline was run
  once against a real hypothesis (Mikel Brown Jr., 70% Offensive Engine /
  30% Shooting Specialist) and the real distribution of all three
  combined-exposure measures was read off all 1,820 actual combos:

  | Check | min | p10 | **p25** | median | p75 | p90 | max | threshold |
  |---|---|---|---|---|---|---|---|---|
  | Rim share | 0.000 | 0.044 | **0.061** | 0.100 | 0.149 | 0.197 | 0.289 | < 0.06 |
  | Spacing share | 0.026 | 0.204 | **0.267** | 0.345 | 0.418 | 0.481 | 0.637 | < 0.25 |
  | Ball share | 0.063 | 0.188 | 0.238 | 0.297 | **0.357** | 0.412 | 0.543 | > 0.36 |

  i.e. rim and spacing flag roughly the bottom quartile of the real
  distribution; ball dominance flags roughly the top quartile. At these
  exact thresholds: rim flags 440/1,820 (24.2%), spacing flags 356/1,820
  (19.6%), ball dominance flags 432/1,820 (23.7%), and 935/1,820 (51.4%)
  of all combos are flag-free on all three checks — a sensible,
  non-degenerate split (roughly half the space survives every check;
  each individual check catches a meaningful-but-minority share),
  confirmed by direct computation, not assumed from the round numbers
  alone.
- **Ranking**: flag-free combos first (`any_flag` ascending), then by
  environment match descending, with **near-ties broken by talent**.
  "Near-tie" is defined and computed explicitly, not eyeballed: sort by
  environment match descending, then walk the sorted list keeping a
  moving "group leader" score; a combo starts a *new* tie-group only once
  it falls more than **`UNIT_ENV_MATCH_EPSILON` = 0.005** below its
  group's leader (chained groups, not a fixed-width bucket grid — a fixed
  grid risks splitting two nearly-identical scores that straddle a bucket
  boundary, which chaining from a moving leader avoids). Within each tie
  group, combos are re-sorted by `sum_bpm` descending. 0.005 was chosen
  by reading the real gaps between consecutive combos near the top of
  the same real ranked list — several adjacent pairs there sit well under
  0.005 apart (e.g. 0.0003, 0.0008, 0.0011 between neighbors in the top
  20), while the full observed range across all 1,820 combos under that
  hypothesis was about 0.44 wide (0.48 to 0.93) — so 0.005 groups
  genuinely-indistinguishable fits without conflating scores that are
  meaningfully apart. Confirmed live in the running app: under the
  default Brown hypothesis, the #2-by-raw-fit combo (Sharpe/Randle/
  Minott/Etienne, env. match 0.9225, BPM +2.6) out-ranks the #1-by-raw-fit
  combo (Wolf/Clowney/Randle/Johnson, 0.9262, BPM −4.3) precisely because
  the two are within epsilon of each other and the first has more talent.
- **"Worst flag-heavy unit"** (the contrast card): among flagged combos,
  the one with the **most simultaneous flags** first, environment match
  ascending as the tiebreak — not simply the single lowest-scoring combo
  overall (which might trip only one flag). Under the default Brown
  hypothesis this surfaces Powell/Dëmin/Traoré/Saraf: all three checks
  fail at once (no rim protection, spacing risk, and a clashing second
  engine), env. match 0.50, sum BPM −21.7 — a genuinely illustrative
  "what not to run," not just a mediocre fit.
- **Per-check "key name"** (e.g. "Rim protection: Day'Ron Sharpe"): among
  the 4 teammates, whichever one individually carries the largest share
  of that check's archetype pair, computed from his own recipe row (not
  the 4-man mean) — credit for a passed check, or (for the ball-dominance
  check specifically, the only one of the three where a *specific player*
  is the problem) who the clash is with on a failed check. Rim and
  spacing checks show a key name only when passed (there's no single
  player to blame for an absence); the lone-engine check shows one only
  when it fails ("Second on-ball engine: clashes with —").
- **Caching**: `load_recommended_units_cached()` is `@st.cache_data`,
  keyed on the hypothesis target vector itself (passed as a plain numpy
  array — confirmed directly that Streamlit's cache hasher treats two
  arrays with equal values as the same cache key, not by object
  identity, so recomputation happens exactly when the *hypothesis*
  changes, never on an incidental rerun) plus `recipes`/`k`/`season`/the
  roster list, matching this page's existing `recipes`-unprefixed
  hashing convention (`load_role_sensitivity_cached` etc.) rather than
  underscore-prefixing it out of the cache key.

**UI**: `N_RECOMMENDED_UNITS = 4` top cards (2×2 grid; spec asked for
"3-5"), each showing the rookie's headshot + the 4 teammates' headshots
(`hull_callout_chart.get_headshot_data_uri` for the rookie, who has no
real `PLAYER_ID`; the real NBA headshot CDN URL for teammates, who all
do), environment-match score, total BPM, and the 3-line ✓/✗ checklist
above. One additional full-width contrast card ("What not to run") below
them. A closing caption states plainly: "Ranked by environment fit +
talent under structural constraints - descriptive construction, not a
lineup outcome prediction (see the Future Work page for why outcome
prediction was ruled out)" — the same standing discipline as every other
number on this page, extended to a new feature rather than relaxed for
it. All of this sits *above* the (now-collapsed) `st.expander("The
environment math behind these picks")` holding the original archetype-
gap chart and its headline sentence, byte-for-byte unchanged, and above
the pre-existing "Sensitivity: does the top recommendation hold..."
section, also unchanged and still running on every rerun.

**Verified, not assumed**: exercised end-to-end with
`streamlit.testing.v1.AppTest` (the same tool used throughout this
project for UI checks — no browser-automation tool is available in this
environment) — zero exceptions navigating to the page, across all three
rookies (Brown, Bilodeau, Jefferson; the latter two exercise the "no
documented college-identity default" selectbox branch, not just Brown's),
and after moving the "Primary weight" slider (confirmed the top card's
environment-match value actually changes, 92%→91%, proving the block is
live off the hypothesis sliders rather than frozen from first render).
This process caught one real bug before it shipped: the photo caption
under each headshot used a plain `name.split()[-1]` to get a last name,
which mis-labels every suffixed name as the suffix itself ("Mikel Brown
Jr." → "Jr.", and "Michael Porter Jr." → "Jr." for any card that includes
him) — visible directly in the rendered card HTML pulled out of
`AppTest`'s markdown elements, not spotted by reasoning about the code.
Fixed with a small `_last_name()` helper that strips trailing
Jr./Sr./II/III/IV/V tokens before taking the last remaining word.
