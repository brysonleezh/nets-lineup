# Project Summary — NCAA → NBA Rookie Archetype Translator

*A standalone summary for a reader who has never seen this project. If you
read nothing else in this repository, read this.*

## The problem

The Nets drafted three players in 2026 — Mikel Brown Jr., Tyler Bilodeau,
Joshua Jefferson — who have never played an NBA minute. Every statistical
model the front office already trusts for evaluating NBA players (the
league-wide "archetype" model that classifies every rostered player into
one of 8 statistical roles, used elsewhere in this project for lineup and
roster-construction work) requires NBA data to run. These three players
don't have any. The question this project answers: **given only a
player's college statistics, what NBA role is he likely to play as a
rookie?**

## The approach

**Two archetype spaces, not one.** College basketball and the NBA track
different things — college data (from collegebasketballdata.com) has no
shot-location or play-type tracking, so a college player's statistical
"role" is described in a narrower, 12-feature space than an NBA player's
(29 features). Rather than force college players into the NBA's own
archetype system, we fit a **second, independent 8-archetype model on
league-wide college data** (2016-17 through 2025-26, ~29,000 player-
seasons) using the same statistical method (Archetypal Analysis) as the
existing NBA-side model. Every player — college or NBA — gets a "recipe":
an 8-number mixture describing how much he looks like each of the 8
archetypes in his league's own space.

**A learned translator, not a lookup table.** The naive approach would be
to match each college archetype to whichever NBA archetype looks most
similar by the numbers (cosine similarity on the 12 features both sides
share) and assume a player carries his college recipe straight across.
We built and tested this exact approach (called T4 in the technical
reports) as the baseline the real model had to beat. **It failed badly** —
worse than simply predicting the league-average rookie recipe for every
player. The reason is informative: two archetypes can look similar on the
12 shared features while being genuinely different roles once you account
for the 16 NBA-side dimensions (shot location, play type) that have no
college equivalent at all.

Instead, we built a **learned statistical translator**: a Bayesian
regression model (Dirichlet regression, chosen for producing genuine
probability-mixture outputs rather than an arbitrary score) trained on
**273 real players** who exist in both worlds — every draft pick from
2017 through 2025 who (a) played in the NCAA, (b) has a matched, complete
college statistical season, and (c) went on to earn at least 300 NBA
minutes as a rookie. For each of these 273 "anchors," the model sees his
real college recipe plus 8 draft/college covariates (pick number, age,
years in college, conference strength, efficiency, shot mix), and learns
to predict his real, already-known rookie-season NBA recipe. Testing
confirmed those covariates matter: adding them to the college recipe
alone improved prediction accuracy by ~18-19%, answering the project's
original design question directly.

## The validation

This is the part that matters most for trusting the output. Following a
standard machine-learning discipline (train/test separation, pre-
registration), the 2025 draft class (36 players) was set aside entirely
during model development — never touched, never peeked at, verified by
file-hash at every step — and used exactly once, at the end, as a genuine
test of players the model had never seen. The evaluation plan (which
metrics, what counts as success) was written down and frozen *before* the
test ran.

**Result: the model passed.** On those 36 real, held-out 2025 rookies:

- It named the exact correct top archetype **52.8%** of the time, and had
  the correct archetype somewhere in its top two **69.4%** of the time.
- It beat the "just predict the league average" baseline by a wide
  margin, and beat the cosine-similarity structural-matching baseline
  (T4) by **more than four-fold** on the primary accuracy metric — the
  headline comparison this project set out to make.
- Performance on this genuinely unseen test class was close to what
  internal testing during development predicted, which is itself a good
  sign — it means the earlier testing wasn't fooling itself.

**One thing the model got wrong, and we said so.** The model also
produces a confidence range (a Bayesian posterior interval) alongside
each prediction. Testing that specific claim found it doesn't hold up —
the model is overconfident, and widening the interval doesn't fix it. We
are not showing those intervals anywhere in the final output. Instead,
every prediction ships with (a) a single honest sentence stating the
model's real, measured track record on real held-out data, and (b) a
table of the 5 most statistically similar past players and what they
actually became — a real, concrete range of outcomes in place of a
number that turned out not to be trustworthy.

## Honest limits

- **The model cannot see everything a scout can.** College data has no
  shot-location or play-type tracking, so it cannot reliably distinguish,
  for example, a rim-protecting screen-setter from a mobile switchable
  big, or a spot-up shooter from a movement shooter — both pairs look
  statistically identical in the 12 features available on the college
  side.
- **It has never seen an elite prospect with an injury-shortened final
  college season.** Players like that were excluded from training because
  their final season didn't clear a basic playing-time filter — a real
  and important gap, since that exact profile (a one-and-done star who
  got hurt) recurs in real draft rooms.
- **It answers "what role, if he plays" — not "will he play."** Training
  only included players who actually earned a real rookie role (300+
  minutes); the model has no opinion on whether a given rookie will get
  the opportunity at all.
- **It doesn't know about coaching, scheme, or the roster in front of a
  player.** No statistical model does.
- **The sample is real but not huge**: 273 training examples, 36 in the
  one held-out test. The reported accuracy numbers carry real statistical
  uncertainty (documented with confidence intervals in the technical
  report) — treat them as a track record, not a guarantee.
- **One caveat is data-driven rather than statistically proven**: a rule
  flagging when a player's college profile is unusually one-dimensional
  (his real rookie role may end up even more concentrated than predicted)
  is grounded in exactly one dramatic real example, not a broad pattern —
  disclosed as such everywhere it appears.

## What to check when the season ends

The three predictions were frozen — recorded with a cryptographic hash,
timestamped, committed to version control — **before** the 2026-27 season
began, specifically so no one (including the people who built this) can
quietly adjust the story after seeing what actually happened. A review
script is already written and tested (against synthetic data, since real
season data doesn't exist yet) so that grading these predictions later
uses the *identical* method, metrics, and code that validated the model
in the first place — no new metric gets introduced after the fact just
because it would look more favorable. Once each rookie has played enough
minutes to have a real season (the same 300-minute bar used throughout),
running that one script will show, honestly, whether the prediction was
right.

## Where everything lives

- **Predictions**: `data/projections/nets_rookies_2026.csv`, and one
  readable card per rookie in `reports/rookie_cards/`.
- **The frozen record**: `data/projections/predictions_frozen.json`.
- **Full technical detail, phase by phase**: `reports/phase1_data_report.md`
  through `reports/phase6_deployment_report.md`, each with its own
  worklog of what was tried, what broke, and how it was fixed.
- **What the model structurally cannot see, in full**:
  `reports/college_archetypes.md`'s appendix.
- **The grading script for whenever the 2026-27 season is real**:
  `src/eval/review_2026_predictions.py`.
