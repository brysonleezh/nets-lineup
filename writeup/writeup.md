# The Archertype Portal for the Brooklyn Nets

## Why this idea

The idea generated from *"Scouting Anyone: Probabilistic Player Archetypes for Any League"([https://www.sloansportsconference.com/research-papers/scouting-anyone-probabilistic-player-archetypes-for-any-league])* MIT Sloan Sports Analytics Conference 2026. Traditional basketball players are labeled as guard/forward/center or position 1-5. Unsupervised machine learning model can work here - but standard clustering (KNN, K-Means) apply players by **cluser center**. That is backwards for basketball analytics. For example, if a strech big who occassionaly post up, like Chet Holmgren or Jaren Jackson Jr. They doesn't belong to any single average role, and real basketball roles are defined at the **extremes**, not the middle.

Archetypoid Analysis(ADA) provide assistance with this: it finds the K most **extreme real players** (vertices of the convex hull of the player cloud) and express everyone else as a mixture of them - e.g., "31% 3&D Wing + 21% Traditional Playmaker + ...". Every archetype is an actual player, so the output stays readable for coaches and front office.

## What the portal asnwers

Two questions, in seqeunce, all build on ADA recipes:

1. **How do we describe a player?** - ADA on measture role language (The 8 Player Types).
2. **Is a player being used correctly in his role?** - a per-player diagnosis (Player breakdown)

## Step 0 — Data

Four data layers, all pulled and validated from primary sources.

**1. Player-feature table (the ADA input).** Three sources joined per player-season:
NBA.com box/advanced stats (via `nba_api`), Basketball-Reference shot-location profiles
(FGA share by distance, corner-3 rate, % of FG assisted), and NBA.com Synergy play-type
shares (9 categories: isolation, P&R ball-handler/roll man, spot-up, post-up, cut,
handoff, off-screen, off-rebound). After preprocessing — per-100 rates, MIN ≥ 300 floor,
diacritics-insensitive name matching across sources — this yields a **29-dimensional
feature vector per player-season**, for 2023-24 through 2025-26 (~430 players per
season). All cross-source joins resolve to NBA player IDs; names are only used at the
initial handoff, then never again.

**2. Lineup & on-court tables.** NBA.com lineup data at 2–5 man group sizes (shared
minutes, net ratings) powering the teammate-environment diagnostics, plus per-player
on-court net rating and team on/off splits.

**3. Possession-level stint table (built from scratch).** The planned parser (`pbpstats`)
broke — its NBA.com endpoint stopped serving data as of 2025-26 — so we built a
possession parser directly on the current v3 endpoints. It walks every play-by-play
event, tracks the 10 players on court through substitutions (direction inferred from
on-court state, since "SUB: A FOR B" has no fixed in/out order), cross-checks period
openings against observed actions (some between-period lineup changes are never logged),
and counts possessions from real events (made shots, turnovers, final free throws,
defensive rebounds) rather than the FGA+0.44×FTA approximation. Output: **75,587
directed stints** (5 offense + 5 defense IDs, possessions, points) across all **1,230
games**, cached per game and fully resumable.

Validation: 1230/1230 games parsed; **100% of games exactly reconcile with their real
final scores** (getting there surfaced and fixed two real bugs — 5 games with malformed
home/away text, and ~3% of points silently dropped when substitutions occur between
free throws); every row passes the 5-distinct-offense + 5-distinct-defense check. One
documented residual: possession counts run ~5% low (tied to ambiguous substitutions),
which shifts scale but not the relative comparisons the portal uses.

**4. Event-to-stint attribution.** A per-game event table linking each play-by-play
event (shooter, rebounder, turnover, free throws) to its stint, validated by
reconciling per-stint scoring sums. This is what lets the portal measure an
*individual's* usage within lineup contexts (Tab 2's elasticity), not just team totals.

## Step 1 - Fit ADA

- Fit once on 2025-26 (433 players, K = 8, chosen by explained-variance elbow). The 8 archetypoids are real players; each player gets a **recipe** (mixture weights summing to 1).
- Historical seasons (2023-24, 2024-25) are **projected onto the same fixed basis** — never refit — so recipes are comparable across seasons.

## Step 2 - Build up portal

**Tab 1: The 8 Player Types.** The vocabulary. A convex-hull view of the league with the 8 archetypoids as corners, plus the full Nets roster described in recipe terms.

**Tab 2: Player Breakdown.** One player, five diagnostics:
- **Who he is** — recipe + purity/entropy, with a bootstrap stability band (game-level resampling; wide band = small sample, don't over-trust).
- **His signature** — how he deviates from his own archetype's typical profile.
- **How his role changed** — season-over-season drift, its league percentile, and the features driving it.
- **How his environment shapes him** — what teammate types he gets vs. his style's norm, what has actually worked, and whether his own usage expands or shrinks depending on who's on the floor (elasticity, from play-by-play event attribution).
- **Is he used the way he produces?** — two partial recipes (deployment-only features vs. outcome-only features) projected onto the same basis; their gap flags miscasting (e.g., "deployed as a 3&D wing, produces like a combo guard").

**Tab 3: Player Report**
- Exports a one-click PDF scouting report.

## Obstacles
We also built the ambitious version: a possession-level, skill-weighted archetype-RAPM (75,587 stints, exogenous prior-season skill, 56 synergy interaction terms,
leave-one-team-out validation). It failed its own gates, and we escalated systematically: 30-team LOTO (n=1,317 lineups, r = 0.12–0.21), feature ablations (no change), 3 seasons pooled (no change), split-half reliability of single-lineup net rating (≈ 0 — the target is ~90% sampling noise; our r ≈ 0.21 sits at the attainable ceiling), and finally a team-season test on 90 reliable points: full model r = 0.770 vs. talent-only 0.767.

**Conclusion**: box-score archetype composition + pairwise synergy adds no detectable predictive increment over individual talent at this data scale — at any granularity tested. So this portal uses archetypes for what they're validated for — description, diagnosis, structural constraints — and never sells them as an outcome predictor. Every number on every page is labeled as measurement, inference, or assumption accordingly

## Limitations

- Archetype describe style, not quality, K=8
- 