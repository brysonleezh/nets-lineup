# NBA Archetype Portal

**Live app:** [nets-lineup.streamlit.app](https://nets-lineup.streamlit.app)

**Bowen Li** · [bowenli@gmail.com](mailto:bowenli@gmail.com) · +1 (323) 630-5773 · [bl-website-peach.vercel.app](https://bl-website-peach.vercel.app/)

## Why this idea

The idea comes from *“Scouting Anyone: Probabilistic Player Archetypes for Any League”* ([MIT Sloan Sports Analytics Conference 2026](https://www.sloansportsconference.com/research-papers/scouting-anyone-probabilistic-player-archetypes-for-any-league)). Traditional basketball language reduces players to guard/forward/center or positions 1–5. Standard clustering improves on that, but still describes players through average cluster centers. Basketball roles are often easier to understand from the extremes: the real players whose statistical profiles define the boundaries of the league.

Archetypoid Analysis (ADA) finds those **extreme real players** and describes everyone else as a convex blend of them. Instead of assigning one opaque label, the portal can describe a player as, for example, `48% Perimeter Defender + 26% Inside Scoring Big + ...`. The weights sum to 100%, and each corner is anchored by an actual player, so the output reads like a scouting recipe rather than a black-box classification.

## What the portal answers

Four questions, connected through the same eight-part NBA archetype language:

1. **What types of players define the league?** — an interactive 3D vocabulary of the eight archetypes and all 433 recipe-eligible players.
2. **How should we describe and investigate one player?** — roster screening followed by a five-part individual diagnosis.
3. **What NBA role might a 2026 draftee play?** — a separately trained NCAA-to-NBA archetype translator, filtered by the team that selected him.
4. **How do we carry the analysis into a meeting?** — a two-page player report generated from the same computed evidence shown in the portal.

The default presentation flow opens on **Player Breakdown**, scoped to the curated **New Orleans Pelicans 2026–27 roster**, with **Zion Williamson** selected. This is a call-ready case-study entry point, not a Pelicans-only model: the user can switch to all 30 teams or another team at any time.

## Step 0 — Data

Five validated data layers support the portal.

**1. NBA player-feature table (the ADA input).** Three sources are joined per player-season: NBA.com box and advanced statistics (via `nba_api`), Basketball-Reference shot-location profiles, and NBA.com Synergy play-type shares. After preprocessing — per-100 rates, a MIN ≥ 300 floor, and accent-insensitive name matching — each player-season becomes a **29-dimensional feature vector**. The data covers 2023–24 through 2025–26, with roughly 430 eligible players per season. All downstream joins use NBA player IDs.

**2. Lineup and on-court tables.** NBA.com lineup data at 2–5 player group sizes, per-player on-court net rating, and team on/off splits support the teammate-environment diagnostics.

**3. Possession-level stint table.** When the standard `pbpstats` NBA.com endpoint stopped serving the required 2025–26 data, the project added a parser for the current v3 play-by-play endpoints. It tracks the ten players on court through substitutions, reconciles period openings, and counts possessions from real events. The result is **75,587 directed stints** across all **1,230 games**. Every game exactly matches its real final score; the documented residual is that possession totals run about 5% low because of ambiguous substitution sequences.

**4. Event-to-stint attribution.** Each shot, rebound, turnover, and free throw is linked back to its stint. This allows the diagnostic page to measure a player's own usage inside different lineup contexts instead of relying only on team totals.

**5. NCAA-to-NBA translation data.** A separate college table contains **29,312 player-seasons** from 2016–17 through 2025–26 in a 12-feature space available consistently on the college side. Model development used **237 drafted players** with both a complete final college season and at least 300 NBA rookie minutes. The 2025 draft class — another 36 players — was held out until final evaluation; only after that test was complete was the deployment model refit on all **273 historical anchors**. Current 2026 draft selections and team assignments are loaded separately so the portal can show the complete draft class by full team name, including players for whom NCAA projection inputs are unavailable.

## Step 1 — Fit ADA

- The NBA basis is fit once on the 2025–26 regular season: **433 players, K = 8**. K follows the paper's intra-archetype-variance diagnostic rather than a plain RSS elbow. The eight archetypoids are real players: Nicolas Batum, Jonas Valančiūnas, Clint Capela, D'Angelo Russell, Ryan Kalkbrenner, Tim Hardaway Jr., Shai Gilgeous-Alexander, and Bez Mbeng.
- The resulting labels are **3&D Wing, Inside Scoring Big, Rim Protector / Roll Man, Combo Guard, Play-Finishing Big, Shooting Specialist, Offensive Engine,** and **Perimeter Defender**. Six are strong matches to the paper's NBA types; two are deliberately relabeled from the fitted feature profiles rather than forced into a paper label that the data does not support.
- Historical NBA seasons are projected onto the same frozen 2025–26 basis, never refit, so player recipes remain comparable over time.
- The 3D scene is a visualization of those recipes, not a second model. The eight corners are assigned to a balanced square antiprism, and every player's position is the same convex combination of the corners as his eight recipe weights. This gives every archetype equal visual importance while preserving the blend relationship.
- NCAA basketball uses its own independent eight-archetype space because college data lacks the NBA's shot-location and play-type dimensions. A Bayesian Dirichlet regression then learns the mapping from a player's college recipe and draft/college context to his likely rookie NBA recipe.

## Step 2 — Build the portal

**Tab 1: The 8 Player Types.** The league-wide vocabulary page. The eight exemplar headshots define the corners of an auto-rotating 3D space; the neutral-grey interior cloud represents every 2025–26 player with 300+ minutes. Archetype color encodes position family — guards, wings, and bigs — while shade distinguishes types inside each family. Hovering a type isolates its players. Zooming into the space reveals selected player portraits, and clicking a corner or player opens the relevant archetype or similarity result. The layout collapses to a single-column mobile view without changing the underlying camera, geometry, or recipe positions.

**Tab 2: Player Breakdown (`Diagnostic Analysis`).** The page begins with a team-scoped screening chart and sortable player table; selecting a chart point or sidebar player updates the same analysis. Player links from the 3D page enter this flow in the current browser tab. It then answers five questions:

- **Who is he?** — profile, eight-part recipe, purity, and entropy: whether the player is specialized or hybrid.
- **What makes him different from his role?** — the player's largest feature-level deviations from statistically similar recipe neighbors.
- **How has his role changed?** — season-over-season recipe drift, its league percentile, and the features behind the change.
- **How does his environment shape him?** — which teammate archetypes he receives, what has worked, and how his own usage changes across lineup contexts.
- **Is he being used the way he produces?** — deployment-only and outcome-only recipes projected onto the same basis; their gap is a descriptive flag for possible miscasting.

For the Pelicans presentation preset, the team selector uses the curated 2026–27 roster while every recipe, minute total, and performance diagnostic remains frozen to the 2025–26 regular season. A new addition may therefore display his prior team's season context. Players without enough 2025–26 NBA data are not assigned fabricated recipes.

**Tab 3: NCAA Bridge — 2026 Draft Archetype Projections.** This page is about **2026 draftees**, not current NBA rookies. A full-name team filter shows the players selected by that team, with profile images, college context, and a projected three-part NBA archetype receipt when NCAA input data is available. Each projection can be opened to see statistically similar college profiles and what those players actually became as NBA rookies. A second section provides measured college-to-rookie transitions for historical players, plus the complete sortable set of 273 historical anchors.

The translator was evaluated once on the untouched 2025 class. It identified the exact top rookie archetype **52.8%** of the time and placed the correct archetype in its top two **69.4%** of the time. It materially beat both the league-average guess and a direct college-to-NBA structural relabeling baseline. Its posterior intervals failed calibration, so the portal does not display them; real historical comparables provide the uncertainty context instead.

**Tab 4: Player Report.** A one-click two-page PDF built from the same selected player's diagnostic results. The page and export share the same computed values, preventing a report from silently disagreeing with the interactive analysis.

## Conclusion

The portal gives a coaching staff, front office, or scouting group:

- **A shared role language.** Eight real-player anchors and one readable recipe for every eligible NBA player.
- **A navigable league map.** The 3D space makes “everyone is a blend” visible, then supports direct drill-down from archetype to player to comparable-player evidence.
- **A five-question individual diagnosis.** Identity, differentiation, development, environment, and deployment-versus-production are presented as one connected flow.
- **A draft translation tool.** NCAA roles are translated into likely rookie NBA roles with a frozen model whose held-out track record and limitations are disclosed.
- **A portable report.** The same evidence can leave the portal as a meeting-ready PDF.
- **A focused presentation path.** The Pelicans/Zion default provides one concrete case study for a conversation without changing the portal's league-wide scope.

The model is intentionally descriptive. It is strongest when organizing evidence, defining roles, and identifying questions worth investigating. Talent evaluation, scheme decisions, health, personality, and outcome prediction remain human decisions.

## Obstacles

The project also tested a more ambitious possession-level, skill-weighted archetype-RAPM: 75,587 stints, prior-season skill controls, 56 archetype-pair interactions, ridge regression, and leave-one-team-out validation.

**It failed its pre-committed gate.** The required out-of-sample correlation against real lineup net ratings was at least 0.30; the initial Brooklyn result was 0.116. The investigation then expanded rather than hiding the failure:

- **Thirty-team leave-one-team-out validation** on 1,317 real lineups remained weak at approximately `r = 0.21`, `R² = 0.04`.
- **Ablations** — fewer features, different possession floors, no skill weighting, three pooled seasons, and alternate targets such as eFG% and TOV% — did not materially improve the result.
- **Split-half reliability** was the decisive test. Independent halves of the same lineup's possessions correlated at or below zero, showing that lineup outcomes at this sample size were not stable enough to support the intended model.
- **Team-season aggregation** produced `r = 0.770` for the full model versus `r = 0.767` for a talent-only baseline. The archetype interactions added no defensible predictive value.

The failed model is not used to rank lineups or recommend transactions in the visible portal. The validated archetype recipes remain because they answer a different, descriptive question that the evidence supports.

## Limitations

- Archetypes describe **style, not quality**. A large weight is not a talent rating.
- K = 8 is a defensible modeling choice, not a uniquely provable answer.
- The current Pelicans selector is a roster context layered over frozen 2025–26 evidence; it is not a 2026–27 performance forecast.
- Players without the required NBA minutes do not receive an NBA recipe. The portal shows missing coverage rather than inventing one.
- Small lineup samples are explicitly flagged and should not be treated as stable causal evidence.
- NCAA Bridge answers **“what role might he play if he earns minutes?”**, not whether he will play, how good he will be, or how his team will deploy him.
- College data cannot fully observe NBA shot-location, play-type, coaching, scheme, medical, and roster-context effects.
- No salary, contract, trade, or win-projection model is included.
