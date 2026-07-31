# The Archetype Portal for the Brooklyn Nets

**Live app:** [nets-lineup.streamlit.app](https://nets-lineup.streamlit.app)

**Bowen Li** · [bowenli@gmail.com](mailto:bowenli@gmail.com) · +1 (323) 630-5773 · [bl-website-peach.vercel.app](https://bl-website-peach.vercel.app/)

## Why this idea

The idea comes from *"Scouting Anyone: Probabilistic Player Archetypes for Any League"* ([MIT Sloan Sports Analytics Conference 2026](https://www.sloansportsconference.com/research-papers/scouting-anyone-probabilistic-player-archetypes-for-any-league)). Traditional basketball labels players as guard/forward/center, or positions 1–5. Unsupervised learning can do better — but standard clustering (K-means, GMM) describes players by **cluster centers**: average types. That's backwards for basketball. A stretch big who occasionally posts up, like Chet Holmgren or Jaren Jackson Jr., fits no single average role. Real basketball roles live at the **extremes**, not the middle.

Archetypoid Analysis (ADA) fixes this. It finds the K most **extreme real players** — the corners of the player cloud — and describes everyone else as a blend of them: "31% 3&D Wing + 21% Traditional Playmaker + ...". Every archetype is an actual player, so the output reads like scouting language, not a black-box cluster label.

## What the portal answers

Two questions, in sequence, both built on ADA recipes:

1. **How do we describe a player?** — an ADA-derived role language (Tab 1: The 8 Player Types).
2. **Is a player being used correctly for his role?** — a per-player diagnosis (Tab 2: Player Breakdown).

## Step 0 — Data

Four data layers, all pulled and validated from primary sources.

**1. Player-feature table (the ADA input).** Three sources joined per player-season: NBA.com box and advanced stats (via `nba_api`), Basketball-Reference shot-location profiles (FGA share by distance, corner-3 rate, % of FG assisted), and NBA.com Synergy play-type shares (9 categories). After preprocessing — per-100 rates, a MIN ≥ 300 floor, accent-insensitive name matching — each player-season becomes a **29-dimensional feature vector**, covering 2023-24 through 2025-26 (~430 players per season). All joins run on NBA player IDs; names are used once at the initial handoff, then never again.

**2. Lineup and on-court tables.** NBA.com lineup data at 2–5 man group sizes (shared minutes, net ratings), plus per-player on-court net rating and team on/off splits. These power the teammate-environment diagnostics.

**3. Possession-level stint table (built from scratch).** The standard parser (`pbpstats`) broke — its NBA.com endpoint stopped serving data in 2025-26 — so we built our own on the current v3 endpoints. It walks every play-by-play event, tracks the 10 players on court through substitutions (in/out direction inferred from on-court state, since "SUB: A FOR B" has no fixed word order), cross-checks period openings against observed actions (some between-period lineup changes are never logged), and counts possessions from real events rather than the FGA + 0.44×FTA estimate. Output: **75,587 directed stints** (5 offense + 5 defense IDs, possessions, points) across all **1,230 games**, cached per game and fully resumable.

Validation: 1,230/1,230 games parsed, and **100% of games exactly match their real final scores**. Getting there surfaced two real bugs — 5 games with malformed home/away text, and ~3% of points silently dropped when substitutions happen between free throws. Every row passes the 5-distinct-offense + 5-distinct-defense check. One documented residual: possession counts run ~5% low (from ambiguous substitutions), which shifts the scale but not the relative comparisons the portal uses.

**4. Event-to-stint attribution.** A per-game table linking each play-by-play event (shooter, rebounder, turnover, free throws) to its stint, validated by reconciling per-stint scoring sums. This lets the portal measure an *individual's* usage inside specific lineup contexts (Tab 2's elasticity), not just team totals.

## Step 1 — Fit ADA

- Fit once on 2025-26 (433 players, K = 8). K is chosen the paper's own way — intra-archetype variance (Section 4.3), not the plain RSS elbow, which the paper flags as misleading here. Our own K sweep on this season's real data shows the same diminishing-returns shape. The 8 archetypoids are real players (e.g. Nicolas Batum, Clint Capela, Shai Gilgeous-Alexander); every other player gets a **recipe** — mixture weights over the 8, summing to 1. Cross-checked against the paper's own NBA archetype table: 6 of 8 are strong matches, 2 are partial matches, reported as such rather than forced.
- Historical seasons (2023-24, 2024-25) are **projected onto the same fixed basis** — never refit — so recipes stay comparable across seasons.

## Step 2 — Build the portal

**Tab 1: The 8 Player Types.** The vocabulary. A convex-hull view of the league with the 8 archetypoids as corners, plus the full Nets roster described in recipe terms.

**Tab 2: Player Breakdown.** One player, five diagnostics:

- **Who is he?** — recipe + purity/entropy: what type he is, and how specialized or hybrid that identity is.
- **What makes him different from his role?** — where he deviates from his own archetype's typical profile.
- **How has his role changed?** — season-over-season drift, its league percentile, and the features driving it.
- **How does his environment shape him?** — what teammate types he gets vs. the norm for his style, what has actually worked, and whether his own usage grows or shrinks depending on who shares the floor (elasticity, from play-by-play event attribution).
- **Is he being used the way he produces?** — two partial recipes (deployment-only features vs. outcome-only features) projected onto the same basis; the gap between them flags miscasting.

Michael Porter Jr. as a real example: **67% Shooting Specialist**, purity in the **88th percentile** (unusually pure, not a hybrid), miscast score **High** (0.539 JS distance; a bigger usage-production gap than 74% of the league), and **Elastic** (a 6.9pp usage swing depending on lineup). The diagnosis: his deployment sits well below his production as an Offensive Engine — a +20.3pp untapped gap, backed by an assist rate that runs ahead of his usage.

**Tab 3: Report.** A one-click, 2-page PDF of the same diagnosis — same computed values, nothing recomputed, so the portal and the exported report can never silently disagree.

## Conclusion

What the portal and its report actually give a coaching staff or front office:

- **A shared role language.** Eight archetypes, each anchored to a real player, and every player on the roster described as a readable recipe — so "what kind of player is he" has one measurable answer instead of five opinions.
- **A five-question diagnosis for any player.** Who is he, what makes him different from his role, how his role has changed, how his environment shapes him, and whether he's being used the way he produces. For Michael Porter Jr., that chain surfaces a concrete, checkable claim: deployed as a pure shooter, producing like an offensive engine — a specific usage gap the staff can test on the floor.
- **A portable report.** The same diagnosis exports as a two-page PDF with identical numbers, so what's discussed in a meeting is exactly what the portal shows.

Each claim in the portal is backed by real validated data, and the model was tested at every level before use. The result of those tests draws a clear line: archetypes are reliable for describing players and diagnosing usage, while predicting outcomes remains the job of talent. The portal respects this boundary on every page — which means when it does flag something, like the Porter gap, that flag is worth a conversation.

## Obstacles

We also built the ambitious version: a possession-level, skill-weighted archetype-RAPM — the 75,587-stint table above, prior-season skill terms (to avoid circularity), 56 archetype-pair interaction terms, ridge regression, validated leave-one-team-out.

**It failed our own gate.** GATE 2 required out-of-sample correlation ≥ 0.3 against real lineup net ratings; Brooklyn came in at 0.116 (n = 5). Rather than stop there, we escalated step by step — each a specific hypothesis tested against real data:

- **30-team leave-one-team-out** (n = 1,317 real lineups, ruling out "Brooklyn is just unlucky"): still weak — corr = 0.21, R² ≈ 0.04.
- **Ablations**, each ruled out in turn: dropping to 8 features with no interactions (worse, 0.13); sweeping the possession floor from 1 to 12 (flat at 0.20–0.22); removing skill-weighting (no change, 0.20); pooling 3 seasons instead of 1 (no change, 0.20); swapping the target to eFG%/TOV% (same range).
- **Split-half reliability — the decisive test.** Split each lineup's own possessions randomly in half and correlate the two halves. This is the empirical ceiling: it just asks whether the target agrees with itself. It came in **at or below zero**. A lineup's own net rating, at this sample size, doesn't even correlate with an independent half of its own possessions. Our model's ~0.21 isn't below some hidden ceiling — it *is* the ceiling.
- **Team-season aggregation** (the least noisy test — 90 team-seasons over 3 years, checked against official NET_RATING): full model **r = 0.770** vs. a talent-only baseline **r = 0.767**. Statistically indistinguishable — and talent-only wins outright in 2 of 3 seasons.

## Limitations

- Archetypes describe **style, not quality** — nothing here is a talent rating.
- K = 8 is a defensible choice rather than a provably optimal one, and earlier seasons are projected onto the 2025-26 fit rather than independently re-fit.
- Small-sample estimates (e.g., under 100 shared minutes) are flagged in the portal and should be read as noise, not signal.
- No salary, contract, or trade modeling — roster outputs are fit and composition flags, never trade advice.
