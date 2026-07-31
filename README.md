**Step0 - Data Collection and Preprocess
Pulls league-wide player and lineup data from NBA.com (via nba_api) and Basketball-Reference...


**Step 1 — Player archetype recipes (ADA).**
Following SSAC 2026 paper *"Scouting Anyone: Probabilistic Player Archetypes for Any League"* (openly cited in write-up). Archetypoid Analysis on league-wide player-season-team observations, ≥300 minutes filter. Features mirror the paper's NBA table: box-score rates (TRB%/AST%/TOV%/STL%/BLK%/USG%/FT rate/TS or eFG/pts per 100/3PA share) + height + shot-distance distribution + play-type shares. z-scored. K selected via intra-archetype variance (paper found K=8; we need to validate). ADA (not AA) so each archetype is a real player — interpretability.
*Why not K-means/GMM: cluster centers drift to the average player; basketball roles are defined at extremes.*

**Step 2 — 8×8 archetype-pair synergy matrix (my main extension; NOT in the paper).**
Each 2-man lineup → outer product of the two players' recipe vectors → 64-dim archetype-pair exposure. Weighted regression of pair net rating on exposure; weights = shared minutes; filter MIN ≥ 100.
**CRITICAL: control for individual player quality** (each player's own on-court net rating or BPM as covariates) — without this the matrix degenerates to "star × anything = positive." Keep the uncontrolled v0 for comparison.
*Why 2-man not 3/5-man: empirical — 3-man combos don't survive the minutes filter at density supporting a 64-cell space. Day 1 produces the sample-density figure proving this.*

**Step 3 — Nets lineup scoring.**
Five-man score = sum over 10 pairs; each pair = pₓᵀ M p_y (quadratic form). Additivity assumption stated openly with 3-part defense (data reality / first-order approximation convention / upgrade path with tracking data). Rank all Nets 5-man combos from the 13 data-eligible players.

**Step 4 — Rookie slot query (inverse question; handles zero-data rookies).**
2026 rookies (Mikel Brown Jr. #6, Joshua Jefferson #28, Tyler Bilodeau) have zero NBA data. The framework never estimates their recipes. Instead: fix a 4-man core → substitute each of 8 archetypes as hypothetical 5th man → demand ranking = what the slot NEEDS. Coach maps the rookie onto it. Brown gets two paths analyzed: (a) reverse-search best 4-man groups if he plays his college identity (on-ball engine), (b) development direction if he joins the default core.
**+ Dessert:** same query, candidate pool = all ~450 NBA players → league-wide fit shortlist (framed as fit shortlist, NOT trade advice — no salary/contract modeling).
**+ NCAA bridge = Future Work paragraph only** (anchor-player mapping NCAA recipe → NBA rookie recipe; ridge/Dirichlet on ~300-400 anchors). Do NOT implement unless I explicitly say Plan B is on.
