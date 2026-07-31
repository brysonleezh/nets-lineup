# CLAUDE.md — Project Context for Claude Code

> Put this file at the repo root. Claude Code reads it automatically at session start.

## Who I am & what this is

I'm Bowen Li, completing a **take-home project for the Brooklyn Nets Data Scientist (Basketball Operations) role**. Assigned by Glenn DuPaul (VP Basketball Analytics, baseball-analytics background, values methodological rigor and honesty). **Deadline: 10 calendar days from receipt (received 2026-07-22). Target finish: Day 7, keeping Days 8-10 as buffer.**

The deliverables will be evaluated on: creativity, skillset, thought process, basketball knowledge. Per the prompt: *"Your reasoning and choices made along the way matter just as much (if not more) than the final output."*

## ⚠️ Two hard constraints (read first)

1. **AI usage must be documented.** The assignment requires: *"If you used AI to assist in writing any of the code, you must include what prompts you used as part of the documentation."* → **Every time you (Claude Code) write or substantially modify code, append an entry to `AI_USAGE.md`** in this format:
   ```
   ## Entry NNN — <short title>
   - Date: YYYY-MM-DD
   - Tool: Claude Code
   - Prompt (summarized): "<what I asked>"
   - What was used: <what code/output was adopted>
   - What was NOT AI-generated: <decisions that were mine>
   ```
2. **The main idea must be mine (it is).** The research question, method selection, and all modeling decisions were made by me before this session (documented below). Your role: implement code, debug, review prose. Do NOT propose replacing the core research question or method. Refinements and bug-catches are welcome; direction changes are not.

## Research Question (final, do not change)

> **"Which lineup combinations should the Nets play, and what kind of player should surround Mikel Brown Jr.?** A probabilistic archetype model + a league-wide archetype synergy matrix, applied to the current Nets roster."

## Method — four-step pipeline (all decided, with rationale)



**Step 1 — Player archetype recipes (ADA).**
Following SSAC 2026 paper *"Scouting Anyone: Probabilistic Player Archetypes for Any League"* (openly cited in write-up). Archetypoid Analysis on league-wide player-season-team observations, ≥300 minutes filter. Features mirror the paper's NBA table: box-score rates (TRB%/AST%/TOV%/STL%/BLK%/USG%/FT rate/TS or eFG/pts per 100/3PA share) + height + shot-distance distribution + play-type shares. z-scored. K selected via intra-archetype variance (paper found K=8; we validate, expect ~8). ADA (not AA) so each archetype is a real player — interpretability.
*Why not K-means/GMM: cluster centers drift to the average player; basketball roles are defined at extremes.*

**Step 2 — 8×8 archetype-pair synergy matrix (my main extension; NOT in the paper).**
Each 2-man lineup → outer product of the two players' recipe vectors → 64-dim archetype-pair exposure. Weighted regression of pair net rating on exposure; weights = shared minutes; filter MIN ≥ 100.
**CRITICAL: control for individual player quality** (each player's own on-court net rating or BPM as covariates) — without this the matrix degenerates to "star × anything = positive." Keep the uncontrolled v0 for comparison.
*Why 2-man not 3/5-man: empirical — 3-man combos don't survive the minutes filter at density supporting a 64-cell space. Day 1 produces the sample-density figure proving this.*

**Step 3 — Nets lineup scoring.**
Five-man score = sum over 10 pairs; each pair = pₓᵀ M p_y (quadratic form). Additivity assumption stated openly with 3-part defense (data reality / first-order approximation convention / upgrade path with tracking data). Rank all Nets 5-man combos from the 13 data-eligible players.

**Step 4 — Rookie slot query (inverse question; handles zero-data rookies).**
Zero/insufficient-NBA-data players (Mikel Brown Jr. #6 2026, Joshua Jefferson, Tyler Bilodeau #43 2026, plus several others per the 2026-07-27 roster update below) have no usable NBA recipe. The framework never estimates their recipes from NBA data directly. Instead: fix a 4-man core → substitute each of 8 archetypes as hypothetical 5th man → demand ranking = what the slot NEEDS. Coach maps the rookie onto it. Brown gets two paths analyzed: (a) reverse-search best 4-man groups if he plays his college identity (on-ball engine), (b) development direction if he joins the default core.
**+ Dessert:** same query, candidate pool = all ~450 NBA players → league-wide fit shortlist (framed as fit shortlist, NOT trade advice — no salary/contract modeling).
**+ NCAA bridge — Plan B is now ON (as of 2026-07-27, explicit go-ahead given).** Anchor-player mapping: NCAA recipe → NBA rookie recipe. Separate ADA (K=8) fit on college data from collegebasketballdata.com, on a reduced feature set (that source has no Synergy-style play-type shares — 9 of the NBA model's 29 features have no college-side equivalent and are dropped, not faked). Anchor set = current NBA players (2025-26, MIN≥300) with a findable final college season, matched via CBBD's `/draft/picks`. Ridge regression per NBA-archetype-dimension, college recipe → NBA recipe, trained on anchors, validated via leave-one-out (report real accuracy, not assumed) before trusting it on the actual zero-data players.

## Nets roster context (corrected 2026-07-27 — the "full roster turnover" claim below was wrong)

**CORRECTION:** the previous version of this section claimed the entire veteran core (MPJ, Randle, Terance Mann, Ziaire Williams, Day'Ron Sharpe, Noah Clowney, Josh Minott, Malachi Smith) had been "traded away." That was false. Verified live against ESPN's roster page (https://www.espn.com/nba/team/roster/_/name/bkn/brooklyn-nets, checked 2026-07-27): all six are still on the roster. The 9 names previously listed in this section (John Ukomadu, Aaron Scott, Ben Humrichous, Grant Nelson, Dwight Murray Jr., Dion Brown, Duke Brennan, Dain Dainja, Hunter Sallis) do **not** appear on ESPN's roster page at all — removed below as unconfirmed, not verified as real. Any case-study subject (e.g. the Intro page's MPJ worked example) is valid again; no replacement needed.

Real, ESPN-verified 19-man roster (#, position, height, weight, age, college all from ESPN; "years pro"/"how acquired" aren't shown on ESPN's roster page and are not guessed here):

| Player | # | Pos | Ht | Wt | Age | College |
|---|---|---|---|---|---|---|
| Mikel Brown Jr. | 0 | G | 6-5 | 180 | 20 | Louisville |
| Danny Wolf | 2 | F | 6-11 | 250 | 22 | Michigan |
| Drake Powell | 4 | G | 6-5 | 195 | 20 | North Carolina |
| Egor Dëmin | 8 | G | 6-8 | 200 | 20 | BYU |
| E.J. Liddell | 9 | F | 6-6 | 240 | 25 | Ohio State |
| Tyson Etienne | 10 | G | 6-0 | 200 | 26 | Wichita State |
| Nolan Traoré | 13 | G | 6-3 | 185 | 20 | — |
| Terance Mann | 14 | G | 6-6 | 215 | 29 | Florida State |
| Michael Porter Jr. | 17 | F | 6-10 | 218 | 28 | Missouri |
| Day'Ron Sharpe | 20 | C | 6-10 | 265 | 24 | North Carolina |
| Noah Clowney | 21 | F | 6-10 | 210 | 22 | Alabama |
| Julius Randle | 30 | F | 6-9 | 250 | 31 | Kentucky |
| Chaney Johnson | 31 | F | 6-8 | 225 | 24 | Auburn |
| Tyler Bilodeau | 34 | F | 6-8 | 228 | 22 | UCLA |
| Ben Saraf | 77 | G | 6-6 | 200 | 20 | — |
| Josh Minott | 00 | F | 6-8 | 205 | 23 | Memphis |
| Joshua Jefferson | — | F | 6-8 | 246 | 22 | Iowa State |
| Keon Ellis | — | G | 6-4 | 175 | 26 | Alabama |
| Moritz Wagner | — | F | 6-11 | 245 | 29 | Michigan |

Checked directly against the local DB (`build_nba_side_tables()`, 2025-26 season), not assumed from ESPN or the label alone:
- **Real 2025-26 NBA data, clears MIN≥300, real fitted K=8 archetype recipe already exists in `data/basis_2025_26/recipes.csv`** (16 players — the archetype model was fit league-wide, so a player's recipe doesn't care which team he's on): Danny Wolf (1187 min, BKN), Drake Powell (1320 min, BKN), Egor Dëmin (1308 min, BKN), E.J. Liddell (349 min, BKN), Nolan Traoré (1243 min, BKN), Terance Mann (1529 min, BKN), Day'Ron Sharpe (1160 min, BKN), Noah Clowney (1780 min, BKN), Julius Randle (2610 min, **still shows team=MIN in the season-cumulative row**), Chaney Johnson (348 min, BKN — barely clears), Ben Saraf (916 min, BKN), Josh Minott (834 min, BKN), Michael Porter Jr. (1689 min, BKN), Keon Ellis (1479 min, **still shows team=CLE**), Tyson Etienne (380 min, BKN), Moritz Wagner (427 min, **still shows team=ORL**). The three "still shows" players were traded to Brooklyn recently enough that their season-cumulative box-score row hasn't switched teams in the source data — exact trade dates not confirmed, and the recipe join is PLAYER_ID-based, so this doesn't block using their recipe.
- **Zero NBA data anywhere in the DB, any season → NCAA bridge** (3 players, true rookies, confirmed by a direct query with zero rows returned at any season): Mikel Brown Jr., Tyler Bilodeau, Joshua Jefferson. NCAA-side data pull/model not yet built — deferred, per explicit instruction (Plan B mechanics to be specified in a later message).
- 16 + 3 = 19, matching ESPN's live roster count exactly.

## Repo structure (updated Day 2: no notebooks — plain .py scripts only, each
## runnable standalone with a `Prompt (summarized): ...` header when AI-assisted)

```
nets-lineup-synergy/
├── README.md          # includes daily Progress Log block at top
├── AI_USAGE.md        # every AI-assisted code change logged
├── writeup.md         # skeleton exists with <!-- guidance comments --> per section
├── writeup.pdf        # generated Day 7 via pandoc
├── requirements.txt
├── data/              # SQLite; gitignored data files
├── src/               # data.py, inspect_data.py, viz.py, features.py,
│                       archetypes_model.py (not archetypes.py — would shadow
│                       the installed `archetypes` package on import), synergy.py
└── figures/
```

## Stack (decided)

Python only. pandas/numpy/scikit-learn, `nba_api` (data), `archetypes` package (ADA — validate it works on Day 2 toy run; fallback: implement ADA or use R archetypes for cross-check), SQLite for storage (SQL visible in code = JD required skill), matplotlib/seaborn static figures (NO Streamlit until Day 8, NO D3). JD requires GitHub practices → small commits, message format below.

**Added later (Day 8 portal build-out):** Streamlit itself (per the Day 8 plan below), plus `jinja2` + `playwright` for the per-player PDF diagnosis report (HTML/CSS template → headless-Chromium print-to-PDF, replacing an earlier reportlab-based version — see DIAGNOSTICS_README.md's "Report (PDF export)" section). **One-time environment setup step, not covered by `pip install -r requirements.txt`:** `playwright install chromium` must be run once per environment before the PDF report feature will work; without it, the feature fails gracefully with an actionable in-app error rather than crashing the page.

## 7-day schedule & current status

- **Day 1:** repo + data pulls (B-Ref advanced, NBA.com shot-distance/play-type, leaguedashlineups group_quantity 2/3/4/5, six seasons 2020-21 to 2025-26) + **2/3/4/5-man MIN≥100 sample-density check + figure**
- **Day 2:** features aligned to paper Table 3.3 → ADA toy validation → full run prep 
- **Day 3:** synergy matrix v0 (no baseline) → v1 (with baseline control)
- **Day 4:** matrix robustness (thresholds 50/100/200, season splits, sanity checks) + heatmap final
- **Day 5:** Nets lineup ranking + diagnostics; slot queries (Brown/Jefferson/Bilodeau); league shortlist
- **Day 6:** write-up draft (fill writeup.md skeleton; JD language in motivation)
- **Day 7:** code into src/, README, AI_USAGE audit, pandoc → PDF
- **Day 8 (only if Days 1-7 debt-free):** Streamlit portal — 3 pages (Lineup Builder / Slot Finder / Synergy Matrix), pure lookups of precomputed recipes.parquet + matrix.npy, deploy to my existing self-hosted server
- **Day 9-10:** buffer / optional NCAA-bridge mini (Plan B) / early submission

> **Update this section's status line as days complete.** Current: Day 1 complete (all 6 raw tables pulled/validated, sample-density figure done — see `figures/day1_sample_density.png` and `notebooks/01_data_collection.ipynb`). Day 2 in progress.

## Commit conventions

Format: `<Area>: <what and, when relevant, why>` — areas: Data / Feature / Model / Analysis / Fix / Writeup. Record decisions, not just actions (e.g., `Model: add BPM baseline to synergy regression — v0 degenerated to star-pairs-always-positive`). Keep v0→v1 history visible; never squash/rebase. Push at least daily. End-of-day ritual: README progress block + writeup.md raw notes + `Day N progress log` commit.

## Known pitfalls (from planning)

- NBA.com API rate limits → sleep between calls, batch, cache to SQLite immediately
- `archetypes` package is niche → Day 2 toy validation before committing to it
- Lineup net-rating noise → minutes-weighted regression, MIN≥100 filter, robustness checks Day 4
- Uncontrolled synergy matrix = star-contamination (the single most likely way this project fails silently)
- Write-up must stand alone without the Day 8 app

## Interview context (for tone/framing decisions)

Glenn probes methodology honesty — limitations stated plainly beat overclaiming. Write-up motivation should echo JD language: "roster construction, player evaluation, player development, game strategy." The Scouting Anyone paper is cited openly; my extensions (synergy matrix + slot query) are clearly delineated from it.