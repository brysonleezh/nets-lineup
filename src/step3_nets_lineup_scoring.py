"""
Step 3 - Nets lineup scoring.

Five-man score = sum over the 10 pairs in a 5-man group, each pair scored
as p_x.T @ M @ p_y using step2's v1 (individual-quality-controlled) matrix.
This is an additivity assumption - a lineup's chemistry is treated as the
sum of its pairwise chemistries, not a genuine 5-way interaction.
CLAUDE.md's 3-part defense for this, stated here rather than left implicit:
  1. data reality - Day 1's own density check showed even 3-man combos
     don't survive MIN>=100 at a density that could support fitting a true
     interaction model past pairs.
  2. it's the standard first-order-approximation convention for exactly
     this reason in lineup-construction work.
  3. there's a stated upgrade path once real on-court tracking data
     (spacing, shot quality allowed, etc.) is available.

Part A  the 13 data-eligible Nets players (CLAUDE.md roster context)
Part B  resolve each to a PLAYER_ID + current-season archetype recipe
Part C  score every 5-man combo (sum of C(5,2)=10 pairwise terms)
Part D  rank + report

Explicitly out of scope here: Mikel Brown Jr., Joshua Jefferson, Tyler
Bilodeau (zero NBA data - CLAUDE.md routes them to Step 4's slot query,
which asks what archetype a 5th-man slot NEEDS rather than trying to score
a rookie who has no recipe to plug in).

STATUS: skeleton, written while step1's basis was still fitting. Part A/B's
roster-resolution logic is tested against the real database (doesn't need
step1); Part C/D's scoring math is tested against synthetic recipes/M, not
real ones yet - step1's recipes.csv and step2's synergy_matrix_v1.npy don't
exist on disk yet. Re-run phase_score() once both do.
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from step0_data_collect_process import build_nba_side_tables, _normalize_name

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# --- Part A: the current Nets roster (corrected 2026-07-27) --------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: the previous version of this list ("full roster turnover") was
wrong - user pasted a roster screenshot/list showing MPJ, Randle, Terance
Mann, Day'Ron Sharpe, Noah Clowney, and Josh Minott still active, then
asked to verify against ESPN's live roster page
(https://www.espn.com/nba/team/roster/_/name/bkn/brooklyn-nets). Fetched
it directly: confirms all 6 are still on the roster, plus names never
before seen in this project (E.J. Liddell, Nolan Traore, Julius Randle,
Keon Ellis, Tyson Etienne, Moritz Wagner) - 19 total. The 9 names the old
list had instead (John Ukomadu, Aaron Scott, Ben Humrichous, Grant Nelson,
Dwight Murray Jr., Dion Brown, Duke Brennan, Dain Dainja, Hunter Sallis) do
not appear on ESPN's page and are dropped as unconfirmed.
Used: re-checked every one of the 19 names directly against
build_nba_side_tables() for season 2025-26 (not assumed from ESPN age/exp)
- 16 clear MIN>=300 and already have a real fitted recipe in
data/basis_2025_26/recipes.csv (the archetype model was fit league-wide,
so a recipe doesn't care which team a player's minutes were logged under -
this is exactly why Julius Randle/Keon Ellis/Moritz Wagner, whose
season-cumulative row still shows their previous team (MIN/CLE/ORL - they
were evidently traded to Brooklyn recently enough that the row hasn't
caught up), still resolve fine: the join is PLAYER_ID+SEASON only, no team
filter, so resolve_roster()'s existing logic needed zero code changes -
only the roster data itself was wrong). Mikel Brown Jr., Tyler Bilodeau,
and Joshua Jefferson return zero rows in player_base at ANY season
(confirmed via a direct LIKE query, not inferred) - true zero-data
rookies, same NCAA-bridge path as before.
Not AI: the roster itself (from ESPN, the user's source), which players
count as data-eligible (re-derived here directly from the DB per the
project's own "check, don't assume" convention), and the decision to defer
the NCAA-bridge mechanics for the 3 true rookies to a later message - all
the user's calls.
"""

NETS_ROSTER = [
    "Mikel Brown Jr.",
    "Danny Wolf",
    "Drake Powell",
    "Egor Dëmin",
    "E.J. Liddell",
    "Tyson Etienne",
    "Nolan Traore",
    "Terance Mann",
    "Michael Porter Jr.",
    "Day'Ron Sharpe",
    "Noah Clowney",
    "Julius Randle",
    "Chaney Johnson",
    "Tyler Bilodeau",
    "Ben Saraf",
    "Josh Minott",
    "Joshua Jefferson",
    "Keon Ellis",
    "Moritz Wagner",
]

# Real 2025-26 NBA data, confirmed MIN>=300 via build_nba_side_tables() -
# resolve_roster() below finds these directly, no bridge needed. Team label
# in the season-cumulative row may lag a recent trade (Randle=MIN,
# Ellis=CLE, Wagner=ORL as of this check) - doesn't matter, the recipe join
# is PLAYER_ID-based, not team-based.
NETS_ROSTER_NBA_DATA = [
    "Danny Wolf", "Drake Powell", "Egor Dëmin", "E.J. Liddell", "Nolan Traore",
    "Terance Mann", "Day'Ron Sharpe", "Noah Clowney", "Julius Randle",
    "Chaney Johnson", "Ben Saraf", "Josh Minott", "Michael Porter Jr.",
    "Keon Ellis", "Tyson Etienne", "Moritz Wagner",
]

# Zero real NBA data at any season (confirmed via a direct DB query, not
# inferred) - each needs step4's NCAA bridge. Bridge mechanics (CBBD school/
# season per player, ridge-regression mapping) deferred to a later message
# per explicit instruction - this list only marks who needs it.
NETS_ROSTER_NCAA_BRIDGE = [
    "Mikel Brown Jr.",
    "Tyler Bilodeau",
    "Joshua Jefferson",
]

assert len(NETS_ROSTER) == 19, "roster list drifted from the 2026-07-27 ESPN-verified correction"
assert len(NETS_ROSTER_NBA_DATA) + len(NETS_ROSTER_NCAA_BRIDGE) == len(NETS_ROSTER)


# --- Part B: resolve roster names -> recipes -----------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: 帮我把 step3 的框架写好. Asked how to look the roster up safely,
given this session's own history of name-matching bugs (the Egor Dёmin
Cyrillic-homoglyph miss, the Martin-twins GROUP_NAME collision). Checked
that reusing step0's own _normalize_name is possible (it's importable, not
duplicated here) so a fix to the join logic in one place fixes both.
Used: resolve-and-report rather than assume - fail loudly on 0 or >1
matches instead of silently taking the first row, since a silent wrong
match here would score a different real player without any visible error.
Not AI: MIN>=300 recipe floor already decided in step1/step0; treating
"MISSING" as a real, expected possibility rather than a bug (a rookie with
<300 minutes even across a full season is plausible, not a code fault).
"""


def resolve_roster(season="2025-26"):
    """Look up each NETS_ROSTER_NBA_DATA name's PLAYER_ID for the given
    season - the 16 current-roster players confirmed to have real NBA data
    at MIN>=300 (league-wide, regardless of which team logged those
    minutes - a recent trade doesn't invalidate the recipe join, since it's
    PLAYER_ID+SEASON, not team-scoped). The other 3 (NETS_ROSTER_NCAA_BRIDGE)
    are NOT looked up here at all - they have no real NBA row to find
    regardless of season, and belong to step4's NCAA-bridge path instead.

    Uses step0's own _normalize_name so this matches the exact same rules
    (accents, Cyrillic homoglyphs, suffixes, case) as the B-Ref join - not
    a second, possibly-inconsistent name-matching implementation.
    """
    pop = build_nba_side_tables()
    season_pop = pop[pop["SEASON"] == season].copy()
    season_pop["_norm"] = season_pop["PLAYER_NAME"].apply(_normalize_name)

    resolved, missing = [], []
    for name in NETS_ROSTER_NBA_DATA:
        norm = _normalize_name(name)
        match = season_pop[season_pop["_norm"] == norm]
        if len(match) == 0:
            missing.append(name)
            continue
        if len(match) > 1:
            raise ValueError(f"{name!r} matched {len(match)} rows in {season} - ambiguous, fix before scoring")
        row = match.iloc[0]
        resolved.append({
            "roster_name": name, "PLAYER_ID": int(row["PLAYER_ID"]),
            "PLAYER_NAME": row["PLAYER_NAME"], "MIN": float(row["MIN"]),
        })

    out = pd.DataFrame(resolved)
    print(f"resolved {len(out)}/{len(NETS_ROSTER_NBA_DATA)} NBA-data roster players in {season}")
    if missing:
        print(f"  MISSING (not in {season} at MIN>=300, or a name-spelling mismatch "
              f"- check against player_bio before assuming it's just low minutes): {missing}")
    return out


def attach_recipes(roster_df, recipes, k, season="2025-26"):
    """Join resolved roster PLAYER_IDs to their archetype recipe vectors."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    season_recipes = recipes[recipes["SEASON"] == season][["PLAYER_ID"] + arch_cols]

    out = roster_df.merge(season_recipes, on="PLAYER_ID", how="left")
    n_before = len(out)
    out = out.dropna(subset=["arch_0"]).reset_index(drop=True)
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"attach_recipes: dropped {n_dropped}/{n_before} roster players "
              f"(resolved in build_nba_side_tables() but missing from recipes.csv - "
              f"check step1's projection floor)")
    if len(out) < 5:
        raise ValueError(f"only {len(out)} roster players have recipes - can't form a 5-man lineup")
    return out


# --- Part C: score every 5-man combo --------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: 帮我把 step3 的框架写好, the p_x.T @ M @ p_y sum-over-pairs scoring
CLAUDE.md specifies. Asked for a vectorized version once it's more than a
toy roster, since 13 players is only C(13,5)=1287 combos now but this
should not become the bottleneck if the roster grows.
Used: precompute all pairwise p_x.T @ M @ p_y scores once (C(13,2)=78 of
them, not 1287*10), then each 5-man lineup's score is just a sum over its
10 already-computed pair scores - avoids repeating the same matrix
multiply for the same pair across every combo that contains it.
Not AI: the additivity model itself (CLAUDE.md's, defended in the module
docstring above), which pairs count (all C(5,2)=10, no double-counting).
"""


def precompute_pair_scores(recipe_matrix, player_ids, M):
    """All pairwise p_x.T @ M @ p_y scores, once. recipe_matrix: (n, k),
    rows aligned with player_ids. Returns {(id_a, id_b): score} for a < b.
    """
    n = len(player_ids)
    scores = {}
    for a, b in combinations(range(n), 2):
        s = float(recipe_matrix[a] @ M @ recipe_matrix[b])
        key = tuple(sorted((player_ids[a], player_ids[b])))
        scores[key] = s
    return scores


def score_all_combos(roster_recipes, M):
    """roster_recipes: DataFrame with PLAYER_ID, PLAYER_NAME, arch_0..arch_{k-1}.
    Returns every 5-man combo from the roster, ranked by summed pair score.
    """
    ids = roster_recipes["PLAYER_ID"].tolist()
    names = dict(zip(roster_recipes["PLAYER_ID"], roster_recipes["PLAYER_NAME"]))
    arch_cols = [c for c in roster_recipes.columns if c.startswith("arch_")]
    recipe_matrix = roster_recipes[arch_cols].values

    pair_scores = precompute_pair_scores(recipe_matrix, ids, M)

    results = []
    for combo in combinations(ids, 5):
        total = sum(pair_scores[tuple(sorted(pair))] for pair in combinations(combo, 2))
        results.append({
            "players": " / ".join(names[i] for i in combo),
            "player_ids": combo,
            "score": total,
        })

    out = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"scored {len(out)} five-man combinations from {len(ids)} roster players "
          f"(C({len(ids)},5)={len(out)})")
    return out


# --- Part D: rank + report -------------------------------------------------

def report_top(ranked, n=10):
    print(f"\ntop {n} lineups:")
    print(ranked.head(n)[["players", "score"]].to_string(index=False))
    print(f"\nbottom {min(n, 3)} lineups (sanity check - should look like real bad-fit combos, not noise):")
    print(ranked.tail(min(n, 3))[["players", "score"]].to_string(index=False))


# --- Phase runner ------------------------------------------------------

def phase_score(basis_dir=None, matrix_path=None, season="2025-26"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from step1_archetypes_model import load_basis

    basis_dir = Path(basis_dir) if basis_dir else DATA_DIR / "basis_full"
    matrix_path = Path(matrix_path) if matrix_path else DATA_DIR / "synergy_matrix_v1.npy"

    basis = load_basis(basis_dir)
    k = basis["k"]
    recipes = pd.read_csv(basis_dir / "recipes.csv")
    M = np.load(matrix_path)

    roster = resolve_roster(season=season)
    roster = attach_recipes(roster, recipes, k, season=season)

    ranked = score_all_combos(roster, M)
    ranked.to_csv(DATA_DIR / "nets_lineup_rankings.csv", index=False)
    report_top(ranked)
    return ranked


if __name__ == "__main__":
    # phase_score()  # once step1's basis_full and step2's synergy_matrix_v1.npy exist
    pass
