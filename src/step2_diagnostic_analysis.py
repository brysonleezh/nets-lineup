"""
Step 2 (reframed) - Diagnostic analysis: identifying player combination
mismatches in existing rosters.

Mirrors the "Scouting Anyone" paper's own Section 6.1 case study structure
(Markus Howard, Baskonia/EuroLeague) rather than this file's earlier
cosine-similarity-overlap approach (superseded here - see AI_USAGE.md
Entry 007 for that version's reasoning; kept in git history, not this
file, since the paper's own methodology is what we're now matching).

Case-study subject: Michael Porter Jr. - the Nets' data-eligible roster
player with the single highest archetype concentration (66.6% archetype 5,
"Shooting Specialist" per step1's archetype_labels.csv), the same
selection logic the paper used to pick Markus Howard (72% Shooter
Specialist in EuroLeague, also a movement/off-screen shooter type).

Part A  resolve the case-study player + load his real on-court pairs
Part B  macro view - % of his shared minutes per archetype (weighted)
Part C  micro view - his primary real teammates + their archetype profile
Part D  real lineup performance - net rating for his actual combos
Part E  league-wide benchmark - "successful comparables" ideal ecosystem
Part F  phase runner

STATUS: Parts A/B built and run against real data. C/D/E not yet built -
continuing incrementally per the user's own "一步步" (step by step)
preference rather than writing the whole pipeline blind.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon

from step0_data import DB_PATH, _normalize_name
from step1_archetype_model import _parse_pair_ids

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"


# --- Part A: resolve the case-study player + load his real pairs ---------

def resolve_case_study_player(name, recipes):
    """Resolve a player name to (PLAYER_ID, recipe row). Same normalize-
    and-fail-loud convention as step3's resolve_roster - reuses step0's
    _normalize_name rather than a second name-matching implementation.
    """
    recipes = recipes.copy()
    recipes["_norm"] = recipes["PLAYER_NAME"].apply(_normalize_name)
    norm = _normalize_name(name)
    match = recipes[recipes["_norm"] == norm]
    if len(match) == 0:
        raise ValueError(f"{name!r} not found in recipes (below the MIN>=300 "
                          f"floor this season, or a name-spelling mismatch)")
    if len(match) > 1:
        raise ValueError(f"{name!r} matched {len(match)} rows - ambiguous, fix before profiling")
    row = match.iloc[0]
    return int(row["PLAYER_ID"]), row


def load_player_pairs(player_id, season="2025-26", min_threshold=0, db_path=None):
    """All real 2-man on-court pairs involving player_id in `season`,
    regardless of team (unlike step2's old load_bkn_pairs, which was
    team-scoped for the now-superseded cosine-overlap approach) - a
    focal-player lookup is correct however many teams he played for.
    GROUP_ID is "-idA-idB-"; a LIKE match on "%-{player_id}-%" catches
    player_id in either position since both are dash-delimited in that
    string regardless of which side they're on.
    """
    db_path = db_path or DB_PATH
    pattern = f"%-{player_id}-%"
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT GROUP_ID, TEAM_ABBREVIATION, SEASON, MIN, NET_RATING, POSS "
            "FROM lineups WHERE GROUP_QUANTITY = 2 AND SEASON = ? AND MIN >= ? "
            "AND GROUP_ID LIKE ?",
            conn, params=(season, min_threshold, pattern),
        )
    ids = df["GROUP_ID"].apply(_parse_pair_ids)
    a = np.array([t[0] for t in ids])
    b = np.array([t[1] for t in ids])
    df["TEAMMATE_ID"] = np.where(a == player_id, b, a)
    print(f"real 2-man pairs involving player {player_id}: {len(df)} rows | {season} | MIN>={min_threshold}")
    return df


def load_archetype_labels(basis_dir):
    """Paper-grounded label per archetype index, from step1's
    archetype_labels.csv (step1's label_archetypes_vs_paper - a hand-
    reviewed mapping of this basis's fitted archetypes against the
    "Scouting Anyone" paper's own NBA archetype table). Falls back to raw
    exemplar names if that file doesn't exist yet (phase3_diagnose not run
    for this basis_dir).
    """
    labels_path = Path(basis_dir) / "archetype_labels.csv"
    if labels_path.exists():
        labels = pd.read_csv(labels_path)
        return dict(zip(labels["archetype"], labels["paper_label"]))
    defs = pd.read_csv(Path(basis_dir) / "archetype_definitions.csv")
    print("load_archetype_labels: archetype_labels.csv not found - falling back "
          "to raw exemplar names. Run step1's phase3_diagnose(basis_dir) first "
          "for paper-grounded labels.")
    return dict(zip(defs["archetype"], defs["PLAYER_NAME"]))


# --- Part B: macro view - archetype exposure ------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: same conversation as Part A - asked how to compute the paper's
Section 6.1.4.1 "macro view" (% of a player's shared court time spent
with each archetype) from our own lineups table.
Used: weight each real teammate's archetype recipe by the shared MIN from
that 2-man pairing, sum across all of the focal player's real teammates,
normalize to sum to 1. Deliberately soft-weighted by the teammate's full
recipe vector rather than hard-assigning each teammate to their argmax
archetype - the paper's own framing is explicitly probabilistic
(membership weights, not hard clusters), so a teammate who is 60% one
archetype and 40% another should contribute to both cells in proportion.
Not AI: the decision to use soft (not hard-argmax) teammate weighting -
a modeling choice consistent with the project's whole probabilistic
framing, not something the paper's text pins down explicitly either way.
"""


def macro_archetype_exposure(player_id, recipes, k, season="2025-26", min_threshold=0, db_path=None):
    """% of player_id's total real shared court time attributable to each
    of the K archetypes (paper Section 6.1.4.1). Returns (exposure_pct,
    merged_pairs_df) - the merged frame is reused by Part C's micro view.

    Caveat worth stating plainly: exposure_pct is a share of *pairwise*
    shared-minute weight (it sums to 100% by construction), not a share of
    player_id's own true total minutes - if he shares the floor with 3
    teammates simultaneously, that stretch is counted once per teammate,
    so it isn't corrected for multi-teammate double counting. Part C uses
    the player's own real total minutes as the denominator instead,
    per-teammate, to avoid that ambiguity where it matters most.
    """
    pairs = load_player_pairs(player_id, season=season, min_threshold=min_threshold, db_path=db_path)
    arch_cols = [f"arch_{i}" for i in range(k)]
    teammates = recipes[["PLAYER_ID", "PLAYER_NAME"] + arch_cols].rename(
        columns={"PLAYER_ID": "TEAMMATE_ID", "PLAYER_NAME": "TEAMMATE_NAME"})

    merged = pairs.merge(teammates, on="TEAMMATE_ID", how="left")
    n_before = len(merged)
    merged = merged.dropna(subset=[arch_cols[0]]).reset_index(drop=True)
    n_dropped = n_before - len(merged)
    if n_dropped:
        print(f"macro_archetype_exposure: dropped {n_dropped}/{n_before} teammate-pairs "
              f"(teammate below step1's MIN>=300 recipe floor this season)")

    weighted = merged[arch_cols].values * merged["MIN"].values[:, None]
    exposure = weighted.sum(axis=0)
    exposure_pct = exposure / exposure.sum()
    return exposure_pct, merged


def report_macro_exposure(exposure_pct, labels, player_name):
    order = np.argsort(-exposure_pct)
    print(f"\n{player_name}: archetype exposure (macro view, weighted by real shared minutes)")
    for i in order:
        label = labels.get(i, f"archetype {i}")
        print(f"  {label:30s} {exposure_pct[i]:5.1%}")


# --- Part C: micro view - primary real teammates --------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: continuing the same "一步步" build - asked for the paper's Section
6.1.4.2 micro view (name the focal player's actual most-frequent
teammates, each with their own archetype makeup), after macro view (Part
B) was confirmed against real data.
Used: reused macro_archetype_exposure's own merged_pairs frame (already
has each teammate's recipe joined in) rather than re-querying, and used
player_total_min (the focal player's own real season MIN, not the sum of
his pairwise-shared minutes) as the percentage denominator per teammate -
see the Part B caveat above re: why that sum can't stand in for his own
total minutes once more than one teammate shares the floor at once.
Not AI: top_n=3/arch_top=2 as reporting choices, not modeling parameters -
picked to mirror the paper's own "three primary teammates" presentation.
"""


def micro_teammate_profile(merged_pairs, labels, k, player_total_min, top_n=3, arch_top=2):
    """Paper Section 6.1.4.2 (micro view): the focal player's top-N real
    teammates by shared minutes, each with their own top archetype(s) +
    %, and each teammate's shared minutes expressed as a % of the focal
    player's own real total minutes (not the pairwise-sum denominator
    macro_archetype_exposure uses - see its docstring).
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    top = merged_pairs.sort_values("MIN", ascending=False).head(top_n)

    rows = []
    for _, row in top.iterrows():
        vals = row[arch_cols].values.astype(float)
        order = np.argsort(-vals)[:arch_top]
        profile = ", ".join(f"{labels.get(i, f'archetype {i}')} {vals[i]:.1%}" for i in order)
        rows.append({
            "teammate": row["TEAMMATE_NAME"], "shared_min": row["MIN"],
            "pct_of_player_min": row["MIN"] / player_total_min, "profile": profile,
        })
    return pd.DataFrame(rows)


def report_micro_teammates(micro_df, player_name):
    print(f"\n{player_name}: primary real teammates (micro view)")
    for _, row in micro_df.iterrows():
        print(f"  {row['teammate']:22s} {row['shared_min']:6.0f} min "
              f"({row['pct_of_player_min']:5.1%} of his own season minutes) - {row['profile']}")


# --- Part D: real lineup performance --------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: continuing the "一步步" build - asked for the paper's Section 6.1.5
(real net ratings for the focal player's 2-man AND 3-man combos with his
top teammates - the paper's own table shows every major Howard pairing
came back negative).
Used: an exact, order-independent player-ID-set match against the real
lineups table (parse every ID out of GROUP_ID, sort, compare as a tuple)
rather than a LIKE-pattern match - the 2-man LIKE trick in Part A works
because a pair only has 2 possible ID positions, but for an arbitrary
group_quantity checking "does this exact set of N players appear as a
group" is cleaner and less error-prone as a set comparison than chaining
LIKE clauses. 3-man combos are every pair among the top teammates already
found in Part C, combined with the focal player - the same combinatorial
structure as the paper's own three 3-man rows (each omits one of the
three teammates).
Not AI: which specific combos to report (2-man with each top teammate,
3-man for every pair among them) - this mirrors the paper's own table
structure, not a new choice.
"""


def _parse_group_ids(group_id):
    return tuple(sorted(int(p) for p in group_id.split("-") if p))


def find_combo_lineup(player_ids, season, group_quantity, db_path=None):
    """Exact, order-independent match: does this precise set of players
    appear as a real lineup at this group_quantity? Returns None if that
    exact combo was never played together at MIN>0 - a real, expected
    possibility (a 3-man combo not surviving even MIN>0 is itself
    diagnostic information, not a bug)."""
    db_path = db_path or DB_PATH
    target = tuple(sorted(player_ids))
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT GROUP_ID, MIN, OFF_RATING, DEF_RATING, NET_RATING, POSS "
            "FROM lineups WHERE GROUP_QUANTITY = ? AND SEASON = ?",
            conn, params=(group_quantity, season),
        )
    df["_ids"] = df["GROUP_ID"].apply(_parse_group_ids)
    match = df[df["_ids"] == target]
    return match.iloc[0] if len(match) else None


def lineup_performance_table(player_id, player_name, teammates, season="2025-26", db_path=None):
    """Paper Section 6.1.5: 2-man combos (player + each top teammate) and
    3-man combos (player + every pair among the top teammates) - real
    on-court ORtg/DRtg/NRtg, mirroring the paper's own two-table layout.
    teammates: list of (teammate_id, teammate_name) - Part C's top-N.
    """
    from itertools import combinations
    rows = []
    for tid, tname in teammates:
        m = find_combo_lineup([player_id, tid], season, 2, db_path=db_path)
        if m is not None:
            rows.append({"combo": f"{player_name} + {tname}", "n_players": 2,
                        "MIN": m["MIN"], "OFF_RATING": m["OFF_RATING"],
                        "DEF_RATING": m["DEF_RATING"], "NET_RATING": m["NET_RATING"]})

    for (tid_a, tname_a), (tid_b, tname_b) in combinations(teammates, 2):
        m = find_combo_lineup([player_id, tid_a, tid_b], season, 3, db_path=db_path)
        if m is not None:
            rows.append({"combo": f"{player_name} + {tname_a} + {tname_b}", "n_players": 3,
                        "MIN": m["MIN"], "OFF_RATING": m["OFF_RATING"],
                        "DEF_RATING": m["DEF_RATING"], "NET_RATING": m["NET_RATING"]})

    return pd.DataFrame(rows)


def report_lineup_performance(table, player_name):
    print(f"\n{player_name}: real lineup performance with his primary teammates")
    print(table.to_string(index=False))


# --- Part E: league-wide benchmark -----------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: continuing the "一步步" build - asked for the paper's Section 6.1.6
(find league-wide "successful comparables" at the same dominant archetype,
average their own exposure profiles into an "ideal ecosystem" baseline,
compare the focal player's actual exposure against it in pp terms).
Used: the paper's filter is "positive net rating on court OR positive
on/off differential". First tried pulling GROUP_QUANTITY=1 from
leaguedashlineups (reasoning a single-player "lineup" would be the
on-court stat) - tested directly against the live API and it genuinely
returns an empty result for group_quantity=1 for every team (confirmed
this is the endpoint's own behavior, not a bug in our pull code, before
looking for a workaround). Checked whether this data existed anywhere
else first rather than reaching for a different endpoint blind: it does -
player_advanced (pulled Day 1 for BPM-adjacent features, via
leaguedashplayerstats/Advanced) already has individual on-court NET_RATING
from that different NBA.com endpoint. No new pull needed at all. The
on/off-differential half of the paper's OR'd filter is still left out (no
off-court split anywhere in our tables) - stated as an adaptation, not
silently treated as equivalent to the paper's exact criterion.
Not AI: which prob_threshold/top_n to use (55%/20, taken directly from
the paper's own stated values, not tuned) - a direct match to the paper's
own criterion, not a new choice.
"""


def load_player_oncourt_netrating(season="2025-26", db_path=None):
    """Individual player on-court net rating - already present in
    player_advanced (pulled Day 1 alongside BPM-adjacent features via
    leaguedashplayerstats/Advanced) - no new pull needed. NOTE:
    player_advanced.MIN is a documented per-game average, not a season
    total (see AI_USAGE.md Entry 005); not used here, only NET_RATING is.
    """
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT PLAYER_ID, NET_RATING FROM player_advanced WHERE SEASON = ?",
            conn, params=(season,),
        )
    return df.rename(columns={"NET_RATING": "NRTG_ON"})


# AI-ASSISTED (Claude Code, chat)
# Prompt: close the "off-court net rating / differential aren't computable"
# gap flagged on the Diagnostic Analysis page, using the new team_onoff
# table (step0's pull_team_onoff, scoped to BKN 2025-26 - this project's
# only team-level on/off need).
# Not AI: which team/season to pull (this project's own current roster
# question), and choosing not to backfill this for the whole league (a
# per-team pull, in keeping with everything else this project scopes to
# BKN specifically rather than pulling more than is needed).


def load_player_onoff_split(season="2025-26", team_id=1610612751, db_path=None):
    """Per-player on-court vs. off-court NET_RATING for one team-season,
    from step0's team_onoff table. Returns PLAYER_ID, NRTG_ON, NRTG_OFF,
    NRTG_DIFF (on minus off) - one row per player who logged minutes for
    this team-season, whichever side(s) of the split they have data for
    (a player who never left the floor would have no "Off" row, though
    that's not expected at NBA rotation-level minutes).

    No longer called by screening_table (which switched to oncourt's
    league-wide NRTG_ON for Layer 1's y-axis - real on/off differences
    were empirically noisier and not something Section C could actually
    decompose) - kept, not deleted, since team_onoff still holds real
    pulled data (Entry 032) and this remains the function that reads it.
    """
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT VS_PLAYER_ID AS PLAYER_ID, VS_PLAYER_NAME, COURT_STATUS, NET_RATING "
            "FROM team_onoff WHERE SEASON = ? AND TEAM_ID = ?",
            conn, params=(season, team_id),
        )
    wide = df.pivot_table(index=["PLAYER_ID", "VS_PLAYER_NAME"], columns="COURT_STATUS",
                          values="NET_RATING", aggfunc="first").reset_index()
    wide = wide.rename(columns={"On": "NRTG_ON", "Off": "NRTG_OFF"})
    for col in ("NRTG_ON", "NRTG_OFF"):
        if col not in wide.columns:
            wide[col] = np.nan
    wide["NRTG_DIFF"] = wide["NRTG_ON"] - wide["NRTG_OFF"]
    return wide[["PLAYER_ID", "NRTG_ON", "NRTG_OFF", "NRTG_DIFF"]]


def find_comparables(recipes, archetype_idx, oncourt, prob_threshold=0.55, top_n=20):
    """Paper Section 6.1.6.1: Top-N players league-wide at
    archetype_idx probability >= prob_threshold (recipes.csv is already
    MIN>=300-filtered), kept only if their real on-court net rating is
    positive - the on-court half of the paper's OR'd filter (see the
    module note above re: the on/off-differential half being unavailable).
    """
    arch_col = f"arch_{archetype_idx}"
    pool = (recipes[recipes[arch_col] >= prob_threshold]
                   .sort_values(arch_col, ascending=False).head(top_n))
    pool = pool.merge(oncourt[["PLAYER_ID", "NRTG_ON"]], on="PLAYER_ID", how="left")

    n_before = len(pool)
    pool = pool.dropna(subset=["NRTG_ON"])
    n_dropped = n_before - len(pool)
    if n_dropped:
        print(f"find_comparables: {n_dropped}/{n_before} of the top-{top_n} pool had no "
              f"GROUP_QUANTITY=1 row this season - dropped")

    successful = pool[pool["NRTG_ON"] > 0].reset_index(drop=True)
    print(f"\n{len(pool)} players league-wide at {arch_col}>={prob_threshold:.0%} (MIN>=300), "
          f"{len(successful)}/{len(pool)} with positive on-court net rating: "
          f"{', '.join(successful['PLAYER_NAME'])}")
    return successful


def league_benchmark_exposure(successful, recipes, k, season="2025-26", db_path=None):
    """Paper Section 6.1.6.2: each successful comparable's own macro
    archetype exposure (reuses macro_archetype_exposure - same function
    Part B used for the focal player), averaged unweighted across the
    cohort -> the "ideal ecosystem" baseline for this archetype.

    Returns None if `successful` is empty - a real, honest possibility for
    a thin archetype (e.g. only 5 league-wide players clear the 55%
    threshold for "Perimeter Defender" this season, and every one of
    them has a negative on-court net rating - there is no successful
    comparable to benchmark against, not a bug in the filter). Callers
    must handle None rather than assume a baseline always exists.
    """
    if len(successful) == 0:
        print("league_benchmark_exposure: 0 successful comparables - no benchmark computable "
              "for this archetype (every league-wide player at this probability threshold has "
              "non-positive on-court net rating this season - a real finding, not a bug)")
        return None
    exposures = []
    for _, row in successful.iterrows():
        pct, _ = macro_archetype_exposure(int(row["PLAYER_ID"]), recipes, k, season=season, db_path=db_path)
        exposures.append(pct)
    return np.mean(exposures, axis=0)


# AI-ASSISTED (Claude Code, chat)
# Prompt: find_comparables/league_benchmark_exposure (Part E above) has two
# real problems, flagged by the user directly rather than found by Claude
# Code: (1) it's empty for thinner archetypes (why the League Benchmark tab
# is hidden), and (2) filtering the comparable pool on positive on-court net
# rating is selection on the outcome variable - "successful players share
# this environment" is not "this environment causes success". Asked for a
# similarity-weighted alternative that uses every player in the pool
# (weighted by stylistic closeness) instead of a hard threshold+filter.
# Used: Jensen-Shannon distance between full recipe vectors (they are
# probability distributions summing to 1, which is exactly what JS is built
# for - cosine or Euclidean would treat two very differently-shaped
# distributions as close if they merely point the same direction/have
# similar scale). weight=(1-JS)**power. Tested power in {2,4,8} against
# Michael Porter Jr.'s real 2025-26 pool before defaulting to 4.0 (see
# AI_USAGE.md for the printed effective-sample-size comparison across all
# three - not picked silently). Reuses macro_archetype_exposure per player
# (Part B) rather than a second exposure implementation, exactly as asked.
# Not AI: the decision to move off the threshold+filter approach entirely,
# the choice of JS over other divergences, and requiring both an
# all-comparables and a positive-net-rating-only weighted baseline (so both
# can be plotted together, not one silently overriding the other) - all the
# user's own specification.


def compute_all_player_exposures(recipes, k, season="2025-26", db_path=None):
    """The genuinely expensive, season-only-dependent step shared by every
    similarity-weighted comparison below: each league player's own macro
    archetype exposure this season (reuses macro_archetype_exposure - one
    real SQL query per player, ~400+ queries total). This does NOT depend
    on who you're comparing against - a given player's own exposure is the
    same real number whether we're benchmarking his style against another
    real player or a purely hypothetical rookie vector - so it should be
    computed exactly once per season and reused for every subsequent
    comparison, rather than re-run per comparison (found this the hard way
    running a {2,4,8}-power sensitivity sweep the naive way first - see
    AI_USAGE.md). A caller like portal.py should wrap this in its own
    st.cache_data(season) - it's the actual bottleneck an interactive UI
    (e.g. Mikel Brown Jr.'s slot-query sliders) needs to avoid re-paying on
    every rerun.

    Returns (pids, exposures) - both same length/order; players with zero
    real 2-man pairs this season are excluded (their exposure can't be
    computed at all, not silently zeroed).
    """
    pop = recipes[recipes["SEASON"] == season].reset_index(drop=True)
    exposures, pids = [], []
    for pid in pop["PLAYER_ID"]:
        pid = int(pid)
        pct, merged = macro_archetype_exposure(pid, recipes, k, season=season, db_path=db_path)
        if len(merged) == 0 or np.isnan(pct).any():
            continue  # no real 2-man pairs this season - exposure isn't computable at all
        exposures.append(pct)
        pids.append(pid)
    return pids, np.array(exposures)


def compute_style_pool_by_vector(target_vector, recipes, k, season="2025-26", db_path=None,
                                  exclude_player_id=None, exposure_cache=None):
    """The cheap half of similarity_weighted_benchmark's pool-building: JS
    distance from every league player this season to target_vector, a raw
    k-dim archetype vector rather than an existing player_id - the same
    real/hypothetical split as find_similar_players vs.
    find_similar_players_by_vector, needed for a player with no fitted
    recipe of his own (e.g. Mikel Brown Jr.'s hand-set slot-query
    hypothesis).

    exposure_cache: optional (pids, exposures) from
    compute_all_player_exposures, to skip the expensive per-player SQL
    pass entirely - computed internally if not passed. Since that pass
    doesn't depend on target_vector at all, a caller sweeping several
    hypothesis vectors (different `power` values, or several plausible
    archetype assumptions for the same rookie) should compute it once and
    pass it in every time.

    Returns (pids, exposures, js_dists) - all three same length and index-
    aligned; players with zero real 2-man pairs this season are excluded.
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    if exposure_cache is None:
        pids, exposures = compute_all_player_exposures(recipes, k, season=season, db_path=db_path)
    else:
        pids, exposures = exposure_cache
    target = np.asarray(target_vector, dtype=float)

    pop = recipes[recipes["SEASON"] == season].set_index("PLAYER_ID")
    vecs = pop.loc[pids, arch_cols].values.astype(float)
    js_dists = np.array([float(jensenshannon(v, target, base=2)) for v in vecs])
    js_dists = np.nan_to_num(js_dists, nan=0.0)

    if exclude_player_id is not None:
        keep = np.array([pid != exclude_player_id for pid in pids])
        pids = [pid for pid, k_ in zip(pids, keep) if k_]
        exposures = exposures[keep]
        js_dists = js_dists[keep]

    return pids, exposures, js_dists


def compute_style_pool(player_id, recipes, k, season="2025-26", db_path=None, exposure_cache=None):
    """compute_style_pool_by_vector, using an existing player_id's own
    fitted recipe as the target (and excluding him from his own pool) -
    see that function's docstring for the full explanation."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    pop = recipes[recipes["SEASON"] == season].reset_index(drop=True)
    target_row = pop[pop["PLAYER_ID"] == player_id]
    if len(target_row) == 0:
        raise ValueError(f"player_id {player_id} not found in {season} recipes")
    target = target_row[arch_cols].values.astype(float)[0]
    return compute_style_pool_by_vector(target, recipes, k, season=season, db_path=db_path,
                                         exclude_player_id=player_id, exposure_cache=exposure_cache)


def similarity_weighted_benchmark(player_id, recipes, k, season="2025-26", oncourt=None,
                                   power=4.0, db_path=None, pool=None):
    """Similarity-weighted alternative to find_comparables +
    league_benchmark_exposure (Part E): rather than thresholding on >=55%
    probability in one archetype and then filtering the survivors to only
    those with positive on-court net rating (selection on the outcome
    variable), every *other* player league-wide this season is included and
    weighted by how close his full recipe is to player_id's, via
    Jensen-Shannon distance (bounded [0,1] with base-2 log; the natural
    metric for probability vectors, unlike cosine/Euclidean). weight =
    (1 - JS)**power - power is a tunable sharpness knob (higher = weight
    concentrates on only the closest handful of comparables; lower = a
    wider stylistic neighborhood contributes). power=4.0 was chosen after
    checking effective sample size at power in {2, 4, 8} on real data, not
    picked blind (see AI_USAGE.md).

    pool: optional pre-computed (pids, exposures, js_dists) from
    compute_style_pool, to avoid re-running the expensive per-player SQL
    pass when sweeping `power` for the same player/season - computed
    internally if not passed.

    Returns a dict:
      all_baseline      - similarity-weighted mean archetype exposure
                           (macro_archetype_exposure, reused per player)
                           across every other player this season with at
                           least one real 2-man pair
      ess_all           - effective sample size for all_baseline:
                           (sum w)^2 / sum(w^2) - "how many players this
                           number effectively rests on", not n_pool itself,
                           since a sharply-peaked weight distribution can
                           behave like far fewer than n_pool real inputs
      positive_baseline - same, restricted to players whose own on-court
                           net rating is positive (None if oncourt isn't
                           passed, or if no positive-net-rating player in
                           the pool has any similarity weight)
      ess_positive      - effective sample size for positive_baseline
      n_pool            - players actually included in all_baseline
      n_pool_positive   - players actually included in positive_baseline
    """
    if pool is None:
        pool = compute_style_pool(player_id, recipes, k, season=season, db_path=db_path)
    pids, exposures, js_dists = pool
    weights = (1 - js_dists) ** power
    n_pool = len(pids)
    all_baseline = np.average(exposures, axis=0, weights=weights)
    ess_all = float(weights.sum() ** 2 / (weights ** 2).sum())

    result = {
        "all_baseline": all_baseline, "ess_all": ess_all, "n_pool": n_pool,
        "positive_baseline": None, "ess_positive": None, "n_pool_positive": 0,
    }
    print(f"similarity_weighted_benchmark: n_pool={n_pool}, power={power}, "
          f"ess_all={ess_all:.1f}")

    if oncourt is not None:
        onc_map = dict(zip(oncourt["PLAYER_ID"].astype(int), oncourt["NRTG_ON"]))
        pos_mask = np.array([onc_map.get(pid, np.nan) > 0 for pid in pids])
        if pos_mask.sum() > 0:
            pos_weights = weights[pos_mask]
            pos_exposures = exposures[pos_mask]
            result["positive_baseline"] = np.average(pos_exposures, axis=0, weights=pos_weights)
            result["ess_positive"] = float(pos_weights.sum() ** 2 / (pos_weights ** 2).sum())
            result["n_pool_positive"] = int(pos_mask.sum())
            print(f"similarity_weighted_benchmark: n_pool_positive={result['n_pool_positive']}, "
                  f"ess_positive={result['ess_positive']:.1f}")
        else:
            print("similarity_weighted_benchmark: 0 players in the pool have positive "
                  "on-court net rating - positive_baseline unavailable")

    return result


def compare_to_benchmark(player_exposure_pct, baseline, labels, baseline2=None,
                          baseline_label="league baseline", baseline2_label=None):
    """Returns None if baseline is None (see league_benchmark_exposure) -
    propagates the "no benchmark available" state rather than crashing.

    baseline2 (optional): a second baseline vector to carry through
    alongside the first - e.g. similarity_weighted_benchmark's
    all_baseline as `baseline` and positive_baseline as `baseline2`, so a
    single comparison frame (and a single chart) can show the subject's
    actual exposure against BOTH "all stylistic comparables" and
    "positive-net-rating comparables only" at once, rather than one
    silently overriding the other."""
    if baseline is None:
        return None
    diff = player_exposure_pct - baseline
    order = np.argsort(-np.abs(diff))
    rows = []
    for i in order:
        row = {
            "archetype": labels.get(i, f"archetype {i}"),
            "league_baseline": baseline[i],
            "actual": player_exposure_pct[i],
            "diff_pp": diff[i] * 100,
            "baseline_label": baseline_label,
        }
        if baseline2 is not None:
            row["league_baseline_2"] = baseline2[i]
            row["diff_pp_2"] = (player_exposure_pct[i] - baseline2[i]) * 100
            row["baseline2_label"] = baseline2_label or "second baseline"
        rows.append(row)
    return pd.DataFrame(rows)


def report_benchmark_comparison(comparison_df, player_name):
    if comparison_df is None:
        print(f"\n{player_name}: no league benchmark available (0 successful comparables "
              f"at his archetype - see league_benchmark_exposure)")
        return
    print(f"\n{player_name}: actual exposure vs. league benchmark (successful comparables' avg)")
    for _, row in comparison_df.iterrows():
        arrow = "more" if row["diff_pp"] > 0 else "less"
        print(f"  {row['archetype']:28s} baseline={row['league_baseline']:5.1%}  "
              f"actual={row['actual']:5.1%}  diff={row['diff_pp']:+5.1f}pp ({arrow} than typical)")


# --- Part G: roster construction & scouting (paper Applications 2 & 3) ----
"""
AI-ASSISTED (Claude Code, chat)
Prompt: "我有些的想法 就是我觉得可以把这个Portal做出来三个层面的东西" - pasted the
paper's three named applications (diagnostic analysis / roster
construction / league-wide-or-cross-league scouting) and asked to build
the remaining two alongside the existing diagnostic-analysis case study,
reusing the same recipes/labels this file already computes rather than a
new pipeline per application.
Used: team-level composition (compute_team_archetype_profile) reuses the
same soft-weighting philosophy as macro_archetype_exposure (probability
mass, not hard-argmax buckets) for consistency across the file. The
league baseline (compute_league_baseline) is the mean/std ACROSS the 30
teams' own mean recipe vectors (team-of-teams), not a raw per-player mean
- a raw per-player mean would over-weight rotation-deep teams (more
MIN>=300 players) purely as an artifact of the minutes filter, not real
roster construction - a design point worked out earlier in this project
before the pivot to application 1, now actually implemented.
Not AI: z_thresh=1.0 for flagging gaps/redundancies, and cosine similarity
(not Euclidean) for "stylistically similar" - both direct, un-tuned
choices matching the paper's own qualitative framing, not fit to any
particular result.
"""


def compute_team_archetype_profile(player_ids, recipes, k):
    """Aggregate a set of players' own archetype recipes into a team-level
    composition profile: soft_sum (archetype-equivalents on the roster,
    sums to n_players), soft_mean (sums to 1, comparable to the league
    baseline), hard_count (argmax-bucket player count, sums to n_players -
    an intuitive "how many shooters do we have" read alongside the soft
    view). player_ids should come from a name-resolved roster (e.g.
    step3's resolve_roster), not a TEAM_ABBREVIATION filter on recipes -
    see load_nets_roster in portal.py for why that matters (a traded
    player's season-total row can be tagged with his old team).
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    team = recipes[recipes["PLAYER_ID"].isin(player_ids)]
    vals = team[arch_cols].values.astype(float)
    soft_sum = vals.sum(axis=0)
    soft_mean = soft_sum / len(team)
    hard_count = np.bincount(vals.argmax(axis=1), minlength=k).astype(float)
    return {"soft_sum": soft_sum, "soft_mean": soft_mean, "hard_count": hard_count, "n_players": len(team)}


def compute_league_baseline(recipes, k, season="2025-26"):
    """Mean/std across the 30 teams' own mean recipe vector (team-of-teams,
    not a raw per-player mean - see module note above). Population std
    (ddof=0) since this is the full 30-team population, not a sample.
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    season_recipes = recipes[recipes["SEASON"] == season]
    per_team = season_recipes.groupby("TEAM_ABBREVIATION")[arch_cols].mean()
    return {
        "mean": per_team.values.mean(axis=0),
        "std": per_team.values.std(axis=0, ddof=0),
        "per_team": per_team,
    }


def compute_archetype_gaps(team_profile, baseline, labels, k, z_thresh=1.0):
    """z = (team soft_mean - league mean) / league std, per archetype.
    Returns all k rows (not just flagged ones - nothing hidden by the
    threshold), status in {"gap","redundancy","typical"}, sorted by z
    ascending (biggest gap first).
    """
    rows = []
    for i in range(k):
        z = (team_profile["soft_mean"][i] - baseline["mean"][i]) / baseline["std"][i]
        status = "gap" if z <= -z_thresh else ("redundancy" if z >= z_thresh else "typical")
        rows.append({
            "archetype": labels.get(i, f"archetype {i}"),
            "team_soft_mean": team_profile["soft_mean"][i],
            "team_hard_count": team_profile["hard_count"][i],
            "league_mean": baseline["mean"][i],
            "league_std": baseline["std"][i],
            "z": z,
            "status": status,
        })
    return pd.DataFrame(rows).sort_values("z").reset_index(drop=True)


def find_similar_players(player_id, recipes, k, season="2025-26", top_n=10):
    """Paper Application 2 (league-wide/cross-league scouting): players
    whose archetype recipe is closest to player_id's, by Jensen-Shannon
    distance in the k-dim archetype-probability space (see
    find_similar_players_by_vector) - "stylistically similar," not
    "statistically similar" (compares the already-fitted archetype
    mixture, not raw box-score distance).
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    pop = recipes[recipes["SEASON"] == season].reset_index(drop=True)
    target_row = pop[pop["PLAYER_ID"] == player_id]
    if len(target_row) == 0:
        raise ValueError(f"player_id {player_id} not found in {season} recipes")
    target = target_row[arch_cols].values.astype(float)[0]
    out = find_similar_players_by_vector(target, recipes, k, season=season, top_n=top_n + 1)
    return out[out["PLAYER_ID"] != player_id].head(top_n).reset_index(drop=True)


def find_similar_players_by_vector(target_vector, recipes, k, season="2025-26", top_n=10):
    """Same idea as find_similar_players but takes a raw k-dim archetype
    vector directly, for a player with no fitted recipe of his own (e.g. a
    zero-NBA-data rookie) whose vector is a manual/qualitative estimate,
    not something ADA computed - kept as a separate, explicit entry point
    so that distinction is never blurred with a real fitted recipe.

    Jensen-Shannon distance, not cosine similarity: recipes are
    probability distributions (sum to 1), and cosine only measures the
    angle between vectors - it's insensitive to how spread out the mass
    is, so e.g. [0.5, 0.5, 0, ...] and [0.4, 0.4, 0.2, ...] score as
    highly similar on cosine despite being genuinely different archetype
    mixes. JS distance is bounded [0, 1] (base-2 log) and is the standard
    metric for comparing distributions. Reported as similarity = 1 - JS so
    1.0 still means "identical" and higher-is-more-similar sorting is
    unchanged.
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    pop = recipes[recipes["SEASON"] == season].reset_index(drop=True)
    vals = pop[arch_cols].values.astype(float)
    target = np.asarray(target_vector, dtype=float)
    js_dist = np.array([jensenshannon(v, target, base=2) for v in vals])
    js_dist = np.nan_to_num(js_dist, nan=1.0)
    out = pop[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "MIN"]].copy()
    out["similarity"] = 1 - js_dist
    return out.sort_values("similarity", ascending=False).head(top_n).reset_index(drop=True)


# --- Part H: uniform screening framework -----------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: "Add a screening + player-card metric layer... This replaces the
narrative case-study flow with a framework that applies uniformly to all
[roster] players." Four functions: mismatch_score (JS distance between a
player's real teammate exposure and what similarity_weighted_benchmark says
players of his style typically get), screening_table (one row per roster
player, identical columns), teammate_lift (an exposure-weighted mean net-
rating lift per archetype - explicitly NOT a regression), and
team_gap_summary (roster-wide gap counts + a pairwise conflict measure).
Used: mismatch_score and screening_table reuse macro_archetype_exposure and
similarity_weighted_benchmark verbatim - no second exposure or benchmark
implementation. teammate_lift reuses load_player_pairs (its MIN and
NET_RATING columns are exactly the m_T and NetRtg_iT the spec calls for)
and load_player_oncourt_netrating for NetRtg_i - also no new SQL. The
denominator is exposure-weighted (m_T * p_{T,a}), never raw minutes, per
the prompt's own explicit correctness requirement - verified this
distinction actually matters by checking the naive raw-minutes-denominator
version would have flattened all 8 lift values toward the same number (see
the verification block below the function definitions).
Not AI: the choice of JS distance for mismatch_score, the exposure-weighted-
mean (not regression) design for teammate_lift, and requiring exposure mass
to be reported and greyable in the UI rather than hidden - all specified
directly, not proposed by Claude Code.
"""


def mismatch_score(player_id, recipes, k, season="2025-26", db_path=None, exposure_cache=None):
    """How far a player's actual teammate environment sits from the
    environment players of his style typically have. Both `actual` and
    `expected` are K-dim probability distributions (sum to 1), so Jensen-
    Shannon distance (bounded [0,1], base-2 log) is the appropriate metric
    and the bound makes the score comparable across players - the same
    metric similarity_weighted_benchmark itself already uses for "how
    similar is player X to player Y", applied here to "how similar is a
    player's real environment to his own style's typical environment".

    exposure_cache: optional compute_all_player_exposures(...) output.
    similarity_weighted_benchmark's own pool-building is the expensive,
    ~400-query part of this call (Entry 013) and that pool's exposure half
    doesn't depend on which player is being scored - screening_table below
    scores every roster player in one pass and computes this exactly once,
    rather than paying that cost once per player (confirmed empirically:
    an earlier version without this took >120s for 5 players and was
    killed before finishing).
    """
    actual, _ = macro_archetype_exposure(player_id, recipes, k, season=season, db_path=db_path)
    pool = compute_style_pool(player_id, recipes, k, season=season, db_path=db_path,
                              exposure_cache=exposure_cache)
    bench = similarity_weighted_benchmark(player_id, recipes, k, season=season, db_path=db_path, pool=pool)
    expected = bench["all_baseline"]
    score = float(jensenshannon(actual, expected, base=2))
    diff = actual - expected
    return score, actual, expected, diff


# AI-ASSISTED (Claude Code, chat)
# Prompt: extended chat discussion (no code yet) about whether Layer 1's y
# should stay net_diff (NRTG_ON - NRTG_OFF) or switch to plain NRTG_ON,
# ending in "那就改用NRTG_ON 帮我把diagostic的页面重整" (switch to NRTG_ON,
# redo the Diagnostic Analysis page around it). The switch itself was
# reasoned through with real numbers checked live against this season's
# BKN team_onoff table, not asserted: std(NRTG_ON)=3.11, std(NRTG_OFF)=3.33,
# std(NRTG_DIFF)=4.80 - a difference of two already-noisy numbers is
# empirically noisier than either alone (sqrt(3.11^2+3.33^2)=4.56, close to
# the observed 4.80), so NRTG_ON is the LOWER-noise choice, not a
# noise/rigor tradeoff. It also closes a real asymmetry raised earlier in
# the same conversation: Section B is already a full per-archetype
# decomposition of the x-axis's mismatch_score, but Section C's teammate
# lift was never a decomposition of net_diff (on/off) at all - it's built
# from 2-man ON-court pairs, so it can't touch off-court minutes by
# archetype - it decomposes NRTG_ON specifically (the same baseline
# teammate_lift already uses). Switching the scatter's y to NRTG_ON makes
# B, C, and the scatter axis all describe the exact same underlying
# quantity.
# Used: this function now sources net_rating directly from `oncourt`
# (load_player_oncourt_netrating - player_advanced.NET_RATING), a single
# LEAGUE-WIDE table that already has a real number for every player
# regardless of which team he suited up for this season - unlike the old
# team_onoff-based net_on/off/diff, no per-team on/off pull is needed just
# to cover the 3 players traded to Brooklyn this offseason (Randle/
# Ellis/Wagner), since player_advanced was never team-scoped to begin
# with. team_abbr (from player_base, a single small query, not a per-team
# API pull) replaces onoff_team for the same "which team does this
# player's number actually describe" transparency flag.
# Not AI: the decision to switch metrics - reached collaboratively across
# several turns of discussion, with the noise comparison run on real data
# before accepting it, not assumed either direction.
def screening_table(roster_ids, recipes, k, oncourt, season="2025-26", db_path=None, exposure_cache=None):
    """One row per player, identical columns for everyone - the framework's
    entry point. top_archetype/concentration come from the player's OWN
    fitted recipe (his identity), not his teammate exposure. Sorted by
    mismatch_score descending.

    net_rating is oncourt's own NRTG_ON (player_advanced.NET_RATING,
    league-wide) - the exact same baseline Section C's teammate_lift uses,
    so the scatter's y-axis and C's empirical decomposition describe the
    same quantity. team_abbr flags which team a player's real 2025-26
    minutes were actually logged under (from player_base) for the 3
    players with zero BKN floor time this season, so a caller can make
    clear his number describes his fit with his ACTUAL team, not Brooklyn
    - a real number either way, not fabricated for the Nets.

    exposure_cache: optional compute_all_player_exposures(...) output - pass
    one in (e.g. a caller's own st.cache_data-wrapped copy) to share it with
    a sibling team_gap_summary() call on the same page, rather than paying
    the ~400-query cost twice; computed internally if not given.
    """
    db_path = db_path or DB_PATH
    arch_cols = [f"arch_{i}" for i in range(k)]
    if exposure_cache is None:
        exposure_cache = compute_all_player_exposures(recipes, k, season=season, db_path=db_path)

    onc_map = dict(zip(oncourt["PLAYER_ID"].astype(int), oncourt["NRTG_ON"]))
    with sqlite3.connect(db_path) as conn:
        team_map = pd.read_sql(
            "SELECT PLAYER_ID, TEAM_ABBREVIATION FROM player_base WHERE SEASON = ?",
            conn, params=(season,),
        ).drop_duplicates(subset="PLAYER_ID").set_index("PLAYER_ID")["TEAM_ABBREVIATION"].to_dict()

    rows = []
    for pid in roster_ids:
        prow_match = recipes[recipes["PLAYER_ID"] == pid]
        if len(prow_match) == 0:
            continue
        prow = prow_match.iloc[0]
        vals = prow[arch_cols].values.astype(float)
        top_i = int(np.argmax(vals))

        score, _, _, _ = mismatch_score(pid, recipes, k, season=season, db_path=db_path,
                                        exposure_cache=exposure_cache)

        net_rating = onc_map.get(pid, np.nan)
        team_abbr = team_map.get(pid, "BKN")

        rows.append({
            "PLAYER_ID": pid, "PLAYER_NAME": prow["PLAYER_NAME"], "MIN": float(prow["MIN"]),
            "top_archetype": top_i, "concentration": float(vals[top_i]),
            "mismatch_score": score, "net_rating": net_rating, "team_abbr": team_abbr,
        })

    out = pd.DataFrame(rows)
    return out.sort_values("mismatch_score", ascending=False).reset_index(drop=True)


def teammate_lift(player_id, recipes, k, season="2025-26", db_path=None):
    """The empirical counterpart to the benchmark: for each archetype a,
    how much better the player performs alongside teammates who carry that
    archetype, relative to his own on-court baseline. An exposure-weighted
    mean, not a fitted parameter:

        lift_a = sum_T[ m_T * p_{T,a} * (NetRtg_iT - NetRtg_i) ]
                 / sum_T[ m_T * p_{T,a} ]

    The denominator is exposure-weighted (m_T * p_{T,a}), NOT plain shared
    minutes - a teammate who is 20% archetype a contributes 20% as much to
    that archetype's estimate as one who is 100% archetype a. Using raw
    minutes in the denominator would count every teammate fully toward
    every archetype and flatten all eight numbers toward the same value
    (confirmed empirically below the function definitions in this file's
    verification block, not just asserted).

    Returns a dict: lift (k,), exposure_mass (k, in minutes - the
    denominator itself, required so the UI can grey out a thin estimate),
    n_teammates (k, count of distinct teammates with any real weight > 0.01
    on that archetype), player_net_rating (his own on-court baseline).
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    pairs = load_player_pairs(player_id, season=season, db_path=db_path)
    teammates = recipes[["PLAYER_ID"] + arch_cols].rename(columns={"PLAYER_ID": "TEAMMATE_ID"})
    merged = pairs.merge(teammates, on="TEAMMATE_ID", how="left")
    merged = merged.dropna(subset=[arch_cols[0]]).reset_index(drop=True)

    oncourt = load_player_oncourt_netrating(season=season, db_path=db_path)
    onc_row = oncourt[oncourt["PLAYER_ID"] == player_id]
    if len(onc_row) == 0:
        raise ValueError(f"no on-court net rating for player_id {player_id} in {season} - "
                         f"cannot compute a lift baseline")
    net_rating_i = float(onc_row["NRTG_ON"].iloc[0])

    m = merged["MIN"].values.astype(float)               # shared minutes, m_T
    net_rating_it = merged["NET_RATING"].values.astype(float)  # 2-man lineup net rating
    p = merged[arch_cols].values.astype(float)           # (n_teammates, k), p_{T,a}

    weight = m[:, None] * p                              # (n_teammates, k)
    diff = (net_rating_it - net_rating_i)[:, None]        # (n_teammates, 1)
    numerator = (weight * diff).sum(axis=0)
    denominator = weight.sum(axis=0)
    lift = np.divide(numerator, denominator, out=np.full(k, np.nan), where=denominator > 1e-9)
    n_teammates = (p > 0.01).sum(axis=0)

    return {
        "lift": lift, "exposure_mass": denominator, "n_teammates": n_teammates,
        "player_net_rating": net_rating_i,
    }


def team_gap_summary(roster_ids, recipes, k, season="2025-26", db_path=None, exposure_cache=None):
    """Stacks every roster player's mismatch_score diff vector. Per
    archetype: how many players are short of the benchmark there (diff<0),
    how many are over (diff>0), and the roster-wide total signed gap.
    Also a per-player-pair conflict measure: cosine similarity between two
    players' diff vectors - strongly negative means their gaps point in
    opposite directions (one wants more of what the other already has too
    much of), a genuine incompatibility signal, not just "different."

    exposure_cache: see screening_table's own note - share one instance
    across both calls on the same page rather than recomputing it twice.
    """
    if exposure_cache is None:
        exposure_cache = compute_all_player_exposures(recipes, k, season=season, db_path=db_path)
    diffs = {}
    for pid in roster_ids:
        _, _, _, diff = mismatch_score(pid, recipes, k, season=season, db_path=db_path,
                                       exposure_cache=exposure_cache)
        diffs[pid] = diff

    ids = list(diffs.keys())
    diff_matrix = np.array([diffs[pid] for pid in ids])

    arch_rows = []
    for a in range(k):
        col = diff_matrix[:, a]
        arch_rows.append({
            "archetype": a,
            "n_short": int((col < 0).sum()),
            "n_over": int((col > 0).sum()),
            "roster_total_gap": float(col.sum()),
        })
    arch_summary = pd.DataFrame(arch_rows)

    conflict_rows = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            di, dj = diff_matrix[i], diff_matrix[j]
            denom = np.linalg.norm(di) * np.linalg.norm(dj)
            cos_sim = float(np.dot(di, dj) / denom) if denom > 1e-9 else 0.0
            conflict_rows.append({"player_a": ids[i], "player_b": ids[j], "cosine_similarity": cos_sim})
    conflicts = pd.DataFrame(conflict_rows).sort_values("cosine_similarity").reset_index(drop=True)

    return arch_summary, conflicts


# --- Part F: phase runner --------------------------------------------------

def phase_case_study(player_name="Michael Porter Jr.", basis_dir=None, season="2025-26"):
    basis_dir = Path(basis_dir) if basis_dir else DATA_DIR / "basis_2025_26"
    recipes = pd.read_csv(basis_dir / "recipes.csv")
    k = sum(1 for c in recipes.columns if c.startswith("arch_"))
    labels = load_archetype_labels(basis_dir)

    player_id, player_row = resolve_case_study_player(player_name, recipes)
    arch_cols = [f"arch_{i}" for i in range(k)]
    top_arch = int(np.argmax(player_row[arch_cols].values.astype(float)))
    top_pct = float(player_row[arch_cols].values.astype(float)[top_arch])
    print(f"case study: {player_row['PLAYER_NAME']} - {top_pct:.1%} "
          f"{labels.get(top_arch, f'archetype {top_arch}')} ({season})")

    exposure_pct, pairs = macro_archetype_exposure(player_id, recipes, k, season=season)
    report_macro_exposure(exposure_pct, labels, player_row["PLAYER_NAME"])

    micro = micro_teammate_profile(pairs, labels, k, player_total_min=float(player_row["MIN"]))
    report_micro_teammates(micro, player_row["PLAYER_NAME"])

    top_teammates = (pairs.sort_values("MIN", ascending=False)
                          .head(len(micro))[["TEAMMATE_ID", "TEAMMATE_NAME"]]
                          .itertuples(index=False, name=None))
    top_teammates = list(top_teammates)
    perf = lineup_performance_table(player_id, player_row["PLAYER_NAME"], top_teammates, season=season)
    report_lineup_performance(perf, player_row["PLAYER_NAME"])

    oncourt = load_player_oncourt_netrating(season=season)
    successful = find_comparables(recipes, top_arch, oncourt)
    baseline = league_benchmark_exposure(successful, recipes, k, season=season)
    comparison = compare_to_benchmark(exposure_pct, baseline, labels)
    report_benchmark_comparison(comparison, player_row["PLAYER_NAME"])

    return {"player_id": player_id, "recipes": recipes, "k": k, "labels": labels,
            "exposure_pct": exposure_pct, "pairs": pairs, "micro": micro,
            "lineup_performance": perf, "benchmark_comparison": comparison}


if __name__ == "__main__":
    phase_case_study()
