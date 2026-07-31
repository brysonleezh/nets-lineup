"""
Step 2b - per-player diagnostic report (Sections A-F).

Extends step2_diagnostic_analysis.py's screening/mismatch/teammate_lift
framework (Layer 1 + B/C, unchanged, reused here as Section C1/C2) with six
report sections for whichever player is selected in the portal's Diagnostic
Analysis screening layer. Nothing here re-implements macro_archetype_exposure,
similarity_weighted_benchmark, mismatch_score, teammate_lift, screening_table,
load_player_pairs, project(), or recipes_frame() - all reused directly from
step1_archetypes_model.py / step2_diagnostic_analysis.py.

Every claim this module produces is DESCRIPTIVE per CLAUDE.md's own
talent-vs-archetype design principle (see docs/RESEARCH_FINDINGS.md SS6) -
none of it is a validated predictor of wins. Gates fail loudly (raise/return
an explicit "insufficient/degraded" flag) rather than silently interpolating
or padding.

Phase 0   per-season recipe projection (2023-24, 2024-25) onto the FIXED
          2025-26 basis - never a re-fit. Gate 0a (feature availability),
          Gate 0b (projection smoothness on stable veterans).
Section A purity/entropy with league percentile
Section B role drift (B1) + age-matched development comps (B2)
Section C role elasticity (C3) - team shot-profile split by teammate
          presence; C1/C2 (mismatch/teammate_lift) stay in
          step2_diagnostic_analysis.py, just re-exported by the portal
Section D miscasting test (opportunity-only vs outcome-only partial
          projections onto the same fixed basis)
Section E signature radar vs. archetype centroid (E2 - value vs. salary -
          is out of scope: no salary table exists in this project, see
          DIAGNOSTICS_README.md)
Section F game-level bootstrap of the box-score-derived features only
          (season-level-only features held fixed - see DIAGNOSTICS_README.md
          for exactly which 4 features are actually resampled and why the
          other 25 are not)
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon

from step0_data_collect_process import DB_PATH
from step1_archetypes_model import load_basis, load_population, project, recipes_frame
from step2_diagnostic_analysis import load_player_pairs

REPO_ROOT = Path(__file__).resolve().parent.parent
BASIS_DIR = REPO_ROOT / "data" / "basis_2025_26"
STINTS_DIR = REPO_ROOT / "data" / "stints"
RAW_CACHE_DIR = STINTS_DIR / "raw_cache"

HIST_SEASONS = ["2023-24", "2024-25"]
ALL_SEASONS = HIST_SEASONS + ["2025-26"]


# ===========================================================================
# Phase 0 - per-season recipes on the fixed 2025-26 basis
# ===========================================================================

def historical_recipes_path(season, basis_dir=BASIS_DIR):
    return Path(basis_dir) / f"recipes_{season.replace('-', '_')}.csv"


def build_historical_recipes(basis_dir=BASIS_DIR, min_threshold=300, seasons=HIST_SEASONS):
    """Project each historical season onto the 2025-26 basis (fixed
    basis/mu/sd - never a re-fit; a player's role can differ season to
    season, which is exactly what Section B measures, so pooling seasons
    into one fit would erase the signal this section reports on).

    GATE 0a: every one of the basis's own feature columns (BBRef shot-
    location + Synergy playtype shares included) must be fully populated
    for the historical season's MIN>=min_threshold population - checked
    directly against real query results, not assumed from the fact that
    the multi-season RAPM work already pulled this data. Raises loudly on
    any missing feature rather than imputing.
    """
    fit = load_basis(basis_dir)
    out = {}
    for season in seasons:
        df = load_population(min_threshold=min_threshold, season_min=season, season_max=season)
        n_null = int(df[fit["feature_columns"]].isna().sum().sum())
        if n_null > 0:
            missing_cols = df[fit["feature_columns"]].isna().sum()
            missing_cols = missing_cols[missing_cols > 0]
            raise RuntimeError(
                f"GATE 0a FAILED for {season}: {n_null} missing feature values across "
                f"columns {missing_cols.to_dict()} - stop, do not impute. Check whether "
                f"BBRef shot-location/playtype data was actually pulled for this season."
            )
        P = project(df, fit["basis"], fit["mu"], fit["sd"], fit["feature_columns"])
        recipes = recipes_frame(df, P, fit["k"])
        path = historical_recipes_path(season, basis_dir)
        recipes.to_csv(path, index=False)
        out[season] = recipes
        print(f"GATE 0a passed for {season}: {n_null} missing feature values, "
              f"{len(recipes)} players projected -> {path}")
    return out


def gate0b_stability_check(recipes_all, k, min_min=1200, n_show=10):
    """Print (does not assert - a human judgment call, not a hard
    threshold) the 3-season recipe of the highest-minute players who
    appear in all 3 seasons, so a reader can see the projection evolves
    smoothly rather than jumping archetype wildly season to season -
    exactly the failure mode a normalization bug (mu/sd recomputed per
    season instead of reused from the 2025-26 basis) would produce."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    counts = recipes_all.groupby("PLAYER_ID").size()
    stable_ids = counts[counts == len(ALL_SEASONS)].index
    sub = recipes_all[recipes_all["PLAYER_ID"].isin(stable_ids)]
    min_by_player = sub.groupby("PLAYER_ID")["MIN"].min()
    show_ids = min_by_player[min_by_player >= min_min].sort_values(ascending=False).head(n_show).index

    for pid in show_ids:
        rows = recipes_all[recipes_all["PLAYER_ID"] == pid].sort_values("SEASON")
        name = rows.iloc[0]["PLAYER_NAME"]
        print(f"--- {name} ---")
        for _, r in rows.iterrows():
            top3 = sorted(range(k), key=lambda i: -r[f"arch_{i}"])[:3]
            s = ", ".join(f"arch{i}={r[f'arch_{i}']:.2f}" for i in top3)
            print(f"  {r['SEASON']} (MIN={r['MIN']:.0f}): {s}")
    return show_ids.tolist()


def load_all_season_recipes(basis_dir=BASIS_DIR):
    """Concatenate the 2025-26 production recipes with the two projected
    historical seasons - builds recipes_2023_24.csv/recipes_2024_25.csv
    via build_historical_recipes() first if they don't exist yet (Phase 0
    is a prerequisite, not an optional extra)."""
    frames = []
    cur = pd.read_csv(Path(basis_dir) / "recipes.csv")
    cur["SEASON"] = "2025-26"
    frames.append(cur)
    missing = [s for s in HIST_SEASONS if not historical_recipes_path(s, basis_dir).exists()]
    if missing:
        build_historical_recipes(basis_dir, seasons=missing)
    for season in HIST_SEASONS:
        df = pd.read_csv(historical_recipes_path(season, basis_dir))
        df["SEASON"] = season
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_player_bio_all_seasons(seasons=ALL_SEASONS, db_path=None):
    """Per-season AGE (and name) for every player - player_bio has one row
    per player-season, unlike load_player_bio's single-season query."""
    db_path = db_path or DB_PATH
    placeholders = ",".join("?" * len(seasons))
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            f"SELECT PLAYER_ID, PLAYER_NAME, SEASON, AGE FROM player_bio "
            f"WHERE SEASON IN ({placeholders})",
            conn, params=list(seasons),
        )


# ===========================================================================
# Section A - purity / entropy
# ===========================================================================

def purity_entropy(alpha_vec):
    """purity = his top-archetype share. entropy = normalized Shannon
    entropy of the recipe, [0,1] - 0 = a pure single archetype, 1 =
    maximally spread across all K."""
    alpha_vec = np.asarray(alpha_vec, dtype=float)
    k = len(alpha_vec)
    purity = float(alpha_vec.max())
    p = np.clip(alpha_vec, 1e-12, None)
    entropy = float(-(p * np.log2(p)).sum() / np.log2(k))
    return purity, entropy


def league_purity_entropy(recipes, k, season="2025-26"):
    """purity/entropy for every player in `season` - the reference
    distribution Section A's percentile and tercile labels are read off."""
    pop = recipes[recipes["SEASON"] == season]
    arch_cols = [f"arch_{i}" for i in range(k)]
    vals = pop[arch_cols].values.astype(float)
    purities = vals.max(axis=1)
    p = np.clip(vals, 1e-12, None)
    entropies = -(p * np.log2(p)).sum(axis=1) / np.log2(k)
    return purities, entropies


def percentile_rank(value, distribution):
    return float((np.asarray(distribution) < value).mean())


def tercile_label(value, distribution, low_label, mid_label, high_label):
    q1, q2 = np.quantile(distribution, [1 / 3, 2 / 3])
    if value <= q1:
        return low_label
    if value >= q2:
        return high_label
    return mid_label


# ===========================================================================
# Section B1 - role drift
# ===========================================================================

def role_drift(player_id, recipes_all, k, seasons=ALL_SEASONS, major_threshold=0.15):
    """His recipe across up to 3 seasons. `insufficient=True` (with
    whatever seasons ARE available) if he has <2 seasons of NBA data -
    never padded to look like a real trend."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    rows = []
    for s in seasons:
        m = recipes_all[(recipes_all["PLAYER_ID"] == player_id) & (recipes_all["SEASON"] == s)]
        if len(m):
            rows.append((s, m.iloc[0][arch_cols].values.astype(float)))

    if len(rows) < 2:
        return {
            "available_seasons": [r[0] for r in rows],
            "insufficient": True,
        }

    mat = np.stack([r[1] for r in rows])
    seasons_present = [r[0] for r in rows]
    major_idx = np.where((mat >= major_threshold).any(axis=0))[0].tolist()

    delta = mat[-1] - mat[0]
    riser_idx = int(np.argmax(delta))
    faller_idx = int(np.argmin(delta))

    return {
        "insufficient": False,
        "seasons": seasons_present,
        "matrix": mat,               # (n_seasons, k)
        "major_idx": major_idx,
        "riser_idx": riser_idx, "riser_pp": float(delta[riser_idx] * 100),
        "faller_idx": faller_idx, "faller_pp": float(delta[faller_idx] * 100),
        "first_season": seasons_present[0], "last_season": seasons_present[-1],
    }


# ===========================================================================
# Section C1/C2 rework - per-transition drift magnitude + feature attribution
# ===========================================================================
# AI-ASSISTED (Claude Code, chat)
# Prompt: full C1+C2 rework spec, given in complete detail (exact formulas,
# thresholds, caveats, graceful-degradation rules) - see AI_USAGE.md for the
# full prompt text. "No new data pulls - everything derives from the
# existing three season recipe files... and the existing per-season
# feature tables."
# Used: every function below reuses recipes_all (already has TEAM_ABBREVIATION
# and MIN per season - no new query needed for the team-changed tag or the
# MIN>=300-in-both-seasons filter) and load_population() (already used
# elsewhere in this module for raw per-season features) - no new data
# source. jensenshannon (already imported) computes the drift magnitude;
# tercile_label (already defined above) computes the verdict word from a
# real distribution, never a hardcoded cutoff.
# Not AI: every formula, threshold (|dz|>=0.3, top-5), and the toward-R/
# away-from-F attribution logic - all specified directly in the task spec.

def season_transitions(seasons_present):
    """Consecutive (old, new) season pairs from whichever seasons a player
    actually has, e.g. ['2023-24','2024-25','2025-26'] ->
    [('2023-24','2024-25'), ('2024-25','2025-26')]. At most 2 for this
    project's 3-season window."""
    return list(zip(seasons_present[:-1], seasons_present[1:]))


def transition_drift_magnitude(player_id, recipes_all, k, season_old, season_new):
    """JS distance (base-2) between one player's own alpha vectors in two
    seasons. None if he doesn't have a recipe in both."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    row_old = recipes_all[(recipes_all["PLAYER_ID"] == player_id) & (recipes_all["SEASON"] == season_old)]
    row_new = recipes_all[(recipes_all["PLAYER_ID"] == player_id) & (recipes_all["SEASON"] == season_new)]
    if len(row_old) == 0 or len(row_new) == 0:
        return None
    v_old = row_old.iloc[0][arch_cols].values.astype(float)
    v_new = row_new.iloc[0][arch_cols].values.astype(float)
    return float(jensenshannon(v_old, v_new, base=2))


def league_transition_drift_distribution(recipes_all, k, season_old, season_new, min_minutes=300):
    """Drift magnitude for EVERY league player with a recipe in BOTH
    seasons of this transition (MIN>=300 in both, per the task spec) - the
    real distribution a single player's own magnitude is read against.
    Computed once per transition (season_old, season_new), not per player -
    the caller (portal.py) caches this at the transition level, reused for
    every player."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    old_df = recipes_all[(recipes_all["SEASON"] == season_old) & (recipes_all["MIN"] >= min_minutes)]
    new_df = recipes_all[(recipes_all["SEASON"] == season_new) & (recipes_all["MIN"] >= min_minutes)]
    merged = old_df.merge(new_df, on="PLAYER_ID", suffixes=("_old", "_new"))
    old_vals = merged[[f"{c}_old" for c in arch_cols]].values.astype(float)
    new_vals = merged[[f"{c}_new" for c in arch_cols]].values.astype(float)
    dists = np.array([float(jensenshannon(o, n, base=2)) for o, n in zip(old_vals, new_vals)])
    return dists, int(len(merged))


def transition_team_changed(recipes_all, player_id, season_old, season_new):
    row_old = recipes_all[(recipes_all["PLAYER_ID"] == player_id) & (recipes_all["SEASON"] == season_old)]
    row_new = recipes_all[(recipes_all["PLAYER_ID"] == player_id) & (recipes_all["SEASON"] == season_new)]
    if len(row_old) == 0 or len(row_new) == 0:
        return False
    return bool(row_old.iloc[0]["TEAM_ABBREVIATION"] != row_new.iloc[0]["TEAM_ABBREVIATION"])


# Features whose natural scale isn't a 0-1 fraction (formatted as a plain
# number, not a percentage, in C2's hover text and attribution sentence).
NON_FRACTION_FEATURES = {"PTS_PER_100", "BPM", "Dist.", "PLAYER_HEIGHT_INCHES"}

# AI-ASSISTED (Claude Code, chat) - a real, visible bug: these 6 features
# are ALREADY stored as percentage points (e.g. USG% = 20.2 meaning
# 20.2%), not 0-1 fractions like TS%/the shot-distance/play-type shares -
# with no third bucket, they fell into the fraction branch below and got
# multiplied by 100 a SECOND time ("Usage rate (2020% -> 3050%)" instead
# of "20.2% -> 30.5%"). Classified by checking real 2025-26 population
# values directly (USG% mean ~19.1, range 8.5-38.1 - clearly
# percent-points already, not 0-1) rather than assuming one rule fits
# every feature - this project has hit this exact class of bug before
# (see the Box-score-rates format note in portal.py's
# render_player_stats_tab/adv_defs, which already classified these same 6
# features correctly for a different table on the page; this formatter
# just never matched it).
ALREADY_PERCENT_FEATURES = {"USG%", "AST%", "TOV%", "STL%", "BLK%", "TRB%"}


def format_raw_feature_value(feature_name, value):
    if feature_name in NON_FRACTION_FEATURES:
        return f"{value:.1f}"
    if feature_name in ALREADY_PERCENT_FEATURES:
        return f"{value:.1f}%"
    return f"{value:.0%}"


def drift_attribution(player_id, recipes_all, fit, k, season_old, season_new,
                      dz_threshold=0.3, n_features=5, min_threshold=0):
    """dz = z_new - z_old across all 29 basis features (both seasons
    z-scored with the basis's own FIXED mu/sd - never recomputed per
    season, same rule Phase 0's projection already follows). Kept features:
    |dz| >= dz_threshold, top n_features by |dz| (fewer if fewer qualify -
    never padded). Ties each kept feature to the transition's own top
    RISING archetype R / top FALLING archetype F (from the two alpha
    vectors for THIS specific transition, not the overall first-to-last
    role_drift() value, which could span a different window) via whether
    the feature's move points toward R's real archetypoid z-profile
    (fit['basis'][R]) or away from F's (fit['basis'][F])."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    feature_columns = fit["feature_columns"]
    mu, sd = fit["mu"], fit["sd"]

    pop_old = load_population(min_threshold=min_threshold, season_min=season_old, season_max=season_old)
    pop_new = load_population(min_threshold=min_threshold, season_min=season_new, season_max=season_new)
    row_old = pop_old[pop_old["PLAYER_ID"] == player_id]
    row_new = pop_new[pop_new["PLAYER_ID"] == player_id]
    if len(row_old) == 0 or len(row_new) == 0:
        return {"available": False, "reason": "raw feature row missing for one of the two seasons"}
    raw_old, raw_new = row_old.iloc[0], row_new.iloc[0]

    z_old = (raw_old[feature_columns].astype(float).values - mu) / sd
    z_new = (raw_new[feature_columns].astype(float).values - mu) / sd
    dz = z_new - z_old

    qualifying = np.where(np.abs(dz) >= dz_threshold)[0]
    if len(qualifying) == 0:
        return {"available": False, "reason": f"no feature moved by >= {dz_threshold} SD between the two seasons"}
    kept = qualifying[np.argsort(-np.abs(dz[qualifying]))][:n_features]

    rec_old = recipes_all[(recipes_all["PLAYER_ID"] == player_id) & (recipes_all["SEASON"] == season_old)]
    rec_new = recipes_all[(recipes_all["PLAYER_ID"] == player_id) & (recipes_all["SEASON"] == season_new)]
    alpha_old = rec_old.iloc[0][arch_cols].values.astype(float)
    alpha_new = rec_new.iloc[0][arch_cols].values.astype(float)
    alpha_delta = alpha_new - alpha_old
    R = int(np.argmax(alpha_delta))
    F = int(np.argmin(alpha_delta))

    basis = np.asarray(fit["basis"], dtype=float)
    features = []
    for i in kept:
        i = int(i)
        toward_R = bool(np.sign(dz[i]) == np.sign(basis[R, i] - z_old[i])) if basis[R, i] != z_old[i] else False
        away_F = bool(np.sign(dz[i]) == np.sign(z_old[i] - basis[F, i])) if basis[F, i] != z_old[i] else False
        features.append({
            "idx": i, "name": feature_columns[i], "dz": float(dz[i]),
            "raw_old": float(raw_old[feature_columns[i]]), "raw_new": float(raw_new[feature_columns[i]]),
            "toward_R": toward_R, "away_F": away_F,
        })

    return {
        "available": True, "features": features, "R": R, "F": F,
        "alpha_delta_R_pp": float(alpha_delta[R] * 100), "alpha_delta_F_pp": float(alpha_delta[F] * 100),
    }


# AI-ASSISTED (Claude Code, chat)
# Prompt: "REPLACE the two per-transition diverging bar panels with ONE
# feature-trajectory line chart... y = the feature's league-standardized
# value (z, using the basis's fixed mu/sd)... if the player has only 2
# seasons, the chart naturally shows 2 points."
# Used: drift_attribution() only ever z-scores a feature across the TWO
# seasons of ONE transition; the redesigned C2 chart plots the UNION of
# BOTH transitions' top movers across the player's FULL season history (up
# to 3 points, not 2), so a feature flagged only in transition 2 still
# gets its transition-1 point plotted too. Reuses the exact same
# fixed-mu/sd standardization drift_attribution() already uses (never
# refit per season) rather than a different method for the same quantity.
# Not AI: the requirement itself - given directly.
def feature_trajectory(player_id, feature_names, fit, seasons, min_threshold=0):
    """Per-season raw value + z-score (fixed basis mu/sd) for a specific
    set of features, across ALL of `seasons` - not just one transition's
    two endpoints. Returns {feature_name: {"raw": [...], "z": [...]}},
    one entry per season in the same order as `seasons`; a season with no
    raw feature row for this player (not expected in practice, since
    `seasons` already comes from a player who cleared the MIN floor each
    season - handled defensively, not for an anticipated real case) gets
    None in both lists rather than raising.
    """
    feature_columns = fit["feature_columns"]
    mu, sd = fit["mu"], fit["sd"]
    idx_by_name = {name: i for i, name in enumerate(feature_columns)}

    result = {name: {"raw": [], "z": []} for name in feature_names}
    for s in seasons:
        pop_s = load_population(min_threshold=min_threshold, season_min=s, season_max=s)
        row = pop_s[pop_s["PLAYER_ID"] == player_id]
        if len(row) == 0:
            for name in feature_names:
                result[name]["raw"].append(None)
                result[name]["z"].append(None)
            continue
        row = row.iloc[0]
        for name in feature_names:
            i = idx_by_name[name]
            raw_v = float(row[name])
            result[name]["raw"].append(raw_v)
            result[name]["z"].append((raw_v - mu[i]) / sd[i])
    return result


# ===========================================================================
# Section B2 - development comps at matched age
# ===========================================================================

def development_comps(player_id, recipes_all, bio_all, k, subject_season="2025-26",
                       age_tol=1, top_n=5, seasons=ALL_SEASONS):
    """Players who had a similar recipe to the SUBJECT's CURRENT recipe,
    at the SAME age the subject is now, in whichever of our 3 seasons they
    were that age - then what those players' recipes became the following
    season we have. With 3 projected seasons total, the lookahead is at
    most 2 seasons past the matched one - never implied to be a full
    career arc.
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    subj_row = recipes_all[(recipes_all["PLAYER_ID"] == player_id) & (recipes_all["SEASON"] == subject_season)]
    if len(subj_row) == 0:
        raise ValueError(f"player_id {player_id} has no {subject_season} recipe")
    subj_vec = subj_row.iloc[0][arch_cols].values.astype(float)

    subj_age_row = bio_all[(bio_all["PLAYER_ID"] == player_id) & (bio_all["SEASON"] == subject_season)]
    if len(subj_age_row) == 0:
        return {"subject_age": None, "comps": [], "note": "no bio/AGE row for the subject in the current season"}
    subj_age = float(subj_age_row.iloc[0]["AGE"])

    bio_idx = bio_all.set_index(["SEASON", "PLAYER_ID"])["AGE"]

    candidates = []
    for s in seasons:
        pop_s = recipes_all[recipes_all["SEASON"] == s]
        for _, r in pop_s.iterrows():
            pid = int(r["PLAYER_ID"])
            if pid == player_id:
                continue
            key = (s, pid)
            if key not in bio_idx.index:
                continue
            age = float(bio_idx.loc[key])
            if abs(age - subj_age) > age_tol:
                continue
            v = r[arch_cols].values.astype(float)
            dist = float(jensenshannon(v, subj_vec, base=2))
            candidates.append({
                "player_id": pid, "name": r["PLAYER_NAME"], "matched_season": s,
                "age": age, "js_dist": dist, "vec": v,
            })

    candidates.sort(key=lambda c: c["js_dist"])
    top = candidates[:top_n]

    season_order = {s: i for i, s in enumerate(seasons)}
    out = []
    for c in top:
        i = season_order[c["matched_season"]]
        next_season = seasons[i + 1] if i + 1 < len(seasons) else None
        next_info = None
        if next_season is not None:
            nrow = recipes_all[(recipes_all["PLAYER_ID"] == c["player_id"]) & (recipes_all["SEASON"] == next_season)]
            if len(nrow):
                nvec = nrow.iloc[0][arch_cols].values.astype(float)
                delta = nvec - c["vec"]
                biggest_i = int(np.argmax(np.abs(delta)))
                next_info = {
                    "season": next_season,
                    "top_archetype": int(np.argmax(nvec)),
                    "biggest_change_idx": biggest_i,
                    "biggest_change_pp": float(delta[biggest_i] * 100),
                }
        out.append({
            "player_id": c["player_id"], "name": c["name"], "matched_season": c["matched_season"],
            "age": c["age"], "js_dist": c["js_dist"],
            "top_archetype_then": int(np.argmax(c["vec"])),
            "next": next_info,
        })

    max_lookahead = len(seasons) - 1 - min((season_order[c["matched_season"]] for c in top), default=len(seasons) - 1)
    return {"subject_age": subj_age, "comps": out, "max_lookahead_seasons": max_lookahead}


# ===========================================================================
# Section C3 - role elasticity (downgraded per Gate C3 - see module docstring)
# ===========================================================================

def _load_season_stints_v2_or_v1(season):
    """The process-metrics-extended stint table only exists for the primary
    season (stints_2025_26_v2.parquet); other seasons fall back to the
    plain stint table (no off_fga/off_tov columns -> C3 degrades further,
    handled by the caller checking for those columns)."""
    v2_path = STINTS_DIR / f"stints_{season.replace('-', '_')}_v2.parquet"
    v1_path = STINTS_DIR / f"stints_{season.replace('-', '_')}.parquet"
    path = v2_path if v2_path.exists() else v1_path
    return pd.read_parquet(path), path


def highest_shared_minutes_teammate(player_id, season="2025-26", db_path=None):
    pairs = load_player_pairs(player_id, season=season, db_path=db_path)
    if len(pairs) == 0:
        return None
    row = pairs.loc[pairs["MIN"].idxmax()]
    return int(row["TEAMMATE_ID"]), float(row["MIN"])


def _summarize_stint_block(df):
    """Team-level offensive-profile summary for a block of stints - shared
    by role_elasticity() and role_sensitivity_profile() so both compute
    the exact same 6 metrics off the exact same stint columns."""
    poss = float(df["off_possessions"].sum())
    fga = float(df["off_fga"].sum())
    fgm = float(df["off_fgm"].sum())
    fg3a = float(df["off_fg3a"].sum())
    fg3m = float(df["off_fg3m"].sum())
    tov = float(df["off_tov"].sum())
    points = float(df["off_points"].sum())
    return {
        "possessions": poss,
        "ortg": 100.0 * points / poss if poss else np.nan,
        "efg_pct": (fgm + 0.5 * fg3m) / fga if fga else np.nan,
        "fg_pct": fgm / fga if fga else np.nan,
        "fg3_pct": fg3m / fg3a if fg3a else np.nan,
        "fg3a_rate": fg3a / fga if fga else np.nan,
        "tov_pct": 100.0 * tov / poss if poss else np.nan,
    }


def role_elasticity(player_id, season="2025-26", min_poss=500, db_path=None):
    """GATE C3 (downgraded path, taken deliberately - see module docstring):
    true per-event individual usage-share attribution would require
    rebuilding stint-to-play-by-play event boundaries (the substitution-
    tracking machinery build_stints.py already implements internally but
    does not expose), which is a project of its own scope for one
    subsection of six. Building that was judged not worth it here - this
    computes the TEAM's offensive shot profile (eFG%, 3PA rate, TOV%)
    while the focal player shares the floor with his highest-shared-
    minutes teammate vs. when he's on the floor without him, both directly
    from the already-validated per-stint team-level counts
    (stints_*_v2.parquet). This is a genuinely different quantity than an
    individual usage split and is labeled as such everywhere it's shown -
    it answers "does the team play differently around him with vs.
    without this teammate", not "does HIS OWN usage change."
    """
    teammate = highest_shared_minutes_teammate(player_id, season=season, db_path=db_path)
    if teammate is None:
        return {"available": False, "reason": "no real 2-man on-court pairs this season"}
    teammate_id, teammate_min = teammate

    stints, path = _load_season_stints_v2_or_v1(season)
    required = {"off_fga", "off_fgm", "off_fg3a", "off_fg3m", "off_tov", "off_points"}
    if not required.issubset(stints.columns):
        return {"available": False, "reason": f"{path.name} lacks process-metric columns "
                                              f"({required - set(stints.columns)}) - process-metrics "
                                              f"extension not built for this season"}

    off_cols = [f"off_p{i}" for i in range(1, 6)]
    on_floor = stints[stints[off_cols].eq(player_id).any(axis=1)].copy()
    if len(on_floor) == 0:
        return {"available": False, "reason": "player never appears in an offensive 5-man unit this season"}

    with_teammate = on_floor[on_floor[off_cols].eq(teammate_id).any(axis=1)]
    without_teammate = on_floor[~on_floor[off_cols].eq(teammate_id).any(axis=1)]

    with_summary = _summarize_stint_block(with_teammate)
    without_summary = _summarize_stint_block(without_teammate)
    thin = with_summary["possessions"] < min_poss or without_summary["possessions"] < min_poss

    return {
        "available": True,
        "teammate_id": teammate_id, "teammate_min": teammate_min,
        "with_teammate": with_summary, "without_teammate": without_summary,
        "thin": thin, "min_poss": min_poss,
        "metric_level": "team",  # explicit tag: this is NOT an individual usage split
    }


# SUPERSEDED (Claude Code, chat) - kept, not deleted, per this project's
# established "kept but not wired up" convention (see SHOW_STABILITY_BOOTSTRAP/
# render_roster_construction in portal.py for the same pattern).
# Prompt (original): D3 upgrade spec - "from a single-context split to a
# full 8-archetype sensitivity profile: for each context archetype, split
# the player's own stints into HIGH vs LOW by median teammate-exposure to
# that archetype, compute the usage-proxy (team shot-profile) delta, grey
# out any archetype under a possession floor."
# Prompt (superseding): "Fix D3 properly by closing the data gap it
# downgraded around... PHASE 2 - REBUILD D3 AS DESIGNED (individual
# elasticity, not a team proxy)... the removed team-ORtg proxy version:
# delete its UI; keep the function as superseded-but-visible per project
# convention, noting it duplicated D2's question."
# Why superseded: this function's own ORtg-delta metric asks "does the
# TEAM's offense perform better or worse around him" - the same question
# D2 (teammate lift) already answers, just measured a different way (net-
# rating lift by teammate archetype vs. ORtg delta by teammate-exposure
# split). Once Phase 1 (build_stints.py's event-to-stint link) made a real
# INDIVIDUAL usage/assist/shot-mix measurement possible, this team-level
# proxy stopped being the best available answer to "does HIS OWN game
# change" and became a near-duplicate of D2 instead. See
# `individual_role_sensitivity_profile()` below for the replacement.
def role_sensitivity_profile(player_id, recipes, k, season="2025-26", min_poss=500, db_path=None):
    """ORtg delta (HIGH-minus-LOW teammate exposure) for each of the k
    context archetypes, off the player's own on-floor offensive stints
    this season. `min_poss` gates EACH archetype's split independently -
    a player can have some archetypes thin and others not."""
    stints, path = _load_season_stints_v2_or_v1(season)
    required = {"off_fga", "off_fgm", "off_fg3a", "off_fg3m", "off_tov", "off_points"}
    if not required.issubset(stints.columns):
        return {"available": False, "reason": f"{path.name} lacks process-metric columns "
                                              f"({required - set(stints.columns)}) - process-metrics "
                                              f"extension not built for this season"}

    off_cols = [f"off_p{i}" for i in range(1, 6)]
    on_floor = stints[stints[off_cols].eq(player_id).any(axis=1)].copy()
    if len(on_floor) == 0:
        return {"available": False, "reason": "player never appears in an offensive 5-man unit this season"}

    arch_cols = [f"arch_{i}" for i in range(k)]
    rec_season = recipes[recipes["SEASON"] == season] if "SEASON" in recipes.columns else recipes
    arch_lookup = rec_season.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")[arch_cols]

    teammate_ids = on_floor[off_cols].values  # (n_stints, 5)
    is_teammate = teammate_ids != player_id   # excludes the focal player's own slot

    profiles = []
    for c in range(k):
        arch_share = arch_lookup[f"arch_{c}"]
        exposure = np.zeros(len(on_floor))
        for j in range(5):
            vals = pd.Series(teammate_ids[:, j]).map(arch_share).fillna(0.0).values
            exposure += np.where(is_teammate[:, j], vals, 0.0)

        median_exp = float(np.median(exposure))
        high_mask = exposure >= median_exp
        low_mask = ~high_mask
        if high_mask.sum() == 0 or low_mask.sum() == 0:
            profiles.append({
                "archetype_idx": c, "available": False,
                "reason": "degenerate median split (every stint landed on one side)",
            })
            continue

        high_summary = _summarize_stint_block(on_floor[high_mask])
        low_summary = _summarize_stint_block(on_floor[low_mask])
        thin = high_summary["possessions"] < min_poss or low_summary["possessions"] < min_poss
        profiles.append({
            "archetype_idx": c, "available": True, "thin": thin, "min_poss": min_poss,
            "median_exposure": median_exp,
            "high": high_summary, "low": low_summary,
            "ortg_delta": float(high_summary["ortg"] - low_summary["ortg"]),
        })

    return {"available": True, "profiles": profiles, "n_stints": int(len(on_floor)),
            "metric_level": "team"}


# AI-ASSISTED (Claude Code, chat)
# Prompt: D3 upgrade spec - "GATE an optional rim-vs-3PT shot-mix
# elasticity (and an assist-rate delta) on whether shot-distance data
# exists in cached PBP; if not, state the limitation."
# Used: checked the actual cached raw play-by-play
# (data/stints/raw_cache/*_pbp.parquet, sourced from NBA's PlayByPlayV3
# endpoint) directly - it DOES carry a real `shotDistance` column per shot
# event, and assist credit as free text inside `description` (e.g. "...
# (Bridges 3 AST)"), never a structured column. The gate still resolves
# to NOT AVAILABLE for a different, load-bearing reason: neither field is
# usable at the STINT level, because stints_*.parquet (the table every
# other Section D computation reads) stores only pre-aggregated per-stint
# counts with no pointer back to which raw PBP rows fall inside which
# stint - that event-to-stint join is exactly the "rebuilding stint-to-
# play-by-play event boundaries" scope tradeoff role_elasticity's own
# docstring already declined for individual usage attribution. Extending
# it here would mean re-deriving a chunk of build_stints.py's own
# substitution-boundary walk (extensively fixed for edge cases like
# subs-between-free-throws - see that file's own history) against ~3700
# cached games, which is a project of its own for one optional gate on one
# subsection. Reports this as a clear, checked "why not," not a silent
# omission.
# Not AI: the gate structure itself ("if not, state the limitation") -
# specified directly in the task spec.
# SUPERSEDED (Claude Code, chat) - kept, not deleted, per this project's
# "kept but not wired up" convention. This gate's own reasoning (below) was
# correct AT THE TIME it was written - the join it describes as too costly
# to build ("rebuilding the substitution-boundary walk") is exactly what
# build_stints.py's Phase 1 event export then went and built, closing the
# gap this function documents. Superseded by checking for that export
# directly rather than re-deriving the old per-column check.
def shot_mix_gate_status(season="2025-26"):
    """Whether Section D3's rim-vs-3PT / assist-rate elasticity can be
    built from currently cached data. Checked directly against whether the
    Phase 1 event-to-stint export exists for this season, not assumed."""
    events_path = STINTS_DIR / f"events_{season.replace('-', '_')}.parquet"
    if events_path.exists():
        return {"available": True, "reason": f"{events_path.name} (event-to-stint link) exists for {season}."}
    import glob
    sample = glob.glob(str(RAW_CACHE_DIR / "*_pbp.parquet"))
    if not sample:
        return {"available": False, "reason": "no cached play-by-play found"}
    cols = set(pd.read_parquet(sample[0]).columns)
    has_shot_distance = "shotDistance" in cols
    return {
        "available": False,
        "has_shot_distance": has_shot_distance,
        "reason": (
            "shot distance IS cached per-event (raw play-by-play), but stints_*.parquet - the table this "
            "section's other splits read - stores only pre-aggregated per-stint counts with no link back "
            "to individual play-by-play rows. Joining shot distance (or assist credit, which only exists "
            "as free text in the play description, never a structured column) to a specific stint would "
            "mean rebuilding the substitution-boundary walk build_stints.py already implements internally "
            "but doesn't expose - the same scope tradeoff already made for individual usage attribution "
            "(see role_elasticity's own note). Run build_stints.py's event export to close this gap."
        ),
    }


# A "shot at the rim" cutoff in feet - matches this project's own basis
# feature's "% of FGA by Distance_0-3" bucket boundary (the closest B-Ref-
# style bucket to "at the rim"), so this new individual-level metric reads
# on the same scale as the archetype model's own shot-profile features.
RIM_DISTANCE_FT = 3

_SCORING_ACTION_TYPES = ("Made Shot", "Missed Shot")


def _individual_event_summary(ev_df, player_id, possessions, n_stints):
    """HIS OWN behavior (not team-level) over one side (HIGH or LOW
    exposure) of a context-archetype split: usage proxy, assist rate, rim
    share, 3PT share - all computed from the Phase 1 event-to-stint link
    table. `ev_df` must already be restricted to this player's OWN TEAM's
    events during the relevant stints (his 4 teammates' events are needed
    for the usage/assist-rate DENOMINATOR - "his share of his team's
    events" - not just his own numerator)."""
    his = ev_df[ev_df["personId"] == player_id]
    his_fga = int(his["actionType"].isin(_SCORING_ACTION_TYPES).sum())
    his_tov = int((his["actionType"] == "Turnover").sum())
    his_fta = int((his["actionType"] == "Free Throw").sum())
    his_ast = int((ev_df["assistPersonId"] == player_id).sum())  # assists on ANY teammate's make, team-wide

    team_fga = int(ev_df["actionType"].isin(_SCORING_ACTION_TYPES).sum())
    team_tov = int((ev_df["actionType"] == "Turnover").sum())
    team_fta = int((ev_df["actionType"] == "Free Throw").sum())
    team_fgm = int((ev_df["actionType"] == "Made Shot").sum())

    his_num = his_fga + his_tov + 0.44 * his_fta
    team_num = team_fga + team_tov + 0.44 * team_fta
    usage_proxy = his_num / team_num if team_num > 0 else float("nan")
    assist_rate = his_ast / team_fgm if team_fgm > 0 else float("nan")

    his_shots = his[his["actionType"].isin(_SCORING_ACTION_TYPES)]
    if len(his_shots) > 0:
        rim_share = float((his_shots["shotDistance"] <= RIM_DISTANCE_FT).mean())
        three_share = float((his_shots["shotValue"] == 3).mean())
    else:
        rim_share, three_share = float("nan"), float("nan")

    return {
        "possessions": float(possessions), "n_stints": int(n_stints),
        "usage_proxy": usage_proxy, "assist_rate": assist_rate,
        "rim_share": rim_share, "three_share": three_share,
        "his_fga": his_fga, "his_tov": his_tov, "his_fta": his_fta, "his_ast": his_ast,
    }


# AI-ASSISTED (Claude Code, chat)
# Prompt: "PHASE 2 - REBUILD D3 AS DESIGNED (individual elasticity, not a
# team proxy): For each of the 8 context archetypes: split the subject's
# on-floor POSSESSIONS (not stints) by his 4 teammates' combined exposure
# to that archetype (median split as now). From the new event table
# compute HIS OWN behavior per side: usage proxy = his (FGA + TOV +
# 0.44*FTA) share of his team's events while on floor; assist rate. Report
# the HIGH-minus-LOW delta per archetype - this replaces the team-ORtg
# delta entirely. If shotDistance survived into the event table, also
# report his rim-vs-3PT share delta per context."
# Used: the median-split mechanic itself (exposure score per stint, HIGH
# vs LOW by the real median) is UNCHANGED from the superseded
# role_sensitivity_profile() above - "possessions not stints" is honored by
# weighting every aggregate by real off_possessions/event counts rather
# than by stint COUNT, not by literally enumerating individual possessions
# (which would produce an identical split, since teammate exposure is
# constant for the whole lineup-window a stint represents - splitting by
# stint and summing possessions within each side already IS a possession-
# weighted split). The events table is first filtered ONCE to this
# player's own team's events during his on-floor stints (a small subset)
# before the 8-way archetype split, rather than re-scanning the full
# ~311k-row season table inside every one of the 16 (8 archetypes x
# high/low) per-side summaries.
# Not AI: the exact usage-proxy formula, the assist-rate/rim-share/3PT-
# share additions, and "replaces the team-ORtg delta entirely" - all
# specified directly in the task spec.
def individual_role_sensitivity_profile(player_id, recipes, k, season="2025-26", min_poss=500):
    stints, path = _load_season_stints_v2_or_v1(season)
    required = {"off_fga", "off_fgm", "off_fg3a", "off_fg3m", "off_tov", "off_points"}
    if not required.issubset(stints.columns):
        return {"available": False, "reason": f"{path.name} lacks process-metric columns "
                                              f"({required - set(stints.columns)})"}

    events_path = STINTS_DIR / f"events_{season.replace('-', '_')}.parquet"
    if not events_path.exists():
        return {"available": False, "reason": f"{events_path.name} not found - run build_stints.py's "
                                              f"event export (Phase 1) for this season first"}
    events = pd.read_parquet(events_path)

    off_cols = [f"off_p{i}" for i in range(1, 6)]
    on_floor = stints[stints[off_cols].eq(player_id).any(axis=1)].copy()
    if len(on_floor) == 0:
        return {"available": False, "reason": "player never appears in an offensive 5-man unit this season"}

    arch_cols = [f"arch_{i}" for i in range(k)]
    rec_season = recipes[recipes["SEASON"] == season] if "SEASON" in recipes.columns else recipes
    arch_lookup = rec_season.drop_duplicates("PLAYER_ID").set_index("PLAYER_ID")[arch_cols]

    teammate_ids = on_floor[off_cols].values
    is_teammate = teammate_ids != player_id

    # restrict once to this player's own team's events during his own
    # on-floor stints (his 4 teammates' events included - needed for the
    # usage/assist-rate denominator) - a (game_id, stint_id, team) triple
    # match, since `on_floor`'s own offense_team IS his team for that
    # stint (on_floor only kept rows where he's literally one of off_p1..5).
    on_floor_keys = set(zip(on_floor["game_id"], on_floor["stint_id"], on_floor["offense_team"]))
    ev_key_tuples = list(zip(events["game_id"], events["stint_id"], events["teamId"]))
    team_events = events[[t in on_floor_keys for t in ev_key_tuples]].copy()
    if len(team_events) == 0:
        return {"available": False, "reason": "no team events found for this player's on-floor stints"}

    profiles = []
    for c in range(k):
        arch_share = arch_lookup[f"arch_{c}"]
        exposure = np.zeros(len(on_floor))
        for j in range(5):
            vals = pd.Series(teammate_ids[:, j]).map(arch_share).fillna(0.0).values
            exposure += np.where(is_teammate[:, j], vals, 0.0)

        median_exp = float(np.median(exposure))
        high_mask = exposure >= median_exp
        low_mask = ~high_mask
        if high_mask.sum() == 0 or low_mask.sum() == 0:
            profiles.append({
                "archetype_idx": c, "available": False,
                "reason": "degenerate median split (every stint landed on one side)",
            })
            continue

        high_keys = set(zip(on_floor.loc[high_mask, "game_id"], on_floor.loc[high_mask, "stint_id"]))
        low_keys = set(zip(on_floor.loc[low_mask, "game_id"], on_floor.loc[low_mask, "stint_id"]))
        te_key_pairs = list(zip(team_events["game_id"], team_events["stint_id"]))
        high_events = team_events[[t in high_keys for t in te_key_pairs]]
        low_events = team_events[[t in low_keys for t in te_key_pairs]]

        high_summary = _individual_event_summary(
            high_events, player_id, on_floor.loc[high_mask, "off_possessions"].sum(), int(high_mask.sum()))
        low_summary = _individual_event_summary(
            low_events, player_id, on_floor.loc[low_mask, "off_possessions"].sum(), int(low_mask.sum()))
        thin = high_summary["possessions"] < min_poss or low_summary["possessions"] < min_poss

        def _delta(key):
            hv, lv = high_summary[key], low_summary[key]
            return float(hv - lv) if not (np.isnan(hv) or np.isnan(lv)) else float("nan")

        profiles.append({
            "archetype_idx": c, "available": True, "thin": thin, "min_poss": min_poss,
            "median_exposure": median_exp,
            "high": high_summary, "low": low_summary,
            "usage_delta": _delta("usage_proxy"), "assist_delta": _delta("assist_rate"),
            "rim_delta": _delta("rim_share"), "three_delta": _delta("three_share"),
        })

    return {"available": True, "profiles": profiles, "n_stints": int(len(on_floor)), "metric_level": "individual"}


# ===========================================================================
# Section D - miscasting test (opportunity-only vs outcome-only projection)
# ===========================================================================

OPPORTUNITY_FEATURES = [
    "USG%", "Dist.",
    "% of FGA by Distance_0-3", "% of FGA by Distance_3-10", "% of FGA by Distance_10-16",
    "% of FGA by Distance_16-3P", "% of FGA by Distance_3P", "Corner 3s_%3PA",
    "% of FG Ast'd_2P", "% of FG Ast'd_3P",
    "PLAYTYPE_CUT", "PLAYTYPE_HANDOFF", "PLAYTYPE_ISOLATION", "PLAYTYPE_OFFREBOUND",
    "PLAYTYPE_OFFSCREEN", "PLAYTYPE_PRBALLHANDLER", "PLAYTYPE_PRROLLMAN",
    "PLAYTYPE_POSTUP", "PLAYTYPE_SPOTUP",
]
OUTCOME_FEATURES = ["PTS_PER_100", "TS%", "AST%", "TOV%", "STL%", "BLK%", "TRB%", "FTr", "BPM"]
# PLAYER_HEIGHT_INCHES is neither a deployment choice nor a production
# outcome - a fixed physical trait - so it's excluded from BOTH partial
# projections (an explicit, disclosed judgment call, not an oversight;
# see DIAGNOSTICS_README.md). Dist. (average shot distance) is treated as
# an opportunity/shot-selection feature, not an outcome, for the same
# reason - also disclosed there rather than left implicit.
NEUTRAL_FEATURES = ["PLAYER_HEIGHT_INCHES"]


def _partial_projection(raw_row, fit, feature_block):
    """Project using only `feature_block`'s real values; every other
    feature's raw value is replaced with the basis's own population mean
    for that feature, which z-scores to exactly 0 (a neutral, non-
    separating contribution) - reuses project() unmodified rather than
    hand-rolling a second NNLS call."""
    feature_columns = fit["feature_columns"]
    mu = pd.Series(fit["mu"], index=feature_columns)
    x = raw_row[feature_columns].astype(float).copy()
    for col in feature_columns:
        if col not in feature_block:
            x[col] = mu[col]
    df1 = pd.DataFrame([x])
    P = project(df1, fit["basis"], fit["mu"], fit["sd"], feature_columns)
    return P[0]


def miscasting_score(player_id, season, fit, min_threshold=0, pop=None):
    """opportunity-only and outcome-only partial recipes for one player,
    and the JS distance between them (the miscast score). `pop`: optional
    pre-loaded load_population(...) frame - league_miscasting_scores below
    loads it once and passes it through rather than re-running
    build_nba_side_tables()'s full join per player (433x otherwise)."""
    if pop is None:
        pop = load_population(min_threshold=min_threshold, season_min=season, season_max=season)
    row_match = pop[pop["PLAYER_ID"] == player_id]
    if len(row_match) == 0:
        raise ValueError(f"player_id {player_id} not found in {season} population (min_threshold={min_threshold})")
    raw_row = row_match.iloc[0]

    opp_vec = _partial_projection(raw_row, fit, OPPORTUNITY_FEATURES)
    out_vec = _partial_projection(raw_row, fit, OUTCOME_FEATURES)
    score = float(jensenshannon(opp_vec, out_vec, base=2))
    return {
        "opportunity_recipe": opp_vec, "outcome_recipe": out_vec, "score": score,
        "underused_idx": int(np.argmax(out_vec - opp_vec)),
        "gap_pp": float((out_vec - opp_vec).max() * 100),
    }


def league_miscasting_scores(recipes, fit, season="2025-26", min_threshold=0):
    """Miscast score for every recipe player this season - the real
    percentile distribution Section D's number is read against (computed
    once, cached by the caller). Loads the population ONCE and reuses it
    across all players (see miscasting_score's own note)."""
    pop = load_population(min_threshold=min_threshold, season_min=season, season_max=season)
    ids = recipes[recipes["SEASON"] == season]["PLAYER_ID"].astype(int)
    scores = []
    for pid in ids:
        try:
            scores.append(miscasting_score(pid, season, fit, min_threshold=min_threshold, pop=pop)["score"])
        except ValueError:
            continue
    return np.array(scores)


# AI-ASSISTED (Claude Code, chat)
# Prompt (original): Section E redesign spec, tiered-verdict item 3c -
# "ground it with one feature clause per side: the outcome feature with
# the largest positive z contribution toward that archetype, and the
# opportunity feature with the largest deficit."
# Prompt (this fix): "fix the feature-evidence clause: restrict the
# auto-picked supporting features to an offense-relevant, intuitive
# whitelist (TS%, 3PA rate/3P%, USG%, AST%, spot-up/off-screen playtype
# shares, %FGA assisted) - never cite defensive stats like steal rate as
# evidence for an offensive archetype resemblance. If no whitelisted
# feature clears the contribution threshold, omit the clause rather than
# citing an unintuitive one."
# Used: a NEW, separate function - miscasting_score()/_partial_projection()
# themselves are untouched, per the original spec's explicit "the two
# partial projections and the JS score computation stay exactly as they
# are." "Contribution toward archetype R" is a disclosed heuristic (one
# term of a similarity product, player_z * that archetype's own
# archetypoid z-profile at fit["basis"][R] - the same archetypoid
# reference point drift_attribution already uses for its toward-R/away-
# from-F logic), not a decomposition of the NNLS projection itself, which
# has no clean per-feature attribution. "Deficit" = how far below that
# archetype's own archetypoid profile the player's real z-score sits.
# `INTUITIVE_EVIDENCE_FEATURES` is the exact whitelist given in the prompt,
# mapped onto this project's real feature names (STL%/BLK%/TRB%/TOV%/FTr/
# PTS_PER_100/BPM are all in OUTCOME_FEATURES but NOT on the whitelist -
# the bug being fixed is that the old version's unrestricted argmax over
# OUTCOME_FEATURES could and did surface exactly these as "evidence"). Both
# the outcome search AND the opportunity search are now restricted to this
# SAME whitelist (not just the outcome side) and each requires its own
# argmax to clear `GROUNDING_CONTRIBUTION_THRESHOLD` (0.3 SD - reusing the
# same disclosed cutoff step2b's own drift_attribution() already uses for
# "a meaningful feature movement", rather than inventing a second number)
# before being reported at all - each side is independently None if
# nothing whitelisted clears the bar, letting the caller build a one-sided,
# two-sided, or fully-omitted sentence rather than forcing a non-sequitur.
# Not AI: the exact whitelist and the "omit rather than cite an unintuitive
# feature" requirement - specified directly in the task spec.
INTUITIVE_EVIDENCE_FEATURES = {
    "TS%", "AST%", "USG%", "% of FGA by Distance_3P",
    "PLAYTYPE_SPOTUP", "PLAYTYPE_OFFSCREEN",
    "% of FG Ast'd_2P", "% of FG Ast'd_3P",
}
GROUNDING_CONTRIBUTION_THRESHOLD = 0.3  # SD - same convention as drift_attribution's dz cutoff


def miscasting_feature_grounding(player_id, season, fit, underused_idx, min_threshold=0, pop=None):
    """One outcome feature + one opportunity feature that "ground" the
    underused_idx archetype gap in real numbers, for Section E's
    largest-gap verdict line - each half independently None if nothing on
    the intuitive whitelist clears the contribution threshold."""
    if pop is None:
        pop = load_population(min_threshold=min_threshold, season_min=season, season_max=season)
    row_match = pop[pop["PLAYER_ID"] == player_id]
    if len(row_match) == 0:
        return {"outcome_feature": None, "opportunity_feature": None}
    raw_row = row_match.iloc[0]
    feature_columns = fit["feature_columns"]
    mu, sd = fit["mu"], fit["sd"]
    z = (raw_row[feature_columns].astype(float).values - mu) / sd
    basis_row = np.asarray(fit["basis"], dtype=float)[underused_idx]
    col_idx = {c: i for i, c in enumerate(feature_columns)}

    result = {"outcome_feature": None, "outcome_z": None,
             "opportunity_feature": None, "opportunity_z": None, "opportunity_deficit": None}

    outcome_idx = [col_idx[c] for c in OUTCOME_FEATURES if c in col_idx and c in INTUITIVE_EVIDENCE_FEATURES]
    if outcome_idx:
        contribution = z[outcome_idx] * basis_row[outcome_idx]
        best_local = int(np.argmax(contribution))
        if contribution[best_local] >= GROUNDING_CONTRIBUTION_THRESHOLD:
            best_outcome_i = outcome_idx[best_local]
            result["outcome_feature"] = feature_columns[best_outcome_i]
            result["outcome_z"] = float(z[best_outcome_i])

    opportunity_idx = [col_idx[c] for c in OPPORTUNITY_FEATURES if c in col_idx and c in INTUITIVE_EVIDENCE_FEATURES]
    if opportunity_idx:
        deficit = basis_row[opportunity_idx] - z[opportunity_idx]
        worst_local = int(np.argmax(deficit))
        if deficit[worst_local] >= GROUNDING_CONTRIBUTION_THRESHOLD:
            worst_opportunity_i = opportunity_idx[worst_local]
            result["opportunity_feature"] = feature_columns[worst_opportunity_i]
            result["opportunity_z"] = float(z[worst_opportunity_i])
            result["opportunity_deficit"] = float(deficit[worst_local])

    return result


# ===========================================================================
# Section B (signature radar) - similarity-weighted peer centroid
# ===========================================================================
# AI-ASSISTED (Claude Code, chat)
# Prompt: "我觉得这个radar chart不应该是针对单一的archetype 而是针对比如和他风格
# 最相近的比如10个人 他们在features上的差距 你觉得这个猜想合理么" (this radar
# chart shouldn't compare against a single archetype - it should compare
# against, say, the ~10 players whose STYLE is most similar to him, and the
# feature gap against THEM - is this a reasonable idea?), confirmed against
# a real case first: Danny Wolf is 27.9% Combo Guard / 23.6% 3&D Wing /
# 20.5% Rim Protector - a near three-way tie (purity percentile 4.8%, more
# hybrid than 95% of the league) - so bucketing him with "everyone whose TOP
# archetype is Combo Guard" compares him against players who are mostly
# MUCH more purely guards than he is, which is exactly the flaw the user
# was describing.
# Used: replaced archetype_centroids() (a hard top-archetype MEMBERSHIP
# bucket) with similarity_weighted_feature_centroid() - a continuous
# Jensen-Shannon-similarity-weighted centroid over EVERY other player's
# FULL recipe vector, not just people who share his nominal top archetype.
# Deliberately reused the exact weighting convention
# similarity_weighted_benchmark() (step2_diagnostic_analysis.py) already
# uses and this project already validated (JS distance, power=4.0, ESS
# reported) rather than a literal "top-10 nearest neighbors" average (small
# n, noisier, and inconsistent with how "similar players" is computed
# everywhere else in this app) - same idea, more stable estimator, and now
# Section B's peer concept matches Section D's (mismatch score) peer
# concept instead of being a third, different definition of "similar."
# Not AI: the core idea (compare against style-neighbors, not a single
# archetype bucket) and catching the Danny Wolf case that motivated it -
# both the user's own.

def similarity_weighted_feature_centroid(player_id, recipes, fit, k, season="2025-26",
                                          power=4.0, min_threshold=0, pop=None):
    """Weighted mean z-scored feature vector across the league, weighted by
    Jensen-Shannon similarity of each OTHER player's full archetype recipe
    to this player's own recipe - his "style neighbors'" typical feature
    profile, not his nominal top archetype's typical profile. Returns
    (centroid, ess) - ess (effective sample size) is reported alongside so
    a thin, unstable peer group is never silently treated as solid."""
    if pop is None:
        pop = load_population(min_threshold=min_threshold, season_min=season, season_max=season)
    arch_cols = [f"arch_{i}" for i in range(k)]
    rec = recipes[recipes["SEASON"] == season][["PLAYER_ID"] + arch_cols]
    target_row = rec[rec["PLAYER_ID"] == player_id]
    if len(target_row) == 0:
        raise ValueError(f"player_id {player_id} not found in {season} recipes")
    target_vec = target_row.iloc[0][arch_cols].values.astype(float)

    merged = pop.merge(rec, on="PLAYER_ID", how="inner")
    merged = merged[merged["PLAYER_ID"] != player_id].reset_index(drop=True)

    feature_columns = fit["feature_columns"]
    mu, sd = fit["mu"], fit["sd"]
    Z = (merged[feature_columns].astype(float).values - mu) / sd
    vecs = merged[arch_cols].values.astype(float)

    js_dists = np.array([float(jensenshannon(v, target_vec, base=2)) for v in vecs])
    js_dists = np.nan_to_num(js_dists, nan=0.0)
    weights = (1 - js_dists) ** power
    ess = float((weights.sum() ** 2) / (weights ** 2).sum())

    centroid = (Z * weights[:, None]).sum(axis=0) / weights.sum()
    return centroid, ess, feature_columns


def player_signature(player_id, centroid, fit, feature_columns,
                     season="2025-26", min_threshold=0, n_axes=10, pop=None):
    if pop is None:
        pop = load_population(min_threshold=min_threshold, season_min=season, season_max=season)
    row_match = pop[pop["PLAYER_ID"] == player_id]
    if len(row_match) == 0:
        raise ValueError(f"player_id {player_id} not found in {season} population")
    raw_row = row_match.iloc[0]
    mu, sd = fit["mu"], fit["sd"]
    z = (raw_row[feature_columns].astype(float).values - mu) / sd

    loading_rank = np.argsort(-np.abs(centroid))
    deviation = np.abs(z - centroid)
    forced_in = np.where(deviation > 1.0)[0]
    axis_idx = list(loading_rank[:n_axes])
    for i in forced_in:
        if i not in axis_idx:
            axis_idx.append(int(i))

    diffs = z - centroid
    top_dev_idx = np.argsort(-np.abs(diffs))[:3]

    return {
        "axis_idx": axis_idx,
        "player_z": z, "centroid_z": centroid,
        "top_deviations": [(int(i), float(diffs[i])) for i in top_dev_idx],
    }


# ===========================================================================
# Section E2 - value vs. role peers: OUT OF SCOPE
# ===========================================================================
# No salary table exists anywhere in this project (checked: no
# data/salaries*.csv, no salary column in any of the 8 tables in
# nets_synergy.db) and CLAUDE.md's own stated scope explicitly excludes
# salary/contract modeling ("no salary/contract modeling" - see the Brown
# study's Phase B4/B5 note). Per the task spec's own gate ("IF a salary
# table is available... check; if absent, SKIP with a visible one-line
# note, do not scrape mid-task"), this section is skipped - the portal
# shows the one-line note directly, no function needed here.


# ===========================================================================
# Section F - game-level bootstrap (GATE F: downgraded scope, disclosed)
# ===========================================================================
# GATE F: individually recomputing all 11 box-score-shaped features per
# resampled game (USG%/AST%/TRB%/STL%/BLK%/BPM) needs each game's TEAM and
# OPPONENT totals (shared-minutes-weighted usage/rebound/assist-rate
# formulas), which the box cache supports in principle but was judged too
# heavy to build reliably for one subsection of six - downgrading per the
# gate's own explicit permission. Bootstrapped here: the 4 features whose
# standard formula only needs the player's OWN per-game counts (no team
# denominator): TS%, FTr, TOV%, and PTS_PER_100 (via a per-minute-scoring
# ratio applied to his real season PTS_PER_100 - an analytic approximation,
# not a re-derivation of true team-pace-adjusted PTS_PER_100). The other 25
# features (including USG%/AST%/TRB%/STL%/BLK%/BPM, all 9 playtype shares,
# all 6 BBRef shot-location features, height) are held at their real season
# value in every resample - this makes the reported interval UNDERSTATE
# true uncertainty even more than the general season-level-features-fixed
# caveat already implies. Both caveats are surfaced on-page, not just here.

BOOTSTRAP_FEATURES = ["TS%", "FTr", "TOV%", "PTS_PER_100"]


def _season_from_game_id(game_id):
    """Standard NBA game-ID convention: '0022' + 2-digit season-start-year
    + 5-digit sequence, regular season only (preseason/playoffs use other
    prefixes, out of scope here since the stint tables are regular-season
    only already)."""
    game_id = str(game_id)
    if not game_id.startswith("0022"):
        return None
    start_year = 2000 + int(game_id[4:6])
    return f"{start_year}-{str(start_year + 1)[2:]}"


def player_game_ids(player_id, season, stints=None):
    """Every game this player appears in this season, found from the
    already-loaded stint table's own off_p/def_p roster columns (fast,
    vectorized) rather than scanning every box file in raw_cache."""
    if stints is None:
        stints, _ = _load_season_stints_v2_or_v1(season)
    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]
    mask = stints[off_cols].eq(player_id).any(axis=1) | stints[def_cols].eq(player_id).any(axis=1)
    return sorted(stints.loc[mask, "game_id"].unique().tolist())


def _parse_minutes(raw):
    raw = str(raw)
    if ":" not in raw:
        return 0.0
    m, s = raw.split(":")
    return float(m) + float(s) / 60.0


def load_player_game_box(player_id, season, stints=None):
    """GATE F check: this is the function whose output length determines
    whether the bootstrap can run at all - a traded player's games span
    two teams' box files, which is fine here since game discovery goes
    through the stint table's player-roster columns, not a team filter."""
    game_ids = player_game_ids(player_id, season, stints=stints)
    rows = []
    for gid in game_ids:
        path = RAW_CACHE_DIR / f"{gid}_box.parquet"
        if not path.exists():
            continue
        box = pd.read_parquet(path)
        prow = box[box["personId"] == player_id]
        if len(prow) == 0:
            continue
        r = prow.iloc[0]
        minutes = _parse_minutes(r["minutes"])
        if minutes <= 0:
            continue
        rows.append({
            "game_id": gid, "MIN": minutes,
            "PTS": float(r["points"]), "FGA": float(r["fieldGoalsAttempted"]),
            "FTA": float(r["freeThrowsAttempted"]), "TOV": float(r["turnovers"]),
        })
    return pd.DataFrame(rows)


def bootstrap_recipe(player_id, season, fit, recipes, k, B=500, seed=0, stints=None):
    """Game-level bootstrap of the 4 BOOTSTRAP_FEATURES; the other 25
    features held fixed at their real season value in every resample (see
    module note above). Returns available=False with a reason if GATE F's
    own extractability check fails, rather than silently downgrading
    further to a fabricated interval."""
    games = load_player_game_box(player_id, season, stints=stints)
    if len(games) == 0:
        return {"available": False,
                "reason": "no per-game box rows extractable from raw_cache for this player/season"}

    feature_columns = fit["feature_columns"]
    pop = load_population(min_threshold=0, season_min=season, season_max=season)
    row_match = pop[pop["PLAYER_ID"] == player_id]
    if len(row_match) == 0:
        return {"available": False, "reason": "player not found in season population (below MIN floor?)"}
    base_row = row_match.iloc[0]

    season_pts_per_100 = float(base_row["PTS_PER_100"])
    season_min_total = float(games["MIN"].sum())
    season_pts_total = float(games["PTS"].sum())
    season_pts_per_min = season_pts_total / season_min_total if season_min_total else np.nan

    rng = np.random.default_rng(seed)
    n = len(games)
    P = np.full((B, k), np.nan)
    for b in range(B):
        sample = games.sample(n=n, replace=True, random_state=int(rng.integers(0, 2**31 - 1)))
        fga, fta = sample["FGA"].sum(), sample["FTA"].sum()
        pts, tov, minutes = sample["PTS"].sum(), sample["TOV"].sum(), sample["MIN"].sum()

        ts_pct = pts / (2 * (fga + 0.44 * fta)) if (fga + fta) else np.nan
        ftr = fta / fga if fga else np.nan
        tov_pct = 100.0 * tov / (fga + 0.44 * fta + tov) if (fga + fta + tov) else np.nan
        pts_per_min = pts / minutes if minutes else np.nan
        pts_per_100 = (season_pts_per_100 * (pts_per_min / season_pts_per_min)
                       if season_pts_per_min else season_pts_per_100)

        x = base_row[feature_columns].astype(float).copy()
        x["TS%"], x["FTr"], x["TOV%"], x["PTS_PER_100"] = ts_pct, ftr, tov_pct, pts_per_100
        df1 = pd.DataFrame([x])
        P[b] = project(df1, fit["basis"], fit["mu"], fit["sd"], feature_columns)[0]

    lo = np.nanpercentile(P, 5, axis=0)
    hi = np.nanpercentile(P, 95, axis=0)

    arch_cols = [f"arch_{i}" for i in range(k)]
    season_match = recipes[(recipes["PLAYER_ID"] == player_id) & (recipes["SEASON"] == season)]
    point_estimate = (season_match.iloc[0][arch_cols].values.astype(float)
                       if len(season_match) else np.nanmean(P, axis=0))

    return {
        "available": True, "n_games": n, "B": B,
        "point_estimate": point_estimate, "lo5": lo, "hi95": hi,
        "bootstrapped_features": BOOTSTRAP_FEATURES,
        "fixed_features": [c for c in feature_columns if c not in BOOTSTRAP_FEATURES],
    }
