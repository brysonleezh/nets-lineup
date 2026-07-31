"""
Step 4 - Roster construction: does archetype mixture predict lineup outcomes,
and does that relationship survive going from a partial lineup to a full one.

GATE RESULT (already checked before this file was written): 5-man lineups at
MIN>=100 in 2025-26 number only 78 - too few to fit a 10-predictor model
(sum_bpm + 7 archetype shares + entropy) without massively overfitting. The
4-man unit (915 rows) is the primary fitting unit here; 2-man (2,705) and
3-man (3,143) are robustness checks in the OTHER direction - do the same
coefficients hold at a smaller n, where more of the lineup is unobserved.
The 78 five-man lineups become an extrapolation check on the 4-man-fitted
model, not a training set and not a clean holdout (same players, same season).

Every model here is: outcome ~ sum_bpm + (K-1 archetype shares) + entropy,
fit by weighted least squares with weights = shared MIN. Three requirements
drive that spec (see fit_outcome_model's docstring for the full reasoning):
sum_bpm so the archetype coefficients aren't secretly detecting where stars
play; K-1 shares (not K) because shares sum to 1 and are collinear with the
intercept - one archetype must be dropped as reference; entropy because a
model linear in shares alone has a degenerate optimum (concentrate all
weight on the single best-coefficient archetype), and entropy is the one
extra number that answers "does balance help, holding composition and talent
fixed."

Part A  build_lineup_table - per-lineup mixture/entropy/sum_bpm/outcomes
Part B  fit_outcome_model - WLS + reference-archetype handling + team-cluster
        bootstrap CIs
Part C  cross_validate - GroupKFold, baseline (sum_bpm only) vs full
Part D  granularity_comparison - same model at n=2/3/4, coefficients aligned
        to one shared reference archetype so they're actually comparable;
        also reports each share's weighted SD and the standardized
        (coef x SD) coefficient at each n, since raw coefficients alone
        aren't comparable across n - a 4-man mixture averages four recipes
        and so has less share variance than a 2-man one, which mechanically
        inflates raw coefficients at higher n with no contamination required
Part E  five_man_extrapolation - score the 78 five-man MIN>=100 lineups with
        the 4-man-fitted model AND a sum_bpm-only baseline fit on the same
        4-man table, so the full model's correlation has something to beat
Part F  predict_substitution - the actionable output: shift delta of weight
        from one archetype to another, report the predicted change
Part G  score_roster_combos (enumerate_roster_combos + score_combos) -
        enumerate every N-man combo from a SPECIFIC roster (not just the
        combos that already exist as real `lineups` rows) and apply an
        already-fitted model's coefficients - added to back the portal's
        Roster Construction lineup-ranking feature (see portal.py) after
        a parallel possession-level RAPM effort failed its own
        out-of-sample validation gate (see RAPM_README.md's GATE 2 section)
Part H  phase runner + report

STATUS: run end-to-end against real 2025-26 data (data/basis_2025_26,
K=8). Printed numbers in this file's own __main__ block are real output,
not placeholders - see the run log for the actual R^2s, coefficients, and
granularity comparison.
"""

from __future__ import annotations

import sqlite3
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

from step0_data_collect_process import DB_PATH, build_nba_side_tables
from step2_diagnostic_analysis import _parse_group_ids

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

N_BOOTSTRAP = 1000
BOOTSTRAP_SEED = 0

OUTCOME_COLUMNS_RAW = [
    "OFF_RATING", "DEF_RATING", "NET_RATING", "EFG_PCT",
    "TM_TOV_PCT", "REB_PCT", "OREB_PCT", "AST_PCT", "PACE",
]
# Final outcome columns after the sign flips below - every one of these
# reads "higher is better" (PACE has no better/worse direction and is
# passed through unchanged).
OUTCOME_COLUMNS = [
    "OFF_RATING", "DEF_RATING_INV", "NET_RATING", "EFG_PCT",
    "TM_TOV_PCT_INV", "REB_PCT", "OREB_PCT", "AST_PCT", "PACE",
]


# --- Part A: build_lineup_table -------------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: build a per-lineup table (mixture/entropy/sum_bpm/outcomes) generalized
over GROUP_QUANTITY, given the already-checked gate result that 5-man MIN>=100
lineups (78 rows) are too thin to fit directly and 4-man (915) is the primary
unit.
Used: reuses step2_diagnostic_analysis's own `_parse_group_ids` (GROUP_ID
split on '-', empty strings dropped, sorted tuple of ints) rather than
writing a third copy of the same dash-parsing logic - that function already
IS the n-agnostic generalization of step2_synergy_matrix's original 2-only
`_parse_pair_ids` (it was written for Part D's exact-set lineup lookups,
which needed to work at any group_quantity). Player-ID -> recipe/BPM lookups
are a single vectorized merge over an exploded long-format table (one row
per lineup x per player slot), not a per-lineup python loop - the same
scaling reasoning step2's attach_recipes/attach_quality_covariate used, just
generalized from 2 slots to n.
Not AI: which two outcome columns needed a sign flip (DEF_RATING - points
allowed per 100, lower is better defense; TM_TOV_PCT - the lineup's own
turnover rate, lower is better) and the decision to do the flip once here
so no downstream chart or coefficient needs a "remember this one is
backwards" footnote.
"""


def _entropy_vectorized(mixture_matrix):
    """exp(-sum p log p) per row - the effective number of archetypes a
    mixture spans. Range is [1, K]: exactly 1 when all weight sits on a
    single archetype (no diversity), K when weight is perfectly uniform
    across all K. 0*log(0) := 0 by the usual convention (an archetype with
    zero weight contributes nothing to the sum, not a NaN)."""
    p = mixture_matrix
    safe_p = np.where(p > 0, p, 1.0)  # log(1)=0, paired with p=0 -> contributes 0
    plogp = p * np.log(safe_p)
    return np.exp(-plogp.sum(axis=1))


def build_lineup_table(recipes, k, season, group_quantity, min_threshold=100, db_path=None):
    """One row per lineup at this group_quantity, MIN >= min_threshold, in
    `season`. Columns: GROUP_ID, TEAM_ABBREVIATION, MIN, arch_0..arch_{k-1}
    (mixture - mean of the n players' archetype recipes, sums to 1),
    entropy, sum_bpm, then the 9 outcome columns.

    DEF_RATING_INV = -DEF_RATING and TM_TOV_PCT_INV = -TM_TOV_PCT: both
    source columns are "lower is better" (points allowed, turnover rate),
    flipped once here so every outcome in this table reads "higher is
    better" and no chart or coefficient sign needs to explain a reversed
    axis downstream.

    Players are joined to their archetype recipe and BPM by PLAYER_ID only
    (never by name - this project's own recurring bug class). A lineup is
    dropped if ANY of its n players is missing a recipe or a BPM (below the
    MIN>=300 individual-season floor that both recipes.csv and
    build_nba_side_tables() already enforce) - drop count is reported since
    that's the real usable sample size, not the raw row count.
    """
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            f"SELECT GROUP_ID, TEAM_ABBREVIATION, MIN, {', '.join(OUTCOME_COLUMNS_RAW)} "
            "FROM lineups WHERE SEASON = ? AND GROUP_QUANTITY = ? AND MIN >= ?",
            conn, params=(season, group_quantity, min_threshold),
        )
    n_raw = len(df)
    if n_raw == 0:
        raise ValueError(f"no lineups at group_quantity={group_quantity}, MIN>={min_threshold}, season={season}")

    df["DEF_RATING_INV"] = -df["DEF_RATING"]
    df["TM_TOV_PCT_INV"] = -df["TM_TOV_PCT"]
    df = df.drop(columns=["DEF_RATING", "TM_TOV_PCT"])

    ids_per_lineup = df["GROUP_ID"].apply(lambda g: list(_parse_group_ids(g)))
    bad_n = ids_per_lineup.apply(len) != group_quantity
    if bad_n.any():
        raise ValueError(
            f"{int(bad_n.sum())} lineup(s) parsed to a player count != "
            f"group_quantity={group_quantity} - GROUP_ID format check needed"
        )

    long_df = pd.DataFrame({
        "_lineup_idx": np.repeat(df.index.values, group_quantity),
        "PLAYER_ID": np.concatenate(ids_per_lineup.values),
    })

    arch_cols = [f"arch_{i}" for i in range(k)]
    season_recipes = recipes[recipes["SEASON"] == season][["PLAYER_ID"] + arch_cols]
    bpm = build_nba_side_tables()
    bpm = bpm[bpm["SEASON"] == season][["PLAYER_ID", "BPM"]]

    long_df = long_df.merge(season_recipes, on="PLAYER_ID", how="left")
    long_df = long_df.merge(bpm, on="PLAYER_ID", how="left")

    # A merge miss leaves every arch_ column AND BPM null together for that
    # slot (single-row merge) - checking arch_0 stands in for "recipe missing".
    long_df["_ok"] = long_df["arch_0"].notna() & long_df["BPM"].notna()
    n_ok = long_df.groupby("_lineup_idx")["_ok"].sum()
    complete_idx = n_ok[n_ok == group_quantity].index

    n_dropped = n_raw - len(complete_idx)
    if n_dropped:
        print(f"build_lineup_table[n={group_quantity}]: dropped {n_dropped}/{n_raw} lineups "
              f"(>=1 of {group_quantity} players missing a recipe or BPM - "
              f"below the MIN>=300 individual-season floor)")

    complete_long = long_df[long_df["_lineup_idx"].isin(complete_idx)]
    mixture = complete_long.groupby("_lineup_idx")[arch_cols].mean()
    sum_bpm = complete_long.groupby("_lineup_idx")["BPM"].sum().rename("sum_bpm")

    out = df.loc[complete_idx].join(mixture).join(sum_bpm)
    out["entropy"] = _entropy_vectorized(out[arch_cols].values)

    out = out[["GROUP_ID", "TEAM_ABBREVIATION", "MIN"] + arch_cols + ["entropy", "sum_bpm"] + OUTCOME_COLUMNS]
    out = out.reset_index(drop=True)

    print(f"build_lineup_table[n={group_quantity}]: {len(out)} lineups, "
          f"MIN>={min_threshold}, season={season}, {out['TEAM_ABBREVIATION'].nunique()} teams")
    return out


# --- Part B: fit_outcome_model ---------------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: fit outcome ~ sum_bpm + (K-1 archetype shares) + entropy by weighted
least squares (weights=MIN), drop one archetype as reference (default:
largest total exposure, must be returned since every coefficient is
"relative to that archetype"), and get bootstrap CIs that account for the
same player appearing in dozens of lineups by resampling TEAMS rather than
rows, 1000 draws.
Used: sklearn LinearRegression(sample_weight=...) for the WLS fit - this
project's own stack decision (sklearn, not statsmodels; statsmodels isn't
even in requirements.txt) rather than adding a new dependency. Reference
archetype's "largest total exposure" is defined as minutes-weighted total
exposure (sum over lineups of MIN * mixture share) rather than a raw
unweighted mean - consistent with weighting everything else in this model
by MIN, and it answers a concrete question ("which archetype actually
soaks up the most played minutes"), not an arbitrary tiebreak. The
team-cluster bootstrap resamples whole teams with replacement (a sampled
team contributes ALL of its rows, possibly more than once if resampled
more than once) and refits WLS on each resampled table - the standard
pairs-cluster bootstrap for clustered/panel data, not a per-row resample
that would understate variance given the same player recurring across many
rows for the same team.
Not AI: the model specification itself (sum_bpm + shares + entropy, weights
= MIN) and the three requirements in the module docstring - all given
directly, not proposed here.
"""


def _archetype_share_cols(k, reference_archetype):
    return [f"arch_{i}" for i in range(k) if i != reference_archetype]


def _build_design(lineup_df, k, reference_archetype):
    """Design matrix + column names for outcome ~ sum_bpm + (K-1 shares) +
    entropy. Column order fixed: sum_bpm, then the K-1 non-reference shares
    (ascending archetype index), then entropy - same order fit_outcome_model,
    cross_validate, five_man_extrapolation, and predict_substitution all
    assume."""
    share_cols = _archetype_share_cols(k, reference_archetype)
    X = np.column_stack([
        lineup_df["sum_bpm"].to_numpy(dtype=float),
        lineup_df[share_cols].to_numpy(dtype=float),
        lineup_df["entropy"].to_numpy(dtype=float),
    ])
    feature_names = ["sum_bpm"] + [f"{c}_share" for c in share_cols] + ["entropy"]
    return X, feature_names


def _pick_reference_archetype(lineup_df, k):
    """Default reference: the archetype with the largest minutes-weighted
    total exposure, i.e. sum(MIN * mixture_share) across lineups - the
    archetype actually soaking up the most played minutes in this table."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    weighted_exposure = (lineup_df[arch_cols].to_numpy() * lineup_df["MIN"].to_numpy()[:, None]).sum(axis=0)
    return int(np.argmax(weighted_exposure))


def _weighted_r2(y_true, y_pred, w):
    wmean = np.average(y_true, weights=w)
    ss_res = np.sum(w * (y_true - y_pred) ** 2)
    ss_tot = np.sum(w * (y_true - wmean) ** 2)
    return 1.0 - ss_res / ss_tot


def _weighted_std(x, w):
    """Population weighted SD (weights=MIN, same convention as _weighted_r2's
    weighted mean/SS_tot) - not a bias-corrected sample estimate."""
    wmean = np.average(x, weights=w)
    return float(np.sqrt(np.average((x - wmean) ** 2, weights=w)))


def _weighted_corr(x, y, w):
    wx = np.average(x, weights=w)
    wy = np.average(y, weights=w)
    cov = np.average((x - wx) * (y - wy), weights=w)
    var_x = np.average((x - wx) ** 2, weights=w)
    var_y = np.average((y - wy) ** 2, weights=w)
    return float(cov / np.sqrt(var_x * var_y))


def _bootstrap_teams(lineup_df, outcome, k, reference_archetype, n_boot=N_BOOTSTRAP, seed=BOOTSTRAP_SEED):
    """Resample TEAMS with replacement (not rows) and refit WLS each draw -
    the same player appears in dozens of lineups for the same team, so a
    row-level bootstrap would treat those as independent when they aren't
    and understate the true standard errors. A sampled team contributes
    every one of its rows; a team sampled twice contributes its rows twice.
    """
    rng = np.random.default_rng(seed)
    teams = lineup_df["TEAM_ABBREVIATION"].unique()
    n_teams = len(teams)
    by_team = {t: g for t, g in lineup_df.groupby("TEAM_ABBREVIATION")}

    n_features = 2 + (k - 1)  # sum_bpm + (k-1) shares + entropy
    coefs = np.empty((n_boot, n_features))
    intercepts = np.empty(n_boot)

    for b in range(n_boot):
        sampled_teams = rng.choice(teams, size=n_teams, replace=True)
        boot_df = pd.concat([by_team[t] for t in sampled_teams], ignore_index=True)
        X, _ = _build_design(boot_df, k, reference_archetype)
        y = boot_df[outcome].to_numpy(dtype=float)
        w = boot_df["MIN"].to_numpy(dtype=float)

        m = LinearRegression(fit_intercept=True)
        m.fit(X, y, sample_weight=w)
        coefs[b] = m.coef_
        intercepts[b] = m.intercept_

    return coefs, intercepts


def fit_outcome_model(lineup_df, outcome, k, reference_archetype=None):
    """Weighted least squares: outcome ~ sum_bpm + (K-1 archetype shares) +
    entropy, weights = MIN.

    Three requirements baked in (see module docstring for the full
    reasoning): (1) shares sum to 1 and are collinear with the intercept,
    so one archetype is dropped as reference - defaults to the largest
    minutes-weighted total exposure, and is ALWAYS returned in the result
    dict, since every coefficient means "relative to that archetype" and is
    unreadable without it; (2) sum_bpm is always in the design, so the
    archetype coefficients can't just be detecting where the stars play;
    (3) entropy is the one extra predictor that separates "does balance
    help" from "which composition works," since a model linear in shares
    alone has a degenerate optimum (put all weight on the single best
    coefficient).

    Standard errors come from a team-cluster bootstrap (1000 draws,
    resampling TEAMS not rows - see _bootstrap_teams), since the same
    player recurs across dozens of lineups and a row-level bootstrap or
    naive OLS standard error would understate that non-independence.
    """
    if reference_archetype is None:
        reference_archetype = _pick_reference_archetype(lineup_df, k)

    X, feature_names = _build_design(lineup_df, k, reference_archetype)
    y = lineup_df[outcome].to_numpy(dtype=float)
    w = lineup_df["MIN"].to_numpy(dtype=float)

    model = LinearRegression(fit_intercept=True)
    model.fit(X, y, sample_weight=w)
    r2 = model.score(X, y, sample_weight=w)

    boot_coefs, boot_intercepts = _bootstrap_teams(lineup_df, outcome, k, reference_archetype)
    ci_lower = np.percentile(boot_coefs, 2.5, axis=0)
    ci_upper = np.percentile(boot_coefs, 97.5, axis=0)
    intercept_ci = tuple(np.percentile(boot_intercepts, [2.5, 97.5]))

    print(f"fit_outcome_model[{outcome}, n={len(y)}]: reference=arch_{reference_archetype}, "
          f"weighted R^2={r2:.4f}")

    return {
        "outcome": outcome, "k": int(k), "n": int(len(y)),
        "reference_archetype": reference_archetype,
        "feature_names": feature_names,
        "coef": model.coef_, "intercept": float(model.intercept_),
        "ci_lower": ci_lower, "ci_upper": ci_upper, "intercept_ci": intercept_ci,
        "r2_weighted": float(r2),
    }


def report_coefficients(model):
    """Full coefficient table with bootstrap CIs, for the 'Report before
    returning' requirement."""
    rows = [{"feature": name, "coef": c, "ci_lower": lo, "ci_upper": hi}
            for name, c, lo, hi in zip(model["feature_names"], model["coef"], model["ci_lower"], model["ci_upper"])]
    rows.append({"feature": "intercept", "coef": model["intercept"],
                 "ci_lower": model["intercept_ci"][0], "ci_upper": model["intercept_ci"][1]})
    table = pd.DataFrame(rows)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    return table


# --- Part C: cross_validate --------------------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: GroupKFold on TEAM_ABBREVIATION comparing baseline (sum_bpm only)
against the full model, returning out-of-sample R^2 for both plus a boolean
for whether full beats baseline, so the UI can grey out an outcome that
fails this rather than present coefficients that predict nothing out of
sample.
Used: GroupKFold (not plain KFold) so no team's lineups leak across the
train/test split - a random row split would let e.g. two different 4-man
BOS combos from the same team end up on opposite sides, which isn't a real
out-of-sample test of whether the model generalizes to a new team's rosters.
Weighted out-of-sample R^2 uses the TEST fold's own weighted mean as the
SS_tot baseline (the standard R^2 convention, not the train set's mean).
Not AI: the baseline-vs-full comparison itself and the requirement to flag
outcomes where full doesn't beat baseline - given directly.
"""


def cross_validate(lineup_df, outcome, k, n_splits=5, reference_archetype=None):
    """GroupKFold on TEAM_ABBREVIATION. Compares baseline (sum_bpm only)
    against full (sum_bpm + K-1 shares + entropy). Returns out-of-sample
    weighted R^2 for both plus a boolean for whether full beats baseline -
    an outcome that fails this should be flagged so downstream (e.g. the
    portal) can grey it out rather than present coefficients that don't
    predict anything out of sample.
    """
    if reference_archetype is None:
        reference_archetype = _pick_reference_archetype(lineup_df, k)

    X_full, _ = _build_design(lineup_df, k, reference_archetype)
    X_base = lineup_df[["sum_bpm"]].to_numpy(dtype=float)
    y = lineup_df[outcome].to_numpy(dtype=float)
    w = lineup_df["MIN"].to_numpy(dtype=float)
    groups = lineup_df["TEAM_ABBREVIATION"].to_numpy()

    gkf = GroupKFold(n_splits=n_splits)
    r2_full_folds, r2_base_folds = [], []
    for train_idx, test_idx in gkf.split(X_full, y, groups=groups):
        m_full = LinearRegression(fit_intercept=True).fit(X_full[train_idx], y[train_idx], sample_weight=w[train_idx])
        m_base = LinearRegression(fit_intercept=True).fit(X_base[train_idx], y[train_idx], sample_weight=w[train_idx])

        r2_full_folds.append(_weighted_r2(y[test_idx], m_full.predict(X_full[test_idx]), w[test_idx]))
        r2_base_folds.append(_weighted_r2(y[test_idx], m_base.predict(X_base[test_idx]), w[test_idx]))

    r2_full = float(np.mean(r2_full_folds))
    r2_base = float(np.mean(r2_base_folds))
    full_beats_baseline = r2_full > r2_base

    print(f"cross_validate[{outcome}]: baseline (sum_bpm only) OOS R^2={r2_base:.4f}, "
          f"full OOS R^2={r2_full:.4f} -> full beats baseline: {full_beats_baseline}")

    return {
        "outcome": outcome, "r2_baseline": r2_base, "r2_full": r2_full,
        "full_beats_baseline": bool(full_beats_baseline),
        "r2_baseline_folds": r2_base_folds, "r2_full_folds": r2_full_folds,
    }


# --- Part D: granularity_comparison ------------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: fit the same model at group_quantity 2, 3, 4 and return the
coefficients side by side - the key robustness result, since an n-man
lineup's rating includes the (5-n) unobserved teammates' contribution and
sum_bpm only controls for the n observed ones; going 2->4 shrinks the
unobserved count from 3 to 1, so coefficients that hold across all three
aren't being driven by that contamination, and coefficients that drift
systematically with n are, with the drift's direction saying how.
Used: fits group_quantity=4 first to pick ONE reference archetype (via
_pick_reference_archetype), then re-fits 2 and 3 with that SAME reference
passed in explicitly, rather than letting each group_quantity pick its own
argmax independently - if 2-man and 4-man picked different reference
archetypes, "the coefficient on arch_3_share" would mean a different
baseline comparison at each n and the side-by-side table would compare
apples to oranges, defeating the point of this function.
Not AI: the reasoning connecting unobserved-teammate count to coefficient
drift, and the choice of 2/3/4 (not 5) as the compared granularities -
given directly (5-man is the extrapolation check in Part E, not part of
this comparison, per the already-checked gate result).
"""


def granularity_comparison(recipes, k, season, outcome, min_threshold=100):
    """Fit outcome ~ sum_bpm + (K-1 shares) + entropy at group_quantity
    2, 3, and 4, holding ONE reference archetype fixed across all three
    (picked from the 4-man table, the primary unit) so the coefficients are
    directly comparable.

    The reasoning this result tests: an n-man lineup's rating already
    includes the (5-n) unobserved teammates' contribution, and sum_bpm only
    controls for the n players actually in the design. Going from 2-man to
    4-man shrinks that unobserved count from 3 to 1. A coefficient that
    holds roughly constant across n=2/3/4 is not being driven by that
    contamination; a coefficient that drifts systematically with n is, and
    the sign/direction of the drift says which way.

    CAVEAT this function now checks directly: raw coefficients are not
    comparable across n on their own. A 4-man mixture averages four
    recipes, so each share variable has less variance at n=4 than at n=2 -
    the same underlying effect produces a mechanically larger raw
    coefficient at higher n, with no unobserved-teammate contamination
    required to produce that pattern. For each archetype share, this
    reports the raw coefficient, that share's weighted SD at this n, and
    the standardized coefficient (raw coef x SD) side by side - if the
    standardized coefficients are flat across n while the raw ones drift,
    the drift is variance scaling, not contamination, and any contamination
    claim built on the raw numbers alone has to be withdrawn.
    """
    lineup_4 = build_lineup_table(recipes, k, season, 4, min_threshold=min_threshold)
    reference_archetype = _pick_reference_archetype(lineup_4, k)
    share_cols = _archetype_share_cols(k, reference_archetype)

    fits, sds, lineup_tables = {}, {}, {4: lineup_4}
    for gq in (2, 3, 4):
        if gq not in lineup_tables:
            lineup_tables[gq] = build_lineup_table(recipes, k, season, gq, min_threshold=min_threshold)
        lineup_df = lineup_tables[gq]
        fits[gq] = fit_outcome_model(lineup_df, outcome, k, reference_archetype=reference_archetype)
        w = lineup_df["MIN"].to_numpy(dtype=float)
        sds[gq] = {c: _weighted_std(lineup_df[c].to_numpy(dtype=float), w) for c in share_cols}

    feature_names = fits[4]["feature_names"]
    rows = []
    for i, feat in enumerate(feature_names):
        row = {"feature": feat}
        raw_col = feat[:-len("_share")] if feat.endswith("_share") else None
        for gq in (2, 3, 4):
            coef = fits[gq]["coef"][i]
            row[f"coef_n{gq}"] = coef
            if raw_col is not None:
                sd = sds[gq][raw_col]
                row[f"sd_n{gq}"] = sd
                row[f"std_coef_n{gq}"] = coef * sd
            else:
                row[f"sd_n{gq}"] = np.nan
                row[f"std_coef_n{gq}"] = np.nan
        rows.append(row)
    comparison_table = pd.DataFrame(rows)

    print(f"\ngranularity_comparison[{outcome}]: reference=arch_{reference_archetype} "
          f"(fixed across n=2/3/4 from the 4-man table)")
    print("raw coefficients, weighted SD of each share at this n, and standardized coefficient (coef x SD):")
    print(comparison_table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    for gq in (2, 3, 4):
        print(f"  n={gq}: n_lineups={fits[gq]['n']}, weighted R^2={fits[gq]['r2_weighted']:.4f}")

    return {"reference_archetype": reference_archetype, "fits": fits, "sds": sds, "comparison_table": comparison_table}


# --- Part E: five_man_extrapolation ------------------------------------------

def five_man_extrapolation(model, recipes, k, season, min_threshold=100):
    """Score the 78 five-man MIN>=100 lineups (2025-26) with the
    4-man-fitted `model`, report the weighted correlation between predicted
    and actual outcome - alongside the same score for a sum_bpm-only
    baseline (fit on the same 4-man training table `model` used), since the
    full model's correlation is uninterpretable in isolation: if the
    baseline reaches a similar number, the archetype-mixture terms added
    nothing at n=5 even if they looked useful at n=4.

    NOT a clean holdout - the same players and season fed the 4-man fit, so
    this doesn't estimate out-of-sample error the way cross_validate does.
    It's an extrapolation check on one specific question: does the
    mixture-to-outcome relationship fit at n=4 still hold shape at full
    (n=5) lineup size, where CLAUDE.md's own gate result says there isn't
    enough data to fit directly.
    """
    lineup_5 = build_lineup_table(recipes, k, season, 5, min_threshold=min_threshold)
    outcome = model["outcome"]

    X, feature_names = _build_design(lineup_5, k, model["reference_archetype"])
    assert feature_names == model["feature_names"], "reference archetype mismatch between model and 5-man design"

    pred_full = model["intercept"] + X @ model["coef"]
    actual = lineup_5[outcome].to_numpy(dtype=float)
    w = lineup_5["MIN"].to_numpy(dtype=float)
    corr_full = _weighted_corr(actual, pred_full, w)

    lineup_4 = build_lineup_table(recipes, k, season, 4, min_threshold=min_threshold)
    baseline = LinearRegression(fit_intercept=True).fit(
        lineup_4[["sum_bpm"]].to_numpy(dtype=float), lineup_4[outcome].to_numpy(dtype=float),
        sample_weight=lineup_4["MIN"].to_numpy(dtype=float),
    )
    pred_baseline = baseline.predict(lineup_5[["sum_bpm"]].to_numpy(dtype=float))
    corr_baseline = _weighted_corr(actual, pred_baseline, w)

    print(f"five_man_extrapolation[{outcome}]: n={len(actual)} five-man lineups (MIN>={min_threshold})")
    print(f"  full model (sum_bpm + shares + entropy): weighted corr(predicted, actual) = {corr_full:.4f}")
    print(f"  sum_bpm-only baseline:                   weighted corr(predicted, actual) = {corr_baseline:.4f}")
    print("  NOTE: extrapolation check, not validation - same players/season as the 4-man fit, "
          "not an out-of-sample holdout.")

    return {
        "outcome": outcome, "n": int(len(actual)),
        "weighted_corr_full": corr_full, "weighted_corr_baseline": corr_baseline,
        "predicted_full": pred_full, "predicted_baseline": pred_baseline,
        "actual": actual, "weights": w,
    }


# --- Part F: predict_substitution ---------------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: shift delta of weight from one archetype to another and return the
predicted change - the actionable output, sidestepping solving for an
optimum.
Used: sum_bpm is NOT a parameter here and doesn't need to be - a
substitution query holds personnel/talent fixed (same lineup, only its
archetype composition shifts), so sum_bpm's coefficient contributes
identically to the "before" and "after" score and cancels exactly out of
the difference. entropy is NOT held fixed - it's recomputed from the new
mixture, since concentrating or spreading weight across archetypes is
exactly what the entropy term measures, and holding it fixed while shares
move would silently misstate the predicted effect of an entropy-changing
substitution as pure composition effect.
Not AI: the decision to omit sum_bpm as an argument (a consequence of the
model spec's structure, but the choice to lean on that rather than requiring
callers to supply an arbitrary sum_bpm value was made here) - flagged, not
independently decided.
"""


def predict_substitution(model, mixture, from_archetype, to_archetype, delta=0.10):
    """Predicted change in `model`'s outcome from shifting `delta` of
    archetype weight from `from_archetype` to `to_archetype`, holding
    sum_bpm (personnel/talent) fixed - same lineup, only the mix changes,
    so sum_bpm's coefficient cancels exactly out of the before/after
    difference and isn't needed as an argument. entropy is NOT held fixed:
    it's recomputed from the shifted mixture, since a substitution that
    concentrates or spreads weight is exactly what the entropy term is
    meant to capture.
    """
    mixture = np.asarray(mixture, dtype=float)
    if mixture[from_archetype] < delta - 1e-9:
        raise ValueError(f"only {mixture[from_archetype]:.3f} of weight is on arch_{from_archetype}, "
                          f"can't shift {delta}")

    new_mixture = mixture.copy()
    new_mixture[from_archetype] -= delta
    new_mixture[to_archetype] += delta

    k = model["k"]
    ref = model["reference_archetype"]
    share_idx = [i for i in range(k) if i != ref]
    coef_by_feature = dict(zip(model["feature_names"], model["coef"]))

    delta_pred = 0.0
    for i in share_idx:
        delta_pred += coef_by_feature[f"arch_{i}_share"] * (new_mixture[i] - mixture[i])

    ent_before = float(_entropy_vectorized(mixture[None, :])[0])
    ent_after = float(_entropy_vectorized(new_mixture[None, :])[0])
    delta_pred += coef_by_feature["entropy"] * (ent_after - ent_before)

    print(f"predict_substitution[{model['outcome']}]: shift {delta:.2f} weight "
          f"arch_{from_archetype} -> arch_{to_archetype}: predicted change = {delta_pred:+.4f} "
          f"(entropy {ent_before:.3f} -> {ent_after:.3f})")

    return {
        "outcome": model["outcome"], "delta_predicted": float(delta_pred),
        "mixture_before": mixture, "mixture_after": new_mixture,
        "entropy_before": ent_before, "entropy_after": ent_after,
    }


# --- Part G: score_roster_combos ---------------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: wire this file's already-validated WLS mixture model into the
Streamlit portal's Roster Construction page, after a parallel possession-
level RAPM effort (src/fit_rapm.py) failed its own out-of-sample validation
gate (GATE 2: Net correlation 0.12-0.17 at real sample sizes) and the user
chose to redirect to this file instead.
Used: this file's fit_outcome_model only ever fits on lineups that actually
EXIST in the `lineups` table (real group_quantity=4 combos, plus a n=5
extrapolation check on the 78 real five-man lineups) - it has no function
that scores an ARBITRARY hypothetical n-man combo the way step3's
score_all_combos does for the (separate, unrun) synergy-matrix pipeline.
Added exactly that, reusing this file's own _entropy_vectorized and
_archetype_share_cols rather than a third reimplementation - a combo's
mixture is the mean of its players' recipe vectors (same convention
build_lineup_table uses for real lineups), sum_bpm is a plain sum, entropy
is computed off the mixture, and the model's fitted (already-cross-
validated) coefficients are applied via the same linear formula
_build_design encodes, not a fresh fit - these are the SAME 5-man
combinations step5_man_extrapolation already found the model holds up on
(corr=0.556 vs a sum_bpm-only baseline's 0.516, n=78 real five-man lineups),
so applying it to the Nets' own hypothetical combos is the same
extrapolation already validated, just enumerated over a specific roster
instead of only the 78 that happen to already exist as real lineups.
Not AI: the decision to redirect Roster Construction to this file at all -
made directly by the user after GATE 2's failure was presented.
"""


def enumerate_roster_combos(roster_ids, recipes, season, k, bpm_lookup, combo_size=5):
    """Every combo_size-man combination from roster_ids that has BOTH a real
    archetype recipe AND a real BPM (season, per-player - the same
    eligibility rule build_lineup_table applies to real lineups; a
    roster player missing either is dropped from the ELIGIBLE POOL, not
    silently zero-filled, so a combo including him is never enumerated).
    Returns a DataFrame: one row per combo (player_ids tuple, arch_0..
    arch_{k-1} mean mixture, entropy, sum_bpm) - NOT yet scored; pass to
    score_combos() with a fitted model.
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    season_recipes = recipes[recipes["SEASON"] == season].set_index("PLAYER_ID")

    eligible = [pid for pid in roster_ids if pid in season_recipes.index and pid in bpm_lookup.index]
    dropped = sorted(set(roster_ids) - set(eligible))
    if dropped:
        print(f"enumerate_roster_combos: {len(dropped)}/{len(roster_ids)} roster player(s) "
              f"excluded from the combo pool (no recipe and/or no BPM this season): {dropped}")
    if len(eligible) < combo_size:
        raise ValueError(f"only {len(eligible)} eligible roster players, need >= {combo_size}")

    rows = []
    for combo in combinations(eligible, combo_size):
        mixture = season_recipes.loc[list(combo), arch_cols].mean(axis=0)
        sum_bpm = float(bpm_lookup.loc[list(combo)].sum())
        rows.append({"player_ids": combo, **mixture.to_dict(), "sum_bpm": sum_bpm})
    out = pd.DataFrame(rows)
    out["entropy"] = _entropy_vectorized(out[arch_cols].to_numpy(dtype=float))
    print(f"enumerate_roster_combos: {len(out)} combo_size={combo_size} combinations "
          f"from {len(eligible)} eligible players")
    return out


def score_combos(model, combo_df, k):
    """Applies `model`'s already-fitted (already cross-validated - see
    cross_validate) coefficients to combo_df via the SAME linear formula
    _build_design encodes for real lineups - no refit, this is pure
    application of an existing model to new (hypothetical) rows. Returns
    combo_df with a `predicted` column, sorted descending."""
    share_cols = _archetype_share_cols(k, model["reference_archetype"])
    X = np.column_stack([
        combo_df["sum_bpm"].to_numpy(dtype=float),
        combo_df[share_cols].to_numpy(dtype=float),
        combo_df["entropy"].to_numpy(dtype=float),
    ])
    out = combo_df.copy()
    out["predicted"] = model["intercept"] + X @ model["coef"]
    return out.sort_values("predicted", ascending=False).reset_index(drop=True)


# --- Part H: phase runner + report -------------------------------------------

def report_model(model, cv):
    print(f"\n=== {model['outcome']} @ group_quantity=4 ===")
    print(f"n = {model['n']}")
    print(f"reference archetype = arch_{model['reference_archetype']}")
    print(f"weighted R^2 (in-sample) = {model['r2_weighted']:.4f}")
    print(f"cross-validated R^2: baseline (sum_bpm only) = {cv['r2_baseline']:.4f}, "
          f"full (sum_bpm + shares + entropy) = {cv['r2_full']:.4f} "
          f"({'full beats baseline' if cv['full_beats_baseline'] else 'full does NOT beat baseline'})")
    print("\ncoefficient table (95% CI, team-cluster bootstrap, 1000 draws):")
    report_coefficients(model)


def phase_report(basis_dir=None, season="2025-26", min_threshold=100, outcome="OFF_RATING"):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from step1_archetypes_model import load_basis

    basis_dir = Path(basis_dir) if basis_dir else DATA_DIR / "basis_2025_26"
    basis = load_basis(basis_dir)
    k = basis["k"]
    recipes = pd.read_csv(basis_dir / "recipes.csv")

    lineup_4 = build_lineup_table(recipes, k, season, group_quantity=4, min_threshold=min_threshold)

    model = fit_outcome_model(lineup_4, outcome, k)
    cv = cross_validate(lineup_4, outcome, k)
    report_model(model, cv)

    granularity = granularity_comparison(recipes, k, season, outcome, min_threshold=min_threshold)
    extrapolation = five_man_extrapolation(model, recipes, k, season, min_threshold=min_threshold)

    return {"model": model, "cross_validate": cv, "granularity": granularity, "extrapolation": extrapolation}


if __name__ == "__main__":
    phase_report()
