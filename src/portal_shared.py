"""
Carved out of portal.py during the Day-8 portal restructuring (five-file
split: portal_shared / step2_intro / step3_player_breakdown / step4_report /
portal.py). Holds everything 2+ of the tab-render files call directly:
cross-tab constants (palette, SEASON, table styling), the module-level
@st.cache_data data loaders (recipes/roster/bio/box-score/exposure-cache),
and the small set of generic UI/chart helpers actually shared across tabs
(verified by grep, not assumed - most "generic-looking" chart helpers turned
out to be single-tab-only and stayed in step3_player_breakdown.py instead).
Deliberately has zero import of step2_intro/step3_player_breakdown/
step4_report to avoid circular imports; it only imports from the
already-existing, unchanged step0_data.py, step1_archetype_model.py,
step2_diagnostic_analysis.py, and step2b_player_diagnostics.py.
"""

from __future__ import annotations

import sqlite3

import numpy as np
import pandas as pd
import streamlit as st

from step0_data import DB_PATH, build_nba_side_tables
from step1_archetype_model import resolve_roster
from step2_diagnostic_analysis import (
    DATA_DIR,
    compare_to_benchmark,
    compute_all_player_exposures,
    compute_style_pool,
    find_comparables,
    find_similar_players_by_vector,
    league_benchmark_exposure,
    load_archetype_labels,
    load_player_oncourt_netrating,
    macro_archetype_exposure,
    mismatch_score,
    teammate_lift,
)
import step2b_player_diagnostics as diag2b



# Same palette as bl-website-peach.vercel.app (pulled from its own styles.css
# :root variables) - warm paper background, forest green primary, coral/gold
# accents, soft rounded corners - rather than inventing a second, unrelated
# visual identity for this project.
BL_PAPER = "#f7f2ea"
BL_WHITE = "#fffdf8"
BL_INK = "#20242a"
BL_MUTED = "#687078"
BL_LINE = "#ddd6ca"
BL_GREEN = "#004b2b"
BL_GREEN_SOFT = "#185b2d"
BL_CORAL = "#ee735f"
BL_GOLD = "#f6bd2e"

BASIS_DIR = DATA_DIR / "basis_2025_26"
SEASON = "2025-26"
# AI-ASSISTED (Claude Code, chat) - Prompt: "一开始加载默认选择的Michael Porter
# Jr." (default the Diagnostic Analysis page to Michael Porter Jr. on
# initial load). Not AI: the choice of player - the user's own call (he's
# also this project's recurring worked-example case study, per CLAUDE.md).
DEFAULT_DIAG_PLAYER_ID = 1629008  # Michael Porter Jr.

# NBA.com's own public static CDN - the standard headshot/logo paths used
# throughout the nba_api community for exactly this purpose. PNG, not SVG -
# Streamlit's sandboxed HTML rendering (used for the raw <img> tags below)
# silently failed to load SVGs in testing; PNG loads fine in the same
# context.
HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"


# --- shared data loaders ----------------------------------------------------

@st.cache_data
def load_static():
    recipes = pd.read_csv(BASIS_DIR / "recipes.csv")
    k = sum(1 for c in recipes.columns if c.startswith("arch_"))
    labels = load_archetype_labels(BASIS_DIR)
    oncourt = load_player_oncourt_netrating(season=SEASON)
    return recipes, k, labels, oncourt



@st.cache_data
def load_full_features(season="2025-26"):
    """The same joined NBA.com + Basketball-Reference feature table step1
    fits archetypes on (box-score rates, shot distance mix, play-type
    shares) - already built and name-matched, no new query needed for a
    detailed real-stats view per player."""
    df = build_nba_side_tables()
    return df[df["SEASON"] == season].reset_index(drop=True)


# AI-ASSISTED (Claude Code, chat) - Prompt: "每次初次加载的时候速度都会特别慢
# 怎么解决这个问题", then "如果我后续是要部署怎么解决这个问题" (first load is
# always especially slow - how to fix; then, a deployment-specific
# follow-up). Timed every cold-start loader directly rather than guessing:
# this one function alone took 36.8s of a 38.7s total critical path
# (everything else <0.5s combined). Two layers, cheapest-first: (1)
# persist="disk" so a long-running dev server survives restarts without
# repaying this - the return value is a plain, picklable tuple, safe to
# serialize, and Streamlit's disk cache lives under ~/.streamlit/cache,
# outside this repo. (2) Not deployment-robust alone - a fresh
# container/redeploy has no guarantee that cache survives - so also check
# for a precomputed file first (src/pipeline/precompute_exposure_cache.py),
# built the same "precompute offline, ship the file" way as
# data/basis_2025_26/recipes.csv itself. Falls back to the live ~400-query
# path (still persisted) if that file hasn't been generated yet, so a fresh
# clone isn't broken, just slower once - same guard-and-fall-back
# convention as the Intro page's missing-hull-basis message.
@st.cache_data(show_spinner="Computing league-wide archetype exposures (one-time per season)...", persist="disk")
def load_exposure_cache(recipes, k, season):
    """Every league player's own macro archetype exposure this season
    (compute_all_player_exposures) - the genuinely expensive, ~400-query
    part of similarity_weighted_benchmark's pool, and the ONE thing that
    does not depend on which player/hypothesis is being compared against.
    Cached here (not inside step2_diagnostic_analysis.py, which stays
    Streamlit-agnostic) so Mikel Brown Jr.'s interactive slot-query
    sliders - which change the target vector, not the season - never
    re-pay this cost on rerun; only the cheap per-target JS-distance step
    (compute_style_pool_by_vector) runs on every slider move.

    Checks for a precomputed data/basis_2025_26/exposure_cache_{season}.npz
    first (see src/pipeline/precompute_exposure_cache.py) - if present,
    this returns near-instantly regardless of whether the server process
    or its disk cache has ever run before, which is what a deployed copy
    needs. Falls back to the real ~400-query computation otherwise.
    """
    precomputed_path = BASIS_DIR / f"exposure_cache_{season}.npz"
    if precomputed_path.exists():
        data = np.load(precomputed_path)
        if int(data["k"]) == k:
            return list(data["pids"]), data["exposures"]
        # k doesn't match this precomputed file (e.g. re-fit at a different
        # K) - stale artifact, fall through to a real computation rather
        # than silently returning the wrong-shaped result.
    return compute_all_player_exposures(recipes, k, season=season)


@st.cache_data
def load_nets_roster(season="2025-26"):
    """This is a Nets portal - scope player selection to the actual current
    roster, not all 433 league players. Reuses step3's resolve_roster
    (name-matched against NETS_ROSTER, not a TEAM_ABBREVIATION filter on
    recipes.csv) rather than re-deriving this - recipes.csv attributes a
    traded player to whichever team his season-total row landed on (Julius
    Randle shows "MIN", not "BKN", from before this offseason's trade), so
    filtering by team here would silently drop him again, the exact bug
    step3 already found and fixed by matching on name instead.
    """
    return resolve_roster(season=season)


@st.cache_data
def load_player_bio(season="2025-26", db_path=None):
    """Basic bio/season-line info for the profile card - already pulled on
    Day 1 (player_bio table), no new data collection needed."""
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT PLAYER_ID, PLAYER_NAME, TEAM_ID, TEAM_ABBREVIATION, AGE, PLAYER_HEIGHT, "
            "PLAYER_HEIGHT_INCHES, PLAYER_WEIGHT, COLLEGE, COUNTRY, DRAFT_YEAR, DRAFT_ROUND, "
            "DRAFT_NUMBER, GP, PTS, REB, AST FROM player_bio WHERE SEASON = ?",
            conn, params=(season,),
        )


@st.cache_data
def load_player_base_stats(season="2025-26", db_path=None):
    """Season box-score line from player_base - the roster table's other
    named source (alongside player_bio). Only the box-score columns not
    already covered by load_player_bio (which has its own PTS/REB/AST from
    the bio endpoint) - skips the ~40 *_RANK columns (league-wide ranks,
    not meaningful for a 16-player subset) and team-level W/L/W_PCT."""
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        return pd.read_sql(
            "SELECT PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION, GP, MIN, PTS, REB, AST, STL, BLK, "
            "FG_PCT, FG3_PCT, FT_PCT, PLUS_MINUS FROM player_base WHERE SEASON = ?",
            conn, params=(season,),
        )


# --- Page 1: Diagnostic Analysis --------------------------------------------
#
# AI-ASSISTED (Claude Code, chat)
# Prompt: "Add a screening + player-card metric layer... This replaces the
# narrative case-study flow with a framework that applies uniformly to all
# [roster] players", followed by "Restructure render_diagnostic_analysis()
# ... into the three-layer framework backed by Prompt A's functions." Full
# spec: Layer 1 a quadrant scatter (mismatch_score x, net_diff y [later
# switched to plain on-court net rating - see build_quadrant_chart's own
# note], size=MIN,
# reference lines at the roster's own median, quadrants labeled in the
# caption not on the chart, click-to-select via on_select="rerun" carrying
# PLAYER_ID in customdata) plus a sortable screening_table below it; Layer 2
# an identical-structure player card (A: recipe + profile + stats; B: signed
# benchmark gap via mismatch_score; C: teammate_lift, greyed/hatched below
# an exposure-mass threshold) with a computed (never hardcoded) comparison
# of whether B and C agree, disagree, or C is flat (a player-level, not
# fit, issue); Layer 3 team_gap_summary's roster-wide short/over counts and
# pairwise conflict measure; and two always-visible limitation sentences.
# Used: every number in Layers 1-3 comes from Prompt A's own functions
# (mismatch_score/screening_table/teammate_lift/team_gap_summary) - no new
# computation in this file, per that prompt's own explicit constraint.
# Discovered live that mismatch_score's own similarity_weighted_benchmark
# call is the ~400-query-per-player bottleneck (Entry 013) and neither
# screening_table nor team_gap_summary shared that cost across roster
# players by default - added an exposure_cache passthrough to both
# (mirroring the existing load_exposure_cache pattern) so a 5-player
# roster page costs one ~35s pass, not five.
# Not AI: the quadrant framing and what each corner means, the B/C
# agreement logic (including the specific "C is flat -> player-level
# issue" rule), the exposure-mass threshold requirement for Block C, and
# the two limitation sentences - all specified directly by the user.

DIAG_MASS_THRESHOLD = 100  # shared minutes; matches CLAUDE.md's own MIN>=100 pair-significance convention


@st.cache_data(show_spinner="Computing per-player mismatch scores...")
def load_mismatch(player_id, recipes, k, season, exposure_cache):
    return mismatch_score(player_id, recipes, k, season=season, exposure_cache=exposure_cache)


@st.cache_data(show_spinner="Computing teammate-archetype lift...")
def load_teammate_lift(player_id, recipes, k, season):
    return teammate_lift(player_id, recipes, k, season=season)


# AI-ASSISTED (Claude Code, chat)
# Prompt: full six-section (A-F) per-player diagnostic report spec, given in
# complete detail (exact formulas, gates, UI structure, which existing
# functions must be reused verbatim) - see AI_USAGE.md for the full prompt
# text. This block is the thin Streamlit-side wiring for
# step2b_player_diagnostics.py: cached loaders (small hashable args; a
# leading underscore on `_fit`/`_centroids`-style params is Streamlit's own
# convention for skipping hash-by-value on an object it can't hash, e.g. the
# basis dict) plus one render_section_x per report section.
# Used: every new NUMBER comes from step2b_player_diagnostics.py's own
# functions - no computation duplicated here. C1/C2 reuse load_mismatch/
# load_teammate_lift/lift_bar_chart/render_bc_comparison unchanged (already
# defined above in this file); Section A's profile/chart/stats-expander
# block reuses render_profile_card/archetype_column_chart/
# render_player_stats_tab unchanged, per the spec's own "extends the
# existing card" instruction.
# Not AI: the six-section structure, every formula (purity/entropy,
# opportunity/outcome feature split, the bootstrap's 4-vs-25 feature split),
# and every honesty/gate requirement - all given directly in the task spec.

@st.cache_data(show_spinner="Loading 3-season archetype recipes (2023-24 to 2025-26)...")
def load_recipes_all_seasons():
    return diag2b.load_all_season_recipes()


@st.cache_data
def load_league_purity_entropy(recipes_all, k, season=SEASON):
    return diag2b.league_purity_entropy(recipes_all, k, season=season)


@st.cache_data(show_spinner="Computing miscast scores league-wide...")
def load_league_miscasting(recipes_all, _fit, k, season=SEASON):
    return diag2b.league_miscasting_scores(recipes_all, _fit, season=season)


@st.cache_data(show_spinner="Finding style neighbors...")
def load_peer_centroid(player_id, recipes_all, _fit, k, season=SEASON):
    return diag2b.similarity_weighted_feature_centroid(player_id, recipes_all, _fit, k, season=season)


@st.cache_data(show_spinner=False)
def load_role_drift_cached(player_id, recipes_all, k):
    return diag2b.role_drift(player_id, recipes_all, k)


# AI-ASSISTED (Claude Code, chat)
# Prompt: C1/C2 role-drift rework spec - "cache this league distribution
# per transition, compute once, reuse for every player" (the league-wide
# drift-magnitude distribution for a given (season_old, season_new) pair
# doesn't depend on which player is being viewed, so keying the cache by
# player_id the way load_role_drift_cached does would recompute the same
# ~450-player distribution on every single player click).
# Used: two new cache wrappers keyed ONLY on the transition
# (season_old, season_new) or on (player_id, transition) as appropriate -
# following this file's existing _fit-leading-underscore convention so
# Streamlit doesn't try to hash the fit dict.
# Not AI: the caching granularity requirement itself, specified directly.
@st.cache_data(show_spinner=False)
def load_transition_drift_distribution_cached(recipes_all, k, season_old, season_new):
    return diag2b.league_transition_drift_distribution(recipes_all, k, season_old, season_new)


@st.cache_data(show_spinner=False)
def load_drift_attribution_cached(player_id, recipes_all, _fit, k, season_old, season_new):
    return diag2b.drift_attribution(player_id, recipes_all, _fit, k, season_old, season_new)


@st.cache_data(show_spinner="Computing individual role sensitivity...")
def load_individual_role_sensitivity_cached(player_id, recipes, k, season=SEASON):
    return diag2b.individual_role_sensitivity_profile(player_id, recipes, k, season=season)


# AI-ASSISTED (Claude Code, chat)
# Prompt: "As for D3 and E, i think a big difference is that I can
# generate some ideas to coach or front office... So that I can generate
# two number for D3 and E D3: Elastic ability? E: Miscast? 我的意思这里在
# 结尾可以放两个卡片 这样会看起来更加合适" (D3 and E can each produce a headline
# number a coach or front office could act on - put two cards at the end).
# Used: for E's card, the league distribution already exists
# (load_league_miscasting) - no new computation needed. D3 has no
# equivalent yet (_elasticity_verdict only checks a fixed 6.0pp bar, not a
# real league distribution), so the two cards would have looked
# inconsistent side by side (one with a real percentile, one without).
# Added this league-wide sweep to match house style - every other
# verdict card on this page (purity, entropy, drift magnitude, mismatch,
# miscast) is {tercile word} + {real league percentile} + {raw number},
# and a bottom-line pair meant for a coach/front-office reader deserves
# the same rigor, not a shortcut. Reuses load_individual_role_sensitivity_
# cached + _elasticity_verdict per player (both already exist, unchanged)
# - this function only adds the LOOP + percentile distribution around
# them. Timed directly before committing to this design: ~0.125s/player
# measured on a real 8-player sample (individual_role_sensitivity_profile
# touches season-wide event/stint parquet files, notably heavier than
# this page's other league-wide sweeps, which are vectorized over an
# already-small recipes table) - ~54s extrapolated for the full league,
# a real but one-time, cacheable cost, not assumed cheap.
# Not AI: the idea itself (two coach-facing headline cards for D3/E) and
# the "why not just reuse what's already computed" framing - given
# directly; the "give D3 a real percentile too, for consistency" call was
# a judgment made here.
@st.cache_data(show_spinner="Computing role-elasticity spreads league-wide (about a minute, cached after)...")
def load_league_elasticity_spreads(recipes, k, season=SEASON):
    pids = recipes["PLAYER_ID"].astype(int).unique().tolist()
    spreads = []
    for pid in pids:
        profile = load_individual_role_sensitivity_cached(pid, recipes, k, season)
        if not profile.get("available"):
            continue
        elasticity = _elasticity_verdict(profile["profiles"])
        if elasticity.get("available"):
            spreads.append(elasticity["spread_pp"])
    return np.array(spreads), len(spreads)


@st.cache_data(show_spinner=False)
def load_miscasting_cached(player_id, _fit, season=SEASON):
    return diag2b.miscasting_score(player_id, season, _fit)


@st.cache_data(show_spinner=False)
def load_miscasting_grounding_cached(player_id, _fit, underused_idx, season=SEASON):
    return diag2b.miscasting_feature_grounding(player_id, season, _fit, underused_idx)


@st.cache_data(show_spinner=False)
def load_signature_cached(player_id, _centroid, _fit, feature_columns, season=SEASON):
    return diag2b.player_signature(player_id, _centroid, _fit, feature_columns, season=season)


# AI-ASSISTED (Claude Code, chat)
# Prompt: "Simplify D3's copy - display only, no computation changes. (1)
# Replace the dense method caption with one plain line... Move the current
# individual-measurement/method text into the existing 'how this works'
# expander style. (2) Add plain-language axis end labels... Keep the
# technical axis title only in hover/expander. (3) Add a computed one-word
# verdict chip next to the header, from the spread of well-supported
# deltas: 'elastic'/'rigid'... (4) Keep the auto sentence and the
# D1-comparison note, but reword the D1 note to one simpler line."
# Then: "Remove D3's 'Drill down into one context archetype' expander
# entirely (UI removed; functions kept as superseded-but-visible)... move
# its essential evidence into the main chart's hover..."
# Used: `_elasticity_verdict()` is a NEW single-source-of-truth function -
# both the header chip and the auto sentence read from its ONE computed
# result (word/spread_pp/best/worst/biggest), so they can never disagree
# with each other the way two independently-thresholded pieces of text
# could. "Spread" = the range between his biggest well-supported riser and
# faller (best - worst), tested against ELASTIC_SPREAD_THRESHOLD_PP - a
# NEW, disclosed threshold (this is a display-logic/wording decision, not
# a change to individual_role_sensitivity_profile()'s own measurement, so
# it stays within "display only, no computation changes"). The D1-
# comparison note is now the exact fixed sentence given, no longer
# computed by comparing D3's vs D1's top archetype - the `d1_diff`
# parameter this comparison used is fully removed (confirmed via grep it
# has no other caller) rather than left as a dead, unused argument.
# `individual_sensitivity_chart()`'s hover now carries the SAME evidence
# the removed drill-down showed for the actually-plotted usage metric
# (both raw usage-proxy values, both possession counts, assist-rate delta)
# - Python-rounded strings in customdata, this file's established hover
# convention. Verified directly against the current code (not assumed)
# that the drill-down's OWN metrics were already individual (usage_proxy/
# assist_rate/rim_share/three_share), not the old team-level ORtg/TOV%
# stats the task's parenthetical warned about - that inconsistency had
# already been fixed in the D3 Phase 2 rebuild, so no extra note was
# needed for it here. `sensitivity_profile_chart()`/`_role_sensitivity_verdict()`/
# `SENSITIVITY_SIGNIFICANT_DELTA` (the superseded team-ORtg versions) and
# the drill-down's own rendering code are otherwise untouched by this
# change - only the individual-metric versions and their D3 call site
# changed.
# Not AI: the plain-language wording (caption, axis labels, chip words,
# D1 note), the "spread" framing, and "remove the drill-down, fold its
# evidence into hover" - all specified directly in the two prompts.
ELASTIC_SPREAD_THRESHOLD_PP = 6.0  # percentage points (best usage_delta - worst usage_delta, well-supported archetypes only) - see DIAGNOSTICS_README.md


def _elasticity_verdict(profiles, threshold_pp=ELASTIC_SPREAD_THRESHOLD_PP):
    """Single source of truth for D3's elastic/rigid read - the header
    chip and the auto sentence both read from this ONE result so they can
    never disagree."""
    usable = [p for p in profiles if p["available"] and not p["thin"] and not np.isnan(p["usage_delta"])]
    if not usable:
        return {"available": False}
    best = max(usable, key=lambda p: p["usage_delta"])
    worst = min(usable, key=lambda p: p["usage_delta"])
    biggest = max(usable, key=lambda p: abs(p["usage_delta"]))
    spread_pp = (best["usage_delta"] - worst["usage_delta"]) * 100
    word = "elastic" if spread_pp >= threshold_pp else "rigid"
    return {"available": True, "word": word, "spread_pp": spread_pp,
           "best": best, "worst": worst, "biggest": biggest}


# AI-ASSISTED (Claude Code, chat)
# Prompt: "我觉得不需要添加这个 我觉得针对这些Features看起来会很乱 我觉得其实如果
# 可以解释一下这些features具体是什么意思 并且如果在这个features上他的数据优于
# similar style playes 可以标注出来 用不一样的颜色 并且avatar plot的这些维度都是
# 随机杂乱的扔在这里 我觉得可以按照一个特定顺序排列 把相同类型的放一起" (three
# asks: (1) plain-language explanations for the raw feature codes, (2) color
# the deviation bullets when his number is better than his style neighbors',
# (3) the radar's axes are thrown on in a random/mixed order - group same-
# type features together).
# Used: FEATURE_LABELS (raw code -> short plain-language name) and
# FEATURE_CATEGORY/CATEGORY_ORDER (5 groups: box score, physical, shot
# profile, shot creation, play type) - both reused for axis labels AND the
# deviation bullets, so the same wording appears in both places. Color
# logic reuses step2b's OWN existing opportunity/outcome split (built for
# Section E's miscasting test) rather than inventing a second "which
# features are good/bad" classification: only the 9 OUTCOME_FEATURES
# (TS%, AST%, etc.) get a green/coral "better/worse than his style
# neighbors" judgment (TOV% is the one feature where lower, not higher, is
# better - flagged explicitly rather than assumed); every deployment/style
# feature (playtype shares, shot-location mix, height) stays neutral ink -
# "he does more hand-offs than his style neighbors" isn't better or worse,
# just different, and coloring it would imply a judgment the data doesn't
# support.
# Not AI: all three requests - given directly.

FEATURE_LABELS = {
    "PTS_PER_100": "Scoring rate", "TS%": "Shooting efficiency", "USG%": "Usage rate",
    "AST%": "Assist rate", "TOV%": "Turnover rate", "STL%": "Steal rate", "BLK%": "Block rate",
    "TRB%": "Rebound rate", "FTr": "Free-throw rate", "BPM": "Overall impact (BPM)",
    "PLAYER_HEIGHT_INCHES": "Height",
    "Dist.": "Avg. shot distance",
    "% of FGA by Distance_0-3": "Shots at the rim",
    "% of FGA by Distance_3-10": "Shots, short mid-range",
    "% of FGA by Distance_10-16": "Shots, long mid-range",
    "% of FGA by Distance_16-3P": "Shots, deep mid-range",
    "% of FGA by Distance_3P": "Shots that are 3s",
    "Corner 3s_%3PA": "3s from the corner",
    "% of FG Ast'd_2P": "Assisted 2s", "% of FG Ast'd_3P": "Assisted 3s",
    "PLAYTYPE_CUT": "Cuts", "PLAYTYPE_HANDOFF": "Hand-offs", "PLAYTYPE_ISOLATION": "Isolation",
    "PLAYTYPE_OFFREBOUND": "Putbacks", "PLAYTYPE_OFFSCREEN": "Off-screen",
    "PLAYTYPE_PRBALLHANDLER": "P&R ball-handler", "PLAYTYPE_PRROLLMAN": "P&R roll man",
    "PLAYTYPE_POSTUP": "Post-ups", "PLAYTYPE_SPOTUP": "Spot-ups",
}
# NBA.com Synergy play-type codes are run-together compounds (PRROLLMAN,
# OFFREBOUND, ...) that generic "".title() mangles into
# "Prrollman"/"Offrebound" - an explicit label map, not a smarter
# string-casing rule, since these are just names to memorize (P&R Roll
# Man, Post Up, Spot Up, etc.), not a general pattern. Hoisted to module
# level (was local to render_player_stats_tab) so collect_report_data can
# reuse it for the PDF report's Play-Type Usage section too.
PLAYTYPE_LABELS = {
    "PLAYTYPE_CUT": "Cut", "PLAYTYPE_HANDOFF": "Handoff",
    "PLAYTYPE_ISOLATION": "Isolation", "PLAYTYPE_OFFREBOUND": "Off. Rebound",
    "PLAYTYPE_OFFSCREEN": "Off Screen", "PLAYTYPE_PRBALLHANDLER": "P&R Ball Handler",
    "PLAYTYPE_PRROLLMAN": "P&R Roll Man", "PLAYTYPE_POSTUP": "Post Up",
    "PLAYTYPE_SPOTUP": "Spot Up",
}
# AI-ASSISTED (Claude Code, chat) - Prompt: "table可以加一个column feature
# description就是描述一下这一列是做干什么的" (add a column describing what each
# feature/row actually measures). One line per feature, matching this
# project's own definitions (box-score rates, shot-distance buckets,
# Synergy-style play-type shares - see CLAUDE.md's feature list), not
# generic textbook wording where the two differ.
FEATURE_DESCRIPTIONS = {
    "PTS_PER_100": "Points scored per 100 team possessions while on court.",
    "TS%": "Points per shot attempt, accounting for free throws and 3-pointers.",
    "USG%": "Share of his team's offensive possessions he used (shots, free throws, turnovers) while on court.",
    "AST%": "Share of teammates' made field goals he assisted while on court.",
    "TOV%": "Turnovers per 100 plays he used.",
    "STL%": "Share of opponent possessions ending in a steal by him while on court.",
    "BLK%": "Share of opponent 2-point attempts he blocked while on court.",
    "TRB%": "Share of available rebounds he grabbed while on court.",
    "FTr": "Free-throw attempts per field-goal attempt.",
    "BPM": "Box Plus-Minus - estimated points per 100 possessions vs. league-average.",
    "PLAYER_HEIGHT_INCHES": "Height, in inches.",
    "Dist.": "Average distance of his shot attempts, in feet.",
    "% of FGA by Distance_0-3": "Share of his shot attempts taken within 3 feet of the rim.",
    "% of FGA by Distance_3-10": "Share of his shot attempts taken 3-10 feet out.",
    "% of FGA by Distance_10-16": "Share of his shot attempts taken 10-16 feet out.",
    "% of FGA by Distance_16-3P": "Share of his shot attempts taken from 16 feet to the 3-point line.",
    "% of FGA by Distance_3P": "Share of his shot attempts that are 3-pointers.",
    "Corner 3s_%3PA": "Share of his 3-point attempts taken from the corner.",
    "% of FG Ast'd_2P": "Share of his made 2-pointers that were assisted, not self-created.",
    "% of FG Ast'd_3P": "Share of his made 3-pointers that were assisted, not self-created.",
    "PLAYTYPE_CUT": "Share of his offensive possessions finished as a cutter.",
    "PLAYTYPE_HANDOFF": "Share of his offensive possessions finished off a hand-off.",
    "PLAYTYPE_ISOLATION": "Share of his offensive possessions finished in isolation.",
    "PLAYTYPE_OFFREBOUND": "Share of his offensive possessions finished off his own offensive rebound (putback).",
    "PLAYTYPE_OFFSCREEN": "Share of his offensive possessions finished coming off an off-ball screen.",
    "PLAYTYPE_PRBALLHANDLER": "Share of his offensive possessions finished as the ball-handler in a pick-and-roll.",
    "PLAYTYPE_PRROLLMAN": "Share of his offensive possessions finished as the roll man in a pick-and-roll.",
    "PLAYTYPE_POSTUP": "Share of his offensive possessions finished posting up.",
    "PLAYTYPE_SPOTUP": "Share of his offensive possessions finished on a spot-up catch-and-shoot or closeout attack.",
}
CATEGORY_ORDER = ["Box score", "Physical", "Shot profile", "Shot creation", "Play type"]
FEATURE_CATEGORY = {
    **{f: "Box score" for f in ["PTS_PER_100", "TS%", "USG%", "AST%", "TOV%", "STL%", "BLK%", "TRB%", "FTr", "BPM"]},
    "PLAYER_HEIGHT_INCHES": "Physical",
    **{f: "Shot profile" for f in ["Dist.", "% of FGA by Distance_0-3", "% of FGA by Distance_3-10",
                                    "% of FGA by Distance_10-16", "% of FGA by Distance_16-3P",
                                    "% of FGA by Distance_3P", "Corner 3s_%3PA"]},
    **{f: "Shot creation" for f in ["% of FG Ast'd_2P", "% of FG Ast'd_3P"]},
    **{f: "Play type" for f in ["PLAYTYPE_CUT", "PLAYTYPE_HANDOFF", "PLAYTYPE_ISOLATION",
                                 "PLAYTYPE_OFFREBOUND", "PLAYTYPE_OFFSCREEN", "PLAYTYPE_PRBALLHANDLER",
                                 "PLAYTYPE_PRROLLMAN", "PLAYTYPE_POSTUP", "PLAYTYPE_SPOTUP"]},
}


# AI-ASSISTED (Claude Code, chat)
# Prompt (this revision): "表格添加针对每一个column的排序功能 并且每一行结尾添加浅色的
# 线 去掉Team column 然后最左侧添加profile image列 把没有数据的三个球员先都默认放在
# 最下面 并且每个球员要保证row的长度是一样的" - six changes: per-column click-to-sort,
# a light row-separator line, drop Team, add a leftmost profile-image column, the 3
# no-data rookies default-sorted to the bottom (not interspersed alphabetically), and
# every row forced to the same height regardless of cell content.
# Used: switched from st.markdown(unsafe_allow_html=True) to
# st.components.v1.html() - Streamlit's markdown renderer does not execute
# <script> tags (a documented, deliberate restriction), and click-to-sort
# needs real JS; components.html() runs in its own sandboxed iframe that
# does. Sort is a small vanilla-JS row-reorder: each <td> carries a
# data-sort attribute (a plain sortable value, separate from its displayed
# text - e.g. Ht's data-sort is PLAYER_HEIGHT_INCHES while the cell shows
# "6-6"; Mixture's data-sort is its own top-1 archetype share, a defensible
# single sort key for a column that displays three numbers), clicking a
# <th> re-sorts <tbody>'s rows by that column's data-sort values (numeric
# compare if parseFloat succeeds on both sides, else string compare) and
# toggles ascending/descending on repeat clicks of the same header - no
# other custom sort framework needed for 19 rows. Default (pre-sort) row
# order is the 16 recipe-having players alphabetically, THEN the 3 no-data
# rookies alphabetically - built as two separately-sorted lists concatenated,
# not one sorted(roster_names) call (which would intersperse "Joshua
# Jefferson"/"Mikel Brown Jr."/"Tyler Bilodeau" alphabetically among the
# rest instead of pinning them to the bottom). Fixed row height: every <td>
# gets the same explicit `height` (ROW_HEIGHT_PX) with vertical-align:middle,
# since without it the 3-line Mixture cell (recipe players) would render
# taller than the one-line '-' cell (the 3 rookies), making rows visibly
# uneven - checked the Mixture cell's actual stacked-row height (3 bar+label
# rows plus gaps) before picking a value big enough for it, not just eyeballed.
# Profile image column has no headshot to show for the 3 rookies (confirmed
# zero PLAYER_ID resolves anywhere for them, same as every other column) -
# renders a flat empty grey circle placeholder there instead of broken <img>.
# Not AI: the six specific requirements - all given directly, not proposed.

ROW_HEIGHT_PX = 74


def _build_sortable_table_html(table_id, columns, rows_cells, row_height=ROW_HEIGHT_PX, row_styles=None,
                                font_size_px=14):
    """Generic sortable HTML table shell, shared by every sortable table in
    this app. `columns`: list of (label, sort_key_or_None - None means not
    sortable, e.g. a Photo column). `rows_cells`: list of rows, each a list
    of (html, sort_value_or_None) tuples aligned with `columns`. `table_id`
    must be unique per table on the page - namespaces both the DOM id and
    the JS sort function/state so two sortable tables never collide.
    `row_styles`: optional list of extra inline-CSS strings, one per row
    (parallel to `rows_cells`, "" for no extra styling) - e.g. a highlighted
    background for a currently-selected row. Left generic (a raw style
    string, not a hardcoded "highlight" flag) so any caller can reuse it for
    whatever per-row styling it needs, not just selection. `font_size_px`:
    base table font size (default 14, unchanged for existing callers) -
    individual cells can still override it locally via their own inline
    `<span style="font-size:...">`, same as before.
    Returns (table_html, iframe_height); caller still calls
    components.html() itself since the height budget/scrolling behavior can
    differ by context.
    """
    if not table_id.isidentifier():
        # table_id gets interpolated directly into JS variable/function names
        # (sortTable_{table_id}, sortState_{table_id}) - a hyphen (valid in an
        # HTML id, invalid in a JS identifier) silently breaks the sort
        # script with no Python-side error, only caught by actually running
        # the JS in a browser. Fail loudly here instead.
        raise ValueError(f"table_id={table_id!r} must be a valid JS identifier (no hyphens) - "
                         f"it's interpolated directly into this table's sort function/variable names")

    header_cells = []
    for i, (label, sort_key) in enumerate(columns):
        if sort_key is None:
            header_cells.append(f'<th style="text-align:left;padding:8px 10px;color:{BL_INK};">{label}</th>')
        else:
            header_cells.append(
                f'<th onclick="sortTable_{table_id}({i})" style="text-align:left;padding:8px 10px;'
                f'color:{BL_INK};cursor:pointer;user-select:none;" title="Click to sort">{label}</th>'
            )
    header_html = "".join(header_cells)

    row_styles = row_styles or [""] * len(rows_cells)
    body_rows = []
    for cells, extra_style in zip(rows_cells, row_styles):
        tds = []
        for html, sort_val in cells:
            attr = f' data-sort="{sort_val}"' if sort_val is not None else ""
            tds.append(f'<td{attr} style="padding:6px 10px;color:{BL_INK};white-space:nowrap;'
                       f'height:{row_height}px;box-sizing:border-box;vertical-align:middle;">{html}</td>')
        body_rows.append(f'<tr style="border-bottom:1px solid {BL_LINE};{extra_style}">{"".join(tds)}</tr>')

    table_html = f"""
    <div style="font-family:system-ui,-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;
                background:{BL_PAPER};overflow-x:auto;">
    <table id="{table_id}" style="border-collapse:collapse;width:100%;font-size:{font_size_px}px;">
      <thead><tr style="border-bottom:2px solid {BL_LINE};">{header_html}</tr></thead>
      <tbody>{"".join(body_rows)}</tbody>
    </table>
    </div>
    <script>
    let sortState_{table_id} = {{col: -1, asc: true}};
    function sortTable_{table_id}(colIdx) {{
        const table = document.getElementById('{table_id}');
        const tbody = table.tBodies[0];
        const rows = Array.from(tbody.rows);
        const asc = (sortState_{table_id}.col === colIdx) ? !sortState_{table_id}.asc : true;
        sortState_{table_id} = {{col: colIdx, asc: asc}};

        rows.sort((a, b) => {{
            const av = a.cells[colIdx].dataset.sort, bv = b.cells[colIdx].dataset.sort;
            const an = parseFloat(av), bn = parseFloat(bv);
            const cmp = (!isNaN(an) && !isNaN(bn)) ? (an - bn) : av.localeCompare(bv);
            return asc ? cmp : -cmp;
        }});
        rows.forEach(r => tbody.appendChild(r));

        const ths = table.tHead.rows[0].cells;
        for (let i = 0; i < ths.length; i++) {{
            ths[i].innerHTML = ths[i].innerHTML.replace(/ [\\u25b2\\u25bc]$/, '');
        }}
        ths[colIdx].innerHTML += asc ? ' \\u25b2' : ' \\u25bc';
    }}
    </script>
    """
    iframe_height = 56 + len(rows_cells) * row_height + 20
    return table_html, iframe_height
