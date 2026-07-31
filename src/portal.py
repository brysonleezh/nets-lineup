"""
Portal - interactive Streamlit app. Sidebar nav has an Intro orientation
page plus three pages matching the "Scouting Anyone" paper's three named
practical applications:

  0. Intro - plain-language "what is an archetype" explainer with a
     worked example and a link to the source paper. Its own top-level
     page (not a tab under Diagnostic Analysis) since it's orientation
     for the whole portal, not specific to any one player.
  1. Diagnostic Analysis - identifying player combination mismatches in
     existing rosters (the Michael Porter Jr. case study, generalized to
     any Nets player - step2_diagnostic_analysis.py Parts A-E).
  2. Roster Construction - balancing archetype distributions for tactical
     coherence (team-level composition vs. a 30-team league baseline -
     step2_diagnostic_analysis.py Part G).
  3. Scouting - finding stylistically similar players league-wide (cosine
     similarity in archetype-recipe space - step2_diagnostic_analysis.py
     Part G). Mikel Brown Jr. case study TBD - he has zero NBA data, so
     this page currently only demos the mechanism on players who already
     have a fitted recipe.

Every computation here calls step1/step2's already-validated functions -
this file only adds the UI layer, navigation, and chart rendering.

Run: streamlit run src/portal.py
"""

from __future__ import annotations

import io
import math
import sqlite3
from datetime import date

import numpy as np
import pandas as pd
import pdf2image
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components
from scipy.spatial import ConvexHull
from scipy.spatial.distance import jensenshannon
from sklearn.decomposition import PCA

from step0_data_collect_process import DB_PATH, build_nba_side_tables, _normalize_name
from step2_diagnostic_analysis import (
    DATA_DIR,
    compare_to_benchmark,
    compute_all_player_exposures,
    compute_archetype_gaps,
    compute_league_baseline,
    compute_style_pool,
    compute_style_pool_by_vector,
    compute_team_archetype_profile,
    find_comparables,
    find_similar_players_by_vector,
    league_benchmark_exposure,
    load_archetype_labels,
    load_player_oncourt_netrating,
    macro_archetype_exposure,
    mismatch_score,
    screening_table,
    similarity_weighted_benchmark,
    team_gap_summary,
    teammate_lift,
)
from step1_archetypes_model import (
    ARCHETYPE_TO_PAPER,
    load_basis as load_ada_basis,
    load_population as load_ada_population,
    project as ada_project,
)
from step3_nets_lineup_scoring import resolve_roster, NETS_ROSTER, NETS_ROSTER_NCAA_BRIDGE
import hull_callout_chart
import player_report
import report_svg_charts as rsc
import step2b_player_diagnostics as diag2b

st.set_page_config(page_title="Nets Archetype Portal", layout="wide")

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


def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"


st.markdown(
    f"""
    <style>
    .stApp {{ background-color: {BL_PAPER}; }}
    section[data-testid="stSidebar"] {{ background-color: {BL_WHITE}; border-right: 1px solid {BL_LINE}; }}
    /* Forced regardless of the browser's own light/dark toggle (a separate,
    per-browser Streamlit setting independent of this file's config.toml) -
    without !important here, that toggle can override these to a dark
    default on a returning visitor's browser. */
    header[data-testid="stHeader"] {{ background-color: {BL_PAPER} !important; }}
    div[data-baseweb="select"] > div {{
        background-color: {BL_WHITE} !important; color: {BL_INK} !important;
        border-color: {BL_LINE} !important;
    }}
    div[data-baseweb="select"] svg {{ fill: {BL_INK} !important; }}
    ul[data-baseweb="menu"] {{ background-color: {BL_WHITE} !important; }}
    ul[data-baseweb="menu"] li {{ color: {BL_INK} !important; }}
    div[data-baseweb="radio"] > div {{ border-color: {BL_GREEN} !important; }}
    div[data-baseweb="radio"] > div > div {{ background-color: {BL_GREEN} !important; }}
    /* Slider (K selector): BaseWeb bakes the filled/unfilled split into a
    per-value linear-gradient on this div (verified live via Playwright -
    rgb(0,75,43) filled, rgba(217,188,120,.25) unfilled - a warm gold/red-
    leaning tone, not the theme's own green), regenerated per slider value
    with no CSS custom property to hook into, so the dynamic two-tone
    gradient can't be recolored while keeping its fill position - flattened
    to one solid grey track instead (position is still fully readable from
    the thumb alone). Targeted structurally (nth child of the slider's own
    wrapper), not by emotion-cache class hash, since those are regenerated
    per value/render and not a stable selector. */
    div[data-baseweb="slider"] div[role="slider"] {{ background-color: {BL_MUTED} !important; }}
    div[data-baseweb="slider"] > div > div > div:last-child {{
        background: {BL_LINE} !important; background-image: none !important;
    }}
    h1, h2, h3, h4 {{ color: {BL_INK} !important; font-weight: 700 !important; letter-spacing: -0.01em; }}
    [data-testid="stMetricValue"] {{ color: {BL_INK}; }}
    .stCaption, [data-testid="stCaptionContainer"] {{ color: {BL_MUTED} !important; font-size: 18px; }}
    [data-testid="stMarkdownContainer"] p, [data-testid="stMarkdownContainer"] span,
    [data-testid="stMarkdownContainer"] li {{ color: {BL_INK}; font-size: 18px; }}
    h1 {{ font-size: 3rem !important; }}
    h2 {{ font-size: 2rem !important; }}
    h3 {{ font-size: 1.75rem !important; }}
    h4 {{ font-size: 1.4rem !important; }}
    div[data-testid="stTable"] th, div[data-testid="stTable"] td {{ font-size: 17px; padding: 12px 14px; }}
    div[data-testid="stSelectbox"] label, div[data-testid="stSelectbox"] div {{ font-size: 17px; }}
    [data-testid="stMetricValue"] {{ font-size: 2rem !important; }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {BL_LINE}; border-radius: 14px; overflow: hidden; }}
    /* st.dataframe renders via canvas (glide-data-grid) and reads the
    browser's own light/dark toggle directly in JS - CSS can't touch its
    cell colors at all, which is why it kept showing up black regardless
    of the rules above. Tables in this app use st.table instead (plain
    HTML), fully styleable here. */
    div[data-testid="stTable"] table {{ background-color: {BL_WHITE}; border-collapse: collapse; width: 100%; }}
    div[data-testid="stTable"] th {{
        background-color: {BL_PAPER} !important; color: {BL_INK} !important;
        border-bottom: 2px solid {BL_LINE} !important; font-weight: 600;
    }}
    div[data-testid="stTable"] td {{
        background-color: {BL_WHITE} !important; color: {BL_INK} !important;
        border-bottom: 1px solid {BL_LINE} !important;
    }}
    div[data-testid="stTable"] {{ border: 1px solid {BL_LINE}; border-radius: 14px; overflow: hidden; }}
    div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 14px; }}
    .stSelectbox label {{ color: {BL_INK} !important; font-weight: 600; }}
    button[data-baseweb="tab"] {{ font-weight: 600; color: {BL_MUTED}; font-size: 17px; padding: 10px 4px; }}
    button[data-baseweb="tab"] p {{ font-size: 17px !important; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: {BL_GREEN}; }}
    div[data-baseweb="tab-highlight"] {{ background-color: {BL_GREEN} !important; }}
    .stButton button {{ border-radius: 100px; }}
    section[data-testid="stSidebar"] .stButton button {{ border-radius: 8px; }}
    div[role="radiogroup"] label {{ padding: 6px 4px; }}
    hr {{ border-color: {BL_LINE}; }}
    body, .stApp {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    </style>
    """,
    unsafe_allow_html=True,
)

BASIS_DIR = DATA_DIR / "basis_2025_26"
SEASON = "2025-26"
NETS_TEAM_ID = 1610612751
# AI-ASSISTED (Claude Code, chat) - Prompt: "一开始加载默认选择的Michael Porter
# Jr." (default the Diagnostic Analysis page to Michael Porter Jr. on
# initial load). Not AI: the choice of player - the user's own call (he's
# also this project's recurring worked-example case study, per CLAUDE.md).
DEFAULT_DIAG_PLAYER_ID = 1629008  # Michael Porter Jr.

# AI-ASSISTED (Claude Code, chat) - Prompt: PDF diagnosis report spec,
# "black (team accent, Nets) #0B0B0C". Used as a fixed constant, not a
# 30-team color dict - this report's rosterLabel is unconditionally
# "Brooklyn Nets Roster" (this whole portal is Nets-scoped), so the
# report's own chrome (masthead rule, team tag background, bottom-line
# panel) is a fixed design choice, not a per-player fact - unlike the
# `team` field itself, which stays the player's REAL current
# TEAM_ABBREVIATION (may be non-BKN for Randle/Ellis/Wagner - see
# DIAGNOSTICS_README.md).
# Not AI: the exact hex value - given directly in the spec.
REPORT_TEAM_COLOR = "#0B0B0C"

# Intro page's convex-hull scatter: K=8 reuses the official basis_2025_26
# fit above (this project's one authoritative K=8, referenced everywhere
# else in the portal) - only K=4..7,9,10 are precomputed separately (see
# precompute_hull_bases.py; ADA is far too slow to refit live on a slider).
HULL_BASES_DIR = DATA_DIR / "hull_bases"
HULL_K_RANGE = (4, 5, 6, 7, 8, 9, 10)
HULL_DEFAULT_K = 8
# data/k_selection.csv reflects a different, now-superseded population
# (multi-season / higher MIN floor, from before this project's single-
# season direction change - see step1_archetypes_model.py's own Part A
# comment). data/k_selection_2025_26.csv is the sweep actually run against
# the same population (2025-26, MIN>=300) this hull chart uses - pairing
# the stale file here would mislead the "what does each extra corner buy"
# comparison, so this deliberately does NOT point at k_selection.csv.
K_SELECTION_PATH = DATA_DIR / "k_selection_2025_26.csv"
# No longer read anywhere since Diagnostic Analysis's tabs were replaced
# by a vertical narrative (Section 3 there uses similarity_weighted_
# benchmark directly, unconditionally) - left in place only because
# render_league_benchmark_tab (the code this flag used to gate) is still
# kept in the file as superseded-but-visible prior work.
SHOW_LEAGUE_BENCHMARK_TAB = False

# NBA.com's own public static CDN - the standard headshot/logo paths used
# throughout the nba_api community for exactly this purpose. PNG, not SVG -
# Streamlit's sandboxed HTML rendering (used for the raw <img> tags below)
# silently failed to load SVGs in testing; PNG loads fine in the same
# context.
HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"
TEAM_LOGO_URL = "https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png"


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


@st.cache_data(show_spinner="Computing league-wide archetype exposures (one-time per season)...")
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
    """
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


# --- shared UI helpers -------------------------------------------------------

def circular_avatar(url, size=110):
    """Round profile-picture crop via raw <img> + CSS (border-radius:50%,
    object-fit:cover) - st.image() alone can't crop a rectangular headshot
    into a circle, and a plain border-radius on a non-square image renders
    as an oval, not a circle."""
    # flex-shrink:0 + min-width/min-height guard against a narrow flex
    # column compressing the image's WIDTH only (its explicit HEIGHT
    # doesn't shrink to match, since nothing else constrains it) - that
    # mismatch is what was rendering this as an oval instead of a circle.
    st.markdown(
        f'<img src="{url}" style="width:{size}px;height:{size}px;'
        f'min-width:{size}px;min-height:{size}px;flex-shrink:0;border-radius:50%;'
        f'object-fit:cover;object-position:top center;border:3px solid {BL_WHITE};'
        f'box-shadow:0 0 0 1px {BL_LINE};display:block;margin:0 auto;">',
        unsafe_allow_html=True,
    )


def render_profile_card(player_name, player_id, bio_row, avatar_size=150):
    photo_col, info_col = st.columns([1, 3])
    with photo_col:
        circular_avatar(HEADSHOT_URL.format(player_id=player_id), size=avatar_size)
    with info_col:
        st.markdown(f"### {player_name}")
        draft = (
            f"{bio_row['DRAFT_YEAR']} Rd {bio_row['DRAFT_ROUND']} Pick {bio_row['DRAFT_NUMBER']}"
            if str(bio_row["DRAFT_YEAR"]) not in ("Undrafted", "nan", "None")
            else "Undrafted"
        )
        st.markdown(
            f'<span style="color:{BL_INK};">'
            f"<b>{bio_row['TEAM_ABBREVIATION']}</b> &nbsp;·&nbsp; "
            f"Age {bio_row['AGE']:.0f} &nbsp;·&nbsp; {bio_row['PLAYER_HEIGHT']}, {bio_row['PLAYER_WEIGHT']} lb &nbsp;·&nbsp; "
            f"{bio_row['COLLEGE'] or bio_row['COUNTRY']} &nbsp;·&nbsp; {draft}"
            f"</span>",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<span style="color:{BL_INK};">'
            f"{bio_row['GP']:.0f} GP &nbsp;·&nbsp; {bio_row['PTS']:.0f} PTS &nbsp;·&nbsp; "
            f"{bio_row['REB']:.0f} REB &nbsp;·&nbsp; {bio_row['AST']:.0f} AST ({SEASON} season totals)"
            f"</span>",
            unsafe_allow_html=True,
        )


def bubble_chart(categories, values, dominant_idx=None, n_cols=4, height=380):
    """Compact bubble grid: bubbles sit at fixed grid coordinates (n_cols
    per row), size is the only thing that varies with value (area-
    proportional via sizemode="area"), with a size floor so a 0% category
    still shows as a small visible ring rather than disappearing. Sorted
    descending so the dominant category reads first (top-left). No axes -
    this is a composition display, not a bivariate scatter.
    """
    order = np.argsort(values)[::-1]
    cats = [categories[i] for i in order]
    vals = [values[i] for i in order]
    is_dominant = [bool(dominant_idx is not None and i == dominant_idx) for i in order]

    n = len(cats)
    n_rows = int(np.ceil(n / n_cols))
    xs = [i % n_cols for i in range(n)]
    ys = [-(i // n_cols) for i in range(n)]

    max_val = max(vals) if len(vals) and max(vals) > 0 else 1.0
    sizeref = 2.0 * max_val / (76 ** 2)
    bubble_colors = [BL_GREEN if d else BL_LINE for d in is_dominant]
    pct_colors = [BL_GOLD if d else BL_INK for d in is_dominant]

    fig = go.Figure(go.Scatter(
        x=xs, y=ys, mode="markers",
        marker=dict(size=[max(v, max_val * 0.05) for v in vals], sizemode="area",
                    sizeref=sizeref, color=bubble_colors, line=dict(width=1.5, color=BL_WHITE)),
        hovertext=[f"{c}: {v:.1%}" for c, v in zip(cats, vals)],
        hoverinfo="text",
        showlegend=False,
    ))
    for x, y, v, tcolor in zip(xs, ys, vals, pct_colors):
        fig.add_annotation(x=x, y=y, text=f"{v:.0%}", showarrow=False,
                            font=dict(size=13, color=tcolor, family="Arial Black"))
    for x, y, c in zip(xs, ys, cats):
        fig.add_annotation(x=x, y=y - 0.44, text=c, showarrow=False,
                            font=dict(size=11, color=BL_MUTED), align="center")

    fig.update_layout(
        # Fixed width, same reasoning as archetype_pie_chart: scaleanchor
        # locks the bubbles' own aspect ratio, but the canvas around them
        # would otherwise stretch to fill a wide bordered card, leaving the
        # grid floating centered rather than flush left with everything
        # else on the page.
        width=620, height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        xaxis=dict(visible=False, range=[-0.65, n_cols - 0.35], fixedrange=True),
        yaxis=dict(visible=False, range=[-(n_rows - 1) - 0.65, 0.65], fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER,
    )
    return fig


# AI-ASSISTED (Claude Code, chat)
# Prompt (first revision): "显示其archtype的柱状图使用从灰色图像 颜色越深比例越高
# 并且我觉得这个显示效果是不是太单调了" (use grayscale, darker = higher share -
# though I wonder if that'll be too monotonous) - answered with a brand-green
# ramp instead of literal grayscale. Prompt (this revision): "用灰色 like
# brooklyn nets的背景 黑灰白 而不是绿色" (use grey - like the Brooklyn Nets'
# own black/grey/white identity - not green) - the user redirected back to
# grayscale, specifically anchored to the real Nets black/white palette
# rather than this app's own bl-website-peach-derived green accent (this
# file's colors were explicitly never a literal Nets palette - see the
# BL_* constants' own comment - so "black/grey/white" is a real, separate
# ask, not the same thing restated).
# Used: renamed to _sequential_ink_ramp, dark end = BL_INK (this file's
# near-black ink color, not BL_GREEN), light end = a neutral grey (not
# green-tinted) - same darker=higher-share logic and single-hue-sequential
# structure as before, just restyled to the requested black/grey/white
# ramp instead of reverting to a flat mid-grey (still addresses the
# earlier "too monotonous" concern via the ink-black anchor's contrast).
# Not AI: the specific direction to grey, anchored to the real Nets colors
# rather than this app's own green accent - the user's own call.
def _sequential_ink_ramp(n, light="#d8dadd", dark=BL_INK):
    """n colors from `light` to `dark`, index 0 = darkest (assign to the
    highest-value bar first, since callers already sort descending)."""
    lr, lg, lb = (int(light.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    dr, dg, db = (int(dark.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
    colors = []
    for i in range(n):
        t = 1.0 if n == 1 else 1.0 - i / (n - 1)  # i=0 -> t=1 (darkest/dark end)
        r = round(lr + (dr - lr) * t)
        g = round(lg + (dg - lg) * t)
        b = round(lb + (db - lb) * t)
        colors.append(f"#{r:02x}{g:02x}{b:02x}")
    return colors


# AI-ASSISTED (Claude Code, chat)
# Prompt: part of the A-F report restructure - "merge Section F into A...
# the recipe column chart in A gains the bootstrap's 5th-95th percentile
# ERROR BARS on each archetype bar... not a second chart, not a separate
# section." archetype_column_chart is shared by Section A and Section D
# (both of D's partial-recipe charts) and the paused PDF export - adding
# optional err_lo/err_hi params (default None, no behavior change for any
# existing caller) rather than a second chart function, so only Section A's
# post-bootstrap call needs to pass them.
# Not AI: the merge requirement itself - given directly in the task spec.
def archetype_column_chart(categories, values, height=380, width=None, min_display=0.02,
                           bargap=0.25, bar_width=0.42, err_lo=None, err_hi=None):
    """Vertical column chart of archetype composition - replaces the pie
    chart per explicit request. Bars ordered descending left-to-right
    (biggest first), colored by a single-hue black/grey sequential ramp
    (darkest = highest share - see _sequential_ink_ramp's own note).
    Archetypes below min_display (2%) are dropped entirely rather than
    shown as a token sliver - a player who has ~0% presence in an
    archetype isn't meaningfully "a little bit" of that type. width=None
    (the default) leaves layout.width unset so st.plotly_chart's own
    width="stretch" fully controls sizing instead of fighting a fixed
    figure width - pass an explicit width only for a genuinely
    fixed-size use. err_lo/err_hi: optional, full-length arrays index-
    aligned with `values` (e.g. a bootstrap's lo5/hi95) - when given,
    draws asymmetric error bars on each bar; omitted entirely otherwise.
    """
    order = [i for i in np.argsort(values)[::-1] if values[i] >= min_display]
    cats = [categories[i] for i in order]
    vals = [values[i] for i in order]

    colors = _sequential_ink_ramp(len(cats))

    bar_kwargs = dict(
        x=cats, y=vals,
        # Thinner bars (width=0.42 of the category slot, not Plotly's
        # default ~0.8) + rounded tops - the previous version used the
        # full slot width, which read as thick, blocky columns rather
        # than a refined chart.
        width=bar_width,
        marker=dict(color=colors, cornerradius=10),
        text=[f"{v:.0%}" for v in vals], textposition="outside",
        textfont=dict(color=BL_INK, size=16),
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    )
    y_top = max(vals) if vals else 1
    if err_lo is not None and err_hi is not None:
        err_plus = [float(err_hi[i] - values[i]) for i in order]
        err_minus = [float(values[i] - err_lo[i]) for i in order]
        bar_kwargs["error_y"] = dict(type="data", symmetric=False, array=err_plus,
                                     arrayminus=err_minus, color=BL_INK, thickness=1.5)
        y_top = max(y_top, max((err_hi[i] for i in order), default=y_top))

    fig = go.Figure(go.Bar(**bar_kwargs))
    # AI-ASSISTED (Claude Code, chat) - bugfix found while building the PDF
    # report export (not asked for, caught by inspecting the rendered PDF):
    # the rotated archetype-name tick labels rendered fine on-page but were
    # clipped at the bottom in the PDF's static chart images. Root cause:
    # a browser doesn't clip SVG content that overflows a nominal "height",
    # so the old b=10 margin never visibly mattered live - but kaleido's
    # static PNG export rasterizes to a hard pixel canvas with no overflow,
    # so the same tiny margin cut the labels off. automargin=True (Plotly
    # expands the margin to fit whatever it actually renders) plus a larger
    # margin floor fixes both contexts, not just the PDF - a real, shared
    # bug in a chart used on 3 different sections of the page.
    layout_kwargs = dict(
        height=height, margin=dict(l=10, r=10, t=30, b=90),
        showlegend=False,
        bargap=bargap,
        xaxis=dict(tickfont=dict(color=BL_INK, size=15), automargin=True),
        yaxis=dict(visible=False, range=[0, y_top * 1.25]),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
    )
    if width is not None:
        layout_kwargs["width"] = width
    fig.update_layout(**layout_kwargs)
    return fig


def diverging_bar(df, x_col, y_col, hover_extra_cols, height=320, x_title=""):
    """Shared diverging-bar-chart look (green=more/positive,
    coral=less/negative than reference) used by both the league-benchmark
    tab and the roster-construction gap chart - same visual language for
    "how does this compare to a reference distribution" everywhere in the
    portal."""
    d = df.sort_values(x_col).copy()
    # Rounded here (not left to px.bar's own auto-generated hovertemplate,
    # which shows the raw float at full precision) - see build_quadrant_chart's
    # note on the live-confirmed Plotly bug where an explicit fixed-decimal
    # format spec (e.g. "%{x:+.2f}") can still render a long unrounded digit
    # string for a noisy float, whereas a bare spec on an already-rounded
    # value displays cleanly.
    d[x_col] = d[x_col].round(2)
    colors = [BL_CORAL if v < 0 else BL_GREEN for v in d[x_col]]
    fig = px.bar(d, x=x_col, y=y_col, orientation="h", custom_data=hover_extra_cols)
    fig.update_traces(marker_color=colors, hovertemplate="%{y}: %{x:+}<extra></extra>")
    fig.add_vline(x=0, line_color=BL_LINE, line_width=1)
    # AI-ASSISTED (Claude Code, chat) - bugfix found while building the PDF
    # report export: same class of bug as archetype_column_chart's own fix
    # (see its comment) - a browser doesn't clip an axis label that
    # overflows the nominal margin, so l=10 never visibly mattered live,
    # but kaleido's static PNG export rasterizes to a hard canvas and cut
    # the y-axis archetype-name labels down to single letters. automargin
    # fixes this in both contexts, not just the PDF.
    fig.update_layout(
        xaxis_title=x_title, yaxis_title=None, height=height,
        margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        # font_color alone doesn't reliably reach axis tick labels - Plotly
        # gives ticks their own tickfont default that otherwise renders
        # low-contrast gray regardless of the figure-wide font color.
        xaxis=dict(tickfont=dict(color=BL_INK, size=14),
                   title_font=dict(color=BL_INK, size=15), automargin=True),
        yaxis=dict(tickfont=dict(color=BL_INK, size=14), automargin=True),
    )
    return fig


# AI-ASSISTED (Claude Code, chat)
# Prompt: "shot distance distribution使用court作为背景图画出来" (draw shot
# distance distribution using a court as the background image).
# Used: a real half-court diagram (standard NBA dimensions in feet, hoop at
# the origin) drawn from plain Scatter line segments, with the 5 distance
# buckets rendered as concentric, hoop-centered bands underneath it.
# Not AI: the redesign direction itself - given directly.
def _court_line_segments():
    """(x, y) vertices for every court line, NBA regulation dimensions in
    feet, hoop at the origin, +y toward half court. Segments are joined
    with None gaps into one Scatter trace - simplest way to draw many
    disconnected line pieces (straight + arcs) without juggling multiple
    traces or SVG path shapes."""
    BASELINE_Y, HALFCOURT_Y = -5.25, 41.75

    def arc(r, deg0, deg1, cx=0.0, cy=0.0, n=48):
        t = np.radians(np.linspace(deg0, deg1, n))
        return cx + r * np.cos(t), cy + r * np.sin(t)

    xs, ys = [], []

    def seg(sx, sy):
        xs.extend(list(sx) + [None])
        ys.extend(list(sy) + [None])

    # court boundary (baseline, sidelines, half-court line)
    seg([-25, 25], [BASELINE_Y, BASELINE_Y])
    seg([-25, -25], [BASELINE_Y, HALFCOURT_Y])
    seg([25, 25], [BASELINE_Y, HALFCOURT_Y])
    seg([-25, 25], [HALFCOURT_Y, HALFCOURT_Y])
    # backboard (4ft from baseline) + rim + restricted area (4ft radius)
    seg([-3, 3], [-1.25, -1.25])
    seg(*arc(0.75, 0, 360))
    seg(*arc(4, 0, 180))
    # lane / paint (16ft wide, 19ft from baseline) + free throw circle
    seg([-8, -8], [BASELINE_Y, 13.75])
    seg([8, 8], [BASELINE_Y, 13.75])
    seg([-8, 8], [13.75, 13.75])
    seg(*arc(6, 0, 360, cy=13.75))
    # three-point line: straight corners (x=+-22) up to where the 23.75ft
    # arc meets them, then the arc over the top
    corner_y = math.sqrt(23.75 ** 2 - 22 ** 2)
    seg([-22, -22], [BASELINE_Y, corner_y])
    seg([22, 22], [BASELINE_Y, corner_y])
    corner_deg = math.degrees(math.atan2(corner_y, 22))
    seg(*arc(23.75, corner_deg, 180 - corner_deg))
    # half-court circle
    seg(*arc(6, 0, 360, cy=HALFCOURT_Y))

    return xs, ys


def build_court_shot_chart(dist_pairs, height=460):
    """Zone shot chart: the 5 distance-bucket shares as concentric bands
    centered on the hoop, radially symmetric (not corner/wing-resolved),
    over a real half-court diagram. Concentric because the underlying
    feature (`% of FGA by Distance_X-Y`) is pure straight-line distance
    from the hoop with no corner/wing split anywhere in the source data -
    a concentric band is the honest shape for what this number actually
    measures, not a stand-in for the true zone chart. The real NBA
    three-point line is drawn on top for reference, so the stylized
    16-3P/3PT band boundary (fixed at a 23.75ft radius) can be visually
    compared to where the line actually sits (22ft in the corners).
    dist_pairs: exactly 5 (label, pct) tuples in canonical distance order
    (0-3, 3-10, 10-16, 16-3P, 3PT) - the full set, not a top-N subset,
    since a spatial chart with a zone left out would just show unlabeled
    blank court.
    """
    BASELINE_Y, HALFCOURT_Y = -5.25, 41.75
    band_edges = [0, 3, 10, 16, 23.75, 33]
    palette = [BL_GREEN, BL_GOLD, BL_CORAL, BL_GREEN_SOFT, BL_MUTED]
    theta = np.linspace(0, np.pi, 90)

    fig = go.Figure()
    for i, (label, pct) in enumerate(dist_pairs):
        r_in, r_out = band_edges[i], band_edges[i + 1]
        outer_x, outer_y = r_out * np.cos(theta), r_out * np.sin(theta)
        inner_x, inner_y = r_in * np.cos(theta[::-1]), r_in * np.sin(theta[::-1])
        xs = np.clip(np.concatenate([outer_x, inner_x]), -25, 25)
        ys = np.clip(np.concatenate([outer_y, inner_y]), BASELINE_Y, HALFCOURT_Y)
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", fill="toself", line=dict(width=0),
            fillcolor=_hex_to_rgba(palette[i % len(palette)], 0.55),
            hoveron="fills", hoverinfo="text", text=f"{label}: {pct:.0%}",
            showlegend=False,
        ))
        mid_r = min((r_in + r_out) / 2, HALFCOURT_Y - 3)
        fig.add_annotation(
            x=0, y=mid_r, text=f"<b>{pct:.0%}</b><br>{label}", showarrow=False,
            font=dict(color=BL_INK, size=13), align="center",
        )

    line_x, line_y = _court_line_segments()
    fig.add_trace(go.Scatter(x=line_x, y=line_y, mode="lines",
                             line=dict(color=BL_MUTED, width=1.5),
                             hoverinfo="skip", showlegend=False))

    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER,
        xaxis=dict(visible=False, range=[-27, 27], scaleanchor="y", scaleratio=1),
        yaxis=dict(visible=False, range=[BASELINE_Y - 2, HALFCOURT_Y + 2]),
    )
    return fig


def build_playtype_pie(pt_pairs_top3, other_pct, height=320):
    """Play-type usage as a pie: top-3 slices plus one 'Other' slice for
    the remainder, so the pie still sums to 100% (unlike a bar chart, a
    pie implies parts of a whole - dropping the rest without an 'Other'
    slice would misrepresent how much of his usage the top 3 actually
    cover)."""
    labels = [p[0] for p in pt_pairs_top3] + (["Other"] if other_pct > 0.001 else [])
    values = [p[1] for p in pt_pairs_top3] + ([other_pct] if other_pct > 0.001 else [])
    palette = [BL_GREEN, BL_GOLD, BL_CORAL, BL_LINE]

    fig = go.Figure(go.Pie(
        labels=labels, values=values, hole=0.45,
        marker=dict(colors=palette[:len(labels)], line=dict(color=BL_PAPER, width=2)),
        textinfo="label+percent", textfont=dict(color=BL_INK, size=13),
        hovertemplate="%{label}: %{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
    )
    return fig


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

# AI-ASSISTED (Claude Code, chat)
# Prompt: "could you hide this part for now, cuz I don't know how to
# explain it" - after a back-and-forth about what the stability bootstrap
# actually does and whether to extend it, the user opted to hide the
# feature entirely rather than ship something they can't explain in an
# interview setting.
# Used: one module-level flag gating the bootstrap button/error-bars/
# stability-caption UI in render_section_a - everything else in Section A
# (profile, chart, purity/entropy cards) is untouched. The underlying
# computation (bootstrap_recipe, load_bootstrap_cached,
# archetype_column_chart's err_lo/err_hi support) is NOT deleted - same
# "kept but not wired up" convention this file already uses for
# render_roster_construction, so it's a one-line flip to re-enable later
# rather than a rebuild.
# Not AI: the decision to hide it - the user's own call.
SHOW_STABILITY_BOOTSTRAP = False

# AI-ASSISTED (Claude Code, chat)
# Prompt: "隐藏掉C3 Development comps — players who played like him at his
# age" (hide C3 Development comps).
# Used: same "kept but not wired up" convention as SHOW_STABILITY_BOOTSTRAP
# just above - one module-level flag gating C3's markdown header, caption,
# and table in render_section_b; load_development_comps_cached and
# development_comps() themselves are untouched, so this is a one-line flip
# to re-enable, not a rebuild.
# Not AI: the decision to hide it - the user's own call.
SHOW_C3_DEVELOPMENT_COMPS = False

# AI-ASSISTED (Claude Code, chat) - Prompt: "Future work & Obstacles先隐藏起来"
# (hide Future Work & Obstacles for now). Used: same "kept but not wired
# up" convention as the two flags above, applied to a whole PAGE rather
# than a section within one - gates the page's inclusion in the sidebar
# nav list, so `page` can never actually take this value while hidden.
# render_future_work() itself is untouched - a one-line flip to re-enable.
# Not AI: the decision to hide it - the user's own call.
SHOW_FUTURE_WORK_PAGE = False

# AI-ASSISTED (Claude Code, chat)
# Prompt (this revision): "把Player Breakdown最后的Report preview and
# download单独放一个新的tab Report" (take the Report preview/download off
# the end of Player Breakdown and put it back in its own new "Report"
# tab) - a THIRD reversal of this page's placement history: originally
# its own page, folded into the bottom of Diagnostic Analysis ("since it
# is the result of diagnostic analysis"), now split back out again.
# Used: flipped back to True - `render_player_report_page()` and its own
# selectbox-based player picker were never touched by any of these
# reversals, so this is still a one-line flip either direction. The
# render_report_section() call at the bottom of render_diagnostic_analysis
# (added when this flag went False) is removed in this same revision -
# see that function's own note.
# Not AI: the decision to split it back into its own tab - the user's own
# call.
SHOW_PLAYER_REPORT_PAGE = True

# AI-ASSISTED (Claude Code, chat) - Prompt: "把Building Around Rookie给隐藏
#掉吧 因为我觉得来不及都弄完了" (hide Building Around Rookie - not enough
# time to finish it all). Used: same "kept but not wired up" convention as
# every other SHOW_* flag on this page - gates the page out of the sidebar
# nav list; render_rookie_slot_query_page() and everything under it
# (render_rookie_slot_query, the recommended-units feature, etc.) are
# completely untouched, a one-line flip to bring it back.
# Not AI: the decision to hide it, and the stated reason (time) - the
# user's own call.
SHOW_ROOKIE_SLOT_QUERY_PAGE = False

# AI-ASSISTED (Claude Code, chat) - Prompt: 'hide "see the usage-vs-
# production gap chart" please'. Used: same "kept but not wired up"
# convention as the flags above - gates the expander in render_section_d
# (Section E). usage_production_gap_chart(), _select_gap_chart_indices(),
# and the gap/omitted-archetypes computation are all untouched - a
# one-line flip re-enables the expander, not a rebuild.
# Not AI: the decision to hide it - the user's own call.
SHOW_GAP_CHART_EXPANDER = False

@st.cache_data(show_spinner="Running the screening framework across the roster...")
def load_screening_table(roster_ids, recipes, k, oncourt, season, exposure_cache):
    return screening_table(roster_ids, recipes, k, oncourt, season=season, exposure_cache=exposure_cache)


@st.cache_data(show_spinner="Computing per-player mismatch scores...")
def load_mismatch(player_id, recipes, k, season, exposure_cache):
    return mismatch_score(player_id, recipes, k, season=season, exposure_cache=exposure_cache)


@st.cache_data(show_spinner="Computing teammate-archetype lift...")
def load_teammate_lift(player_id, recipes, k, season):
    return teammate_lift(player_id, recipes, k, season=season)


@st.cache_data(show_spinner="Summarizing team-wide archetype gaps...")
def load_team_gap_summary(roster_ids, recipes, k, season, exposure_cache):
    return team_gap_summary(roster_ids, recipes, k, season=season, exposure_cache=exposure_cache)


# AI-ASSISTED (Claude Code, chat)
# Prompt: "Layer 1的二维散点图不要用点 每个点使用球员profile image 并且我觉得可以
# 在quadrants的背景图像上写上...就是四个维度代表的意思 用虚线背景" - replace the
# plain colored dots with each player's actual headshot, and print the four
# quadrant meanings directly on the chart's background (dashed) rather than
# only in a caption underneath.
# Used: same fig.add_layout_image approach already verified working in a
# live browser for the Intro page's hull scatter (Entry 028/029) - a
# same-position, fully transparent (opacity=0) marker trace is kept
# underneath purely so Streamlit's on_select="rerun" click-to-select still
# has something real to click (layout images aren't clickable on their
# own), with size still ~ minutes so higher-minute players keep a larger
# hit target. Photos go through hull_callout_chart.get_headshot_data_uri
# (local cache + base64 + initials fallback) rather than a second, separate
# image-loading implementation. The four quadrant meanings are dashed-
# bordered background rectangles (fig.add_shape, layer="below") each with a
# low-opacity label anchored in its own far corner - the shared inner edges
# of the four rects sit exactly where the old add_vline/add_hline median
# dividers were, so those were dropped as redundant rather than kept
# alongside a second dashed line at the same position. Axis range is now
# explicit (padded past the real data extent) rather than autoranged, since
# the quadrant rectangles and image sizing both need a known, fixed data
# range to size against.
# Not AI: the decision to use real photos instead of dots and to put the
# quadrant meanings on the chart itself - the user's own reaction to the
# plain-dot version.
def build_quadrant_chart(screening_df, labels, height=560, selected_pid=None):
    df = screening_df.dropna(subset=["net_rating"]).reset_index(drop=True)
    if len(df) == 0:
        return None
    top_label = df["top_archetype"].map(lambda i: labels.get(i, f"Archetype {i}"))
    med_x = round(float(screening_df["mismatch_score"].median()), 3)
    med_y = round(float(screening_df["net_rating"].median()), 1)

    # AI-ASSISTED (Claude Code, chat)
    # Prompt: "在Hover里显示的value 最多只保留3位小数" - hover values should never
    # show a long floating-point digit string. Live-verified this is a real
    # Plotly.js hovertemplate bug, not something a format spec alone fixes:
    # a minimal repro (a single point at y=-9.000000000000002, hovertemplate
    # "%{y:+.1f}") still rendered the full unrounded number in the hover box
    # - the ".1f" spec was silently ignored. net_diff = NRTG_ON - NRTG_OFF
    # (step2_diagnostic_analysis.py) is exactly the kind of float subtraction
    # that produces this noise even when both inputs look like clean
    # one-decimal numbers in the DB.
    # Used: round every value that reaches a Plotly trace/customdata here in
    # Python BEFORE it reaches the chart, rather than trusting the
    # hovertemplate's own format spec to do it client-side - since the
    # client-side rounding is exactly what was just demonstrated to be
    # unreliable. mismatch_score to 3dp, net_diff/MIN to sensible display
    # precision, concentration left as a fraction (still formatted via
    # Plotly's %{...:.0%}, a percentage format rather than fixed-point,
    # which wasn't implicated in the repro).
    # Not AI: the 3-decimal ceiling itself - given directly.
    df = df.copy()
    df["mismatch_score"] = df["mismatch_score"].round(3)
    df["net_rating"] = df["net_rating"].round(1)
    df["MIN"] = df["MIN"].round(0)

    x_min, x_max = float(df["mismatch_score"].min()), float(df["mismatch_score"].max())
    y_min, y_max = float(df["net_rating"].min()), float(df["net_rating"].max())
    x_pad = (x_max - x_min) * 0.18 or 0.02
    y_pad = (y_max - y_min) * 0.18 or 1.0
    x_range = [x_min - x_pad, x_max + x_pad]
    y_range = [y_min - y_pad, y_max + y_pad]
    img_w = (x_range[1] - x_range[0]) * 0.085
    img_h = (y_range[1] - y_range[0]) * 0.11

    min_vals = df["MIN"].values.astype(float)
    size_ref = 2.0 * min_vals.max() / (46 ** 2) if len(min_vals) else 1.0

    customdata = np.stack([
        df["PLAYER_ID"].values, df["PLAYER_NAME"].values,
        top_label.values, df["concentration"].values, df["MIN"].values,
    ], axis=-1)

    fig = go.Figure(go.Scatter(
        x=df["mismatch_score"], y=df["net_rating"], mode="markers",
        marker=dict(size=min_vals, sizemode="area", sizeref=size_ref, sizemin=28,
                   color=BL_GREEN, opacity=0.0, line=dict(width=0)),  # invisible - click hit-target only
        customdata=customdata,
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>%{customdata[2]}: %{customdata[3]:.0%}<br>"
            "mismatch=%{x} · net rating=%{y:+} · %{customdata[4]:.0f} min<extra></extra>"
        ),
    ))

    for _, row in df.iterrows():
        photo = hull_callout_chart.get_headshot_data_uri(int(row["PLAYER_ID"]), row["PLAYER_NAME"])
        fig.add_layout_image(dict(
            source=photo, xref="x", yref="y", x=row["mismatch_score"], y=row["net_rating"],
            xanchor="center", yanchor="middle", sizex=img_w, sizey=img_h,
            sizing="contain", layer="above",
        ))
        # AI-ASSISTED (Claude Code, chat)
        # Prompt (original): "点击选中哪个球员 球员会有选中效果" - the currently-
        # selected player's photo needs a visible selected state on the
        # chart itself, not just Layer 2 below changing. Prompt (this
        # revision): "screening scatter plot里选中球员不要用绿色圆圈 可以用
        # 别的方式" (don't use a green circle for the selected player, some
        # other way is fine) - green is BKN's own jersey color, so a green
        # ring around a player's headshot (which often shows him IN a dark
        # green Nets jersey/background) could visually blend into the photo
        # itself rather than standing out as a distinct "selected" marker.
        # Used: a filled-none circle shape, centered on the same point as
        # the photo and sized slightly larger (1.15x), drawn on its own
        # dedicated layer="above" pass so it rings the photo rather than
        # sitting behind it - a layout image has no native "selected" style
        # of its own (it's a plain raster, not a stylable marker), so a
        # separate ring shape is the only way to add one. Swapped the ring
        # color to BL_GOLD (this app's other primary accent, already used
        # for the cross-team tag elsewhere on this same chart) - gold has
        # no overlap with BKN's own black/green identity, so it can't blend
        # into a jersey the way green could.
        # Not AI: the selection-ring visual language, and flagging green
        # specifically as the problem - both given directly.
        if selected_pid is not None and int(row["PLAYER_ID"]) == int(selected_pid):
            ring_w, ring_h = img_w * 1.15, img_h * 1.15
            fig.add_shape(
                type="circle", xref="x", yref="y",
                x0=row["mismatch_score"] - ring_w / 2, x1=row["mismatch_score"] + ring_w / 2,
                y0=row["net_rating"] - ring_h / 2, y1=row["net_rating"] + ring_h / 2,
                line=dict(color=BL_GOLD, width=4), fillcolor="rgba(0,0,0,0)", layer="above",
            )
        # net_rating for a player with zero BKN floor time still comes from
        # player_advanced (league-wide, not team-scoped - screening_table's
        # team_abbr), so it's a real number either way - flagged here too,
        # not just in the table, so a reader doesn't assume it describes
        # his fit specifically with the Nets.
        team_abbr = row.get("team_abbr", "BKN")
        if team_abbr != "BKN":
            # yshift is a PIXEL offset (applied after the data-space anchor is
            # resolved) - img_h is in data units, so the actual half-photo
            # pixel height is img_h / (y-axis span) * plot height in px.
            photo_half_height_px = (img_h / (y_range[1] - y_range[0])) * height / 2
            fig.add_annotation(
                x=row["mismatch_score"], y=row["net_rating"], xref="x", yref="y",
                xanchor="center", yanchor="top", yshift=-(photo_half_height_px + 3),
                text=team_abbr, showarrow=False,
                font=dict(size=9, color=BL_GOLD), bgcolor="rgba(246,189,46,0.18)",
                borderpad=1,
            )

    # AI-ASSISTED (Claude Code, chat)
    # Prompt: "那就改用NRTG_ON 帮我把diagostic的页面重整" - after switching y
    # from net_diff (on-off) to plain NRTG_ON, the bottom-right quadrant's
    # old label ("ENVIRONMENT PROBLEM") leaned specifically on the on/off
    # comparison (team clearly worse WITHOUT him = a context/fit issue,
    # not him) - a claim NRTG_ON alone can't support as confidently, since
    # an absolute on-court number can't separate "his fit is atypical AND
    # dragging him down" from "he just isn't playing well right now" the
    # way an on/off swing at least partially could. Renamed to "FIT
    # PROBLEM" - still a fair top-line read of "atypical fit + subpar
    # on-court number" (parallel to "PLAYER-LEVEL PROBLEM"'s directness),
    # without asserting the causal on/off claim the old label implied.
    # The other three labels' logic didn't depend on the on/off comparison
    # and are unchanged.
    # Not AI: the decision to keep vs. rename each label - my own read of
    # which claims survive the metric switch, done directly.
    quadrant_specs = [
        (med_x, x_range[1], y_range[0], med_y, "right", "bottom", "FIT PROBLEM"),
        (x_range[0], med_x, y_range[0], med_y, "left", "bottom", "PLAYER-LEVEL PROBLEM"),
        (med_x, x_range[1], med_y, y_range[1], "right", "top", "PRODUCING DESPITE FIT"),
        (x_range[0], med_x, med_y, y_range[1], "left", "top", "WORKING AS EXPECTED"),
    ]
    for x0, x1, y0, y1, xanchor, yanchor, text in quadrant_specs:
        fig.add_shape(
            type="rect", xref="x", yref="y", x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color=BL_LINE, width=1.25, dash="dash"),
            fillcolor="rgba(0,0,0,0)", layer="below",
        )
        label_x = x1 if xanchor == "right" else x0
        label_y = y1 if yanchor == "top" else y0
        fig.add_annotation(
            x=label_x, y=label_y, xref="x", yref="y", xanchor=xanchor, yanchor=yanchor,
            xshift=-10 if xanchor == "right" else 10, yshift=-8 if yanchor == "top" else 8,
            text=text, showarrow=False, font=dict(size=14, color=BL_MUTED), opacity=0.65,
            align=xanchor,
        )

    # AI-ASSISTED (Claude Code, chat)
    # Prompt: "screen散点图中也标注虚线代表的意思" - label what the dashed lines
    # themselves mean (the roster's own per-axis median), not just what each
    # quadrant means - the quadrant corner labels above explain the four
    # regions but nothing on the chart said the dividers were medians.
    # Used: two short labels centered on each divider at a point away from
    # the corner labels (top-center for the vertical mismatch-median line,
    # right-center for the horizontal net-diff-median line) - those two
    # positions are the only parts of the frame the four corner labels don't
    # already occupy.
    # Not AI: asking for this - the user's own follow-up after seeing the
    # quadrant-label version.
    fig.add_annotation(
        x=med_x, y=y_range[1], xref="x", yref="y", xanchor="center", yanchor="top", yshift=-4,
        text=f"MEDIAN MISMATCH ({med_x:.3f})", showarrow=False,
        font=dict(size=11, color=BL_MUTED), opacity=0.75,
    )
    fig.add_annotation(
        x=x_range[1], y=med_y, xref="x", yref="y", xanchor="right", yanchor="middle", xshift=-4,
        text=f"MEDIAN NET RATING ({med_y:+.1f})", showarrow=False,
        font=dict(size=11, color=BL_MUTED), opacity=0.75, textangle=-90,
    )

    # AI-ASSISTED (Claude Code, chat)
    # Prompt: "1. Screening scatter plot背景中出现了一些黑色实线 不需要 给我去掉"
    # (some solid black lines showed up in the Screening scatter plot's
    # background - not needed, remove them).
    # Used: xaxis/yaxis never set showgrid/gridcolor explicitly, so Plotly's
    # own default gridlines (a solid mid-grey, dark enough against BL_PAPER
    # to read as "black lines") were drawing at every tick - redundant with,
    # and visually louder than, the quadrant's own dashed median dividers
    # and rectangle borders, which already carry the chart's structure.
    # Disabled the default grid entirely rather than re-styling it lighter,
    # since the quadrant dividers already do that job.
    # Not AI: flagging the lines as unwanted - the user's own reaction to
    # the rendered page.
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(title="mismatch score (JS distance: actual vs. expected environment)",
                  tickfont=dict(color=BL_INK, size=12), title_font=dict(color=BL_INK, size=13),
                  range=x_range, showgrid=False, zeroline=False),
        yaxis=dict(title="on-court net rating",
                  tickfont=dict(color=BL_INK, size=12), title_font=dict(color=BL_INK, size=13),
                  range=y_range, showgrid=False, zeroline=False),
    )
    return fig


# AI-ASSISTED (Claude Code, chat)
# Prompt: "缺失数据帮我填充 并且在table中我觉得可以标注他们的play数据是来自别的队
# not brooklyn nets" - fill in the missing net_on/off/diff for the 3 players
# with zero BKN floor time, and flag in the table that their numbers come
# from a different team. Superseded by the later NRTG_ON switch below (see
# screening_table's own note) - net_off/net_diff no longer exist as
# columns at all, but the "tag a cross-team number" mechanism this entry
# built lives on as the single `team_abbr` tag on the new "Net Rating"
# column.
# Not AI: the decision to pull real data rather than declare it permanently
# unavailable, and the specific ask to label it in the table - both given
# directly.
#
# AI-ASSISTED (Claude Code, chat)
# Prompt: "那就改用NRTG_ON 帮我把diagostic的页面重整" - Net On/Off/Diff
# collapse into a single Net Rating column matching the scatter's new y.
# Used: same cross-team tag mechanism as before, now applied once instead
# of three times per row (one column, not three), sourced from
# screening_table's `team_abbr`.
def render_screening_table(screening_df, labels, selected_pid=None):
    """Same sortable-HTML-table shell as the Intro page's roster table
    (_build_sortable_table_html) - click a header to sort, leftmost Photo
    column, fixed row height. Net Rating shows '-' only if player_advanced
    has no row at all for this player (shouldn't happen for the current
    roster - see screening_table's own docstring); a value sourced from a
    team other than BKN carries a small muted team tag right next to it.
    `selected_pid`'s row gets a highlighted background + left accent bar,
    matching the quadrant chart's own selection ring so the same player is
    visually tied together across both."""
    NA_SORT = -999.0
    columns = [("", None), ("Player", "name"), ("MIN", "min"), ("Top Archetype", "arch"),
               ("Concentration", "conc"), ("Mismatch", "mismatch"), ("Net Rating", "net_rating")]

    def _fmt_net(v, team):
        if pd.isna(v):
            return "—", NA_SORT
        tag = (f' <span style="font-size:10px;font-weight:600;color:{BL_GOLD};'
               f'background:rgba(246,189,46,0.18);border-radius:3px;padding:1px 4px;">{team}</span>'
               if team != "BKN" else "")
        return f"{v:+.1f}{tag}", float(v)

    # NOT clickable, despite an attempt (see AI_USAGE.md Entry for the full
    # story): components.html() renders in an iframe sandboxed with
    # "allow-forms allow-modals allow-popups allow-popups-to-escape-sandbox
    # allow-same-origin allow-scripts allow-downloads" - critically missing
    # "allow-top-navigation"/"allow-top-navigation-by-user-activation".
    # Without either flag, the browser hard-blocks ANY top-frame navigation
    # from inside that iframe (confirmed live: a direct
    # `window.top.location.search = ...` call throws
    # "SecurityError: ...does not have permission to navigate the target
    # frame"), regardless of technique (JS assignment, a real <a
    # target="_top"> click, etc.) - this is a Streamlit-controlled sandbox
    # attribute with no public API to add flags, not something fixable
    # from this file. Making the table itself clickable would require
    # replacing this custom sortable HTML table with a native Streamlit
    # widget (e.g. st.dataframe's own on_select row-selection), which is a
    # different, bigger change than what was asked - flagged to the user
    # rather than shipped half-working.
    rows_cells, row_styles = [], []
    for _, row in screening_df.iterrows():
        pid = int(row["PLAYER_ID"])
        photo_html = (
            f'<img src="{HEADSHOT_URL.format(player_id=pid)}" style="width:36px;height:36px;'
            f'border-radius:50%;object-fit:cover;object-position:top center;'
            f'border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};display:block;">'
        )
        arch_i = int(row["top_archetype"])
        arch_label = labels.get(arch_i, f"Archetype {arch_i}")
        team_abbr = row.get("team_abbr", "BKN")
        net_rating_html, net_rating_sort = _fmt_net(row["net_rating"], team_abbr)

        rows_cells.append([
            (photo_html, None),
            (f"<b>{row['PLAYER_NAME']}</b>", row["PLAYER_NAME"].lower()),
            (f"{row['MIN']:,.0f}", float(row["MIN"])),
            (arch_label, arch_label.lower()),
            (f"{row['concentration']:.0%}", float(row["concentration"])),
            (f"{row['mismatch_score']:.3f}", float(row["mismatch_score"])),
            (net_rating_html, net_rating_sort),
        ])
        row_styles.append(
            f"background:rgba(0,75,43,0.08);box-shadow:inset 3px 0 0 {BL_GREEN};"
            if selected_pid is not None and pid == int(selected_pid) else ""
        )

    table_html, iframe_height = _build_sortable_table_html(
        "screening_table", columns, rows_cells, row_styles=row_styles)
    components.html(table_html, height=iframe_height, scrolling=True)


# AI-ASSISTED (Claude Code, chat)
# Prompt: extended discussion ending in "那就改用NRTG_ON 帮我把diagostic的
# 页面重整" (switch to NRTG_ON, redo the Diagnostic Analysis page around
# it) - see build_quadrant_chart's own note for the full reasoning (real
# std comparison showing net_diff is empirically noisier than NRTG_ON
# alone, and NRTG_ON matching Section C's own baseline so B/C/the scatter
# axis are all finally consistent).
# Used: caption's y-axis description updated to plain on-court net rating;
# the old "zero BKN floor time" caption (which was specifically about
# net_on/off/diff having nothing to fall back to) is replaced with a
# shorter note about where a cross-team net_rating number is sourced from
# - player_advanced is league-wide, so this is now just a transparency
# note, not a missing-data workaround.
# Not AI: the metric switch - reached collaboratively, not assumed.
def render_layer1_screening(screening_df, labels, selected_pid=None):
    # AI-ASSISTED (Claude Code, chat) - Prompt: "字太多了 先简单一点 简单解释
    # Screening and Screening table就可以了" (too much text - simplify to just
    # a simple explanation of "Screening" and "Screening table"). Used:
    # condensed both captions - the detailed x/y/marker-size breakdown is
    # already printed directly on the chart itself (quadrant meaning +
    # axis titles), so this text no longer duplicates it. The traded-
    # player note (real content, not just explanatory prose) is kept but
    # shortened - same 3 names/teams, fewer words around them.
    # Not AI: "too much text, simplify" - the user's own reaction.
    st.markdown("### 1. Screening")
    st.caption("Every data-eligible roster player, one view - click a photo to select him below.")

    cross_team = screening_df[screening_df["team_abbr"] != "BKN"] if "team_abbr" in screening_df else screening_df.iloc[0:0]
    if len(cross_team):
        team_list = ", ".join(f"{n} ({t})" for n, t in zip(cross_team["PLAYER_NAME"], cross_team["team_abbr"]))
        st.caption(f"Traded in this offseason - shown with his real net rating from his actual team: {team_list}.")

    fig = build_quadrant_chart(screening_df, labels, height=560, selected_pid=selected_pid)
    if fig is not None:
        st.plotly_chart(fig, on_select="rerun", selection_mode="points",
                        key="screening_quadrant", width="stretch")

    med_x = screening_df["mismatch_score"].median()
    med_y = screening_df["net_rating"].median()
    trouble_quadrant = screening_df[
        (screening_df["mismatch_score"] > med_x) & (screening_df["net_rating"] < med_y)
    ]
    if len(trouble_quadrant) == 0:
        st.info(
            "No player currently lands in the high-mismatch / below-median-net-rating "
            "quadrant - that is a finding, not an empty state. This roster's problems are "
            "not primarily about lineup configuration."
        )

    st.markdown("**Screening table**")
    st.caption("Same players, sortable.")
    render_screening_table(screening_df, labels, selected_pid=selected_pid)


def lift_bar_chart(lift_res, arch_names, height=320, mass_threshold=DIAG_MASS_THRESHOLD):
    """Diverging bar of teammate_lift, extending diverging_bar's own
    green/coral-by-sign visual language with a third state: any archetype
    below mass_threshold shared minutes is greyed and hatched, not colored
    by sign - a thin estimate must not read the same as a well-supported
    one (explicit requirement, not a style choice)."""
    lift, mass, n_teammates = lift_res["lift"], lift_res["exposure_mass"], lift_res["n_teammates"]
    order = np.argsort(lift)
    cats = [arch_names[i] for i in order]
    # rounded in Python, not left to the hovertemplate's own format spec -
    # see build_quadrant_chart's note on the live-confirmed Plotly bug where
    # a "%{x:+.2f}"-style spec can still render an unrounded float when the
    # underlying value carries floating-point noise (e.g. from the
    # exposure-weighted-mean division teammate_lift itself uses).
    vals = [round(float(lift[i]), 2) for i in order]
    m = [round(float(mass[i]), 0) for i in order]
    n = [int(n_teammates[i]) for i in order]
    thin = [mm < mass_threshold for mm in m]

    colors = [BL_LINE if t else (BL_CORAL if v < 0 else BL_GREEN) for v, t in zip(vals, thin)]
    patterns = ["/" if t else "" for t in thin]

    fig = go.Figure(go.Bar(
        x=vals, y=cats, orientation="h",
        marker=dict(color=colors, pattern=dict(shape=patterns, size=6, fgcolor=BL_MUTED)),
        customdata=np.stack([m, n], axis=-1),
        hovertemplate=(
            "%{y}: lift=%{x:+}<br>exposure mass=%{customdata[0]:.0f} min · "
            "%{customdata[1]:.0f} teammates<extra></extra>"
        ),
    ))
    fig.add_vline(x=0, line_color=BL_LINE, line_width=1)
    # automargin fix - see diverging_bar's own note (same underlying bug,
    # same fix, this chart just wasn't touched by that edit since it builds
    # its own go.Bar rather than calling diverging_bar()).
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(title="net rating lift vs. his own on-court baseline",
                  tickfont=dict(color=BL_INK, size=13), title_font=dict(color=BL_INK, size=13),
                  automargin=True),
        yaxis=dict(tickfont=dict(color=BL_INK, size=13), automargin=True),
    )
    return fig


# AI-ASSISTED (Claude Code, chat)
# Prompt: part of building the report/PDF export feature - needed the B/C
# verdict's text available as plain data (not just rendered Streamlit
# widgets) so the same wording could appear in both the on-page report
# preview and the PDF.
# Used: pulled render_bc_comparison's own three-way logic (agree/disagree/
# flat/insufficient) out into compute_bc_verdict(), returning (kind,
# message) rather than calling st.success/st.warning/st.info directly;
# render_bc_comparison now just dispatches on `kind` to the right widget -
# same logic, same wording, computed once.
# Not AI: the three-way verdict logic itself - already established (not
# introduced by this refactor).
def compute_bc_verdict(diff, lift_res, arch_names, mass_threshold=DIAG_MASS_THRESHOLD):
    lift, mass = lift_res["lift"], lift_res["exposure_mass"]
    well_supported = mass >= mass_threshold

    if well_supported.sum() == 0:
        return "insufficient", (
            "No archetype has enough exposure mass (≥100 shared minutes) to "
            "compare B and C for this player."
        )

    supported_lift = np.where(well_supported, lift, np.nan)
    lift_range = float(np.nanmax(supported_lift) - np.nanmin(supported_lift))

    if lift_range < 2.0:
        return "flat", (
            f"**C is flat** across every well-supported archetype (range {lift_range:.1f} "
            f"net-rating points) - his results don't depend on teammate type. This reads "
            f"as a **player-level issue, not a fit issue**."
        )

    biggest_deficit_idx = int(np.argmin(diff))
    supported_idx = np.where(well_supported)[0]
    ranked_by_lift = supported_idx[np.argsort(-lift[supported_idx])]
    top_lift_set = set(ranked_by_lift[:3].tolist())

    if well_supported[biggest_deficit_idx] and biggest_deficit_idx in top_lift_set:
        return "agree", (
            f"**D1 and D2 agree**: his largest deficit ({arch_names[biggest_deficit_idx]}, "
            f"{diff[biggest_deficit_idx] * 100:+.1f}pp vs. typical) is also among his "
            f"highest well-supported lifts ({lift[biggest_deficit_idx]:+.2f}) - two "
            f"independent lines of evidence point the same way."
        )
    return "disagree", (
        f"**D1 and D2 disagree**: his largest deficit is {arch_names[biggest_deficit_idx]} "
        f"({diff[biggest_deficit_idx] * 100:+.1f}pp vs. typical), but that archetype "
        f"isn't among his best-supported lifts - either noise, or he's atypical for "
        f"his own archetype."
    )


# AI-ASSISTED (Claude Code, chat)
# Prompt: "这下面的文字内容不需要蓝色标注出来 就是设置一个下拉框 一点击标题
# 'Do D1 and D2 agree?' 下面的内容就扩展出来就好了" (the text below doesn't
# need to be highlighted in blue - just make it a dropdown: click the title
# 'Do D1 and D2 agree?' and the content expands below it).
# Used: replaced the container + color-coded widget (st.success/info/
# warning/caption by verdict kind) with a single st.expander whose label
# IS the title and whose body is plain text - one treatment for all 4
# verdict kinds (agree/disagree/flat/insufficient), not just dropping the
# blue "disagree" case specifically, since the request was a full-mechanism
# swap (dropdown), not a color swap.
# Not AI: the dropdown-instead-of-color-box direction - given directly.
def render_bc_comparison(diff, lift_res, arch_names, mass_threshold=DIAG_MASS_THRESHOLD):
    """Computed, never hardcoded: does B (benchmark gap) agree with C
    (empirical lift)? Three cases per the spec - agree / disagree / C is
    flat (ruling out fit as the explanation, a player-level issue)."""
    kind, message = compute_bc_verdict(diff, lift_res, arch_names, mass_threshold)
    with st.expander("Do D1 and D2 agree?"):
        st.markdown(message)


    # render_layer2_player_card was fully superseded by the six-section A-F
    # report (render_player_report + render_section_a/b/c/d/e/f below) -
    # deleted rather than kept unreferenced, since every piece of it now has
    # a direct, better replacement (Section A extends its A block with
    # purity/entropy; Section C's C1/C2 are its B/C blocks verbatim), unlike
    # render_roster_construction's own "kept but not wired up" case where no
    # replacement exists yet.


# AI-ASSISTED (Claude Code, chat)
# Prompt: "我觉得还不如把这部分先去掉 因为第三个tab我想做的是roster
# construction 所以说diagostic analysis就是针对个体球员的 roster
# construction是针对于球队的" (I think it's better to just drop this section -
# the third tab I want to be Roster Construction, so Diagnostic Analysis is
# about individual players and Roster Construction is about the team) -
# after I noted the pairwise conflict measure is a genuinely different
# signal from Roster Construction's own team-vs-league gap chart (player-
# vs-player incompatibility, not team-vs-league share) and recommended
# moving it rather than deleting it outright, which the user agreed with.
# Used: former render_layer3_team_summary's per-archetype short/over/gap
# table was dropped entirely (not migrated) - Roster Construction's own
# compute_archetype_gaps already answers "which archetypes is the team
# short on/over on," via a cleaner z-score-vs-league method, so keeping
# both would just be two competing answers to the same question. Only the
# pairwise conflict table (no z-score equivalent exists anywhere else)
# survives, renamed to `render_pairwise_conflict` and moved to be called
# from render_roster_construction instead of render_diagnostic_analysis.
# Not AI: the decision to cut Layer 3 and the reasoning for where the
# split between the two pages should fall - given directly.
def render_pairwise_conflict(recipes, k, roster_ids, exposure_cache):
    st.markdown("**Player-pair conflict**")
    st.caption(
        "Each roster player's mismatch diff vector (his actual archetype mixture minus what "
        "players of his style typically get, one value per archetype - see Diagnostic "
        "Analysis Section D) is compared pairwise via cosine similarity. Strongly negative = "
        "the two players' gaps point in opposite directions (one wants exactly what the other "
        "already has too much of) - a genuine incompatibility signal, distinct from the "
        "team-vs-league share gaps above."
    )
    _, conflicts = load_team_gap_summary(tuple(roster_ids), recipes, k, SEASON, exposure_cache)

    id_to_name = dict(zip(recipes["PLAYER_ID"].astype(int), recipes["PLAYER_NAME"]))
    with st.expander("Pairwise conflict measure (most negative = most incompatible gap vectors)"):
        conflict_columns = [("Player A", "a"), ("Player B", "b"), ("Cosine similarity", "cos")]
        conflict_rows_cells = []
        for _, r in conflicts.iterrows():
            a_name = id_to_name.get(int(r["player_a"]), str(r["player_a"]))
            b_name = id_to_name.get(int(r["player_b"]), str(r["player_b"]))
            cos = float(r["cosine_similarity"])
            conflict_rows_cells.append([
                (a_name, a_name.lower()),
                (b_name, b_name.lower()),
                (f"{cos:+.2f}", cos),
            ])
        conflict_html, conflict_height = _build_sortable_table_html(
            "team_conflict_table", conflict_columns, conflict_rows_cells, row_height=40)
        components.html(conflict_html, height=conflict_height, scrolling=True)


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
def load_bio_all_seasons():
    return diag2b.load_player_bio_all_seasons()


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


@st.cache_data(show_spinner=False)
def load_development_comps_cached(player_id, recipes_all, bio_all, k):
    return diag2b.development_comps(player_id, recipes_all, bio_all, k)


@st.cache_data(show_spinner="Computing role elasticity...")
def load_role_elasticity_cached(player_id, season=SEASON):
    return diag2b.role_elasticity(player_id, season=season)


@st.cache_data(show_spinner="Computing role sensitivity profile...")
def load_role_sensitivity_cached(player_id, recipes, k, season=SEASON):
    return diag2b.role_sensitivity_profile(player_id, recipes, k, season=season)


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


@st.cache_data(show_spinner="Running game-level stability bootstrap...")
def load_bootstrap_cached(player_id, season, _fit, recipes_all, k, B=500):
    return diag2b.bootstrap_recipe(player_id, season, _fit, recipes_all, k, B=B)


# AI-ASSISTED (Claude Code, chat)
# Prompt: "Puriy and Entropy用卡片或者其他形式来display 只放一个数字这里感觉不好看" -
# a bare st.metric + caption read as too plain for these two numbers;
# redesign as a card.
# Used: a small reusable HTML gauge-card - big value, a percentile bar
# (0-100%, a tick marking exactly where this player sits on the real
# league distribution) and the tercile label underneath, matching this
# file's own established convention of a raw-HTML component wherever
# st.metric's plain look doesn't carry enough information (the file's
# sortable tables and mixture cells use the same "reach for a small custom
# component before reaching for the plain built-in" convention). The
# percentile bar - not a value/max bar - is the meaningful visual here,
# since purity/entropy don't have a natural upper bound worth anchoring a
# gauge to, but "where does he sit vs. the league" does.
# Not AI: the "make it a card" direction itself - the user's own reaction
# to the plain st.metric version.
#
# AI-ASSISTED (Claude Code, chat), later revision:
# Prompt: "Purity and entropy不要带白边" (Purity/entropy shouldn't have a white
# edge/border) - the card's own BL_WHITE fill + BL_LINE border read as a
# visible white box+outline sitting inside the already-BL_PAPER-colored
# Section A container, once seen live.
# Used: dropped the card's own background and border entirely - it now
# sits flush on the parent container's BL_PAPER background (padding only,
# for spacing between the two side-by-side cards), rather than swapping to
# a different fill color guessing at what was wanted.
# Not AI: flagging the white edge as unwanted - the user's own reaction to
# the rendered page.
# AI-ASSISTED (Claude Code, chat) - bugfix found while implementing the
# Purity/Entropy copy rewrite below (not asked for as its own item, caught
# while re-reading this function to satisfy "bar = league percentile, tick
# = league median"): `tick_pct` was set to `pct * 100` - the EXACT SAME
# value as the fill bar's own width - so the tick always sat right at the
# fill's edge instead of marking the league median. A percentile-space
# median is always exactly 50 by definition (half the population is below
# it), so the fix is a constant, not a second computation.
# AI-ASSISTED (Claude Code, chat) - Prompt: "Role elasticity, their Player
# text mark at the wrong place of the bar plot" - a real screenshot showed
# "PLAYER" sitting at ~66% while the actual fill bar (52nd percentile)
# ended at ~52%. Root cause: the PREVIOUS version of this function
# deliberately pushed the label away from 50 whenever it landed within
# `min_gap`, to stop it colliding with "Median" - which is exactly what
# happened here (52 is only 2pp from 50, well inside the old 16pp gap),
# but the result reads as the label pointing at the WRONG spot on the
# bar, not just a cosmetically-adjusted one. Fixed by dropping the
# anti-collision push entirely - "Player" now always sits at his TRUE
# percentile, full stop. Only the edge-safety clamp remains (so the label
# can't render half off the card at 0%/100%), which was never the part
# that was wrong.
# Not AI: "wrong place" - the user's own screenshot and reaction.
def _clamp_gauge_label_x(pct_val, lo=6, hi=94):
    return max(lo, min(hi, pct_val * 100))


# AI-ASSISTED (Claude Code, chat) - Prompt (full sequence): "Please mark it
# mediam and their value for player and league median please" -> "我的意思
# 是标注在轴上 不是单独放在下面" (label it on the axis, not a row below) ->
# "I want their value mark on the bar plot, not just add text below the
# chart. I think format can be Median and Player are up at the bar, their
# vaule are below at the bar. And add axis for all those bar plots
# please."
# Used: `player_value`/`median_value` stay NEW, OPTIONAL parameters
# (default None -> byte-for-byte the same as before for any caller that
# doesn't pass them). When both given: "Median" (fixed at 50, matching the
# tick mark, which never moves) and "Player" (at the fill's own pct,
# color-matched to the bar) render ABOVE the bar; their real raw values
# render BELOW, at the SAME two x-positions so each number sits under its
# own label; a plain 0%/50%/100% percentile axis renders below that. The
# fill bar and the tick mark themselves are completely unchanged - only
# label/value/axis TEXT is added around them.
# A real collision risk found by reasoning through it, not assumed away
# (mirroring the same class of bug already found and fixed once this
# session for the C2 trajectory chart's direct end-labels): if the
# player's own percentile lands close to 50, "Median" and "Player" would
# print on top of each other. `_clamp_gauge_label_x()` keeps the PLAYER
# label/value at least `min_gap` away from 50 and away from the card's own
# edges - the tick and fill bar stay at the true position regardless; only
# the label/value TEXT shifts, the same "adjust the label, never the
# data" principle used there.
# Given this HTML/CSS still can't be visually re-verified in this
# environment (no browser tool; not a Plotly figure kaleido can export to
# check), spacing throughout was kept generous rather than tuned tight.
# Not AI: the on-bar-labels-plus-axis format and its exact framing ("Median
# and Player up top, values below, axis too") - given directly. The
# specific collision-avoidance mechanism was a judgment call made here.
def _stat_gauge_card(label, value_display, pct, sentence, color=BL_GREEN, player_value=None, median_value=None):
    # AI-ASSISTED (Claude Code, chat) - Prompt: pasted screenshot showing
    # the raw HTML source (literal `<div style="...">Median</div>` text)
    # rendering ON THE PAGE instead of being parsed as HTML. Root cause,
    # confirmed by reproducing it with markdown_it (the same markdown
    # parser dependency installed in this Streamlit environment, not
    # assumed): the PREVIOUS version built `overlay` as a triple-quoted,
    # indented multi-line f-string, then spliced it into the outer
    # template at an indented `{overlay}` position. That combination - an
    # indentation-inconsistent seam plus a blank-ish line where the two
    # strings joined - is exactly CommonMark's trigger for "this is an
    # indented code block," which prints the HTML as escaped literal text
    # instead of rendering it. Fixed by rebuilding this function's ENTIRE
    # output as single-line, concatenated f-strings with NO embedded
    # newlines anywhere - not just patching the one spot that visibly
    # broke, since the same risk could just as easily hide in a
    # differently-shaped card later. This trades source readability
    # (long lines) for a class of bug that's otherwise invisible from the
    # Python side alone and was only caught because the user could see the
    # real rendered page.
    # Not AI: reporting the visible breakage - the user's own screenshot.
    # AI-ASSISTED (Claude Code, chat) - Prompt: "First of all, no axis for
    # now" - dropped the 0%/50%/100% percentile axis row entirely (was
    # the last three divs below). bar_margin_bottom shrinks back down
    # accordingly, since there's now only one row of text (the two
    # values) below the bar instead of two.
    # Not AI: the decision to drop it - the user's own call.
    show_median = player_value is not None and median_value is not None
    bar_margin_bottom = "margin-bottom:10px;"
    bar_margin_top = ""
    overlay = ""
    if show_median:
        player_x = _clamp_gauge_label_x(pct)
        bar_margin_bottom = "margin-bottom:28px;"
        bar_margin_top = "margin-top:20px;"
        overlay = (
            f'<div style="position:absolute; left:50%; top:-17px; transform:translateX(-50%); font-size:10px; letter-spacing:0.03em; text-transform:uppercase; color:{BL_MUTED}; white-space:nowrap;">Median</div>'
            f'<div style="position:absolute; left:{player_x:.1f}%; top:-17px; transform:translateX(-50%); font-size:10px; letter-spacing:0.03em; text-transform:uppercase; color:{color}; font-weight:700; white-space:nowrap;">Player</div>'
            f'<div style="position:absolute; left:50%; top:14px; transform:translateX(-50%); font-size:11px; color:{BL_MUTED}; white-space:nowrap;">{median_value}</div>'
            f'<div style="position:absolute; left:{player_x:.1f}%; top:14px; transform:translateX(-50%); font-size:11px; color:{BL_INK}; font-weight:600; white-space:nowrap;">{player_value}</div>'
        )
    return (
        f'<div style="padding:6px 4px; height:100%; box-sizing:border-box;">'
        f'<div style="font-size:12px; letter-spacing:0.04em; text-transform:uppercase; color:{BL_MUTED}; margin-bottom:8px;">{label}</div>'
        f'<div style="font-size:34px; font-weight:700; color:{BL_INK}; line-height:1.1; margin-bottom:14px;">{value_display}</div>'
        f'<div style="position:relative; height:8px; background:{BL_LINE}; border-radius:4px; {bar_margin_bottom} {bar_margin_top}">'
        f'<div style="position:absolute; left:0; top:0; height:8px; width:{pct * 100:.1f}%; background:{color}; border-radius:4px;"></div>'
        f'<div title="League median" style="position:absolute; left:50%; top:-3px; width:2px; height:14px; background:{BL_INK}; transform:translateX(-1px); border-radius:1px;"></div>'
        f'{overlay}'
        f'</div>'
        f'<div style="font-size:13px; color:{BL_MUTED};">{sentence}</div>'
        f'</div>'
    )


# AI-ASSISTED (Claude Code, chat)
# Prompt: "Purity/Entropy/Stability copy... 1. shared caption explaining
# they're two views of the same trait, opposite directions. 2. Replace 'X%
# league percentile - unusually hybrid' with direction-free plain
# language... Keep the tercile-based verdict word computed as-is. 3. Make
# both progress bars encode the SAME thing (his league percentile)... 4.
# Rewrite the stability line to include a COMPUTED verdict... tight/
# moderate/wide thresholds."
# Used: purity and entropy move in OPPOSITE directions relative to their
# own percentile_rank (high purity percentile = more specialized; high
# entropy percentile = more hybrid), so "what fraction of the league is he
# more X than" needs a different inversion rule per metric per tercile -
# worked out algebraically (see the two comparison_pct branches below)
# rather than assumed, and cross-checked against a real example already in
# this file's own history (E.J. Liddell: pu_pct=13% -> "more hybrid than
# 87% of the league"; en_pct=94% -> "more hybrid than 94% of the league" -
# entropy needs no inversion, purity does, because tercile_label's own
# low/mid/high argument ORDER is opposite between the two calls below).
# Not AI: all four numbered requirements, the example wording, and the
# tight/moderate/wide thresholds (<=3pp / <=8pp / else) - given directly.
def _specialization_sentence(subject_phrase, tercile_word, comparison_pct):
    if tercile_word == "hybrid":
        return f"{subject_phrase} — more hybrid than {comparison_pct:.0%} of the league."
    if tercile_word == "specialized":
        return f"{subject_phrase} — more specialized than {comparison_pct:.0%} of the league."
    return f"{subject_phrase} — a typical blend, near the league median."


# AI-ASSISTED (Claude Code, chat)
# Prompt: full A-F report restructure spec - "merge Section F into A
# instead of keeping F as a separate last section. Concretely: the recipe
# column chart in A gains the bootstrap's 5th-95th percentile ERROR BARS
# on each archetype bar, and the entropy/purity metrics gain a small
# stability note next to them. Keep the existing lazy-compute pattern...
# Move F's honesty caption... into a caption directly under A's chart,
# shown only once the error bars exist." Explicit constraint: reordering +
# one merge only - no computation/formula changes anywhere.
# Used: bootstrap state is read ONCE at the top (same session_state key
# render_section_f used, `bootstrap_{player_id}_{SEASON}` - kept stable
# per the spec's own instruction not to rename keys) before the chart is
# built, so the chart call itself decides whether to pass err_lo/err_hi to
# archetype_column_chart. The button (pre-run) and the honesty caption
# (post-run) are mutually exclusive, matching "not a second chart, not a
# separate section." `load_bootstrap_cached`/`bootstrap_recipe`/every
# other computation function is called exactly as it was in the old
# render_section_f - only where the button and its result are drawn
# changed. st.rerun() was added after the button sets session_state (not
# present in the old F) so the chart above updates in the SAME click,
# matching how the OLD F's own code order (button before its chart)
# already gave same-run feedback - without it, reordering the button below
# the chart would introduce a one-click lag that didn't exist before.
# Not AI: the merge itself, exactly what moves where, and the stability-
# note requirement - all specified directly in the task spec.
def render_section_a(player_id, prow, k, labels, bio, recipes_all, fit):
    arch_cols = [f"arch_{i}" for i in range(k)]
    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    own_vals = prow[arch_cols].values.astype(float)
    top_i = int(np.argmax(own_vals))

    bootstrap_key = f"bootstrap_{player_id}_{SEASON}"
    bootstrap = st.session_state.get(bootstrap_key) if SHOW_STABILITY_BOOTSTRAP else None
    bootstrap_ok = bootstrap is not None and bootstrap.get("available")

    st.markdown("#### A. Who is he?")
    with st.container(border=True):
        profile_col, chart_col = st.columns([1, 1], vertical_alignment="center")
        with profile_col:
            bio_match = bio[bio["PLAYER_ID"] == player_id]
            if len(bio_match):
                render_profile_card(prow["PLAYER_NAME"], player_id, bio_match.iloc[0], avatar_size=120)
            else:
                st.caption(f"No bio record found for {prow['PLAYER_NAME']} in {SEASON} player_bio.")
        with chart_col:
            if bootstrap_ok:
                fig = archetype_column_chart(arch_names, own_vals, height=300,
                                            err_lo=bootstrap["lo5"], err_hi=bootstrap["hi95"])
            else:
                fig = archetype_column_chart(arch_names, own_vals, height=300)
            st.plotly_chart(fig, width="stretch")

        if bootstrap_ok:
            st.caption(
                f"Error bars: 5th-95th percentile band from a game-level bootstrap ({bootstrap['n_games']} "
                f"games resampled, B={bootstrap['B']}). Only 4 of the 29 basis features are actually "
                "re-derived per resample from his own per-game box counts (TS%, FTr, TOV%, and an "
                "approximate PTS_PER_100); the other 25 — including USG%, AST%, TRB%, STL%, BLK%, BPM, and "
                "every shot-location/play-type share — are held fixed at their real season value in every "
                "resample (see DIAGNOSTICS_README.md for why). **This means the band UNDERSTATES true "
                "uncertainty**, likely substantially. A low-minute player will show a wide band by "
                "construction — that's the point, not a bug."
            )
        elif SHOW_STABILITY_BOOTSTRAP:
            if st.button("Run stability bootstrap (~30s)", key=f"bootstrap_btn_{player_id}"):
                st.session_state[bootstrap_key] = load_bootstrap_cached(player_id, SEASON, fit, recipes_all, k, B=500)
                st.rerun()
            if bootstrap is not None and not bootstrap.get("available"):
                st.caption(f"Not available: {bootstrap.get('reason')}")

        purities, entropies = load_league_purity_entropy(recipes_all, k)
        purity, entropy = diag2b.purity_entropy(own_vals)
        pu_pct = diag2b.percentile_rank(purity, purities)
        en_pct = diag2b.percentile_rank(entropy, entropies)
        # word-only tercile verdicts (the computed bucket, not the wording) -
        # purity: low=hybrid/high=specialized. entropy: low=specialized/high=hybrid.
        pu_word = diag2b.tercile_label(purity, purities, "hybrid", "typical", "specialized")
        en_word = diag2b.tercile_label(entropy, entropies, "specialized", "typical", "hybrid")

        st.caption(
            "Same trait, opposite ends: low purity = high entropy = a hybrid player. "
            "Bar = league percentile; tick = league median."
        )

        # purity's own percentile-rank already means "more specialized than X%"
        # directly; "more hybrid than" is the complement. entropy's percentile-
        # rank already means "more hybrid than X%" directly (opposite inversion).
        pu_comparison = (1 - pu_pct) if pu_word == "hybrid" else pu_pct
        en_comparison = en_pct if en_word == "hybrid" else (1 - en_pct)
        pu_sentence = _specialization_sentence(
            f"His top archetype explains {purity:.0%} of his style", pu_word, pu_comparison)
        en_sentence = _specialization_sentence(
            f"His archetype mix has entropy {entropy:.2f}", en_word, en_comparison)

        # AI-ASSISTED (Claude Code, chat) - Prompt: "As for Purity — top-
        # archetype share and Entropy — normalized, 0-1, please also add
        # value and axis on the bar for player and league median." Used:
        # `purities`/`entropies` (the real league distributions already
        # loaded just above) give a real median via np.median - no new
        # data pull, reusing what this section already computed.
        # Not AI: the request itself - given directly.
        m1, m2 = st.columns(2)
        with m1:
            st.markdown(
                _stat_gauge_card(
                    "Purity — top-archetype share", f"{purity:.0%}", pu_pct, pu_sentence, color=BL_GREEN,
                    player_value=f"{purity:.0%}", median_value=f"{np.median(purities):.0%}",
                ),
                unsafe_allow_html=True,
            )
        with m2:
            st.markdown(
                _stat_gauge_card(
                    "Entropy — normalized, 0-1", f"{entropy:.2f}", en_pct, en_sentence, color=BL_GOLD,
                    player_value=f"{entropy:.2f}", median_value=f"{np.median(entropies):.2f}",
                ),
                unsafe_allow_html=True,
            )

        if bootstrap_ok:
            band_pp = (bootstrap["hi95"][top_i] - bootstrap["lo5"][top_i]) * 100
            # thresholds documented in DIAGNOSTICS_README.md - a real,
            # stated cutoff, not just an adjective picked by eye.
            if band_pp <= 3:
                verdict = f"a tight band: the {purity:.0%} purity is reliable, not small-sample noise"
            elif band_pp <= 8:
                verdict = f"a moderate band: the {purity:.0%} purity is reasonably reliable, with some resampling noise"
            else:
                verdict = f"a wide band: treat the {purity:.0%} purity as a rough estimate, not a precise one"
            st.caption(
                f"Across {bootstrap['B']} game-resamples his {arch_names[top_i]} share stays within "
                f"{bootstrap['lo5'][top_i]:.0%}-{bootstrap['hi95'][top_i]:.0%} (±{band_pp / 2:.0f}pp) — {verdict}."
            )
        elif SHOW_STABILITY_BOOTSTRAP:
            st.caption("Stability: not yet checked — run the bootstrap above to see a confidence band.")

    with st.expander("Detailed stats (box score rates, shot profile, play-type usage)"):
        render_player_stats_tab(prow)


# AI-ASSISTED (Claude Code, chat)
# Prompt: "B1. Role drift across seasons 这个图的X轴长度太长了 稍微窄一点" (B1's
# chart X-axis is too long, make it a bit narrower).
# Used: with only 2-3 season categories on the x-axis, width="stretch"
# spread them across the full container width, exaggerating the gaps
# between points. Wrapped the chart in a narrower column
# (st.columns([2, 1]), chart in the first) rather than changing the
# figure's own axis range/padding - constrains the rendered width without
# touching the data or the line/marker styling.
# Not AI: flagging the chart as too wide - the user's own reaction to the
# rendered page.
# AI-ASSISTED (Claude Code, chat)
# Prompt: "REPLACE the two per-transition diverging bar panels with ONE
# feature-trajectory line chart, deliberately mirroring C1's form so the
# two read as a pair" - x = the three seasons (same axis/order as C1), y =
# z-score, one line per feature in the union of both transitions' top
# movers, direct-labeled at its right end (no legend), hover shows the raw
# value per season.
# Used: same 8-color validated categorical palette C1's own chart uses,
# for the "read as a pair" requirement - direct end-labels via one
# add_annotation per line (positioned at that line's own last non-None
# point) rather than a shared legend, since up to 8 lines in a legend box
# would be exactly the clutter this whole redesign is trying to avoid.
# Hover reuses diag2b.format_raw_feature_value (now fixed - see its own
# note) rather than Plotly's raw float default, matching this project's
# house rounding convention.
# Not AI: the chart form/axes/direct-label requirement - given directly.
def feature_trajectory_chart(feature_names, trajectory, seasons, height=280):
    # AI-ASSISTED (Claude Code, chat) - Prompt: "单独给折线图弄一个图例子 因为这些
    # 指标太多导致文字都叠到一起了 分不清楚" (give the line chart its own separate
    # legend - too many metrics are making the direct end-labels pile up on
    # top of each other and become unreadable). Used: replaces the direct
    # end-labels (previously de-collided via _declutter_label_positions,
    # now deleted - it had no other caller) with a real Plotly legend,
    # matching C1's own chart in this exact section EXACTLY
    # (legend=dict(font=dict(color=BL_INK, size=15), orientation="v")) -
    # that config was already tuned twice on direct user feedback for C1
    # ("legend太小了" -> size 15, explicit vertical orientation), so reusing
    # it here keeps C1/C2 visually consistent rather than inventing a
    # second legend style in the same section. The union of two
    # transitions' top-5 movers can still exceed 8 lines (confirmed for
    # real - Julius Randle's union is 9) - a real legend scales to that
    # without collision the way stacked direct labels didn't; the existing
    # repeated-color-gets-a-dashed-line fallback for the 9th+ series is
    # kept, since Plotly's legend swatches render dash pattern too, so
    # two same-hue entries still read as visually distinct in the legend
    # itself, not just on the chart.
    # Not AI: the diagnosis (too many labels, illegible) and the requested
    # fix (a real legend) - given directly.
    palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
    fig = go.Figure()
    all_y = [v for fname in feature_names for v in trajectory[fname]["z"] if v is not None]

    for i, fname in enumerate(feature_names):
        label = FEATURE_LABELS.get(fname, fname)
        z_vals = trajectory[fname]["z"]
        raw_vals = trajectory[fname]["raw"]
        raw_strs = [diag2b.format_raw_feature_value(fname, v) if v is not None else "n/a" for v in raw_vals]
        color = palette[i % len(palette)]
        dash = "solid" if i < len(palette) else "dash"
        fig.add_trace(go.Scatter(
            x=seasons, y=z_vals, mode="lines+markers", name=label,
            line=dict(color=color, width=2, dash=dash), marker=dict(size=8),
            customdata=raw_strs,
            hovertemplate=f"<b>{label}</b><br>" + "%{x}: %{customdata} (%{y:+.2f} SD)<extra></extra>",
        ))

    y_top = max(all_y) if all_y else 1.0
    y_bottom = min(all_y) if all_y else 0.0
    pad = (y_top - y_bottom) * 0.08 or 0.5
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        yaxis=dict(title="feature value (SD vs. league)", tickfont=dict(color=BL_INK, size=12),
                   range=[y_bottom - pad, y_top + pad]),
        xaxis=dict(tickfont=dict(color=BL_INK, size=12)),
        legend=dict(font=dict(color=BL_INK, size=15), orientation="v"),
    )
    return fig


def render_section_b(player_id, recipes_all, bio_all, k, labels, fit, height=340):
    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    st.markdown("#### C. How has his role changed?")
    with st.container(border=True):
        st.markdown("**C1. Role drift across seasons**")
        rd = load_role_drift_cached(player_id, recipes_all, k)
        if rd["insufficient"]:
            st.caption(f"Rookie season — no drift history available (only {rd['available_seasons']} projected).")
        else:
            # AI-ASSISTED (Claude Code, chat)
            # Prompt: "C1. Role drift across seasons 颜色重复了" (the colors
            # repeat/duplicate) - BL_GREEN and BL_GREEN_SOFT (#004b2b vs
            # #185b2d) are both dark forest greens, too close to tell apart
            # in a thin line + small legend swatch, even though they're
            # technically different hex values.
            # Used: loaded the project's dataviz skill and swapped in its
            # validated reference 8-color categorical palette (blue, orange,
            # aqua, yellow, magenta, green, violet, red - references/palette.md)
            # rather than hand-picking more colors and eyeballing them -
            # re-ran the validator against this app's own light surface
            # (BL_PAPER #f7f2ea, not the skill's default) before adopting it:
            # all hard gates pass (lightness band, chroma floor, CVD
            # separation, normal-vision floor); 4 of 8 slots warn on
            # contrast-vs-surface, which the skill's own "relief rule"
            # requires mitigating with visible labels - already satisfied
            # here (a legend + hover tooltip on every line, not color alone).
            # Not AI: flagging the duplicate-looking colors - the user's own
            # reaction to the rendered chart.
            palette = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
            fig = go.Figure()
            minor_mass = np.zeros(len(rd["seasons"]))
            for i in range(k):
                series = rd["matrix"][:, i]
                if i in rd["major_idx"]:
                    fig.add_trace(go.Scatter(
                        x=rd["seasons"], y=series, mode="lines+markers", name=arch_names[i],
                        line=dict(color=palette[i % len(palette)], width=2), marker=dict(size=8),
                    ))
                else:
                    minor_mass = minor_mass + series
            if minor_mass.sum() > 1e-9:
                fig.add_trace(go.Scatter(
                    x=rd["seasons"], y=minor_mass, mode="lines+markers",
                    name="Minor (<15% every season, pooled)",
                    line=dict(color=BL_LINE, width=2, dash="dot"), marker=dict(size=6),
                ))
            # AI-ASSISTED (Claude Code, chat)
            # Prompt: "这表达的2024-2025 and 2025-2026是断开的 有什么更好的表达方式
            # 这个看起来还是很杂乱" (the 2024-25/2025-26 transitions read as
            # disconnected, and it still looks cluttered), then "change both
            # please" confirming the proposed fix.
            # Used: replaces the two side-by-side _stat_gauge_card boxes
            # (each an independent-looking widget) with (a) a short label
            # annotated directly ON the line chart at each segment's own
            # midpoint (paper-coordinate y, below the plot - the same
            # convention this app's other charts already use for axis-end
            # labels - so it can't collide with a data line regardless of
            # this specific player's archetype trajectory), and (b) one
            # flowing markdown line per transition below the chart, in the
            # exact same "**{s_old} → {s_new}**: {verdict} — bigger than
            # {pct:.0%}..." sentence format collect_report_data's PDF export
            # already used for this section - bringing the live page in
            # line with a format that already existed elsewhere in this
            # codebase, not inventing a new one. All underlying computation
            # (magnitude, league percentile, verdict tercile, changed-teams
            # tag) is unchanged from the card version - display only.
            # Not AI: flagging the disconnected/cluttered feel - the user's
            # own reaction to the rendered page.
            transitions = diag2b.season_transitions(rd["seasons"])
            transition_info = []
            for i, (s_old, s_new) in enumerate(transitions):
                magnitude = diag2b.transition_drift_magnitude(player_id, recipes_all, k, s_old, s_new)
                dist, n_league = load_transition_drift_distribution_cached(recipes_all, k, s_old, s_new)
                pct = diag2b.percentile_rank(magnitude, dist)
                verdict = diag2b.tercile_label(magnitude, dist, "stable role", "moderate shift", "major shift")
                changed = diag2b.transition_team_changed(recipes_all, player_id, s_old, s_new)
                tag = " · changed teams across this transition" if changed else ""
                transition_info.append(dict(s_old=s_old, s_new=s_new, magnitude=magnitude,
                                            pct=pct, verdict=verdict, n_league=n_league, tag=tag))
                fig.add_annotation(
                    # y=-0.16/margin.b=70 - the same paper-coordinate
                    # below-axis pairing D3's own chart already uses for
                    # its axis-end labels (an earlier, narrower margin got
                    # this annotation clipped off the bottom of the
                    # rendered image entirely - confirmed by rendering it
                    # and looking, not assumed correct from the numbers).
                    x=i + 0.5, xref="x", y=-0.16, yref="paper", yanchor="top", xanchor="center",
                    text=f"Δ{magnitude:.3f} · {verdict}",
                    showarrow=False, font=dict(color=BL_MUTED, size=12),
                )

            fig.update_layout(
                height=height, margin=dict(l=10, r=10, t=10, b=70),
                plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
                yaxis=dict(title="archetype share", tickformat=".0%", tickfont=dict(color=BL_INK, size=12)),
                xaxis=dict(tickfont=dict(color=BL_INK, size=12)),
                # legend size bumped 11->13->15 across two follow-up requests
                # ("legend太小了 把尺寸调大一点", then "vertical align and font
                # size to 15") - orientation explicitly set to vertical
                # rather than left to Plotly's default, so it stays vertical
                # regardless of container width.
                legend=dict(font=dict(color=BL_INK, size=15), orientation="v"),
            )
            chart_col, _ = st.columns([2, 1])
            with chart_col:
                st.plotly_chart(fig, width="stretch")

            for info in transition_info:
                st.markdown(
                    f"**{info['s_old']} → {info['s_new']}**: {info['verdict']} — bigger than "
                    f"{info['pct']:.0%} of the league's transitions (n={info['n_league']}){info['tag']}."
                )
            # st.caption(
            #     "Drift magnitude = Jensen-Shannon distance between his own archetype-mixture vectors in "
            #     "the two seasons (0 = identical mix, 1 = fully disjoint). A season-over-season change "
            #     "conflates real development with one season's sample noise and, if he changed teams, a "
            #     "new scheme — read as descriptive, not diagnostic."
            # )

        st.markdown("**C2. What's driving the drift**")
        if rd["insufficient"]:
            st.caption("Rookie season — no drift to attribute yet.")
        else:
            # AI-ASSISTED (Claude Code, chat)
            # Prompt (original C1+C2 spec): C2 = feature-level attribution -
            # dz across all 29 basis features between two seasons (both
            # z-scored with the basis's own FIXED mu/sd, never re-fit per
            # season), kept at |dz|>=0.3, top 5, each tied to THIS
            # transition's own top rising/falling archetype via whether the
            # feature moved toward that archetype's real archetypoid
            # z-profile or away from the falling one's - this computation
            # (load_drift_attribution_cached, diag2b.drift_attribution) is
            # UNCHANGED throughout every redesign below.
            #
            # AI-ASSISTED (Claude Code, chat)
            # Prompt: "in C2 I don't want to use gap chart... we want to
            # show their features change in 2023-2024, 2024-2025 right?"
            # (a diverging bar only shows the delta, not the actual
            # season-to-season values) -> "Rework C2... REPLACE the two
            # per-transition diverging bar panels with ONE
            # feature-trajectory line chart, deliberately mirroring C1's
            # form... Merge the two auto-attribution text blocks into one
            # block... plus the existing caveat once, not twice."
            # Used: `transitions` -> `attrs` (one drift_attribution result
            # per transition, unchanged computation) -> `union_names` (each
            # transition's own top-5 qualifying feature names, in order,
            # deduped - a feature flagged in only ONE transition still gets
            # its FULL season history plotted, via the new
            # diag2b.feature_trajectory(), not just the 2 seasons it was
            # originally flagged in). feature_trajectory_chart() (see its
            # own note) renders the union as one line per feature across
            # ALL of rd["seasons"] - 2 points for a 2-season player, 3 for
            # a 3-season player, matching C1's own x-axis exactly. Below the
            # chart, one merged block: one sentence per transition (the
            # same "Biggest single-feature mover" content as before, byte-
            # for-byte, just no longer split into side-by-side columns),
            # then the "one lens, not causation" caveat ONCE at the end
            # instead of once per column.
            # Not AI: the chart-form swap, the union-across-seasons
            # requirement, and the text-merge - all specified directly.
            transitions = diag2b.season_transitions(rd["seasons"])
            attrs = [load_drift_attribution_cached(player_id, recipes_all, fit, k, s_old, s_new)
                    for s_old, s_new in transitions]
            union_names = []
            for attr in attrs:
                if attr["available"]:
                    for f in attr["features"]:
                        if f["name"] not in union_names:
                            union_names.append(f["name"])

            if not union_names:
                st.caption("No clear driver: no feature moved by ≥0.3 SD in either transition.")
            else:
                traj = diag2b.feature_trajectory(player_id, union_names, fit, rd["seasons"])
                # Narrower column, matching C1's own chart just above -
                # leaves room for the vertical legend (see
                # feature_trajectory_chart's own note) instead of it
                # crowding a full-width chart.
                chart_col, _ = st.columns([2, 1])
                with chart_col:
                    st.plotly_chart(
                        feature_trajectory_chart(union_names, traj, rd["seasons"], height=280),
                        width="stretch",
                    )

                for (s_old, s_new), attr in zip(transitions, attrs):
                    if not attr["available"]:
                        st.markdown(f"**{s_old} → {s_new}**: no clear driver ({attr['reason']}).")
                        continue
                    top = attr["features"][0]
                    top_label = FEATURE_LABELS.get(top["name"], top["name"])
                    raw_str = (f"{diag2b.format_raw_feature_value(top['name'], top['raw_old'])} → "
                              f"{diag2b.format_raw_feature_value(top['name'], top['raw_new'])}")
                    direction = ""
                    if top["toward_R"] and top["away_F"]:
                        direction = (f", part of a move toward his {arch_names[attr['R']]} share "
                                   f"({attr['alpha_delta_R_pp']:+.1f}pp) and away from "
                                   f"{arch_names[attr['F']]} ({attr['alpha_delta_F_pp']:+.1f}pp)")
                    elif top["toward_R"]:
                        direction = (f", part of a move toward his rising {arch_names[attr['R']]} share "
                                   f"({attr['alpha_delta_R_pp']:+.1f}pp)")
                    elif top["away_F"]:
                        direction = (f", part of a move away from his falling {arch_names[attr['F']]} share "
                                   f"({attr['alpha_delta_F_pp']:+.1f}pp)")
                    st.markdown(
                        f"**{s_old} → {s_new}** biggest mover: **{top_label}** "
                        f"({raw_str}, {top['dz']:+.2f} SD){direction}."
                    )
                st.caption(
                    "Features shown moved by ≥0.3 SD between two seasons (top 5 per transition); a feature can "
                    "move without a matching archetype-share change, and vice versa — this is one lens, not "
                    "proof of causation."
                )

        if SHOW_C3_DEVELOPMENT_COMPS:
            st.markdown("**C3. Development comps — players who played like him at his age**")
            comps = load_development_comps_cached(player_id, recipes_all, bio_all, k)
            if comps["subject_age"] is None:
                st.caption(comps.get("note", "No age data available for this player."))
            elif len(comps["comps"]) == 0:
                st.caption(f"No player found at age {comps['subject_age']:.0f}±1 with a comparable recipe "
                          f"in our 3-season window.")
            else:
                st.caption(
                    f"Players at age {comps['subject_age']:.0f} (±1) whose recipe THEN most resembles his "
                    f"CURRENT recipe, and what they became the following season we have. With only 3 projected "
                    f"seasons total, the lookahead here is at most {comps['max_lookahead_seasons']} season(s) — "
                    f"not a full career arc."
                )
                # AI-ASSISTED (Claude Code, chat)
                # Prompt: "B2里的表格使用和上面的table一样的格式 不要看起来觉得不一致"
                # (B2's table should use the same format as the table above -
                # don't let it look inconsistent).
                # Used: added the leftmost Photo column + bold player name, the
                # exact same cell pattern render_screening_table already uses
                # (circular HEADSHOT_URL image, <b> name) - B2 was the one
                # sortable table in this file missing it, which is what read as
                # inconsistent next to the screening table directly above it.
                # Not AI: flagging the inconsistency - the user's own reaction.
                columns = [("", None), ("Comp", "name"), ("Matched season", "season"), ("Age", "age"),
                          ("Similarity", "sim"), ("Top archetype then", "then"), ("Next season", "next")]
                rows_cells = []
                for c in comps["comps"]:
                    sim = 1 - c["js_dist"]
                    then_label = arch_names[c["top_archetype_then"]]
                    if c["next"] is None:
                        next_label = "— (no further season in our window)"
                    else:
                        n = c["next"]
                        next_label = (f"{n['season']}: {arch_names[n['top_archetype']]} "
                                      f"(biggest change: {arch_names[n['biggest_change_idx']]} "
                                      f"{n['biggest_change_pp']:+.1f}pp)")
                    photo_html = (
                        f'<img src="{HEADSHOT_URL.format(player_id=c["player_id"])}" style="width:36px;height:36px;'
                        f'border-radius:50%;object-fit:cover;object-position:top center;'
                        f'border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};display:block;">'
                    )
                    rows_cells.append([
                        (photo_html, None),
                        (f"<b>{c['name']}</b>", c["name"].lower()),
                        (c["matched_season"], c["matched_season"]),
                        (f"{c['age']:.0f}", c["age"]),
                        (f"{sim:.0%}", sim),
                        (then_label, then_label.lower()),
                        (next_label, next_label.lower()),
                    ])
                table_html, h = _build_sortable_table_html("dev_comps_table", columns, rows_cells, row_height=44)
                components.html(table_html, height=h, scrolling=True)


def render_section_c(player_id, recipes, k, labels, exposure_cache):
    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    st.markdown("#### D. How does his environment shape him?")

    score, actual, expected, diff = load_mismatch(player_id, recipes, k, SEASON, exposure_cache)
    with st.container(border=True):
        st.markdown("**D1. What he gets vs. what his type usually gets**")
        st.caption(f"mismatch score (JS distance) = {score:.3f}")
        b_df = pd.DataFrame({"archetype": arch_names, "diff_pp": diff * 100})
        b_fig = diverging_bar(b_df, "diff_pp", "archetype", [],
                              x_title="pp vs. typical for his style (actual − expected)")
        st.plotly_chart(b_fig, width="stretch")

    lift_res = load_teammate_lift(player_id, recipes, k, SEASON)
    with st.container(border=True):
        st.markdown("**D2. What actually works**")
        st.caption("Exposure-weighted mean net-rating lift by teammate archetype - a weighted mean, not a "
                  "fitted regression. Hatched/grey bars have <100 exposure-minutes of support and should be "
                  "read as noise.")
        c_fig = lift_bar_chart(lift_res, arch_names)
        st.plotly_chart(c_fig, width="stretch")

    render_bc_comparison(diff, lift_res, arch_names)

    with st.container(border=True):
        _render_role_elasticity(player_id, recipes, k, labels, SEASON)


# AI-ASSISTED (Claude Code, chat)
# Prompt (first revision): "展示的eFG% 3PA rate TOV% 我觉得首先可以挑选更多指标
# 其次是这个显示效果不好 或者说很差 可以用别的方式么" (pick more metrics; the
# display is poor, use a different form) - answered with a custom paired-
# bar HTML component (6 metrics, with/without bars). Prompt (this revision,
# after seeing that component rendered): "这个C3的显示效果很差感觉" (C3's
# display still feels poor) - the custom widget still read as visually
# weak (long thin bars, low with/without contrast) and, more importantly,
# looked like nothing else on this page.
# Used: dropped the custom HTML bars entirely for a real Plotly diverging
# bar chart via the SAME diverging_bar() helper C1 and Section D already
# use - the one component on this page already proven to look good and
# consistent. Reframed the 6 metrics as one number each: % difference
# (with-teammate vs. without), which also solves the scale-mismatch
# problem the old component's per-row normalization was working around
# (ORtg ~100+ vs. percentages ~0-1 now share one meaningful axis - a
# relative % change is dimensionless). The raw with/without values are
# folded into each y-axis label (e.g. "ORtg (121.7 vs 112.8)") rather than
# a hover tooltip, since diverging_bar's own hovertemplate is shared
# across all 3 of its call sites and changing it risks the other two.
# Not AI: flagging the display as still poor - the user's own reaction to
# the rendered page.
# AI-ASSISTED (Claude Code, chat)
# Prompt: D3 upgrade spec - "UPGRADE to the D3 primary view - sensitivity
# PROFILE instead of a single context: one diverging bar per context
# archetype (usage-proxy delta, HIGH-minus-LOW teammate exposure), grey/
# hatched under a possession floor; a role-aware auto-verdict that frames
# 'no significant deltas' as a POSITIVE plug-and-play finding; demote the
# old single-teammate context view to a drill-down-only detail; GATE an
# optional rim-vs-3PT/assist elasticity on cached-PBP shot-distance
# availability."
# Used: a NEW chart (not diverging_bar - that shared helper has no notion
# of a per-bar "thin" flag, and editing its hovertemplate/coloring would
# touch its other 2 call sites) - thin bars render BL_MUTED + a diagonal
# pattern fill instead of green/coral, a redundant (color AND texture) low-
# confidence flag rather than color alone. SENSITIVITY_SIGNIFICANT_DELTA
# (3.0 points/100) is a disclosed, fixed threshold (documented in
# DIAGNOSTICS_README.md), the same convention as this file's other fixed,
# stated-not-derived cutoffs (C2's 0.3 SD, the old miscast gate's 5pp) -
# not derived from a league distribution, since that would need this same
# stint-level computation run for every league player, a much heavier cost
# than C1/C2's distributions (which only needed pre-existing recipe
# vectors). The gate check (shot_mix_gate_status()) is read directly, not
# assumed - see its own note for what's actually cached vs. joinable.
# Not AI: the profile/gate/verdict structure and the "positive framing for
# near-zero" requirement - specified directly in the task spec.
SENSITIVITY_SIGNIFICANT_DELTA = 3.0  # points per 100 possessions - see DIAGNOSTICS_README.md


def sensitivity_profile_chart(arch_names, profiles, height=380):
    rows = [{"archetype": arch_names[p["archetype_idx"]], "delta": p["ortg_delta"], "thin": p["thin"]}
           for p in profiles if p["available"]]
    df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    colors = [BL_MUTED if t else (BL_CORAL if d < 0 else BL_GREEN) for d, t in zip(df["delta"], df["thin"])]
    patterns = ["/" if t else "" for t in df["thin"]]
    delta_str = [f"{d:+.1f}" for d in df["delta"]]

    fig = go.Figure(go.Bar(
        x=df["delta"], y=df["archetype"], orientation="h",
        marker=dict(color=colors, pattern=dict(shape=patterns, fgcolor=BL_PAPER, size=6)),
        customdata=np.array(delta_str).reshape(-1, 1),
        hovertemplate="%{y}: %{customdata[0]} pts/100<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=BL_LINE, line_width=1)
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        xaxis_title="ORtg delta, high vs. low teammate exposure to this archetype (points/100 poss.)",
        yaxis_title=None,
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(tickfont=dict(color=BL_INK, size=13), title_font=dict(color=BL_INK, size=13), automargin=True),
        yaxis=dict(tickfont=dict(color=BL_INK, size=14), automargin=True),
    )
    return fig


def _role_sensitivity_verdict(arch_names, profiles, own_top_idx, min_poss):
    usable = [p for p in profiles if p["available"] and not p["thin"]]
    if not usable:
        return (f"Every context split has under {min_poss} possessions on one side this season — too "
                f"thin to say how his environment shapes his production.")
    best = max(usable, key=lambda p: p["ortg_delta"])
    worst = min(usable, key=lambda p: p["ortg_delta"])
    biggest = max(usable, key=lambda p: abs(p["ortg_delta"]))
    if abs(biggest["ortg_delta"]) < SENSITIVITY_SIGNIFICANT_DELTA:
        return (
            f"His team's offense barely moves regardless of who's beside him — the largest shift is "
            f"{arch_names[biggest['archetype_idx']]} at {biggest['ortg_delta']:+.1f} points/100, under "
            f"this report's {SENSITIVITY_SIGNIFICANT_DELTA:.1f}-point bar for a meaningful shift. Read as "
            f"a positive: a plug-and-play profile that should slot into most lineup constructions."
        )
    own_note = ""
    if own_top_idx in (best["archetype_idx"], worst["archetype_idx"]):
        own_note = f" Notably, his own archetype ({arch_names[own_top_idx]}) is among the biggest movers."
    return (
        f"His team's offense responds most to **{arch_names[best['archetype_idx']]}** exposure "
        f"({best['ortg_delta']:+.1f} points/100) and least to **{arch_names[worst['archetype_idx']]}** "
        f"({worst['ortg_delta']:+.1f}).{own_note}"
    )


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


def _elasticity_chip_html(elasticity):
    if not elasticity["available"]:
        return ""
    word = elasticity["word"]
    title = (f"Spread of well-supported usage deltas: {elasticity['spread_pp']:+.1f}pp "
            f"(bar: {ELASTIC_SPREAD_THRESHOLD_PP:.1f}pp)")
    return (
        f'<span style="display:inline-block; padding:3px 10px; border-radius:12px; background:{BL_GOLD}; '
        f'color:{BL_INK}; font-size:12px; font-weight:700; letter-spacing:0.03em; text-transform:uppercase; '
        f'margin-left:10px; vertical-align:middle;" title="{title}">{word}</span>'
    )


def individual_sensitivity_chart(arch_names, profiles, height=380):
    rows = [{"archetype": arch_names[p["archetype_idx"]], "delta": p["usage_delta"], "thin": p["thin"],
            "assist_delta": p["assist_delta"], "high_usage": p["high"]["usage_proxy"],
            "low_usage": p["low"]["usage_proxy"], "high_poss": p["high"]["possessions"],
            "low_poss": p["low"]["possessions"]}
           for p in profiles if p["available"] and not np.isnan(p["usage_delta"])]
    df = pd.DataFrame(rows).sort_values("delta").reset_index(drop=True)
    colors = [BL_MUTED if t else (BL_CORAL if d < 0 else BL_GREEN) for d, t in zip(df["delta"], df["thin"])]
    patterns = ["/" if t else "" for t in df["thin"]]
    usage_str = [f"{d * 100:+.1f}pp" for d in df["delta"]]
    # AI-ASSISTED (Claude Code, chat) - the drill-down's own evidence
    # (both raw usage-proxy values, both possession counts, assist-rate
    # delta), folded into one hover line per the exact format given:
    # "usage 29.8% -> 24.1% - assist rate +0.4pp - 1,180 vs 1,240 poss".
    evidence_str = []
    for _, r in df.iterrows():
        assist_txt = f"{r['assist_delta'] * 100:+.1f}pp" if pd.notna(r["assist_delta"]) else "n/a"
        evidence_str.append(
            f"usage {r['high_usage']:.1%} → {r['low_usage']:.1%} · assist rate {assist_txt} · "
            f"{r['high_poss']:,.0f} vs {r['low_poss']:,.0f} poss"
        )

    fig = go.Figure(go.Bar(
        x=df["delta"] * 100, y=df["archetype"], orientation="h",
        marker=dict(color=colors, pattern=dict(shape=patterns, fgcolor=BL_PAPER, size=6)),
        customdata=np.array(list(zip(usage_str, evidence_str))),
        hovertemplate="<b>%{y}</b><br>Usage-proxy delta: %{customdata[0]}<br>%{customdata[1]}<extra></extra>",
    ))
    fig.add_vline(x=0, line_color=BL_LINE, line_width=1)
    # AI-ASSISTED (Claude Code, chat) - "Add plain-language axis end
    # labels... Keep the technical axis title only in hover/expander" -
    # the same annotation pattern Section E's gap chart already
    # established (paper-coordinate text at both x-ends, no xaxis_title).
    # AI-ASSISTED (Claude Code, chat) - Prompt: "in D3, I think your cedes
    # the ball and takes over more of the offense is too small." Used:
    # size 12 -> 14 for these two end-labels specifically (Section E's gap
    # chart uses the identical size-12 pattern but wasn't flagged - left
    # unchanged, so the two charts are no longer pixel-identical in this
    # one respect, until/unless asked to match).
    # Not AI: flagging the small text - the user's own reaction.
    fig.add_annotation(
        text="← cedes the ball",
        xref="paper", yref="paper", x=0, y=-0.16, xanchor="left", yanchor="top",
        showarrow=False, font=dict(color=BL_MUTED, size=14),
    )
    fig.add_annotation(
        text="takes over more of the offense →",
        xref="paper", yref="paper", x=1, y=-0.16, xanchor="right", yanchor="top",
        showarrow=False, font=dict(color=BL_MUTED, size=14),
    )
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=70),
        yaxis_title=None,
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(tickfont=dict(color=BL_INK, size=13), automargin=True),
        yaxis=dict(tickfont=dict(color=BL_INK, size=14), automargin=True),
    )
    return fig


def _individual_sensitivity_verdict(arch_names, elasticity, own_top_idx, min_poss):
    if not elasticity["available"]:
        return (f"Every context split has under {min_poss} possessions on one side this season — too "
                f"thin to say how his own game shifts around different teammates.")
    best, worst, biggest = elasticity["best"], elasticity["worst"], elasticity["biggest"]
    if elasticity["word"] == "rigid":
        return (
            f"His own usage barely moves regardless of who's beside him — the largest shift is "
            f"{arch_names[biggest['archetype_idx']]} at {biggest['usage_delta'] * 100:+.1f}pp, and the "
            f"full spread across well-supported archetypes is under this report's "
            f"{ELASTIC_SPREAD_THRESHOLD_PP:.1f}pp bar for a meaningful role change. Read as a positive: "
            f"a plug-and-play profile that should slot into most lineup constructions."
        )
    own_note = ""
    if own_top_idx in (best["archetype_idx"], worst["archetype_idx"]):
        own_note = f" Notably, his own archetype ({arch_names[own_top_idx]}) is among the biggest movers."
    return (
        f"His own usage rises most alongside **{arch_names[best['archetype_idx']]}** exposure "
        f"({best['usage_delta'] * 100:+.1f}pp) and falls most alongside **{arch_names[worst['archetype_idx']]}** "
        f"({worst['usage_delta'] * 100:+.1f}pp).{own_note}"
    )


def _render_role_elasticity(player_id, recipes, k, labels, season):
    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    profile = load_individual_role_sensitivity_cached(player_id, recipes, k, season)
    elasticity = _elasticity_verdict(profile["profiles"]) if profile.get("available") else {"available": False}

    st.markdown(
        f"**D3. Does his game change with who's on the floor?**{_elasticity_chip_html(elasticity)}",
        unsafe_allow_html=True,
    )
    st.caption(
        "Each bar = one teammate type. Green: with them on the floor, he takes a bigger share of the offense. Red: he cedes the ball."
    )
    with st.expander("How this works"):
        st.caption(
            "A real INDIVIDUAL measurement, not a team-level proxy — for each archetype, his own "
            "on-floor stints are split into HIGH vs. LOW by his teammates' combined exposure to that "
            "archetype (a median split), then his own usage-proxy share [(FGA + TOV + 0.44×FTA) ÷ his "
            "team's same total while he's on the floor] and assist rate are compared between the two "
            "halves, from real play-by-play event attribution."
        )

    if not profile.get("available"):
        st.caption(f"Not available: {profile.get('reason')}")
        return

    arch_cols = [f"arch_{i}" for i in range(k)]
    own_row = recipes.loc[recipes["PLAYER_ID"] == player_id, arch_cols]
    own_top_idx = int(np.argmax(own_row.iloc[0].values.astype(float))) if len(own_row) else None

    st.plotly_chart(individual_sensitivity_chart(arch_names, profile["profiles"]), width="stretch")
    min_poss = next((p["min_poss"] for p in profile["profiles"] if p["available"]), 500)
    st.markdown(_individual_sensitivity_verdict(arch_names, elasticity, own_top_idx, min_poss))
    # st.caption(
    #     "Note: D1 asks what environment he GETS; this asks how he RESPONDS to it - the two can point "
    #     "at different archetypes without contradiction."
    # )

    # AI-ASSISTED (Claude Code, chat) - Prompt: "把这两个Bottom Line的两个卡片
    # Role Elasticity and Miscast Risk放在D3 and E的图下面" (put the two
    # Bottom Line cards under D3's and E's own charts instead of together
    # at the report's end). Used: relocated from the now-removed
    # render_bottom_line_cards, reusing `elasticity` already computed just
    # above in this SAME function - no recomputation. `profile["available"]`
    # (checked earlier, this line only runs when True) and
    # `elasticity["available"]` (checked again here) are two different
    # flags - a player's individual profile can compute fine with no
    # well-supported spread to call elastic/rigid (confirmed on E.J.
    # Liddell), so both branches are kept exactly as they were in the
    # removed function, not simplified away.
    # Not AI: the relocation itself - the user's own call.
    if not elasticity.get("available"):
        st.markdown(
            _simple_stat_card(
                "Role elasticity", "Not enough data",
                "Too few well-supported possession splits this season to call this.", color=BL_MUTED),
            unsafe_allow_html=True,
        )
    else:
        league_spreads, _ = load_league_elasticity_spreads(recipes, k, season)
        pct_d3 = diag2b.percentile_rank(elasticity["spread_pp"], league_spreads) if len(league_spreads) else 0.5
        best_name = arch_names[elasticity["best"]["archetype_idx"]]
        worst_name = arch_names[elasticity["worst"]["archetype_idx"]]
        median_spread = float(np.median(league_spreads)) if len(league_spreads) else None
        st.markdown(
            _stat_gauge_card(
                "Role elasticity", elasticity["word"].capitalize(), pct_d3,
                f"{pct_d3:.0%} league percentile for usage swing · {elasticity['spread_pp']:+.1f}pp "
                f"between {best_name} and {worst_name}",
                color=BL_GOLD,
                player_value=f"{elasticity['spread_pp']:.1f}pp swing",
                median_value=f"{median_spread:.1f}pp" if median_spread is not None else None,
            ),
            unsafe_allow_html=True,
        )


# AI-ASSISTED (Claude Code, chat)
# Prompt: Section E redesign spec item 2 - "replace the two side-by-side
# bar charts with ONE dumbbell chart. One row per archetype - ALL 8, no
# minimum-display cutoff... sorted by |productive - used| descending. Two
# dots per row... connected by a line. Hover: both values + the signed
# gap, rounded in Python."
# Used: raw go.Figure (not px.bar/archetype_column_chart, which both drop
# near-zero archetypes below a min_display floor - here a near-zero value
# in EITHER series is itself the signal a comparison chart needs to show,
# not noise to hide) - one grey connecting-line trace per row
# (showlegend=False, hoverinfo="skip", since the two marker traces already
# carry the real per-row hover) plus two shared marker traces ("Role-as-
# used"/"Role-as-productive") so the legend has exactly 2 entries, not 8.
# Colors are the dataviz skill's own validated categorical slots 1
# (blue)/2 (orange) - already re-validated against this app's BL_PAPER
# surface and already in use for Section C1's line chart, so no new
# palette to re-check. Hover values are Python-rounded into plain
# %{customdata} strings (no Plotly format spec) - this file's established
# hover convention (see diverging_bar's own note on the live-confirmed
# Plotly bug where a format-spec on a noisy float can still render an
# unrounded digit string; a pre-rounded string sidesteps it entirely).
# Not AI: the dumbbell form itself, "no cutoff", and the sort order -
# specified directly in the task spec.
def dumbbell_chart(names, used_vals, productive_vals, height=440):
    gaps = [p - u for u, p in zip(used_vals, productive_vals)]
    order = sorted(range(len(names)), key=lambda i: -abs(gaps[i]))
    sorted_names = [names[i] for i in order]
    sorted_used = [used_vals[i] for i in order]
    sorted_prod = [productive_vals[i] for i in order]

    USED_COLOR, PRODUCTIVE_COLOR = "#2a78d6", "#eb6834"  # categorical slots 1/2

    used_str = [f"{v:.0%}" for v in sorted_used]
    prod_str = [f"{v:.0%}" for v in sorted_prod]
    gap_str = [f"{(p - u) * 100:+.1f}pp" for u, p in zip(sorted_used, sorted_prod)]
    customdata = list(zip(used_str, prod_str, gap_str))
    hovertemplate = ("<b>%{y}</b><br>Role-as-used: %{customdata[0]}<br>Role-as-productive: "
                     "%{customdata[1]}<br>Gap: %{customdata[2]}<extra></extra>")

    fig = go.Figure()
    for name, u, p in zip(sorted_names, sorted_used, sorted_prod):
        fig.add_trace(go.Scatter(
            x=[u, p], y=[name, name], mode="lines",
            line=dict(color=BL_LINE, width=2), showlegend=False, hoverinfo="skip",
        ))
    fig.add_trace(go.Scatter(
        x=sorted_used, y=sorted_names, mode="markers", name="Role-as-used",
        marker=dict(color=USED_COLOR, size=13), customdata=customdata, hovertemplate=hovertemplate,
    ))
    fig.add_trace(go.Scatter(
        x=sorted_prod, y=sorted_names, mode="markers", name="Role-as-productive",
        marker=dict(color=PRODUCTIVE_COLOR, size=13), customdata=customdata, hovertemplate=hovertemplate,
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(title="archetype share", tickformat=".0%", tickfont=dict(color=BL_INK, size=12),
                  title_font=dict(color=BL_INK, size=13)),
        # sorted_names[0] (biggest |gap|) is listed first here but a plain
        # cartesian y-axis places an "array" categoryorder's first entry at
        # the BOTTOM (verified live against this app's own C2 chart, whose
        # ascending-sorted dataframe put its largest value at the TOP,
        # i.e. LAST-listed on top) - reversed here so the biggest gap
        # (the most important row) reads at the top, not buried at bottom.
        yaxis=dict(tickfont=dict(color=BL_INK, size=13), automargin=True,
                   categoryorder="array", categoryarray=list(reversed(sorted_names))),
        legend=dict(font=dict(color=BL_INK, size=13), orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


# AI-ASSISTED (Claude Code, chat)
# Prompt: "Replace Section E's dumbbell chart with a gap-first presentation
# - display only, no computation changes. (1) THREE VERDICT CARDS above
# the chart... (2) MAIN CHART: a single diverging bar per archetype...
# value = role-as-productive MINUS role-as-used, sorted by signed value
# descending... (3) Move the dumbbell chart into a collapsed expander...
# (4) fix the feature-evidence clause: restrict the auto-picked supporting
# features to an offense-relevant, intuitive whitelist... (5) Keep the
# causal caveat and summary-box wiring." Then an AMENDMENT: "do not draw
# all 8 rows. Show only archetypes where |productive - used| >= 3pp...
# CRITICAL: the filter is by GAP SIZE, never by the player's top
# archetypes... Edge case: if fewer than 2 rows clear the threshold, show
# the top-3 by |gap| anyway with a note."
# Used: `_simple_stat_card` is a NEW, smaller sibling of `_stat_gauge_card`
# (no percentile bar - these 3 tiles are a name+share pair, not a
# percentile position) built from the same label/value/muted-sub visual
# language for house-style consistency. `_select_gap_chart_indices` is a
# pure filter on `gaps_pp` alone (never touching used_top_idx/
# productive_top_idx), satisfying the amendment's own explicit warning
# that a tiny-usage/large-production archetype must always survive.
# `usage_production_gap_chart` reuses `diverging_bar()` itself (not a
# lookalike) for the "deliberately consistent" requirement, post-
# processing the returned figure's own hover/annotations only (the same
# established pattern C2's drift-attribution chart already uses, so
# D1/D2/D3's other diverging_bar call sites are untouched) - hover
# customdata is built by replicating diverging_bar's own internal
# ascending sort on the exact same values, the same discipline applied
# after the chart/text-mismatch bug found once already this session (see
# Section B's radar chart note), so a hover bubble can never label the
# wrong bar. `dumbbell_chart` itself is unchanged, just moved inside an
# expander. `miscasting_feature_grounding()`'s new per-side None handling
# (see step2b's own note) lets this render a two-sided, one-sided, or
# fully-omitted clause depending on what actually clears the whitelist +
# threshold, rather than a forced non-sequitur.
# Not AI: the card contents/merge rule, the gap-chart formula/sort/
# threshold/edge-case, the axis-end plain-language wording, and the
# whitelist - all specified directly in the two prompts.
GAP_CHART_THRESHOLD_PP = 3.0  # percentage points - disclosed, fixed (see DIAGNOSTICS_README.md)


def _simple_stat_card(label, value_display, sub=None, color=BL_INK):
    sub_html = (f'<div style="font-size:13px; color:{BL_MUTED}; margin-top:6px;">{sub}</div>'
               if sub else "")
    return f"""
    <div style="padding:10px 4px; height:100%; box-sizing:border-box;">
      <div style="font-size:12px; letter-spacing:0.04em; text-transform:uppercase;
                  color:{BL_MUTED}; margin-bottom:8px;">{label}</div>
      <div style="font-size:21px; font-weight:700; color:{color}; line-height:1.25;">{value_display}</div>
      {sub_html}
    </div>
    """


def _select_gap_chart_indices(gaps_pp, threshold_pp=GAP_CHART_THRESHOLD_PP):
    """Which archetypes the gap chart draws. Filtered by GAP SIZE only,
    never by the player's top archetype in either projection - an
    archetype tiny in usage but large in production (an "untapped
    direction") must always survive this filter. Returns (shown_idx,
    all_aligned) - all_aligned=True means fewer than 2 archetypes cleared
    the threshold, so the top-3 by |gap| are shown instead: an
    "everything aligned" player is a valid finding, not an empty chart."""
    qualifying = [i for i in range(len(gaps_pp)) if abs(gaps_pp[i]) >= threshold_pp]
    if len(qualifying) >= 2:
        return qualifying, False
    order_by_abs = list(np.argsort(-np.abs(gaps_pp)))
    return order_by_abs[:3], True


def usage_production_gap_chart(names, used, prod, shown_idx, height=380):
    gaps_pp = [(prod[i] - used[i]) * 100 for i in shown_idx]
    shown_names = [names[i] for i in shown_idx]
    df = pd.DataFrame({"archetype": shown_names, "gap_pp": gaps_pp})
    fig = diverging_bar(df, "gap_pp", "archetype", [], height=height, x_title="")

    sort_order = np.argsort(gaps_pp)
    hover_text = [f"used {used[shown_idx[j]]:.0%} → produces {prod[shown_idx[j]]:.0%}" for j in sort_order]
    fig.update_traces(
        customdata=np.array(hover_text).reshape(-1, 1),
        hovertemplate="<b>%{y}</b><br>%{customdata[0]}<br>Gap: %{x:+.1f}pp<extra></extra>",
    )
    fig.add_annotation(
        text="← over-deployed: usage exceeds what production supports",
        xref="paper", yref="paper", x=0, y=-0.16, xanchor="left", yanchor="top",
        showarrow=False, font=dict(color=BL_MUTED, size=12),
    )
    fig.add_annotation(
        text="untapped: production points here more than usage →",
        xref="paper", yref="paper", x=1, y=-0.16, xanchor="right", yanchor="top",
        showarrow=False, font=dict(color=BL_MUTED, size=12),
    )
    fig.update_layout(margin=dict(b=70))
    return fig


def render_section_d(player_id, fit, k, labels, recipes_all):
    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    st.markdown("#### E. Is he being used the way he produces?")
    with st.container(border=True):
        st.caption(
            "Two portraits of the same player: how his team USES him vs. what his PRODUCTION looks "
            "like — do they agree?"
        )
        with st.expander("How this works"):
            st.caption(
                "Two partial archetype recipes are projected onto the same fixed basis: **role-as-used** "
                "(deployment features only — usage rate, shot-location mix, assisted-shot share, play-type "
                "shares) vs. **role-as-productive** (outcome features only — TS%, AST%, TOV%, STL%, BLK%, "
                "TRB%, FTr, BPM). Height and average shot distance are held neutral in both projections (see "
            )

        mc = load_miscasting_cached(player_id, fit, SEASON)
        used, prod = mc["opportunity_recipe"], mc["outcome_recipe"]
        used_top_idx = int(np.argmax(used))
        productive_top_idx = int(np.argmax(prod))
        underused_idx = mc["underused_idx"]

        if used_top_idx == productive_top_idx:
            card_cols = st.columns(2)
            with card_cols[0]:
                st.markdown(
                    _simple_stat_card(
                        "Used and produces as", arch_names[used_top_idx],
                        f"{used[used_top_idx]:.0%} used · {prod[productive_top_idx]:.0%} produces"),
                    unsafe_allow_html=True,
                )
            with card_cols[1]:
                st.markdown(
                    _simple_stat_card("Biggest untapped", arch_names[underused_idx], f"{mc['gap_pp']:+.1f}pp"),
                    unsafe_allow_html=True,
                )
        else:
            card_cols = st.columns(3)
            with card_cols[0]:
                st.markdown(
                    _simple_stat_card("Used as", arch_names[used_top_idx], f"{used[used_top_idx]:.0%}"),
                    unsafe_allow_html=True,
                )
            with card_cols[1]:
                st.markdown(
                    _simple_stat_card("Produces like", arch_names[productive_top_idx],
                                      f"{prod[productive_top_idx]:.0%}"),
                    unsafe_allow_html=True,
                )
            with card_cols[2]:
                st.markdown(
                    _simple_stat_card("Biggest untapped", arch_names[underused_idx], f"{mc['gap_pp']:+.1f}pp"),
                    unsafe_allow_html=True,
                )

        # AI-ASSISTED (Claude Code, chat)
        # Prompt: "我觉得我画了很多这种水平柱状图 似乎有点单调了" (I feel like I've
        # drawn a lot of these horizontal bar charts, it's gotten a bit
        # monotonous), followed by "do swap please" confirming the
        # suggested fix: promote the dumbbell (both raw values as two
        # connected points) to the primary chart, demote the diverging
        # gap-bar (introduced in an earlier redesign this session) to a
        # collapsed expander. D1/D2/D3 stay as bars - they're single-value
        # comparisons the dumbbell's two-endpoint form doesn't fit.
        # Used: dumbbell_chart() itself is completely unchanged (still all
        # 8 archetypes, sorted by |gap| descending) - only its container
        # changed, from a collapsed expander to the always-visible slot the
        # gap chart used to occupy. usage_production_gap_chart() and its
        # supporting _select_gap_chart_indices()/"omitted archetypes"
        # caption are also unchanged, just moved into the expander the
        # dumbbell used to live in - display-only, no computation change.
        # Not AI: the swap decision itself - given directly.
        st.plotly_chart(dumbbell_chart(arch_names, used, prod), width="stretch")

        if SHOW_GAP_CHART_EXPANDER:
            gaps_pp = [(prod[i] - used[i]) * 100 for i in range(k)]
            shown_idx, all_aligned = _select_gap_chart_indices(gaps_pp)
            omitted_idx = [i for i in range(k) if i not in shown_idx]
            with st.expander("See the usage-vs-production gap chart"):
                st.plotly_chart(usage_production_gap_chart(arch_names, used, prod, shown_idx), width="stretch")
                if all_aligned:
                    st.caption(
                        f"Every archetype's usage-vs-production gap is under {GAP_CHART_THRESHOLD_PP:.0f}pp — "
                        f"even his 3 largest are shown here. An \"everything aligned\" player is a real finding, "
                        f"not an empty chart."
                    )
                elif omitted_idx:
                    omitted_names = ", ".join(arch_names[i] for i in omitted_idx)
                    st.caption(
                        f"The other {len(omitted_idx)} archetypes are aligned within "
                        f"±{GAP_CHART_THRESHOLD_PP:.0f}pp: {omitted_names}."
                    )

        # AI-ASSISTED (Claude Code, chat) - Prompt: "重复了 保留第二个 miscast
        # risk please" (this repeats [the "Miscast risk" card below], keep
        # the second one). Used: removed this card's own st.markdown/
        # _stat_gauge_card call - it showed the same 74% percentile as
        # "Miscast risk" right below it, just under a different label
        # ("Notably miscast" vs. "High risk"), reading as a duplicate once
        # both were visible in the same section. pct/tercile_word/
        # verdict_color themselves are kept (still feed "Miscast risk" and
        # the text caption below), only the display call is gone.
        # Not AI: the "this repeats" observation and which one to keep -
        # the user's own call.
        league_scores = load_league_miscasting(recipes_all, fit, k)
        pct = diag2b.percentile_rank(mc["score"], league_scores)
        tercile_word = diag2b.tercile_label(mc["score"], league_scores,
                                            "well aligned", "typical alignment", "notably miscast")
        verdict_color = {"well aligned": BL_GREEN, "typical alignment": BL_GOLD,
                         "notably miscast": BL_CORAL}[tercile_word]

        if used_top_idx != productive_top_idx:
            article = lambda name: "an" if name[0].upper() in "AEIOU" else "a"
            st.markdown(
                f"Deployed primarily as {article(arch_names[used_top_idx])} "
                f"**{arch_names[used_top_idx]}**, but produces most like "
                f"{article(arch_names[productive_top_idx])} **{arch_names[productive_top_idx]}**."
            )

        underused = arch_names[underused_idx]
        grounding = load_miscasting_grounding_cached(player_id, fit, underused_idx, SEASON)
        clause = ""
        if grounding["outcome_feature"] and grounding["opportunity_feature"]:
            out_label = FEATURE_LABELS.get(grounding["outcome_feature"], grounding["outcome_feature"])
            opp_label = FEATURE_LABELS.get(grounding["opportunity_feature"], grounding["opportunity_feature"])
            clause = (f" — backed by his {out_label} ({grounding['outcome_z']:+.1f} SD) against a shortfall "
                     f"in {opp_label} ({grounding['opportunity_deficit']:+.1f} SD below a typical {underused})")
        elif grounding["outcome_feature"]:
            out_label = FEATURE_LABELS.get(grounding["outcome_feature"], grounding["outcome_feature"])
            clause = f" — backed by his {out_label} ({grounding['outcome_z']:+.1f} SD)"
        elif grounding["opportunity_feature"]:
            opp_label = FEATURE_LABELS.get(grounding["opportunity_feature"], grounding["opportunity_feature"])
            clause = (f" — backed by a shortfall in {opp_label} ({grounding['opportunity_deficit']:+.1f} SD "
                     f"below a typical {underused})")
        st.caption(
            f"Within {tercile_word}, the largest untapped direction is **{underused}** "
            f"({mc['gap_pp']:+.1f}pp){clause}."
        )

        # AI-ASSISTED (Claude Code, chat) - Prompt: "把这两个Bottom Line的两个
        # 卡片 Role Elasticity and Miscast Risk放在D3 and E的图下面" (put the
        # two Bottom Line cards under D3's and E's own charts instead of
        # together at the report's end). Used: relocated from the now-
        # removed render_bottom_line_cards, reusing tercile_word/pct/
        # verdict_color/underused/mc already computed just above in this
        # SAME function - no recomputation, same re-labeling ("Low/
        # Moderate/High risk" rather than repeating "well aligned"/
        # "notably miscast" verbatim) so it still reads as a distinct
        # takeaway from the alignment card directly above it, not a
        # duplicate.
        # Not AI: the relocation itself - the user's own call.
        #
        # AI-ASSISTED (Claude Code, chat) - Prompt (this revision): "Miscast
        # Risk...Risk是不是有点过" (is "Risk" a bit much) -> "all of them
        # please" (rename everywhere, not just here). Used: "risk" implies a
        # validated probability of a bad outcome; this is a JS-distance
        # between his usage and production profiles, which the report's own
        # standing caveat already calls purely descriptive ("not proof a
        # role change would improve results") - "risk" was overclaiming
        # relative to what the metric actually establishes. Renamed to
        # "Miscast score" ("score" is neutral, and keeps this project's own
        # pre-existing "miscast" vocabulary rather than inventing new
        # terminology) - same variable/color/value, only the label text and
        # the local variable name (risk_word -> level_word) changed.
        # Not AI: the terminology concern and the decision to fix it
        # everywhere, not just the surface it was noticed on - the user's
        # own call.
        level_word = {"well aligned": "Low", "typical alignment": "Moderate", "notably miscast": "High"}[tercile_word]
        median_score = float(np.median(league_scores)) if len(league_scores) else None
        st.markdown(
            _stat_gauge_card(
                "Miscast score", f"{level_word} score", pct,
                f"{pct:.0%} league percentile for gap size · biggest untapped: {underused} ({mc['gap_pp']:+.1f}pp)",
                color=verdict_color,
                player_value=f"{mc['score']:.3f} JS distance",
                median_value=f"{median_score:.3f} JS distance" if median_score is not None else None,
            ),
            unsafe_allow_html=True,
        )

        st.caption("Stylistic-consistency diagnostic, not proof a role change would improve results.")


# AI-ASSISTED (Claude Code, chat)
# Prompt: "把图像放在左侧 文字和说明放在右侧 并且图的字体太小了" (put the chart on
# the left, text/explanation on the right, and the chart's font is too
# small).
# Used: bumped the radar's own axis label size (10->14) and radial-axis/
# legend fonts up to match the rest of the page's chart typography: moved
# render_section_e to a two-column layout (radar in the left/wider column,
# the deviation callouts + E2 note in the right column) instead of a
# single stacked block.
# Not AI: the layout direction and the font-too-small complaint - both the
# user's own reaction to the rendered page.
# AI-ASSISTED (Claude Code, chat)
# Prompt: "和图像对不上 干脆不在这个地方加上我复制你的这段文字说明 就是如果有很
# 明显的和style neighbours不同 直接在radar chart中标注" (this text doesn't match
# the image - don't add this text explanation here at all; instead, if
# there's an obviously different value from his style neighbors, annotate
# it directly on the radar chart) - a real, found bug: `top_deviations`
# (player_signature's own top-3-by-|diff| ACROSS ALL 29 features) was
# computed independent of `axis_idx` (the chart's own top-10-loading +
# forced-deviation subset), so the bullet list could reference a feature
# that was never actually plotted on the chart the reader was looking at.
# Used: dropped the separate bullet-list column entirely. Added an optional
# `callouts` param to signature_radar_chart() - a 3rd Scatterpolar trace
# drawn ONLY at points already present in `feature_names`/`player_z` (so a
# callout can never reference an off-chart feature by construction), with
# per-point color (reusing _deviation_color's own better/worse/neutral
# logic) and a text label showing the SD value directly beside the point.
# render_section_e now computes its "biggest deviations" by restricting
# the search to axis_idx itself, not all 29 features.
# Not AI: the fix direction (annotate on the chart, don't keep an
# inconsistent separate list) - the user's own diagnosis and instruction.
def signature_radar_chart(feature_names, player_z, centroid_z, height=460, callouts=None):
    categories = list(feature_names) + [feature_names[0]]
    player_vals = list(player_z) + [player_z[0]]
    centroid_vals = list(centroid_z) + [centroid_z[0]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=centroid_vals, theta=categories, name="Style neighbors",
        line=dict(color=BL_MUTED, width=2), fill="toself", fillcolor=_hex_to_rgba(BL_MUTED, 0.18),
    ))
    fig.add_trace(go.Scatterpolar(
        r=player_vals, theta=categories, name="Him",
        line=dict(color=BL_GREEN, width=2), fill="toself", fillcolor=_hex_to_rgba(BL_GREEN, 0.25),
    ))
    if callouts:
        # AI-ASSISTED (Claude Code, chat)
        # Prompt: "我有一个想法 就是把radar chart放在左边 然后右边配一个表格
        # 显示前五个feature差异性大的 就是他明显better或者worse的feature能力"
        # (radar on the left, a table on the right showing the top-5
        # biggest-difference features - the ones where he's clearly better
        # or worse) - the precise numbers moved to a table (see
        # render_section_e), so permanent floating text labels on the chart
        # (which also had their own overlap problem - see the earlier
        # textposition fix) are no longer needed; kept as hover text
        # instead, so the exact SD value is still one hover away without
        # permanently cluttering the chart.
        # Not AI: the chart-left/table-right redesign - given directly.
        fig.add_trace(go.Scatterpolar(
            r=[c["r"] for c in callouts], theta=[c["theta"] for c in callouts],
            mode="markers", marker=dict(size=12, color=[c["color"] for c in callouts],
                                        line=dict(color=BL_WHITE, width=1.5)),
            text=[c["text"] for c in callouts], hovertemplate="%{theta}: %{text}<extra></extra>",
            showlegend=False,
        ))
    # AI-ASSISTED (Claude Code, chat) - bugfix found while verifying the
    # category-grouping request live (not asked for separately): the axes
    # were fed in already grouped by category, but Plotly's default
    # categorical angular-axis ordering doesn't reliably follow trace order
    # once the FIRST category is repeated to close the loop
    # (`categories = feature_names + [feature_names[0]]`) - it silently
    # regrouped them back to something else. `categoryorder="array"` +
    # `categoryarray` (the de-duplicated, already-grouped list) forces
    # Plotly to honor the exact order this function was given, regardless
    # of the closing-loop repeat.
    fig.update_layout(
        height=height, margin=dict(l=155, r=130, t=50, b=50),
        paper_bgcolor=BL_PAPER, font_color=BL_INK,
        polar=dict(bgcolor=BL_PAPER, radialaxis=dict(color=BL_INK, gridcolor=BL_LINE, tickfont=dict(size=11)),
                  angularaxis=dict(color=BL_INK, gridcolor=BL_LINE, tickfont=dict(size=13),
                                   categoryorder="array", categoryarray=list(feature_names))),
        legend=dict(font=dict(color=BL_INK, size=13), orientation="h", y=-0.12),
    )
    return fig


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
# AI-ASSISTED (Claude Code, chat) - Prompt: PDF diagnosis report's radar
# chart needs tight, uppercase spoke labels (7.5px, e.g. "HAND-OFFS",
# "AVG DIST") - FEATURE_LABELS above is sized for body text ("Hand-offs",
# "Avg. shot distance") and overflows the radar's own SVG viewBox for
# longer names (confirmed - "Scoring rate" clipped before this map was
# added). Covers all 29 feature columns, since the radar's real axis
# selection (load_signature_cached) is dynamic per player - any of the 29
# could appear, not just the 11 the reference's one sample happens to show.
# Not AI: none of the specific abbreviations were given - authored here to
# match the reference's own style for the 11 that overlap with its sample.
RADAR_AXIS_SHORT_LABELS = {
    "PTS_PER_100": "SCORING", "TS%": "TS%", "USG%": "USAGE", "AST%": "AST%",
    "TOV%": "TOV", "STL%": "STL%", "BLK%": "BLK%", "TRB%": "REB%", "FTr": "FT RATE",
    "BPM": "BPM", "PLAYER_HEIGHT_INCHES": "HEIGHT", "Dist.": "AVG DIST",
    "% of FGA by Distance_0-3": "RIM", "% of FGA by Distance_3-10": "SHORT MID",
    "% of FGA by Distance_10-16": "LONG MID", "% of FGA by Distance_16-3P": "DEEP MID",
    "% of FGA by Distance_3P": "3PT RATE", "Corner 3s_%3PA": "CORNER 3",
    "% of FG Ast'd_2P": "AST'D 2P", "% of FG Ast'd_3P": "AST'D 3P",
    "PLAYTYPE_CUT": "CUTS", "PLAYTYPE_HANDOFF": "HAND-OFFS", "PLAYTYPE_ISOLATION": "ISO",
    "PLAYTYPE_OFFREBOUND": "PUTBACKS", "PLAYTYPE_OFFSCREEN": "OFF-SCR",
    "PLAYTYPE_PRBALLHANDLER": "PR HANDLER", "PLAYTYPE_PRROLLMAN": "PR ROLLER",
    "PLAYTYPE_POSTUP": "POST-UPS", "PLAYTYPE_SPOTUP": "SPOT-UPS",
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
# Only these 9 have a defensible "higher/lower is better" reading (they're
# step2b's own OUTCOME_FEATURES, reused verbatim) - TOV% is the one
# feature in the set where LOWER is better.
HIGHER_IS_BETTER = {"PTS_PER_100": True, "TS%": True, "AST%": True, "TOV%": False,
                   "STL%": True, "BLK%": True, "TRB%": True, "FTr": True, "BPM": True}


def _deviation_color(feature_name, dv):
    if feature_name not in HIGHER_IS_BETTER:
        return BL_INK
    is_better = (dv > 0) if HIGHER_IS_BETTER[feature_name] else (dv < 0)
    return BL_GREEN if is_better else BL_CORAL


def render_section_e(player_id, prow, fit, k, recipes_all):
    st.markdown("#### B. What makes him different from his role?")
    with st.container(border=True):
        centroid, ess, feature_columns = load_peer_centroid(player_id, recipes_all, fit, k, SEASON)
        sig = load_signature_cached(player_id, centroid, fit, feature_columns, SEASON)
        st.caption(
            f"Compared against his style neighbors, not a single archetype label - every other "
            f"player weighted by how similar his full archetype mixture is to {prow['PLAYER_NAME']}'s "
            f"(Jensen-Shannon distance), so a hybrid player is compared to other hybrids, not to "
            f"players who happen to share only his nominal top archetype. Effective sample size "
            f"≈{ess:.0f} players."
        )
        axis_idx = sig["axis_idx"]
        # group same-category features together instead of leaving them in
        # loading-magnitude order - a stable sort keyed on CATEGORY_ORDER's
        # position, so within a category the original relative order holds.
        axis_idx = sorted(
            axis_idx,
            key=lambda i: CATEGORY_ORDER.index(FEATURE_CATEGORY.get(feature_columns[i], "Box score"))
        )
        names_subset = [FEATURE_LABELS.get(feature_columns[i], feature_columns[i]) for i in axis_idx]
        player_subset = [sig["player_z"][i] for i in axis_idx]
        centroid_subset = [sig["centroid_z"][i] for i in axis_idx]

        # "biggest deviations" restricted to axis_idx (what's actually
        # plotted) - never the full 29-feature search player_signature's
        # own top_deviations uses, which could point at an off-chart
        # feature (see this block's own note above).
        local_diffs = np.array(player_subset) - np.array(centroid_subset)
        n_show = min(5, len(local_diffs))
        top_local_idx = np.argsort(-np.abs(local_diffs))[:n_show]
        callouts = []
        for j in top_local_idx[:3]:
            fname = feature_columns[axis_idx[j]]
            dv = float(local_diffs[j])
            callouts.append({
                "theta": names_subset[j], "r": player_subset[j],
                "text": f"{dv:+.1f} SD", "color": _deviation_color(fname, dv),
            })

        radar_height = 460  # must match signature_radar_chart's own default height
        radar_col, table_col = st.columns([5, 7])
        with radar_col:
            st.plotly_chart(
                signature_radar_chart(names_subset, player_subset, centroid_subset,
                                      height=radar_height, callouts=callouts),
                width="stretch",
            )
        with table_col:
            st.markdown("**Top 5 differences vs. his style neighbors**")
            # AI-ASSISTED (Claude Code, chat)
            # Prompt: "表格字体太小了 而且和标题...有很多空白 我觉得table可以加一个
            # column feature description就是描述一下这一列是做干什么的" (font too
            # small, too much blank space under the title, add a column
            # describing what each feature measures).
            # Used: widened this column (radar_col/table_col 7/5 -> 5/7) and
            # gave the table its own larger base font (16px, up from the
            # shared 14px default - via _build_sortable_table_html's new
            # font_size_px param, which every OTHER sortable table in the
            # app still gets to skip) since a real sentence-length
            # Description column needs real width, not just a wider font.
            # Not AI: the three complaints themselves - given directly.
            columns = [("Feature", "name"), ("Diff", "diff"), ("", None), ("Description", None)]
            rows_cells = []
            for j in top_local_idx:
                fname = feature_columns[axis_idx[j]]
                dv = float(local_diffs[j])
                label = names_subset[j]
                color = _deviation_color(fname, dv)
                direction = "above" if dv > 0 else "below"
                desc = FEATURE_DESCRIPTIONS.get(fname, "")
                rows_cells.append([
                    (label, label.lower()),
                    (f'<span style="color:{color}; font-weight:600;">{dv:+.1f} SD</span>', dv),
                    (f'<span style="color:{BL_MUTED}; font-size:13px;">{direction} typical</span>', direction),
                    (f'<span style="white-space:normal; color:{BL_MUTED}; font-size:14px; '
                     f'line-height:1.35;">{desc}</span>', None),
                ])
            table_html, table_content_height = _build_sortable_table_html(
                "signature_top5_table", columns, rows_cells, row_height=64, font_size_px=16)
            # AI-ASSISTED (Claude Code, chat)
            # Prompt (earlier this session): "this table vertical align
            # please" - the 5-row table sat flush at the top of its column
            # while the much taller radar chart beside it left a lot of
            # dead space below the table. That fix centered the table
            # within a container sized to the radar's own height, which
            # then produced THIS session's complaint (blank space between
            # the title and the table, not below it). Switched
            # align-items:center -> flex-start so the table sits directly
            # under its own title instead of floating mid-column; any
            # leftover space (much smaller now that a wider font + a
            # wrapping Description column make the table itself taller)
            # falls below the table instead, which reads as normal next to
            # a taller chart rather than as a gap under the table's own
            # caption.
            wrapper_height = max(radar_height - 40, table_content_height)
            wrapped_html = (
                f'<div style="display:flex; align-items:flex-start; height:{wrapper_height}px;">'
                f'{table_html}</div>'
            )
            components.html(wrapped_html, height=wrapper_height, scrolling=False)


def bootstrap_band_chart(arch_names, point, lo, hi, height=380):
    order = np.argsort(point)[::-1]
    cats = [arch_names[i] for i in order]
    vals = [float(point[i]) for i in order]
    err_plus = [float(hi[i] - point[i]) for i in order]
    err_minus = [float(point[i] - lo[i]) for i in order]
    fig = go.Figure(go.Bar(
        x=cats, y=vals, marker=dict(color=BL_GREEN, cornerradius=6),
        error_y=dict(type="data", symmetric=False, array=err_plus, arrayminus=err_minus,
                    color=BL_INK, thickness=1.5),
        text=[f"{v:.0%}" for v in vals], textposition="outside", textfont=dict(color=BL_INK, size=13),
        hovertemplate="%{x}: %{y:.1%}<extra></extra>",
    ))
    # automargin fix - same underlying bug as archetype_column_chart's own
    # note (rotated category labels clip in kaleido's static export without it).
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=30, b=90), showlegend=False,
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(tickfont=dict(color=BL_INK, size=13), automargin=True),
        yaxis=dict(visible=False, range=[0, (max(hi) * 1.3) if len(hi) else 1]),
    )
    return fig


# render_section_f was fully merged into render_section_a (the A-F report
# restructure's "one merge") - deleted rather than kept unreferenced, same
# reasoning as the earlier render_layer2_player_card removal: every piece
# of it now has a direct home inside render_section_a, unlike
# render_roster_construction's own "kept but not wired up" case.
# bootstrap_band_chart() itself is kept, unreferenced ("kept but not wired
# up") - Section A's own chart (archetype_column_chart's err_lo/err_hi)
# already covers the with-bootstrap case wherever bootstrap data is shown
# now, live or in the PDF report (src/player_report.py).


def render_player_report(player_id, recipes, k, labels, bio, exposure_cache):
    prow_match = recipes[recipes["PLAYER_ID"] == player_id]
    if len(prow_match) == 0:
        st.caption("No recipe found for this player.")
        return
    prow = prow_match.iloc[0]
    arch_cols = [f"arch_{i}" for i in range(k)]
    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    own_vals = prow[arch_cols].values.astype(float)
    top_i = int(np.argmax(own_vals))

    recipes_all = load_recipes_all_seasons()
    bio_all = load_bio_all_seasons()
    fit = load_ada_basis(BASIS_DIR)

    st.markdown(f"### 2. {prow['PLAYER_NAME']} — {own_vals[top_i]:.0%} {arch_names[top_i]}")

    # AI-ASSISTED (Claude Code, chat)
    # Prompt: full A-F report restructure spec - new three-act order:
    # Act 1 (identity) = A then E ("What makes him different" deepens A,
    # so it follows directly); Act 2 (context) = B then C unchanged; Act 3
    # (verdict) = D last, followed by a new computed summary box. F no
    # longer appears here - fully merged into render_section_a.
    # Used: only the CALL ORDER changed - every render_section_x function's
    # own internals, arguments, and computation are untouched, per the
    # spec's explicit "do NOT change any computation... every section's
    # internals stay exactly as they are" constraint. Function names still
    # match their ORIGINAL content (render_section_e still renders the
    # "What makes him different" content, now displayed as header "B") -
    # renaming functions to match new display letters wasn't requested and
    # would add rename risk with no behavioral benefit; each function's own
    # st.markdown("#### ...") call is what changed, not its Python name.
    # Not AI: the new order and the merge/summary requirements - given
    # directly in the task spec.
    render_section_a(player_id, prow, k, labels, bio, recipes_all, fit)
    render_section_e(player_id, prow, fit, k, recipes_all)
    render_section_b(player_id, recipes_all, bio_all, k, labels, fit)
    render_section_c(player_id, recipes, k, labels, exposure_cache)
    render_section_d(player_id, fit, k, labels, recipes_all)


# ===========================================================================
# Report - PDF scouting report (src/player_report.py builds it; this
# function ONLY assembles the data, reusing the exact same cached loaders
# and chart-building functions every render_section_x above already calls)
# ===========================================================================
# AI-ASSISTED (Claude Code, chat)
# Prompt: "Add a 'Report' section at the very end of the Diagnostic
# Analysis page: a per-player PDF scouting report generator... CORE RULE:
# The PDF is assembled ONLY from values, verdict texts, and figures the
# page has ALREADY computed for the selected player (the same cached
# objects the sections render from). Never recompute analytics inside the
# report builder, never hardcode any player-specific text, never
# fabricate a number that isn't on the page." Replaces a much older,
# fully-stale Report feature (deleted, not superseded-but-kept, since it
# referenced field names and a team-level D3 proxy from BEFORE this
# session's C1/C2 rework, Section E redesign, and D3 Phase 1/2 rebuild -
# it would have needed a near-total rewrite regardless).
# Used: every value below comes from calling the SAME @st.cache_data
# loader or chart-building function the corresponding render_section_x
# call already uses (load_mismatch, load_teammate_lift,
# compute_bc_verdict, load_individual_role_sensitivity_cached,
# _elasticity_verdict, _individual_sensitivity_verdict, load_miscasting_
# cached, load_league_miscasting, load_miscasting_grounding_cached,
# _select_gap_chart_indices, usage_production_gap_chart,
# load_role_drift_cached, diag2b.season_transitions/transition_drift_
# magnitude/tercile_label/transition_team_changed, load_drift_attribution_
# cached, load_peer_centroid, load_signature_cached, load_league_purity_
# entropy, diag2b.purity_entropy/percentile_rank/tercile_label,
# archetype_column_chart, signature_radar_chart, diverging_bar,
# individual_sensitivity_chart) - by the time a user reaches the bottom of
# the page every one of these is already a cache hit, so assembling the
# report costs nothing beyond re-running the (cheap, non-cached)
# string-formatting and re-building the (cheap, non-cached) Plotly figure
# objects those loaders feed - never a second, independently-computed
# version of any analytic value.
# A real conflict found while inventorying the current page (not assumed
# away): the new spec's item 2 asks for "the auto-generated summary
# paragraph exactly as rendered on the page" - but the Scouting Summary
# feature was fully deleted earlier this session per an explicit user
# request, so no such paragraph exists anywhere on the current page to
# source from. Per the CORE RULE's own "never fabricate" instruction,
# this function does NOT attempt to regenerate an equivalent paragraph -
# the PDF simply has no Scouting Summary page, and this is called out
# here and in AI_USAGE.md rather than silently working around it.
# Similarly, "each section's one-line honesty caveat" reuses whichever
# real caveat sentence that section CURRENTLY shows verbatim (e.g. E's
# "Stylistic-consistency diagnostic..."); for sections without one single
# isolable caveat sentence anymore (the old global Limitations block was
# ALSO removed earlier this session), the closest already-shown
# explanatory line is used instead of inventing a new one.
# Not AI: the CORE RULE, the page/section structure, and the exact chart
# selection per section - all given directly. Both conflicts above (the
# missing Scouting Summary and the missing global Limitations block) were
# found by checking the current code, not assumed, and resolved by
# omission/substitution rather than fabrication, per the CORE RULE itself.
_MISCAST_MAGNITUDE_PHRASE = {
    "well aligned": "matches his production",
    "typical alignment": "runs close to his production",
    "notably miscast": "sits well below his production",
}


def _build_diagnosis_line(arch_names, top_i, own_vals, tercile_word, underused, gap_pp):
    phrase = _MISCAST_MAGNITUDE_PHRASE.get(tercile_word, "compares to his production")
    return (
        f"{own_vals[top_i]:.0%} {arch_names[top_i]} whose usage {phrase} — "
        f"{underused} ({rsc.signed(gap_pp)}pp) is the untapped lever."
    )


def _build_style_profile_read(shot_mix, play_types):
    dominant_label, dominant_pct = max(shot_mix, key=lambda t: t[1])
    top3_names = ", ".join(name for name, _ in play_types[:3])
    return f"{dominant_pct:.0f}% of his shot attempts come from {dominant_label}, fed primarily by {top3_names}."


def _build_lift_caption(lift_rows):
    """Describes the best/worst archetypes from the SAME lift_rows the D2
    bars actually render (already top-3 + bottom-3, or fewer) - never an
    independently re-filtered set. An earlier version filtered by
    shared-minutes mass on its own, which could (and did, for a
    thin-minutes player - caught by testing) name archetypes the D2 chart
    right above the caption didn't even show."""
    positive = [(n_, v) for n_, v in lift_rows if v > 0]
    negative = [(n_, v) for n_, v in lift_rows if v < 0]
    best = [n_ for n_, v in sorted(positive, key=lambda t: -t[1])[:2]]
    worst = [n_ for n_, v in sorted(negative, key=lambda t: t[1])[:2]]
    parts = []
    if best:
        parts.append(f"Wins next to {' & '.join(best)}")
    if worst:
        parts.append(f"loses next to {' & '.join(worst)}")
    if not parts:
        return "No clear win/loss pattern by teammate archetype yet."
    sentence = "; ".join(parts)
    return sentence if sentence.endswith(".") else sentence + "."


def _build_minutes_caption(arch_names, diff):
    over_name = player_report.abbreviate_archetype(arch_names[int(np.argmax(diff))])
    under_name = player_report.abbreviate_archetype(arch_names[int(np.argmin(diff))])
    return f"He plays next to {over_name} more than his type usually does — and next to {under_name} far less."


def _select_dumbbell_rows(arch_names, used, prod, n=5):
    idx = sorted(range(len(arch_names)), key=lambda i: -max(used[i], prod[i]))[:n]
    return [(player_report.abbreviate_archetype(arch_names[i]), float(used[i]) * 100, float(prod[i]) * 100) for i in idx]


def _build_recommendation_lift(arch_names, lift, n=2):
    ranked = sorted(range(len(arch_names)), key=lambda i: -lift[i])[:n]
    names = [player_report.abbreviate_archetype(arch_names[i]) for i in ranked]
    parts = ", ".join(f"{names[j]} {rsc.signed(lift[ranked[j]])}" for j in range(len(ranked)))
    return {"title": f"Lean into {' + '.join(names)} lineups.", "body": f"Best lifts: {parts}."}


def _build_recommendation_overlap(arch_names, diff, lift_res, profile, mass_threshold=DIAG_MASS_THRESHOLD):
    """Lever 2: the archetype he's simultaneously over-exposed to (D1,
    diff>0) and gets hurt by (D2, negative well-supported lift) - "worst
    overlap from D1xD3xD2" per the report spec. Mirrors compute_bc_verdict's
    own well-supported-mass + graceful-fallback style (portal.py above)."""
    lift, mass = lift_res["lift"], lift_res["exposure_mass"]
    well_supported = mass >= mass_threshold
    candidates = np.where(well_supported & (diff > 0))[0]
    if len(candidates) == 0:
        return None
    worst_idx = int(candidates[np.argmin(lift[candidates])])
    if lift[worst_idx] >= 0:
        return None  # over-exposed but not actually hurting him - no real overlap to flag
    display_name = player_report.abbreviate_archetype(arch_names[worst_idx])
    body = f"{rsc.signed(diff[worst_idx] * 100)}pp over-exposed there; lift {rsc.signed(lift[worst_idx])}"
    if profile.get("available"):
        prof_row = next((p for p in profile["profiles"] if p["archetype_idx"] == worst_idx and p.get("available")), None)
        if prof_row is not None:
            body += f" and usage {rsc.signed(prof_row['usage_delta'] * 100)}pp"
    return {"title": f"Trim {display_name} overlap.", "body": body + "."}


def _build_recommendation_untapped(underused, gap_pp, support_text):
    body = f"{rsc.signed(gap_pp)}pp untapped {underused}"
    if support_text:
        body += f"; {support_text}"
    body += ". Re-measure in 10 games."
    return {"title": f"Grow the {underused} role.", "body": body}


def collect_report_data(player_id, recipes, k, labels, bio, exposure_cache):
    """Builds the PDF diagnosis report's full data contract (see
    report_template/report.html.jinja) for one player - reshaping, not
    recomputing: every value here comes from the SAME @st.cache_data
    loaders the live Diagnostic Analysis sections already call. CORE RULE
    unchanged from this function's original reportlab-era version: never
    recompute analytics here, never hardcode player-specific text, never
    fabricate a number that isn't traceable to a real loader. Prose fields
    (diagnosisLine/reads/recommendations) are deterministic templates over
    already-real numbers, same discipline the old report_summary sentence
    pattern used - never free text."""
    prow_match = recipes[recipes["PLAYER_ID"] == player_id]
    if len(prow_match) == 0:
        return None
    prow = prow_match.iloc[0]
    arch_cols = [f"arch_{i}" for i in range(k)]
    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    own_vals = prow[arch_cols].values.astype(float)
    top_i = int(np.argmax(own_vals))
    name = prow["PLAYER_NAME"]

    bio_match = bio[bio["PLAYER_ID"] == player_id]
    base_stats = load_player_base_stats(season=SEASON)
    base_match = base_stats[base_stats["PLAYER_ID"] == player_id]
    full_features = load_full_features(season=SEASON)
    ff_match = full_features[full_features["PLAYER_ID"] == player_id]
    if len(bio_match) == 0 or len(base_match) == 0 or len(ff_match) == 0:
        return None
    bio_row, base_row, ff_row = bio_match.iloc[0], base_match.iloc[0], ff_match.iloc[0]

    recipes_all = load_recipes_all_seasons()
    fit = load_ada_basis(BASIS_DIR)

    draft = (
        f"{bio_row['DRAFT_YEAR']} Rd {bio_row['DRAFT_ROUND']} Pick {bio_row['DRAFT_NUMBER']}"
        if str(bio_row["DRAFT_YEAR"]) not in ("Undrafted", "nan", "None") else "Undrafted"
    )

    # --- A. archetype mix / purity / entropy ---------------------------
    archetype_mix = sorted(zip(arch_names, (own_vals * 100).tolist()), key=lambda t: -t[1])[:4]
    purities, entropies = load_league_purity_entropy(recipes_all, k)
    purity, entropy = diag2b.purity_entropy(own_vals)
    pu_pct = diag2b.percentile_rank(purity, purities)
    en_pct = diag2b.percentile_rank(entropy, entropies)

    dist_map = [
        ("0-3ft", "% of FGA by Distance_0-3"), ("3-10ft", "% of FGA by Distance_3-10"),
        ("10-16ft", "% of FGA by Distance_10-16"), ("16ft-3P", "% of FGA by Distance_16-3P"),
        ("3PT", "% of FGA by Distance_3P"),
    ]
    shot_mix = [(label, float(ff_row.get(col) or 0) * 100) for label, col in dist_map]
    playtype_cols = [c for c in full_features.columns if c.startswith("PLAYTYPE_")]
    pt_all = sorted(
        ((PLAYTYPE_LABELS.get(c, c.replace("PLAYTYPE_", "").replace("_", " ").title()), float(ff_row.get(c) or 0))
         for c in playtype_cols),
        key=lambda p: -p[1],
    )
    # AI-ASSISTED (Claude Code, chat) - "Other" here is defined as
    # 100% - top3 (fills the stacked bar to exactly 100%, matching the
    # report spec's own sample data, which sums to 100.0 across its 4
    # bars) - a DIFFERENT definition from render_player_stats_tab's own
    # "other_pct" (1 - sum of all 9 tracked play types, representing the
    # share OUTSIDE this project's tracked Synergy taxonomy specifically).
    # An earlier version of this function reused that other definition
    # directly, which left the PDF's stacked bar visibly short of 100% -
    # confirmed by rendering a real player and noticing the bar didn't
    # reach the end.
    # Not AI: none - a bug caught and fixed here by checking the reference
    # sample's own numbers sum to 100%, not by guessing.
    top3_sum = sum(v for _, v in pt_all[:3])
    play_types = [(n_, v * 100) for n_, v in pt_all[:3]] + [("Other", max(0.0, 1.0 - top3_sum) * 100)]

    # --- B. vs. style neighbors (signature/radar) -----------------------
    centroid, ess, feature_columns = load_peer_centroid(player_id, recipes_all, fit, k, SEASON)
    sig = load_signature_cached(player_id, centroid, fit, feature_columns, SEASON)
    axis_idx = sorted(sig["axis_idx"],
                      key=lambda i: CATEGORY_ORDER.index(FEATURE_CATEGORY.get(feature_columns[i], "Box score")))
    radar_axis_labels = [RADAR_AXIS_SHORT_LABELS.get(feature_columns[i], feature_columns[i].upper()[:10]) for i in axis_idx]
    names_subset = [FEATURE_LABELS.get(feature_columns[i], feature_columns[i]) for i in axis_idx]
    player_subset = [float(sig["player_z"][i]) for i in axis_idx]
    centroid_subset = [float(sig["centroid_z"][i]) for i in axis_idx]
    local_diffs = np.array(player_subset) - np.array(centroid_subset)
    n_show = min(5, len(local_diffs))
    top_local_idx = np.argsort(-np.abs(local_diffs))[:n_show]
    top_diffs = [
        (names_subset[j], float(local_diffs[j]), FEATURE_DESCRIPTIONS.get(feature_columns[axis_idx[j]], ""))
        for j in top_local_idx
    ]
    dot_axis_indices = [int(j) for j in top_local_idx[:3]]
    b_top_feat, b_top_dz = names_subset[top_local_idx[0]], float(local_diffs[top_local_idx[0]])

    # --- C. role drift across seasons ------------------------------------
    rd = load_role_drift_cached(player_id, recipes_all, k)
    drift_ctx, drift_insufficient_text = None, None
    if rd["insufficient"]:
        # AI-ASSISTED (Claude Code, chat) - `available_seasons` is a list
        # (e.g. ["2025-26"], possibly empty) - an earlier version
        # interpolated it directly into the f-string, rendering the raw
        # Python repr ("only ['2025-26'] projected") - caught by rendering
        # a real short-history player (E.J. Liddell) and looking at the
        # actual PDF, not assumed from the code.
        seasons_str = ", ".join(rd["available_seasons"]) if rd["available_seasons"] else "no seasons"
        drift_insufficient_text = f"Rookie season — no drift history available (only {seasons_str} projected)."
    else:
        drift_series = [(arch_names[i], (rd["matrix"][:, i] * 100).tolist(), "major") for i in rd["major_idx"]]
        minor_mass = np.zeros(len(rd["seasons"]))
        for i in range(k):
            if i not in rd["major_idx"]:
                minor_mass = minor_mass + rd["matrix"][:, i]
        if minor_mass.sum() > 1e-9:
            drift_series.append(("Minor", (minor_mass * 100).tolist(), "minor"))

        transitions_ctx = []
        for s_old, s_new in diag2b.season_transitions(rd["seasons"]):
            magnitude = diag2b.transition_drift_magnitude(player_id, recipes_all, k, s_old, s_new)
            dist, n_league = load_transition_drift_distribution_cached(recipes_all, k, s_old, s_new)
            t_pct = diag2b.percentile_rank(magnitude, dist)
            verdict = diag2b.tercile_label(magnitude, dist, "stable role", "moderate shift", "major shift")
            severity = {"stable role": "stable", "moderate shift": "moderate", "major shift": "major"}[verdict]
            changed = diag2b.transition_team_changed(recipes_all, player_id, s_old, s_new)
            attr = load_drift_attribution_cached(player_id, recipes_all, fit, k, s_old, s_new)
            if attr["available"]:
                top = attr["features"][0]
                top_label = FEATURE_LABELS.get(top["name"], top["name"])
                old_fmt = diag2b.format_raw_feature_value(top["name"], top["raw_old"])
                new_fmt = diag2b.format_raw_feature_value(top["name"], top["raw_new"])
                mover = f"{top_label} {old_fmt} → {new_fmt} ({rsc.signed(top['dz'], 2)} SD)"
                context = (f"toward {arch_names[attr['R']]} ({rsc.signed(attr['alpha_delta_R_pp'])}pp), "
                          f"away from {arch_names[attr['F']]} ({rsc.signed(attr['alpha_delta_F_pp'])}pp)")
            else:
                mover, context = f"Drift magnitude {magnitude:.3f} ({verdict})", ""
            transitions_ctx.append({
                "delta": magnitude, "severity": severity, "pct": t_pct * 100, "n": n_league,
                "teamChange": bool(changed), "mover": mover, "context": context,
            })
        drift_ctx = {"seasons": rd["seasons"], "series": drift_series, "transitions": transitions_ctx}

    # --- D. environment: lift / minutes-vs-typical / elasticity --------
    score, actual, expected, diff = load_mismatch(player_id, recipes, k, SEASON, exposure_cache)
    lift_res = load_teammate_lift(player_id, recipes, k, SEASON)
    lift, mass = lift_res["lift"], lift_res["exposure_mass"]

    lift_pairs = sorted(zip(arch_names, lift.tolist()), key=lambda t: -t[1])
    lift_rows = [(player_report.abbreviate_archetype(n_), v) for n_, v in
                (lift_pairs[:3] + lift_pairs[-3:] if len(lift_pairs) > 6 else lift_pairs)]

    diff_pairs = sorted(zip(arch_names, (diff * 100).tolist()), key=lambda t: -t[1])
    minutes_rows = [(player_report.abbreviate_archetype(n_), v) for n_, v in
                    (diff_pairs[:2] + diff_pairs[-2:] if len(diff_pairs) > 4 else diff_pairs)]

    profile = load_individual_role_sensitivity_cached(player_id, recipes, k, SEASON)
    if profile.get("available"):
        available_deltas = [(arch_names[p["archetype_idx"]], p["usage_delta"] * 100)
                            for p in profile["profiles"] if p.get("available")]
        available_deltas.sort(key=lambda t: -t[1])
        elasticity_pairs = available_deltas[:2] + available_deltas[-2:] if len(available_deltas) > 4 else available_deltas
        elasticity_rows = [(player_report.abbreviate_archetype(n_), v) for n_, v in elasticity_pairs]
        elasticity_verdict = _elasticity_verdict(profile["profiles"])
        elasticity_label = elasticity_verdict["word"].capitalize() if elasticity_verdict.get("available") else "Stable"
        elasticity_swing = elasticity_verdict.get("spread_pp", 0.0)
        spreads, _n = load_league_elasticity_spreads(recipes, k, SEASON)
        elasticity_pct = diag2b.percentile_rank(elasticity_swing, spreads) * 100 if len(spreads) else 50.0
    else:
        elasticity_rows, elasticity_label, elasticity_swing, elasticity_pct = [], "Stable", 0.0, 50.0

    environment_ctx = {
        "lift_caption": _build_lift_caption(lift_rows),
        "lift_rows": lift_rows,
        "minutesVsTypical": {"mismatchJs": score, "rows": minutes_rows, "caption": _build_minutes_caption(arch_names, diff)},
        "elasticity": {"label": elasticity_label, "swing": elasticity_swing, "pct": elasticity_pct, "rows": elasticity_rows},
    }

    # --- E. used vs. produces (miscast) ---------------------------------
    mc = load_miscasting_cached(player_id, fit, SEASON)
    used, prod = mc["opportunity_recipe"], mc["outcome_recipe"]
    used_top_idx, productive_top_idx = int(np.argmax(used)), int(np.argmax(prod))
    league_scores = load_league_miscasting(recipes_all, fit, k)
    miscast_pct = diag2b.percentile_rank(mc["score"], league_scores) * 100
    tercile_word = diag2b.tercile_label(mc["score"], league_scores, "well aligned", "typical alignment", "notably miscast")
    miscast_level = {"well aligned": "Low", "typical alignment": "Medium", "notably miscast": "High"}[tercile_word]

    underused_idx = mc["underused_idx"]
    underused = arch_names[underused_idx]
    grounding = load_miscasting_grounding_cached(player_id, fit, underused_idx, SEASON)
    # AI-ASSISTED (Claude Code, chat) - `opportunity_deficit` is a magnitude
    # by construction (miscasting_feature_grounding: basis_row - z, kept
    # only when >= a positive threshold - see step2b_player_diagnostics.py),
    # never itself negative. Displayed here as "X.X SD below a typical Y",
    # matching the SAME wording convention render_section_b already uses
    # for this exact value on the live page (portal.py, D3/C section) -
    # not signed, since the word "below" already carries the direction; an
    # earlier draft displayed it as "+0.5 SD" instead, which reads
    # backwards for something called a shortfall.
    # Not AI: none directly - a bug caught and fixed here by checking the
    # source function's real definition rather than assuming a sign.
    support_parts = []
    if grounding["outcome_feature"]:
        out_label = FEATURE_LABELS.get(grounding["outcome_feature"], grounding["outcome_feature"])
        support_parts.append(f"backed by {out_label} ({rsc.signed(grounding['outcome_z'])} SD)")
    if grounding["opportunity_feature"]:
        opp_label = FEATURE_LABELS.get(grounding["opportunity_feature"], grounding["opportunity_feature"])
        support_parts.append(f"a {opp_label} shortfall ({grounding['opportunity_deficit']:.1f} SD below a typical {underused})")
    support_text = " against ".join(support_parts) if support_parts else None

    miscast_ctx = {
        "level": miscast_level, "js": mc["score"],
        "medianJs": float(np.median(league_scores)) if len(league_scores) else mc["score"],
        "pct": miscast_pct,
        "usedAs": (arch_names[used_top_idx], float(used[used_top_idx]) * 100),
        "producesAs": float(prod[used_top_idx]) * 100,
        "untapped": (underused, mc["gap_pp"]),
        "support": support_text,
        "dumbbell": _select_dumbbell_rows(arch_names, used, prod),
    }

    diagnosis_line = _build_diagnosis_line(arch_names, top_i, own_vals, tercile_word, underused, mc["gap_pp"])
    reads_ctx = {
        "styleProfile": _build_style_profile_read(shot_mix, play_types),
        "neighbors": f"His signature stands out most in {b_top_feat} ({b_top_dz:+.1f} SD {'above' if b_top_dz > 0 else 'below'} typical).",
    }
    recommendations = [_build_recommendation_lift(arch_names, lift)]
    overlap_rec = _build_recommendation_overlap(arch_names, diff, lift_res, profile)
    if overlap_rec:
        recommendations.append(overlap_rec)
    recommendations.append(_build_recommendation_untapped(underused, mc["gap_pp"], support_text))

    return {
        "player": {
            "name": name,
            "headshotDataUri": hull_callout_chart.get_headshot_data_uri(int(player_id), name, fallback_color=REPORT_TEAM_COLOR),
            "team": bio_row["TEAM_ABBREVIATION"], "teamName": "Brooklyn Nets", "teamColor": REPORT_TEAM_COLOR,
            "rosterLabel": "Brooklyn Nets Roster", "age": int(bio_row["AGE"]), "height": bio_row["PLAYER_HEIGHT"],
            "weight": int(bio_row["PLAYER_WEIGHT"]) if pd.notna(bio_row["PLAYER_WEIGHT"]) else None,
            "college": bio_row["COLLEGE"] or bio_row["COUNTRY"], "draft": draft,
        },
        "season": {
            "label": SEASON, "generatedDate": date.today().strftime("%b %d, %Y"),
            "totals": {
                "gp": int(base_row["GP"]), "min": int(round(base_row["MIN"])), "pts": int(base_row["PTS"]),
                "reb": int(base_row["REB"]), "ast": int(base_row["AST"]),
            },
        },
        "diagnosisLine": diagnosis_line,
        "archetypeMix": archetype_mix,
        "purity": {"value": purity * 100, "median": float(np.median(purities)) * 100, "pct": pu_pct * 100},
        "entropy": {"value": entropy, "median": float(np.median(entropies)), "pct": en_pct * 100},
        "shotMix": shot_mix,
        "playTypes": play_types,
        "neighbors": {
            "ess": ess, "radar_axes": radar_axis_labels, "radar_him": player_subset,
            "radar_neighbors": centroid_subset, "radar_dot_indices": dot_axis_indices, "topDiffs": top_diffs,
        },
        "drift": drift_ctx, "drift_insufficient_text": drift_insufficient_text,
        "environment": environment_ctx,
        "miscast": miscast_ctx,
        "boxScore": {
            "GP": int(base_row["GP"]), "MIN": int(round(base_row["MIN"])), "PTS": int(base_row["PTS"]),
            "REB": int(base_row["REB"]), "AST": int(base_row["AST"]), "STL": int(base_row["STL"]),
            "BLK": int(base_row["BLK"]), "FG%": float(base_row["FG_PCT"]) * 100, "3P%": float(base_row["FG3_PCT"]) * 100,
            "FT%": float(base_row["FT_PCT"]) * 100, "TS%": float(ff_row["TS%"]) * 100, "USG%": float(ff_row["USG%"]),
            "AST%": float(ff_row["AST%"]), "TRB%": float(ff_row["TRB%"]), "STL%": float(ff_row["STL%"]),
            "BLK%": float(ff_row["BLK%"]), "TOV%": float(ff_row["TOV%"]), "BPM": float(ff_row["BPM"]),
        },
        "recommendations": recommendations,
        "reads": reads_ctx,
        "ARCHETYPE_BAR_GREYS": ["#23262B", "#55585E", "#8B8E93", "#C4C1BA"],
        "SHOT_MIX_COLORS": ["#0B3D2C", "#E9A93B", "#E2725B", "#6B8F5E", "#A8A196"],
        "PLAY_TYPE_COLORS": ["#0B3D2C", "#E9A93B", "#E2725B", "#B9B3A6"],
    }


def render_report_section(player_id, recipes, k, labels, bio, exposure_cache):
    state_key = f"report_pdf_{player_id}_{SEASON}"
    cached = st.session_state.get(state_key)

    def _generate_report():
        data = collect_report_data(player_id, recipes, k, labels, bio, exposure_cache)
        if data is None:
            return {"error": "No recipe found for this player."}
        # AI-ASSISTED (Claude Code, chat) - Prompt: "For PDF page, please
        # refer to README.md and reference-1b-briefing.html file I just
        # sent to you, I want this format please" - the pixel-precise
        # 2-page A4 design replaced the old one-page reportlab report
        # wholesale (see AI_USAGE.md); `build_pdf` now means the new
        # Jinja2+Playwright pipeline, not the old multi-page reportlab
        # function of the same name that used to live here (deleted, not
        # kept - confirmed via grep that nothing else in src/*.py called
        # it, git history preserves it if ever needed again).
        try:
            pdf_bytes = player_report.build_pdf(data)
        except player_report.PDFExportUnavailable as e:
            return {"error": str(e)}
        # AI-ASSISTED (Claude Code, chat) - Prompt: "This
        # page has been blocked by Chrome" (reported twice:
        # once against a data: URI iframe, again against a
        # blob: URL nested-iframe fix that still didn't
        # render in the user's real Chrome). Rather than
        # keep guessing at Chrome's iframe/sandbox/PDF-
        # viewer restrictions with no browser to verify
        # against, switched to the page-by-page PNG
        # fallback the original spec itself anticipated for
        # exactly this case ("if neither renders reliably,
        # show page-by-page PNG previews instead") - plain
        # images can't be blocked by a PDF-specific browser
        # restriction since there's no PDF viewer involved.
        # Converted HERE, at generation time, and cached
        # alongside the PDF bytes - converting on every
        # rerun instead (generation now happens at most once
        # per player+season, see the auto-generate-on-load
        # note below, but the PREVIEW still runs on every
        # rerun of this page) measured at ~0.43s for a 3-page
        # report, non-trivial to pay on every widget
        # interaction elsewhere on the page.
        page_images = pdf2image.convert_from_bytes(pdf_bytes, dpi=150)
        page_pngs = []
        for img in page_images:
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            page_pngs.append(buf.getvalue())
        return {"pdf_bytes": pdf_bytes, "player_name": data["player"]["name"], "page_pngs": page_pngs}

    with st.container(border=True):
        st.caption(
            "A 2-page PDF diagnosis report for this player - built from the same computed values, "
            "verdicts, and charts the Diagnostic Analysis page shows, nothing recomputed."
        )

        # AI-ASSISTED (Claude Code, chat) - Prompt: "when loading the
        # diagnostic analysis page, it will generate preview of pdf report
        # as well" - this section used to require a manual "Generate
        # preview" click before showing anything (the button/spinner/
        # st.rerun() logic just below is unchanged from that version).
        # Used: the first time a given player+season has no cached PDF yet,
        # it's built right here, inline, same render pass - no button click
        # or rerun round-trip needed before a preview appears. `state_key`
        # still gates this to run at most once per player+season - clicking
        # a DIFFERENT player in the quadrant chart triggers one fresh
        # auto-build for him; returning to an already-viewed player, or a
        # rerun triggered by something unrelated elsewhere on the page,
        # hits the cache instead of rebuilding.
        # Not AI: the auto-generate-on-load requirement itself - given
        # directly.
        if cached is None:
            with st.spinner("Building PDF report..."):
                cached = _generate_report()
                st.session_state[state_key] = cached

        col1, col2, col3 = st.columns([1.3, 1.3, 3])
        with col1:
            if st.button("Regenerate", key=f"report_gen_{player_id}"):
                with st.spinner("Building PDF report..."):
                    cached = _generate_report()
                    st.session_state[state_key] = cached
                st.rerun()
        with col2:
            if "pdf_bytes" in cached:
                st.download_button(
                    "Download PDF", data=cached["pdf_bytes"],
                    file_name=f"{cached['player_name'].replace(' ', '_')}_diagnostic_report_{date.today().isoformat()}.pdf",
                    mime="application/pdf", key=f"report_dl_{player_id}",
                )

        if "error" in cached:
            st.caption(cached["error"])
            return

        st.markdown("**Preview**")
        # AI-ASSISTED (Claude Code, chat)
        # Prompt (this feature's original spec): "use st.pdf if the
        # installed Streamlit version has it, else a base64 data-URI
        # iframe... if neither renders reliably, show page-by-page PNG
        # previews instead." Then, in the real browser: "This page has been
        # blocked by Chrome" - reported TWICE, first against a data: URI
        # iframe, then again against a blob: URL nested-iframe fix that
        # still didn't render. Both looked correct at the Python level (no
        # exception - components.html succeeds regardless of what the
        # browser does with the HTML/JS it's given), so the exception-based
        # fallback chain could never actually detect either failure and
        # step past it on its own.
        # Used: `st.pdf` still tried first (checked directly, not assumed,
        # that `hasattr(st, "pdf")` is True in this Streamlit version but
        # the command's own required `streamlit-pdf` extra is NOT
        # installed - deliberately not installing it, since this fallback
        # chain already anticipates its absence). On failure, renders each
        # already-converted page (see `_generate_report()` above, which
        # builds `page_pngs` at generation time) via plain `st.image`
        # - a real `<img>` tag, not a PDF viewer, so there's no PDF-
        # specific browser restriction left to trigger regardless of
        # sandbox/iframe/CSP specifics this environment has no way to test.
        # Not AI: reporting the exact browser error text, twice - the
        # user's own find both times, not visible from this environment's
        # own tools (no browser available here - see Entry 079's note).
        try:
            if not hasattr(st, "pdf"):
                raise RuntimeError("st.pdf not present in this Streamlit version")
            st.pdf(cached["pdf_bytes"])
        except Exception:
            # AI-ASSISTED (Claude Code, chat) - Prompt: "画面太大了 显示一张A4纸
            # 的尺寸" (the image is too big, show it at a page-of-paper
            # size) - width="stretch" filled the entire column, several
            # times wider than a real printed page reads. Centered in a
            # narrower middle column instead of a hardcoded pixel width,
            # so it stays responsive to different screen sizes rather than
            # a fixed size that could be too small or too large elsewhere.
            # AI-ASSISTED (Claude Code, chat) - Prompt: "PDF的preview在页面上
            # 是左右展开的 不是上下的" (the PDF preview should lay out left-right,
            # not up-down) - the original centered/stacked layout (one page,
            # a divider, the next page below it) is replaced with one
            # column per page, side by side, so both pages of the real A4
            # report are visible at once without scrolling past the first.
            # Not AI: the layout direction itself - given directly.
            page_pngs = cached.get("page_pngs", [])
            preview_cols = st.columns(len(page_pngs)) if page_pngs else []
            for col, png_bytes in zip(preview_cols, page_pngs):
                with col:
                    st.image(png_bytes, width="stretch")


def render_diagnostic_analysis(recipes, k, labels, oncourt, bio, roster):
    st.title("Diagnostic Analysis")

    roster_ids = roster["PLAYER_ID"].astype(int).tolist()
    exposure_cache = load_exposure_cache(recipes, k, SEASON)
    screening_df = load_screening_table(tuple(roster_ids), recipes, k, oncourt, SEASON, exposure_cache)

    # AI-ASSISTED (Claude Code, chat)
    # Prompt: "接着修改下一条 针对2 到具体哪个球员的...我想实现的效果就是在screen的
    # 时候如果点击选中哪个球员 球员会有选中效果 下面的screen table这名球员所在的
    # 行会高亮 然后同样的 下面的数据也会显示这名球员的个人信息" - clicking a player
    # in the quadrant chart should visibly highlight him there AND highlight
    # his row in the screening table below, in addition to (already working)
    # Layer 2 showing his card.
    # Used: `selected_pid` is now resolved ONCE, up front, by reading the
    # quadrant chart's own persisted widget state directly
    # (st.session_state["screening_quadrant"]) rather than from
    # render_layer1_screening's return value - a keyed Streamlit widget's
    # selection is already synced into session_state by the time the script
    # starts re-executing on the very rerun a click triggers, so reading it
    # here, before the chart/table are even built, lets both the chart's
    # highlight ring and the table's highlighted row reflect a fresh click on
    # the SAME render pass, instead of lagging one rerun behind (which the
    # previous structure - compute selected_pid only after
    # render_layer1_screening returns - would have caused, since Layer 1
    # would already have finished drawing before the click was accounted for).
    # Not AI: the interaction spec itself (chart highlight + table row
    # highlight, both driven by the same click) - given directly.
    default_row = screening_df[screening_df["PLAYER_ID"] == DEFAULT_DIAG_PLAYER_ID]
    if len(default_row):
        default_pid = int(default_row.iloc[0]["PLAYER_ID"])
    elif len(screening_df):
        default_pid = int(screening_df.iloc[0]["PLAYER_ID"])
    else:
        default_pid = None
    selected_pid = st.session_state.get("diag_selected_pid", default_pid)
    prior_selection = st.session_state.get("screening_quadrant")
    if prior_selection and prior_selection.selection and prior_selection.selection.points:
        pt = prior_selection.selection.points[0]
        cd = pt.get("customdata")
        if cd:
            selected_pid = int(cd[0])
    # AI-ASSISTED (Claude Code, chat)
    # Prompt: "在screning table里点击球员头像也可以更新球员" (clicking a photo
    # in the screening table should also update the selected player) - see
    # render_screening_table's own note for why a query param (not a
    # native widget callback) is the mechanism a raw-HTML iframe table has
    # to signal a selection back to the app.
    # Used: st.query_params is checked AFTER the chart-derived selection so
    # a fresh table click (which always arrives with diag_pid freshly set
    # on that exact rerun) wins over a stale chart selection - then
    # immediately cleared so it acts as a one-shot trigger, not a
    # permanent override that would block a LATER chart click from ever
    # taking effect again.
    # Not AI: the requirement itself - given directly.
    query_pid = st.query_params.get("diag_pid")
    if query_pid is not None:
        selected_pid = int(query_pid)
        del st.query_params["diag_pid"]
    st.session_state["diag_selected_pid"] = selected_pid

    render_layer1_screening(screening_df, labels, selected_pid)

    st.divider()
    if selected_pid is not None:
        render_player_report(selected_pid, recipes, k, labels, bio, exposure_cache)
    else:
        st.caption("No data-eligible roster player to show.")


# AI-ASSISTED (Claude Code, chat)
# Prompt: "Replace the Intro page in portal.py with a geometric explainer.
# The current version uses an animated bar chart; it shows ADA's output
# but not its mechanism. Archetypoids are vertices of the convex hull of
# the player point cloud, and that is what the page should show." Full
# spec: PCA fit on the K archetypoid rows only (never on all players -
# explicitly flagged as a correctness requirement, not a style choice,
# since a hull vertex in ~29-dim space need not sit on the boundary of a
# PCA plane fit to everyone), report explained variance of that fit and
# how many of the K archetypoids land on the 2D ConvexHull of the
# projected points, a K=4-10 slider backed by bases precomputed offline
# (precompute_hull_bases.py - ADA is too slow to refit live), hover with
# player/team/minutes/top-archetype on all points, and a click-driven
# detail panel (added in a follow-up message) reusing render_profile_card/
# load_player_bio/archetype_column_chart as-is, showing an archetypoid's
# own high/low feature z-scores when the clicked point IS one of the K.
# Used: see compute_hull_projection / build_hull_scatter / render_hull_
# detail_panel below - both fully implement the spec above, including the
# 2D-projection caveat caption and the <6-on-hull honesty warning.
# Not AI: the geometric framing itself (archetypoids as convex-hull
# vertices, not abstract cluster centers), requiring the PCA-fit-on-
# archetypoids-only correctness constraint, and every interaction spec
# (hover vs. click, what the detail panel must show, visual encoding per
# point group) - all specified directly, not proposed by Claude Code.
PAPER_URL = (
    "https://cdn.prod.website-files.com/68d6be744d7efccc2207f571/"
    "699f0e4d8c6ef2a7ca76022e_Scouting%20Anyone_%20Probabilistic%20"
    "Player%20Archetypes%20for%20Any%20League.pdf"
)

@st.cache_data
def load_hull_population(season=SEASON, min_threshold=300):
    """Raw (not yet z-scored) feature rows for every eligible 2025-26
    player - reuses step1's own load_population, the exact population its
    bases are fit/projected on, rather than re-deriving a feature matrix a
    second way."""
    return load_ada_population(min_threshold=min_threshold, season_min=season, season_max=season)


@st.cache_data
def load_hull_basis(k):
    """K=8 reuses this project's one authoritative production basis
    (data/basis_2025_26) rather than a second K=8 fit, so the slider's K=8
    position always matches every other page in the portal. K=4-7/9-10
    load precompute_hull_bases.py's offline output - never refit here."""
    path = BASIS_DIR if k == 8 else HULL_BASES_DIR / f"k{k}"
    return load_ada_basis(path)


@st.cache_data
def load_hull_archetype_defs(k):
    path = BASIS_DIR if k == 8 else HULL_BASES_DIR / f"k{k}"
    return pd.read_csv(path / "archetype_definitions.csv")


def _resolve_row_by_id(pop, player_id, min_hint=None):
    """Match a PLAYER_ID back to its row position in `pop` - never by name
    (this project's own past name-collision bugs). A midseason-traded
    player can have more than one row this season (step1's own Part E
    note), so when min_hint is given, disambiguate by whichever row's MIN
    is closest rather than silently taking the first match."""
    cand = pop.index[pop["PLAYER_ID"].astype(int) == int(player_id)]
    if len(cand) == 0:
        return None
    if len(cand) == 1 or min_hint is None:
        return int(cand[0])
    mins = pop.loc[cand, "MIN"].values
    return int(cand[int(np.argmin(np.abs(mins - min_hint)))])


@st.cache_data(show_spinner="Projecting the player cloud onto the archetypoid plane...")
def compute_hull_projection(k):
    """Everything the hull scatter needs for one K: the 2D PCA plane
    (fit on the K archetypoid rows only - see the module note above for
    why fitting it on all players would be wrong, not just a different
    choice), every player's position on that plane, their own simplex
    recipe at this K (project(), not a refit), and the honesty check of
    how many archetypoids actually land on the 2D hull of the projection.
    """
    fit = load_hull_basis(k)
    defs = load_hull_archetype_defs(k)
    pop = load_hull_population().reset_index(drop=True)

    feature_cols = fit["feature_columns"]
    X = pop[feature_cols].astype(float).values
    X_z = (X - fit["mu"]) / fit["sd"]
    basis = np.asarray(fit["basis"], dtype=float)

    pca = PCA(n_components=2)
    pca.fit(basis)                 # fit on the K archetypoids ONLY
    basis_2d = pca.transform(basis)
    all_2d = pca.transform(X_z)    # then transform everyone onto that plane

    P = ada_project(pop, fit["basis"], fit["mu"], fit["sd"], feature_cols)

    archetype_row_idx = []
    for _, d in defs.iterrows():
        archetype_row_idx.append(_resolve_row_by_id(pop, int(d["PLAYER_ID"]), float(d["MIN"])))

    hull_all = ConvexHull(all_2d)
    hull_vertex_set = set(hull_all.vertices.tolist())
    n_on_hull = sum(1 for idx in archetype_row_idx if idx is not None and idx in hull_vertex_set)

    hull_of_archetypoids = ConvexHull(basis_2d)
    outline_idx = list(hull_of_archetypoids.vertices) + [hull_of_archetypoids.vertices[0]]

    return {
        "k": fit["k"], "fit": fit, "defs": defs, "pop": pop,
        "all_2d": all_2d, "basis_2d": basis_2d, "P": P,
        "explained_variance_ratio": pca.explained_variance_ratio_,
        "archetype_row_idx": archetype_row_idx, "n_on_hull": n_on_hull,
        "outline_idx": outline_idx,
    }


# build_hull_scatter (the Plotly-native archetypoid scatter + Nets layout-
# image markers) and its archetypoid-list companion column were replaced by
# hull_callout_chart.py's leader-line-callout component (see
# build_intro_hull_html below) - superseded, not layered alongside; kept in
# git history, not this file.


# AI-ASSISTED (Claude Code, chat)
# Prompt: "把intro的这部分改造一下 就是这里list的是每个brooklyn nets球员的的profile
# image + name + archtype mixture. Archtype mixture用三个非常小的柱状图 然后把现有
# 的brooklyn nets的球员都列举在这里" - replace the click-to-select single-player
# detail panel with a persistent gallery of every current Nets player (photo +
# name + a compact 3-bar top-archetype chart), always visible rather than
# gated behind clicking a specific point.
# Used: mini_archetype_bars() is a deliberately separate, smaller chart
# (11-12px fonts, tight margins) from archetype_column_chart - reusing the
# big chart's own defaults at a squeezed-down height left overlapping labels
# at gallery scale. render_nets_roster_gallery reuses circular_avatar (not
# render_profile_card, which shows a full bio block - more than the "photo +
# name + mixture" spec asked for) and resolves each Nets player's row via
# _resolve_row_by_id (PLAYER_ID, never name) into whichever K's projection
# (proj["P"]) is currently active, so the gallery updates with the slider
# like the rest of the page rather than staying pinned to one K.
# Not AI: the decision to replace (not add alongside) the click-driven panel,
# and top-3-only (not the full K-length recipe) per player - both the user's
# own specification.


# AI-ASSISTED (Claude Code, chat)
# Prompt (this revision): "NCAA桥接方案先不做...这里的排列方式很丑 我觉得可以一行
# 放两个球员 然后每个Mixture archtype用灰色 颜色从深到浅 并且使用比较短的长度
# 不是这种占据一长整行 并且球员name不要用白色" - four fixes to the now-16-player
# gallery: (1) two players per row instead of one, since the taller roster made
# the single-column layout feel sparse/uglier; (2) drop the per-archetype
# identity-color scheme (Entry above) for a plain rank-ordered grey ramp -
# dark=highest share, light=lowest, within each player's own top 3; (3) shorter
# bars (the chart no longer needs to fill a wide single-column chart_col now
# that two players share a row); (4) the player name was rendering
# near-invisible against the cream background - the file's global CSS rule
# only targets stMarkdownContainer p/span/li (line ~110), not the raw <div>/<b>
# this row injects, so it fell through to some unstyled default instead of
# BL_INK - fixed by setting color explicitly on the injected HTML rather than
# relying on inheritance.
# Used: loaded the project's dataviz skill before touching chart color, since
# this is exactly a magnitude encoding within one player's row (a sequential
# job, not identity) - per that skill's rule ("sequential = one hue,
# light->dark"), replaced ARCHETYPE_COLOR_PALETTE/_hex_to_rgba (both now dead
# code, removed - nothing else in the file used them) with a fixed 3-step grey
# ramp (GRAY_RAMP_TOP3), one shade per rank position since bars are already
# sorted high to low. Checked the ramp's own contrast against BL_PAPER before
# picking hex values (relative-luminance contrast ratios: darkest 8.7:1,
# mid 3.4:1, lightest 1.5:1 against the paper background) - the lightest bar's
# contrast against the cream background is low enough on its own that it could
# nearly disappear, so added a BL_MUTED 1px outline (4.5:1 contrast) to every
# bar regardless of fill shade, keeping even the lightest bar's edge visible.
# Two-per-row uses a plain st.columns(2) outer split with the existing
# name+chart inner columns nested one level inside each - Streamlit supports
# one level of column nesting.
# Not AI: the four specific complaints and the two-per-row layout choice -
# all the user's own reaction to the rendered screenshot.

# AI-ASSISTED (Claude Code, chat)
# Prompt (this revision): "我改变想法了...我想在这显示一个表格 去掉这个filter 然后
# 按照球员的首字母排序 然后按照这些column显示 player_bio + player_base 这两个表格
# ...然后最右侧的显示一列是mixture type 并且使用从左到右 从高到低的...vertical
# histogram plot 长度和颜色深度表示该archtype的占比 只显示top 3 archtype" - drop
# the card gallery (two-per-row photo+bars) for an actual data table: one row
# per player, alphabetical by name, no filter control, columns drawn from
# player_bio + player_base (step0's two raw source tables, not the engineered
# build_nba_side_tables() feature set), plus a rightmost "Mixture" column - 3
# small bar segments placed left to right (highest share first), grey shade
# and length both encoding the percentage, top-3 archetypes only.
# Used: st.table's own plain-HTML-table implementation (already this file's
# stated convention for styleable tables, per the comment above the
# stTable/th/td CSS rules) can't hold a custom per-cell widget - a data
# table's real value here is column density, and st.table only accepts plain
# cell values, not embedded bar charts. Built the whole thing as one raw HTML
# <table> via st.markdown(unsafe_allow_html=True) instead, giving full control
# over the mixture cell's inline flex-box bar segments while everything else
# renders as an ordinary table cell. Picked a moderate column set from the two
# named tables rather than every column either exposes - player_bio's ~40
# *_RANK columns and player_base's team-level W/L/W_PCT aren't per-player
# information worth a column in a 16-row table; DRAFT_YEAR/ROUND/NUMBER
# collapse into one "Draft" string rather than three near-empty columns.
# Mixture bar widths scale against ONE shared reference (the single largest
# top-1 percentage across the whole roster) rather than each player's own max
# (the card version's convention) - a shared table column should let 56% read
# visibly longer than 28% across rows, not just within one player's own three
# bars. Reused GRAY_RAMP_TOP3 (same rank-based dark->light grey, same BL_MUTED
# outline) from the card version for visual continuity rather than a new
# scheme. Each segment carries a `title` hover tooltip with the exact
# archetype name + percentage rather than inline text, since three full
# archetype names (e.g. "Rim Protector / Roll Man") would blow out the row
# height/column width of what's supposed to be a compact table cell.
# Not AI: the pivot from card gallery to data table, dropping the filter, sort
# order, and the two named source tables to draw columns from - all the
# user's own change of direction, not a Claude Code design choice.

GRAY_RAMP_TOP3 = ["#3f454d", "#7c848c", "#c3c9cf"]  # rank 0 (highest share) -> rank 2, dark -> light
MIXTURE_BAR_MAX_PX = 70
MIXTURE_BAR_MIN_PX = 5


# AI-ASSISTED (Claude Code, chat)
# Prompt (this revision): "1,这里只显示了16个球员 3个rookie有bio的数据就填写在这里
# 没有就用 - / 2,不需要显示College Country Draft / 3,Mixture archtype没有显示数字
# + archtype name 你可以在这自己设计一下怎么样的显示方案更好" - three fixes: (1)
# show all 19 roster players, not just the 16 with a fitted recipe - the 3 true
# rookies get a row with '-' in every column that has no data for them, rather
# than being silently dropped; (2) drop College/Country/Draft; (3) the hover-
# only mixture cell didn't satisfy "show the number + archetype name" - given
# free rein to redesign, switched from 3 side-by-side bars (compact but
# text-only-on-hover) to 3 STACKED bar+label rows (still dark->light by rank,
# still bar length ~ percentage) since removing 3 columns freed enough cell
# width to show "56% Combo Guard" etc. directly, without needing to shrink
# archetype names or rely on a tooltip nobody has to discover.
# Used: switched the row-building loop from `roster["PLAYER_ID"]` (already
# filtered to the 16 by resolve_roster()) to the full 19-name NETS_ROSTER list,
# resolving each name to a PLAYER_ID via player_bio's own name column
# (_normalize_name, same convention step3's resolve_roster already uses to
# turn a human roster name into a PLAYER_ID - the one place name-matching is
# unavoidable, since that's the actual identity handoff point) rather than
# reusing pop's PLAYER_NAME column a second time - once a PLAYER_ID is known,
# archetype-row lookup still goes through _resolve_row_by_id (PLAYER_ID only,
# never a second name match) exactly as before, so the "never resolve
# archetypes by name" rule this file's own comments call out isn't relaxed.
# A name with zero match anywhere (Mikel Brown Jr., Tyler Bilodeau - confirmed
# zero rows in player_bio/player_base at any season, not just this one) gets
# player_id=None and every data cell renders '-'.
# Not AI: the three specific complaints and the decision to keep 16 recipe
# columns' full text alongside the bars rather than dropping the bars
# entirely - all the user's own call (bars explicitly not asked to be removed,
# just not TEXT-less anymore).


def _mixture_cell(vals, arch_names, global_max):
    """Top-3 archetype mixture as 3 stacked rows: a small bar (length +
    grey shade, dark->light by rank, both ~ the percentage - GRAY_RAMP_TOP3)
    followed by its visible '56% Combo Guard' label. `vals` is None for a
    player with no fitted recipe at all (the 3 true rookies) - renders '-'.
    Bar width scales against `global_max` (the single largest top-1 share
    anywhere in the table) so bars stay comparable across rows."""
    if vals is None:
        return "—"
    top3 = np.argsort(vals)[::-1][:3]
    rows_html = []
    for rank, i in enumerate(top3):
        pct = float(vals[i])
        width_px = max(MIXTURE_BAR_MIN_PX, round(pct / global_max * MIXTURE_BAR_MAX_PX))
        color = GRAY_RAMP_TOP3[min(rank, len(GRAY_RAMP_TOP3) - 1)]
        rows_html.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:{width_px}px;min-width:{width_px}px;height:12px;background:{color};'
            f'border:1px solid {BL_MUTED};border-radius:3px;"></div>'
            f'<span style="font-size:12px;color:{BL_INK};white-space:nowrap;">{pct:.0%} {arch_names[i]}</span>'
            f'</div>'
        )
    return f'<div style="display:flex;flex-direction:column;gap:3px;">{"".join(rows_html)}</div>'


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


# ESPN-verified (checked live 2026-07-27, see CLAUDE.md's own roster
# table) - these 3 have ZERO NBA data anywhere in this project's DB
# (confirmed via a direct zero-row query at any season), so there is no
# stats-based source to pull Age/Ht/Wt from. Height stored as both the
# display string and its inches equivalent, matching PLAYER_HEIGHT/
# PLAYER_HEIGHT_INCHES's own two-column convention in the real bio table.
ROOKIE_BIO_FALLBACK = {
    "Mikel Brown Jr.": {"age": 20, "height": "6-5", "height_inches": 77, "weight": 180},
    "Tyler Bilodeau": {"age": 22, "height": "6-8", "height_inches": 80, "weight": 228},
    "Joshua Jefferson": {"age": 22, "height": "6-8", "height_inches": 80, "weight": 246},
}


def render_nets_roster_table(proj, roster_names, labels, bio, base_stats):
    pop, P = proj["pop"], proj["P"]
    k = proj["k"]
    arch_names = [labels.get(i, f"Archetype {i}") if k == 8 else f"Archetype {i}" for i in range(k)]

    bio_by_norm = bio.copy()
    bio_by_norm["_norm"] = bio_by_norm["PLAYER_NAME"].apply(_normalize_name)
    bio_idx = bio.set_index("PLAYER_ID")
    base_idx = base_stats.set_index("PLAYER_ID")

    resolved = {}  # name -> (player_id or None, row_idx or None)
    for name in roster_names:
        match = bio_by_norm[bio_by_norm["_norm"] == _normalize_name(name)]
        if len(match) == 0:
            resolved[name] = (None, None)
            continue
        pid = int(match.iloc[0]["PLAYER_ID"])
        resolved[name] = (pid, _resolve_row_by_id(pop, pid))

    n_with_recipe = sum(1 for _, row_idx in resolved.values() if row_idx is not None)
    st.markdown(f"#### Nets roster at K={k}")
    st.caption(
        f"{len(roster_names)} roster players ({n_with_recipe} with a fitted K={k} recipe - "
        f"the rest are true rookies with no 2025-26 NBA row of any kind, shown with '-' and "
        f"sorted to the bottom by default). Bio columns from player_bio, box score from "
        f"player_base. Mixture: top 3 archetypes, highest share first, bar length and grey "
        f"shade both ~ the percentage. Click a column header to sort."
    )

    global_max = max((float(P[row_idx].max()) for _, row_idx in resolved.values() if row_idx is not None), default=1.0)

    # default order: recipe-having players alphabetically, THEN the no-data
    # rookies alphabetically - two lists concatenated, not one sorted() call,
    # so the rookies land at the bottom rather than interspersed.
    with_recipe = sorted(n for n in roster_names if resolved[n][1] is not None)
    without_recipe = sorted(n for n in roster_names if resolved[n][1] is None)
    ordered_names = with_recipe + without_recipe

    # (header label, data-sort key name) - Photo has no sort key (None).
    columns = [("", None), ("Player", "name"), ("Age", "age"), ("Ht", "ht"), ("Wt", "wt"),
               ("GP", "gp"), ("MIN", "min"), ("PTS", "pts"), ("REB", "reb"), ("AST", "ast"),
               ("STL", "stl"), ("BLK", "blk"), ("FG%", "fg"), ("3P%", "fg3"), ("FT%", "ft"),
               ("Mixture (top 3)", "mix")]

    rows_cells = []
    for name in ordered_names:
        pid, row_idx = resolved[name]
        b = bio_idx.loc[pid] if pid is not None and pid in bio_idx.index else None
        s = base_idx.loc[pid] if pid is not None and pid in base_idx.index else None
        recipe_vals = P[row_idx] if row_idx is not None else None
        # AI-ASSISTED (Claude Code, chat) - Prompt: "Intro Tab, please also
        # include Joshua Jefferson, Mike Brown JR, Tyler Bilodeau, their
        # basic information in the table, Age, Ht, Wt, Profile image,
        # leave - for other columns please." Used: these 3 have ZERO NBA
        # data anywhere in the DB (confirmed via a direct zero-row query -
        # see CLAUDE.md), so `bio`/`base_stats` have nothing to look up by
        # player_id - `b` is None for all three, same as before. Their
        # Age/Ht/Wt are hardcoded from the SAME ESPN-verified source
        # CLAUDE.md's own roster table already cites (checked 2026-07-27),
        # not re-derived here. GP/MIN/PTS/etc. and Mixture are left
        # completely untouched (still "—") - only Age/Ht/Wt/Photo change
        # for these 3 specific names, exactly as asked.
        # Not AI: the request itself, and which 3 names - given directly.
        rookie_fallback = ROOKIE_BIO_FALLBACK.get(name) if b is None else None

        if pid is not None:
            photo_html = (
                f'<img src="{HEADSHOT_URL.format(player_id=pid)}" style="width:36px;height:36px;'
                f'border-radius:50%;object-fit:cover;object-position:top center;'
                f'border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};display:block;">'
            )
        elif rookie_fallback is not None:
            # get_headshot_data_uri(None, name) always returns the graceful
            # initials-SVG data URI (no player_id -> no network attempt at
            # all, per its own docstring) - a real "profile image" (colored
            # circle + initials) instead of the blank grey dot every other
            # unmatched name still gets below.
            photo_html = (
                f'<img src="{hull_callout_chart.get_headshot_data_uri(None, name)}" '
                f'style="width:36px;height:36px;border-radius:50%;object-fit:cover;'
                f'border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};display:block;">'
            )
        else:
            photo_html = (
                f'<div style="width:36px;height:36px;border-radius:50%;background:{BL_LINE};'
                f'border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};"></div>'
            )

        mix_sort = float(recipe_vals.max()) if recipe_vals is not None else -1.0
        if b is not None:
            age_cell = (f"{b['AGE']:.0f}", float(b["AGE"]))
            ht_cell = (b["PLAYER_HEIGHT"], float(b["PLAYER_HEIGHT_INCHES"]))
            wt_cell = ((f"{b['PLAYER_WEIGHT']} lbs", float(b["PLAYER_WEIGHT"]))
                      if pd.notna(b["PLAYER_WEIGHT"]) else ("—", -1.0))
        elif rookie_fallback is not None:
            age_cell = (str(rookie_fallback["age"]), float(rookie_fallback["age"]))
            ht_cell = (rookie_fallback["height"], float(rookie_fallback["height_inches"]))
            wt_cell = (f"{rookie_fallback['weight']} lbs", float(rookie_fallback["weight"]))
        else:
            age_cell = ht_cell = wt_cell = ("—", -1.0)

        rows_cells.append([
            (photo_html, None),
            (f"<b>{name}</b>", name.lower()),
            age_cell,
            ht_cell,
            wt_cell,
            (f"{s['GP']:.0f}" if s is not None else "—", float(s["GP"]) if s is not None else -1.0),
            (f"{s['MIN']:.0f}" if s is not None else "—", float(s["MIN"]) if s is not None else -1.0),
            (f"{s['PTS']:.0f}" if s is not None else "—", float(s["PTS"]) if s is not None else -1.0),
            (f"{s['REB']:.0f}" if s is not None else "—", float(s["REB"]) if s is not None else -1.0),
            (f"{s['AST']:.0f}" if s is not None else "—", float(s["AST"]) if s is not None else -1.0),
            (f"{s['STL']:.0f}" if s is not None else "—", float(s["STL"]) if s is not None else -1.0),
            (f"{s['BLK']:.0f}" if s is not None else "—", float(s["BLK"]) if s is not None else -1.0),
            (f"{s['FG_PCT']:.1%}" if s is not None else "—", float(s["FG_PCT"]) if s is not None else -1.0),
            (f"{s['FG3_PCT']:.1%}" if s is not None else "—", float(s["FG3_PCT"]) if s is not None else -1.0),
            (f"{s['FT_PCT']:.1%}" if s is not None else "—", float(s["FT_PCT"]) if s is not None else -1.0),
            (_mixture_cell(recipe_vals, arch_names, global_max), mix_sort),
        ])

    table_html, iframe_height = _build_sortable_table_html("roster_table", columns, rows_cells)
    components.html(table_html, height=iframe_height, scrolling=True)


def render_ev_vs_k_chart(summary, chosen_k, height=280):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=summary["k"], y=summary["explained_var"], mode="lines+markers",
        line=dict(color=BL_GREEN, width=2), marker=dict(size=7, color=BL_GREEN),
        hovertemplate="K=%{x}: %{y:.1%} explained variance<extra></extra>",
    ))
    chosen = summary[summary["k"] == chosen_k]
    if len(chosen):
        fig.add_trace(go.Scatter(
            x=chosen["k"], y=chosen["explained_var"], mode="markers",
            marker=dict(size=15, color=BL_CORAL, line=dict(width=2, color=BL_WHITE)),
            hoverinfo="skip", showlegend=False,
        ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(title="K", tickmode="linear", tickfont=dict(color=BL_INK, size=13)),
        yaxis=dict(title="explained variance", tickformat=".0%", tickfont=dict(color=BL_INK, size=13)),
    )
    return fig


# AI-ASSISTED (Claude Code, chat)
# Prompt: full leader-line-callout redesign of the hull scatter, given in
# complete spec (geometry algorithm, colors, hover-card content/style,
# Nets-dot interaction) - implemented in the new hull_callout_chart module;
# this is just the thin Streamlit-side wiring.
# Used: cached on (hull_k, nets_ids_tuple) rather than passing `proj`/`labels`
# straight through to a cached function - both are large (proj holds several
# numpy arrays over the whole ~430-player population) and Streamlit's cache
# would have to hash them on every rerun, whereas hull_k/nets_ids_tuple are
# small and stable, and the expensive pieces (compute_hull_projection,
# load_static) are already independently @st.cache_data'd, so re-deriving
# them inside on a cache miss costs nothing on a cache hit. This is what
# lets an unrelated widget interaction elsewhere on the page (which reruns
# this whole script) skip rebuilding the chart's ~24 embedded base64 photos
# every time - only a real hull_k or roster change does that.
# Not AI: the whole component being replaced (build_hull_scatter + the
# archetypoid list column) - the user's own change of direction.
@st.cache_data
def build_intro_hull_html(hull_k, nets_ids_tuple):
    proj = compute_hull_projection(hull_k)
    _, _, static_labels, _ = load_static()
    use_labels = static_labels if hull_k == 8 else {}
    roster_df = pd.DataFrame({"PLAYER_ID": list(nets_ids_tuple)})
    spec = hull_callout_chart.build_figure_spec(proj, roster_df, use_labels, hull_k)
    return hull_callout_chart.render_html(spec)


def render_intro_page(roster, labels):
    st.title("Intro")

    with st.container(border=True):
        st.markdown("### What is an \"archetype\" here?")
        st.markdown(
            "ADA finds the **K most extreme real players** in the league and expresses "
            "everyone else as a blend of them - not abstract types, real players. Below: "
            "the colored corners are those K archetypes, black dots are Nets players - "
            "hover either for details."
        )

        hull_k = st.slider("Number of archetypes (K)", min_value=min(HULL_K_RANGE),
                           max_value=max(HULL_K_RANGE), value=HULL_DEFAULT_K, step=1)

        if hull_k != 8 and not (HULL_BASES_DIR / f"k{hull_k}").exists():
            st.warning(f"K={hull_k}'s precomputed basis isn't on disk yet - run "
                       f"precompute_hull_bases.py, then reload.")
            return

        proj = compute_hull_projection(hull_k)

        # AI-ASSISTED (Claude Code, chat)
        # Prompt (this revision): "把这些文字都去掉 字太多了" - drop the <6-on-hull
        # warning/caption, the PCA-plane-fit caption, and the "2D projection"
        # caption under the chart entirely - not reworded/shortened, removed.
        # Not AI: which text to cut - the user's own reaction to the rendered
        # page having too much caption text.

        nets_ids_tuple = tuple(sorted(roster["PLAYER_ID"].astype(int).tolist()))
        chart_html = build_intro_hull_html(hull_k, nets_ids_tuple)
        components.html(chart_html, height=int(hull_callout_chart.FIGURE_HEIGHT_PX * 1.05) + 20, scrolling=False)

        # Hidden for now per explicit request ("先不要标注这个论文链接" / don't show
        # the paper link for now) - not removed, just commented out (as '#'
        # lines, not a bare triple-quoted string - a bare string here would
        # trigger Streamlit's own "magic" auto-st.write(), same gotcha as the
        # explained-variance block below).
        # st.markdown(f"[Scouting Anyone: Probabilistic Player Archetypes for Any League]({PAPER_URL})")

    with st.container(border=True):
        bio = load_player_bio(season=SEASON)
        base_stats = load_player_base_stats(season=SEASON)
        render_nets_roster_table(proj, NETS_ROSTER, labels, bio, base_stats)

    # Hidden for now per explicit request ("这部分内容先不显示 先comment掉" / "we
    # don't need to show this image for now") - not removed, just commented
    # out (as '#' lines, not a bare triple-quoted string, since a bare string
    # here would trigger Streamlit's own "magic" auto-st.write() - see the
    # AI_USAGE.md entry for the bug that caused earlier this session).
    # st.divider()
    # st.markdown("#### Explained variance vs. K")
    # st.markdown(
    #     "The player cloud's own boundary is fixed by the data - K only controls how many "
    #     "corners are used to approximate it. This curve shows what each extra corner buys."
    # )
    # if K_SELECTION_PATH.exists():
    #     k_summary = pd.read_csv(K_SELECTION_PATH)
    #     st.plotly_chart(render_ev_vs_k_chart(k_summary, hull_k, height=280), width="stretch")
    # else:
    #     st.caption(f"{K_SELECTION_PATH} not found - run step1_archetypes_model.phase1_select_k() first.")


# AI-ASSISTED (Claude Code, chat)
# Prompt: "改造这个图 shot distance distribution使用court作为背景图画出来
# play type usage 使用pie chart Box Score rates也包含基础的Base stats not
# those advanced" (redesign this: shot distance distribution should be
# drawn over a court background, play type usage should be a pie chart,
# and Box score rates should also include basic Base stats, not just the
# advanced ones) - a follow-up redesign of the same section from Entry 034.
# Used: `build_court_shot_chart`/`build_playtype_pie` (see their own
# docstrings for the concentric-band and top-3+Other design choices);
# `load_player_base_stats` (already existed, used by the roster gallery
# table) reused verbatim as the Base stats source rather than a new query,
# rendered as its own sortable-style table above the existing Advanced one.
# Not AI: the three-part redesign spec - given directly.
def render_player_stats_tab(row):
    with st.container(border=True):
        pid = int(row["PLAYER_ID"])
        full_features = load_full_features(season=SEASON)
        prow_match = full_features[full_features["PLAYER_ID"] == pid]
        if len(prow_match) == 0:
            st.caption(f"No detailed stats found for {row['PLAYER_NAME']} in {SEASON}.")
            return
        prow = prow_match.iloc[0]

        base_stats = load_player_base_stats(season=SEASON)
        base_match = base_stats[base_stats["PLAYER_ID"] == pid]
        base_row = base_match.iloc[0] if len(base_match) else None

        st.markdown("**Box score rates**")
        # AI-ASSISTED (Claude Code, chat)
        # Prompt: "针对diagostic analysis tab, 请把base stats and advanced都放到
        # 一行上" (put Base stats and Advanced on one row) - previously two
        # separate stacked tables (Entry ~036), now merged into a single row
        # since it's one player's full box-score profile either way.
        # Used: base_defs + adv_defs concatenated into one column list, one
        # _build_sortable_table_html call instead of two - values pulled from
        # their own original sources unchanged (base_row for Base stats,
        # prow for Advanced), just rendered together.
        # Not AI: the merge instruction itself - given directly.
        base_defs = [
            ("GP", "GP", lambda v: f"{v:.0f}"),
            ("MIN", "MIN", lambda v: f"{v:.0f}"),
            ("PTS", "PTS", lambda v: f"{v:.0f}"),
            ("REB", "REB", lambda v: f"{v:.0f}"),
            ("AST", "AST", lambda v: f"{v:.0f}"),
            ("STL", "STL", lambda v: f"{v:.0f}"),
            ("BLK", "BLK", lambda v: f"{v:.0f}"),
            ("FG%", "FG_PCT", lambda v: f"{v:.1%}"),
            ("3P%", "FG3_PCT", lambda v: f"{v:.1%}"),
            ("FT%", "FT_PCT", lambda v: f"{v:.1%}"),
        ]
        # NOTE: these columns are NOT all on the same scale - TS% is a
        # fraction (0-1) while USG%/AST%/TRB%/STL%/BLK%/TOV% are already
        # percentage points (0-100) and BPM is a raw rating, not a percent
        # at all. Checked against real values before picking formats, not
        # assumed - formatting all of these as "%. 1%" would have been
        # wrong for most of them.
        adv_defs = [
            ("TS%", "TS%", lambda v: f"{v:.1%}"),
            ("USG%", "USG%", lambda v: f"{v:.1f}%"),
            ("AST%", "AST%", lambda v: f"{v:.1f}%"),
            ("TRB%", "TRB%", lambda v: f"{v:.1f}%"),
            ("STL%", "STL%", lambda v: f"{v:.1f}%"),
            ("BLK%", "BLK%", lambda v: f"{v:.1f}%"),
            ("TOV%", "TOV%", lambda v: f"{v:.1f}%"),
            ("BPM", "BPM", lambda v: f"{v:+.1f}"),
        ]
        combined_cells = [
            (fmt(base_row[col]) if base_row is not None and pd.notna(base_row[col]) else "—", None)
            for label, col, fmt in base_defs
        ] + [
            (fmt(prow.get(col)) if pd.notna(prow.get(col)) else "—", None)
            for label, col, fmt in adv_defs
        ]
        combined_columns = [(label, None) for label, _, _ in base_defs + adv_defs]
        combined_html, combined_height = _build_sortable_table_html(
            "player_stats_combined", combined_columns, [combined_cells], row_height=44)
        components.html(combined_html, height=combined_height, scrolling=True)

        dist_map = [
            ("0-3 ft", "% of FGA by Distance_0-3"), ("3-10 ft", "% of FGA by Distance_3-10"),
            ("10-16 ft", "% of FGA by Distance_10-16"), ("16ft-3P", "% of FGA by Distance_16-3P"),
            ("3PT", "% of FGA by Distance_3P"),
        ]
        dist_pairs = [(label, float(prow.get(col) or 0)) for label, col in dist_map]

        playtype_cols = [c for c in full_features.columns if c.startswith("PLAYTYPE_")]
        pt_all = sorted(
            ((PLAYTYPE_LABELS.get(c, c.replace("PLAYTYPE_", "").replace("_", " ").title()),
              float(prow.get(c) or 0))
             for c in playtype_cols),
            key=lambda p: p[1], reverse=True,
        )
        pt_top3 = pt_all[:3]
        # The true gap to 100%, not just the sum of the other 6 tracked-but-
        # smaller columns - this project's feature set only carries 9 of
        # Synergy's full play-type taxonomy (no Transition/Misc/etc., a
        # documented choice, not missing data), so most players' 9 tracked
        # shares alone sum well under 1.0 (league mean ~0.69, checked
        # directly) - the shortfall is real usage, just outside this
        # project's tracked categories, and needs to land in "Other" for
        # the pie to represent his full shot diet rather than only the
        # tracked slice of it.
        other_pct = max(0.0, 1.0 - sum(v for _, v in pt_all))

        court_col, pie_col = st.columns(2)
        with court_col:
            st.markdown("**Shot distance distribution**")
            st.plotly_chart(build_court_shot_chart(dist_pairs), width="stretch")
        with pie_col:
            st.markdown("**Play type usage** (top 3 + other)")
            st.plotly_chart(build_playtype_pie(pt_top3, other_pct), width="stretch")
            st.caption(
                "'Other' includes both his own smaller tracked play types and Synergy "
                "categories outside this project's 9-type feature set (e.g. Transition, Misc)."
            )


# Orphaned since the Diagnostic Analysis narrative restructure - Section 3
# there now uses similarity_weighted_benchmark (Prompt A) directly, plotting
# both baselines, superseding this find_comparables/league_benchmark_exposure
# (threshold-then-filter-on-outcome) version. Kept, not deleted, same
# "superseded but visible" convention as step2_synergy_matrix.py - not
# wired to a live `result` dict shape anymore, so do not call this without
# updating it first.
def render_league_benchmark_tab(result, top_label):
    with st.container(border=True):
        n_comp = len(result["successful"])
        if result["comparison"] is None:
            st.info(
                f"No league benchmark available for {top_label}: 0 players league-wide at "
                f"≥55% probability in this archetype have a positive on-court net rating "
                f"this season - a real data finding for this (thin) archetype, not a bug."
            )
        else:
            st.markdown(f"{n_comp} successful comparables league-wide at {top_label}: "
                        + ", ".join(result["successful"]["PLAYER_NAME"]))
            comp_fig = diverging_bar(
                result["comparison"], "diff_pp", "archetype", ["league_baseline", "actual"],
                x_title="percentage points vs. league benchmark",
            )
            comp_fig.update_traces(
                hovertemplate=(
                    "%{y}<br>league baseline=%{customdata[0]:.1%}<br>"
                    "actual=%{customdata[1]:.1%}<br>diff=%{x:+}pp<extra></extra>"
                ),
            )
            st.plotly_chart(comp_fig, width="stretch")


# --- Page 2: Roster Construction ---------------------------------------------
# AI-ASSISTED (Claude Code, chat)
# Prompt: wire step4_roster_construction.py's already-validated (real
# cross-validated R^2: 0.312 full vs. 0.295 sum_bpm-only baseline, out-of-
# sample by team) WLS mixture model into this page's lineup-ranking feature,
# after a parallel possession-level skill-weighted archetype-RAPM effort
# (src/fit_rapm.py) failed its own out-of-sample validation gate (Net
# correlation 0.12-0.17 at real sample sizes, not just a small-n fluke - see
# RAPM_README.md's GATE 2 section) and the user chose to redirect here
# instead rather than ship an unvalidated model's roster recommendations.
# Used: `load_lineup_rankings` fits the model + scores every eligible 5-man
# Nets combo in ONE cached call (not two separately-cached functions) since
# the fitted model dict (containing numpy arrays) isn't cleanly hashable for
# a second @st.cache_data layer keyed on it - simplest to keep the model
# internal to one cache entry, keyed only on the actually-hashable inputs
# (the recipes DataFrame - already a working @st.cache_data argument
# elsewhere in this file, e.g. load_exposure_cache - k, season, outcome, and
# a sorted roster-id tuple).
# Not AI: the decision to redirect here - the user's own, made after GATE 2's
# failure was presented directly.


@st.cache_data
def load_bpm_lookup(season):
    df = build_nba_side_tables()
    return df[df["SEASON"] == season].set_index("PLAYER_ID")["BPM"]


@st.cache_data(show_spinner="Fitting the lineup outcome model and scoring every 5-man combination...")
def load_lineup_rankings(recipes, k, season, roster_ids, outcome="NET_RATING",
                         min_threshold=100, combo_size=5):
    from step4_roster_construction import (
        build_lineup_table, fit_outcome_model, cross_validate,
        enumerate_roster_combos, score_combos,
    )
    lineup_df = build_lineup_table(recipes, k, season, group_quantity=4, min_threshold=min_threshold)
    model = fit_outcome_model(lineup_df, outcome, k)
    cv = cross_validate(lineup_df, outcome, k, reference_archetype=model["reference_archetype"])
    bpm_lookup = load_bpm_lookup(season)
    combos = enumerate_roster_combos(list(roster_ids), recipes, season, k, bpm_lookup, combo_size=combo_size)
    scored = score_combos(model, combos, k)
    return model, cv, scored


def render_lineup_rankings(recipes, k, labels, roster, team_profile):
    st.divider()
    st.markdown("### Lineup rankings")
    st.caption(
        "A separate, simpler model from the composition analysis above: fits "
        "outcome ~ sum_bpm + archetype shares + entropy on real 4-man, MIN≥100, "
        "2025-26 lineups pooled across all 30 teams, cross-validates it (held-out "
        "teams, not held-out rows), then applies the fitted coefficients to every "
        "eligible 5-man combination from the Nets' own roster. (A more granular, "
        "possession-level version of this idea was also tried and did **not** pass "
        "its own out-of-sample validation check - see RAPM_README.md's GATE 2 "
        "section - so this simpler, already-validated model is what's shown here, "
        "not the more sophisticated one.)"
    )

    OUTCOME_LABELS = {
        "NET_RATING": "Net rating", "OFF_RATING": "Offensive rating",
        "DEF_RATING_INV": "Defensive rating (higher = better)",
    }
    outcome = st.selectbox("Outcome", list(OUTCOME_LABELS.keys()),
                           format_func=lambda o: OUTCOME_LABELS[o], key="lineup_rank_outcome")

    roster_ids = tuple(sorted(roster["PLAYER_ID"].astype(int).tolist()))
    model, cv, scored = load_lineup_rankings(recipes, k, SEASON, roster_ids, outcome=outcome)

    beats = "beats" if cv["full_beats_baseline"] else "does NOT beat"
    st.markdown(
        f"Cross-validated (out-of-sample, held out by team) R²: **{cv['r2_full']:.3f}** "
        f"full model vs. **{cv['r2_baseline']:.3f}** sum_bpm-only baseline — archetype "
        f"composition **{beats}** a pure-talent baseline out of sample "
        f"(n={model['n']} real 4-man lineups league-wide)."
    )

    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    ref_label = arch_names[model["reference_archetype"]]
    st.caption(
        f"Coefficients below are relative to **{ref_label}** (this table's reference "
        f"archetype - the one soaking up the most minutes in the training data); "
        f"95% CI from a 1000-draw team-cluster bootstrap."
    )
    coef_rows = []
    for name, c, lo, hi in zip(model["feature_names"], model["coef"], model["ci_lower"], model["ci_upper"]):
        if not name.endswith("_share"):
            continue
        arch_i = int(name.split("_")[1])
        coef_rows.append({"archetype": arch_names[arch_i], "coef": c, "ci_lower": lo, "ci_upper": hi})
    coef_df = pd.DataFrame(coef_rows)
    coef_fig = diverging_bar(coef_df, "coef", "archetype", ["ci_lower", "ci_upper"],
                             x_title=f"coefficient vs. {ref_label} (holding sum_bpm, entropy fixed)")
    coef_fig.update_traces(
        hovertemplate="%{y}: %{x:+.2f} (95%% CI %{customdata[0]:+.2f} to %{customdata[1]:+.2f})<extra></extra>",
    )
    st.plotly_chart(coef_fig, width="stretch")

    id_to_name = dict(zip(recipes["PLAYER_ID"].astype(int), recipes["PLAYER_NAME"]))

    def _names(ids):
        return ", ".join(id_to_name.get(int(i), str(i)) for i in ids)

    st.markdown(f"**Top 10 of {len(scored)} eligible 5-man combinations**")
    columns = [("Lineup", "lineup"), ("Predicted", "pred"), ("Sum BPM", "bpm"), ("Entropy", "ent")]
    rows_cells = []
    for _, r in scored.head(10).iterrows():
        names = _names(r["player_ids"])
        rows_cells.append([
            (names, names.lower()),
            (f"{r['predicted']:+.1f}", float(r["predicted"])),
            (f"{r['sum_bpm']:+.1f}", float(r["sum_bpm"])),
            (f"{r['entropy']:.2f}", float(r["entropy"])),
        ])
    html, height = _build_sortable_table_html("lineup_rankings_table", columns, rows_cells, row_height=40)
    components.html(html, height=height, scrolling=True)

    with st.expander(f"Bottom 5 of {len(scored)}"):
        bottom_rows_cells = []
        for _, r in scored.tail(5).iterrows():
            names = _names(r["player_ids"])
            bottom_rows_cells.append([
                (names, names.lower()),
                (f"{r['predicted']:+.1f}", float(r["predicted"])),
                (f"{r['sum_bpm']:+.1f}", float(r["sum_bpm"])),
                (f"{r['entropy']:.2f}", float(r["entropy"])),
            ])
        bottom_html, bottom_height = _build_sortable_table_html(
            "lineup_rankings_bottom_table", columns, bottom_rows_cells, row_height=40)
        components.html(bottom_html, height=bottom_height, scrolling=True)

    st.divider()
    st.markdown("**What if we shifted the team's overall composition?**")
    st.caption(
        "Holds the roster's current talent (sum_bpm) fixed and shifts weight between two "
        "archetypes in the TEAM'S overall composition (the same mixture shown at the top of "
        "this page) - a coach-facing 'what if we leaned more into X and less into Y' query, "
        "using this model's own fitted coefficients."
    )
    col1, col2, col3 = st.columns([2, 2, 1.6])
    with col1:
        from_label = st.selectbox("Shift weight FROM", arch_names, index=int(np.argmax(team_profile["soft_mean"])),
                                  key="sub_from")
    with col2:
        to_options = [a for a in arch_names if a != from_label]
        to_label = st.selectbox("Shift weight TO", to_options, index=0, key="sub_to")
    with col3:
        delta = st.slider("Amount shifted", min_value=0.05, max_value=0.30, value=0.10, step=0.05, key="sub_delta")

    from step4_roster_construction import predict_substitution
    from_idx, to_idx = arch_names.index(from_label), arch_names.index(to_label)
    try:
        sub = predict_substitution(model, team_profile["soft_mean"], from_idx, to_idx, delta=delta)
        st.markdown(
            f"Shifting **{delta:.0%}** of team weight from **{from_label}** to **{to_label}**: "
            f"predicted **{sub['delta_predicted']:+.2f}** change in {OUTCOME_LABELS[outcome].lower()} "
            f"(entropy {sub['entropy_before']:.2f} → {sub['entropy_after']:.2f})."
        )
    except ValueError as e:
        st.warning(str(e))


# AI-ASSISTED (Claude Code, chat)
# Prompt: "在这里解释一下我尝试做regression model 但是失败了 并且详细解释我都做出了
# 哪些努力" (explain here that I tried to build a regression model but it failed,
# and explain in detail what efforts were made) - after Roster Construction moved
# to the last nav slot.
# Used: condenses docs/RESEARCH_FINDINGS.md's full chronological evidence chain
# (already written, every number already verified during that investigation) into
# a portal-readable expander, rather than just the one-line mention already in
# render_lineup_rankings' own caption - reuses the SAME real numbers, not new ones.
# Not AI: the investigation itself and its numbers - already done; only the
# decision to surface it here, and its placement (right before the lineup-ranking
# section it explains the absence of a fancier version of), were made now.
def render_rapm_investigation_note():
    st.markdown("**What actually happened when it was tested**")
    st.caption(
        "It failed its own validation gate, and shipping it anyway would have "
        "meant presenting numbers that could genuinely mislead a real personnel "
        "decision - so no page in this portal uses it. Full detail and every "
        "number: `RAPM_README.md` and `docs/RESEARCH_FINDINGS.md` in the repo."
    )
    with st.expander("What was tried, and why it didn't hold up (7 angles, all tested against real data)"):
        st.markdown(
            "**The validation gate**: compare the model's predicted Net rating for "
            "real Brooklyn lineups it never saw in training against their actual "
            "observed Net rating. Bar: correlation ≥ 0.3. **Result: 0.12–0.21 "
            "depending on sample size — fails.** True out-of-sample R² was ≈0.04 "
            "(worse than the correlation alone implied): predicted Net barely "
            "varied across different real lineups (std≈5.4) while real lineups' "
            "actual Net varies enormously (std≈28.9)."
        )
        st.markdown("Seven specific, falsifiable explanations were tested, one at a time, against real data:")
        rows = [
            ("Model too complex (56 interaction terms overfitting)",
             "Simplified to 8 features, no interactions", "corr = 0.131 — worse, not better"),
            ("Stints too short (~3 possessions) — filter them",
             "Swept a minimum-possession floor 1→16", "Flat at 0.20–0.22 the whole way — no effect"),
            ("Skill-weighting itself is the problem",
             "Refit with every player's skill set to 1.0", "corr = 0.199 — statistically the same"),
            ("Not enough data (1 season)",
             "Pulled 2 more full seasons (2023-24, 2024-25)", "corr = 0.204 pooled — no improvement"),
            ("Net rating specifically is too noisy a target",
             "Refit on eFG% and turnover rate instead", "Same weak range (corr 0.13–0.15)"),
            ("Is the target even measurable at this sample size?",
             "Split a real lineup's own possessions randomly in half, correlated the two halves against each other", "**-0.43 to -0.10 — at or below zero.** A lineup's own Net rating doesn't correlate with an independent random half of itself."),
            ("Does it work at the least noisy level — team-season?",
             "30 teams × 3 seasons vs. real official NET_RATING", "corr = 0.770 — but a talent-only baseline (no archetype/synergy at all) scores 0.767, and wins outright in 2 of 3 seasons"),
        ]
        columns = [("Hypothesis", "h"), ("What was actually tried", "t"), ("Result", "r")]
        rows_cells = [[(h, h), (t, t), (r, r)] for h, t, r in rows]
        html, height = _build_sortable_table_html("rapm_investigation_table", columns, rows_cells, row_height=64)
        components.html(html, height=height, scrolling=True)
        st.markdown(
            "**Conclusion**: six independent angles, plus the most favorable "
            "aggregation level tested, all converge on the same answer — "
            "archetype composition and pairwise synergy show no detectable "
            "predictive increment over individual talent at this data scale, at "
            "any granularity tested. This doesn't mean lineup fit doesn't matter "
            "on the court - it means this specific approach, at this data volume, "
            "couldn't detect it if it's smaller than the noise floor found above. "
            "Archetype recipes remain a validated **descriptive** tool (the "
            "composition/gap/conflict analysis on this page) - just not, so far, "
            "a validated **predictor** of lineup outcomes on their own."
        )


# AI-ASSISTED (Claude Code, chat)
# Prompt: "或者我把Roster Constructio改成Future Work and obstacle 然后阐述一下我
# 当时是怎么构想建模的 但是怎么失败的" (or, change Roster Construction into "Future
# Work and Obstacle" and explain how I originally envisioned building the model,
# but how it failed) - superseded the earlier "hide Roster Construction" request
# with repurposing the tab instead of just hiding it.
# Used: a new top-level page telling the full story in order - what was
# envisioned and why (citing the same precedent CLAUDE.md itself cites for the
# base archetype model, and naming the synergy/RAPM extension as this project's
# own idea beyond that paper), the model that was actually built, then reuses
# render_rapm_investigation_note() verbatim for the failure + 7-hypothesis
# investigation (same real numbers already verified in Entries 045-048, not
# duplicated/re-derived) - and a closing section distinguishing "abandoned" from
# "genuinely still worth trying," reusing the exact same prioritized list already
# discussed and evaluated in chat, not a new brainstorm.
# Not AI: the decision to repurpose the tab instead of hiding it, and the
# structure requested (vision -> failure -> effort in detail) - given directly.
def render_future_work():
    st.title("Future Work & Obstacles")
    st.caption(
        "An honest account of a real research attempt that didn't pan out - kept "
        "visible rather than deleted, since methodology honesty (stating what "
        "was tried and what the data actually showed) is worth more here than "
        "only showing what worked."
    )

    st.markdown("### The original idea")
    st.markdown(
        "This project's archetype model (the K=8 recipes used throughout this "
        "portal) follows *\"Scouting Anyone: Probabilistic Player Archetypes for "
        "Any League\"* (SSAC 2026). That paper stops at describing player style. "
        "**The idea tried here, beyond the paper**: if archetype composition "
        "captures real basketball roles, then which *pairs* of archetypes share "
        "the floor should predict how well a lineup actually performs - beyond "
        "what individual talent alone would predict. Concretely: does putting a "
        "shot-creating guard next to a rim-running big produce better possessions "
        "than the same two players' individual talent would suggest on its own? "
        "That's a real, testable hypothesis, and it's what the rest of this page "
        "is about."
    )

    st.markdown("### The model actually built")
    st.markdown(
        "A possession-level, skill-weighted archetype-RAPM: for every one of "
        "75,587 real 2025-26 stints, a feature vector of talent (`Soff`/`Sdef`) "
        "+ skill-weighted archetype exposure per side (8+8 dims) + all 56 "
        "within-side archetype-*pair* interaction terms, fit by a differentially-"
        "penalized ridge regression (talent unpenalized, archetype tilt + "
        "synergy penalized), with the skill values themselves required to be "
        "exogenous (prior-season only, leakage-checked) and Brooklyn's own "
        "possessions held out entirely from training - so evaluating Brooklyn's "
        "own lineups would be genuinely out-of-sample, not circular."
    )

    st.divider()
    render_rapm_investigation_note()

    st.divider()
    st.markdown("### What's actually still worth trying (and what isn't)")
    st.markdown(
        "Not every angle here has equal promise - ranked by what's actually "
        "worth spending more time on, not just listed:\n\n"
        "1. **Predict a more process-level outcome** (partially tried - eFG%/TOV% "
        "were tested and landed in the same weak range as Net rating at the "
        "lineup level, so this specific pair didn't pan out, but a narrower, "
        "theory-driven process metric closer to what archetype composition "
        "should mechanically affect hasn't been ruled out).\n"
        "2. **A few specific, theory-driven archetype-pair hypotheses** (not all "
        "28+28 interaction terms at once) - lower statistical burden than the "
        "full interaction matrix, but requires committing to a hypothesis "
        "*before* looking at results, not after.\n"
        "3. **More seasons of data** - already tested (3 seasons vs. 1) at both "
        "the lineup and team-season level; didn't move the needle either way. "
        "**Not recommended as a next step** - the evidence points at the "
        "approach, not the data volume.\n\n"
        "What this page does *not* do: claim the underlying basketball idea is "
        "wrong. It states plainly what this specific approach, at this specific "
        "data scale, could and couldn't detect - and lets the archetype recipes "
        "keep doing the job they're actually validated for (Diagnostic Analysis, "
        "Scouting) without overstating what they can predict."
    )


def render_roster_construction(roster, recipes, k, labels):
    st.title("Roster Construction")

    team_profile = compute_team_archetype_profile(roster["PLAYER_ID"].tolist(), recipes, k)
    baseline = compute_league_baseline(recipes, k, season=SEASON)
    gaps = compute_archetype_gaps(team_profile, baseline, labels, k)

    st.markdown(f"**Team composition** — {team_profile['n_players']} data-eligible players, "
               f"soft archetype mass (sums to {team_profile['n_players']})")
    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    team_fig = bubble_chart(arch_names, team_profile["soft_mean"],
                            dominant_idx=int(np.argmax(team_profile["soft_mean"])))
    st.plotly_chart(team_fig, width="content")

    st.markdown("**Gaps & redundancies vs. the 30-team league baseline** "
               "(z-score of the Nets' share in each archetype vs. how much teams normally vary)")
    gap_fig = diverging_bar(gaps, "z", "archetype", ["team_soft_mean", "league_mean", "team_hard_count"],
                            x_title="z vs. league (std devs)")
    gap_fig.update_traces(
        hovertemplate=(
            "%{y}<br>team share=%{customdata[0]:.1%} (%{customdata[2]:.0f} players)<br>"
            "league avg=%{customdata[1]:.1%}<br>z=%{x:+}<extra></extra>"
        ),
    )
    st.plotly_chart(gap_fig, width="stretch")

    biggest_gap = gaps.iloc[0]
    biggest_redundancy = gaps.iloc[-1]
    st.markdown(
        f"Biggest gap: **{biggest_gap['archetype']}** ({biggest_gap['z']:+.2f}z, "
        f"{biggest_gap['team_hard_count']:.0f} players) &nbsp;·&nbsp; "
        f"Biggest redundancy: **{biggest_redundancy['archetype']}** ({biggest_redundancy['z']:+.2f}z, "
        f"{biggest_redundancy['team_hard_count']:.0f} players)"
    )
    st.table(
        gaps[["archetype", "team_soft_mean", "team_hard_count", "league_mean", "z", "status"]],
        hide_index=True, width="stretch",
    )

    st.divider()
    roster_ids = roster["PLAYER_ID"].astype(int).tolist()
    exposure_cache = load_exposure_cache(recipes, k, SEASON)
    render_pairwise_conflict(recipes, k, roster_ids, exposure_cache)

    st.divider()
    render_rapm_investigation_note()

    st.divider()
    render_lineup_rankings(recipes, k, labels, roster, team_profile)


# --- Page 3: Scouting ---------------------------------------------------------

# AI-ASSISTED (Claude Code, chat)
# Prompt: "针对四个tab 第三个Scouting 改名为Rookie Slot Query 并且只针对Nets的
# 三名新秀来做 去除掉上半部分 Players who play like Ben Saraf (Combo Guard)...
# 只保留下面" (for the four tabs, rename the third one "Scouting" to "Rookie
# Slot Query" and scope it to only the Nets' three rookies; remove the
# upper "Players who play like X" section, keep only what's below).
# Used: dropped the "Players who play like X" similarity section entirely
# (it worked for ANY roster player with a real recipe - exactly what the
# three zero-NBA-data rookies this page now exclusively serves don't have,
# so keeping it would mean either breaking on a rookie selection or
# quietly falling back to some other player, neither acceptable). Renamed
# render_scouting -> render_rookie_slot_query_page (the function's PURPOSE
# changed, not just its display header, so this isn't the same case as the
# render_section_x letter-vs-name convention elsewhere in this file, which
# is specifically about pure reordering with unchanged content). The
# selectbox now sources its options from NETS_ROSTER_NCAA_BRIDGE (step3's
# own hardcoded 3-name list - Mikel Brown Jr., Tyler Bilodeau, Joshua
# Jefferson - the exact set CLAUDE.md documents as zero-NBA-data, verified
# directly against resolve_roster() returning 0 rows for all three at any
# season) rather than the full roster - no recipe/player_id lookup is
# needed or attempted for any of them, since none has one.
# Not AI: the rename, the rookie-only scope, and removing the similarity
# section - all specified directly.
def render_rookie_slot_query_page(rookie_name, recipes, k, labels, oncourt):
    st.title("Rookie Slot Query")
    render_rookie_slot_query(rookie_name, recipes, k, labels, oncourt)


# AI-ASSISTED (Claude Code, chat)
# Prompt: "I think in change Rookie Slot Query to Report generate page, how
# do you think... I only have 24 hrs right now to finish this whole
# project" - discussed as an exploratory question first (recommended
# AGAINST replacing Rookie Slot Query, since it's the portal's one direct
# answer to the assignment's own headline Mikel Brown Jr. question; the
# Report feature is already fully built and just needed unhiding, so
# adding it as its own new tab costs far less time than the alternative),
# then confirmed: "yes please."
# Used: a thin page wrapper matching render_rookie_slot_query_page's own
# pattern exactly - st.title, then delegates to the already-built,
# already-validated render_report_section (Entry 079/080). Its own player
# selector is scoped to the SAME data-eligible Nets roster
# render_diagnostic_analysis's Layer 1 already uses (`roster` here has
# already been merged against `recipes` and had rows with no fitted
# recipe dropped, by the time this module-level code runs - see the
# roster-prep block above) - a plain st.selectbox rather than the quadrant-
# chart-click this page doesn't have room for, sorted alphabetically by
# name for a predictable dropdown, the same simple-selector pattern
# Rookie Slot Query's own sidebar selectbox already uses.
# Not AI: the decision to add this page (not replace Rookie Slot Query),
# and the "make it a real page, not a hidden section" framing - given
# directly across this exchange.
def render_player_report_page(player_id, recipes, k, labels, bio, exposure_cache):
    st.title("Player Report")
    st.caption(
        "A per-player PDF scouting report, assembled from everything computed on the Diagnostic "
        "Analysis page for the selected player."
    )
    render_report_section(player_id, recipes, k, labels, bio, exposure_cache)


# --- Rookie slot query (CLAUDE.md Step 4) -----------------------------------
#
# AI-ASSISTED (Claude Code, chat)
# Prompt: rewrite this as "an interactive hypothesis rather than a static
# text block" - Brown has no NBA minutes, so the framework must not fit him
# a recipe, but a user-set primary/secondary archetype + weight slider,
# fed through similarity_weighted_benchmark (what environment players of
# that style normally have) vs. compute_team_archetype_profile (what the
# Nets currently supply), with the difference reported as the gap - plus a
# sensitivity check across several plausible archetype assumptions, so a
# gap that survives varying the assumption is a stronger claim than one
# point estimate.
# Used: st.selectbox x2 + st.slider to hand-build a k-dim hypothesis vector
# (default: Offensive Engine, matching CLAUDE.md's "on-ball scoring engine"
# college-identity note). Fed through compute_style_pool_by_vector +
# similarity_weighted_benchmark (both already built for Prompt A's
# threshold-vs-outcome-selection fix) against a cached, season-only
# exposure table (load_exposure_cache) so every slider move is cheap - the
# real per-player SQL cost is paid once, not per slider drag (confirmed
# empirically: ~38s once vs. ~0.02s for 4 additional hypothesis sweeps on
# top of the same cache - see AI_USAGE.md). The sensitivity sweep holds the
# user's chosen primary archetype fixed (that part of the hypothesis -
# "on-ball scoring engine" - comes directly from CLAUDE.md's college-
# identity note, not a free assumption) and varies only the two genuinely
# uncertain knobs: the secondary archetype and the weight split.
# Not AI: the decision to make this interactive rather than static, to
# default the primary archetype to Brown's stated college identity, and
# the disclaimer language making clear the hypothesis is the user's
# assumption, not a fitted number - all the user's own specification.
#
# AI-ASSISTED (Claude Code, chat) - generalized from render_brown_slot_query
# Prompt: same rename/scope spec as above - the page now needs to serve
# all 3 rookies, not just Brown specifically.
# Used: every "Mikel Brown Jr." string literal replaced with the
# `rookie_name` parameter. CLAUDE.md documents a college identity ("on-ball
# scoring engine") ONLY for Brown - Bilodeau and Jefferson have no such
# note, so `ROOKIE_COLLEGE_IDENTITY` is a dict keyed by name (not a
# positional default) and the selectbox's own label text only claims a
# "college identity" default for whichever rookie the dict actually
# covers; the other two get a plain "Primary archetype" label with no
# unsupported claim about where the default came from.
# Not AI: the generalization itself - specified directly.
ROOKIE_COLLEGE_IDENTITY = {
    "Mikel Brown Jr.": "Offensive Engine",  # CLAUDE.md: "on-ball scoring engine" college identity
}


# AI-ASSISTED (Claude Code, chat)
# Prompt: "Add an execution-level 'Recommended units around him' block to the
# Rookie Slot Query, replacing the gap chart as the primary output..." (full
# spec in AI_USAGE.md) - enumerate all C(16,4)=1820 four-man complements from
# the recipe-holding Nets roster, score each against the current hypothesis's
# typical-environment target (JS distance -> "environment match"), apply 3
# hard structural-flag checks, rank flag-free-first by fit with a talent
# (sum BPM) tiebreak, and surface the top few + one "what not to run"
# contrast as cards - the gap chart moves into a collapsed expander, unchanged.
# Used: enumerate_roster_combos (step4_roster_construction.py) already builds
# exactly this combo table (player_ids, mean arch_0..arch_{k-1} mixture,
# sum_bpm) for the lineup-outcome-model path - reused here with combo_size=4
# instead of writing a new enumerator. load_bpm_lookup already exists
# (used by load_lineup_rankings) - reused as-is, not duplicated. The three
# threshold constants below and UNIT_ENV_MATCH_EPSILON were NOT guessed -
# computed by running this exact scoring pipeline once against a real
# hypothesis (Brown, 70% Offensive Engine / 30% Shooting Specialist) and
# reading real percentiles/gaps off the actual 1820-combo distribution (see
# DIAGNOSTICS_README.md for the numbers). "Key name" attribution
# (e.g. "Rim protection: Sharpe") picks, among the 4 teammates, whichever one
# individually carries the highest share of the relevant archetype pair -
# computed from each player's own recipe row, not the combo mean.
# Not AI: the full computation spec, the 3-axis scoring design, the ranking
# rule (flag-free first, then fit, epsilon-tiebreak by talent), and the card
# UI spec (cards, contrast card, honesty line, caching per hypothesis) -
# given directly, in full, by the user.
UNIT_RIM_SHARE_THRESHOLD = 0.06        # combined Rim Protector/Roll Man + Mobile Big share; below -> "no rim protection" (~p25 of all 1820 four-man Nets complements: p25=0.061)
UNIT_SPACING_SHARE_THRESHOLD = 0.25    # combined Shooting Specialist + 3&D Wing share; below -> "spacing risk" (~p25: p25=0.267)
UNIT_BALL_DOMINANCE_THRESHOLD = 0.36   # combined Offensive Engine + Combo Guard share; above -> "second on-ball engine" (~p75: p75=0.357)
UNIT_ENV_MATCH_EPSILON = 0.005         # env-match units within this of the current tie-group's leader are treated as a near-tie and re-sorted by talent (sum BPM) - consecutive-ranked gaps near the top of the real distribution are frequently smaller than this (observed range across all 1820 combos ~0.44 wide, min 0.48 to max 0.93 under one tested hypothesis)
N_RECOMMENDED_UNITS = 4                # how many top cards to show ("top 3-5" per spec)


@st.cache_data(show_spinner="Scoring every 4-man complement around him...")
def load_recommended_units_cached(target, recipes, k, season, roster_id_list, rim_idx, spacing_idx, ball_idx,
                                   rim_threshold=UNIT_RIM_SHARE_THRESHOLD,
                                   spacing_threshold=UNIT_SPACING_SHARE_THRESHOLD,
                                   ball_threshold=UNIT_BALL_DOMINANCE_THRESHOLD,
                                   epsilon=UNIT_ENV_MATCH_EPSILON):
    """Every 4-man complement from roster_id_list, scored against `target`
    (the current hypothesis's typical-environment archetype vector) by
    environment match (1 - Jensen-Shannon distance), talent (sum BPM), and
    3 structural flags. Ranked flag-free-first by environment match, with
    near-ties (within `epsilon` of the current tie-group's leader) broken
    by talent. Cached on `target` itself, so a new hypothesis vector (and
    only a new hypothesis vector) triggers a recompute.
    """
    from step4_roster_construction import enumerate_roster_combos

    bpm_lookup = load_bpm_lookup(season)
    combos = enumerate_roster_combos(roster_id_list, recipes, season, k, bpm_lookup, combo_size=4)

    arch_cols = [f"arch_{i}" for i in range(k)]
    mixture = combos[arch_cols].to_numpy(dtype=float)
    combos["env_match"] = 1.0 - np.array([jensenshannon(row, target, base=2.0) for row in mixture])
    combos["rim_share"] = mixture[:, rim_idx].sum(axis=1)
    combos["spacing_share"] = mixture[:, spacing_idx].sum(axis=1)
    combos["ball_share"] = mixture[:, ball_idx].sum(axis=1)
    combos["flag_rim"] = combos["rim_share"] < rim_threshold
    combos["flag_spacing"] = combos["spacing_share"] < spacing_threshold
    combos["flag_ball"] = combos["ball_share"] > ball_threshold
    combos["n_flags"] = combos[["flag_rim", "flag_spacing", "flag_ball"]].sum(axis=1)
    combos["any_flag"] = combos["n_flags"] > 0

    combos = combos.sort_values("env_match", ascending=False).reset_index(drop=True)
    scores = combos["env_match"].to_numpy()
    tie_group = np.zeros(len(scores), dtype=int)
    if len(scores):
        anchor, gid = scores[0], 0
        for i in range(1, len(scores)):
            if anchor - scores[i] > epsilon:
                gid += 1
                anchor = scores[i]
            tie_group[i] = gid
    combos["_tie_group"] = tie_group

    return (combos.sort_values(["any_flag", "_tie_group", "sum_bpm"], ascending=[True, True, False])
                  .drop(columns="_tie_group").reset_index(drop=True))


def _unit_key_players(player_ids, recipes, season, k, rim_idx, spacing_idx, ball_idx):
    """Among these 4 teammates, who individually carries the most of each
    checked archetype pair - computed from each player's own recipe row
    (not the combo mean), for the card's "key name" attribution."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    season_recipes = recipes[recipes["SEASON"] == season].set_index("PLAYER_ID")
    indiv = season_recipes.loc[list(player_ids), arch_cols].to_numpy(dtype=float)
    rim_lead = player_ids[int(np.argmax(indiv[:, rim_idx].sum(axis=1)))]
    spacing_lead = player_ids[int(np.argmax(indiv[:, spacing_idx].sum(axis=1)))]
    ball_lead = player_ids[int(np.argmax(indiv[:, ball_idx].sum(axis=1)))]
    return rim_lead, spacing_lead, ball_lead


def _checklist_item_html(passed, text):
    icon = "✓" if passed else "✗"
    color = BL_GREEN if passed else BL_CORAL
    return (
        f'<div style="font-size:12.5px; color:{BL_INK}; margin-bottom:4px;">'
        f'<span style="color:{color}; font-weight:700; margin-right:5px;">{icon}</span>{text}'
        f'</div>'
    )


_NAME_SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def _last_name(full_name):
    """"Michael Porter Jr." -> "Porter", not "Jr." - a plain .split()[-1]
    mis-labels every suffixed name on the roster (Porter among teammates,
    Brown among the rookies this card renders)."""
    parts = full_name.split()
    while len(parts) > 1 and parts[-1].lower().rstrip(".") in _NAME_SUFFIXES:
        parts = parts[:-1]
    return parts[-1] if parts else full_name


def _recommended_unit_card_html(badge_text, badge_color, row, rookie_name, rookie_photo_uri, name_lookup,
                                 recipes, season, k, rim_idx, spacing_idx, ball_idx):
    teammate_ids = list(row["player_ids"])
    rim_lead, spacing_lead, ball_lead = _unit_key_players(
        teammate_ids, recipes, season, k, rim_idx, spacing_idx, ball_idx)

    photo_items = [(rookie_photo_uri, _last_name(rookie_name))] + [
        (HEADSHOT_URL.format(player_id=pid), _last_name(name_lookup.get(pid, str(pid))))
        for pid in teammate_ids
    ]
    photo_html = "".join(
        f'<div style="display:flex;flex-direction:column;align-items:center;width:19%;">'
        f'<img src="{uri}" style="width:42px;height:42px;border-radius:50%;object-fit:cover;'
        f'object-position:top center;border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};display:block;">'
        f'<div style="font-size:10px;color:{BL_MUTED};margin-top:4px;text-align:center;white-space:nowrap;'
        f'overflow:hidden;text-overflow:ellipsis;width:100%;">{nm}</div>'
        f'</div>'
        for uri, nm in photo_items
    )
    photo_row_html = f'<div style="display:flex;justify-content:space-between;gap:2px;margin-bottom:12px;">{photo_html}</div>'

    rim_text = (f"Rim protection: {name_lookup.get(rim_lead, '')}" if not row["flag_rim"]
                else "No rim protection")
    spacing_text = (f"Spacing: {name_lookup.get(spacing_lead, '')}" if not row["flag_spacing"]
                     else "Spacing risk")
    engine_text = ("Lone engine" if not row["flag_ball"]
                   else f"Second on-ball engine: clashes with {name_lookup.get(ball_lead, '')}")
    checklist_html = (
        _checklist_item_html(not row["flag_rim"], rim_text)
        + _checklist_item_html(not row["flag_spacing"], spacing_text)
        + _checklist_item_html(not row["flag_ball"], engine_text)
    )

    badge_html = (
        f'<div style="display:inline-block; padding:2px 8px; border-radius:10px; background:{badge_color}; '
        f'color:{BL_WHITE}; font-size:10px; font-weight:700; letter-spacing:0.04em; text-transform:uppercase; '
        f'margin-bottom:8px;">{badge_text}</div>'
    )
    return (
        f'<div style="padding:8px 4px; height:100%; box-sizing:border-box;">'
        f'{badge_html}'
        f'{photo_row_html}'
        f'<div style="display:flex; justify-content:space-between; margin-bottom:10px;">'
        f'<div><div style="font-size:10px; text-transform:uppercase; letter-spacing:0.04em; color:{BL_MUTED};">Environment match</div>'
        f'<div style="font-size:20px; font-weight:700; color:{BL_INK};">{row["env_match"]:.0%}</div></div>'
        f'<div style="text-align:right;"><div style="font-size:10px; text-transform:uppercase; letter-spacing:0.04em; color:{BL_MUTED};">Total BPM</div>'
        f'<div style="font-size:20px; font-weight:700; color:{BL_INK};">{row["sum_bpm"]:+.1f}</div></div>'
        f'</div>'
        f'{checklist_html}'
        f'</div>'
    )


def render_rookie_slot_query(rookie_name, recipes, k, labels, oncourt):
    st.markdown(f"#### {rookie_name} — rookie slot query")
    st.caption(
        f"{rookie_name} has zero NBA minutes, so this framework never fits him an archetype "
        f"recipe. Instead: set a hypothesis for his style below, and see what kind of "
        f"player typically surrounds someone who plays that way - compared to what the "
        f"Nets currently supply."
    )

    arch_names = [labels.get(i, f"archetype {i}") for i in range(k)]
    college_identity = ROOKIE_COLLEGE_IDENTITY.get(rookie_name)
    if college_identity and college_identity in arch_names:
        default_primary = college_identity
        primary_label_text = "Primary archetype (default: his college identity, an on-ball scoring engine per CLAUDE.md)"
    else:
        default_primary = arch_names[0]
        primary_label_text = "Primary archetype (no documented college-identity default for him yet — pick one)"
    fallback_secondary = [a for a in arch_names if a != default_primary][0]
    default_secondary = "Shooting Specialist" if "Shooting Specialist" in arch_names and \
        "Shooting Specialist" != default_primary else fallback_secondary

    col1, col2, col3 = st.columns([2, 2, 1.6])
    with col1:
        primary_label = st.selectbox(
            primary_label_text,
            arch_names, index=arch_names.index(default_primary),
        )
    with col2:
        secondary_options = [a for a in arch_names if a != primary_label]
        default_idx = (secondary_options.index(default_secondary)
                        if default_secondary in secondary_options else 0)
        secondary_label = st.selectbox("Secondary archetype", secondary_options, index=default_idx)
    with col3:
        primary_weight = st.slider("Primary weight", min_value=0.50, max_value=0.95, value=0.70, step=0.05)

    st.caption(
        f"This is a hand-set hypothesis, not a fitted number — "
        f"**{primary_weight:.0%} {primary_label} / {1 - primary_weight:.0%} {secondary_label}** "
        f"is your assumption about how {rookie_name}'s game translates, stated explicitly so it "
        f"can be second-guessed, not a number this model invented."
    )

    arch_idx = {name: i for i, name in enumerate(arch_names)}

    def _build_vector(p_label, s_label, weight):
        v = np.zeros(k)
        v[arch_idx[p_label]] = weight
        v[arch_idx[s_label]] = 1 - weight
        return v

    roster_ids = load_nets_roster(season=SEASON)
    roster_ids = roster_ids.merge(recipes[["PLAYER_ID"]], on="PLAYER_ID", how="inner")
    team_profile = compute_team_archetype_profile(roster_ids["PLAYER_ID"].tolist(), recipes, k)
    exposure_cache = load_exposure_cache(recipes, k, SEASON)

    def _gap_for(p_label, s_label, weight):
        vec = _build_vector(p_label, s_label, weight)
        pool = compute_style_pool_by_vector(vec, recipes, k, season=SEASON, exposure_cache=exposure_cache)
        res = similarity_weighted_benchmark(None, recipes, k, season=SEASON, oncourt=oncourt,
                                             power=4.0, pool=pool)
        gap = res["all_baseline"] - team_profile["soft_mean"]
        return res, gap

    res, gap = _gap_for(primary_label, secondary_label, primary_weight)
    top_gap_idx = int(np.argmax(gap))
    top_gap_label = arch_names[top_gap_idx]

    st.markdown("### Recommended units around him")
    st.caption(
        f"Every 4-man complement from the Nets' 16 data-eligible roster players "
        f"(C(16,4) = 1,820 combinations), scored against the same typical-environment target "
        f"above and ranked by fit, subject to three structural checks - see exactly how below."
    )

    rim_idx = [arch_idx["Rim Protector / Roll Man"], arch_idx["Mobile Big"]]
    spacing_idx = [arch_idx["Shooting Specialist"], arch_idx["3&D Wing"]]
    ball_idx = [arch_idx["Offensive Engine"], arch_idx["Combo Guard"]]
    roster_id_list = roster_ids["PLAYER_ID"].tolist()
    name_lookup = roster_ids.set_index("PLAYER_ID")["PLAYER_NAME"].to_dict()

    ranked_units = load_recommended_units_cached(
        res["all_baseline"], recipes, k, SEASON, roster_id_list, rim_idx, spacing_idx, ball_idx,
    )
    rookie_photo_uri = hull_callout_chart.get_headshot_data_uri(None, rookie_name)

    top_units = ranked_units.head(N_RECOMMENDED_UNITS)
    unit_cols = st.columns(2)
    for i, (_, unit_row) in enumerate(top_units.iterrows()):
        with unit_cols[i % 2]:
            with st.container(border=True):
                st.markdown(
                    _recommended_unit_card_html(
                        f"#{i + 1}", BL_GREEN, unit_row, rookie_name, rookie_photo_uri, name_lookup,
                        recipes, SEASON, k, rim_idx, spacing_idx, ball_idx,
                    ),
                    unsafe_allow_html=True,
                )

    flagged_units = ranked_units[ranked_units["any_flag"]]
    if len(flagged_units):
        worst_unit = flagged_units.sort_values(["n_flags", "env_match"], ascending=[False, True]).iloc[0]
        st.markdown("**What not to run**")
        with st.container(border=True):
            st.markdown(
                _recommended_unit_card_html(
                    "What not to run", BL_CORAL, worst_unit, rookie_name, rookie_photo_uri, name_lookup,
                    recipes, SEASON, k, rim_idx, spacing_idx, ball_idx,
                ),
                unsafe_allow_html=True,
            )

    st.caption(
        "Ranked by environment fit + talent under structural constraints - descriptive "
        "construction, not a lineup outcome prediction (see the Future Work page for why "
        "outcome prediction was ruled out)."
    )

    with st.expander("The environment math behind these picks"):
        st.markdown(
            f"**Players who share this style are typically surrounded by "
            f"{gap[top_gap_idx] * 100:+.1f}pp more {top_gap_label} than the Nets currently have** "
            f"({res['all_baseline'][top_gap_idx]:.1%} typical environment vs. "
            f"{team_profile['soft_mean'][top_gap_idx]:.1%} on the current roster; "
            f"similarity-weighted across {res['n_pool']} league players, effective sample "
            f"size ≈{res['ess_all']:.0f})."
        )
        _, _, note = ARCHETYPE_TO_PAPER.get(top_gap_idx, ("", "", ""))
        if note:
            st.caption(f"What {top_gap_label} looks like: {note}")

        gap_df = pd.DataFrame({
            "archetype": arch_names,
            "env_pct": res["all_baseline"],
            "nets_pct": team_profile["soft_mean"],
            "gap_pp": gap * 100,
        }).sort_values("gap_pp")
        gap_fig = diverging_bar(gap_df, "gap_pp", "archetype", ["env_pct", "nets_pct"],
                                x_title="gap: typical environment for this style − Nets' current supply (pp)")
        gap_fig.update_traces(hovertemplate=(
            "%{y}<br>typical environment=%{customdata[0]:.1%}<br>"
            "Nets currently supply=%{customdata[1]:.1%}<br>gap=%{x:+}pp<extra></extra>"
        ))
        st.plotly_chart(gap_fig, width="stretch")

    st.markdown("**Sensitivity: does the top recommendation hold across other plausible assumptions?**")
    alt_secondary_pool = [a for a in arch_names if a not in (primary_label, secondary_label)]
    alt_secondary = alt_secondary_pool[0] if alt_secondary_pool else secondary_label
    variants = [
        (primary_label, secondary_label, primary_weight),
        (primary_label, secondary_label, round(min(primary_weight + 0.15, 0.95), 2)),
        (primary_label, secondary_label, round(max(primary_weight - 0.15, 0.50), 2)),
        (primary_label, alt_secondary, primary_weight),
    ]
    sensitivity_rows = []
    for p_label, s_label, w in variants:
        res_v, gap_v = _gap_for(p_label, s_label, w)
        top_v_idx = int(np.argmax(gap_v))
        sensitivity_rows.append({
            "assumption": f"{w:.0%} {p_label} / {1 - w:.0%} {s_label}",
            "top_gap_archetype": arch_names[top_v_idx],
            "gap_pp": gap_v[top_v_idx] * 100,
        })
    sens_df = pd.DataFrame(sensitivity_rows)
    n_unique = sens_df["top_gap_archetype"].nunique()
    if n_unique == 1:
        st.success(
            f"Stable: **{sens_df['top_gap_archetype'].iloc[0]}** is the top recommended "
            f"archetype across all {len(sens_df)} plausible variations of this hypothesis "
            f"tested (weight ±15pp, and one alternate secondary archetype)."
        )
    else:
        st.warning(
            f"Not stable: the top recommended archetype changes across plausible variations "
            f"tested ({', '.join(sens_df['top_gap_archetype'].unique())}) - treat the "
            f"recommendation above as a point estimate under one specific assumption, not a "
            f"firm conclusion."
        )
    with st.expander("Show the assumption variants tested"):
        sens_display = sens_df.copy()
        sens_display["gap_pp"] = sens_display["gap_pp"].apply(lambda v: f"{v:+.1f}")
        st.table(sens_display, hide_index=True, width="stretch")


# --- Sidebar navigation -------------------------------------------------------

recipes, k, labels, oncourt = load_static()
bio = load_player_bio(season=SEASON)
roster = load_nets_roster(season=SEASON)

arch_cols_all = [f"arch_{i}" for i in range(k)]
roster = roster.merge(recipes[["PLAYER_ID"] + arch_cols_all], on="PLAYER_ID", how="left")
roster = roster.dropna(subset=[arch_cols_all[0]]).reset_index(drop=True)
roster["dominant_arch"] = roster[arch_cols_all].values.argmax(axis=1)
roster["role"] = roster["dominant_arch"].map(labels)

with st.sidebar:
    st.markdown(
        f'''
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:4px;">
            <img src="{TEAM_LOGO_URL.format(team_id=NETS_TEAM_ID)}" style="height:2.6rem;width:2.6rem;flex-shrink:0;">
            <span style="font-size:1.7rem;font-weight:700;color:{BL_INK};letter-spacing:-0.01em;line-height:1.1;">
                Nets Archetype Portal
            </span>
        </div>
        ''',
        unsafe_allow_html=True,
    )
    st.caption("Probabilistic Archetype Scouting Portal")
    st.divider()

    # AI-ASSISTED (Claude Code, chat)
    # Prompt: "我觉得还不如把这部分先去掉 因为第三个tab我想做的是roster
    # construction 所以说diagostic analysis就是针对个体球员的 roster
    # construction是针对于球队的" - Roster Construction re-added to the nav
    # as its own page (position 3, before Scouting), reversing its earlier
    # removal, since the user now wants an explicit individual-vs-team split
    # across pages rather than folding team-level content into Diagnostic
    # Analysis's Layer 3 (see render_pairwise_conflict's own note).
    # AI-ASSISTED (Claude Code, chat)
    # Prompt: first "Roster Construction先隐藏掉这个tab..." (hide Roster
    # Construction for now), then superseded by "或者我把Roster Constructio改成
    # Future Work and obstacle..." (or, repurpose it into "Future Work and
    # Obstacle" instead) - render_roster_construction() itself is untouched
    # (unreachable, not deleted) in case the roster-construction analysis
    # feature comes back later; this slot now points at the new page instead.
    # AI-ASSISTED (Claude Code, chat)
    # Prompt: "I think in change Rookie Slot Query to Report generate page,
    # how do you think... I only have 24 hrs right now to finish this
    # whole project" -> "yes please" (confirming: ADD a new Player Report
    # page, do not replace Rookie Slot Query - see render_player_report_page's
    # own note for why replacing it was pushed back on).
    # Used: new nav entry added alongside the existing 4, not swapped in
    # for one of them.
    # Not AI: the decision to add rather than replace - the user's own call.
    nav_options = ["📖 The 8 Player Types", "🔍 Player Breakdown"]
    if SHOW_ROOKIE_SLOT_QUERY_PAGE:
        nav_options.append("🎯 Building Around Rookie")
    if SHOW_PLAYER_REPORT_PAGE:
        nav_options.append("📄 Report")
    if SHOW_FUTURE_WORK_PAGE:
        nav_options.append("🚧 Future Work & Obstacles")
    page = st.radio("Navigate", nav_options, label_visibility="collapsed")
    st.divider()

    # Diagnostic Analysis no longer needs this - Layer 1's own quadrant
    # scatter/table IS the player-selection mechanism for that page now
    # (click a point). Rookie Slot Query still needs an explicit pick -
    # scoped to NETS_ROSTER_NCAA_BRIDGE (the 3 zero-NBA-data rookies) only,
    # not the full roster - see render_rookie_slot_query_page's own note.
    if page in ("🎯 Building Around Rookie",):
        selected_player = st.selectbox("Rookie", NETS_ROSTER_NCAA_BRIDGE)
    elif page in ("📄 Report",):
        # `roster` here is already merged against `recipes` and dropna'd
        # down to the data-eligible players (see the roster-prep block
        # above) - the same 16-player pool Diagnostic Analysis's Layer 1
        # screens, just picked via a plain dropdown instead of a chart
        # click, since a dedicated report page has no quadrant chart to
        # click a point on.
        report_names = sorted(roster["PLAYER_NAME"].tolist())
        default_match = roster.loc[roster["PLAYER_ID"] == DEFAULT_DIAG_PLAYER_ID, "PLAYER_NAME"]
        default_name = default_match.iloc[0] if len(default_match) else report_names[0]
        selected_report_name = st.selectbox("Player", report_names, index=report_names.index(default_name))
        selected_report_pid = int(roster.loc[roster["PLAYER_NAME"] == selected_report_name, "PLAYER_ID"].iloc[0])

if page == "📖 The 8 Player Types":
    render_intro_page(roster, labels)
elif page == "🔍 Player Breakdown":
    render_diagnostic_analysis(recipes, k, labels, oncourt, bio, roster)
elif page == "🎯 Building Around Rookie":
    render_rookie_slot_query_page(selected_player, recipes, k, labels, oncourt)
elif page == "📄 Report":
    exposure_cache = load_exposure_cache(recipes, k, SEASON)
    render_player_report_page(selected_report_pid, recipes, k, labels, bio, exposure_cache)
elif page == "🚧 Future Work & Obstacles":
    render_future_work()
