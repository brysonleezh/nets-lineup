"""
Carved out of portal.py during the Day-8 portal restructuring. The largest
of the four split-out files: everything behind the "Player Breakdown" tab
(render_diagnostic_analysis - the screening scatter/table, Sections A-E,
the Layer-2 player-report card render_player_report), plus two tabs kept
hidden behind their existing SHOW_* flags (Rookie Slot Query / "Building
Around Rookie", and Future Work & Obstacles' RAPM investigation writeup),
plus several fully dead/unreachable functions found while carving this file
out (render_roster_construction, render_league_benchmark_tab, and a few
smaller superseded chart/verdict helpers) - all preserved unchanged and
un-deleted, per this restructuring's own rule.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.spatial.distance import jensenshannon

from step0_data import build_nba_side_tables
from step1_archetype_model import (
    ARCHETYPE_TO_PAPER,
    load_basis as load_ada_basis,
)
from step2_diagnostic_analysis import (
    compute_archetype_gaps,
    compute_league_baseline,
    compute_style_pool_by_vector,
    compute_team_archetype_profile,
    screening_table,
    similarity_weighted_benchmark,
    team_gap_summary,
)
import hull_callout_chart
import step2b_player_diagnostics as diag2b
from portal_shared import (
    BASIS_DIR,
    SEASON,
    BL_PAPER,
    BL_WHITE,
    BL_INK,
    BL_MUTED,
    BL_LINE,
    BL_GREEN,
    BL_GREEN_SOFT,
    BL_CORAL,
    BL_GOLD,
    HEADSHOT_URL,
    DEFAULT_DIAG_PLAYER_ID,
    DIAG_MASS_THRESHOLD,
    ELASTIC_SPREAD_THRESHOLD_PP,
    FEATURE_LABELS,
    PLAYTYPE_LABELS,
    FEATURE_DESCRIPTIONS,
    CATEGORY_ORDER,
    FEATURE_CATEGORY,
    _elasticity_verdict,
    _build_sortable_table_html,
    load_drift_attribution_cached,
    load_exposure_cache,
    load_full_features,
    load_individual_role_sensitivity_cached,
    load_league_elasticity_spreads,
    load_league_miscasting,
    load_league_purity_entropy,
    load_miscasting_cached,
    load_miscasting_grounding_cached,
    load_mismatch,
    load_nets_roster,
    load_peer_centroid,
    load_player_base_stats,
    load_recipes_all_seasons,
    load_role_drift_cached,
    load_signature_cached,
    load_teammate_lift,
    load_transition_drift_distribution_cached,
)




def _hex_to_rgba(hex_color, alpha):
    hex_color = hex_color.lstrip("#")
    r, g, b = (int(hex_color[i:i + 2], 16) for i in (0, 2, 4))
    return f"rgba({r},{g},{b},{alpha})"
# No longer read anywhere since Diagnostic Analysis's tabs were replaced
# by a vertical narrative (Section 3 there uses similarity_weighted_
# benchmark directly, unconditionally) - left in place only because
# render_league_benchmark_tab (the code this flag used to gate) is still
# kept in the file as superseded-but-visible prior work.
SHOW_LEAGUE_BENCHMARK_TAB = False


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

# AI-ASSISTED (Claude Code, chat) - Prompt: "把Building Around Rookie给隐藏
#掉吧 因为我觉得来不及都弄完了" (hide Building Around Rookie - not enough
# time to finish it all). Used: same "kept but not wired up" convention as
# every other SHOW_* flag on this page - gates the page out of the sidebar
# nav list; render_rookie_slot_query_page() and everything under it
# (render_rookie_slot_query, the recommended-units feature, etc.) are
# completely untouched, a one-line flip to bring it back.
# Not AI: the decision to hide it, and the stated reason (time) - the
# user's own call.
#
# Unrelated-systems note: this page and step5_rookie_projections.py's
# "NCAA Bridge — Rookie Archetype Projections" page both concern the same
# three rookies (Mikel Brown Jr., Tyler Bilodeau, Joshua Jefferson) but are
# otherwise unrelated systems that happen to share the word "rookie." This
# page asks what archetype SLOT a rookie should fill next to the existing
# roster (a lineup-construction question, using NETS_ROSTER_NCAA_BRIDGE +
# the original NBA-side archetype model only). step5's page shows the
# actual trained NCAA->NBA translator's own predictions (a player-
# projection question, using the full Phase 1-6 pipeline). Do not merge
# them or assume one supersedes the other.
SHOW_ROOKIE_SLOT_QUERY_PAGE = False

# AI-ASSISTED (Claude Code, chat) - Prompt: "I want to do one more roster
# construction tab, what do you think we can do?" - investigation found
# render_roster_construction() already existed complete and previously
# validated (see portal.py:187-201's own history - this page was live
# once, hidden, then its slot repurposed into Future Work & Obstacles
# instead of being deleted); presented that finding, user chose to
# revive it as-is over extending it or also reviving Rookie Slot Query.
# Used: same "kept but not wired up" flip as every other SHOW_* flag on
# this page - gates the page INTO the sidebar nav list; render_roster_
# construction() and everything under it were already fully built,
# requiring zero code changes beyond this flag and portal.py's nav wiring.
# Not AI: the decision to revive it, and the choice of scope (as-is, no
# extension) - the user's own call.
#
# AI-ASSISTED (Claude Code, chat) - Prompt: "先隐藏掉Roster Construction"
# (hide Roster Construction for now). Used: same "kept but not wired up"
# flip back to False - render_roster_construction() and everything under
# it remain fully built and untouched, a one-line flip to re-enable again.
# Not AI: the decision to hide it - the user's own call.
SHOW_ROSTER_CONSTRUCTION_PAGE = False

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
    # story): st.iframe() renders in an iframe sandboxed with
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
    st.iframe(table_html, height=iframe_height)


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
        st.iframe(conflict_html, height=conflict_height)


@st.cache_data
def load_bio_all_seasons():
    return diag2b.load_player_bio_all_seasons()


@st.cache_data(show_spinner=False)
def load_development_comps_cached(player_id, recipes_all, bio_all, k):
    return diag2b.development_comps(player_id, recipes_all, bio_all, k)


@st.cache_data(show_spinner="Computing role elasticity...")
def load_role_elasticity_cached(player_id, season=SEASON):
    return diag2b.role_elasticity(player_id, season=season)


@st.cache_data(show_spinner="Computing role sensitivity profile...")
def load_role_sensitivity_cached(player_id, recipes, k, season=SEASON):
    return diag2b.role_sensitivity_profile(player_id, recipes, k, season=season)


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
                st.iframe(table_html, height=h)


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
            st.iframe(wrapped_html, height=wrapper_height)


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


def render_diagnostic_analysis(
    recipes,
    k,
    labels,
    oncourt,
    bio,
    roster,
    sidebar_pid=None,
    context_note=None,
):
    st.title("Diagnostic Analysis")
    if context_note:
        st.caption(context_note)

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
    # AI-ASSISTED (Claude Code, chat) - Prompt: "在Player Breakdown 添加一个
    # list可以选择球员 就像 Report Tab那样" (add a player-picker list to Player
    # Breakdown, like the Report tab has). `sidebar_pid` comes from a new
    # st.selectbox portal.py's sidebar now renders for this page (mirroring
    # the Report tab's own sidebar-selectbox pattern) - a THIRD selection
    # input alongside the chart click and table-photo click above, not a
    # replacement for either. Applied last and unconditionally (when
    # present) because portal.py's sidebar code already replicates the same
    # chart/table-click detection (read-only, no state mutation) BEFORE
    # this function ever runs, and pre-syncs the dropdown's own displayed
    # value to match whichever of the two just fired - so by the time
    # `sidebar_pid` gets here it already correctly reflects either a fresh
    # chart/table click (synced) or the user's own fresh dropdown pick.
    # That pre-sync has to happen in portal.py, not here: a keyed Streamlit
    # widget's displayed value is fixed at the point in the SCRIPT where
    # it's instantiated, and the sidebar renders before this function is
    # even called - syncing here would always lag one rerun behind.
    # Not AI: the requirement itself, and "like the Report tab" as the
    # placement/style precedent - given directly.
    if sidebar_pid is not None:
        selected_pid = sidebar_pid
    st.session_state["diag_selected_pid"] = selected_pid

    render_layer1_screening(screening_df, labels, selected_pid)

    st.divider()
    if selected_pid is not None:
        render_player_report(selected_pid, recipes, k, labels, bio, exposure_cache)
    else:
        st.caption("No data-eligible roster player to show.")


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
        st.iframe(combined_html, height=combined_height)

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
    st.iframe(html, height=height)

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
        st.iframe(bottom_html, height=bottom_height)

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
        st.iframe(html, height=height)
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
UNIT_RIM_SHARE_THRESHOLD = 0.06        # combined Rim Protector/Roll Man + Play-Finishing Big share; below -> "no rim protection" (~p25 of all 1820 four-man Nets complements: p25=0.061)
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

    rim_idx = [arch_idx["Rim Protector / Roll Man"], arch_idx["Play-Finishing Big"]]
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
