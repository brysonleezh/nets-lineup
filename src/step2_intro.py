"""
Carved out of portal.py during the Day-8 portal restructuring. Everything
that serves ONLY the "The 8 Player Types" (Intro) tab: the frozen-K=8
convex-hull scatter (compute_hull_projection/build_intro_hull_html, backed
by hull_callout_chart.py), the Nets roster gallery table, and the K-slider
explained-variance chart (currently dead/commented out on the live page,
kept per this restructuring's "don't delete dead code" rule). Entry point
for portal.py's nav dispatch is render_intro_page().

The page used to expose a K=4..10 slider over this chart, backed by bases
precomputed offline in src/pipeline/precompute_hull_bases.py - removed
(see AI_USAGE.md) since K=8 is frozen project-wide (Phase 2's RSS/intra-
variance diagnostics + consensus-basin rule) and every other page reads
the K=8 basis only; a slider implied a choice the visitor never actually
had. precompute_hull_bases.py and its data/hull_bases/k4../k10 output are
left in place as historical pipeline artifacts, not deleted, but nothing
in the live app reads them anymore.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA

from step0_data import _normalize_name
from step1_archetype_model import (
    load_basis as load_ada_basis,
    load_population as load_ada_population,
    project as ada_project,
    NETS_ROSTER,
)
from step2_diagnostic_analysis import DATA_DIR
import hull_callout_chart
from portal_shared import (
    BASIS_DIR,
    SEASON,
    BL_PAPER,
    BL_WHITE,
    BL_INK,
    BL_MUTED,
    BL_LINE,
    BL_GREEN,
    BL_CORAL,
    HEADSHOT_URL,
    load_static,
    load_player_bio,
    load_player_base_stats,
    _build_sortable_table_html,
)



# Intro page's convex-hull scatter reads the official basis_2025_26 fit
# (this project's one authoritative K=8, referenced everywhere else in the
# portal) - K is frozen, not a live parameter (see the module docstring).
# data/k_selection.csv reflects a different, now-superseded population
# (multi-season / higher MIN floor, from before this project's single-
# season direction change - see step1_archetype_model.py's own Part A
# comment). data/k_selection_2025_26.csv is the sweep actually run against
# the same population (2025-26, MIN>=300) this hull chart uses - pairing
# the stale file here would mislead the "what does each extra corner buy"
# comparison, so this deliberately does NOT point at k_selection.csv.
K_SELECTION_PATH = DATA_DIR / "k_selection_2025_26.csv"


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
# (src/pipeline/precompute_hull_bases.py - ADA is too slow to refit live), hover with
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
def load_hull_basis():
    """Reuses this project's one authoritative production basis
    (data/basis_2025_26) rather than a second fit, so this chart always
    matches every other page in the portal - never refit here."""
    return load_ada_basis(BASIS_DIR)


@st.cache_data
def load_hull_archetype_defs():
    return pd.read_csv(BASIS_DIR / "archetype_definitions.csv")


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
def compute_hull_projection():
    """Everything the hull scatter needs: the 2D PCA plane (fit on the 8
    archetypoid rows only - see the module note above for why fitting it
    on all players would be wrong, not just a different choice), every
    player's position on that plane, their own simplex recipe (project(),
    not a refit), and the honesty check of how many archetypoids actually
    land on the 2D hull of the projection. K is always 8 - the frozen,
    project-wide basis - never a live parameter.
    """
    fit = load_hull_basis()
    defs = load_hull_archetype_defs()
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

# AI-ASSISTED (Claude Code, chat) - Prompt: "in The 8 Player Types, since we
# already have their receipt in the prediction for NCAA bridge, I think we can
# show his prediction receipt and point it out in the table please" - the 3
# true rookies' Mixture cells previously rendered '-'; the NCAA->NBA translator
# (Phases 1-6) already produces a real predicted NBA recipe for each, so the
# cell now shows it instead of nothing.
# Used: the green ramp + hollow/DASHED bars + an explicit "projected" pill, so
# a modeled recipe can never be mistaken for a fitted one at a glance. Hue
# alone was deliberately not enough - the dashed/solid difference is the part
# that survives greyscale printing and colour-blind viewing, which matters
# because this table is the first thing a coach sees. The projections are read
# from the frozen Phase-6 output (data/projections/nets_rookies_2026.csv), NOT
# recomputed here, so this table and the NCAA Bridge page cannot silently
# disagree - the same rule step4_report.py follows for the PDF.
# Not AI: the decision to surface the projections in this table at all, and to
# flag them rather than blend them in - the user's own call.
REPO_ROOT_CFG = DATA_DIR.parent / "config.yaml"
PROJECTED_RAMP_TOP3 = ["#004b2b", "#3f7d5c", "#8fb3a1"]  # BL_GREEN family, dark -> light by rank
PROJECTED_PILL_HTML = (
    '<div style="display:flex;align-items:center;gap:4px;">'
    '<span style="font-size:9px;font-weight:700;letter-spacing:0.04em;text-transform:uppercase;'
    'color:#004b2b;background:rgba(0,75,43,0.10);border:1px dashed #004b2b;'
    'border-radius:3px;padding:0 4px;white-space:nowrap;">projected · NCAA bridge</span>'
    '</div>'
)


@st.cache_data
@st.cache_data
def load_rookie_nba_ids():
    """{display_name: nba_player_id} for the 3 zero-NBA-data rookies, read from
    config.yaml — the same frozen input the translator pipeline itself used.

    Read here rather than imported from step5_rookie_projections (which has an
    identical helper) on purpose: this page should not have to import the NCAA
    Bridge page to draw its own roster table. The duplication is four lines and
    one source of truth (config.yaml) is still shared."""
    import yaml
    try:
        cfg = yaml.safe_load((REPO_ROOT_CFG).read_text())
        return {r["display_name"]: r.get("nba_player_id") for r in cfg.get("rookies", [])}
    except Exception:
        return {}


def load_rookie_projections():
    """The frozen Phase-6 NCAA->NBA predictions for the 3 zero-NBA-data
    rookies, keyed by the SAME display name NETS_ROSTER uses. Returns
    {name: np.ndarray(k)} - the predicted NBA-side recipe (y_pred_0..7),
    which lives in the same 8-archetype space as every fitted recipe in
    this table, so the two are directly comparable (that comparability is
    the whole point of the bridge).

    Missing file -> {} rather than an exception: the projections are a
    separate pipeline's output, and this page must still render the other
    16 players if that pipeline hasn't been run in a given checkout."""
    path = DATA_DIR / "projections" / "nets_rookies_2026.csv"
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    y_cols = [c for c in df.columns if c.startswith("y_pred_") and c != "y_pred_argmax"]
    y_cols.sort(key=lambda c: int(c.rsplit("_", 1)[1]))
    out = {}
    for _, row in df.iterrows():
        vec = row[y_cols].to_numpy(dtype=float)
        # A projection that doesn't sum to 1 is not a recipe - skip it rather
        # than render a number the rest of the page would treat as a mixture.
        if not np.isfinite(vec).all() or abs(vec.sum() - 1.0) > 1e-4:
            continue
        out[str(row["display_name"])] = vec
    return out


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


def _mixture_cell(vals, arch_names, global_max, projected=False):
    """Top-3 archetype mixture as 3 stacked rows: a small bar (length +
    grey shade, dark->light by rank, both ~ the percentage - GRAY_RAMP_TOP3)
    followed by its visible '56% Combo Guard' label. `vals` is None for a
    player with neither a fitted recipe nor a projection - renders '-'.
    Bar width scales against `global_max` (the single largest top-1 share
    anywhere in the table) so bars stay comparable across rows.

    projected=True renders the SAME geometry in a visually distinct
    treatment (green ramp, hollow/dashed bars, "projected" pill) - see
    PROJECTED_RAMP_TOP3's own note for why these two kinds of number must
    never look alike."""
    if vals is None:
        return "—"
    ramp = PROJECTED_RAMP_TOP3 if projected else GRAY_RAMP_TOP3
    top3 = np.argsort(vals)[::-1][:3]
    rows_html = []
    for rank, i in enumerate(top3):
        pct = float(vals[i])
        # Clamped at both ends: global_max is derived from FITTED recipes only,
        # so a projected share above it would otherwise render a bar wider than
        # the column (today 0.55 vs 0.67, but nothing enforces that ordering).
        width_px = min(MIXTURE_BAR_MAX_PX,
                       max(MIXTURE_BAR_MIN_PX, round(pct / global_max * MIXTURE_BAR_MAX_PX)))
        color = ramp[min(rank, len(ramp) - 1)]
        # Projected bars are hollow + dashed (measured ones are solid): the
        # distinction survives greyscale printing and colour-blind viewing,
        # which a hue-only difference would not.
        bar_style = (
            f'background:transparent;border:1px dashed {color};'
            if projected else
            f'background:{color};border:1px solid {BL_MUTED};'
        )
        rows_html.append(
            f'<div style="display:flex;align-items:center;gap:6px;">'
            f'<div style="width:{width_px}px;min-width:{width_px}px;height:12px;{bar_style}'
            f'border-radius:3px;"></div>'
            f'<span style="font-size:12px;color:{BL_INK};white-space:nowrap;">{pct:.0%} {arch_names[i]}</span>'
            f'</div>'
        )
    if projected:
        rows_html.insert(0, PROJECTED_PILL_HTML)
    return f'<div style="display:flex;flex-direction:column;gap:3px;">{"".join(rows_html)}</div>'


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

    rookie_projections = load_rookie_projections()
    rookie_nba_ids = load_rookie_nba_ids()

    resolved = {}  # name -> (player_id or None, row_idx or None)
    for name in roster_names:
        match = bio_by_norm[bio_by_norm["_norm"] == _normalize_name(name)]
        if len(match) == 0:
            resolved[name] = (None, None)
            continue
        pid = int(match.iloc[0]["PLAYER_ID"])
        resolved[name] = (pid, _resolve_row_by_id(pop, pid))

    n_with_recipe = sum(1 for _, row_idx in resolved.values() if row_idx is not None)
    n_projected = sum(1 for n in roster_names
                      if resolved[n][1] is None and n in rookie_projections)
    st.markdown(f"#### Nets roster at K={k}")
    st.caption(
        f"{len(roster_names)} roster players ({n_with_recipe} with a fitted K={k} recipe "
        f"measured from their own 2025-26 NBA minutes). Bio columns from player_bio, box "
        f"score from player_base. Mixture: top 3 archetypes, highest share first, bar "
        f"length and shade both ~ the percentage. Click a column header to sort."
    )
    if n_projected:
        # Stated as its own line, not buried in the caption above: this is the
        # one place in the portal where a measured and a modeled recipe sit in
        # the same column, so the distinction has to be impossible to miss.
        st.caption(
            f"The remaining {n_projected} are true rookies with no NBA minutes of any kind. "
            f"Their mixture is **projected** from college data by the NCAA→NBA translator "
            f"(dashed green bars, marked *projected · NCAA bridge*) — a model output with real "
            f"error bars, not a measurement. Everything else in their row is still '—' because "
            f"no NBA box score exists. See the **NCAA Bridge** page for how each was derived "
            f"and how accurate the translator was on players whose rookie season we already know."
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
            # AI-ASSISTED (Claude Code, chat) - Prompt: "可以把这三名球员profile
            # image Include进来" - these three rendered as initials discs because
            # they have no row in player_bio, so `pid` is None. They DO have real
            # NBA headshots; their nba_player_id just lives in config.yaml rather
            # than in the DB this table joins against.
            # Used: get_headshot_data_uri(pid, name) rather than the raw
            # HEADSHOT_URL the other 16 rows use. That helper downloads once,
            # caches, and — critically for brand-new draftees — detects the CDN's
            # generic silhouette placeholder and falls back to the initials disc
            # instead of rendering a grey non-photo. The other 16 are established
            # players whose photos certainly exist; these three are exactly the
            # case where a missing photo is plausible, so they get the guarded
            # path. Verified all three resolve to real photos today.
            # Not AI: the request to show their photos - the owner's own.
            photo_html = (
                f'<img src="{hull_callout_chart.get_headshot_data_uri(rookie_nba_ids.get(name), name)}" '
                f'style="width:36px;height:36px;border-radius:50%;object-fit:cover;'
                f'object-position:top center;'
                f'border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};display:block;">'
            )
        else:
            photo_html = (
                f'<div style="width:36px;height:36px;border-radius:50%;background:{BL_LINE};'
                f'border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};"></div>'
            )

        # A fitted recipe always wins; the projection is only ever a fallback
        # for a player who has no 2025-26 NBA row at all, so no player can
        # ever show a modeled number when a measured one exists.
        projected_vals = rookie_projections.get(name) if recipe_vals is None else None
        mixture_vals = recipe_vals if recipe_vals is not None else projected_vals
        mix_sort = float(mixture_vals.max()) if mixture_vals is not None else -1.0
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
            (_mixture_cell(mixture_vals, arch_names, global_max,
                           projected=recipe_vals is None and projected_vals is not None), mix_sort),
        ])

    table_html, iframe_height = _build_sortable_table_html("roster_table", columns, rows_cells)
    st.iframe(table_html, height=iframe_height)


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
# Used: cached on nets_ids_tuple rather than passing `proj`/`labels`
# straight through to a cached function - both are large (proj holds several
# numpy arrays over the whole ~430-player population) and Streamlit's cache
# would have to hash them on every rerun, whereas nets_ids_tuple is
# small and stable, and the expensive pieces (compute_hull_projection,
# load_static) are already independently @st.cache_data'd, so re-deriving
# them inside on a cache miss costs nothing on a cache hit. This is what
# lets an unrelated widget interaction elsewhere on the page (which reruns
# this whole script) skip rebuilding the chart's ~24 embedded base64 photos
# every time - only a real roster change does that.
# Not AI: the whole component being replaced (build_hull_scatter + the
# archetypoid list column) - the user's own change of direction.
#
# AI-ASSISTED (Claude Code, chat) - Prompt: "先隐藏掉Roster Construction" was
# followed by a separate request to drop this chart's own K=4-10 slider
# entirely (K=8 is frozen project-wide; a slider implied a choice the
# visitor never actually had) - dropped the hull_k parameter, this now
# always builds the K=8 chart.
# Used: same cache-on-small-stable-args pattern as above, now with one
# fewer argument. Not AI: the decision to remove the slider - the user's
# own call, given with the full rationale (Phase 2 diagnostics froze K=8;
# every other page in the portal already reads only the K=8 basis).
@st.cache_data
def build_intro_hull_html(nets_ids_tuple):
    proj = compute_hull_projection()
    _, _, static_labels, _ = load_static()
    roster_df = pd.DataFrame({"PLAYER_ID": list(nets_ids_tuple)})
    spec = hull_callout_chart.build_figure_spec(proj, roster_df, static_labels, k=8)
    return hull_callout_chart.render_html(spec)


def render_intro_page(roster, labels):
    st.title("Intro")

    with st.container(border=True):
        st.markdown("### What is an \"archetype\" here?")
        st.markdown(
            "ADA finds the **8 most extreme real players** in the league and expresses "
            "everyone else as a blend of them - not abstract types, real players. Below: "
            "the colored corners are those 8 archetypes, black dots are Nets players - "
            "hover either for details."
        )
        st.caption("K = 8 - selected by the Phase 2 diagnostics and matching the NBA basis.")

        proj = compute_hull_projection()

        # AI-ASSISTED (Claude Code, chat)
        # Prompt (this revision): "把这些文字都去掉 字太多了" - drop the <6-on-hull
        # warning/caption, the PCA-plane-fit caption, and the "2D projection"
        # caption under the chart entirely - not reworded/shortened, removed.
        # Not AI: which text to cut - the user's own reaction to the rendered
        # page having too much caption text.

        nets_ids_tuple = tuple(sorted(roster["PLAYER_ID"].astype(int).tolist()))
        chart_html = build_intro_hull_html(nets_ids_tuple)
        st.iframe(chart_html, height=int(hull_callout_chart.FIGURE_HEIGHT_PX * 1.05) + 20)

        st.markdown(
            "Every point above earned its coordinates by playing real NBA minutes - "
            "the incoming rookies projected on the NCAA Bridge page have none, which is "
            "exactly the gap that page's translator addresses."
        )

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
    #     st.caption(f"{K_SELECTION_PATH} not found - run step1_archetype_model.phase1_select_k() first.")
