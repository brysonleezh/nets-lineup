"""
Portal page: NCAA Bridge Rookie Projections (v1). Presents the completed
NCAA->NBA rookie-translation pipeline (Phases 1-6, see reports/phase1_data_report.md
through reports/phase6_deployment_report.md and reports/project_summary.md
for the full technical record).

Naming note: "step5" here refers to this file's position in the ARCHETYPE
PORTAL's own page sequence (step2_intro, step3_player_breakdown, step4_report,
this file) - it has NO relationship to the translator pipeline's own "Phase 5"
(holdout validation) or "Phase 6" (deployment) numbering. Two unrelated
numbering schemes that happen to collide; don't conflate them.

Unrelated-systems note (also flagged at SHOW_ROOKIE_SLOT_QUERY_PAGE's own
definition in step3_player_breakdown.py): this page and the legacy
"Building Around Rookie" page (SHOW_ROOKIE_SLOT_QUERY_PAGE, still False)
both concern the same three rookies but are otherwise unrelated systems.
The legacy page queries what archetype SLOT a rookie should fill next to
the existing roster (a lineup-construction question, using
NETS_ROSTER_NCAA_BRIDGE + the original NBA-side archetype model only).
This page shows this session's actual trained NCAA->NBA translator's own
predictions (a player-projection question, using the Phase 1-6 pipeline).
Do not merge them or assume one supersedes the other.

Architecture principle: this page READS, it does not COMPUTE. Every
number traces to a file already on disk (data/projections/, data/translator/,
data/anchors/, data/college/). The one exception is the Section 4
pick-slot counterfactual, which evaluates the frozen deployment model's
own already-fitted posterior (softmax(Bx), averaged over all 3000 stored
posterior samples - the same method that produced every frozen prediction
elsewhere in this pipeline) with only the pick-derived input features
varying - a closed-form readout of already-frozen numbers, not a new fit.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

import hull_callout_chart  # noqa: E402
import player_report  # noqa: E402
from portal_shared import BL_PAPER, BL_WHITE, BL_INK, BL_MUTED, BL_LINE, BL_GREEN, BL_CORAL, _build_sortable_table_html  # noqa: E402

# AI-ASSISTED (Claude Code, chat) - Prompt: the full "Portal Page — NCAA
# Bridge Rookie Projections (v1)" spec, given directly, requesting this new
# page be added "immediately before the Reports page" in the nav.
# Used: same "kept but not wired up until ready" flip as every other
# SHOW_* flag on this app - gates the page into the sidebar nav list.
# Not AI: the decision to build this page now (deferred earlier in this
# session until Phase 6 finished), and its placement in the nav order -
# the owner's own calls.
# AI-ASSISTED (Claude Code, chat) - Prompt: "针对NCAA Bridge可以先hidden" -
# hidden while the portal is refocused league-wide. Same gate-don't-delete
# convention: the page, the translator pipeline behind it and their tests are
# untouched, and one flip brings it back.
SHOW_ROOKIE_PROJECTIONS_PAGE = False

# AI-ASSISTED (Claude Code, chat) - Prompt: "Try a different draft slot — same
# player, different opportunity / 把这一章节去掉" (drop this section).
# Used: the same "kept but not wired up" flag convention every other hidden
# surface in this app uses - render_section4_counterfactual(), its
# closed-form posterior path (predict_mu_mean_numpy, _rookie_raw_row,
# build_raw_row_from_anchor) and run_consistency_guard() are all left fully
# built and tested; only the call site is gated, so this is a one-line flip
# to bring back.
# The consistency guard is gated with it, not left running: the guard exists
# solely to decide whether that section may render (it re-derives the frozen
# predictions through the closed-form path and refuses to show a second,
# silently-divergent set of numbers). With the section hidden it would load
# the 3000-sample posterior on every page view to answer a question nobody
# asks. tests/test_portal_rookie_page.py still exercises the guard directly,
# so hiding it here does not reduce coverage.
SHOW_PICK_SLOT_COUNTERFACTUAL = False

# AI-ASSISTED (Claude Code, chat) - Prompt: "3 · Versus the benchmarks, and
# where it can improve 直接把这个隐藏掉" (hide this outright).
# Used: the same gate-don't-delete convention - the benchmark chart, the
# accuracy diagnosis, the standing limitations and the frozen record all stay
# fully built (and their loaders/tests untouched); only the act's call site is
# gated. One flip restores it.
# Consequence worth naming: with this off the page makes NO accuracy claim at
# all - it shows the mapping and its results, and stops there. The sanctioned
# confidence statement lived in the limitations block, so it goes quiet with
# it rather than being restated somewhere less careful.
SHOW_BENCHMARK_ACT = False

K = 8


@st.cache_data
def _rookie_nba_ids() -> dict:
    """{display_name: nba_player_id} from config.yaml - the same frozen input
    Phase 6 used. Real NBA headshots exist for all three 2026 draftees; when
    one is missing the shared helper degrades to an initials avatar on its
    own, so this never has to guard."""
    import yaml
    try:
        cfg = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
        return {r["display_name"]: r.get("nba_player_id") for r in cfg.get("rookies", [])}
    except Exception:
        return {}


TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
PROJECTIONS_DIR = REPO_ROOT / "data" / "projections"
ANCHORS_DIR = REPO_ROOT / "data" / "anchors"

# AI-ASSISTED (Claude Code, chat) - Prompt: "Match the portal's existing
# visual conventions... Consistent archetype color mapping across every
# chart on the page" - found the app's own existing validated 8-color
# categorical palette (step3_player_breakdown.py's C1 role-drift chart,
# itself sourced from this project's dataviz skill and re-validated
# against BL_PAPER) already used for archetype-adjacent charts elsewhere.
# Used: reused verbatim rather than inventing a second palette.
ARCHETYPE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]


# --- data loaders (cached, graceful degradation) ----------------------------

REQUIRED_FILES = {
    "nets_rookies_2026.csv": PROJECTIONS_DIR / "nets_rookies_2026.csv",
    "predictions_frozen.json": PROJECTIONS_DIR / "predictions_frozen.json",
    "holdout_predictions.csv": TRANSLATOR_DIR / "holdout_predictions.csv",
    "holdout_metrics.csv": TRANSLATOR_DIR / "holdout_metrics.csv",
    "holdout_baseline_comparison.csv": TRANSLATOR_DIR / "holdout_baseline_comparison.csv",
    "deployment/manifest.json": TRANSLATOR_DIR / "deployment" / "manifest.json",
    "deployment/posterior.npz": TRANSLATOR_DIR / "deployment" / "posterior.npz",
    "deployment_preprocessing.json": TRANSLATOR_DIR / "deployment_preprocessing.json",
    "anchors.csv": ANCHORS_DIR / "anchors.csv",
    "recipes.csv": REPO_ROOT / "data" / "college" / "recipes.csv",
    "college_archetypes.md": REPO_ROOT / "reports" / "college_archetypes.md",
    "archetype_labels.csv": REPO_ROOT / "data" / "basis_2025_26" / "archetype_labels.csv",
    "rookie_projections_full.json": TRANSLATOR_DIR / "rookie_projections_full.json",
    "saturation_rule_evidence.csv": TRANSLATOR_DIR / "saturation_rule_evidence.csv",
}


def check_required_files() -> list[str]:
    """Returns a list of missing (label, path) - empty if everything's there."""
    return [label for label, path in REQUIRED_FILES.items() if not path.exists()]


@st.cache_data
def load_rookie_projections() -> pd.DataFrame:
    return pd.read_csv(REQUIRED_FILES["nets_rookies_2026.csv"])


@st.cache_data
def load_rookie_projections_full() -> list:
    return json.loads(REQUIRED_FILES["rookie_projections_full.json"].read_text())


@st.cache_data
def load_frozen_record() -> dict:
    return json.loads(REQUIRED_FILES["predictions_frozen.json"].read_text())


@st.cache_data
def load_holdout_predictions() -> pd.DataFrame:
    return pd.read_csv(REQUIRED_FILES["holdout_predictions.csv"])


@st.cache_data
def load_holdout_metrics() -> pd.DataFrame:
    return pd.read_csv(REQUIRED_FILES["holdout_metrics.csv"])


@st.cache_data
def load_baseline_comparison() -> pd.DataFrame:
    return pd.read_csv(REQUIRED_FILES["holdout_baseline_comparison.csv"])


@st.cache_data
def load_deployment_manifest() -> dict:
    return json.loads(REQUIRED_FILES["deployment/manifest.json"].read_text())


@st.cache_resource
def load_deployment_posterior() -> tuple[np.ndarray, np.ndarray]:
    with np.load(REQUIRED_FILES["deployment/posterior.npz"]) as z:
        return z["B"], z["phi"]


@st.cache_data
def load_deployment_preprocessing() -> dict:
    return json.loads(REQUIRED_FILES["deployment_preprocessing.json"].read_text())


@st.cache_data
def load_anchors() -> pd.DataFrame:
    return pd.read_csv(REQUIRED_FILES["anchors.csv"])


@st.cache_data
def load_college_recipes() -> pd.DataFrame:
    return pd.read_csv(REQUIRED_FILES["recipes.csv"])


@st.cache_data
def load_labels() -> tuple[dict, dict]:
    """(college_labels, nba_labels), both {index: label_string}.
    load_nba_labels() itself returns a richer {index: {label, exemplar,
    match_quality, note}} dict (used separately by load_archetypoids() for
    "closest real comparison" prose) - flattened to just the label string
    here so every chart/table call site can treat college_labels and
    nba_labels identically, rather than each caller needing to know one of
    the two is secretly a dict of dicts."""
    from rookie_card_labels import load_college_labels, load_nba_labels
    nba_labels_full = load_nba_labels()
    nba_labels = {j: v["label"] for j, v in nba_labels_full.items()}
    # Cleaned HERE rather than at each chart call site: three college labels
    # carry a "(weak - see note above)" aside written for a reader of
    # reports/college_archetypes.md, and every consumer of this function renders
    # labels straight onto an axis where that note does not exist. Doing it at
    # the single load point means no future call site can reintroduce the
    # dangling reference by forgetting to sanitize.
    return clean_labels(load_college_labels()), clean_labels(nba_labels)


@st.cache_data
def load_archetypoids() -> dict:
    """{index: {player, season, team}} - the real nearest-player exemplar
    per NBA archetype, for "closest real comparison" prose."""
    from rookie_card_labels import load_nba_archetypoids
    return load_nba_archetypoids()


@st.cache_data
def load_saturation_evidence() -> pd.DataFrame:
    return pd.read_csv(REQUIRED_FILES["saturation_rule_evidence.csv"])


@st.cache_data
def load_rookie_pick_numbers() -> dict:
    """{display_name: draft_pick_overall} - not in any of the listed data
    sources (nets_rookies_2026.csv doesn't carry the pick number), so this
    reads config.yaml, the same static input file Phase 6 itself used to
    build these predictions - still a read of an already-frozen input,
    not a computation."""
    import yaml
    config = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
    return {r["display_name"]: r["draft_pick_overall"] for r in config["rookies"]}


# --- label hygiene -----------------------------------------------------------

# AI-ASSISTED (Claude Code, chat) - Prompt: "整个页面关于NCAA Bridge的flow很乱
# 我想把这部分重写改一下". Found while auditing the page: three of the college
# archetype labels in reports/college_archetypes.md carry an editorial aside
# aimed at a READER OF THAT REPORT - "(weak - see note above)" - and those
# strings were being rendered verbatim onto this page's chart axes, where no
# such note exists. A dangling cross-reference in the UI reads as a bug to a
# visitor and as sloppiness to a reviewer.
# Used: strip the "see note above" pointer while KEEPING the "(weak)" marker
# itself - that marker is real, load-bearing information (those archetypes are
# genuinely noisy and the project says so openly), so it is trimmed, not hidden.
# Not AI: the labels' own content and the decision to flag those archetypes as
# weak in the first place - the owner's, in the source report.
_LABEL_ASIDE = re.compile(r"\s*\(\s*(weak[^)]*?)\s*[—-]\s*see note above\s*\)", re.IGNORECASE)


def clean_label(label: str) -> str:
    """Archetype label as it should appear IN THE UI: '(weak - see note above)'
    -> '(weak)'. Pointers to a report the visitor cannot see are removed; the
    weak/catch-all qualifier itself is preserved."""
    if not isinstance(label, str):
        return label
    return _LABEL_ASIDE.sub(lambda m: f" ({m.group(1).strip()})", label).strip()


def clean_labels(labels: dict) -> dict:
    return {j: clean_label(v) for j, v in labels.items()}


# --- shared chart helper -----------------------------------------------------

def archetype_bar_chart(mix: np.ndarray, nba_labels: dict, height: int = 320,
                          title: str | None = None,
                          xaxis_title: str = "predicted weight") -> go.Figure:
    """Horizontal bar chart, one bar per archetype, sorted descending,
    colored by the app's shared ARCHETYPE_COLORS palette, labeled with
    real archetype names (never a bare index).

    xaxis_title defaults to "predicted weight" so every existing call site on
    this page is unchanged, but it MUST be overridden for a chart showing a
    measured recipe. step6_draft_class.py puts a measured college recipe and a
    predicted NBA recipe side by side, and labelling the measured one
    "predicted" would blur the single distinction that page exists to draw."""
    order = np.argsort(-mix)
    names = [nba_labels.get(int(j), f"archetype {j}") for j in order]
    values = mix[order] * 100
    colors = [ARCHETYPE_COLORS[int(j) % 8] for j in order]
    fig = go.Figure(go.Bar(
        x=values, y=names, orientation="h",
        marker_color=colors,
        text=[f"{v:.0f}%" for v in values], textposition="outside",
        hovertemplate="%{y}: %{x:.1f}%<extra></extra>",
    ))
    fig.update_layout(
        height=height, margin=dict(l=10, r=40, t=30 if title else 10, b=10),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(title=xaxis_title, range=[0, max(values.max() * 1.25, 10)], gridcolor=BL_LINE),
        yaxis=dict(autorange="reversed"),
    )
    if title:  # passing title=None explicitly to update_layout rendered a literal
        # "undefined" string in the chart (a real, confirmed-live Plotly.js quirk in
        # this version) - only set the key at all when there's a real title to show
        fig.update_layout(title=title)
    return fig


def fmt_top3(top3: list, labels: dict) -> str:
    parts = []
    for j, w in top3:
        parts.append(f"**{labels.get(int(j), f'archetype {j}')}** ({w*100:.0f}%)")
    return " → ".join(parts)


# --- Section 1: the three rookie projections --------------------------------

# AI-ASSISTED (Claude Code, chat) - Prompt: a design handoff bundle
# (design_handoff_rookie_archetype_cards: README.md + a .dc.html reference)
# for this section, given directly - "针对我页面中NCAA Bridge中 1, What the
# model says修改". The handoff replaces the section's three horizontal bar
# charts with compact profile cards: circular headshot, name, meta line, a
# donut of the top-3 archetype weights with the remainder grouped as
# "Others", and a legend.
# Used: the handoff's layout, spacing and type scale as specified (card
# padding 36/30/32, headshot 104px, donut 150px with a 26px hole, legend
# rows 9px apart, grid auto-fit minmax(300px,1fr) at 24px gap, max-width
# 1120px). The donut is a pure CSS conic-gradient - no charting library and
# no JS - so it renders through st.markdown without an iframe, and the
# rank-by-darkness encoding the handoff calls out (segment 1 darkest) is
# preserved.
# COLOR: the handoff is explicit that its cream/amber palette is placeholder
# and that all colors must be derived from the host app's own tokens, by
# ROLE not by hex. Mapped accordingly: paper->BL_PAPER, card->BL_WHITE,
# ink->BL_INK, muted->BL_MUTED, hairline->BL_LINE, ramp-1/2/3 -> three tones
# of this app's primary green (dark->light), others -> BL_INK at 10%. The
# green ramp is the same one step2_intro.py already uses to mark projected
# recipes, so "modelled, not measured" reads consistently across both pages.
# TYPE: the handoff names Newsreader/IBM Plex Mono but defers to "the app's
# existing font stack if it has one". Loading two webfonts would add an
# external network dependency to a page that currently has none, so the
# serif/mono CONTRAST the design depends on is kept via system stacks
# instead of the named families.
# Not AI: the design itself - layout, type scale, the donut-with-Others
# encoding, and the decision to apply it to this section - all the owner's,
# via the handoff.
CARD_RAMP = ["#004b2b", "#3f7d5c", "#8fb3a1"]   # ramp-1/2/3, darkest = top archetype
CARD_OTHERS = "rgba(32,36,42,0.10)"             # BL_INK at ~10%
CARD_SERIF = "Georgia, 'Times New Roman', serif"
CARD_MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def _legend_row_html(label: str, pct: int, swatch: str, color: str,
                     emphasis: bool = False) -> str:
    """emphasis=True is the owner's "前三的纬度使用加粗更大字体" rule: the three
    archetypes that actually carry the projection are set bigger and bolder
    than the rest, so the recipe's shape is readable without reading numbers."""
    # Sized for the three-across card, not the earlier full-width row: the
    # emphasis is RELATIVE, so it has to stay below the 19px player name
    # above it or the legend outshouts the card's own title.
    name_size = "15px" if emphasis else "12.5px"
    name_weight = "600" if emphasis else "400"
    pct_size = "13px" if emphasis else "11px"
    pct_weight = "600" if emphasis else "500"
    dot = "10px" if emphasis else "8px"
    return (
        '<div style="display:flex;align-items:center;gap:10px;">'
        f'<span style="width:{dot};height:{dot};border-radius:2px;flex:none;background:{swatch};"></span>'
        f'<span style="flex:1;text-align:left;font-size:{name_size};font-weight:{name_weight};'
        f'line-height:1.35;color:{color};">{label}</span>'
        f'<span style="font:{pct_weight} {pct_size} {CARD_MONO};font-variant-numeric:tabular-nums;'
        f'color:{color};">{pct}%</span>'
        '</div>'
    )


# AI-ASSISTED (Claude Code, chat) - Prompt: "把三个卡片铺开 并且针对显示球员
# receipt 我希望画这个图像 然后因为我们一共有8个纬度 在8个纬度显示球员的能力
# 并且前三的纬度使用加粗更大字体的展示这个receipt 然后我希望 profile + chart
# 是左右排列显示 不是上下的" - given with a reference screenshot of this app's
# own signature radar (step3_player_breakdown.signature_radar_chart).
# Used: a radar over the 8 NBA archetypes, matching that existing chart's
# conventions (Scatterpolar, closed loop, BL_GREEN line + translucent fill,
# BL_PAPER polar background, categoryorder="array" - that last one is not
# cosmetic: step3's own comment records that Plotly silently regroups a
# categorical angular axis once the first category is repeated to close the
# loop, so the explicit order is what keeps the axes where they were put).
# TWO DELIBERATE CHOICES THE PROMPT DID NOT SPECIFY, both so the three
# radars can actually be compared side by side rather than merely coexist:
#   - the 8 axes are in FIXED archetype order for every player (not sorted
#     by each player's own weights) - sorting per player would put a
#     different archetype at 12 o'clock on each chart, and the shapes would
#     be visually comparable while meaning nothing;
#   - the radial range is shared across all three, so a 55% lobe on one
#     chart is the same distance from centre as a 55% lobe on another.
# Not AI: the radar itself, the 8-dimension framing, the bold/larger
# treatment for the top 3, and the profile-left/chart-right layout - all
# given directly.
def recipe_radar_chart(mix: np.ndarray, nba_labels: dict, r_max: float,
                       height: int = 300) -> go.Figure:
    """Radar of one rookie's projected recipe across all 8 NBA archetypes.
    The three heaviest axes are labelled bold and a size larger; `r_max` is
    passed in (not derived per player) so every rookie's chart shares one
    radial scale.

    Axis labels use player_report.ARCHETYPE_ABBREV - the project's own,
    already-used short forms - because three side-by-side radars leave each
    chart ~350px wide and the full names ("Rim Protector / Roll Man",
    "Traditional Playmaker") collide with each other and with the plot at
    that size. Reusing the existing map rather than inventing a second set
    of short names keeps the radar consistent with the PDF report, which
    abbreviates the same eight archetypes the same way."""
    top3 = set(int(j) for j in np.argsort(-mix)[:3])
    labels = []
    for j in range(K):
        short = player_report.abbreviate_archetype(nba_labels.get(j, f"archetype {j}"))
        if j in top3:
            # Plotly.js renders this HTML subset in categorical tick labels;
            # the category string doubles as the label, so the emphasis has
            # to live in the string itself.
            labels.append(f'<b><span style="font-size:12.5px">{short}</span></b>')
        else:
            labels.append(f'<span style="font-size:10px">{short}</span>')

    vals = [float(mix[j]) * 100 for j in range(K)]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals + [vals[0]], theta=labels + [labels[0]],
        mode="lines+markers",
        line=dict(color=BL_GREEN, width=2),
        fill="toself", fillcolor="rgba(0,75,43,0.22)",
        marker=dict(size=5, color=BL_GREEN),
        hovertemplate="%{theta}: %{r:.0f}%<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(
        height=height, margin=dict(l=58, r=58, t=28, b=28),
        paper_bgcolor=BL_PAPER, font_color=BL_INK,
        polar=dict(
            bgcolor=BL_PAPER,
            # dtick=20 rather than Plotly's default ~5 ticks: the radial
            # labels sit over the filled shape, and at 10% intervals they
            # crowded the very lobes they were there to measure.
            radialaxis=dict(range=[0, r_max], color=BL_MUTED, gridcolor=BL_LINE,
                            tickfont=dict(size=8), ticksuffix="%", dtick=20,
                            angle=90, tickangle=90),
            angularaxis=dict(color=BL_INK, gridcolor=BL_LINE,
                             categoryorder="array", categoryarray=labels),
        ),
    )
    return fig


def rookie_profile_html(name: str, meta: str, photo_uri: str, top3: list,
                        others: int) -> str:
    """The identity half of a rookie card: headshot, name, meta, and the
    recipe as a legend, with the radar rendered directly beneath it. Sized
    for a three-across layout, so every dimension here is smaller than the
    single-row version this replaced. Top-3 rows stay bold and a size larger
    than the rest (the owner's rule) so the three archetypes that carry the
    projection still read first at the reduced size."""
    legend = "".join(
        _legend_row_html(lbl, pct, CARD_RAMP[i] if i < len(CARD_RAMP) else CARD_OTHERS,
                         BL_INK, emphasis=True)
        for i, (lbl, pct) in enumerate(top3)
    )
    legend += _legend_row_html("Others", others, CARD_OTHERS, BL_MUTED, emphasis=False)

    return (
        f'<div style="display:flex;flex-direction:column;align-items:center;text-align:center;'
        f'padding:20px 18px 4px;">'
        f'<div style="width:76px;height:76px;border-radius:50%;border:1px solid {BL_LINE};'
        f'padding:3px;flex:none;">'
        f'<img src="{photo_uri}" alt="" style="width:100%;height:100%;border-radius:50%;'
        f'object-fit:cover;object-position:top center;display:block;">'
        f'</div>'
        f'<div style="margin:12px 0 0;font-family:{CARD_SERIF};font-size:19px;font-weight:600;'
        f'letter-spacing:-0.01em;color:{BL_INK};line-height:1.2;">{name}</div>'
        f'<div style="margin-top:5px;font:400 10.5px {CARD_MONO};color:{BL_MUTED};">{meta}</div>'
        f'<div style="width:100%;max-width:240px;margin-top:16px;display:flex;'
        f'flex-direction:column;gap:7px;">{legend}</div>'
        f'</div>'
    )


def comp_card_html(name: str, sub: str, college_role: str, nba_role: str, pct: int) -> str:
    """One comparable player, as a card rather than a table row: who he was in
    COLLEGE, and what he actually became as an NBA rookie. The table this
    replaced only ever showed the NBA side, which is the half that isn't the
    transformation - the college role is where the change starts, and it is
    already in the data (`college_top_archetype`), just never displayed."""
    return (
        f'<div style="background:{BL_WHITE};border:1px solid {BL_LINE};border-radius:10px;'
        f'padding:11px 13px;margin-bottom:8px;">'
        f'<div style="display:flex;align-items:baseline;justify-content:space-between;gap:8px;">'
        f'<span style="font-family:{CARD_SERIF};font-size:15px;font-weight:600;'
        f'color:{BL_INK};line-height:1.2;">{name}</span>'
        f'<span style="font:400 10px {CARD_MONO};color:{BL_MUTED};white-space:nowrap;">{sub}</span>'
        f'</div>'
        # EVERY span carries an explicit font-size. This app's global CSS sets
        # `[data-testid="stMarkdownContainer"] span { font-size: 18px }`, so a
        # span styled for colour/weight alone silently renders at 18px - which
        # is exactly what happened here on first render: the NBA role came out
        # visibly LARGER than the college role it was supposed to follow.
        f'<div style="margin-top:7px;font-size:12.5px;line-height:1.5;color:{BL_MUTED};">'
        f'<span style="font-size:12.5px;color:{BL_MUTED};">{college_role}</span>'
        f'<span style="font-size:12.5px;color:{BL_GREEN};font-weight:700;padding:0 5px;">&rarr;</span>'
        f'<span style="font-size:12.5px;color:{BL_INK};font-weight:600;">{nba_role}</span> '
        f'<span style="font:600 11.5px {CARD_MONO};color:{BL_INK};'
        f'font-variant-numeric:tabular-nums;">{pct}%</span>'
        f'</div>'
        f'</div>'
    )


def render_act_header(index: str, title: str) -> None:
    """The one heading style for all three acts: a mono index baseline-aligned
    with a serif title.

    Acts 2 and 3 used `st.markdown("## ...")` until the owner asked for one
    consistent size - and that was never going to match by tweaking numbers,
    because this app's global CSS forces `h2 { font-size: 2rem !important;
    font-weight: 700 !important }`. Rendering all three through the same divs
    is what actually makes them identical."""
    st.markdown(
        f'<div style="display:flex;align-items:baseline;gap:16px;margin-top:4px;">'
        f'<span style="font:500 12px {CARD_MONO};letter-spacing:.2em;color:{BL_MUTED};">{index}</span>'
        f'<span style="font-family:{CARD_SERIF};font-size:28px;font-weight:600;'
        f'letter-spacing:-0.01em;color:{BL_INK};">{title}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_rookie_cards(projections: list, college_labels: dict, nba_labels: dict,
                        pick_numbers: dict) -> None:
    """Three cards left to right, in draft order. Each is profile on top and
    its 8-archetype radar directly beneath, with that player's comparable
    college profiles collapsed under his own column."""
    nba_ids = _rookie_nba_ids()

    # Fixed display order, by name, so the three always read Brown ->
    # Bilodeau -> Jefferson regardless of how the source file happens to be
    # ordered (it is a pipeline output, not a display artifact).
    order = {"Mikel Brown Jr.": 0, "Tyler Bilodeau": 1, "Joshua Jefferson": 2}
    ordered = sorted(projections, key=lambda r: order.get(r["display_name"], 99))

    # One radial scale for all three charts, computed across every rookie, so
    # a 55% lobe on one chart sits at the same distance from centre as a 55%
    # lobe on another. Per-player autoscaling would make three charts that
    # look comparable and are not.
    peak = max(float(np.max(np.array(r["y_pred"]))) for r in ordered) * 100
    r_max = float(np.ceil(peak / 10.0) * 10)

    for col, rookie in zip(st.columns(3, gap="medium"), ordered):
        with col:
            name = rookie["display_name"]
            mix = np.array(rookie["y_pred"])
            top3 = [(nba_labels.get(int(j), f"archetype {j}"), int(round(float(mix[j]) * 100)))
                    for j in np.argsort(-mix)[:3]]
            others = 100 - sum(p for _, p in top3)
            meta = (f"{rookie['college_team']} ({rookie['conference']}) · "
                    f"pick #{pick_numbers.get(name, '?')}")
            photo = hull_callout_chart.get_headshot_data_uri(nba_ids.get(name), name)

            with st.container(border=True):
                st.markdown(rookie_profile_html(name, meta, photo, top3, others),
                            unsafe_allow_html=True)
                st.plotly_chart(recipe_radar_chart(mix, nba_labels, r_max),
                                width="stretch", key=f"s1_radar_{name}")

            with st.expander("Similar college profiles — what they became"):
                for c in rookie["comps"]:
                    nba_j, nba_w = c["true_rookie_top2"][0]
                    st.markdown(
                        comp_card_html(
                            # display-only: a few comp names carry a double
                            # space in the source data ("Antonio  Reeves")
                            " ".join(str(c["name"]).split()),
                            f"{c['draft_year']} · #{c['pick']} · {c['college']}",
                            clean_label(college_labels.get(int(c["college_top_archetype"]),
                                                           "—")),
                            nba_labels.get(int(nba_j), f"archetype {nba_j}"),
                            int(round(float(nba_w) * 100)),
                        ),
                        unsafe_allow_html=True,
                    )


def render_section1_projections(projections: list, college_labels: dict, nba_labels: dict,
                                  archetypoids: dict, pick_numbers: dict) -> None:
    """ACT 1 - the answer. The prose that used to sit here (a paragraph
    narrating each rookie's certainty level) was removed at the owner's
    request: the cards state the same thing in the numbers, and the radar
    shows it in the shape."""
    render_rookie_cards(projections, college_labels, nba_labels, pick_numbers)


# --- Section 2: why naive relabeling fails (headline chart) -----------------

BASELINE_DISPLAY_NAMES = {
    "T1_b (chosen model)": "This model",
    "T4 Hungarian": "Naive structural matching (Hungarian)",
    "T5(i) global mean": "Guess the league-average rookie",
    "T5(ii) height tercile": "Guess by height tercile",
}


def render_section2_headline(baseline_df: pd.DataFrame) -> None:
    """ACT 2, part 1 - why a learned translator is needed at all. No st.header
    here: Act 2 owns one header, and this and the holdout browser below it are
    two halves of the same question ("can you trust this?"), which the old
    six-flat-header layout split into two unrelated-looking sections."""
    st.caption("Could we not just relabel college archetypes as NBA ones?")

    df = baseline_df.copy()
    df["display"] = df["model"].map(BASELINE_DISPLAY_NAMES).fillna(df["model"])
    df = df[df["model"] != "T5(ii) height tercile"]  # keep the headline to the 3 named in the spec table
    df = df.sort_values("top1_hit_mean")

    colors = [BL_CORAL if "This model" in d else BL_MUTED for d in df["display"]]
    fig = go.Figure(go.Bar(
        x=df["top1_hit_mean"] * 100, y=df["display"], orientation="h",
        marker_color=colors, text=[f"{v:.1f}%" for v in df["top1_hit_mean"] * 100],
        textposition="outside",
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=60, t=10, b=10),
        plot_bgcolor=BL_PAPER, paper_bgcolor=BL_PAPER, font_color=BL_INK,
        xaxis=dict(title="top-1 hit rate on the 36-player 2025 holdout", range=[0, 65], gridcolor=BL_LINE),
    )
    st.plotly_chart(fig, width="stretch", key="s2_headline")

    st.caption(
        "Roles transform on the way up, so similarity-matching loses to guessing the average "
        "rookie. That asymmetry is what the translator learns."
    )
    # The JSD line that used to sit here is gone: act 3's metric row already
    # shows this model's JSD, and repeating all four models' figures as prose
    # was the single densest sentence on the page.


# --- Section 3: holdout browser ----------------------------------------------

def _jensen_shannon_row(p: np.ndarray, q: np.ndarray) -> float:
    p = np.clip(p, 1e-12, None); p = p / p.sum()
    q = np.clip(q, 1e-12, None); q = q / q.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float((a * (np.log(a) - np.log(b))).sum())
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)) / np.log(2)


def render_section3_holdout_browser(holdout_df: pd.DataFrame, college_labels: dict, nba_labels: dict,
                                      anchors_df: pd.DataFrame) -> None:
    # No heading or preamble here any more: this renders inside an expander
    # already labelled "All 36 held-out players", under an act whose caption
    # states the hold-out discipline. Repeating it was two-thirds of the prose
    # on the page.

    df = holdout_df.copy()
    # holdout_predictions.csv (a Phase 5 deliverable) carries no college name column -
    # anchors.csv (already loaded elsewhere on this page) has one and covers all 36
    # holdout players by player_name with zero misses; joined here for display only,
    # doesn't touch either underlying artifact.
    df = df.merge(anchors_df[["player_name", "college_team"]], on="player_name", how="left")
    df["jsd"] = [_jensen_shannon_row(np.array([r[f"y_pred_{j}"] for j in range(K)]),
                                       np.array([r[f"y_true_{j}"] for j in range(K)]))
                 for _, r in df.iterrows()]
    df["hit"] = df["pred_argmax"] == df["true_argmax"]
    df = df.sort_values("jsd")  # default sort: JSD ascending, per spec

    columns = [("Player", "name"), ("Pick", "pick"), ("College", None), ("Predicted", None),
               ("Actual", None), ("Hit", "hit"), ("JSD", "jsd")]
    rows_cells = []
    for _, r in df.iterrows():
        pred_label = nba_labels.get(int(r["pred_argmax"]), str(r["pred_argmax"]))
        true_label = nba_labels.get(int(r["true_argmax"]), str(r["true_argmax"]))
        hit_html = f'<span style="color:{BL_GREEN};font-weight:700;">✓</span>' if r["hit"] else \
                   f'<span style="color:{BL_CORAL};font-weight:700;">✗</span>'
        rows_cells.append([
            (r["player_name"], r["player_name"]),
            (f"#{int(r['pick_overall'])}", int(r["pick_overall"])),
            (str(r["college_team"]) if pd.notna(r["college_team"]) else "—", None),
            (pred_label, None),
            (true_label, None),
            (hit_html, 1 if r["hit"] else 0),
            (f"{r['jsd']:.3f}", r["jsd"]),
        ])
    table_html, iframe_height = _build_sortable_table_html("holdoutBrowserTable", columns, rows_cells,
                                                             row_height=34, font_size_px=14)
    # No expander of its own: the CALLER now wraps this whole function in one
    # ("All 36 held-out players"), and Streamlit raises on a nested expander.
    st.iframe(table_html, height=iframe_height)  # uncapped, matching every other
    # _build_sortable_table_html caller in this app - st.iframe() in this Streamlit
    # version has no `scrolling` kwarg (confirmed: TypeError on a first attempt), so
    # capping the height without a way to scroll would just clip the table



# --- Section 4: pick-slot counterfactual -------------------------------------

# AI-ASSISTED (Claude Code, chat) - Prompt: the spec's counterfactual
# architecture described softmax(B_mean . x) using the posterior-MEAN
# coefficient matrix; found this disagrees with nets_rookies_2026.csv's
# actual frozen predictions (E[softmax(Bx)], averaged over all 3000
# posterior samples) by up to 0.58 percentage points on real data - a
# genuine Jensen's-inequality gap, not a bug (see worklog). Owner decided:
# use the full posterior (E[softmax(Bx)]) for exact agreement.
# Used: pure-numpy implementation (no jax/numpyro import in this file at
# all) - the forward pass is a small, fixed matrix operation over the
# already-stored 3000 samples (~460K flops), not a new model fit or
# sampling step, so plain numpy is both correct and avoids jax's own
# import/JIT startup cost on every portal page load.
# Not AI: the decision to use the full posterior over the mean-only
# reading - the owner's own call, made after the gap was measured and
# presented.
def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def predict_mu_mean_numpy(X: np.ndarray, B_samples: np.ndarray) -> np.ndarray:
    """X: (n, p) standardized inputs, NO intercept column (added here).
    B_samples: (S, p+1, K-1). Returns (n, K) posterior-mean mu - the exact
    same quantity (same averaging order) as every frozen prediction in
    this pipeline."""
    Xaug = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    logits_free = np.einsum("np,spk->snk", Xaug, B_samples)
    logits = np.concatenate([logits_free, np.zeros((*logits_free.shape[:-1], 1))], axis=-1)
    mu = softmax_np(logits)
    return mu.mean(axis=0)


def build_raw_row_from_anchor(anchor_row: pd.Series) -> dict:
    """anchors.csv already has every raw feature apply_frozen_transform
    needs, just under its own column names - maps them once here."""
    from input_variants import Z_FEATURE_COLS, SHOT_TYPE_COLS
    row = {c: anchor_row[c] for c in Z_FEATURE_COLS}
    row["overall"] = float(anchor_row["overall"])
    row["log_pick"] = float(np.log(anchor_row["overall"]))
    row["age_at_draft"] = float(anchor_row["age_at_draft"])
    row["years_in_college"] = float(anchor_row["years_in_college"])
    row["PORPAG"] = float(anchor_row["PORPAG"])
    row["rim_finishing_share"] = anchor_row.get("rim_finishing_share")
    row["three_pt_jumper_share"] = anchor_row.get("three_pt_jumper_share")
    row["conf_tier"] = bool(anchor_row["conf_tier"])
    return row


def _rookie_raw_row(display_name: str, config: dict, pick_override: float | None = None) -> dict:
    """Rebuilds one rookie's raw feature row from the same sources Phase 6
    Step 2 used (config.yaml + shared_features.parquet + roster startSeason
    data), optionally overriding the pick number - the single shared path
    used by both the consistency guard and Section 4's live slider, so
    there's exactly one implementation of "how a rookie's row is built,"
    not two that could silently drift apart."""
    from phase6_step2_predict_rookies import build_rookie_raw_row, roster_min_start_by_athlete_id
    from datetime import date

    sf = pd.read_parquet(REPO_ROOT / "data" / "college" / "shared_features.parquet")
    roster_starts = roster_min_start_by_athlete_id()
    draft_date = date.fromisoformat(config["draft_date_2026"])
    rookie_cfg = next(r for r in config["rookies"] if r["display_name"] == display_name)
    row = build_rookie_raw_row(rookie_cfg, sf, None, roster_starts, draft_date)
    if pick_override is not None:
        row["overall"] = float(pick_override)
        row["log_pick"] = float(np.log(pick_override))
    return row


def run_consistency_guard(projections_full: list, preprocessing: dict, B_samples: np.ndarray,
                           config: dict, tolerance: float = 1e-6) -> dict:
    """Evaluates the closed-form counterfactual path for the 3 real
    rookies at their REAL pick numbers and checks element-wise agreement
    with the frozen y_pred in rookie_projections_full.json. Pure function
    (nothing from st.* touched) so this is directly unit-testable without
    Streamlit and without a running portal."""
    from phase6_step2_predict_rookies import apply_frozen_transform

    max_diff = 0.0
    details = []
    for rookie in projections_full:
        name = rookie["display_name"]
        row = _rookie_raw_row(name, config)  # no override - real pick number, from config itself
        X = apply_frozen_transform([row], preprocessing)
        mu = predict_mu_mean_numpy(X, B_samples)[0]

        frozen = np.array(rookie["y_pred"])
        diff = float(np.abs(mu - frozen).max())
        max_diff = max(max_diff, diff)
        details.append({"name": name, "max_diff": diff, "pick": row["overall"]})

    return {"passed": max_diff < tolerance, "max_diff": max_diff, "tolerance": tolerance, "details": details}


def render_section4_counterfactual(anchors_df: pd.DataFrame, projections_full: list,
                                     preprocessing: dict, B_samples: np.ndarray,
                                     college_labels: dict, nba_labels: dict,
                                     pick_numbers: dict, guard_result: dict) -> None:
    st.markdown(
        "Same college statistical profile, different draft slot — draft position is one of the "
        "strongest predictors in the model, because it determines the opportunity a rookie is "
        "handed. **Trae Young (pick 5) kept the offense as a rookie (77% concentrated on his top "
        "archetype); Jalen Brunson (pick 33), a similar-shaped college engine, came off the bench "
        "and diluted into a far more spread rookie mix (55% on the same top archetype, real weight "
        "on three others)** — same top archetype, very different concentration."
    )

    if not guard_result["passed"]:
        st.warning(
            "⚠️ This section is disabled: the closed-form counterfactual path did not reproduce the "
            "frozen predictions within tolerance on page load (see the portal worklog for the "
            "logged diagnostic). Showing a second, silently-divergent set of numbers next to the "
            "frozen ones would be worse than not showing this section at all."
        )
        return

    pool_names = sorted(anchors_df["player_name"].unique().tolist())
    rookie_names = [p["display_name"] for p in projections_full]
    all_names = rookie_names + pool_names
    default_name = rookie_names[0]

    selected_name = st.selectbox("College player-season", all_names,
                                  index=all_names.index(default_name), key="s4_player_select")
    pick_slider = st.slider("Draft pick", 1, 60,
                             value=int(pick_numbers.get(selected_name, 30)), key="s4_pick_slider")

    from phase6_step2_predict_rookies import apply_frozen_transform

    if selected_name in rookie_names:
        rookie = next(p for p in projections_full if p["display_name"] == selected_name)
        c_alpha = np.array(rookie["c_alpha"])
        c_argmax = rookie["c_argmax"]
        import yaml
        config = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
        row = _rookie_raw_row(selected_name, config, pick_override=pick_slider)
    else:
        anchor_row = anchors_df[anchors_df["player_name"] == selected_name].iloc[0]
        c_alpha = np.array([anchor_row[f"c_alpha_{j}"] for j in range(K)])
        c_argmax = int(anchor_row["c_argmax"])
        row = build_raw_row_from_anchor(anchor_row)
        row["overall"] = float(pick_slider)
        row["log_pick"] = float(np.log(pick_slider))

    X = apply_frozen_transform([row], preprocessing)
    mu = predict_mu_mean_numpy(X, B_samples)[0]

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**{selected_name}'s college profile (fixed input)**")
        # xaxis_title overridden: this is his MEASURED college profile (the fixed
        # input to the counterfactual), not a prediction - the helper's default
        # label would have called it "predicted weight" right next to a chart
        # that genuinely is predicted, blurring the one distinction that matters.
        st.plotly_chart(archetype_bar_chart(c_alpha, {j: clean_label(college_labels.get(j, f"archetype {j}"))
                         for j in range(K)}, height=260, xaxis_title="measured weight"),
                        width="stretch", key="s4_college_bar")
    with col2:
        st.markdown(f"**Predicted rookie mix at pick #{pick_slider}**")
        st.plotly_chart(archetype_bar_chart(mu, nba_labels, height=260), width="stretch", key="s4_pred_bar")

    # extrapolation flag: has any training anchor with this SAME college archetype
    # ever been drafted in this pick's bucket?
    bucket = "1-14" if pick_slider <= 14 else ("15-30" if pick_slider <= 30 else "31-60")
    anchors_bucket = anchors_df.copy()
    anchors_bucket["bucket"] = pd.cut(anchors_bucket["overall"], bins=[0, 14, 30, 60],
                                        labels=["1-14", "15-30", "31-60"])
    same_archetype_here = anchors_bucket[(anchors_bucket["c_argmax"] == c_argmax) &
                                          (anchors_bucket["bucket"] == bucket)]
    if len(same_archetype_here) == 0:
        st.caption(
            f"⚠️ **Extrapolation**: no training anchor whose college profile's top archetype is "
            f"\"{clean_label(college_labels.get(c_argmax, c_argmax))}\" was ever drafted in the #{bucket} range — "
            f"this combination doesn't exist in the training data. Read this prediction with extra caution."
        )
    else:
        st.caption(
            f"{len(same_archetype_here)} training anchor(s) with this same college archetype were "
            f"drafted in the #{bucket} range — this combination is represented in training data."
        )


# --- Act 2/3 support: the gap, and the honest accuracy read ------------------

# AI-ASSISTED (Claude Code, chat) - Prompt: "2 · Why you should believe it /
# 3 · Where it breaks, and what was frozen 我觉得这两点表现的很奇怪 修改一下
# 2, 展示我的model把collenge player project => rookie season的结果和真实
# rookie season的差距 3, VS benchmark 我这个model都做了什么 并且相比于普通的
# 有什么进步 ... 准确率是不是还是有点低 哪里可以进行提高".
# Used: the two acts re-partitioned along the owner's own split - act 2 is now
# purely "projection vs. what actually happened", and the benchmark comparison
# (which used to open act 2) moves into act 3 where it belongs alongside "what
# did this model actually buy us". The accuracy diagnosis below is new, and
# every figure in it is computed from the frozen holdout artifact at render
# time rather than asserted.
# Not AI: the re-partition itself and the question "is the accuracy still a
# bit low, where can it be improved" - the owner's own, and the reason this
# diagnosis exists at all.

CONCENTRATION_BANDS = [
    (0.00, 0.35, "Genuinely hybrid (top weight &lt; 35%)"),
    (0.35, 0.55, "Moderately defined (35-55%)"),
    (0.55, 1.01, "Clear-cut (&gt; 55%)"),
]


@st.cache_data
def holdout_error_profile() -> dict:
    """Where the 36 holdout misses actually fall. Read straight off the frozen
    predictions - no model is run here."""
    hp = load_holdout_predictions()
    P = hp[[f"y_pred_{j}" for j in range(K)]].values
    T = hp[[f"y_true_{j}" for j in range(K)]].values
    pa, ta = P.argmax(1), T.argmax(1)
    hit = pa == ta
    tmax = T.max(1)

    by_true = {}
    for j in range(K):
        m = ta == j
        if m.sum():
            by_true[j] = (int(m.sum()), float(hit[m].mean()))

    confusions = {}
    for t, p in zip(ta[~hit], pa[~hit]):
        confusions[(int(t), int(p))] = confusions.get((int(t), int(p)), 0) + 1

    bands = []
    for lo, hi, label in CONCENTRATION_BANDS:
        m = (tmax >= lo) & (tmax < hi)
        if m.sum():
            bands.append((label, int(m.sum()), float(hit[m].mean())))

    return {
        "n": int(len(hp)),
        "hit": float(hit.mean()),
        "by_true": by_true,
        "confusions": sorted(confusions.items(), key=lambda kv: -kv[1]),
        "bands": bands,
        "true_peak": float(tmax.mean()),
        "pred_peak": float(P.max(1).mean()),
    }


def _role_panel_html(kicker: str, role: str, pct: str, accent: str, dim: bool = False) -> str:
    """One of the three stages in a player's story. Same box three times so the
    eye compares roles, not layouts."""
    body = BL_MUTED if dim else BL_INK
    return (
        f'<div style="background:{BL_WHITE};border:1px solid {BL_LINE};border-left:3px solid {accent};'
        f'border-radius:8px;padding:12px 14px;height:100%;">'
        f'<div style="font:500 9.5px {CARD_MONO};letter-spacing:.14em;text-transform:uppercase;'
        f'color:{BL_MUTED};">{kicker}</div>'
        f'<div style="margin-top:6px;font-family:{CARD_SERIF};font-size:17px;font-weight:600;'
        f'line-height:1.25;color:{body};">{role}</div>'
        f'<div style="margin-top:3px;font:500 11.5px {CARD_MONO};color:{BL_MUTED};'
        f'font-variant-numeric:tabular-nums;">{pct}</div>'
        f'</div>'
    )


@st.cache_data
def load_browsable_players() -> pd.DataFrame:
    """All 273 drafted anchors with BOTH sides measured: his college recipe
    (`c_alpha_*`) and the rookie recipe he actually produced (`y_*`).

    No prediction column, deliberately. Act 2 shows the real college -> rookie
    transition; the model's projections belong to act 1 (the three players who
    have no rookie season yet) and to the validation act. Mixing a projection
    into a view of observed data invites the reader to read the model's output
    as if it were also measured.

    (The out-of-sample vectors generated by
    src/translator/phase5_step4_oof_recipes.py are left on disk and unread -
    a valid artifact, just not what this act is for.)"""
    a = load_anchors()
    keep = (["player_name", "draft_year", "overall", "college_team", "nba_player_id",
             "c_argmax", "c_alpha_max", "y_argmax", "y_max"]
            + [f"c_alpha_{j}" for j in range(K)] + [f"y_{j}" for j in range(K)])
    return a[keep].sort_values(["draft_year", "player_name"],
                               ascending=[False, True]).reset_index(drop=True)


def recipe_radar_compare(y_pred: np.ndarray, y_true: np.ndarray, nba_labels: dict,
                         r_max: float, height: int = 380) -> go.Figure:
    """The same 8-archetype radar act 1 uses, with TWO traces: what we
    projected and what he actually became.

    Bold labels mark the ACTUAL top-3, not the projected one - the truth is
    what the projection is being judged against, so emphasising the prediction
    would flatter it."""
    top3 = set(int(j) for j in np.argsort(-y_true)[:3])
    labels = []
    for j in range(K):
        short = player_report.abbreviate_archetype(nba_labels.get(j, f"archetype {j}"))
        labels.append(f'<b><span style="font-size:12.5px">{short}</span></b>' if j in top3
                      else f'<span style="font-size:10.5px">{short}</span>')

    fig = go.Figure()
    for name, vec, colour, fill in (
        ("We projected", y_pred, BL_GREEN, "rgba(0,75,43,0.20)"),
        ("He became", y_true, BL_CORAL, "rgba(238,115,95,0.20)"),
    ):
        vals = [float(vec[j]) * 100 for j in range(K)]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=labels + [labels[0]], name=name,
            mode="lines+markers", line=dict(color=colour, width=2),
            fill="toself", fillcolor=fill, marker=dict(size=5, color=colour),
            hovertemplate="%{theta}: %{r:.0f}%<extra>" + name + "</extra>",
        ))
    fig.update_layout(
        height=height, margin=dict(l=70, r=70, t=46, b=34),
        paper_bgcolor=BL_PAPER, font_color=BL_INK,
        legend=dict(orientation="h", y=1.13, font=dict(size=12)),
        polar=dict(
            bgcolor=BL_PAPER,
            radialaxis=dict(range=[0, r_max], color=BL_MUTED, gridcolor=BL_LINE,
                            tickfont=dict(size=8), ticksuffix="%", dtick=20,
                            angle=90, tickangle=90),
            angularaxis=dict(color=BL_INK, gridcolor=BL_LINE,
                             categoryorder="array", categoryarray=labels),
        ),
    )
    return fig


def render_player_transition(college_labels: dict, nba_labels: dict) -> None:
    """Act 2: pick any drafted player and see the transition that actually
    happened — the role he played in college, and the role he produced as an
    NBA rookie. Both sides are measured; nothing here is a projection."""
    df = load_browsable_players()
    labels = [f"{r.player_name} ({int(r.draft_year)})" for r in df.itertuples()]
    picked = st.selectbox(f"Player ({len(df)} drafted players, 2017–2025)", labels,
                          index=0, key="s2_pick")
    r = df.loc[labels.index(picked)]

    c_alpha = np.array([float(r[f"c_alpha_{j}"]) for j in range(K)])
    y_true = np.array([float(r[f"y_{j}"]) for j in range(K)])

    # Fixed 60% floor, widened only when this player needs it - a handful of
    # real rookie recipes reach 100%, and letting one of those set the scale
    # for everyone shrinks the large majority into an unreadable blob.
    r_max = max(60.0, float(np.ceil(y_true.max() * 100 / 20.0) * 20))

    pid = int(r["nba_player_id"]) if pd.notna(r["nba_player_id"]) else None
    photo = hull_callout_chart.get_headshot_data_uri(pid, r["player_name"])

    left, right = st.columns([0.9, 1.1], gap="medium", vertical_alignment="center")
    with left:
        st.markdown(
            f'<div style="background:{BL_WHITE};border:1px solid {BL_LINE};border-radius:12px;'
            f'padding:22px 20px;text-align:center;">'
            f'<img src="{photo}" style="width:76px;height:76px;border-radius:50%;object-fit:cover;'
            f'object-position:top center;border:1px solid {BL_LINE};padding:3px;display:block;'
            f'margin:0 auto;">'
            f'<div style="margin-top:12px;font-family:{CARD_SERIF};font-size:20px;font-weight:600;'
            f'color:{BL_INK};line-height:1.2;">{r["player_name"]}</div>'
            f'<div style="margin-top:5px;font:400 10.5px {CARD_MONO};color:{BL_MUTED};">'
            f'{r["college_team"] if pd.notna(r["college_team"]) else "—"} · '
            f'{int(r["draft_year"])} pick #{int(r["overall"])}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
        st.write("")
        st.markdown(_role_panel_html(
            "In college", clean_label(college_labels.get(int(r["c_argmax"]), "—")),
            f"{c_alpha.max()*100:.0f}% of his college recipe", BL_MUTED, dim=True),
            unsafe_allow_html=True)
        st.write("")
        st.markdown(_role_panel_html(
            "As an NBA rookie", nba_labels.get(int(r["y_argmax"]), "—"),
            f"{y_true.max()*100:.0f}% of his rookie recipe", BL_GREEN),
            unsafe_allow_html=True)
    with right:
        st.plotly_chart(recipe_radar_chart(y_true, nba_labels, r_max, height=380),
                        width="stretch", key=f"s2_radar_{r['player_name']}")
        st.caption("His measured rookie-season recipe across the 8 NBA archetypes.")


def render_mapping_table(college_labels: dict, nba_labels: dict) -> None:
    """The same measured transition for everyone at once. This is the only view
    where the divergence is visible at a glance — one college archetype landing
    on several different NBA roles, which is the fact the whole translator
    exists to model."""
    df = load_browsable_players()
    df["_college"] = [clean_label(college_labels.get(int(j), "—")) if pd.notna(j) else "—"
                      for j in df["c_argmax"]]
    df["_rookie"] = [nba_labels.get(int(j), f"archetype {j}") for j in df["y_argmax"]]

    columns = [("Player", "name"), ("Yr", "yr"), ("School", "school"),
               ("In college", "college"), ("As an NBA rookie", "rookie")]
    rows = []
    for _, r in df.iterrows():
        rows.append([
            (f'<b>{r["player_name"]}</b>', str(r["player_name"]).lower()),
            (str(int(r["draft_year"])), int(r["draft_year"])),
            (str(r["college_team"]) if pd.notna(r["college_team"]) else "—",
             str(r["college_team"]).lower() if pd.notna(r["college_team"]) else ""),
            (f'<span style="color:{BL_MUTED};">{r["_college"]}</span>', r["_college"].lower()),
            (f'<span style="font-weight:600;color:{BL_INK};">{r["_rookie"]}</span>',
             r["_rookie"].lower()),
        ])
    table_html, height = _build_sortable_table_html(
        "mappingTable", columns, rows, row_height=32, font_size_px=13)
    st.iframe(table_html, height=height)


def render_accuracy_diagnosis(prof: dict, metrics_df: pd.DataFrame, nba_labels: dict) -> None:
    """Act 3: the numbers, then the short honest read on them.

    Deliberately terse. An earlier version of this said the same things in
    four long paragraphs and the owner's verdict was that acts 2 and 3 were
    "太多了 ... 复杂冗长" - so each finding is now one line with its figure,
    and the supporting detail is gone rather than merely collapsed."""
    m = metrics_df.iloc[0]
    lo, hi = m["top1_hit_boot_lo"] * 100, m["top1_hit_boot_hi"] * 100
    gap = (prof["true_peak"] - prof["pred_peak"]) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("Top archetype exactly right", f"{m['top1_hit_mean']*100:.1f}%",
              help=f"95% bootstrap CI {lo:.0f}–{hi:.0f}% (n={int(m['n'])})")
    c2.metric("Right archetype in the top two", f"{m['top1_within_top2_mean']*100:.1f}%")
    c3.metric("Recipe distance (JSD)", f"{m['jsd_mean']:.3f}",
              help="Scores the whole 8-number recipe, not just which archetype came first. "
                   "0 = exact match.")

    worst = sorted(prof["by_true"].items(), key=lambda kv: (kv[1][1], -kv[1][0]))
    worst_j, (worst_n, worst_rate) = next((j, v) for j, v in worst if v[0] >= 3)
    conf = prof["confusions"][0] if prof["confusions"] else None

    st.markdown("**Why it is not higher**")
    st.markdown(
        f"- **n = {prof['n']}.** The 95% interval is {lo:.0f}–{hi:.0f}% — wide enough to cover the "
        f"41.7% baseline.\n"
        f"- **{nba_labels.get(worst_j, worst_j)}** is the weak class ({worst_rate*100:.0f}%), "
        f"usually lost to *{nba_labels.get(conf[0][1], conf[0][1]) if conf else '—'}* — two "
        f"perimeter roles split by shot location and play type, which college data lacks.\n"
        f"- Projections run **{gap:.0f}pp flatter** than reality; the model hedges."
    )
    st.markdown("**What would move it**")
    st.markdown(
        "- More anchors — 237 is the ceiling.\n"
        "- College shot-location / play-type data.\n"
        "- Split-half the rookie recipe first, to see how much of the gap is the target moving."
    )


# --- Section 5: what this model cannot see -----------------------------------

SANCTIONED_CONFIDENCE_SENTENCE = (
    "On the 36 rookies of the 2025 draft class, the top predicted archetype was correct 52.8% of "
    "the time, and the actual archetype fell in the top two 69.4% of the time."
)


def render_section5_limits() -> None:
    """ACT 3, part 1. Kept fully visible rather than collapsed: a reader who
    stops before the limits has been misled, so these do not go behind a click."""
    st.markdown("**What this model cannot see**")
    st.markdown(
        "- **No shot location or play type** — 16 of the NBA basis's 29 dimensions have no college "
        "equivalent.\n"
        "- **Trained on rookies who played ≥300 minutes** — it answers *what role if he plays*, not "
        "*whether he plays*.\n"
        "- **Injury-truncated prospects are absent** from training, so those profiles are "
        "extrapolation.\n"
        "- **Coaching and usage are unobservable** to any statistical model.\n"
        "- **Intervals are not calibrated**, which is why none are shown.\n"
        "- **237 training anchors, 36 held out.**"
    )
    st.caption(f"The one sanctioned confidence statement: \"{SANCTIONED_CONFIDENCE_SENTENCE}\"")


# --- Section 6: frozen record -------------------------------------------------

def render_section6_frozen_record(frozen: dict, manifest: dict) -> None:
    """ACT 3, part 2 - the audit trail."""
    full_hash = frozen["nets_rookies_2026_csv_sha256"]
    st.markdown(
        f"Frozen **{frozen['frozen_at_utc']}** · commit `{frozen['git_hash'][:10]}` · "
        f"model {manifest.get('model', 'T1_b')} (n={manifest.get('n_train', '—')})"
    )
    # Shown inline rather than in its own expander: this whole function now
    # renders inside a "Frozen record" expander, and Streamlit raises on a
    # nested one. The hash is the audit artifact - it belongs visible here.
    st.code(full_hash, language=None)

    st.markdown("**Predicted top-3, plain text:**")
    for r in frozen["rookies"]:
        # One decimal, deliberately: this reads the FROZEN artifact (which stores
        # weights rounded to 3dp), while the cards in Act 1 read full precision.
        # At .0f a stored 0.345 prints "34%" beside a card printing "35%" - the
        # same number, contradicting itself across the page. One decimal prints
        # "34.5%", which agrees with the card instead of contradicting it, and
        # does so without altering a single stored value.
        top3_str = ", ".join(f"archetype {j} ({w*100:.1f}%)" for j, w in r["predicted_top3"])
        st.markdown(f"- **{r['display_name']}**: {top3_str}")

    st.caption("Frozen before the 2026-27 season; graded at midseason via "
               "`src/eval/review_2026_predictions.py`.")

    refreeze_log = frozen.get("refreeze_log", [])
    if refreeze_log:
        st.warning(f"⚠️ Refrozen {len(refreeze_log)} time(s) since the original freeze:")
        for entry in refreeze_log:
            st.markdown(f"- **{entry['refrozen_at_utc']}**: {entry['reason']}")


# --- main entry point ---------------------------------------------------------

def render_rookie_projections_page() -> None:
    st.title("NCAA Bridge — Rookie Archetype Projections")

    missing = check_required_files()
    if missing:
        st.error(
            f"This page can't render: {len(missing)} required data file(s) are missing "
            f"({', '.join(missing)}). This is expected if the Phase 1-6 translator pipeline hasn't "
            f"been run yet — see reports/project_summary.md for how to build it. The rest of the "
            f"portal is unaffected."
        )
        return

    try:
        projections_df = load_rookie_projections()
        projections_full = load_rookie_projections_full()
        frozen = load_frozen_record()
        holdout_df = load_holdout_predictions()
        baseline_df = load_baseline_comparison()
        manifest = load_deployment_manifest()
        B_samples, phi_samples = load_deployment_posterior()
        preprocessing = load_deployment_preprocessing()
        anchors_df = load_anchors()
        college_labels, nba_labels = load_labels()
        archetypoids = load_archetypoids()
        pick_numbers = load_rookie_pick_numbers()
    except Exception as e:
        st.error(f"This page can't render: failed to load a required data file ({e}). "
                 f"The rest of the portal is unaffected.")
        return

    if not college_labels or not nba_labels or len(college_labels) != K or len(nba_labels) != K:
        st.error("This page can't render: archetype labels are not available for both sides "
                 "(reports/college_archetypes.md and data/basis_2025_26/archetype_labels.csv). "
                 "This page never displays a bare archetype index.")
        return

    guard_result = {"passed": False, "max_diff": None, "error": "section hidden"}
    if SHOW_PICK_SLOT_COUNTERFACTUAL:
        import yaml
        config = yaml.safe_load((REPO_ROOT / "config.yaml").read_text())
        try:
            guard_result = run_consistency_guard(projections_full, preprocessing, B_samples, config)
        except Exception as e:
            guard_result = {"passed": False, "max_diff": None, "error": str(e)}
        if not guard_result["passed"]:
            print(f"[step5_rookie_projections] CONSISTENCY GUARD FAILED: {guard_result}")  # logged, per spec

    # AI-ASSISTED (Claude Code, chat) - Prompt: "我觉得没必要写这么复杂 整个页面
    # 关于NCAA Bridge的flow很乱 我想把这部分重写改一下" (this is over-complicated,
    # the whole NCAA Bridge flow is a mess, I want to rewrite it). Scope confirmed
    # by the owner: REORGANISE, keep every piece of content.
    # The old layout was six st.header() sections of equal visual weight, in build
    # order, with no narrative spine - so a reader could not tell which parts were
    # the answer, which were evidence, and which were appendix. Worse, two of them
    # (naive-relabeling and the holdout browser) answer the SAME question and were
    # separated by nothing, while the frozen record - an audit artifact - got the
    # same prominence as the projections themselves.
    # Restructured into three acts that follow how the page is actually read:
    #   1. what the model says   (the answer)
    #   2. why you should believe it   (both evidence sections, now together)
    #   3. where it breaks / the audit trail   (appendix)
    # Nothing was deleted: the comps and the 36-row holdout table moved into
    # expanders under the act they belong to, so the page opens short and every
    # artifact is one click away. (The pick-slot counterfactual was moved the
    # same way and later hidden outright at the owner's request - see
    # SHOW_PICK_SLOT_COUNTERFACTUAL; its code is still here, unrun.)
    # Not AI: the judgment that the page was too complex and the flow was a mess,
    # and the decision to reorganise rather than cut - the owner's own calls.

    # --- ACT 1: what the model says -----------------------------------------
    render_act_header("1", "What the model says")
    st.caption("Projections from college data — not observed NBA seasons.")
    render_section1_projections(projections_full, college_labels, nba_labels, archetypoids, pick_numbers)

    if SHOW_PICK_SLOT_COUNTERFACTUAL:
        with st.expander("Try a different draft slot — same player, different opportunity"):
            render_section4_counterfactual(anchors_df, projections_full, preprocessing, B_samples,
                                             college_labels, nba_labels, pick_numbers, guard_result)

    st.divider()

    # --- ACT 2: the mapping, as one table -----------------------------------
    # Owner: "就是展示一个table 我们所有的player如何从college => rookie season,
    # and our project for each one of them". One table, all 36, whole chain.
    render_act_header("2", "College to rookie season, as it actually happened")
    st.caption("Pick any drafted player: the role he played in college, and the role he actually "
               "produced as an NBA rookie. Both measured.")
    render_player_transition(college_labels, nba_labels)
    with st.expander("All 273 drafted players, sortable"):
        render_mapping_table(college_labels, nba_labels)

    # --- ACT 3: benchmarks and the honest read (hidden) ----------------------
    if SHOW_BENCHMARK_ACT:
        st.divider()
        render_act_header("3", "Versus the benchmarks, and where it can improve")
        metrics_df = load_holdout_metrics()
        prof = holdout_error_profile()
        render_section2_headline(baseline_df)
        st.write("")
        render_accuracy_diagnosis(prof, metrics_df, nba_labels)
        st.write("")
        with st.expander("What this model cannot see"):
            render_section5_limits()
        with st.expander("Frozen record"):
            render_section6_frozen_record(frozen, manifest)
