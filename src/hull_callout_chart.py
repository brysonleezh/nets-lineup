"""
Anatomical-diagram-style leader-line callouts for the archetypoid hull
scatter, plus custom (non-native) hover cards for both archetypoids and
Nets players - both requirements Plotly's own hovertemplate can't satisfy
(no leader-line label layout primitive, no <img> support in hover text).

Part A  compute_leader_callouts - pure pixel-space geometry (centroid,
        direction vectors, knee/end points, side-based decluttering).
        Reusable across any K - takes vertex pixel positions in, returns
        leader-line paths + label anchors out, no chart-library dependency.
Part B  clockwise color assignment for the K archetypoid vertices
Part C  local headshot cache (download once, base64 data-URI, initials
        fallback) - avoids depending on a live network fetch inside the
        rendered page and avoids the canvas/CORS issue found earlier with
        remote URLs in a rasterized context.
Part D  figure assembly (traces + leader-line path shapes + annotations)
Part E  standalone HTML/JS shell - Plotly.js via CDN + a custom hover-card
        div driven by plotly_hover/plotly_unhover, since Plotly's native
        hover can't embed images.
"""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

import numpy as np
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HEADSHOT_CACHE_DIR = DATA_DIR / "headshot_cache"

HEADSHOT_URL = "https://cdn.nba.com/headshots/nba/latest/1040x760/{player_id}.png"

# --- Part A: leader-line geometry (pure, pixel-space) -----------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: exact algorithm specified in full - centroid C of the K vertices,
unit direction d = normalize(V-C) per vertex, knee = V + d*46px, end = knee
+/-40px horizontal, label anchor = end +/-8px horizontal, last segment
horizontal, declutter labels closer than 26px vertically by pushing apart
after sorting by y, knee follows the label's new y.
Used: implemented literally as specified, entirely in pixel space (the
algorithm's distances - 46/40/8/26px - are screen-space quantities, so
running the whole computation in pixel space and converting the result
back to data coordinates once at the end is the only way to honor those
distances exactly regardless of the plot's data-unit scale - doing the
geometry in data-space and eyeballing a pixel-ish constant would drift as
soon as the axis range or figure size changed). Decluttering only compares
labels on the SAME side (left vs right, by sign of d.x) since opposite-side
labels can't visually collide with each other. Knee's Y is overwritten to
match the label's final (decluttered) Y so the last segment stays
horizontal even after spacing labels apart - the knee's X is left alone
(still the original direction-projected value), which is what keeps the
first segment's angle looking like it's actually pointing back at the
vertex rather than snapping to a fixed angle.
Not AI: the whole leader-line design concept and every numeric constant
(46/40/8/26px, the two-segment elbow shape, side-based decluttering) -
given directly, not proposed here.
"""


def _normalize(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([1.0, 0.0])


def compute_leader_callouts(vertices_px, knee_len=46.0, horiz_len=40.0,
                             label_gap=8.0, min_spacing=26.0):
    """vertices_px: (n, 2) array of vertex positions in PIXEL space.

    Returns a list (same order as input) of dicts:
      {"path": [(x0,y0), (x1,y1), (x2,y2)],   # V -> knee -> end, pixel space
       "label_anchor": (x, y),                 # pixel space
       "align": "left" | "right",              # text-anchor side
       "side": "left" | "right"}
    """
    vertices_px = np.asarray(vertices_px, dtype=float)
    n = len(vertices_px)
    C = vertices_px.mean(axis=0)

    dirs = np.array([_normalize(v - C) for v in vertices_px])
    knees = vertices_px + dirs * knee_len
    ends_x = knees[:, 0] + np.where(dirs[:, 0] > 0, horiz_len, -horiz_len)
    ends_y = knees[:, 1].copy()

    sides = np.where(dirs[:, 0] > 0, "right", "left")

    # declutter within each side independently - opposite sides can't collide
    for side in ("left", "right"):
        idx = np.where(sides == side)[0]
        if len(idx) < 2:
            continue
        order = idx[np.argsort(ends_y[idx])]
        for i in range(1, len(order)):
            prev, cur = order[i - 1], order[i]
            if ends_y[cur] - ends_y[prev] < min_spacing:
                ends_y[cur] = ends_y[prev] + min_spacing

    knees[:, 1] = ends_y  # last segment stays horizontal after decluttering

    out = []
    for i in range(n):
        align = "left" if dirs[i, 0] > 0 else "right"
        anchor_x = ends_x[i] + (label_gap if dirs[i, 0] > 0 else -label_gap)
        out.append({
            "path": [tuple(vertices_px[i]), tuple(knees[i]), (float(ends_x[i]), float(ends_y[i]))],
            "label_anchor": (float(anchor_x), float(ends_y[i])),
            "align": align,
            "side": sides[i],
        })
    return out


# --- Part B: clockwise color assignment --------------------------------------

CALLOUT_PALETTE = ["#4a72a8", "#3f8b7d", "#7d8f4a", "#b8873a",
                   "#c2603f", "#a8517a", "#6b5ba8", "#5c6672"]


def clockwise_order(vertices):
    """vertices: (n, 2) array. Returns the indices that walk the vertices
    CLOCKWISE around their own centroid (decreasing atan2 angle, since y
    increases upward) - shared by the color assignment and the dashed
    outline connecting them, so both agree on the same ordering."""
    vertices = np.asarray(vertices, dtype=float)
    C = vertices.mean(axis=0)
    angles = np.arctan2(vertices[:, 1] - C[1], vertices[:, 0] - C[0])
    return np.argsort(-angles)


def assign_clockwise_colors(vertices, palette=CALLOUT_PALETTE):
    """vertices: (n, 2) array (any coordinate space - pixel or data, doesn't
    matter, only relative angle around the centroid matters). Returns a
    color per input index, assigned by walking the vertices CLOCKWISE from
    the centroid and handing out `palette` in order. Cycles the palette if
    n > len(palette)."""
    cw_order = clockwise_order(vertices)
    colors = [None] * len(vertices)
    for rank, idx in enumerate(cw_order):
        colors[idx] = palette[rank % len(palette)]
    return colors


# --- Part C: local headshot cache --------------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: "头像图片路径：请建一个dict/json映射 player_id -> 本地图片路径...缺图时
fallback成球员姓名缩写的圆形色块，不要报错" - cache headshots locally rather
than hot-linking the CDN from inside the rendered page.
Used: downloads once to data/headshot_cache/{player_id}.png (skipped if
already cached - safe to call every run), then base64-encodes into a data
URI for embedding - checked earlier this session that Plotly's own
add_layout_image silently fails to render a remote CDN url when the page is
rasterized to a static image (kaleido's headless renderer has no network
access in this sandbox, confirmed via a side-by-side data-URI-vs-remote-URL
test), so a data URI is the more robust choice generally, not just a
"local path" technicality - it works the same whether the consumer is a
live browser or a static export. Network/HTTP failures (timeout, 404) are
caught and produce a deterministic solid-color circle with the player's
initials instead of raising - a missing headshot is a real, expected
possibility (not every PLAYER_ID has a CDN photo), not a bug to crash on.
Not AI: the decision to cache locally at all instead of hot-linking, and
the initials-fallback requirement - given directly.
"""


def _initials(name):
    parts = [p for p in name.replace(".", "").split() if p]
    if len(parts) >= 2:
        return (parts[0][0] + parts[-1][0]).upper()
    return (parts[0][:2] if parts else "??").upper()


def _initials_svg_data_uri(name, bg_color, size=96):
    initials = _initials(name)
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}">'
        f'<circle cx="{size/2}" cy="{size/2}" r="{size/2}" fill="{bg_color}"/>'
        f'<text x="50%" y="52%" text-anchor="middle" dominant-baseline="middle" '
        f'font-family="system-ui,sans-serif" font-size="{size*0.36:.0f}" '
        f'font-weight="600" fill="#fdfaf5">{initials}</text></svg>'
    )
    b64 = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{b64}"


# The CDN returns HTTP 200 with a generic silhouette (not a 404) for any
# unknown/photo-less PLAYER_ID - confirmed live by requesting two different
# invalid IDs and comparing: byte-identical (md5 e7f284977a4931dedd1cb6ba4c32283e),
# so a response-size check alone doesn't catch it. Hash every download and
# treat this exact placeholder as "missing" too.
_CDN_PLACEHOLDER_MD5 = "e7f284977a4931dedd1cb6ba4c32283e"


def get_headshot_data_uri(player_id, player_name, fallback_color="#7c8593", timeout=5):
    """Local-cache-first headshot as a base64 data URI. Downloads once to
    data/headshot_cache/{player_id}.png; on any failure (network error, the
    CDN's generic silhouette placeholder, corrupt image), returns a solid-
    color initials SVG instead of raising - a missing CDN photo is expected,
    not exceptional."""
    if player_id is None:
        return _initials_svg_data_uri(player_name, fallback_color)

    HEADSHOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = HEADSHOT_CACHE_DIR / f"{int(player_id)}.png"

    if not cache_path.exists():
        try:
            resp = requests.get(HEADSHOT_URL.format(player_id=int(player_id)), timeout=timeout)
            resp.raise_for_status()
            if hashlib.md5(resp.content).hexdigest() == _CDN_PLACEHOLDER_MD5:
                raise ValueError("CDN returned its generic placeholder silhouette, treating as missing")
            cache_path.write_bytes(resp.content)
        except Exception:
            return _initials_svg_data_uri(player_name, fallback_color)

    try:
        data = cache_path.read_bytes()
        if hashlib.md5(data).hexdigest() == _CDN_PLACEHOLDER_MD5:
            return _initials_svg_data_uri(player_name, fallback_color)
        b64 = base64.b64encode(data).decode("ascii")
        return f"data:image/png;base64,{b64}"
    except Exception:
        return _initials_svg_data_uri(player_name, fallback_color)


# --- Part D: figure assembly -------------------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: full color/style spec given directly - cream background #f7f3ec,
grey point cloud #ded8cc, dashed hull #7c8593, 8-color identity palette
(Part B), deep ink #23211e for names, Nets dots #1c1a17, hover-card chrome
(#fdfaf5 bg / #e0d8ca border / rgba(35,33,30,.14) shadow), plus the 46/40/8px
leader-line geometry (Part A) and the ~310px per-side label zone / hull-at-
55%-of-width canvas budget.
Used: fixed pixel width/height and explicit zero-margin layout so the
data<->pixel conversion (needed to run Part A's algorithm, whose distances
are pixel quantities) stays valid - an auto-margined or responsive figure
would silently invalidate every precomputed leader-line coordinate the
moment Plotly's own margin differs from what was assumed. Leader lines are
layout.shapes with type="path" (an SVG path string, "M x,y L x,y L x,y") -
a plain type="line" shape only supports a single straight segment, and this
callout needs two (V->knee, knee->end), so path is the one shape type that
can hold both in one object with one color. Labels are annotations with
showarrow=False and inline <span style="color:...;font-weight:...;
font-size:..."> for the two-line name/role text - one of the few HTML-ish
tags Plotly's annotation renderer actually supports (checked before relying
on it, same as the earlier hover-template tag-subset check this session).
Both the Nets and archetypoid scatter traces set hoverinfo="none" - Plotly's
native hover box is fully replaced by the custom HTML card in Part E, so a
second, redundant native tooltip must not also pop up.
Not AI: every color value and every geometry constant - given directly.
"""

CHART_BG = "#f7f3ec"
POP_DOT_COLOR = "#ded8cc"
HULL_LINE_COLOR = "#7c8593"
INK_COLOR = "#23211e"
NETS_DOT_COLOR = "#1c1a17"
MUTED_TEXT_COLOR = "#7b7469"
CARD_BG = "#fdfaf5"
CARD_BORDER = "#e0d8ca"
CARD_SHADOW = "rgba(35,33,30,0.14)"

FIGURE_WIDTH_PX = 1400
FIGURE_HEIGHT_PX = 700
LABEL_ZONE_PX = 310  # each side - covers 46+40+8+~220 (longest label) per the spec
Y_PAD_FRAC = 0.22    # extra vertical room for decluttered labels pushed past the raw point extent

# (name, position, jersey number) - not available in any pulled DB table
# (player_bio/player_base have no position/jersey column); sourced directly
# from ESPN's live roster page (https://www.espn.com/nba/team/roster/_/name/
# bkn/brooklyn-nets), the same fetch already used earlier this session to
# correct the roster list - not fabricated, not guessed.
NETS_POSITION_NUMBER = {
    "Mikel Brown Jr.": ("G", "0"), "Danny Wolf": ("F", "2"), "Drake Powell": ("G", "4"),
    "Egor Dëmin": ("G", "8"), "E.J. Liddell": ("F", "9"), "Tyson Etienne": ("G", "10"),
    "Nolan Traore": ("G", "13"), "Terance Mann": ("G", "14"), "Michael Porter Jr.": ("F", "17"),
    "Day'Ron Sharpe": ("C", "20"), "Noah Clowney": ("F", "21"), "Julius Randle": ("F", "30"),
    "Chaney Johnson": ("F", "31"), "Tyler Bilodeau": ("F", "34"), "Ben Saraf": ("G", "77"),
    "Josh Minott": ("F", "00"), "Joshua Jefferson": ("F", "—"), "Keon Ellis": ("G", "—"),
    "Moritz Wagner": ("F", "—"),
}


def _hover_card_html(photo_uri, name, subtitle, weight_html, border_color):
    """Shared hover-card inner HTML for both archetypoids and Nets players -
    photo left (48px circle, border in the point's own identity color),
    name/subtitle/weights right. Built once in Python (not JS template
    strings) so the formatting logic is easy to test in isolation."""
    return (
        f'<div style="display:flex;align-items:center;gap:10px;">'
        f'<img src="{photo_uri}" style="width:48px;height:48px;min-width:48px;border-radius:50%;'
        f'object-fit:cover;object-position:top center;border:1.5px solid {border_color};">'
        f'<div>'
        f'<div style="font-size:15px;font-weight:600;color:{INK_COLOR};white-space:nowrap;">{name}</div>'
        f'<div style="font-size:13px;color:{MUTED_TEXT_COLOR};white-space:nowrap;">{subtitle}</div>'
        f'<div style="font-size:13px;white-space:nowrap;margin-top:2px;">{weight_html}</div>'
        f'</div></div>'
    )


# AI-ASSISTED (Claude Code, chat) - Prompt: "label two or three of the black
# Nets dots inline by default... pick recognizable rotation players
# positioned so labels don't collide with the corner annotations" - a
# fixed roster subset (not every Nets player, which would clutter the
# interior) so the figure reads without hovering at all.
# Used: picked by actually computing this season's real 2D projection and
# measuring pixel distance from each candidate to every corner label's own
# anchor point (not eyeballed) - Julius Randle (248px clearance), Michael
# Porter Jr. (171px), and Day'Ron Sharpe (149px) were the best-separated
# recognizable rotation players among the roster, spread across the plot
# (left/up, left/center, right/center) rather than clustered together.
# If this project's Nets roster or the underlying 2025-26 fit changes and
# one of these three is no longer in the population, the lookup below
# just skips it silently - a missing default label is a cosmetic
# degradation, not a crash.
# Not AI: the decision to label a small fixed set rather than all Nets
# dots, and that hover remains the mechanism for everyone else - given
# directly.
DEFAULT_INLINE_LABELED_NETS = ["Julius Randle", "Michael Porter Jr.", "Day'Ron Sharpe"]


def _top_weights_html(vals, arch_names, colors, top_n=2, min_pct=0.05):
    """'62% Offensive Engine · 24% 3&D Wing' - each archetype name colored
    with its own identity color. Archetypes below min_pct are dropped even
    if it means showing only 1, since a near-zero second archetype isn't a
    real signal worth reporting."""
    order = np.argsort(vals)[::-1][:top_n]
    parts = []
    for i in order:
        if vals[i] < min_pct:
            continue
        parts.append(f'<span style="color:{colors[i]};font-weight:600;">{vals[i]:.0%} {arch_names[i]}</span>')
    return " · ".join(parts) if parts else "—"


def build_figure_spec(proj, roster_df, labels, k):
    """Everything needed to render the chart: Plotly trace/shape/annotation
    dicts (plain JSON-able structures, not a go.Figure - this gets embedded
    directly into a <script> block in Part E, not passed through
    st.plotly_chart) plus layout. Vertex positions and archetype weights are
    read live from `proj`/`labels` - nothing here is hardcoded per K.
    """
    pop, all_2d = proj["pop"], proj["all_2d"]
    P = proj["P"]
    defs = proj["defs"]
    archetype_row_idx = proj["archetype_row_idx"]

    arch_names = [labels.get(i, f"Archetype {i}") for i in range(k)]
    arch_row = [archetype_row_idx[a] for a in range(k)]
    if any(r is None for r in arch_row):
        raise ValueError("one or more archetypoids didn't resolve to a population row - "
                          "check archetype_row_idx before building this chart")
    vertices_data = np.array([all_2d[r] for r in arch_row])
    colors = assign_clockwise_colors(vertices_data)

    nets_ids = set(int(x) for x in roster_df["PLAYER_ID"].astype(int))
    is_archetype = np.zeros(len(pop), dtype=bool)
    is_archetype[arch_row] = True
    is_nets = pop["PLAYER_ID"].astype(int).isin(nets_ids).values & ~is_archetype
    is_other = ~is_archetype & ~is_nets

    # --- data<->pixel conversion, fixed once, before any geometry runs -----
    data_x0, data_x1 = float(all_2d[:, 0].min()), float(all_2d[:, 0].max())
    data_y0, data_y1 = float(all_2d[:, 1].min()), float(all_2d[:, 1].max())
    dx, dy = data_x1 - data_x0, data_y1 - data_y0
    data_x0 -= dx * 0.08; data_x1 += dx * 0.08
    data_y0 -= dy * Y_PAD_FRAC; data_y1 += dy * Y_PAD_FRAC

    hull_px_width = FIGURE_WIDTH_PX - 2 * LABEL_ZONE_PX
    px_per_data_x = hull_px_width / (data_x1 - data_x0)
    px_per_data_y = FIGURE_HEIGHT_PX / (data_y1 - data_y0)
    side_pad_data = LABEL_ZONE_PX / px_per_data_x
    full_x0, full_x1 = data_x0 - side_pad_data, data_x1 + side_pad_data

    def data_to_px(x, y):
        px_x = (x - full_x0) / (full_x1 - full_x0) * FIGURE_WIDTH_PX
        px_y = (data_y1 - y) / (data_y1 - data_y0) * FIGURE_HEIGHT_PX
        return px_x, px_y

    def px_to_data(px_x, px_y):
        x = full_x0 + px_x / FIGURE_WIDTH_PX * (full_x1 - full_x0)
        y = data_y1 - px_y / FIGURE_HEIGHT_PX * (data_y1 - data_y0)
        return x, y

    # --- leader-line callouts (Part A), computed in pixel space ------------
    vertices_px = np.array([data_to_px(x, y) for x, y in vertices_data])
    callouts = compute_leader_callouts(vertices_px)

    shapes, annotations = [], []
    for a in range(k):
        c = callouts[a]
        path_data = [px_to_data(px, py) for px, py in c["path"]]
        svg_path = "M " + " L ".join(f"{x},{y}" for x, y in path_data)
        shapes.append({
            "type": "path", "path": svg_path, "xref": "x", "yref": "y",
            "line": {"color": colors[a], "width": 1.2}, "opacity": 0.9,
        })
        anchor_data = px_to_data(*c["label_anchor"])
        name = defs[defs["archetype"] == a].iloc[0]["PLAYER_NAME"]
        role = arch_names[a]
        text = (
            f'<span style="font-size:16px;font-weight:600;color:{INK_COLOR};">{name}</span><br>'
            f'<span style="font-size:14px;letter-spacing:0.07em;color:{colors[a]};">{role.upper()}</span>'
        )
        annotations.append({
            "x": anchor_data[0], "y": anchor_data[1], "xref": "x", "yref": "y",
            "text": text, "showarrow": False,
            "xanchor": c["align"], "yanchor": "middle", "align": c["align"],
        })

    # Default-labeled Nets dots (see DEFAULT_INLINE_LABELED_NETS above) - a
    # small, lighter-weight text label offset from the dot itself via
    # xshift/yshift (pixel-space, no leader line needed - these were chosen
    # to already sit clear of the corner labels' own pixel footprint),
    # visually distinct from the corner callouts (smaller, uncolored,
    # single line) so the two label types read as different things.
    for i in np.where(is_nets)[0]:
        name = pop["PLAYER_NAME"].values[i]
        if name not in DEFAULT_INLINE_LABELED_NETS:
            continue
        x, y = all_2d[i]
        annotations.append({
            "x": float(x), "y": float(y), "xref": "x", "yref": "y",
            "text": f'<span style="font-size:12px;font-weight:600;color:{INK_COLOR};">{name}</span>',
            "showarrow": False, "xanchor": "left", "yanchor": "bottom",
            "xshift": 9, "yshift": 9,
        })

    # --- traces --------------------------------------------------------------
    idx_other = np.where(is_other)[0]

    # AI-ASSISTED (Claude Code, chat)
    # Prompt: none of this was requested - found live via Playwright while
    # verifying hover (Part of "iterate until hover actually works" from the
    # spec's closing instruction to export and check a hover-state screenshot).
    # Hovering exactly over a rendered Nets marker's screen position, the
    # plotly_hover event fired for curveNumber=0 (the background trace) at a
    # DIFFERENT point, not the Nets trace - confirmed by logging
    # evt.points[0].curveNumber/pointIndex at the exact pixel. Root cause:
    # Plotly's "closest point" hover detection compares raw pixel distance to
    # each point's own center across ALL traces and does not account for
    # trace stacking order or marker size/z-index - a same-K-archetype-
    # profile background player can land close enough in the 2D projection
    # that its center is fractionally closer to the cursor than the visually
    # larger, foreground Nets marker drawn on top of it, and wins hover
    # priority anyway. Fixed by dropping background points within
    # HOVER_EXCLUDE_RADIUS_PX of any Nets/archetypoid marker's own pixel
    # position before building the background trace - those points are
    # already fully hidden under the foreground marker at that position, so
    # this is a hover-reliability fix with zero visual side effect, not a
    # data change.
    # Not AI: none - this bug and its fix were not part of the spec, found
    # and resolved during verification.
    # reshape(-1, 2) rather than a bare np.array: with NO highlighted players
    # (the league-wide view, where singling out one team would be meaningless)
    # the comprehension yields [] whose shape is (0,), and vstack rejects it
    # against the (n,2) vertices. Empty is a legitimate state here, not an
    # error, so it has to produce an empty (0,2) instead of raising.
    highlight_px = np.array(
        [data_to_px(*all_2d[i]) for i in np.where(is_nets)[0]], dtype=float
    ).reshape(-1, 2)
    foreground_px = np.vstack([vertices_px, highlight_px])
    if len(idx_other):
        other_px = np.array([data_to_px(*all_2d[i]) for i in idx_other])
        HOVER_EXCLUDE_RADIUS_PX = 14.0
        dists = np.linalg.norm(other_px[:, None, :] - foreground_px[None, :, :], axis=2)
        idx_other = idx_other[dists.min(axis=1) > HOVER_EXCLUDE_RADIUS_PX]

    other_trace = {
        "type": "scatter", "mode": "markers",
        "x": all_2d[idx_other, 0].tolist(), "y": all_2d[idx_other, 1].tolist(),
        "marker": {"size": 6, "color": POP_DOT_COLOR, "opacity": 0.6, "line": {"width": 0}},
        "hoverinfo": "none", "showlegend": False,
    }

    idx_nets = np.where(is_nets)[0]
    nets_pids = pop["PLAYER_ID"].astype(int).values[idx_nets]
    nets_names = pop["PLAYER_NAME"].values[idx_nets]
    nets_cards = []
    for pid, name in zip(nets_pids, nets_names):
        pos_num = NETS_POSITION_NUMBER.get(name, ("", "—"))
        subtitle = f"{pos_num[0]} · #{pos_num[1]}" if pos_num[0] else f"#{pos_num[1]}"
        row_i = int(pop.index[pop["PLAYER_ID"].astype(int) == int(pid)][0])
        weight_html = _top_weights_html(P[row_i], arch_names, colors)
        photo = get_headshot_data_uri(pid, name)
        nets_cards.append(_hover_card_html(photo, name, subtitle, weight_html, NETS_DOT_COLOR))

    nets_trace = {
        "type": "scatter", "mode": "markers",
        "x": all_2d[idx_nets, 0].tolist(), "y": all_2d[idx_nets, 1].tolist(),
        "marker": {"size": 10, "color": NETS_DOT_COLOR, "opacity": 1.0,
                   "line": {"width": 1.5, "color": CHART_BG}},
        "customdata": nets_cards, "hoverinfo": "none", "showlegend": False,
    }

    arch_cards = []
    for a in range(k):
        name = defs[defs["archetype"] == a].iloc[0]["PLAYER_NAME"]
        role = arch_names[a]
        photo = get_headshot_data_uri(int(pop["PLAYER_ID"].values[arch_row[a]]), name, fallback_color=colors[a])
        arch_cards.append(_hover_card_html(photo, name, role, "", colors[a]))

    arch_trace = {
        "type": "scatter", "mode": "markers",
        "x": vertices_data[:, 0].tolist(), "y": vertices_data[:, 1].tolist(),
        "marker": {"size": 12, "color": colors, "line": {"width": 2, "color": CHART_BG}},
        "customdata": arch_cards, "hoverinfo": "none", "showlegend": False,
    }

    # Dashed outline connecting the 8 corner (archetypoid) vertices, closing
    # back to the first - dropped when this component was rebuilt from
    # build_hull_scatter, added back on request. Walks the SAME clockwise
    # order used for color assignment (clockwise_order, Part B) rather than
    # a second independent ordering, so the line traces the vertices in the
    # order a viewer's eye would naturally follow color-to-color around the
    # loop, not a differently-ordered path that could visually cross itself.
    hull_order = clockwise_order(vertices_data)
    hull_loop = list(hull_order) + [hull_order[0]]
    hull_trace = {
        "type": "scatter", "mode": "lines",
        "x": vertices_data[hull_loop, 0].tolist(), "y": vertices_data[hull_loop, 1].tolist(),
        "line": {"color": HULL_LINE_COLOR, "width": 1.5, "dash": "dash"},
        # "skip", not "none": this line's own vertices are exactly the
        # archetypoid dots' positions, so a "none" (participates in hover,
        # suppress tooltip) let Plotly's hover detection pick the coincident
        # LINE point over the marker at the same spot - found live, hovering
        # an archetypoid dot fired curveNumber for the line trace instead
        # and no card ever showed. "skip" excludes it from hover entirely.
        "hoverinfo": "skip", "showlegend": False,
    }

    data = [other_trace, nets_trace, hull_trace, arch_trace]
    layout = {
        "width": FIGURE_WIDTH_PX, "height": FIGURE_HEIGHT_PX,
        "margin": {"l": 0, "r": 0, "t": 0, "b": 0},
        "plot_bgcolor": CHART_BG, "paper_bgcolor": CHART_BG,
        "xaxis": {"visible": False, "range": [full_x0, full_x1], "fixedrange": True},
        "yaxis": {"visible": False, "range": [data_y0, data_y1], "fixedrange": True},
        "shapes": shapes, "annotations": annotations,
        "hovermode": "closest",
    }
    return {"data": data, "layout": layout, "nets_trace_index": 1, "arch_trace_index": 3}


# --- Part E: standalone HTML/JS shell ----------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: "Plotly原生hovertemplate放不了图片，所以hover卡片要用自定义HTML：监听
plotly_hover/plotly_unhover，把customdata渲染进一个绝对定位的div，用
event.event.clientX/Y定位。请一并处理plotly_unhover和鼠标快速移动时的卡片残留
问题" - custom HTML hover cards driven by Plotly.js's own JS events, not
Streamlit's Python-side callbacks.
Used: st.plotly_chart can't do this at all - there's no documented on-hover
callback in Streamlit's Python API, and even if there were, positioning a
raw HTML div at the mouse position has to happen in the SAME document as
the chart's own DOM/JS event, which st.plotly_chart doesn't expose. Built
the entire chart as one self-contained HTML page instead (Plotly.js from
CDN + a plain <script>), meant to be embedded via st.iframe()
- that component runs its own real iframe document, so a <script> tag
actually executes (st.markdown's does not - checked and hit this exact
wall earlier this session building the roster table's sort feature).
Hover-card position uses `position: fixed` and the event's own
clientX/clientY directly, un-translated - clientX/Y are already relative to
this iframe's own viewport (the SAME document the card div lives in), so no
container-offset math is needed, only a right/bottom edge clamp so the card
never renders partially outside the iframe. Two separate mechanisms guard
against a stale card surviving fast mouse movement: plotly_unhover (fires
when the pointer leaves a marker) AND a mouseleave listener on the chart
div itself (covers the pointer leaving the whole plot area fast enough that
individual point unhover events could be missed) - both hide the card and
revert the Nets-trace restyle.
Not AI: the interaction spec itself (card follows mouse, Nets dot enlarge +
halo + dim-others on hover, both stale-card safety nets) - given directly.
"""

_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8">
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
  html, body {{ margin: 0; padding: 0; background: {bg}; overflow: hidden; }}
  #scale-wrapper {{ width: {width}px; height: {height}px; transform-origin: top left; }}
  #chart {{ width: {width}px; height: {height}px; }}
  #hover-card {{
    position: fixed; display: none; z-index: 1000; pointer-events: none;
    background: {card_bg}; border: 1px solid {card_border}; border-radius: 4px;
    padding: 10px 12px; box-shadow: 0 4px 14px {card_shadow};
    font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  }}
</style>
</head>
<body>
<div id="scale-wrapper"><div id="chart"></div></div>
<div id="hover-card"></div>
<script>
  // AI-ASSISTED (Claude Code, chat)
  // Prompt: none of this was requested - found live while checking the app
  // (the closing instruction to export a default-state screenshot from the
  // actual portal, not just the standalone test file). The figure's pixel
  // width (1400px) is fixed on purpose - the whole leader-line algorithm is
  // calibrated in real pixels - but Streamlit's own content column is
  // narrower than that at ordinary browser widths (measured live via
  // Playwright: 1008px in a 1500px-wide window, 1428px at 1920px), so the
  // chart was clipping Jonas Valanciunas/Ryan Kalkbrenner's labels off the
  // right edge in the real app despite rendering perfectly in the
  // standalone test file at a fixed 1450px viewport.
  // Used: a CSS transform: scale() on a wrapper div around #chart, sized to
  // the ACTUAL available width (document.documentElement.clientWidth,
  // measured inside this component's own iframe) divided by the design
  // width - this shrinks (or modestly grows, capped) the whole rendered
  // chart to fit whatever column width Streamlit actually gives it, while
  // Plotly itself still lays out and computes hover/hit-testing against the
  // full 1400x700 box internally (a CSS transform changes paint, not
  // layout, so none of the pixel-space geometry this file precomputed in
  // Python needs to change). The hover-card div is a sibling of
  // scale-wrapper, not inside it, and already positions itself from the
  // raw, untransformed event.event.clientX/clientY - so it needed no
  // adjustment at all for the scale to not throw its positioning off.
  // Not AI: none - not part of the spec, found and fixed during
  // verification against the real app.
  const DESIGN_WIDTH = {width};
  const scale = Math.min(document.documentElement.clientWidth / DESIGN_WIDTH, 1.05);
  document.getElementById("scale-wrapper").style.transform = "scale(" + scale + ")";

  const figData = {fig_data_json};
  const figLayout = {fig_layout_json};
  const netsTraceIdx = {nets_idx};
  const archTraceIdx = {arch_idx};
  const netsBaseSize = {nets_base_size};
  const netsHoverSize = {nets_hover_size};
  const netsColor = "{nets_color}";

  const chartDiv = document.getElementById("chart");
  const card = document.getElementById("hover-card");

  // AI-ASSISTED (Claude Code, chat) - Prompt: "点击某个球员的点球员会定在hover的
  // 效果" (clicking a player's point should lock/pin it in the hover state,
  // staying visible after the mouse moves away). Used: a single `pinned`
  // state variable (curveNumber+pointIndex, or null) - `plotly_click`
  // toggles it (click the pinned point again to unpin, click a different
  // point to move the pin there); `plotly_hover`/`plotly_unhover`/
  // `mouseleave` all check it first and skip their own show/hide/reset
  // logic entirely while something is pinned, so a pin can't be knocked
  // loose by an incidental mouse movement. This is a pure client-side JS
  // interaction, entirely inside this chart's own iframe - no
  // communication with the outer Streamlit page is needed or attempted.
  // Not AI: the interaction spec itself - given directly.
  let pinned = null;

  Plotly.newPlot(chartDiv, figData, figLayout, {{displayModeBar: false, staticPlot: false}}).then(function() {{

    function showCard(html, clientX, clientY) {{
      card.innerHTML = html;
      card.style.display = "block";
      const cw = card.offsetWidth, ch = card.offsetHeight;
      let left = clientX + 16, top = clientY + 16;
      if (left + cw > window.innerWidth) left = clientX - cw - 16;
      if (top + ch > window.innerHeight) top = clientY - ch - 16;
      card.style.left = Math.max(0, left) + "px";
      card.style.top = Math.max(0, top) + "px";
    }}

    function hideCard() {{ card.style.display = "none"; }}

    function resetNetsStyle() {{
      const n = figData[netsTraceIdx].x.length;
      Plotly.restyle(chartDiv, {{
        "marker.size": [Array(n).fill(netsBaseSize)],
        "marker.opacity": [Array(n).fill(1.0)],
        "marker.line.width": [Array(n).fill(1.5)],
      }}, [netsTraceIdx]);
    }}

    function applyNetsHighlight(pointIndex) {{
      const n = figData[netsTraceIdx].x.length;
      const sizes = Array(n).fill(netsBaseSize);
      const opacities = Array(n).fill(0.35);
      const lineWidths = Array(n).fill(1.5);
      sizes[pointIndex] = netsHoverSize;
      opacities[pointIndex] = 1.0;
      lineWidths[pointIndex] = 8;
      Plotly.animate(chartDiv, {{
        data: [{{"marker.size": sizes, "marker.opacity": opacities,
                "marker.line.width": lineWidths,
                "marker.line.color": Array(n).fill("rgba(28,26,23,0.28)")}}],
        traces: [netsTraceIdx],
      }}, {{transition: {{duration: 150}}, frame: {{duration: 150, redraw: false}}}});
    }}

    function showForPoint(pt, clientX, clientY) {{
      if (pt.curveNumber === netsTraceIdx) applyNetsHighlight(pt.pointIndex);
      showCard(pt.customdata, clientX, clientY);
    }}

    chartDiv.on("plotly_hover", function(evt) {{
      if (pinned) return;
      const pt = evt.points[0];
      if (pt.curveNumber === netsTraceIdx || pt.curveNumber === archTraceIdx) {{
        showForPoint(pt, evt.event.clientX, evt.event.clientY);
      }}
    }});

    chartDiv.on("plotly_unhover", function(evt) {{
      if (pinned) return;
      hideCard();
      resetNetsStyle();
    }});

    chartDiv.addEventListener("mouseleave", function() {{
      if (pinned) return;
      hideCard();
      resetNetsStyle();
    }});

    chartDiv.on("plotly_click", function(evt) {{
      const pt = evt.points[0];
      if (pt.curveNumber !== netsTraceIdx && pt.curveNumber !== archTraceIdx) return;
      const isSamePoint = pinned && pinned.curveNumber === pt.curveNumber && pinned.pointIndex === pt.pointIndex;
      if (isSamePoint) {{
        pinned = null;
        hideCard();
        resetNetsStyle();
      }} else {{
        if (pinned && pinned.curveNumber === netsTraceIdx) resetNetsStyle();
        pinned = {{curveNumber: pt.curveNumber, pointIndex: pt.pointIndex}};
        showForPoint(pt, evt.event.clientX, evt.event.clientY);
      }}
    }});
  }});
</script>
</body></html>
"""


def render_html(spec):
    """spec: the dict returned by build_figure_spec(). Returns a complete,
    self-contained HTML page string ready for st.iframe() (or a
    standalone file, for the Playwright-driven verification pass)."""
    import json as _json
    nets_trace = spec["data"][spec["nets_trace_index"]]
    return _PAGE_TEMPLATE.format(
        bg=CHART_BG, width=FIGURE_WIDTH_PX, height=FIGURE_HEIGHT_PX,
        card_bg=CARD_BG, card_border=CARD_BORDER, card_shadow=CARD_SHADOW,
        fig_data_json=_json.dumps(spec["data"]), fig_layout_json=_json.dumps(spec["layout"]),
        nets_idx=spec["nets_trace_index"], arch_idx=spec["arch_trace_index"],
        nets_base_size=nets_trace["marker"]["size"], nets_hover_size=16,
        nets_color=NETS_DOT_COLOR,
    )
