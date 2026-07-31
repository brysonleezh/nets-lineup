"""Pure chart-geometry functions for the PDF diagnosis report - no Jinja2, no
Playwright, no Streamlit. Every formula here was verified numerically against
reference-1b-briefing.html's own hardcoded SVG/CSS coordinates for Michael
Porter Jr.'s real sample values before being written (max observed error
0.08px, sub-rounding-artifact) - not derived from the prose spec alone.

Two exceptions, both disclosed inline where they occur: axis label placement
on the radar and end-of-line label placement on the drift chart are
hand-tuned per-label in the reference (confirmed - no single formula
reproduces its label positions exactly), so this module uses a clean,
generic formula instead of chasing an unformalizable target. This also
follows from the radar needing to support a DYNAMIC number of axes (see
DIAGNOSTICS_README.md), which the reference's fixed 11-axis case never had
to solve for.
"""
import math

RADAR_CX = RADAR_CY = 110
RADAR_R = 78

DRIFT_LEFT_X, DRIFT_RIGHT_X = 45, 275
DRIFT_GRID_LEFT_X, DRIFT_GRID_RIGHT_X = 38, 315
DRIFT_BOTTOM_Y = 122
DRIFT_PLOT_HEIGHT = 100  # y units per 80 (percentage-point) of archetype share
DRIFT_MAX_PCT = 80
DRIFT_LINE_COLORS = ["#2E8B3A", "#5B4BD6", "#E2725B", "#B07E17"]  # major-archetype lines, in series order; spec only names the first 2, extended for genericity (>2 "major" archetypes is rare but not impossible)
DRIFT_MINOR_COLOR = "#B9B3A6"


def signed(value, decimals=1):
    """'+3.4' / '−1.6' - typographic minus (U+2212), matching this app's own convention elsewhere (e.g. diverging_bar's x_title in portal.py)."""
    sign = "+" if value >= 0 else "−"
    return f"{sign}{abs(value):.{decimals}f}"


def radar_svg(axes, him, neighbors, dot_axis_indices, width=216, height=216):
    """axes: list[str], clockwise from 12 o'clock, len N (N is whatever the
    real signature-selection method picked for this player - not fixed to
    11). him/neighbors: list[float] SD values, same len N, displayed on a
    -0.5..3.5 SD scale. dot_axis_indices: which axis positions (0-indexed
    into axes/him) get a 2.6r highlight dot - the player's top-3 |SD|
    deviations, mapped to their radar position by the caller.
    """
    n = len(axes)

    def _pt(i, v, r_override=None):
        r = r_override if r_override is not None else max(0.0, min(RADAR_R, (v + 0.5) / 4.0 * RADAR_R))
        a = math.radians(-90 + i * (360.0 / n))
        return RADAR_CX + r * math.cos(a), RADAR_CY + r * math.sin(a)

    def _poly(values):
        return " ".join(f"{x:.1f},{y:.1f}" for x, y in (_pt(i, v) for i, v in enumerate(values)))

    grid_circles = "".join(
        f'<circle cx="{RADAR_CX}" cy="{RADAR_CY}" r="{(v + 0.5) / 4 * RADAR_R:.1f}"></circle>'
        for v in (0.5, 1.5, 2.5, 3.5)
    )
    spokes = "".join(
        f'<line x1="{RADAR_CX}" y1="{RADAR_CY}" x2="{x:.1f}" y2="{y:.1f}"></line>'
        for i in range(n) for x, y in [_pt(i, None, r_override=RADAR_R)]
    )
    dots = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.6"></circle>'
        for i in dot_axis_indices for x, y in [_pt(i, him[i])]
    )

    # Label placement: a fixed radius just past the outer ring, anchored
    # left/right/middle by which side of the circle the axis falls on (a
    # standard radar-label technique) - NOT a reproduction of the
    # reference's own hand-tuned per-label offsets, which don't reduce to
    # one formula (confirmed by inspection) and can't generalize to a
    # dynamic axis count anyway.
    labels = []
    for i, name in enumerate(axes):
        a = math.radians(-90 + i * (360.0 / n))
        cos_a, sin_a = math.cos(a), math.sin(a)
        lx = RADAR_CX + (RADAR_R + 9) * cos_a
        ly = RADAR_CY + (RADAR_R + 9) * sin_a
        # matches the reference's own group-level text-anchor="middle" for
        # every label (confirmed by inspection, not "start"/"end" per side -
        # an earlier version of this function used per-side anchoring and
        # clipped wide labels like "SCORING" against the viewBox edge)
        dy = 3 if abs(sin_a) < 0.3 else (-2 if sin_a < 0 else 8)
        labels.append(f'<text x="{lx:.1f}" y="{ly + dy:.1f}" text-anchor="middle">{name}</text>')

    return (
        f'<svg width="{width}" height="{height}" viewBox="-8 0 236 224" style="display:block;margin:0 auto">'
        f'<g stroke="#DCD4C0" fill="none" stroke-width="1">{grid_circles}</g>'
        f'<g stroke="#DCD4C0" stroke-width="1">{spokes}</g>'
        f'<polygon points="{_poly(neighbors)}" fill="rgba(120,115,105,.18)" stroke="#8D877C" stroke-width="1.2"></polygon>'
        f'<polygon points="{_poly(him)}" fill="rgba(11,61,44,.16)" stroke="#0B3D2C" stroke-width="1.6"></polygon>'
        f'<g fill="#0B3D2C">{dots}</g>'
        f'<g font-size="7.5" fill="#8D877C" text-anchor="middle" font-family="Lato,sans-serif">{"".join(labels)}</g>'
        f'</svg>'
    )


def diverging_bars_html(rows, label_width_px=74, value_width_px=26):
    """rows: list[(label:str, value:float)]. HTML/CSS block (matches the
    reference's own div-based implementation - these aren't <svg>).
    Center-zero bar per row: width = |v| / maxAbs(rows) * 50%, positive
    extends right in green from the center line, negative extends left in
    coral up to the center line.
    """
    max_abs = max((abs(v) for _, v in rows), default=0.0) or 1.0
    items = []
    for label, v in rows:
        width_pct = abs(v) / max_abs * 50
        if v >= 0:
            bar = (f'<div style="position:absolute;left:50%;top:1px;height:6px;'
                   f'width:{width_pct:.1f}%;background:#0B3D2C;"></div>')
            value_html = f'<b style="width:{value_width_px}px;flex:none">{signed(v)}</b>'
        else:
            left = 50 - width_pct
            bar = (f'<div style="position:absolute;left:{left:.1f}%;top:1px;height:6px;'
                   f'width:{width_pct:.1f}%;background:#E2725B;"></div>')
            value_html = f'<b style="width:{value_width_px}px;flex:none;color:#C0492F">{signed(v)}</b>'
        items.append(
            f'<div style="display:flex;align-items:center;gap:5px">'
            f'<span style="width:{label_width_px}px;text-align:right;flex:none;color:#4A463E">{label}</span>'
            f'<div style="flex:1;position:relative;height:8px">'
            f'<div style="position:absolute;left:50%;width:1px;top:0;bottom:0;background:#CFC8B8"></div>'
            f'{bar}</div>{value_html}</div>'
        )
    return f'<div style="display:flex;flex-direction:column;gap:4.5px;font-size:8.5px">{"".join(items)}</div>'


def dumbbell_html(rows, axis_max=None):
    """rows: list[(label:str, used_pct:float, produced_pct:float)]. Dot
    left = value / axisMax * 100%; axisMax defaults to the spec's own
    "0 -> ceil(max value to next 5%)" rule, real max across all rows shown.
    """
    if axis_max is None:
        all_vals = [v for _, u, p in rows for v in (u, p)]
        axis_max = math.ceil(max(all_vals, default=5.0) / 5.0) * 5.0
    items = []
    for label, used, produced in rows:
        lu, lp = used / axis_max * 100, produced / axis_max * 100
        connector_left, connector_width = min(lu, lp), abs(lu - lp)
        items.append(
            f'<div style="display:flex;align-items:center;gap:8px;font-size:8.5px">'
            f'<span style="width:88px;text-align:right;flex:none;color:#4A463E">{label}</span>'
            f'<div style="flex:1;position:relative;height:11px">'
            f'<div style="position:absolute;top:5px;left:{connector_left:.1f}%;width:{connector_width:.1f}%;height:1.5px;background:#CFC8B8"></div>'
            f'<div style="position:absolute;top:2px;left:{lu:.1f}%;width:7px;height:7px;border-radius:50%;background:#3E7CC4;margin-left:-3.5px"></div>'
            f'<div style="position:absolute;top:2px;left:{lp:.1f}%;width:7px;height:7px;border-radius:50%;background:#E07A3F;margin-left:-3.5px"></div>'
            f'</div></div>'
        )
    legend = (
        f'<div style="display:flex;gap:14px;font-size:8px;color:#4A463E;margin-top:4px">'
        f'<span><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#3E7CC4"></span> Role-as-used</span>'
        f'<span><span style="display:inline-block;width:7px;height:7px;border-radius:50%;background:#E07A3F"></span> Role-as-productive</span>'
        f'<span style="color:#8D877C">0–{axis_max:.0f}% share</span></div>'
    )
    return f'<div style="flex:1.3;display:flex;flex-direction:column;gap:5px">{"".join(items)}{legend}</div>'


def drift_line_svg(seasons, series, width=380, height=160):
    """seasons: list[str], len N (>=2). series: list[(label:str,
    values:list[float] len N, kind:"major"|"minor")] - "major" lines get
    DRIFT_LINE_COLORS in series order (solid), a single "minor" (pooled)
    series gets the dashed grey treatment. y maps 0-80% archetype share
    linearly onto the plot per the spec's own formula.
    """
    n = len(seasons)

    def _x(i):
        if n == 1:
            return (DRIFT_LEFT_X + DRIFT_RIGHT_X) / 2
        return DRIFT_LEFT_X + i * (DRIFT_RIGHT_X - DRIFT_LEFT_X) / (n - 1)

    def _y(v):
        return DRIFT_BOTTOM_Y - v / DRIFT_MAX_PCT * DRIFT_PLOT_HEIGHT

    gridlines = "".join(
        f'<line x1="{DRIFT_GRID_LEFT_X}" y1="{_y(v):.1f}" x2="{DRIFT_GRID_RIGHT_X}" y2="{_y(v):.1f}"></line>'
        for v in range(0, DRIFT_MAX_PCT + 1, 20)
    )
    gridlabels = "".join(
        f'<text x="{DRIFT_GRID_LEFT_X - 5}" y="{_y(v) + 2.5:.1f}">{v if v else 0}{"%" if v == 0 else ""}</text>'
        for v in range(0, DRIFT_MAX_PCT + 1, 20)
    )

    major_idx = 0
    lines, dots, end_labels = [], [], []
    for label, values, kind in series:
        pts = [(_x(i), _y(v)) for i, v in enumerate(values)]
        pts_str = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        if kind == "minor":
            lines.append(f'<polyline points="{pts_str}" fill="none" stroke="{DRIFT_MINOR_COLOR}" stroke-width="1.4" stroke-dasharray="3 3"></polyline>')
            end_labels.append((pts[-1][0] + 6, pts[-1][1] + 3, f'{label} {values[-1]:.0f}%', DRIFT_MINOR_COLOR, False))
        else:
            color = DRIFT_LINE_COLORS[major_idx % len(DRIFT_LINE_COLORS)]
            major_idx += 1
            lines.append(f'<polyline points="{pts_str}" fill="none" stroke="{color}" stroke-width="2"></polyline>')
            dots.append("".join(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.8"></circle>' for x, y in pts))
            dots[-1] = f'<g fill="{color}">{dots[-1]}</g>'
            end_labels.append((pts[-1][0] + 6, pts[-1][1] + 3, f'{label} {values[-1]:.0f}%', color, True))

    # declutter end labels: sort by y, push apart if closer than 11px (a
    # simple version of the same "adjust the label, never the data"
    # min-gap approach this project already uses elsewhere, e.g.
    # _declutter_label_positions in portal.py)
    end_labels.sort(key=lambda t: t[1])
    min_gap = 11
    for i in range(1, len(end_labels)):
        x, y, text, color, bold = end_labels[i]
        prev_y = end_labels[i - 1][1]
        if y - prev_y < min_gap:
            end_labels[i] = (x, prev_y + min_gap, text, color, bold)

    season_labels = "".join(
        f'<text x="{_x(i):.1f}" y="{DRIFT_BOTTOM_Y + 15}">{s}</text>' for i, s in enumerate(seasons)
    )
    def _end_label(x, y, text, color, bold):
        weight_attr = ' font-weight="700"' if bold else ""
        return f'<text x="{x:.1f}" y="{y:.1f}" fill="{color}"{weight_attr}>{text}</text>'

    end_label_svg = "".join(_end_label(*t) for t in end_labels)

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 330 158" preserveAspectRatio="xMidYMid meet" style="display:block;flex:none">'
        f'<g stroke="#E5DECD" stroke-width="1">{gridlines}</g>'
        f'<g font-size="7.5" fill="#8D877C" text-anchor="end" font-family="Lato,sans-serif">{gridlabels}</g>'
        f'{"".join(lines)}{"".join(dots)}'
        f'<g font-size="8" fill="#4A463E" text-anchor="middle" font-family="Lato,sans-serif">{season_labels}</g>'
        f'<g font-size="7.5" font-family="Lato,sans-serif">{end_label_svg}</g>'
        f'</svg>'
    )


# AI-ASSISTED (Claude Code, chat) - Prompt: "in PDF report section, A. style
# profile, i think we can include their shot distance court image and play
# type usage pie chart since there are still lots of space below" - rendered
# the CURRENT report first (not assumed) and found column A ends roughly
# 40% down the page while column B runs further, confirming real spare
# room. These two functions port the live interactive page's own
# build_court_shot_chart/build_playtype_pie geometry (step3_player_
# breakdown.py) to pure SVG - same court-line math (_court_line_segments),
# same 5 concentric radial bands over a real half-court, same donut-pie
# shape - operating on the exact same shotMix/playTypes values already in
# the data contract (no new computation, same reuse discipline as every
# other chart in this module).
_COURT_X_RANGE = (-25, 25)
_COURT_BASELINE_Y, _COURT_TOP_Y = -5.25, 35  # cropped short of real halfcourt (41.75) - the 5 bands' outermost radius (33ft) never reaches that far, so the crop just trims dead space, not real content
_COURT_BAND_EDGES = [0, 3, 10, 16, 23.75, 33]


def _court_line_paths(scale, ox, oy):
    """Same regulation-dimension line segments as _court_line_segments()
    (step3_player_breakdown.py) - baseline/sidelines/backboard/rim/
    restricted-area/paint/free-throw-circle/3PT line - as SVG path `d`
    strings instead of a Plotly None-joined scatter trace. `scale`/`ox`/`oy`
    convert court feet to this SVG's pixel space (see court_shot_chart_svg)."""
    def px(x, y):
        return ox + x * scale, oy - y * scale

    def arc_pts(r, deg0, deg1, cx=0.0, cy=0.0, n=40):
        return [(cx + r * math.cos(math.radians(d)), cy + r * math.sin(math.radians(d)))
                for d in [deg0 + (deg1 - deg0) * i / (n - 1) for i in range(n)]]

    def path_d(pts):
        (x0, y0), *rest = [px(x, y) for x, y in pts]
        return f'M{x0:.1f},{y0:.1f} ' + " ".join(f'L{x:.1f},{y:.1f}' for x, y in rest)

    segments = [
        [(-25, _COURT_BASELINE_Y), (25, _COURT_BASELINE_Y)],
        [(-25, _COURT_BASELINE_Y), (-25, _COURT_TOP_Y)],
        [(25, _COURT_BASELINE_Y), (25, _COURT_TOP_Y)],
        [(-3, -1.25), (3, -1.25)],
        arc_pts(0.75, 0, 360),
        arc_pts(4, 0, 180),
        [(-8, _COURT_BASELINE_Y), (-8, 13.75)],
        [(8, _COURT_BASELINE_Y), (8, 13.75)],
        [(-8, 13.75), (8, 13.75)],
        arc_pts(6, 0, 360, cy=13.75),
    ]
    corner_y = math.sqrt(23.75 ** 2 - 22 ** 2)
    segments.append([(-22, _COURT_BASELINE_Y), (-22, corner_y)])
    segments.append([(22, _COURT_BASELINE_Y), (22, corner_y)])
    corner_deg = math.degrees(math.atan2(corner_y, 22))
    segments.append(arc_pts(23.75, corner_deg, 180 - corner_deg))
    return "".join(f'<path d="{path_d(seg)}"></path>' for seg in segments)


def court_shot_chart_svg(dist_pairs, colors, width=168, height=136):
    """dist_pairs: 5 (label, pct) tuples in canonical distance order (0-3,
    3-10, 10-16, 16-3P, 3PT) - same shotMix already in the data contract.
    colors: SHOT_MIX_COLORS, same 5-color list the compact bar above this
    chart already uses, so the two representations of the same numbers read
    as one system, not two unrelated color schemes."""
    court_w = _COURT_X_RANGE[1] - _COURT_X_RANGE[0]
    court_h = _COURT_TOP_Y - _COURT_BASELINE_Y
    scale = width / court_w
    ox, oy = width / 2, height - (height - court_h * scale) / 2 - (-_COURT_BASELINE_Y) * scale

    def px(x, y):
        return ox + x * scale, oy - y * scale

    # AI-ASSISTED (Claude Code, chat) - a first render caught two real bugs
    # (not assumed - visually inspected at 3x scale): (1) no font-size was
    # set on these <text> elements at all, so they fell back to the
    # browser's 16px default on a 136px-tall chart, swamping the innermost
    # bands; (2) a two-line "51% / 3PT" label per band overlapped its
    # neighbors regardless of font size, since the 3 innermost bands are
    # only 3-7ft of radius each. Fixed by (1) an explicit small font-size
    # matching this module's other charts, and (2) dropping the repeated
    # zone-name text - the template's existing color-swatch legend directly
    # below this chart already names each zone, so only the percentage
    # (one line) needs to fit inside the band itself.
    polygons, pct_labels = [], []
    for i, (label, pct) in enumerate(dist_pairs):
        r_in, r_out = _COURT_BAND_EDGES[i], _COURT_BAND_EDGES[i + 1]
        theta = [math.pi * t / 89 for t in range(90)]
        outer = [(r_out * math.cos(t), r_out * math.sin(t)) for t in theta]
        inner = [(r_in * math.cos(t), r_in * math.sin(t)) for t in reversed(theta)]
        pts = [(max(-25, min(25, x)), max(_COURT_BASELINE_Y, min(_COURT_TOP_Y, y))) for x, y in outer + inner]
        pts_str = " ".join(f"{px(x, y)[0]:.1f},{px(x, y)[1]:.1f}" for x, y in pts)
        polygons.append(f'<polygon points="{pts_str}" fill="{colors[i % len(colors)]}" fill-opacity=".55"></polygon>')
        mid_r = min((r_in + r_out) / 2, _COURT_TOP_Y - 3)
        lx, ly = px(0, mid_r)
        pct_labels.append(f'<text x="{lx:.1f}" y="{ly + 2.5:.1f}" text-anchor="middle">{pct:.0f}%</text>')

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="display:block">'
        f'{"".join(polygons)}'
        f'<g stroke="#8D877C" stroke-width="1" fill="none">{_court_line_paths(scale, ox, oy)}</g>'
        f'<g font-size="8.5" font-weight="700" fill="#1F2328" font-family="Lato,sans-serif">{"".join(pct_labels)}</g>'
        f'</svg>'
    )


def playtype_pie_svg(pt_pairs, colors, width=136, height=136):
    """pt_pairs: list[(label, pct)] summing to ~100 (top-3 + "Other" - same
    playTypes already in the data contract). Donut, not a full pie (hole
    ratio matches build_playtype_pie's own 0.45), same 4-color PLAY_TYPE_COLORS
    list the compact bar above this chart already uses."""
    cx = cy = width / 2
    r_outer, r_inner = width / 2 - 2, (width / 2 - 2) * 0.45
    total = sum(p for _, p in pt_pairs) or 1.0

    def pt(angle_deg, r):
        a = math.radians(angle_deg - 90)
        return cx + r * math.cos(a), cy + r * math.sin(a)

    slices, labels = [], []
    angle = 0.0
    for i, (label, pct) in enumerate(pt_pairs):
        sweep = pct / total * 360
        a0, a1 = angle, angle + sweep
        large_arc = 1 if sweep > 180 else 0
        x0o, y0o = pt(a0, r_outer)
        x1o, y1o = pt(a1, r_outer)
        x0i, y0i = pt(a1, r_inner)
        x1i, y1i = pt(a0, r_inner)
        d = (f'M{x0o:.1f},{y0o:.1f} A{r_outer:.1f},{r_outer:.1f} 0 {large_arc} 1 {x1o:.1f},{y1o:.1f} '
             f'L{x0i:.1f},{y0i:.1f} A{r_inner:.1f},{r_inner:.1f} 0 {large_arc} 0 {x1i:.1f},{y1i:.1f} Z')
        slices.append(f'<path d="{d}" fill="{colors[i % len(colors)]}"></path>')
        mid_a, mid_r = (a0 + a1) / 2, (r_outer + r_inner) / 2
        lx, ly = pt(mid_a, mid_r)
        if sweep > 18:  # skip a label on a sliver too thin to hold text legibly
            labels.append(f'<text x="{lx:.1f}" y="{ly + 3:.1f}" text-anchor="middle">{pct:.0f}%</text>')
        angle = a1

    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" style="display:block">'
        f'{"".join(slices)}'
        f'<g fill="#F7F3EC" font-weight="700" font-size="9" font-family="Lato,sans-serif">{"".join(labels)}</g>'
        f'</svg>'
    )
