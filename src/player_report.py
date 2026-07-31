"""
player_report.py - per-player PDF scouting report, pure PDF-building only.

AI-ASSISTED (Claude Code, chat)
Prompt: "For PDF page, please refer to README.md and reference-1b-briefing.html
file I just sent to you, I want this format please" - a pixel-precise 2-page
A4 design spec + a working HTML/CSS/SVG reference file, replacing the
previous one-page reportlab-built report wholesale. Full spec/plan in
AI_USAGE.md and the approved plan this session worked from.

Used: this module still takes a single plain `report_data` dict (built by
portal.py's `collect_report_data()`, same CORE RULE as before - nothing
here is computed, nothing is fabricated), but now renders it through a
Jinja2 HTML template (`report_template/report.html.jinja`) and exports the
result to PDF via Playwright's headless Chromium
(`page.pdf(format="A4", print_background=True)`), instead of reportlab
direct-drawing. Chart geometry (radar/diverging-bars/drift-lines/dumbbell)
lives in the sibling `report_svg_charts.py` module - pure functions over
already-real numbers, independently verified against the reference file's
own hardcoded SVG coordinates before being trusted (see that module's own
docstring). Fonts (Lato 400/700/900) are vendored into
`report_template/fonts/` and embedded as base64 data URIs, so the whole
rendered HTML string is self-contained - no live network fetch, no
`file://` path resolution needed by Playwright.

Still Streamlit-free, still a one-directional dependency (portal.py
imports this module, never the reverse) - unchanged from the previous
version, for the same reason (avoid portal.py's top-level
`st.set_page_config`/sidebar code re-running on import). The only
cross-module import is `hull_callout_chart.get_headshot_data_uri` - same
as before, already Streamlit-free.

Not AI: the CORE RULE, the exact page design (colors/type/spacing/chart
geometry), and the data contract shape - all given directly, in
README.md/reference-1b-briefing.html. The rendering-technology swap
(reportlab -> Jinja2+Playwright) was proposed here after directly
verifying Playwright's Chromium was already installed and working in this
environment (not assumed) and confirmed with the user before implementing
- see AI_USAGE.md for the verification steps and the plan this was built
from.
"""

from __future__ import annotations

import base64
from pathlib import Path

import jinja2
from playwright.sync_api import sync_playwright

import report_svg_charts
from hull_callout_chart import get_headshot_data_uri

TEMPLATE_DIR = Path(__file__).resolve().parent / "report_template"
FONT_DIR = TEMPLATE_DIR / "fonts"

# Long archetype names abbreviate in tight labels (diverging-bar/dumbbell
# row labels, archetype-mix rows) per the spec's own examples ("Shooting
# Spec.", "Rim Prot. / Roll", "Combo Gd.", "Trad. PM"). Covers all K=8
# names; anything not in this map (e.g. a future re-fit with different
# labels) falls back to the real name unabbreviated rather than guessing.
ARCHETYPE_ABBREV = {
    "3&D Wing": "3&D Wing",
    "Inside Scoring Big": "Inside Scor. Big",
    "Rim Protector / Roll Man": "Rim Prot. / Roll",
    "Combo Guard": "Combo Gd.",
    "Mobile Big": "Mobile Big",
    "Shooting Specialist": "Shooting Spec.",
    "Offensive Engine": "Off. Engine",
    "Traditional Playmaker": "Trad. PM",
}


def abbreviate_archetype(name):
    return ARCHETYPE_ABBREV.get(name, name)


def _ordinal(n):
    """88 -> '88th', 82 -> '82nd', 74 -> '74th'."""
    n = int(round(n))
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suffix}"


def _font_data_uri(path):
    b64 = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:font/woff2;base64,{b64}"


class PDFExportUnavailable(Exception):
    """Raised when Playwright's Chromium can't be launched - the one-time
    `playwright install chromium` setup step (see requirements.txt) wasn't
    run in this environment."""


_JINJA_ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATE_DIR)), autoescape=True)
_JINJA_ENV.filters["signed"] = report_svg_charts.signed
_JINJA_ENV.filters["ordinal"] = _ordinal
_JINJA_ENV.filters["abbreviate_archetype"] = abbreviate_archetype

_FONT_DATA_URIS = {
    "regular": _font_data_uri(FONT_DIR / "Lato-Regular.woff2"),
    "bold": _font_data_uri(FONT_DIR / "Lato-Bold.woff2"),
    "black": _font_data_uri(FONT_DIR / "Lato-Black.woff2"),
}


def _build_chart_svgs(report_data: dict) -> dict:
    """Turns collect_report_data()'s raw numbers (archetype shares, z-scores,
    pp deltas) into the actual <svg>/HTML chart fragments via
    report_svg_charts's pure geometry functions - kept here, not in
    portal.py, so collect_report_data's own return value stays a clean data
    contract with no HTML/SVG string-building mixed into it."""
    neighbors = report_data["neighbors"]
    radar_svg = report_svg_charts.radar_svg(
        neighbors["radar_axes"], neighbors["radar_him"], neighbors["radar_neighbors"],
        neighbors["radar_dot_indices"],
    )

    environment = report_data["environment"]
    lift_bars_svg = report_svg_charts.diverging_bars_html(environment["lift_rows"])
    minutes_bars_svg = report_svg_charts.diverging_bars_html(
        environment["minutesVsTypical"]["rows"], value_width_px=30)
    elasticity_bars_svg = report_svg_charts.diverging_bars_html(environment["elasticity"]["rows"])

    dumbbell_svg = report_svg_charts.dumbbell_html(report_data["miscast"]["dumbbell"])

    drift = report_data["drift"]
    drift_svg = report_svg_charts.drift_line_svg(drift["seasons"], drift["series"]) if drift else None

    return {
        "radar_svg": radar_svg, "lift_bars_svg": lift_bars_svg, "minutes_bars_svg": minutes_bars_svg,
        "elasticity_bars_svg": elasticity_bars_svg, "dumbbell_svg": dumbbell_svg, "drift_svg": drift_svg,
    }


def render_report_html(report_data: dict) -> str:
    """report_data -> the full rendered HTML string. Pure Jinja2 render
    (plus building the chart SVG fragments - see _build_chart_svgs), no
    other I/O beyond the template file already loaded at import time -
    easy to eyeball or diff independent of PDF export."""
    template = _JINJA_ENV.get_template("report.html.jinja")
    return template.render(**report_data, **_build_chart_svgs(report_data), fonts=_FONT_DATA_URIS)


def build_pdf(report_data: dict) -> bytes:
    """report_data -> PDF bytes. Same input always produces the same
    output (no hidden state beyond the vendored fonts/template on disk)."""
    html = render_report_html(report_data)
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            try:
                page = browser.new_page()
                page.set_content(html, wait_until="networkidle")
                page.emulate_media(media="print")
                return page.pdf(
                    format="A4", print_background=True,
                    margin={"top": "0", "bottom": "0", "left": "0", "right": "0"},
                )
            finally:
                browser.close()
    except Exception as e:
        raise PDFExportUnavailable(
            f"PDF export unavailable ({e}). Run `playwright install chromium` once in this environment."
        ) from e
