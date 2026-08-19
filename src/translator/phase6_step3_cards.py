"""
Phase 6 Step 3 - rookie cards. One self-contained HTML + one .md per
rookie, plus a combined page, built with jinja2. Content and ordering per
spec: header, predicted mix (point estimates only, Locked Decision 1),
top-3 prose with archetypoid names, college profile (input, clearly
distinguished from the prediction), comps table, the exact Locked
Decision 2 confidence sentence, saturation caveat if flagged, and an
identical limits footnote on every card.

Run: python3 src/translator/phase6_step3_cards.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import jinja2
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from rookie_card_labels import load_college_labels, load_nba_labels, load_nba_archetypoids  # noqa: E402

K = 8
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
CONFIG_PATH = REPO_ROOT / "config.yaml"
CARDS_DIR = REPO_ROOT / "reports" / "rookie_cards"
TEMPLATE_DIR = REPO_ROOT / "src" / "report_template"

# Locked Decision 2 - quoted verbatim, no other confidence language anywhere on a card.
CONFIDENCE_LINE = (
    "On the 2025 draft class (n=36, the only class this model has been tested against that "
    "it never trained on), this model's predictions had a 52.8% chance of naming the correct "
    "top archetype, a 69.4% chance the correct archetype was in its top two, and an average "
    "Jensen-Shannon divergence of 0.169 between predicted and actual archetype mixes."
)

SATURATION_CAVEAT_TEMPLATE = (
    "This player's college profile is unusually concentrated (top archetype weight {pct:.0f}%, "
    "{pctile:.0f}th percentile among all 273 anchors this model has ever seen). The model "
    "systematically under-predicts concentration for such players — see the 2025 case of Ryan "
    "Kalkbrenner, whose true rookie archetype mix was 100% a single archetype while the model "
    "predicted only 17% weight there. The actual rookie role may be more specialized than shown "
    "above, in the direction of the top archetype."
)

LIMITS_FOOTNOTE = [
    "This model is blind to shot-location and play-type data on the college side (16 of the "
    "NBA archetype basis's 29 dimensions have no college counterpart) — it cannot reliably "
    "separate certain role variants, e.g. a rim-protecting roll man from a perimeter-mobile "
    "big, or a spot-up shooter from a movement shooter.",
    "It was trained only on players who earned at least 300 NBA minutes as rookies — it answers "
    "\"what role will he play if he plays,\" never \"will he play.\"",
    "Elite prospects whose final college season was cut short by injury, suspension, or opt-out "
    "are absent from the training data — the model has never seen that profile, so a prediction "
    "for such a player would be extrapolation, not interpolation.",
    "Posterior uncertainty intervals from this model are not calibrated (validated on the 2025 "
    "holdout — see the confidence line above) and are deliberately not shown.",
    "Coaching decisions, scheme fit, and playing-time opportunity are not observable to any "
    "statistical model and are not represented here.",
]


def card_safe_label(label: str) -> str:
    """Strips the "(...see note above)" qualifier from a college archetype
    label for standalone-card display - that phrase points at a note in
    reports/college_archetypes.md that doesn't exist on the card itself,
    which would confuse a reader who has never seen this project (the
    design constraint Step 3 explicitly requires). The underlying weak/
    noisy-archetype caveat is preserved in spirit via a shorter, self-
    contained parenthetical instead of dropped silently."""
    if "see note above" in label:
        base = label.split(" (")[0]
        return f"{base} (a noisier, less distinct archetype in this model)"
    return label


def fmt_outcome(top2: list) -> str:
    nba_labels = load_nba_labels()
    j0, w0 = top2[0]
    j1, w1 = top2[1]
    return f"{nba_labels[j0]['label']} ({w0*100:.0f}%), then {nba_labels[j1]['label']} ({w1*100:.0f}%)"


def build_card_context(rookie: dict, rookie_cfg: dict, college_labels: dict, nba_labels: dict,
                        archetypoids: dict, meta: dict) -> dict:
    pred_bars = sorted(
        [(j, nba_labels[j]["label"], rookie["y_pred"][j] * 100) for j in range(K)],
        key=lambda x: -x[2])
    college_bars = sorted(
        [(j, card_safe_label(college_labels[j]), rookie["c_alpha"][j] * 100) for j in range(K)],
        key=lambda x: -x[2])

    top3_prose, top3_prose_md = [], []
    for rank, (j, pct) in enumerate(rookie["y_top3"], start=1):
        archetypoid = archetypoids[j]
        ord_word = {1: "1st", 2: "2nd", 3: "3rd"}[rank]
        line = (f"<b>{ord_word}: {nba_labels[j]['label']}</b> ({pct*100:.0f}%) — closest real "
                f"comparison: {archetypoid['player']} ({archetypoid['season']})")
        top3_prose.append(line)
        top3_prose_md.append(line.replace("<b>", "**").replace("</b>", "**"))

    comps = []
    for c in rookie["comps"]:
        comps.append({
            "name": c["name"], "draft_year": c["draft_year"], "pick": c["pick"], "college": c["college"],
            "college_archetype_label": card_safe_label(college_labels[c["college_top_archetype"]]),
            "actual_outcome": fmt_outcome(c["true_rookie_top2"]),
        })

    saturation_caveat = None
    if rookie["saturated"]:
        saturation_caveat = SATURATION_CAVEAT_TEMPLATE.format(
            pct=rookie["c_alpha_max"] * 100, pctile=rookie["c_alpha_max_percentile"])

    height_in = rookie_cfg.get("_height_in")
    height_str = f"{height_in // 12}'{height_in % 12}\"" if height_in else "—"

    return {
        "rookie": {"name": rookie["display_name"], "first_name": rookie["display_name"].split()[0],
                   "college": rookie["college_team"], "position": rookie_cfg.get("_position", "—"),
                   "height": height_str, "pick": rookie_cfg["draft_pick_overall"]},
        "meta": meta,
        "pred_bars": pred_bars, "college_bars": college_bars,
        "top3_prose": top3_prose, "top3_prose_md": top3_prose_md,
        "comps": comps, "confidence_line": CONFIDENCE_LINE,
        "saturation_caveat": saturation_caveat, "limits_footnote": LIMITS_FOOTNOTE,
    }


def main() -> None:
    CARDS_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    projections = json.loads((TRANSLATOR_DIR / "rookie_projections_full.json").read_text())

    draft_picks = json.loads((REPO_ROOT / "data" / "raw" / "cbbd" / "draft_picks_all.json").read_text())
    height_pos_by_aid = {p["athleteId"]: (p.get("height"), p.get("position")) for p in draft_picks}

    college_labels = load_college_labels()
    nba_labels = load_nba_labels()
    archetypoids = load_nba_archetypoids()

    git_hash = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip() or "uncommitted"
    manifest = json.loads((TRANSLATOR_DIR / "deployment" / "manifest.json").read_text())
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "git_hash": git_hash,
        "model_version": f"T1_b, deployment refit {manifest['fit_date']}, n={manifest['n_train']}",
    }

    env = jinja2.Environment(loader=jinja2.FileSystemLoader(TEMPLATE_DIR), autoescape=False)
    html_template = env.get_template("rookie_card.html.jinja")
    md_template = env.get_template("rookie_card.md.jinja")

    all_contexts = []
    for rookie, rookie_cfg in zip(projections, config["rookies"]):
        aid = rookie_cfg["cbbd_athlete_id"]
        height_in, position = height_pos_by_aid.get(aid, (None, None))
        rookie_cfg = {**rookie_cfg, "_height_in": height_in, "_position": position}
        ctx = build_card_context(rookie, rookie_cfg, college_labels, nba_labels, archetypoids, meta)
        all_contexts.append(ctx)

        slug = rookie["display_name"].lower().replace(" ", "_").replace(".", "")
        html_out = html_template.render(**ctx)
        md_out = md_template.render(**ctx)
        (CARDS_DIR / f"{slug}.html").write_text(html_out)
        (CARDS_DIR / f"{slug}.md").write_text(md_out)
        print(f"Wrote {slug}.html / {slug}.md")

    # combined page: simple concatenation with a shared header, reusing the same per-rookie
    # sections template structure via string concatenation of the individual bodies
    combined_html = ["<!DOCTYPE html><html><head><meta charset='utf-8'>"
                      "<title>Nets 2026 Rookie Archetype Cards</title></head><body>"
                      "<div style='max-width:900px;margin:0 auto;padding:2rem 1.5rem;"
                      "font-family:Georgia,serif'><h1>Nets 2026 Rookie Archetype Cards</h1>"
                      f"<p style='color:#7A7266;font-family:monospace;font-size:0.8rem'>"
                      f"generated {meta['timestamp_utc']} · commit {meta['git_hash']} · "
                      f"model {meta['model_version']}</p><hr></div>"]
    combined_md = [f"# Nets 2026 Rookie Archetype Cards\n\n*generated {meta['timestamp_utc']} · "
                   f"commit `{meta['git_hash']}` · model {meta['model_version']}*\n\n---\n"]
    for rookie, rookie_cfg in zip(projections, config["rookies"]):
        aid = rookie_cfg["cbbd_athlete_id"]
        height_in, position = height_pos_by_aid.get(aid, (None, None))
        rookie_cfg2 = {**rookie_cfg, "_height_in": height_in, "_position": position}
        ctx = build_card_context(rookie, rookie_cfg2, college_labels, nba_labels, archetypoids, meta)
        combined_html.append(html_template.render(**ctx).split("<body>")[1].split("</body>")[0])
        combined_md.append(md_template.render(**ctx))
        combined_md.append("\n\n---\n")
    combined_html.append("</body></html>")
    (CARDS_DIR / "all_rookies.html").write_text("\n".join(combined_html))
    (CARDS_DIR / "all_rookies.md").write_text("\n".join(combined_md))
    print("Wrote all_rookies.html / all_rookies.md")


if __name__ == "__main__":
    main()
