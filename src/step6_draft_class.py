"""
Portal page: Draft Class 2026 - the portal's landing page.

WHY THIS PAGE EXISTS (and why it is first): every other page in this app is
organised around how the model was BUILT (fit the basis -> describe a player ->
diagnose him -> translate a rookie -> export). That order is right for a
methods reader and wrong for a coach, who opens the app asking "what did we
just draft, and what do we do with him?" - a question whose answer previously
lived four pages deep. This page answers it first, in one screen, and leaves
every existing page untouched below it.

Architecture principle, inherited verbatim from step5_rookie_projections.py:
THIS PAGE READS, IT DOES NOT COMPUTE. Every number here traces to a file
already on disk - specifically the frozen Phase-6 output
data/projections/nets_rookies_2026.csv, the same file step5's own page and
step2_intro.py's roster table read. Nothing is refit, and no projection is
recomputed, so the three surfaces that now show these recipes cannot silently
disagree with each other.

It also deliberately REUSES step5's `load_labels()` and `archetype_bar_chart()`
rather than reimplementing them. That is not just DRY: `load_nba_labels()`
returns a dict-of-dicts, and any reimplementation that forgets to flatten it
renders a literal "{'label': ...}" into the UI - a bug this project already
hit once (see step5's load_labels docstring).

Naming note: "step6" is this file's position in the ARCHETYPE PORTAL's page
sequence, and has no relationship to the translator pipeline's own Phase 6
(deployment). Same two-numbering-schemes collision step5's docstring flags.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

import hull_callout_chart
from portal_shared import BL_INK, BL_MUTED, BL_LINE, BL_GREEN, BL_PAPER, BL_WHITE
import step5_rookie_projections as bridge

# AI-ASSISTED (Claude Code, chat) - Prompt: "I think it looks too complicated
# right now, I think we can show three rookie project receipt at the top. But
# in our model, i think it would be good if we also see their how rookie
# transfer?" - scope confirmed by the owner as a NEW landing page with every
# existing page kept as-is, and the per-player college->NBA translation as the
# "how it transfers" story.
# Used: this whole file, composed from step5's already-validated loaders and
# chart helper.
# Not AI: the decision to restructure around a landing page at all, the choice
# to keep every existing page untouched rather than rewrite them, and the
# choice of the per-player translation (over the naive-baseline or holdout
# stories, both offered and declined) - all the owner's own calls.
# AI-ASSISTED (Claude Code, chat) - Prompt: "我不是很想要Draft Class 2026
# 这个tab 把这个先隐藏掉" (I don't really want the Draft Class 2026 tab, hide it
# for now). Used: same "kept but not wired up" flip as every other SHOW_* flag
# in this app - render_draft_class_page() and everything under it remain fully
# built, tested (tests/test_draft_class_page.py) and untouched; a one-line flip
# back to True re-enables it.
# Not AI: the decision to hide it - the owner's own call.
SHOW_DRAFT_CLASS_PAGE = False

# A projection's top weight is the honest confidence signal here: the model
# outputs a mixture, and how concentrated that mixture is says how strongly it
# is committing. These two cuts are descriptive of the OUTPUT only - they make
# no claim about whether the college role and the NBA role are "the same
# role", because the college and NBA archetype spaces are different spaces
# with different labels. Asserting a semantic match between them by eye is
# precisely the naive-relabeling move this project's own T4 baseline tested
# and disproved, so this page shows both recipes and refuses to claim an
# equivalence the model was never asked to produce.
CONCENTRATED_MIN = 0.45
SPLIT_MIN = 0.30


def _confidence_read(top_weight: float) -> tuple[str, str, str]:
    """(headline word, colour, one-line plain-English gloss) from the single
    largest projected weight."""
    if top_weight >= CONCENTRATED_MIN:
        return ("Concentrated", BL_GREEN,
                "The model commits to one role - the strongest kind of projection it makes.")
    if top_weight >= SPLIT_MIN:
        return ("Split", "#b8860b",
                "Two roles share most of the weight - read both, not just the top one.")
    return ("Diffuse", BL_MUTED,
            "No role clears 30%. The model is saying 'genuinely hybrid, could go several ways' - "
            "that is information, not a failure.")


def _top_n(vec: np.ndarray, labels: dict, n: int = 3) -> list[tuple[str, float]]:
    order = np.argsort(-vec)[:n]
    return [(labels.get(int(j), f"archetype {j}"), float(vec[j])) for j in order]


def _recipe_vec(row: pd.Series, prefix: str, k: int = bridge.K) -> np.ndarray:
    return np.array([float(row[f"{prefix}{j}"]) for j in range(k)])


def _rookie_card(row: pd.Series, nba_labels: dict, picks: dict, archetypoids: dict):
    """One rookie, one card: photo, who he is, and the single bold line a coach
    can read and stop."""
    name = str(row["display_name"])
    y = _recipe_vec(row, "y_pred_")
    top = _top_n(y, nba_labels, 1)[0]
    word, colour, _gloss = _confidence_read(top[1])
    pick = picks.get(name)
    exemplar = archetypoids.get(int(np.argmax(y)), {}).get("player")

    photo = hull_callout_chart.get_headshot_data_uri(None, name)
    pick_txt = f"Pick #{pick}" if pick else "—"
    # Built outside the f-string below: the copy contains an apostrophe, and an
    # f-string expression may not contain a backslash on this Python (3.10).
    exemplar_html = (
        f'<div style="margin-top:6px;font-size:11.5px;color:{BL_MUTED};">'
        f"Closest role in today's NBA: <b>{exemplar}</b></div>"
    ) if exemplar else ""
    st.markdown(
        f'''<div style="background:{BL_WHITE};border:1px solid {BL_LINE};border-radius:10px;
             padding:14px 14px 12px 14px;height:100%;">
          <div style="display:flex;align-items:center;gap:10px;margin-bottom:10px;">
            <img src="{photo}" style="width:46px;height:46px;border-radius:50%;object-fit:cover;
                 border:2px solid {BL_WHITE};box-shadow:0 0 0 1px {BL_LINE};">
            <div style="line-height:1.25;">
              <div style="font-weight:700;font-size:15px;color:{BL_INK};">{name}</div>
              <div style="font-size:11.5px;color:{BL_MUTED};">{row['college_team']} · {pick_txt}</div>
            </div>
          </div>
          <div style="font-size:11px;text-transform:uppercase;letter-spacing:0.05em;
               color:{BL_MUTED};margin-bottom:2px;">Projected NBA role</div>
          <div style="font-size:19px;font-weight:700;color:{BL_INK};line-height:1.2;">
            {top[1]*100:.0f}% {top[0]}</div>
          <div style="margin-top:8px;font-size:11.5px;color:{colour};font-weight:600;">
            {word} projection</div>
          {exemplar_html}
        </div>''',
        unsafe_allow_html=True,
    )


def render_section_what_we_drafted(df: pd.DataFrame, nba_labels: dict, picks: dict,
                                   archetypoids: dict):
    st.markdown("### What we drafted")
    st.caption(
        "Three players with zero NBA minutes. Everything below is the NCAA→NBA translator's "
        "projection from their college statistics - a model output with real error bars, not a "
        "measurement, and a statement about **style, not talent**."
    )
    for col, (_, row) in zip(st.columns(len(df)), df.iterrows()):
        with col:
            _rookie_card(row, nba_labels, picks, archetypoids)


def render_section_how_they_translate(df: pd.DataFrame, college_labels: dict,
                                      nba_labels: dict):
    st.markdown("### How each rookie translates")
    st.caption(
        "Left: what he actually was in college, in the college archetype space. Right: what the "
        "translator projects him to be in the NBA archetype space. These are two different spaces "
        "with different archetypes - a player's college label and his NBA label are not supposed "
        "to match, and the gap between them is the entire reason this model exists rather than a "
        "lookup table."
    )

    for _, row in df.iterrows():
        name = str(row["display_name"])
        c = _recipe_vec(row, "c_alpha_")
        y = _recipe_vec(row, "y_pred_")
        c_top, y_top = _top_n(c, college_labels, 1)[0], _top_n(y, nba_labels, 1)[0]
        word, colour, gloss = _confidence_read(y_top[1])

        st.divider()
        st.markdown(f"#### {name}")
        st.markdown(
            f"<div style='font-size:13.5px;color:{BL_INK};margin:-4px 0 10px 0;'>"
            f"<b>{c_top[1]*100:.0f}% {c_top[0]}</b> at {row['college_team']} "
            f"<span style='color:{BL_MUTED};'>({row['conference']}, "
            f"{row['college_minutes']:.0f} min, {row['years_in_college']:.0f}y)</span> "
            f"&nbsp;→&nbsp; <b>{y_top[1]*100:.0f}% {y_top[0]}</b> projected in the NBA"
            f"</div>",
            unsafe_allow_html=True,
        )

        left, right = st.columns(2)
        with left:
            st.markdown(f"<div style='font-size:12px;color:{BL_MUTED};font-weight:600;'>"
                        f"COLLEGE (measured)</div>", unsafe_allow_html=True)
            st.plotly_chart(
                bridge.archetype_bar_chart(c, college_labels, height=300,
                                           xaxis_title="measured weight"),
                use_container_width=True, key=f"dc_college_{name}",
            )
        with right:
            st.markdown(f"<div style='font-size:12px;color:{BL_GREEN};font-weight:600;'>"
                        f"NBA (projected)</div>", unsafe_allow_html=True)
            st.plotly_chart(
                bridge.archetype_bar_chart(y, nba_labels, height=300),
                use_container_width=True, key=f"dc_nba_{name}",
            )

        st.markdown(
            f"<div style='background:{BL_PAPER};border-left:3px solid {colour};padding:8px 12px;"
            f"font-size:13px;color:{BL_INK};'><b>{word} projection.</b> {gloss}</div>",
            unsafe_allow_html=True,
        )


def render_section_how_to_read():
    st.divider()
    st.markdown("### How to read this")
    st.markdown(
        "- **Style, not talent.** An archetype says what kind of player someone is, never how "
        "good. Nothing on this page is a ranking.\n"
        "- **Confidence varies by player, and that is real signal.** A projection where one role "
        "holds most of the weight is a much stronger claim than one spread across several. Read "
        "the concentration before you read the label.\n"
        "- **The two charts are in different spaces.** The college archetypes and the NBA "
        "archetypes were fit separately, on different feature sets - college data carries no "
        "shot-location or play-type tracking. A rookie's college label is not supposed to equal "
        "his NBA label; the translator learned that mapping from 237 players who lived in both "
        "worlds.\n"
        "- **What the model cannot see:** defense beyond the box score, athleticism, motor, "
        "shot-making touch, or how a specific coaching staff will use him. Those remain the "
        "scout's job, and they are the right grounds to override anything here.\n"
    )
    st.caption(
        "Full derivation, the naive-baseline comparison this model had to beat, and the "
        "accuracy evidence on 36 players whose rookie seasons we already know: see the "
        "**NCAA Bridge** page."
    )


def render_draft_class_page():
    st.markdown("## Draft Class 2026")
    st.markdown(
        f"<div style='font-size:14px;color:{BL_MUTED};margin:-8px 0 14px 0;'>"
        f"What the Nets drafted, and what kind of player the model expects each to be."
        f"</div>",
        unsafe_allow_html=True,
    )

    # Same graceful-degradation contract every other page here follows: a
    # missing artifact must produce an actionable message, never a traceback.
    missing = [f for f in ("nets_rookies_2026.csv", "recipes.csv", "college_archetypes.md",
                           "archetype_labels.csv")
               if f in bridge.REQUIRED_FILES and not bridge.REQUIRED_FILES[f].exists()]
    if missing:
        st.warning(
            "The rookie projections aren't available in this checkout "
            f"(missing: {', '.join(missing)}). Run the Phase 6 deployment step "
            "(`src/translator/phase6_step2_predict_rookies.py`) to generate them."
        )
        return

    df = bridge.load_rookie_projections()
    college_labels, nba_labels = bridge.load_labels()
    if not college_labels or not nba_labels or len(nba_labels) != bridge.K:
        st.warning("Archetype labels failed to load - the projections can't be displayed by name.")
        return

    try:
        picks = bridge.load_rookie_pick_numbers()
    except Exception:
        picks = {}  # pick number is decoration, never a reason to fail the page
    try:
        archetypoids = bridge.load_archetypoids()
    except Exception:
        archetypoids = {}

    # Highest-confidence projection first: the card a coach should read first
    # is the one the model is most sure about, not whoever happens to be row 0.
    df = df.assign(
        _top=[_recipe_vec(r, "y_pred_").max() for _, r in df.iterrows()]
    ).sort_values("_top", ascending=False).reset_index(drop=True)

    render_section_what_we_drafted(df, nba_labels, picks, archetypoids)
    st.write("")
    render_section_how_they_translate(df, college_labels, nba_labels)
    render_section_how_to_read()
