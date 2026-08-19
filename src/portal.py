"""
Portal - interactive Streamlit app entry point. Sidebar nav has an Intro
orientation page plus up to six more pages (two always shown, five gated
behind SHOW_* flags defined in the files that own them):

  0. Intro (step2_intro.render_intro_page) - plain-language "what is an
     archetype" explainer with the K=4..10 convex-hull scatter.
  1. Player Breakdown (step3_player_breakdown.render_diagnostic_analysis) -
     screening + the six-section per-player diagnostic card.
  2. Roster Construction (step3_player_breakdown, shown behind
     SHOW_ROSTER_CONSTRUCTION_PAGE) - team composition, archetype gaps/
     conflicts, and the validated (WLS, cross-validated R^2=0.312 vs.
     0.295 talent-only baseline) lineup-ranking model.
  3. Building Around Rookie (step3_player_breakdown, hidden behind
     SHOW_ROOKIE_SLOT_QUERY_PAGE) - rookie slot query for the 3 zero-NBA-
     data rookies. Unrelated to page 4 below despite both concerning the
     same 3 rookies - see SHOW_ROOKIE_SLOT_QUERY_PAGE's own definition.
  4. NCAA Bridge (step5_rookie_projections, shown behind
     SHOW_ROOKIE_PROJECTIONS_PAGE) - the completed NCAA->NBA rookie-
     translation pipeline's (Phases 1-6) own projections, validation, and
     pick-slot counterfactual for the same 3 rookies.
  5. Report (step4_report, shown behind SHOW_PLAYER_REPORT_PAGE) - per-
     player PDF scouting report preview/download.
  6. Future Work & Obstacles (step3_player_breakdown, hidden behind
     SHOW_FUTURE_WORK_PAGE) - the RAPM investigation writeup.

This file was trimmed down to a thin entry point during the Day-8 portal
restructuring (five-file split: portal_shared / step2_intro /
step3_player_breakdown / step4_report / this file), later joined by a
sixth (step5_rookie_projections). It only holds st.set_page_config +
global CSS, the once-per-run roster/data prep, the sidebar nav
construction (including the SHOW_* flags imported from wherever each
hidden page actually lives), and the dispatch that calls into the render
entry points above. Every actual computation and all UI/chart rendering
now lives in the imported files.

Run: streamlit run src/portal.py
"""

from __future__ import annotations

import streamlit as st

from step1_archetype_model import NETS_ROSTER_NCAA_BRIDGE
from portal_shared import (
    SEASON,
    BL_PAPER,
    BL_WHITE,
    BL_INK,
    BL_MUTED,
    BL_LINE,
    BL_GREEN,
    DEFAULT_DIAG_PLAYER_ID,
    load_static,
    load_player_bio,
    load_nets_roster,
    load_exposure_cache,
)
from step2_intro import render_intro_page
from step3_player_breakdown import (
    SHOW_ROSTER_CONSTRUCTION_PAGE,
    SHOW_ROOKIE_SLOT_QUERY_PAGE,
    SHOW_FUTURE_WORK_PAGE,
    render_diagnostic_analysis,
    render_roster_construction,
    render_rookie_slot_query_page,
    render_future_work,
)
from step4_report import SHOW_PLAYER_REPORT_PAGE, render_player_report_page
from step5_rookie_projections import SHOW_ROOKIE_PROJECTIONS_PAGE, render_rookie_projections_page
from step6_draft_class import SHOW_DRAFT_CLASS_PAGE, render_draft_class_page


st.set_page_config(page_title="Nets Archetype Portal", layout="wide")


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
    /* AI-ASSISTED (Claude Code, chat) - Prompt: "button background不要用黑色
    这个根本看不清" (don't make the button background black, it's completely
    illegible) - same root cause as the header/slider-track rules above:
    the browser's own light/dark toggle (separate from this app's own
    config.toml theme) can flip a plain st.button/st.download_button's
    background to Streamlit's dark-theme default (verified directly -
    rgb(19,23,32)) while this app's existing broad `stMarkdownContainer p`
    rule below still forces the button's OWN label text to the dark BL_INK
    ink color - dark text on a near-black background, illegible. Forcing
    the background (and border) here, the same way the header already is,
    fixes it: the existing BL_INK text rule then has the contrast it was
    always meant to have. Covers st.button AND st.download_button - both
    render as the same `data-testid="stBaseButton-secondary"` (confirmed
    directly from the real DOM, not assumed). */
    button[data-testid="stBaseButton-secondary"] {{
        background-color: {BL_WHITE} !important; border-color: {BL_LINE} !important;
    }}
    div[role="radiogroup"] label {{ padding: 6px 4px; }}
    hr {{ border-color: {BL_LINE}; }}
    body, .stApp {{ font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }}
    </style>
    """,
    unsafe_allow_html=True,
)
NETS_TEAM_ID = 1610612751
TEAM_LOGO_URL = "https://cdn.nba.com/logos/nba/{team_id}/global/L/logo.png"


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
    # AI-ASSISTED (Claude Code, chat) - Prompt: "how do we present to coach/
    # front office more friendly? I think it looks too complicated right now,
    # I think we can show three rookie project receipt at the top" - the nav
    # was ordered by how the model was BUILT (basis -> breakdown -> bridge ->
    # report), which buries the question a coach actually opens with. Draft
    # Class 2026 is inserted FIRST so it becomes the default landing page.
    # Used: one new entry prepended; every existing page kept exactly as-is
    # and simply shifted down, per the owner's explicit "keep all existing
    # pages" scope choice.
    # Not AI: the restructuring decision and its scope - the owner's own call.
    nav_options = []
    if SHOW_DRAFT_CLASS_PAGE:
        nav_options.append("🏆 Draft Class 2026")
    nav_options += ["📖 The 8 Player Types", "🔍 Player Breakdown"]
    if SHOW_ROSTER_CONSTRUCTION_PAGE:
        nav_options.append("🏀 Roster Construction")
    if SHOW_ROOKIE_SLOT_QUERY_PAGE:
        nav_options.append("🎯 Building Around Rookie")
    if SHOW_ROOKIE_PROJECTIONS_PAGE:
        nav_options.append("🌉 NCAA Bridge")
    if SHOW_PLAYER_REPORT_PAGE:
        nav_options.append("📄 Report")
    if SHOW_FUTURE_WORK_PAGE:
        nav_options.append("🚧 Future Work & Obstacles")
    page = st.radio("Navigate", nav_options, label_visibility="collapsed")
    st.divider()

    # AI-ASSISTED (Claude Code, chat) - Prompt: "在Player Breakdown 添加一个
    # list可以选择球员 就像 Report Tab那样" (add a player-picker list to Player
    # Breakdown, like the Report tab has). Diagnostic Analysis already has
    # TWO click-driven selection inputs (the quadrant chart, the screening
    # table's photo click - both write into
    # st.session_state["diag_selected_pid"], see render_diagnostic_analysis's
    # own note) - this dropdown is a THIRD input into that same slot, not a
    # replacement for either.
    # Used: a read-only replica of render_diagnostic_analysis's own chart/
    # table-click detection (same two session_state/query_param reads,
    # same priority order - table wins over chart) lives here so the
    # dropdown's OWN displayed value can be pre-synced to match a fresh
    # chart/table click before this widget renders. This has to happen
    # here, not inside render_diagnostic_analysis: a keyed Streamlit
    # widget's displayed value is fixed at the point in the script where
    # it's instantiated, and the sidebar always renders before
    # render_diagnostic_analysis is even called - syncing after the fact
    # would always lag one rerun behind a chart/table click. The resolved
    # pid is passed down as `sidebar_pid`, which render_diagnostic_analysis
    # applies as the final, authoritative override - correct whether it
    # reflects a synced chart/table click or the user's own fresh pick,
    # since by construction it's never stale on the render pass it's read.
    # Not AI: the requirement itself, and "like the Report tab" as the
    # explicit placement/style precedent - given directly.
    if page in ("🔍 Player Breakdown",):
        diag_names = sorted(roster["PLAYER_NAME"].tolist())
        pid_to_diag_name = dict(zip(roster["PLAYER_ID"].astype(int), roster["PLAYER_NAME"]))
        default_diag_match = roster.loc[roster["PLAYER_ID"] == DEFAULT_DIAG_PLAYER_ID, "PLAYER_NAME"]
        default_diag_name = default_diag_match.iloc[0] if len(default_diag_match) else diag_names[0]

        prior_quadrant = st.session_state.get("screening_quadrant")
        fresh_diag_pid = None
        if prior_quadrant and prior_quadrant.selection and prior_quadrant.selection.points:
            cd = prior_quadrant.selection.points[0].get("customdata")
            if cd:
                fresh_diag_pid = int(cd[0])
        query_diag_pid = st.query_params.get("diag_pid")
        if query_diag_pid is not None:
            fresh_diag_pid = int(query_diag_pid)  # table click wins over chart click - same priority as render_diagnostic_analysis's own resolution

        if fresh_diag_pid is not None and fresh_diag_pid in pid_to_diag_name:
            st.session_state["diag_player_select"] = pid_to_diag_name[fresh_diag_pid]
        elif "diag_player_select" not in st.session_state:
            st.session_state["diag_player_select"] = default_diag_name

        selected_diag_name = st.selectbox("Player", diag_names, key="diag_player_select")
        selected_diag_pid = int(roster.loc[roster["PLAYER_NAME"] == selected_diag_name, "PLAYER_ID"].iloc[0])
    elif page in ("🎯 Building Around Rookie",):
        selected_player = st.selectbox("Rookie", NETS_ROSTER_NCAA_BRIDGE)
    elif page in ("🏆 Draft Class 2026",):
        pass  # landing page takes no sidebar input - it shows all 3 rookies at once
    elif page in ("🌉 NCAA Bridge",):
        pass  # this page owns its own selectbox/slider widgets, rendered inline in its body
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

if page == "🏆 Draft Class 2026":
    render_draft_class_page()
elif page == "📖 The 8 Player Types":
    render_intro_page(roster, labels)
elif page == "🔍 Player Breakdown":
    render_diagnostic_analysis(recipes, k, labels, oncourt, bio, roster, sidebar_pid=selected_diag_pid)
elif page == "🏀 Roster Construction":
    render_roster_construction(roster, recipes, k, labels)
elif page == "🎯 Building Around Rookie":
    render_rookie_slot_query_page(selected_player, recipes, k, labels, oncourt)
elif page == "🌉 NCAA Bridge":
    render_rookie_projections_page()
elif page == "📄 Report":
    exposure_cache = load_exposure_cache(recipes, k, SEASON)
    render_player_report_page(selected_report_pid, recipes, k, labels, bio, exposure_cache)
elif page == "🚧 Future Work & Obstacles":
    render_future_work()
