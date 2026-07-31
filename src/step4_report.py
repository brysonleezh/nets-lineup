"""
Carved out of portal.py during the Day-8 portal restructuring. Everything
that serves ONLY the "Report" tab: collect_report_data() (assembles the PDF
diagnosis report's full data contract, reshaping - never recomputing -
values already produced by the same cached loaders step3_player_breakdown.py
uses), render_report_section() (the Streamlit preview/download UI), and the
thin render_player_report_page() wrapper portal.py's nav dispatch calls.
Imports player_report.py and report_svg_charts.py (both unchanged).
"""

from __future__ import annotations

import io
from datetime import date

import numpy as np
import pandas as pd
import pdf2image
import streamlit as st

from step1_archetype_model import load_basis as load_ada_basis
import hull_callout_chart
import player_report
import report_svg_charts as rsc
import step2b_player_diagnostics as diag2b
from portal_shared import (
    BASIS_DIR,
    SEASON,
    ELASTIC_SPREAD_THRESHOLD_PP,
    CATEGORY_ORDER,
    FEATURE_CATEGORY,
    FEATURE_DESCRIPTIONS,
    FEATURE_LABELS,
    PLAYTYPE_LABELS,
    _elasticity_verdict,
    load_drift_attribution_cached,
    load_full_features,
    load_individual_role_sensitivity_cached,
    load_league_elasticity_spreads,
    load_league_miscasting,
    load_league_purity_entropy,
    load_miscasting_cached,
    load_miscasting_grounding_cached,
    load_mismatch,
    load_peer_centroid,
    load_player_base_stats,
    load_recipes_all_seasons,
    load_role_drift_cached,
    load_signature_cached,
    load_teammate_lift,
    load_transition_drift_distribution_cached,
)



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


# AI-ASSISTED (Claude Code, chat) - Prompt: "as for C. Role drift across
# seasons, D. E. Please all include Read section as like A and B" - three
# new Read builders (C/D/E), same discipline as the two above: stitch
# together sentences from values the section's own chart/caption already
# computed, never a new number. C reuses the MOST RECENT transition's own
# already-built mover/context/severity strings (transitions_ctx[-1] in
# collect_report_data) rather than re-deriving a trend - "most recent" is
# the one most relevant to "how to use him now." D concatenates the 3
# existing per-panel captions (lift/minutes/elasticity) that already sit
# under D1-D3 into one paragraph, rather than inventing new cross-panel
# insight. E is simply the paragraph that already sat inline next to the
# dumbbell chart, unchanged, moved into a function so it can go through
# the same reads.* contract and the same "Read:" box styling as A-D
# (previously it was the one section without either).
def _build_role_drift_read(drift_ctx, insufficient_text):
    if drift_ctx is None:
        return insufficient_text
    last = drift_ctx["transitions"][-1]
    seasons = drift_ctx["seasons"]
    s_old, s_new = seasons[-2], seasons[-1]
    lead = f"Most recently ({s_old} → {s_new}), his role held stable" if last["severity"] == "stable" else \
           f"Most recently ({s_old} → {s_new}), his role saw a {last['severity']} shift"
    return f"{lead}: {last['mover']}{(' ' + last['context']) if last['context'] else ''}."


def _build_environment_read(lift_caption, minutes_caption, elasticity):
    return (
        f"{lift_caption} {minutes_caption} His usage shows {elasticity['label'].lower()} swings "
        f"({rsc.signed(elasticity['swing'])}pp) across lineup contexts ({elasticity['pct']:.0f}th pct league-wide)."
    )


def _build_miscast_read(underused, gap_pp, used_name, used_pct, produces_pct, support_text):
    body = (
        f"Used {used_pct:.0f}% as a {used_name}; produces only {produces_pct:.0f}% there. "
        f"Biggest untapped direction: {underused} {rsc.signed(gap_pp)}pp"
    )
    if support_text:
        body += f" — {support_text}"
    body += ". Stylistic-consistency diagnostic, not proof a role change would improve results."
    return body


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


# AI-ASSISTED (Claude Code, chat) - Prompt: "as for images I upload to you,
# i think how to use him format looks like not very clear to me, you can
# summary all A B C D E read section and summary again here", then "Bottom
# Line写的再简洁一点 文字过多" (write the Bottom Line more concisely, too much
# text) - replaces the Bottom Line panel's previous content wholesale: 3
# tactical "lever" cards built from D/E data only (deleted below, confirmed
# zero other callers via grep), which never touched A/B/C at all. A first
# version of the replacement concatenated the FULL reads.* sentences
# (styleProfile+neighbors, roleDrift+environment, miscast) - correct in
# substance but far too dense once rendered (one card ran 4 sentences).
# This version builds deliberately SHORTER, separate fragments straight
# from the same already-real values (never re-deriving new insight, never
# a re-parse of the longer reads.* strings, which would be fragile - many
# archetype abbreviations contain periods, e.g. "Off. Engine"/"Trad. PM",
# so splitting on "." to shorten an existing sentence isn't safe) - e.g.
# the C/D card drops the specific box-score mover feature and the D1/D3
# detail entirely, keeping just the archetype-direction + best/worst-fit
# lineup fact.
def _build_bottom_line(shot_mix, play_types, b_top_feat, b_top_dz, drift_ctx,
                        lift_caption, underused, gap_pp, used_name, used_pct, produces_pct):
    dominant_label, dominant_pct = max(shot_mix, key=lambda t: t[1])
    style_body = f"{dominant_pct:.0f}% {dominant_label}, {play_types[0][0]}-heavy. Peaks in {b_top_feat} ({rsc.signed(b_top_dz)} SD)."

    if drift_ctx is None:
        traj_lead = "Rookie season — no drift history yet."
    else:
        last = drift_ctx["transitions"][-1]
        trend = last["context"].split(",")[0] if last["context"] else last["mover"]
        traj_lead = f"{last['severity'].capitalize()} shift {trend}."
    traj_body = f"{traj_lead} {lift_caption}"

    opp_body = f"Used {used_pct:.0f}% as {used_name}; produces {produces_pct:.0f}%. Untapped: {underused} {rsc.signed(gap_pp)}pp."

    return [
        {"title": "Style & fit.", "body": style_body},
        {"title": "Trajectory & deployment.", "body": traj_body},
        {"title": "Opportunity.", "body": opp_body},
    ]


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
        "neighbors": f"His signature stands out most in {b_top_feat} ({rsc.signed(b_top_dz)} SD {'above' if b_top_dz > 0 else 'below'} typical).",
        "roleDrift": _build_role_drift_read(drift_ctx, drift_insufficient_text),
        "environment": _build_environment_read(
            environment_ctx["lift_caption"], environment_ctx["minutesVsTypical"]["caption"], environment_ctx["elasticity"]),
        "miscast": _build_miscast_read(
            underused, mc["gap_pp"], arch_names[used_top_idx],
            float(used[used_top_idx]) * 100, float(prod[used_top_idx]) * 100, support_text),
    }
    bottom_line = _build_bottom_line(
        shot_mix, play_types, b_top_feat, b_top_dz, drift_ctx, environment_ctx["lift_caption"],
        underused, mc["gap_pp"], arch_names[used_top_idx], float(used[used_top_idx]) * 100, float(prod[used_top_idx]) * 100)

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
        "bottomLine": bottom_line,
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
