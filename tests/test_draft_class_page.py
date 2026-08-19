"""Tests for the Draft Class 2026 landing page (src/step6_draft_class.py).

Covers the three things most likely to break silently: that the page is
actually reachable as the portal's DEFAULT page, that every number it shows
traces to the frozen projections file rather than being recomputed, and that a
missing artifact degrades to a message instead of a traceback.
"""

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))


# --- AppTest: reachable, and first -------------------------------------------

def test_draft_class_visibility_matches_its_flag():
    """The page is currently HIDDEN (SHOW_DRAFT_CLASS_PAGE = False, the owner's
    call). This test tracks the flag rather than hardcoding either state, so it
    stays correct whichever way the flag is set - and still proves the nav
    wiring works if it is ever flipped back on."""
    from streamlit.testing.v1 import AppTest
    import step6_draft_class as dc
    at = AppTest.from_file(str(REPO_ROOT / "src" / "portal.py"))
    at.run(timeout=90)
    assert not at.exception
    opts = at.radio[0].options
    if dc.SHOW_DRAFT_CLASS_PAGE:
        assert opts[0] == "🏆 Draft Class 2026", "when shown, it must be the landing page"
        assert at.radio[0].value == "🏆 Draft Class 2026"
    else:
        assert "🏆 Draft Class 2026" not in opts


def test_every_pre_existing_page_still_present():
    """The restructure was explicitly scoped as 'add a landing page, keep all
    existing pages'. This fails if a page was dropped rather than shifted."""
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(REPO_ROOT / "src" / "portal.py"))
    at.run(timeout=90)
    opts = at.radio[0].options
    for expected in ("📖 The 8 Player Types", "🔍 Player Breakdown",
                     "🌉 NCAA Bridge", "📄 Report"):
        assert expected in opts, f"{expected} disappeared from the nav"


# --- the numbers must come from the frozen file ------------------------------

def test_headline_projection_matches_frozen_file_exactly():
    """The page must READ data/projections/nets_rookies_2026.csv, never
    recompute. If these drift apart, the landing page and the NCAA Bridge page
    would show different projections for the same player."""
    import step6_draft_class as dc
    import step5_rookie_projections as bridge

    df = pd.read_csv(REPO_ROOT / "data" / "projections" / "nets_rookies_2026.csv")
    _college, nba_labels = bridge.load_labels()

    for _, row in df.iterrows():
        y = dc._recipe_vec(row, "y_pred_")
        top_label, top_w = dc._top_n(y, nba_labels, 1)[0]
        j = int(np.argmax(y))
        assert top_label == nba_labels[j]
        assert top_w == pytest.approx(float(row[f"y_pred_{j}"]))
        # and it must actually be a simplex, or it is not a recipe
        assert y.sum() == pytest.approx(1.0, abs=1e-4)
        assert (y >= 0).all()


def test_college_recipe_is_a_simplex_too():
    import step6_draft_class as dc
    df = pd.read_csv(REPO_ROOT / "data" / "projections" / "nets_rookies_2026.csv")
    for _, row in df.iterrows():
        c = dc._recipe_vec(row, "c_alpha_")
        assert c.sum() == pytest.approx(1.0, abs=1e-4)
        assert (c >= 0).all()


# --- the confidence read is a pure function of the top weight ----------------

@pytest.mark.parametrize("w,expected", [
    (0.90, "Concentrated"), (0.45, "Concentrated"),
    (0.44, "Split"), (0.30, "Split"),
    (0.29, "Diffuse"), (0.10, "Diffuse"),
])
def test_confidence_read_boundaries(w, expected):
    import step6_draft_class as dc
    assert dc._confidence_read(w)[0] == expected


def test_confidence_read_matches_real_rookies():
    """Guards the actual claim the page makes today: Brown is the confident
    one, Jefferson is the uncertain one."""
    import step6_draft_class as dc
    df = pd.read_csv(REPO_ROOT / "data" / "projections" / "nets_rookies_2026.csv").set_index("display_name")
    got = {n: dc._confidence_read(dc._recipe_vec(r, "y_pred_").max())[0]
           for n, r in df.iterrows()}
    assert got["Mikel Brown Jr."] == "Concentrated"
    assert got["Joshua Jefferson"] == "Diffuse"


# --- graceful degradation ----------------------------------------------------

def test_missing_projections_file_warns_instead_of_raising():
    import step6_draft_class as dc
    fake = REPO_ROOT / "data" / "projections" / "__does_not_exist__.csv"
    with patch.dict(dc.bridge.REQUIRED_FILES, {"nets_rookies_2026.csv": fake}):
        warned = []
        with patch.object(dc.st, "warning", lambda t, **k: warned.append(str(t))), \
             patch.object(dc.st, "markdown", lambda *a, **k: None):
            dc.render_draft_class_page()
        assert warned, "a missing artifact must produce a message, not a traceback"
        assert "nets_rookies_2026.csv" in warned[0]


# --- read-only guard ---------------------------------------------------------

def test_page_has_no_write_path():
    """This page reads frozen artifacts; it must never write one."""
    src = (REPO_ROOT / "src" / "step6_draft_class.py").read_text()
    for forbidden in ("to_csv(", "to_parquet(", "np.save", "savez", "open(", "write_text("):
        assert forbidden not in src, f"unexpected write path: {forbidden}"
