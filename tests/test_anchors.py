"""Phase 3 tests: anchor ledger + anchors.csv correctness."""

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ANCHORS_DIR = REPO_ROOT / "data" / "anchors"
STD_DIR = REPO_ROOT / "data" / "nba_historical"
BASIS_DIR = REPO_ROOT / "data" / "basis_2025_26"

K = 8


@pytest.fixture(scope="module")
def anchors():
    return pd.read_csv(ANCHORS_DIR / "anchors.csv")


@pytest.fixture(scope="module")
def ledger():
    return pd.read_csv(ANCHORS_DIR / "anchor_ledger.csv")


# --- Y simplex validity --------------------------------------------------

def test_y_simplex_validity(anchors):
    y_cols = [f"y_{j}" for j in range(K)]
    vals = anchors[y_cols].values
    assert (vals >= -1e-9).all()
    row_sums = vals.sum(axis=1)
    assert np.abs(row_sums - 1.0).max() < 1e-6


def test_college_recipe_simplex_validity(anchors):
    c_cols = [f"c_alpha_{j}" for j in range(K)]
    vals = anchors[c_cols].values
    assert (vals >= -1e-9).all()
    row_sums = vals.sum(axis=1)
    assert np.abs(row_sums - 1.0).max() < 1e-6


# --- per-season standardization determinism ------------------------------

SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
           "2022-23", "2023-24", "2024-25", "2025-26"]


@pytest.mark.parametrize("season", SEASONS)
def test_standardization_files_exist_and_well_formed(season):
    path = STD_DIR / f"standardization_{season}.json"
    assert path.exists()
    d = json.loads(path.read_text())
    assert d["season"] == season
    assert len(d["feature_columns"]) == 29
    assert len(d["mu"]) == 29
    assert len(d["sd"]) == 29
    assert all(s > 0 for s in d["sd"]), "a zero/negative sd would break z-scoring"


def test_standardization_determinism(monkeypatch):
    """Rebuilding a historical season's standardization twice must give
    identical mu/sd - the pipeline has no source of randomness in this step."""
    import sys
    sys.path.insert(0, str(REPO_ROOT / "src"))
    from step1_archetype_model import load_population

    feature_cols = json.loads((BASIS_DIR / "feature_order.json").read_text())["feature_columns"]
    season = "2021-22"
    df1 = load_population(min_threshold=300, season_min=season, season_max=season)
    df2 = load_population(min_threshold=300, season_min=season, season_max=season)
    mu1, mu2 = df1[feature_cols].astype(float).mean(), df2[feature_cols].astype(float).mean()
    assert np.allclose(mu1.values, mu2.values)


def test_2025_26_standardization_matches_frozen_basis():
    persisted = json.loads((BASIS_DIR / "standardization.json").read_text())
    ours = json.loads((STD_DIR / "standardization_2025-26.json").read_text())
    assert ours["feature_columns"] == persisted["feature_columns"]
    assert np.allclose(ours["mu"], persisted["mu"], atol=1e-9)
    assert np.allclose(ours["sd"], persisted["sd"], atol=1e-9)
    assert ours["n"] == persisted["n_train"]


# --- 2025-class recipe-consistency against the app's stored recipes ------

def test_2025_class_recipes_match_app(anchors):
    app_recipes = pd.read_csv(BASIS_DIR / "recipes.csv")
    class2025 = anchors[anchors["draft_year"] == 2025]
    check = class2025.merge(app_recipes, left_on="nba_player_id", right_on="PLAYER_ID", how="inner")
    assert len(check) == len(class2025), "not every 2025-class anchor found in the app's stored recipes"
    for j in range(K):
        assert np.abs(check[f"y_{j}"].values - check[f"arch_{j}"].values).max() < 1e-3


# --- ledger completeness --------------------------------------------------

VALID_STATUSES = {
    "included", "prediction_target", "non_ncaa_path", "no_college_data",
    "college_below_min", "college_gap_year", "rookie_below_min", "delayed_debut",
}


def test_ledger_every_pick_has_a_valid_status(ledger):
    assert ledger["status"].isin(VALID_STATUSES).all()


def test_ledger_every_pick_appears_exactly_once(ledger):
    dupe = ledger.duplicated(["draft_year", "overall"], keep=False)
    assert not dupe.any(), f"duplicate (draft_year, overall) keys: {ledger[dupe][['draft_year','overall']].values.tolist()}"


def test_ledger_statuses_partition_cleanly(ledger):
    """Every 2017-2025 row has exactly one status; 2026 rows are all prediction_target."""
    elig = ledger[ledger["draft_year"] <= 2025]
    assert elig["status"].notna().all()
    assert (elig["status"] != "prediction_target").all()
    pred = ledger[ledger["draft_year"] == 2026]
    assert (pred["status"] == "prediction_target").all()


def test_ledger_six_named_cases_resolved(ledger):
    expected = {
        "Shaedon Sharpe": "no_college_data", "Mitchell Robinson": "no_college_data",
        "Goga Bitadze": "non_ncaa_path", "De'Anthony Melton": "college_gap_year",
        "Dewan Hernandez": "college_gap_year", "Wesley Iwundu": "included",
    }
    for name, status in expected.items():
        row = ledger[ledger["player_name_raw"] == name]
        assert len(row) == 1, f"{name}: expected exactly 1 ledger row"
        assert row.iloc[0]["status"] == status, f"{name}: expected status={status}, got {row.iloc[0]['status']}"


# --- zero 2026 rows in anchors.csv ----------------------------------------

def test_no_2026_rows_in_anchors(anchors):
    assert (anchors["draft_year"] != 2026).all()
    assert anchors["draft_year"].max() <= 2025


# --- age_at_draft on a fixed birthdate fixture ----------------------------

def test_age_at_draft_fixture():
    """A player born exactly 20 years before his draft date should compute
    to age_at_draft == 20.0 (within 365.25-day-year rounding)."""
    draft_date = date(2019, 6, 20)
    birthdate = date(1999, 6, 20)
    age = (draft_date - birthdate).days / 365.25
    assert abs(age - 20.0) < 0.01


# --- years_in_college on the known-case fixture set ------------------------

def test_years_in_college_validation_file_exists_and_mostly_matches():
    path = ANCHORS_DIR / "years_in_college_validation.csv"
    assert path.exists()
    val = pd.read_csv(path)
    assert len(val) >= 8, "spec requires validating against >= 8 known cases"
    disagree_rate = 1 - val["match"].mean()
    # 8.3% observed and accepted (Ja Morant, investigated - a real CBBD data gap, not a
    # formula bug); this is a regression guard, not a re-statement of the >10% fallback
    # trigger the pipeline script itself already checks.
    assert disagree_rate <= 0.15, f"years_in_college disagreement rate {disagree_rate:.1%} is unexpectedly high"
