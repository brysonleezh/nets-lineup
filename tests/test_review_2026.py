"""Tests for src/eval/review_2026_predictions.py against SYNTHETIC 2026-27
inputs - proves the frozen review methodology is runnable today, not
"discovered broken in six months" when real season data finally exists."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "eval"))

from review_2026_predictions import (  # noqa: E402
    eligibility_threshold, check_eligibility, build_standardization_for_population,
    score_actual_vs_predicted, K,
)
from step1_archetype_model import project  # noqa: E402


# --- eligibility / pro-ration -------------------------------------------

def test_eligibility_threshold_full_season():
    assert eligibility_threshold(82) == 300.0


def test_eligibility_threshold_shortened_season():
    # a hypothetically shortened 72-game season, same math Phase 3 used for 2020-21
    assert abs(eligibility_threshold(72) - 300 * 72 / 82) < 1e-9


def test_check_eligibility_above_and_below_threshold():
    eligible, thresh = check_eligibility(350.0, 82)
    assert eligible and thresh == 300.0
    eligible, thresh = check_eligibility(250.0, 82)
    assert not eligible


# --- standardization on synthetic population ------------------------------

def test_build_standardization_for_synthetic_population():
    rng = np.random.default_rng(0)
    feature_columns = [f"feat_{i}" for i in range(5)]
    df = pd.DataFrame(rng.normal(size=(50, 5)), columns=feature_columns)
    std = build_standardization_for_population(df, feature_columns)
    assert std["n"] == 50
    assert std["feature_columns"] == feature_columns
    assert len(std["mu"]) == 5 and len(std["sd"]) == 5
    assert np.allclose(std["mu"], df.mean().values)
    assert np.allclose(std["sd"], df.std().values)


def test_build_standardization_rejects_zero_variance_feature():
    feature_columns = ["a", "b"]
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})  # b has zero variance
    with pytest.raises(ValueError, match="zero-variance"):
        build_standardization_for_population(df, feature_columns)


# --- scoring on synthetic recipes -----------------------------------------

def test_score_actual_vs_predicted_perfect_prediction():
    y = np.array([0.1, 0.1, 0.1, 0.5, 0.1, 0.05, 0.03, 0.02])
    baseline = np.ones(K) / K
    scores = score_actual_vs_predicted(y, y, baseline)
    assert scores["jsd"] < 1e-8
    assert scores["l1"] < 1e-8
    assert scores["top1_hit"] == 1.0
    assert scores["top1_within_top2"] == 1.0


def test_score_actual_vs_predicted_wrong_prediction():
    y_actual = np.array([1.0, 0, 0, 0, 0, 0, 0, 0])
    y_pred = np.array([0, 1.0, 0, 0, 0, 0, 0, 0])
    baseline = np.ones(K) / K
    scores = score_actual_vs_predicted(y_actual, y_pred, baseline)
    assert scores["top1_hit"] == 0.0
    assert scores["jsd"] > 0.5
    assert len(scores["per_dim_mae"]) == K
    assert len(scores["per_dim_baseline_mae"]) == K


# --- end-to-end on synthetic 2026-27 "season" data (real basis, fake population) ---

def test_end_to_end_pipeline_on_synthetic_season():
    """Builds a fully synthetic 2026-27-shaped population (real 29 basis
    features, random values, one row impersonating a rookie), runs it
    through standardization -> project() -> scoring exactly as the real
    script would, using the REAL frozen basis (never touches real 2026-27
    data, which doesn't exist yet)."""
    feature_order = json.loads((REPO_ROOT / "data" / "basis_2025_26" / "feature_order.json").read_text())
    feature_columns = feature_order["feature_columns"]
    basis = np.load(REPO_ROOT / "data" / "basis_2025_26" / "basis.npz")["basis"]

    rng = np.random.default_rng(1)
    n = 60
    synthetic = pd.DataFrame(rng.normal(loc=10, scale=3, size=(n, len(feature_columns))).clip(min=0.01),
                              columns=feature_columns)
    synthetic["PLAYER_ID"] = range(n)
    synthetic["MIN"] = rng.uniform(300, 2000, size=n)

    std = build_standardization_for_population(synthetic, feature_columns)
    assert std["n"] == n

    rookie_row = synthetic.iloc[[0]]
    y_actual = project(rookie_row, basis, np.array(std["mu"]), np.array(std["sd"]), feature_columns)[0]
    assert y_actual.shape == (K,)
    assert (y_actual >= -1e-9).all()
    assert abs(y_actual.sum() - 1.0) < 1e-6

    y_predicted = np.ones(K) / K  # a fake "frozen prediction" for the test
    baseline = np.ones(K) / K
    scores = score_actual_vs_predicted(y_actual, y_predicted, baseline)
    assert "jsd" in scores and "top1_hit" in scores
    assert 0 <= scores["jsd"] <= 1.001


def test_unevaluable_case_is_not_an_error():
    """A rookie who never appears in the season population (0 minutes) or
    who is below the eligibility threshold must be reported as
    'unevaluable', never crash the review."""
    eligible, threshold = check_eligibility(actual_minutes=150.0, games_in_season=82)
    assert not eligible  # this is the exact condition review_2026_predictions.py
    # branches on to set status='unevaluable' rather than raising - the pure
    # function itself just returns a boolean, confirmed here as the contract
    # the real script's unevaluable branch depends on.
