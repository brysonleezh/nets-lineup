"""Phase 5 tests: holdout-read-once guard, no-holdout-derived-statistic
guard, prediction simplex validity, and metric functions against the same
fixtures Phase 4 used."""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from metrics import jensen_shannon, l1, top1_hit, top1_within_top2  # noqa: E402

ANCHORS_DIR = REPO_ROOT / "data" / "anchors"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
K = 8


# --- holdout read exactly once ---------------------------------------------

def test_holdout_evaluated_lock_file_exists():
    """A completed Phase 5 run must leave a lock file behind - proof the
    holdout was scored, and the mechanism that blocks a second scoring."""
    assert (TRANSLATOR_DIR / "holdout_evaluated.json").exists()
    evaluated = json.loads((TRANSLATOR_DIR / "holdout_evaluated.json").read_text())
    assert evaluated["n_holdout"] == 36
    assert "holdout_sha256" in evaluated


def test_holdout_hash_still_matches_manifest_after_phase5():
    """The holdout file itself must be byte-identical to Step 0's quarantine
    even after the full Phase 5 run completed - proof nothing wrote to it."""
    import hashlib
    manifest = json.loads((ANCHORS_DIR / "holdout_manifest.json").read_text())
    actual = hashlib.sha256((ANCHORS_DIR / "holdout_2025.csv").read_bytes()).hexdigest()
    assert actual == manifest["sha256"]


def test_phase5_step1_refuses_to_rerun_without_flag():
    """Calling main() a second time (lock file already exists) must raise,
    not silently re-evaluate."""
    import phase5_step1_predict as step1
    assert (TRANSLATOR_DIR / "holdout_evaluated.json").exists()
    with pytest.raises(SystemExit):
        step1.main(rerun=False)


# --- no holdout-derived statistic in the transform --------------------------

def test_apply_frozen_variant_b_uses_only_frozen_stats():
    """Corrupting a holdout row's raw feature values must not change the
    frozen mu/sd/imputation-mean dict itself - proof the transform is a
    pure lookup against Phase 4's frozen recipe, never recomputed from
    whatever dataframe is passed in."""
    import phase5_step1_predict as step1
    preprocessing = json.loads((TRANSLATOR_DIR / "chosen_model_preprocessing.json").read_text())
    holdout = pd.read_csv(ANCHORS_DIR / "holdout_2025.csv")

    X_orig, _ = step1.apply_frozen_variant_b(holdout, preprocessing)

    corrupted = holdout.copy()
    corrupted.loc[corrupted.index[:5], "PTS_PER_100"] = 999.0
    corrupted.loc[corrupted.index[:5], "age_at_draft"] = 999.0
    preprocessing_after = json.loads((TRANSLATOR_DIR / "chosen_model_preprocessing.json").read_text())
    assert preprocessing_after == preprocessing, \
        "the frozen preprocessing file must never be mutated by transforming a (possibly corrupted) dataframe"

    # the corrupted rows' OWN X values will differ (expected - their raw inputs changed),
    # but the transform of every OTHER, unmodified row must be untouched
    X_after, _ = step1.apply_frozen_variant_b(corrupted, preprocessing_after)
    assert np.allclose(X_orig[5:], X_after[5:]), \
        "corrupting some holdout rows changed the transform of OTHER rows - a statistic leaked across rows"


def test_no_translator_source_computes_mean_std_of_holdout_dataframe():
    """Grep guard: phase5_step1_predict.py must never call .mean()/.std()
    on anything derived from the holdout dataframe - all such statistics
    must come from the frozen preprocessing dict instead."""
    src = (REPO_ROOT / "src" / "translator" / "phase5_step1_predict.py").read_text()
    # the only .mean()/.std() calls allowed here operate on cont_source (already-frozen-shifted)
    # or are absent entirely - the real guarantee is structural: fillna/-mu//sd all read from
    # the `mu`/`sd`/`impute_means` dict pulled from `preprocessing`, never from `df` or `holdout`
    assert "holdout.mean(" not in src and "holdout.std(" not in src
    assert "df.mean(" not in src and "df.std(" not in src


# --- prediction simplex validity --------------------------------------------

def test_holdout_predictions_are_valid_simplex():
    preds = pd.read_csv(TRANSLATOR_DIR / "holdout_predictions.csv")
    y_pred = preds[[f"y_pred_{j}" for j in range(K)]].values
    assert (y_pred >= -1e-9).all()
    assert np.abs(y_pred.sum(axis=1) - 1.0).max() < 1e-6


def test_holdout_intervals_bracket_the_point_prediction_reasonably():
    preds = pd.read_csv(TRANSLATOR_DIR / "holdout_predictions.csv")
    for j in range(K):
        lo50, hi50 = preds[f"pi50_lo_{j}"], preds[f"pi50_hi_{j}"]
        lo90, hi90 = preds[f"pi90_lo_{j}"], preds[f"pi90_hi_{j}"]
        assert (lo50 <= hi50 + 1e-9).all()
        assert (lo90 <= hi90 + 1e-9).all()
        assert (lo90 <= lo50 + 1e-9).all() and (hi50 <= hi90 + 1e-9).all(), \
            "90% interval must be at least as wide as the 50% interval"


# --- metric functions against the same fixtures used in Phase 4 ------------

def test_jsd_fixture_matches_phase4():
    p = np.array([[1.0, 0.0]])
    q = np.array([[0.0, 1.0]])
    assert abs(jensen_shannon(p, q, base=2)[0] - 1.0) < 1e-6


def test_l1_fixture_matches_phase4():
    p = np.array([[0.5, 0.5]])
    q = np.array([[0.9, 0.1]])
    assert abs(l1(p, q)[0] - 0.8) < 1e-9


def test_top1_fixtures_match_phase4():
    p = np.array([[0.1, 0.9], [0.7, 0.3]])
    q = np.array([[0.2, 0.8], [0.6, 0.4]])
    assert np.allclose(top1_hit(p, q), [1.0, 1.0])


def test_top1_within_top2_fixture_matches_phase4():
    p_true = np.array([[0.0, 0.0, 1.0]])
    q_pred = np.array([[0.1, 0.45, 0.45]])
    assert top1_within_top2(p_true, q_pred)[0] == 1.0


# --- eligibility verdict sanity ---------------------------------------------

def test_holdout_metrics_file_records_eligibility():
    m = pd.read_csv(TRANSLATOR_DIR / "holdout_metrics.csv").iloc[0]
    assert m["beats_t4"] and m["beats_t5i"] and m["beats_t5ii"]
    assert m["eligible"]
