"""
Phase 2 tests: recipe simplex validity, projection determinism, K-sweep
reproducibility (structural, not full-refit - re-running the 27-fit sweep
in CI is not practical; checks the persisted artifacts are internally
consistent instead).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from project import project_recipe

REPO_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = REPO_ROOT / "data" / "college" / "model"
RECIPES_PATH = REPO_ROOT / "data" / "college" / "recipes.csv"


def test_recipes_file_exists():
    assert RECIPES_PATH.exists()


def test_recipe_row_count_matches_canonical_table():
    recipes = pd.read_csv(RECIPES_PATH)
    canonical = pd.read_parquet(REPO_ROOT / "data" / "college" / "shared_features.parquet")
    assert len(recipes) == len(canonical)


def test_recipe_simplex_validity():
    recipes = pd.read_csv(RECIPES_PATH)
    alpha_cols = [f"alpha_{j}" for j in range(8)]  # excludes alpha_max, which also starts with "alpha_"
    assert all(c in recipes.columns for c in alpha_cols)
    alpha = recipes[alpha_cols].values
    assert (alpha >= -1e-9).all(), "found a negative archetype weight"
    row_sums = alpha.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6), "some rows don't sum to 1 within 1e-6"


def test_argmax_matches_alpha_max():
    recipes = pd.read_csv(RECIPES_PATH)
    alpha_cols = [f"alpha_{j}" for j in range(8)]
    alpha = recipes[alpha_cols].values
    assert np.array_equal(recipes["argmax_archetype"].values, alpha.argmax(axis=1))
    assert np.allclose(recipes["alpha_max"].values, alpha.max(axis=1))


def test_no_archetype_is_empty():
    recipes = pd.read_csv(RECIPES_PATH)
    counts = recipes["argmax_archetype"].value_counts()
    assert len(counts) == 8, "some archetype has zero player-seasons assigned as argmax"


def test_projection_deterministic_and_reproducible_from_frozen_basis():
    frozen = np.load(MODEL_DIR / "k8_frozen_basis.npz")
    basis = frozen["basis"]
    rng = np.random.default_rng(1)
    z = rng.normal(size=12)
    a1 = project_recipe(z, basis)
    a2 = project_recipe(z, basis)
    assert np.allclose(a1, a2), "projection is not deterministic for the same input"
    assert a1.sum() == pytest.approx(1.0, abs=1e-9)
    assert (a1 >= -1e-9).all()


def test_k_sweep_summary_internally_consistent():
    summary = pd.read_csv(MODEL_DIR / "k_selection_summary.csv")
    assert list(summary["k"]) == list(range(4, 13))
    # RSS must strictly decrease as K increases (more archetypes can only
    # reduce or match reconstruction error, never increase it)
    assert (summary["rss"].diff().dropna() < 0).all()
    # every K's min_group_n must clear the spec's own 30-player flag
    assert (summary["min_group_n"] >= 30).all()


def test_k8_basin_table_consistent_with_frozen_choice():
    basins = pd.read_csv(MODEL_DIR / "k8_basin_table.csv")
    manifest = (MODEL_DIR / "k8_frozen_manifest.txt").read_text()
    consensus_basin = int([l for l in manifest.splitlines() if l.startswith("consensus_basin")][0].split("=")[1])
    row = basins[basins["basin"] == consensus_basin].iloc[0]
    assert row["n_seeds"] == basins["n_seeds"].max(), "frozen basin is not the plurality basin"
