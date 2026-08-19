"""
Phase 2 Step 4 - compute recipes for the ENTIRE canonical college table
(all 29,312 complete-case player-seasons, all 10 seasons, regardless of
fit-pool membership - "fit pool != projection set" per Step 1's
architecture requirement) via project_recipe against the frozen K=8 AA
basis (data/college/model/k8_frozen_basis.npz, confirmed by the owner
after CHECKPOINT 2).

Run: python3 src/college_model/build_recipes.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pool import SHARED_COLS, Z_COLS, load_canonical
from project import project_recipe

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "college"
MODEL_DIR = OUT_DIR / "model"
K = 8


def main() -> None:
    df = load_canonical()
    print(f"Canonical table: {df.shape}", flush=True)

    frozen = np.load(MODEL_DIR / "k8_frozen_basis.npz")
    basis = frozen["basis"]

    Z = df[Z_COLS].values
    alpha = project_recipe(Z, basis)
    assert np.allclose(alpha.sum(axis=1), 1.0, atol=1e-6), "simplex violation"
    assert (alpha >= -1e-9).all(), "negative weight found"

    out = df[["player_id_source", "player_name_raw", "season", "team", "conference", "minutes"]].copy()
    for j in range(K):
        out[f"alpha_{j}"] = alpha[:, j]
    out["argmax_archetype"] = alpha.argmax(axis=1)
    out["alpha_max"] = alpha.max(axis=1)

    out.to_csv(OUT_DIR / "recipes.csv", index=False)
    print(f"Wrote {OUT_DIR / 'recipes.csv'} ({len(out)} rows)")
    print("\nargmax_archetype distribution:")
    print(out["argmax_archetype"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
