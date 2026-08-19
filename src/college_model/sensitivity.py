"""
Phase 2 Step 5 - stability & sensitivity (lightweight, per spec).

1. Restart stability at K=8: summarized from Amendment A's 8-seed basin
   analysis (already run - not redone here).
2. Conference-flag-OFF refit: AA at K=8 on the FULL 29,312-row pool (no
   draft-conference restriction), matched to the frozen ON basis via
   Hungarian cosine similarity, minimum matched similarity reported.

Run: python3 src/college_model/sensitivity.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import archetypes as arch
from scipy.optimize import linear_sum_assignment

from pool import Z_COLS, build_fit_pool, load_canonical

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "college" / "model"
K = 8


def fit_aa(X_z: np.ndarray, seed: int, max_iter: int = 1000) -> dict:
    t0 = time.time()
    model = arch.AA(n_archetypes=K, max_iter=max_iter, tol=1e-4, random_state=seed)
    model.fit(X_z)
    elapsed = time.time() - t0
    n_iter = len(model.loss_)
    tss = float((X_z ** 2).sum())
    return {
        "seed": seed, "basis": np.asarray(model.archetypes_, dtype=float),
        "rss": float(model.rss_), "explained_var": 1.0 - float(model.rss_) / tss,
        "converged": bool(n_iter < max_iter), "n_iter": int(n_iter), "seconds": elapsed,
    }


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a_n = a / np.linalg.norm(a, axis=1, keepdims=True)
    b_n = b / np.linalg.norm(b, axis=1, keepdims=True)
    return a_n @ b_n.T


def main() -> None:
    df = load_canonical()
    pool_off = build_fit_pool(df, restrict=False)
    X_off = pool_off[Z_COLS].values
    print(f"Pool OFF (unrestricted): {X_off.shape}", flush=True)

    fits = []
    for seed in (0, 1):
        fit = fit_aa(X_off, seed)
        fits.append(fit)
        print(f"OFF seed={seed}: RSS={fit['rss']:.2f} EV={fit['explained_var']:.4f} "
              f"({fit['n_iter']} iters, {fit['seconds']:.1f}s) converged={fit['converged']}", flush=True)

    best_off = min(fits, key=lambda f: f["rss"])
    frozen_on = np.load(OUT_DIR / "k8_frozen_basis.npz")["basis"]

    sim = cosine_sim_matrix(frozen_on, best_off["basis"])
    row_idx, col_idx = linear_sum_assignment(-sim)
    matched_sims = sim[row_idx, col_idx]
    min_matched = float(matched_sims.min())

    print(f"\nMatched cosine similarities (ON archetype -> OFF archetype): {np.round(matched_sims, 4).tolist()}")
    print(f"Minimum matched similarity: {min_matched:.4f}")

    pd.DataFrame({
        "on_archetype": row_idx, "off_archetype": col_idx, "cosine_sim": matched_sims,
    }).to_csv(OUT_DIR / "conference_flag_sensitivity.csv", index=False)

    verdict = ("Structure is materially STABLE under the conference restriction - "
               "removing it does not meaningfully change the archetype set."
               if min_matched >= 0.90 else
               "Structure shows MEANINGFUL sensitivity to the conference restriction - "
               "at least one archetype does not have a close match in the unrestricted fit.")
    print(f"\nVerdict: {verdict}")

    (OUT_DIR / "conference_flag_sensitivity_verdict.txt").write_text(
        f"min_matched_cosine={min_matched:.4f}\nverdict={verdict}\n"
        f"pool_off_shape={X_off.shape}\noff_seed_used={best_off['seed']}\noff_rss={best_off['rss']:.2f}\n"
    )


if __name__ == "__main__":
    main()
