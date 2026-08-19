"""
Phase 2 CHECKPOINT 1 Amendment A - K=8 basin analysis, replacing the naive
"lowest-RSS-restart" rule.

Owner's finding: the original K=8 sweep's best-RSS restart (seed 0,
intra_var=0.602) actually disagreed with the other two restarts (seeds
1/2, intra_var~0.564, close to each other) - i.e. the "best RSS" pick was
an outlier basin, not the consensus solution. Picking it as THE K=8 basis
would have been a real methodological error, not a stylistic call.

Fix: refit AA at K=8 only with MORE restarts (seeds 0..7, max_iter=1000 -
both increases are deliberate: more seeds to actually characterize the
basin structure, higher max_iter since none of the original 200/400-
iteration fits fully converged), then cluster restarts into basins via
Hungarian-matched cosine similarity of archetype z-profiles (same basin
iff min matched cosine >= 0.95 - ALL 8 archetypes must align closely, not
just some). The frozen basis is the lowest-RSS restart WITHIN the
consensus basin (the basin most seeds fall into), not the global lowest-
RSS restart regardless of basin.

Run: python3 src/college_model/refit_k8_basins.py
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
SEEDS = range(8)
MAX_ITER = 1000


def fit_aa(X_z: np.ndarray, seed: int) -> dict:
    t0 = time.time()
    model = arch.AA(n_archetypes=K, max_iter=MAX_ITER, tol=1e-4, random_state=seed)
    model.fit(X_z)
    elapsed = time.time() - t0
    n_iter = len(model.loss_)
    tss = float((X_z ** 2).sum())
    rss = float(model.rss_)
    basis = np.asarray(model.archetypes_, dtype=float)
    coef = np.asarray(model.coefficients_, dtype=float)

    assignment = np.argmax(coef, axis=1)
    v_js = []
    min_group_n = X_z.shape[0]
    for j in range(K):
        mask = assignment == j
        n_j = int(mask.sum())
        min_group_n = min(min_group_n, n_j)
        if n_j >= 2:
            v_js.append(float(X_z[mask].var(axis=0).mean()))
    intra_var = float(np.mean(v_js)) if v_js else np.nan

    return {
        "seed": seed, "basis": basis, "coefficients": coef,
        "converged": bool(n_iter < MAX_ITER), "n_iter": int(n_iter),
        "rss": rss, "explained_var": 1.0 - rss / tss,
        "intra_var": intra_var, "min_group_n": min_group_n, "seconds": elapsed,
    }


def cosine_sim_matrix(basis_a: np.ndarray, basis_b: np.ndarray) -> np.ndarray:
    a_norm = basis_a / np.linalg.norm(basis_a, axis=1, keepdims=True)
    b_norm = basis_b / np.linalg.norm(basis_b, axis=1, keepdims=True)
    return a_norm @ b_norm.T


def min_matched_cosine(basis_a: np.ndarray, basis_b: np.ndarray) -> float:
    """Hungarian-match archetypes between two restarts (maximize total
    cosine similarity), return the MINIMUM matched-pair cosine - the
    weakest link in the alignment, per the owner's >=0.95 threshold."""
    sim = cosine_sim_matrix(basis_a, basis_b)
    row_idx, col_idx = linear_sum_assignment(-sim)  # maximize sim = minimize -sim
    return float(sim[row_idx, col_idx].min())


def cluster_into_basins(fits: list[dict], threshold: float = 0.95) -> list[list[int]]:
    """Greedy clustering: each fit joins the first existing basin whose
    representative (first member) it matches >=threshold against, else
    starts a new basin."""
    basins: list[list[int]] = []
    reps: list[np.ndarray] = []
    for i, fit in enumerate(fits):
        placed = False
        for b_idx, rep in enumerate(reps):
            if min_matched_cosine(fit["basis"], rep) >= threshold:
                basins[b_idx].append(i)
                placed = True
                break
        if not placed:
            basins.append([i])
            reps.append(fit["basis"])
    return basins


def main() -> None:
    df = load_canonical()
    pool = build_fit_pool(df, restrict=True)
    X_z = pool[Z_COLS].values
    print(f"Fit pool: {X_z.shape}", flush=True)

    fits = []
    for seed in SEEDS:
        fit = fit_aa(X_z, seed)
        fits.append(fit)
        flag = "" if fit["converged"] else "  NOT CONVERGED"
        print(f"seed={seed}: RSS={fit['rss']:.2f} EV={fit['explained_var']:.4f} "
              f"intra_var={fit['intra_var']:.4f} min_group_n={fit['min_group_n']} "
              f"({fit['n_iter']} iters, {fit['seconds']:.1f}s){flag}", flush=True)

    basins = cluster_into_basins(fits, threshold=0.95)
    print(f"\n{len(basins)} basin(s) found among {len(fits)} restarts:", flush=True)

    basin_rows = []
    for b_idx, members in enumerate(basins):
        rss_vals = [fits[i]["rss"] for i in members]
        iv_vals = [fits[i]["intra_var"] for i in members]
        mgn_vals = [fits[i]["min_group_n"] for i in members]
        n_conv = sum(fits[i]["converged"] for i in members)
        basin_rows.append({
            "basin": b_idx, "seeds": [fits[i]["seed"] for i in members], "n_seeds": len(members),
            "rss_min": min(rss_vals), "rss_max": max(rss_vals),
            "intra_var_min": min(iv_vals), "intra_var_max": max(iv_vals),
            "min_group_n_min": min(mgn_vals), "n_converged": n_conv,
        })
        print(f"  basin {b_idx}: seeds={[fits[i]['seed'] for i in members]}, "
              f"RSS=[{min(rss_vals):.1f},{max(rss_vals):.1f}], "
              f"intra_var=[{min(iv_vals):.4f},{max(iv_vals):.4f}], "
              f"min_group_n_min={min(mgn_vals)}, converged={n_conv}/{len(members)}", flush=True)

    basin_df = pd.DataFrame(basin_rows)
    basin_df.to_csv(OUT_DIR / "k8_basin_table.csv", index=False)

    # consensus basin = most seeds; tie-break lower mean intra-var
    max_size = basin_df["n_seeds"].max()
    tied = basin_df[basin_df["n_seeds"] == max_size]
    if len(tied) > 1:
        tied = tied.copy()
        tied["mean_intra_var"] = tied.apply(
            lambda r: np.mean([fits[i]["intra_var"] for i in basins[r["basin"]]]), axis=1)
        consensus_basin_idx = int(tied.sort_values("mean_intra_var").iloc[0]["basin"])
    else:
        consensus_basin_idx = int(tied.iloc[0]["basin"])

    consensus_members = basins[consensus_basin_idx]
    frozen_i = min(consensus_members, key=lambda i: fits[i]["rss"])
    frozen = fits[frozen_i]

    print(f"\nConsensus basin: {consensus_basin_idx} (seeds {[fits[i]['seed'] for i in consensus_members]})")
    print(f"Frozen restart: seed={frozen['seed']}, RSS={frozen['rss']:.2f}, "
          f"intra_var={frozen['intra_var']:.4f}, min_group_n={frozen['min_group_n']}")

    np.savez(OUT_DIR / "k8_frozen_basis.npz", basis=frozen["basis"], coefficients=frozen["coefficients"])
    (OUT_DIR / "k8_frozen_manifest.txt").write_text(
        f"K=8 frozen AA fit\n"
        f"seed={frozen['seed']}\nmax_iter={MAX_ITER}\nconverged={frozen['converged']}\nn_iter={frozen['n_iter']}\n"
        f"rss={frozen['rss']:.4f}\nexplained_var={frozen['explained_var']:.4f}\n"
        f"intra_var={frozen['intra_var']:.4f}\nmin_group_n={frozen['min_group_n']}\n"
        f"consensus_basin={consensus_basin_idx}\nbasin_seeds={[fits[i]['seed'] for i in consensus_members]}\n"
        f"selection_rule=lowest-RSS restart within consensus basin (tie-break: lower mean intra-var)\n"
        f"fit_pool_shape={X_z.shape}\nfeature_cols={Z_COLS}\n"
    )

    all_seeds_df = pd.DataFrame([{
        "seed": f["seed"], "rss": f["rss"], "explained_var": f["explained_var"],
        "intra_var": f["intra_var"], "min_group_n": f["min_group_n"],
        "converged": f["converged"], "n_iter": f["n_iter"],
    } for f in fits])
    all_seeds_df.to_csv(OUT_DIR / "k8_all_seeds.csv", index=False)
    print(f"\nWrote k8_basin_table.csv, k8_all_seeds.csv, k8_frozen_basis.npz, k8_frozen_manifest.txt")


if __name__ == "__main__":
    main()
