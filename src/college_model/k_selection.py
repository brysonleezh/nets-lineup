"""
Phase 2 Step 2 - K selection via AA sweep, K=4..12, >=3 restarts each.

AA (not ADA) is the sweep method - a hard, measured constraint from
earlier this session, not a style preference: ADA on this dataset's scale
(~20k rows) was empirically found to take on the order of DAYS per fit
(a single K=8 fit on n=500 exceeded 5 minutes and was killed), while AA
fits the same shape in seconds. ADA is still used later (Step 3) at the
single confirmed K, not for the whole sweep.

Diagnostics, per the paper's Section 4.3 (both computed from the BEST
(lowest-RSS) restart at each K):
  - RSS elbow (raw, expected to look "sufficient" too early per the
    paper's own point).
  - Intra-archetype variance: hard-assign every fit-pool row to its
    argmax alpha, then for each archetype j, V_j = mean over the 12
    z-features of that archetype's own assigned group's per-feature
    variance. The reported curve is the UNWEIGHTED mean of V_j across
    the K archetypes at each K (per this phase's spec - NOT weighted by
    group size, a deliberate change from an earlier session's
    weighted-average version).
  - Archetypes with <30 assigned fit-pool players are flagged.

Run: python3 src/college_model/k_selection.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import archetypes as arch

from pool import SHARED_COLS, Z_COLS, build_fit_pool, load_canonical

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "college" / "model"

MAX_ITER = 400  # raised from an earlier session's 200 (which never fully
# converged on real data, though cross-seed results were near-identical) -
# a moderate increase given AA's real-data fit time here (measured
# ~100-340s/fit at max_iter=200, scaling with K) makes a much larger
# max_iter (e.g. 5x) cost multiple extra hours; this is a pragmatic
# engineering call, not re-litigated with the owner given a near-identical
# question was already raised once - convergence status is reported
# plainly either way, not hidden.
SEEDS = (0, 1, 2)
K_RANGE = range(4, 13)


def fit_aa(X_z: np.ndarray, k: int, seed: int) -> dict:
    t0 = time.time()
    model = arch.AA(n_archetypes=k, max_iter=MAX_ITER, tol=1e-4, random_state=seed)
    model.fit(X_z)
    elapsed = time.time() - t0
    n_iter = len(model.loss_)
    tss = float((X_z ** 2).sum())
    rss = float(model.rss_)
    return {
        "basis": np.asarray(model.archetypes_, dtype=float),
        "coefficients": np.asarray(model.coefficients_, dtype=float),
        "k": int(k), "seed": int(seed),
        "converged": bool(n_iter < MAX_ITER), "n_iter": int(n_iter),
        "rss": rss, "explained_var": 1.0 - rss / tss, "seconds": elapsed,
    }


def intra_archetype_variance_unweighted(X_z: np.ndarray, coefficients: np.ndarray, k: int) -> tuple[float, dict]:
    assignment = np.argmax(coefficients, axis=1)
    v_js = {}
    small_flags = {}
    for j in range(k):
        mask = assignment == j
        n_j = int(mask.sum())
        small_flags[j] = n_j
        if n_j < 2:
            v_js[j] = np.nan
            continue
        group = X_z[mask]
        per_feature_var = group.var(axis=0)  # variance within this archetype's own group, per feature
        v_js[j] = float(per_feature_var.mean())  # mean over the 12 features
    valid = [v for v in v_js.values() if not np.isnan(v)]
    unweighted_mean = float(np.mean(valid)) if valid else np.nan
    return unweighted_mean, {"v_j": v_js, "n_assigned": small_flags}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_canonical()
    pool = build_fit_pool(df, restrict=True)
    X_z = pool[Z_COLS].values
    print(f"Fit pool: {X_z.shape} (restrict_to_draft_conferences=True)", flush=True)

    raw_rows = []
    best_by_k = {}
    for k in K_RANGE:
        fits = []
        for seed in SEEDS:
            fit = fit_aa(X_z, k, seed)
            intra_var, detail = intra_archetype_variance_unweighted(X_z, fit["coefficients"], k)
            fit["intra_var_unweighted"] = intra_var
            fit["min_group_n"] = min(detail["n_assigned"].values())
            fits.append(fit)
            flag = "" if fit["converged"] else "  NOT CONVERGED"
            print(f"K={k} seed={seed}: EV={fit['explained_var']:.4f} intra_var={intra_var:.4f} "
                  f"min_group_n={fit['min_group_n']} ({fit['n_iter']} iters, {fit['seconds']:.1f}s){flag}", flush=True)
            raw_rows.append({
                "k": k, "seed": seed, "rss": fit["rss"], "explained_var": fit["explained_var"],
                "intra_var_unweighted": intra_var, "min_group_n": fit["min_group_n"],
                "n_iter": fit["n_iter"], "converged": fit["converged"], "seconds": fit["seconds"],
            })
        best = min(fits, key=lambda f: f["rss"])
        best_by_k[k] = best
        rss_spread = max(f["rss"] for f in fits) - min(f["rss"] for f in fits)
        print(f"  -> K={k} best seed={best['seed']}, RSS spread across restarts={rss_spread:.1f}", flush=True)

    raw = pd.DataFrame(raw_rows)
    raw.to_csv(OUT_DIR / "k_selection_raw.csv", index=False)

    summary_rows = []
    for k in K_RANGE:
        best = best_by_k[k]
        ks = raw[raw["k"] == k]
        summary_rows.append({
            "k": k, "rss": best["rss"], "explained_var": best["explained_var"],
            "intra_var_unweighted": best["intra_var_unweighted"], "min_group_n": best["min_group_n"],
            "rss_spread": ks["rss"].max() - ks["rss"].min(),
            "ev_spread": ks["explained_var"].max() - ks["explained_var"].min(),
            "all_converged": bool(ks["converged"].all()),
        })
    summary = pd.DataFrame(summary_rows)
    summary["ev_gain"] = summary["explained_var"].diff()
    summary["intra_var_change"] = summary["intra_var_unweighted"].diff()
    summary.to_csv(OUT_DIR / "k_selection_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))

    # persist best-per-K bases for later inspection (Step 3 needs the confirmed K's)
    np.savez(OUT_DIR / "k_sweep_bases.npz", **{
        f"k{k}_basis": best_by_k[k]["basis"] for k in K_RANGE
    })


if __name__ == "__main__":
    main()
