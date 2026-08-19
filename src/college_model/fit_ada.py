"""
Phase 2 Step 3 - ADA at the confirmed K (K=8), initialized from the AA
solution: nearest-real-data-point initialization, then swap-based local
improvement - per this phase's own spec, which explicitly offers this as
an alternative to "reuse the repo's existing ADA routine."

Not a stylistic choice: the `archetypes` package's own `arch.ADA` was
empirically found earlier this session to be computationally infeasible
at this dataset's scale (a K=8 fit on n=500 synthetic rows exceeded 5
minutes and was killed; extrapolated, a fit on this ~20.8k-row pool would
take on the order of days). Nearest-point-init + swap is a much cheaper
local search around an already-good AA solution, using the same
simplex-constrained NNLS projection (project.py) as its RSS-evaluation
primitive - confirmed fast (0.11s for a full-pool projection), which is
what makes many swap evaluations tractable where a from-scratch ADA
search was not.

Run: python3 src/college_model/fit_ada.py
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from pool import Z_COLS, build_fit_pool, load_canonical
from project import project_recipe

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "college" / "model"
CHOSEN_K = 8
MAX_SWEEPS = 5
CANDIDATE_CAP = 400  # per-slot candidate pool cap, for tractability


def compute_rss(X_z: np.ndarray, basis_points: np.ndarray) -> float:
    P = project_recipe(X_z, basis_points)
    recon = P @ basis_points
    return float(((X_z - recon) ** 2).sum())


def nearest_point_init(X_z: np.ndarray, aa_basis: np.ndarray) -> np.ndarray:
    """For each AA continuous archetype, find its nearest real row index.
    Ensures distinctness - if a collision occurs, the colliding slot takes
    its next-nearest available row instead."""
    k = aa_basis.shape[0]
    chosen: list[int] = []
    for j in range(k):
        dists = np.linalg.norm(X_z - aa_basis[j], axis=1)
        order = np.argsort(dists)
        for idx in order:
            if idx not in chosen:
                chosen.append(int(idx))
                break
    return np.array(chosen, dtype=int)


def swap_improve(X_z: np.ndarray, init_idx: np.ndarray, aa_coefficients: np.ndarray) -> dict:
    k = len(init_idx)
    current_idx = init_idx.copy()
    current_rss = compute_rss(X_z, X_z[current_idx])
    history = [{"sweep": 0, "slot": None, "rss": current_rss, "accepted": False}]
    print(f"init RSS (nearest-point): {current_rss:.2f}", flush=True)

    assignment = np.argmax(aa_coefficients, axis=1)

    for sweep in range(1, MAX_SWEEPS + 1):
        any_improved = False
        for j in range(k):
            candidates = np.where(assignment == j)[0]
            candidates = candidates[~np.isin(candidates, current_idx)]
            if len(candidates) > CANDIDATE_CAP:
                rng = np.random.default_rng(sweep * 100 + j)
                candidates = rng.choice(candidates, size=CANDIDATE_CAP, replace=False)

            best_candidate, best_rss = None, current_rss
            t0 = time.time()
            for cand in candidates:
                trial_idx = current_idx.copy()
                trial_idx[j] = cand
                trial_rss = compute_rss(X_z, X_z[trial_idx])
                if trial_rss < best_rss:
                    best_rss, best_candidate = trial_rss, cand
            dt = time.time() - t0

            if best_candidate is not None:
                current_idx[j] = best_candidate
                improvement = current_rss - best_rss
                current_rss = best_rss
                any_improved = True
                print(f"  sweep {sweep} slot {j}: swapped in row {best_candidate}, "
                      f"RSS {current_rss:.2f} (-{improvement:.2f}), {len(candidates)} candidates, {dt:.1f}s", flush=True)
                history.append({"sweep": sweep, "slot": j, "rss": current_rss, "accepted": True})
            else:
                print(f"  sweep {sweep} slot {j}: no improving swap among {len(candidates)} candidates ({dt:.1f}s)", flush=True)
                history.append({"sweep": sweep, "slot": j, "rss": current_rss, "accepted": False})

        print(f"-- sweep {sweep} done, RSS={current_rss:.2f} --", flush=True)
        if not any_improved:
            print("no improving swap found in this sweep - converged.", flush=True)
            break

    return {"final_idx": current_idx, "final_rss": current_rss, "history": history}


def main() -> None:
    df = load_canonical()
    pool = build_fit_pool(df, restrict=True)
    X_z = pool[Z_COLS].values
    print(f"Fit pool: {X_z.shape}", flush=True)

    sweep_data = np.load(OUT_DIR / "k_sweep_bases.npz")
    aa_basis = sweep_data[f"k{CHOSEN_K}_basis"]

    raw = pd.read_csv(OUT_DIR / "k_selection_raw.csv")
    best_row = raw[raw["k"] == CHOSEN_K].sort_values("rss").iloc[0]
    print(f"Using AA K={CHOSEN_K} best fit: seed={int(best_row['seed'])}, RSS={best_row['rss']:.2f}", flush=True)

    # re-fit that exact seed to get its coefficients (not persisted in the sweep, only the basis was)
    import archetypes as arch
    model = arch.AA(n_archetypes=CHOSEN_K, max_iter=400, tol=1e-4, random_state=int(best_row["seed"]))
    model.fit(X_z)
    aa_coefficients = np.asarray(model.coefficients_, dtype=float)
    aa_rss = float(model.rss_)
    print(f"AA re-fit RSS check: {aa_rss:.2f} (sweep recorded {best_row['rss']:.2f})", flush=True)

    init_idx = nearest_point_init(X_z, aa_basis)
    print(f"nearest-point init row indices: {init_idx.tolist()}", flush=True)

    result = swap_improve(X_z, init_idx, aa_coefficients)

    identity = pool.iloc[result["final_idx"]][
        ["player_id_source", "player_name_raw", "season", "team", "conference", "minutes"]
    ].reset_index(drop=True)
    identity["archetype"] = range(CHOSEN_K)
    identity.to_csv(OUT_DIR / "ada_archetypoid_identity.csv", index=False)

    np.savez(OUT_DIR / "ada_k8_basis.npz", basis=X_z[result["final_idx"]], row_idx=result["final_idx"])
    pd.DataFrame(result["history"]).to_csv(OUT_DIR / "ada_swap_history.csv", index=False)

    print(f"\nFINAL: AA RSS={aa_rss:.2f}, ADA RSS={result['final_rss']:.2f}, "
          f"gap={result['final_rss']-aa_rss:+.2f} ({100*(result['final_rss']-aa_rss)/aa_rss:+.2f}%)")
    print("\nArchetypoid identities:")
    print(identity.to_string(index=False))


if __name__ == "__main__":
    main()
