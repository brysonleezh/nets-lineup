"""
Phase 2 Step 3 (Amendment B) - "restricted archetypoid search": the Step 3
ADA deliverable, replacing both the from-scratch `archetypes` package ADA
(confirmed infeasible at this dataset's scale earlier this session - a
K=8 fit on n=500 already exceeded 5 minutes) and this file's own earlier,
now-superseded 5-sweep/400-candidate version.

Method, exactly as specified:
  (i)  candns init: snap each of the 8 FROZEN AA archetypes
       (data/college/model/k8_frozen_basis.npz) to its nearest real
       player-season in z-space. Re-solve all alphas ONCE (one full-pool
       projection), record RSS.
  (ii) One single greedy swap-improvement pass: for each archetype (in
       order 0..7), candidates = its 100 nearest real player-seasons to
       the FROZEN AA archetype's own (fixed) position - not the moving
       post-swap position, so the neighborhood is well-defined and
       doesn't shift under the pass. Full-pool projection RSS is cheap
       (0.11s, measured), so each of the 100 candidates is evaluated
       directly rather than through a cheaper proxy; the best-improving
       candidate is accepted per slot (or none, if no candidate improves)
       before moving to the next slot - this is the "greedy" part: a
       local per-slot choice, not a joint re-optimization, and exactly
       one pass (not iterated to convergence).

Run: python3 src/college_model/restricted_ada.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pool import Z_COLS, build_fit_pool, load_canonical
from project import project_recipe

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = REPO_ROOT / "data" / "college" / "model"
K = 8
N_CANDIDATES = 100


def compute_rss(X_z: np.ndarray, basis_points: np.ndarray) -> float:
    P = project_recipe(X_z, basis_points)
    recon = P @ basis_points
    return float(((X_z - recon) ** 2).sum())


def main() -> None:
    df = load_canonical()
    pool = build_fit_pool(df, restrict=True)
    X_z = pool[Z_COLS].values
    print(f"Fit pool: {X_z.shape}", flush=True)

    frozen = np.load(OUT_DIR / "k8_frozen_basis.npz")
    aa_basis = frozen["basis"]
    aa_rss = compute_rss(X_z, aa_basis)  # continuous AA basis (not a real-point set) - reference only
    print(f"AA (continuous, frozen consensus-basin restart) RSS: {aa_rss:.2f}", flush=True)

    # (i) candns init: nearest real point per archetype, ensuring distinctness
    current_idx = []
    for j in range(K):
        dists = np.linalg.norm(X_z - aa_basis[j], axis=1)
        order = np.argsort(dists)
        for idx in order:
            if idx not in current_idx:
                current_idx.append(int(idx))
                break
    current_idx = np.array(current_idx, dtype=int)
    candns_rss = compute_rss(X_z, X_z[current_idx])
    print(f"candns (nearest-real-point snap) RSS: {candns_rss:.2f} "
          f"(vs continuous AA: {candns_rss - aa_rss:+.2f}, {100*(candns_rss-aa_rss)/aa_rss:+.2f}%)", flush=True)
    print(f"candns row indices: {current_idx.tolist()}", flush=True)

    # (ii) one greedy swap pass, 100-nearest-to-the-FROZEN-AA-position candidates per slot
    current_rss = candns_rss
    swap_log = []
    for j in range(K):
        dists = np.linalg.norm(X_z - aa_basis[j], axis=1)
        nearest_100 = np.argsort(dists)[:N_CANDIDATES]
        candidates = [c for c in nearest_100 if c not in current_idx]

        best_candidate, best_rss = None, current_rss
        for cand in candidates:
            trial_idx = current_idx.copy()
            trial_idx[j] = cand
            trial_rss = compute_rss(X_z, X_z[trial_idx])
            if trial_rss < best_rss:
                best_rss, best_candidate = trial_rss, cand

        if best_candidate is not None:
            improvement = current_rss - best_rss
            current_idx[j] = best_candidate
            current_rss = best_rss
            print(f"slot {j}: swapped in row {best_candidate}, RSS {current_rss:.2f} "
                  f"(-{improvement:.2f}), {len(candidates)} candidates tried", flush=True)
            swap_log.append({"slot": j, "accepted": True, "new_row": int(best_candidate),
                              "rss_after": current_rss, "improvement": improvement})
        else:
            print(f"slot {j}: no improving swap among {len(candidates)} candidates", flush=True)
            swap_log.append({"slot": j, "accepted": False, "new_row": None,
                              "rss_after": current_rss, "improvement": 0.0})

    final_rss = current_rss
    print(f"\nRSS trajectory: AA(continuous)={aa_rss:.2f} -> candns={candns_rss:.2f} -> "
          f"post-swap={final_rss:.2f}", flush=True)
    print(f"Total swap improvement: {candns_rss - final_rss:.2f} "
          f"({100*(candns_rss-final_rss)/candns_rss:+.2f}% vs candns)")
    print(f"Final gap vs continuous AA: {final_rss - aa_rss:+.2f} "
          f"({100*(final_rss-aa_rss)/aa_rss:+.2f}%)")

    identity = pool.iloc[current_idx][
        ["player_id_source", "player_name_raw", "season", "team", "conference", "minutes"]
    ].reset_index(drop=True)
    identity["archetype"] = range(K)
    identity.to_csv(OUT_DIR / "ada_archetypoid_identity.csv", index=False)

    np.savez(OUT_DIR / "ada_k8_basis.npz", basis=X_z[current_idx], row_idx=current_idx)
    pd.DataFrame(swap_log).to_csv(OUT_DIR / "ada_swap_log.csv", index=False)
    pd.DataFrame([
        {"stage": "AA (continuous)", "rss": aa_rss},
        {"stage": "candns", "rss": candns_rss},
        {"stage": "post-swap", "rss": final_rss},
    ]).to_csv(OUT_DIR / "ada_rss_trajectory.csv", index=False)

    print("\nArchetypoid identities:")
    print(identity.to_string(index=False))


if __name__ == "__main__":
    main()
