"""
Phase 4 Step 3 - leave-one-draft-class-out CV over classes 2017-2024 (8
folds), all 12 pre-registered cells (3 variants x {T1,T2,T3} + T4 + T5(i) +
T5(ii)). Every data-dependent quantity is nested inside the training fold
(Step 0.2's leakage boundary) - T4/T5 have nothing to fit but are still run
through the identical fold protocol for comparability.

Run: .nets/bin/python src/translator/cv_harness.py
(needs numpyro/jax for T1 - .nets venv only, not the system python3)
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from input_variants import VARIANTS, build_targets  # noqa: E402
from models import (fit_t2_ridge, fit_t3_knn, hungarian_assignment, fit_t4_hungarian,  # noqa: E402
                     fit_t5_global_mean, fit_t5_height_tercile)
from dirichlet_model import fit_t1_dirichlet  # noqa: E402
from metrics import jensen_shannon, l1, top1_hit, top1_within_top2, per_dim_mae  # noqa: E402

K = 8
Y_COLS = [f"y_{j}" for j in range(K)]
TRAIN_PATH = REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv"
OUT_DIR = REPO_ROOT / "data" / "translator"
CLASSES = list(range(2017, 2025))

CELLS = (
    [(f"T1_{v}", "T1", v) for v in "abc"]
    + [(f"T2_{v}", "T2", v) for v in "abc"]
    + [(f"T3_{v}", "T3", v) for v in "abc"]
    + [("T4", "T4", None), ("T5i", "T5i", None), ("T5ii", "T5ii", None)]
)
assert len(CELLS) == 12


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(TRAIN_PATH)
    assert set(df["draft_year"].unique()) == set(CLASSES)

    all_rows = []  # per (player, cell) predictions + metrics
    fold_summaries = []  # per (fold, cell) aggregate, for spread reporting
    calibration_rows = []
    neighbor_previews = []
    t1_diag_rows = []

    assignment, cos_matrix, college_basis = hungarian_assignment()

    t0 = time.time()
    for fold_i, held_class in enumerate(CLASSES):
        train_df = df[df["draft_year"] != held_class].reset_index(drop=True)
        test_df = df[df["draft_year"] == held_class].reset_index(drop=True)
        y_true = test_df[Y_COLS].values
        print(f"\n=== fold {fold_i + 1}/8: held-out class {held_class} "
              f"(train n={len(train_df)}, test n={len(test_df)}) ===", flush=True)

        for cell_name, model, variant in CELLS:
            t_cell = time.time()
            neighbor_return_idx = [0, 1] if (model == "T3" and fold_i < 3) else None

            if model == "T1":
                variant_fn = VARIANTS[variant]
                X_train, X_test, _ = variant_fn(train_df, test_df)
                y_train, _ = build_targets(train_df, test_df)
                preds, diag = fit_t1_dirichlet(X_train, y_train, X_test, seed=fold_i)
                t1_diag_rows.append({"fold": held_class, "variant": variant,
                                      "target_accept": diag["target_accept"],
                                      "max_rhat": diag["max_rhat"], "min_ess": diag["min_ess"],
                                      "n_divergent": diag["n_divergent"],
                                      "num_samples_total": diag["num_samples_total"]})
                # calibration: coverage of 50%/90% posterior predictive intervals, per dim
                y_pred_samples = diag["y_pred_samples"]  # (S, n_test, K)
                for pct, lo_q, hi_q in [(50, 25, 75), (90, 5, 95)]:
                    lo = np.percentile(y_pred_samples, lo_q, axis=0)
                    hi = np.percentile(y_pred_samples, hi_q, axis=0)
                    covered = ((y_true >= lo) & (y_true <= hi)).mean(axis=0)  # per-dim, this fold
                    calibration_rows.append({"fold": held_class, "variant": variant,
                                              "interval_pct": pct,
                                              **{f"y_{j}_coverage": covered[j] for j in range(K)},
                                              "mean_coverage": covered.mean()})

            elif model == "T2":
                variant_fn = VARIANTS[variant]
                preds, info = fit_t2_ridge(train_df, test_df, variant_fn)

            elif model == "T3":
                variant_fn = VARIANTS[variant]
                preds, info, nb = fit_t3_knn(train_df, test_df, variant_fn=variant_fn,
                                              return_neighbors_for=neighbor_return_idx)
                if nb:
                    for entry in nb:
                        entry["fold"] = held_class
                        entry["test_player"] = test_df.iloc[entry["test_row"]]["player_name"]
                        neighbor_previews.append(entry)

            elif model == "T4":
                preds = fit_t4_hungarian(train_df, test_df, assignment)

            elif model == "T5i":
                preds = fit_t5_global_mean(train_df, test_df)

            elif model == "T5ii":
                preds = fit_t5_height_tercile(train_df, test_df)

            assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-5), f"{cell_name}: predictions not a simplex"

            jsd = jensen_shannon(preds, y_true)
            l1v = l1(preds, y_true)
            t1hit = top1_hit(preds, y_true)
            t1t2 = top1_within_top2(y_true, preds)
            mae = per_dim_mae(preds, y_true)

            for i in range(len(test_df)):
                all_rows.append({
                    "cell": cell_name, "model": model, "variant": variant,
                    "fold": held_class, "player_name": test_df.iloc[i]["player_name"],
                    "jsd": jsd[i], "l1": l1v[i], "top1_hit": t1hit[i], "top1_within_top2": t1t2[i],
                })
            fold_summaries.append({
                "cell": cell_name, "fold": held_class, "n": len(test_df),
                "jsd_mean": float(jsd.mean()), "l1_mean": float(l1v.mean()),
                "top1_hit_mean": float(t1hit.mean()), "top1_within_top2_mean": float(t1t2.mean()),
                **{f"mae_y{j}": float(mae[j]) for j in range(K)},
            })
            print(f"  {cell_name}: JSD={jsd.mean():.4f} L1={l1v.mean():.4f} "
                  f"top1={t1hit.mean():.3f} ({time.time() - t_cell:.1f}s)", flush=True)

    print(f"\nTotal CV wall time: {time.time() - t0:.0f}s")

    preds_df = pd.DataFrame(all_rows)
    preds_df.to_csv(OUT_DIR / "cv_predictions_by_player.csv", index=False)

    fold_df = pd.DataFrame(fold_summaries)
    fold_df.to_csv(OUT_DIR / "cv_fold_summaries.csv", index=False)

    # pooled results: equal weight per PLAYER (not per fold - folds differ in size)
    pooled = preds_df.groupby("cell").agg(
        n=("player_name", "count"),
        jsd_mean=("jsd", "mean"), jsd_std=("jsd", "std"),
        l1_mean=("l1", "mean"), l1_std=("l1", "std"),
        top1_hit_mean=("top1_hit", "mean"),
        top1_within_top2_mean=("top1_within_top2", "mean"),
    ).reset_index()
    # fold-level spread (std of the per-fold means, not the per-player std above)
    fold_spread = fold_df.groupby("cell")["jsd_mean"].std().rename("jsd_fold_spread")
    pooled = pooled.merge(fold_spread, on="cell")
    pooled.to_csv(OUT_DIR / "cv_results.csv", index=False)
    print("\n=== Pooled CV results (sorted by mean JSD, primary selection metric) ===")
    print(pooled.sort_values("jsd_mean").to_string(index=False))

    if calibration_rows:
        pd.DataFrame(calibration_rows).to_csv(OUT_DIR / "t1_calibration.csv", index=False)
    if t1_diag_rows:
        pd.DataFrame(t1_diag_rows).to_csv(OUT_DIR / "t1_sampler_diagnostics.csv", index=False)
    if neighbor_previews:
        (OUT_DIR / "t3_neighbor_previews.json").write_text(json.dumps(neighbor_previews, indent=2, default=str))

    # T4 matching table with top-3 loading players per college archetype
    recipes = pd.read_csv(REPO_ROOT / "data" / "college" / "recipes.csv")
    top3 = {}
    for j in range(K):
        top = recipes[recipes["argmax_archetype"] == j].nlargest(3, "alpha_max")
        top3[j] = top[["player_name_raw", "season", "team", "alpha_max"]].to_dict("records")
    t4_table = {
        "assignment_college_to_nba": assignment.tolist(),
        "cosine_matrix": cos_matrix.tolist(),
        "top3_loaders_per_college_archetype": top3,
    }
    (OUT_DIR / "t4_matching_table.json").write_text(json.dumps(t4_table, indent=2, default=str))

    print(f"\nWrote all outputs to {OUT_DIR}")


if __name__ == "__main__":
    main()
