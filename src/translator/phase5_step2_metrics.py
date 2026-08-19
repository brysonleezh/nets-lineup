"""
Phase 5 Step 2 - metrics, exactly the pre-registered list, nothing added.
Computed on the 36 holdout players, equal weight per player.

Run: .nets/bin/python src/translator/phase5_step2_metrics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from metrics import jensen_shannon, l1, top1_hit, top1_within_top2, per_dim_mae  # noqa: E402
from models import hungarian_assignment, fit_t4_hungarian, fit_t5_global_mean, fit_t5_height_tercile  # noqa: E402

K = 8
ANCHORS_DIR = REPO_ROOT / "data" / "anchors"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 2025


def bootstrap_ci(values: np.ndarray, seed: int = BOOTSTRAP_SEED, n_boot: int = BOOTSTRAP_N) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[b] = values[idx].mean()
    return float(values.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    preds = pd.read_csv(TRANSLATOR_DIR / "holdout_predictions.csv")
    n = len(preds)
    assert n == 36

    y_pred = preds[[f"y_pred_{j}" for j in range(K)]].values
    y_true = preds[[f"y_true_{j}" for j in range(K)]].values

    jsd = jensen_shannon(y_pred, y_true)
    l1v = l1(y_pred, y_true)
    t1hit = top1_hit(y_pred, y_true)
    t1t2 = top1_within_top2(y_true, y_pred)
    mae = per_dim_mae(y_pred, y_true)

    print("=== Headline metrics (T1_b on the 36-player holdout) ===")
    print(f"JSD (primary): mean={jsd.mean():.4f}")
    print(f"L1: mean={l1v.mean():.4f}")
    print(f"Top-1 hit rate: {t1hit.mean():.4f}")
    print(f"Top-1-within-top-2: {t1t2.mean():.4f}")

    # --- per-dimension MAE vs. baseline (training-set mean) MAE ---
    train_df = pd.read_csv(ANCHORS_DIR / "train_2017_2024.csv")
    y_cols = [f"y_{j}" for j in range(K)]
    train_mean_recipe = train_df[y_cols].values.mean(axis=0)
    train_mean_recipe = train_mean_recipe / train_mean_recipe.sum()
    baseline_pred = np.tile(train_mean_recipe, (n, 1))
    baseline_mae = per_dim_mae(baseline_pred, y_true)
    print("\nPer-dimension MAE (model vs. training-mean baseline):")
    for j in range(K):
        print(f"  y_{j}: model={mae[j]:.4f}  baseline={baseline_mae[j]:.4f}  "
              f"{'better' if mae[j] < baseline_mae[j] else 'WORSE'}")

    # --- interval coverage, 50%/90% ---
    lo50, hi50 = preds[[f"pi50_lo_{j}" for j in range(K)]].values, preds[[f"pi50_hi_{j}" for j in range(K)]].values
    lo90, hi90 = preds[[f"pi90_lo_{j}" for j in range(K)]].values, preds[[f"pi90_hi_{j}" for j in range(K)]].values
    cov50 = ((y_true >= lo50) & (y_true <= hi50)).mean(axis=0)
    cov90 = ((y_true >= lo90) & (y_true <= hi90)).mean(axis=0)
    print(f"\n50% PI coverage: per-dim={np.round(cov50, 3).tolist()}, mean={cov50.mean():.3f}")
    print(f"90% PI coverage: per-dim={np.round(cov90, 3).tolist()}, mean={cov90.mean():.3f}")

    # --- bootstrap CIs (2000 resamples over the 36 players) ---
    jsd_mean, jsd_lo, jsd_hi = bootstrap_ci(jsd)
    l1_mean, l1_lo, l1_hi = bootstrap_ci(l1v)
    hit_mean, hit_lo, hit_hi = bootstrap_ci(t1hit)
    print(f"\nBootstrap 95% CI (n_boot={BOOTSTRAP_N}, n=36):")
    print(f"  JSD: {jsd_mean:.4f} [{jsd_lo:.4f}, {jsd_hi:.4f}]")
    print(f"  L1: {l1_mean:.4f} [{l1_lo:.4f}, {l1_hi:.4f}]")
    print(f"  top-1 hit: {hit_mean:.4f} [{hit_lo:.4f}, {hit_hi:.4f}]")

    # --- baselines on the identical holdout, identical transform ---
    train_df_full = train_df  # the exact 237 rows T1_b was trained on
    holdout_df = pd.read_csv(ANCHORS_DIR / "holdout_2025.csv")  # already opened/scored in Step 1; re-reading
    # the same already-scored file for baseline comparison is not a second holdout "evaluation"
    # of the chosen model - T4/T5 are reference points computed the same way Phase 4's CV did.
    assignment, _, _ = hungarian_assignment()
    t4_pred = fit_t4_hungarian(train_df_full, holdout_df, assignment)
    t5i_pred = fit_t5_global_mean(train_df_full, holdout_df)
    t5ii_pred = fit_t5_height_tercile(train_df_full, holdout_df)

    baseline_rows = []
    for name, pred in [("T1_b (chosen model)", y_pred), ("T4 Hungarian", t4_pred),
                        ("T5(i) global mean", t5i_pred), ("T5(ii) height tercile", t5ii_pred)]:
        j, l, h = jensen_shannon(pred, y_true), l1(pred, y_true), top1_hit(pred, y_true)
        baseline_rows.append({"model": name, "jsd_mean": j.mean(), "l1_mean": l.mean(), "top1_hit_mean": h.mean()})
    baseline_df = pd.DataFrame(baseline_rows)
    print("\n=== Baseline comparison on the identical holdout ===")
    print(baseline_df.to_string(index=False))

    beats_t4 = baseline_df.loc[baseline_df["model"] == "T1_b (chosen model)", "jsd_mean"].iloc[0] < \
        baseline_df.loc[baseline_df["model"] == "T4 Hungarian", "jsd_mean"].iloc[0]
    beats_t5i = baseline_df.loc[baseline_df["model"] == "T1_b (chosen model)", "jsd_mean"].iloc[0] < \
        baseline_df.loc[baseline_df["model"] == "T5(i) global mean", "jsd_mean"].iloc[0]
    beats_t5ii = baseline_df.loc[baseline_df["model"] == "T1_b (chosen model)", "jsd_mean"].iloc[0] < \
        baseline_df.loc[baseline_df["model"] == "T5(ii) height tercile", "jsd_mean"].iloc[0]
    eligible = beats_t4 and beats_t5i and beats_t5ii
    print(f"\nDeployment eligibility (beats T4 AND both floors, on JSD): "
          f"T4={beats_t4}, T5i={beats_t5i}, T5ii={beats_t5ii} -> ELIGIBLE={eligible}")

    # --- save everything ---
    metrics_out = pd.DataFrame([{
        "model": "T1_b", "n": n,
        "jsd_mean": jsd.mean(), "jsd_boot_lo": jsd_lo, "jsd_boot_hi": jsd_hi,
        "l1_mean": l1v.mean(), "l1_boot_lo": l1_lo, "l1_boot_hi": l1_hi,
        "top1_hit_mean": t1hit.mean(), "top1_hit_boot_lo": hit_lo, "top1_hit_boot_hi": hit_hi,
        "top1_within_top2_mean": t1t2.mean(),
        "coverage50_mean": cov50.mean(), "coverage90_mean": cov90.mean(),
        "beats_t4": beats_t4, "beats_t5i": beats_t5i, "beats_t5ii": beats_t5ii, "eligible": eligible,
        **{f"mae_y{j}": mae[j] for j in range(K)},
        **{f"baseline_mae_y{j}": baseline_mae[j] for j in range(K)},
        **{f"coverage50_y{j}": cov50[j] for j in range(K)},
        **{f"coverage90_y{j}": cov90[j] for j in range(K)},
    }])
    metrics_out.to_csv(TRANSLATOR_DIR / "holdout_metrics.csv", index=False)
    baseline_df.to_csv(TRANSLATOR_DIR / "holdout_baseline_comparison.csv", index=False)
    print(f"\nWrote {TRANSLATOR_DIR / 'holdout_metrics.csv'}")
    print(f"Wrote {TRANSLATOR_DIR / 'holdout_baseline_comparison.csv'}")


if __name__ == "__main__":
    main()
