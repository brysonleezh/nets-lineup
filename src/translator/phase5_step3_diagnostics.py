"""
Phase 5 Step 3 - descriptive diagnostics beyond the headline. None of these
alter the Step 2 verdict; they inform how Phase 6 presents predictions.

Sub-step 4 (empirical calibration) needs per-dimension CV predictions that
Phase 4's cv_harness.py computed but only ever reduced to aggregate scores
before discarding - regenerates them here by rerunning T1 on the same 8
CV folds, variant (b) only (not the full 12-cell grid), using only
data/anchors/train_2017_2024.csv (never the holdout). This reproduces
Phase 4's own already-completed CV, at finer granularity, not a new
result and not a Phase 5 "adjustment" - the deployed model
(chosen_model_posterior.npz) is untouched.

Run: .nets/bin/python src/translator/phase5_step3_diagnostics.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from input_variants import variant_b, build_targets  # noqa: E402
from dirichlet_model import fit_t1_dirichlet  # noqa: E402
from metrics import jensen_shannon  # noqa: E402

K = 8
ANCHORS_DIR = REPO_ROOT / "data" / "anchors"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
CLASSES = list(range(2017, 2025))


def main() -> None:
    preds = pd.read_csv(TRANSLATOR_DIR / "holdout_predictions.csv")
    y_pred = preds[[f"y_pred_{j}" for j in range(K)]].values
    y_true = preds[[f"y_true_{j}" for j in range(K)]].values
    jsd = jensen_shannon(y_pred, y_true)

    # --- 1. error vs. draft position ---
    print("=== 1. JSD by pick bucket ===")
    buckets = pd.cut(preds["pick_overall"], bins=[0, 14, 30, 60], labels=["1-14", "15-30", "31-60"])
    bucket_df = pd.DataFrame({"bucket": buckets, "jsd": jsd, "pick": preds["pick_overall"]})
    bucket_summary = bucket_df.groupby("bucket", observed=True)["jsd"].agg(["mean", "count"])
    print(bucket_summary)

    # --- 2. error vs. rookie minutes ---
    print("\n=== 2. JSD vs. rookie minutes ===")
    corr = np.corrcoef(preds["rookie_minutes"], jsd)[0, 1]
    print(f"Pearson correlation(rookie_minutes, JSD) = {corr:.3f}")
    minutes_df = pd.DataFrame({"player": preds["player_name"], "rookie_minutes": preds["rookie_minutes"], "jsd": jsd})
    minutes_df.to_csv(TRANSLATOR_DIR / "holdout_jsd_vs_minutes.csv", index=False)

    # --- 3. error vs. college archetype ---
    print("\n=== 3. Mean JSD by college archetype (c_argmax) ===")
    carch_df = pd.DataFrame({"c_argmax": preds["c_argmax"], "jsd": jsd})
    carch_summary = carch_df.groupby("c_argmax")["jsd"].agg(["mean", "count"])
    print(carch_summary)

    # --- 4. empirical calibration curve from CV residuals (not holdout) ---
    print("\n=== 4. Empirical calibration (from Phase 4's CV, regenerated with per-dim samples) ===")
    train_df = pd.read_csv(ANCHORS_DIR / "train_2017_2024.csv")
    all_y_true, all_y_pred_samples = [], []
    for held_class in CLASSES:
        tr = train_df[train_df["draft_year"] != held_class].reset_index(drop=True)
        te = train_df[train_df["draft_year"] == held_class].reset_index(drop=True)
        X_tr, X_te, _ = variant_b(tr, te)
        y_tr, _ = build_targets(tr, te)
        _, diag = fit_t1_dirichlet(X_tr, y_tr, X_te, seed=held_class)
        all_y_true.append(te[[f"y_{j}" for j in range(K)]].values)
        all_y_pred_samples.append(diag["y_pred_samples"])  # (S, n_fold, K)
        print(f"  fold {held_class}: refit OK, max_rhat={diag['max_rhat']:.4f}, min_ess={diag['min_ess']:.0f}")

    cv_y_true = np.concatenate(all_y_true, axis=0)  # (237, K)
    cv_y_pred_samples = np.concatenate(all_y_pred_samples, axis=1)  # (S, 237, K)

    # Fine grid up to a near-100% nominal level, so the curve can be inverted:
    # "what nominal percentile-interval width, if used INSTEAD of the literal 90% one,
    # actually achieves 90% true coverage" - this avoids a flawed median+/-k*halfwidth
    # construction (tried first, discarded: it gave 0.90 coverage at k=1.0 versus the
    # direct percentile method's 0.595 at the same nominal level - Dirichlet marginals
    # are skewed near 0/1, so a symmetric interval around the median, clipped to [0,1],
    # is a fundamentally different and misleading construction for this data).
    nominal_levels = list(range(10, 100, 5)) + [99, 99.9]
    raw_coverage = []
    percentile_bounds = {}
    for pct in nominal_levels:
        lo_q, hi_q = (100 - pct) / 2, 100 - (100 - pct) / 2
        lo = np.percentile(cv_y_pred_samples, lo_q, axis=0)
        hi = np.percentile(cv_y_pred_samples, hi_q, axis=0)
        cov = ((cv_y_true >= lo) & (cv_y_true <= hi)).mean()
        raw_coverage.append(cov)
        percentile_bounds[pct] = (lo_q, hi_q)
    calib_curve = pd.DataFrame({"nominal": nominal_levels, "empirical_coverage": raw_coverage})
    print(calib_curve.to_string(index=False))

    # invert: find the nominal level whose interval achieves 90% TRUE coverage
    target = 0.90
    max_reachable = calib_curve["empirical_coverage"].max()
    holdout_lo90 = preds[[f"pi90_lo_{j}" for j in range(K)]].values
    holdout_hi90 = preds[[f"pi90_hi_{j}" for j in range(K)]].values
    raw_cov90 = ((y_true >= holdout_lo90) & (y_true <= holdout_hi90)).mean()

    if max_reachable < target:
        # Genuine finding, not a bug to route around: even the widest percentile interval
        # tested (nominal=99.9%, i.e. nearly the full posterior predictive sample range)
        # only reaches ~63% true coverage on CV - simple interval-widening CANNOT fix this
        # miscalibration. This points to something interval-scaling can't repair (point-
        # prediction bias for a subset of players, or missing between-player variance the
        # Dirichlet's single scalar phi can't represent) rather than "intervals slightly
        # too narrow." Reported as-is - no numerically-forced 90% interval is computed.
        required_nominal = None
        widened_cov = None
        print(f"\n** Even the widest tested percentile interval (nominal=99.9%, i.e. nearly the "
              f"full posterior-predictive sample range) only reaches {max_reachable:.1%} TRUE "
              f"coverage on CV - short of the 90% target. Simple interval-widening cannot fix "
              f"this: it is evidence of miscalibration beyond a scale problem (e.g. point-"
              f"prediction bias, or between-player variance the model's single scalar phi "
              f"cannot represent), not just 'intervals too narrow.' No corrected interval is "
              f"reported - widening further would be numerically forcing percentiles beyond "
              f"[0,100], not a real fix. **")
    else:
        idx = np.searchsorted(calib_curve["empirical_coverage"].values, target)
        idx = min(max(idx, 1), len(calib_curve) - 1)
        x0, x1 = calib_curve["empirical_coverage"].iloc[idx - 1], calib_curve["empirical_coverage"].iloc[idx]
        n0, n1 = calib_curve["nominal"].iloc[idx - 1], calib_curve["nominal"].iloc[idx]
        required_nominal = n0 + (target - x0) * (n1 - n0) / (x1 - x0) if x1 != x0 else n0
        print(f"\nTo achieve 90% TRUE coverage, must request a nominal {required_nominal:.1f}% "
              f"percentile interval instead of the literal 90% one.")
        holdout_y_pred_samples = np.load(TRANSLATOR_DIR / "holdout_y_pred_samples.npy")  # (S, 36, K)
        lo_q, hi_q = (100 - required_nominal) / 2, 100 - (100 - required_nominal) / 2
        widened_lo = np.percentile(holdout_y_pred_samples, lo_q, axis=0)
        widened_hi = np.percentile(holdout_y_pred_samples, hi_q, axis=0)
        widened_cov = ((y_true >= widened_lo) & (y_true <= widened_hi)).mean()
        print(f"On the HOLDOUT (width derived from CV only, applied here for comparison): "
              f"raw-90%-labeled-interval coverage={raw_cov90:.3f}, "
              f"empirically-corrected ({required_nominal:.1f}%-percentile) interval coverage={widened_cov:.3f}")

    calib_curve.to_csv(TRANSLATOR_DIR / "cv_calibration_curve.csv", index=False)
    print(f"\n(For reference: raw-90%-labeled-interval coverage on holdout = {raw_cov90:.3f})")

    # --- 5. three worst and three best predictions by JSD ---
    print("\n=== 5. Three worst / three best predictions by JSD ===")
    ranked = pd.DataFrame({
        "player_name": preds["player_name"], "jsd": jsd, "pick_overall": preds["pick_overall"],
        "rookie_minutes": preds["rookie_minutes"], "pred_argmax": preds["pred_argmax"],
        "true_argmax": preds["true_argmax"], "c_argmax": preds["c_argmax"],
    }).sort_values("jsd")
    print("Best 3:")
    print(ranked.head(3).to_string(index=False))
    print("Worst 3:")
    print(ranked.tail(3).to_string(index=False))
    ranked.to_csv(TRANSLATOR_DIR / "holdout_ranked_by_jsd.csv", index=False)

    summary = {
        "pick_bucket_jsd": bucket_summary.to_dict(),
        "minutes_jsd_correlation": corr,
        "college_archetype_jsd": carch_summary.to_dict(),
        "required_nominal_pct_for_90pct_true_coverage": required_nominal,
        "max_reachable_coverage_via_widening_cv": max_reachable,
        "holdout_raw_90pct_labeled_coverage": raw_cov90,
        "holdout_empirically_corrected_coverage": widened_cov,
        "best3": ranked.head(3)["player_name"].tolist(),
        "worst3": ranked.tail(3)["player_name"].tolist(),
    }
    (TRANSLATOR_DIR / "phase5_diagnostics_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote diagnostics outputs to {TRANSLATOR_DIR}")


if __name__ == "__main__":
    main()
