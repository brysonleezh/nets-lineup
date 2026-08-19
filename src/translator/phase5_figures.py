"""Phase 5 figures: predicted-vs-actual per dimension, calibration curve,
JSD by pick bucket. Reads only already-computed Phase 5 output files -
no new computation on the holdout.

Run: .nets/bin/python src/translator/phase5_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
FIG_DIR = REPO_ROOT / "figures"
K = 8


def fig_predicted_vs_actual():
    preds = pd.read_csv(TRANSLATOR_DIR / "holdout_predictions.csv")
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for j, ax in enumerate(axes.flat):
        ax.scatter(preds[f"y_true_{j}"], preds[f"y_pred_{j}"], alpha=0.6, s=25)
        ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"archetype {j}")
        ax.set_xlabel("actual")
        ax.set_ylabel("predicted")
    fig.suptitle("Phase 5: predicted vs. actual rookie recipe, per archetype dimension (n=36 holdout)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_predicted_vs_actual.png", dpi=130)
    plt.close(fig)


def fig_calibration_curve():
    curve = pd.read_csv(TRANSLATOR_DIR / "cv_calibration_curve.csv")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(curve["nominal"], curve["empirical_coverage"] * 100, "o-", label="empirical (CV)")
    ax.plot([0, 100], [0, 100], "k--", lw=1, label="perfect calibration")
    ax.set_xlabel("nominal coverage (%)")
    ax.set_ylabel("empirical coverage (%)")
    ax.set_title("T1_b calibration curve (CV residuals, pooled 237 players)\nplateaus ~63% - cannot reach 90% by widening")
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_calibration_curve.png", dpi=130)
    plt.close(fig)


def fig_jsd_by_pick_bucket():
    preds = pd.read_csv(TRANSLATOR_DIR / "holdout_predictions.csv")
    sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))
    from metrics import jensen_shannon
    y_pred = preds[[f"y_pred_{j}" for j in range(K)]].values
    y_true = preds[[f"y_true_{j}" for j in range(K)]].values
    jsd = jensen_shannon(y_pred, y_true)
    buckets = pd.cut(preds["pick_overall"], bins=[0, 14, 30, 60], labels=["1-14", "15-30", "31-60"])
    df = pd.DataFrame({"bucket": buckets, "jsd": jsd})

    fig, ax = plt.subplots(figsize=(6, 5))
    order = ["1-14", "15-30", "31-60"]
    data = [df[df["bucket"] == b]["jsd"].values for b in order]
    ax.boxplot(data, tick_labels=order, showmeans=True)
    for i, d in enumerate(data):
        ax.scatter(np.full(len(d), i + 1) + np.random.default_rng(0).uniform(-0.08, 0.08, len(d)),
                   d, alpha=0.5, s=20, color="tab:blue")
    ax.set_xlabel("pick bucket")
    ax.set_ylabel("JSD")
    ax.set_title("JSD by draft-pick bucket (holdout, n=36)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "phase5_jsd_by_pick_bucket.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    fig_predicted_vs_actual()
    fig_calibration_curve()
    fig_jsd_by_pick_bucket()
    print(f"Wrote 3 figures to {FIG_DIR}")
