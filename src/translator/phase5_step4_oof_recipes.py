"""
Phase 5, step 4 — out-of-sample predicted RECIPES for every training anchor.

WHY THIS EXISTS: `cv_harness.py` already produces leave-one-draft-class-out
predictions for all 237 training anchors, but it only persists per-player
SCALARS (jsd, l1, top1_hit) in cv_predictions_by_player.csv — the predicted
8-vector itself is discarded once the metrics are computed. The portal's NCAA
Bridge page wants to show a player's full college -> projected -> actual
chain, which needs the vector, and until now it could only do that for the 36
held-out 2025 players.

Re-running the CV for the chosen cell alone (T1, variant b) and saving the
vectors closes that gap WITHOUT weakening the honesty of what is displayed:
every row here is predicted by a model fit on the other seven draft classes,
so it is genuinely out-of-sample for the player it describes. Showing the
deployment model's own fitted values for these players instead would be
in-sample and would flatter the model exactly where a reader is most likely
to be checking it.

Seeded identically to cv_harness (seed = fold index) so this reproduces that
run rather than being a second, subtly different one — the script verifies
that by re-deriving each player's JSD and comparing against the stored CV
metrics.

Run: python src/translator/phase5_step4_oof_recipes.py
Output: data/translator/oof_recipes_2017_2024.csv
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from input_variants import VARIANTS, build_targets  # noqa: E402
from dirichlet_model import fit_t1_dirichlet  # noqa: E402
from metrics import jensen_shannon  # noqa: E402

K = 8
CLASSES = list(range(2017, 2025))
VARIANT = "b"          # the chosen cell is T1_b
TRAIN_CSV = REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv"
OUT_CSV = REPO_ROOT / "data" / "translator" / "oof_recipes_2017_2024.csv"
CV_METRICS = REPO_ROOT / "data" / "translator" / "cv_predictions_by_player.csv"


def main() -> None:
    df = pd.read_csv(TRAIN_CSV)
    assert set(df["draft_year"].unique()) == set(CLASSES), "unexpected draft classes"

    rows = []
    t0 = time.time()
    for fold_i, held in enumerate(CLASSES):
        train_df = df[df["draft_year"] != held].reset_index(drop=True)
        test_df = df[df["draft_year"] == held].reset_index(drop=True)
        X_train, X_test, _ = VARIANTS[VARIANT](train_df, test_df)
        y_train, _ = build_targets(train_df, test_df)
        # seed=fold_i mirrors cv_harness exactly
        preds, _diag = fit_t1_dirichlet(X_train, y_train, X_test, seed=fold_i)
        assert preds.shape == (len(test_df), K)
        assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-6), "prediction is not a simplex"
        for n, (_, r) in enumerate(test_df.iterrows()):
            rows.append({
                "player_name": r["player_name"],
                "draft_year": int(r["draft_year"]),
                "held_out_fold": held,
                **{f"y_pred_{j}": float(preds[n, j]) for j in range(K)},
            })
        print(f"fold {fold_i + 1}/8 (class {held}): n_test={len(test_df)} "
              f"[{time.time() - t0:.0f}s elapsed]", flush=True)

    out = pd.DataFrame(rows)
    assert len(out) == len(df), f"expected {len(df)} rows, got {len(out)}"

    # --- reproduction check against the stored CV metrics ---------------------
    # If this really reproduced cv_harness's own run, each player's JSD derived
    # from these vectors must match the JSD that run recorded.
    cv = pd.read_csv(CV_METRICS)
    cv = cv[(cv["model"] == "T1") & (cv["variant"] == VARIANT)][["player_name", "jsd"]]
    y_true = df.set_index("player_name")[[f"y_{j}" for j in range(K)]]
    chk = out.merge(cv, on="player_name", how="inner")
    ours = np.array([
        jensen_shannon(y_true.loc[r["player_name"]].values,
                       np.array([r[f"y_pred_{j}"] for j in range(K)]))
        for _, r in chk.iterrows()
    ])
    diff = np.abs(ours - chk["jsd"].values)
    print(f"\nreproduction check vs cv_predictions_by_player.csv: n={len(chk)} "
          f"max|Δjsd|={diff.max():.2e} mean|Δjsd|={diff.mean():.2e}")
    if diff.max() > 1e-6:
        print("  WARNING: does not reproduce the stored CV run exactly — "
              "do not present these as the same numbers without explaining why.")
    else:
        print("  OK — reproduces the stored CV run.")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_CSV, index=False)
    print(f"wrote {OUT_CSV} ({len(out)} rows)")


if __name__ == "__main__":
    main()
