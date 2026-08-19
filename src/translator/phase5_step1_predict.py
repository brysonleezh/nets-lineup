"""
Phase 5 Step 1 - the one-shot prediction. Opens holdout_2025.csv exactly
once. Refuses to run again (writes holdout_evaluated.json as a lock file)
unless --rerun is passed explicitly - any rerun must be disclosed in the
report with the reason, per spec.

Builds variant-(b) inputs using ONLY the frozen preprocessing recipe from
chosen_model_preprocessing.json (Phase 4's pre-registration) - no statistic
is computed from the holdout itself anywhere in this file.

Run: .nets/bin/python src/translator/phase5_step1_predict.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

import numpyro  # noqa: E402
numpyro.set_host_device_count(4)
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpyro.distributions as dist  # noqa: E402

ANCHORS_DIR = REPO_ROOT / "data" / "anchors"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
HOLDOUT_PATH = ANCHORS_DIR / "holdout_2025.csv"
MANIFEST_PATH = ANCHORS_DIR / "holdout_manifest.json"
EVALUATED_LOCK_PATH = TRANSLATOR_DIR / "holdout_evaluated.json"

K = 8
Y_COLS = [f"y_{j}" for j in range(K)]
Z_FEATURE_COLS = ["PLAYER_HEIGHT_INCHES_z", "PTS_PER_100_z", "TS%_z", "USG%_z", "AST%_z",
                   "TOV%_z", "STL%_z", "BLK%_z", "TRB%_z", "FTr_z",
                   "% of FG Ast'd_2P_z", "% of FG Ast'd_3P_z"]
SHOT_TYPE_COLS = ["rim_finishing_share", "three_pt_jumper_share"]
RAW_COLS_FOR_STD = ["overall", "log_pick", "age_at_draft", "years_in_college", "PORPAG"]


def apply_frozen_variant_b(df: pd.DataFrame, preprocessing: dict) -> tuple[np.ndarray, list[str]]:
    """Transform df (the holdout) using ONLY the frozen mu/sd/imputation
    values persisted in preprocessing - never recomputes anything from df.
    Asserts the source of every statistic is the frozen dict, not df."""
    cols = preprocessing["column_order"]
    mu = preprocessing["standardization_mu"]
    sd = preprocessing["standardization_sd"]
    impute_means = preprocessing["shot_type_imputation_means"]

    missing = df[SHOT_TYPE_COLS].isna().any(axis=1).astype(float).values
    shot = df[SHOT_TYPE_COLS].copy()
    for c in SHOT_TYPE_COLS:
        shot[c] = shot[c].fillna(impute_means[c])  # frozen training-fold mean, not df's own

    cont_source = df[[c for c in Z_FEATURE_COLS + RAW_COLS_FOR_STD]].copy()
    cont_source[SHOT_TYPE_COLS] = shot
    standardized = pd.DataFrame(index=df.index)
    for c in Z_FEATURE_COLS + RAW_COLS_FOR_STD + SHOT_TYPE_COLS:
        assert c in mu and c in sd, f"{c} missing from frozen standardization stats"
        standardized[c] = (cont_source[c] - mu[c]) / sd[c]  # frozen mu/sd, not df's own

    X = np.column_stack([
        standardized[c].values for c in Z_FEATURE_COLS + RAW_COLS_FOR_STD + SHOT_TYPE_COLS
    ] + [df["conf_tier"].astype(float).values, missing])
    assert X.shape[1] == len(cols) == 21
    assert not np.isnan(X).any(), "NaN survived the frozen-imputation transform"
    return X, cols


def predict_from_posterior(X: np.ndarray, B_samples: np.ndarray, phi_samples: np.ndarray,
                            seed: int = 777) -> dict:
    Xaug = jnp.concatenate([jnp.ones((X.shape[0], 1)), jnp.asarray(X, dtype=jnp.float32)], axis=1)
    Bj = jnp.asarray(B_samples)
    logits_free = jnp.einsum("np,spk->snk", Xaug, Bj)  # (S, n, K-1)
    logits = jnp.concatenate([logits_free, jnp.zeros((*logits_free.shape[:-1], 1))], axis=-1)
    mu_samples = jax.nn.softmax(logits, axis=-1)  # (S, n, K)
    mu_mean = np.asarray(mu_samples.mean(axis=0))

    conc = jnp.asarray(phi_samples)[:, None, None] * mu_samples
    y_pred_samples = np.asarray(dist.Dirichlet(conc).sample(jax.random.PRNGKey(seed)))  # (S, n, K)
    return {"mu_mean": mu_mean, "y_pred_samples": y_pred_samples}


def main(rerun: bool = False) -> None:
    if EVALUATED_LOCK_PATH.exists() and not rerun:
        raise SystemExit(
            f"{EVALUATED_LOCK_PATH} already exists - this phase evaluates the holdout exactly "
            f"once. Pass --rerun with a disclosed reason in the report if this is intentional."
        )

    # --- re-verify hash at read time, THEN open ---
    manifest = json.loads(MANIFEST_PATH.read_text())
    actual_hash = hashlib.sha256(HOLDOUT_PATH.read_bytes()).hexdigest()
    assert actual_hash == manifest["sha256"], "HOLDOUT HASH MISMATCH - STOP, do not proceed"
    print(f"Holdout hash verified: {actual_hash}")

    holdout = pd.read_csv(HOLDOUT_PATH)
    assert len(holdout) == 36
    print(f"Opened holdout_2025.csv: {len(holdout)} rows")

    preprocessing = json.loads((TRANSLATOR_DIR / "chosen_model_preprocessing.json").read_text())
    X, col_names = apply_frozen_variant_b(holdout, preprocessing)
    print(f"Built variant-(b) inputs: {X.shape}, using only frozen preprocessing statistics")

    with np.load(TRANSLATOR_DIR / "chosen_model_posterior.npz") as z:
        B_samples, phi_samples = z["B"], z["phi"]
    print(f"Loaded posterior: B {B_samples.shape}, phi {phi_samples.shape}")

    result = predict_from_posterior(X, B_samples, phi_samples)
    mu_mean = result["mu_mean"]
    y_pred_samples = result["y_pred_samples"]

    assert (mu_mean >= -1e-9).all()
    row_sums = mu_mean.sum(axis=1)
    max_dev = np.abs(row_sums - 1.0).max()
    assert max_dev < 1e-6, f"prediction simplex violation: max dev {max_dev:.2e}"
    print(f"Simplex validity: PASS (max |sum-1| = {max_dev:.2e})")

    lo50 = np.percentile(y_pred_samples, 25, axis=0)
    hi50 = np.percentile(y_pred_samples, 75, axis=0)
    lo90 = np.percentile(y_pred_samples, 5, axis=0)
    hi90 = np.percentile(y_pred_samples, 95, axis=0)

    y_true = holdout[Y_COLS].values
    out = pd.DataFrame({"player_name": holdout["player_name"], "overall": holdout["overall"],
                         "pick_overall": holdout["overall"], "age_at_draft": holdout["age_at_draft"],
                         "rookie_minutes": holdout["rookie_minutes"], "c_argmax": holdout["c_argmax"]})
    for j in range(K):
        out[f"y_pred_{j}"] = mu_mean[:, j]
        out[f"y_true_{j}"] = y_true[:, j]
        out[f"pi50_lo_{j}"] = lo50[:, j]
        out[f"pi50_hi_{j}"] = hi50[:, j]
        out[f"pi90_lo_{j}"] = lo90[:, j]
        out[f"pi90_hi_{j}"] = hi90[:, j]
    out["pred_argmax"] = mu_mean.argmax(axis=1)
    out["true_argmax"] = y_true.argmax(axis=1)

    out_path = TRANSLATOR_DIR / "holdout_predictions.csv"
    out.to_csv(out_path, index=False)
    np.save(TRANSLATOR_DIR / "holdout_y_pred_samples.npy", y_pred_samples)
    print(f"Wrote {out_path} ({len(out)} rows)")

    git_hash = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                               capture_output=True, text=True).stdout.strip() or "uncommitted"
    manifest_hash = hashlib.sha256(
        (TRANSLATOR_DIR / "chosen_model_manifest.json").read_bytes()).hexdigest()
    evaluated = {
        "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": git_hash,
        "holdout_sha256": actual_hash,
        "chosen_model_manifest_sha256": manifest_hash,
        "n_holdout": len(holdout),
        "rerun": rerun,
    }
    EVALUATED_LOCK_PATH.write_text(json.dumps(evaluated, indent=2))
    print(f"Wrote {EVALUATED_LOCK_PATH} (lock file - further runs blocked without --rerun)")


if __name__ == "__main__":
    main(rerun="--rerun" in sys.argv)
