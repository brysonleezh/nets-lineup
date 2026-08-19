"""
Phase 6 Step 1 - deployment refit. Refits T1_b (the pre-registered model,
unchanged) on ALL 273 anchors (2017-2025) - correct deployment behavior,
since the 2025 holdout's whole purpose (Phase 5) was already served and
the deployed model should use every real anchor available.

Preprocessing statistics are recomputed on the full 273 and persisted
COMPLETE this time - including the shot-type columns' own standardization
mu/sd, which Phase 4's chosen_model_preprocessing.json omitted (the bug
found and fixed in Phase 5 Step 1). Not repeating that gap is a named
Phase 6 test.

Run: .nets/bin/python src/translator/phase6_step1_deploy_refit.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from input_variants import variant_b, build_targets, Z_FEATURE_COLS, SHOT_TYPE_COLS  # noqa: E402
from dirichlet_model import fit_t1_dirichlet, K, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS  # noqa: E402

ANCHORS_PATH = REPO_ROOT / "data" / "anchors" / "anchors.csv"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
DEPLOY_DIR = TRANSLATOR_DIR / "deployment"
DEPLOY_SEED = 42  # same seed policy as the pre-registered 237-fit (chosen_model.md)
RAW_COLS_FOR_STD = ["overall", "log_pick", "age_at_draft", "years_in_college", "PORPAG"]


def build_complete_preprocessing(df: pd.DataFrame, col_names: list[str]) -> dict:
    """Builds the FULL preprocessing dict, including shot-type standardization
    stats (the thing Phase 4's version omitted). Mirrors variant_b()'s own
    internal computation exactly so the persisted values are provably what
    was actually used to build X."""
    impute_means = df[SHOT_TYPE_COLS].mean()
    shot_imputed = df[SHOT_TYPE_COLS].fillna(impute_means)
    cont_cols = Z_FEATURE_COLS + RAW_COLS_FOR_STD + SHOT_TYPE_COLS
    cont_source = df[Z_FEATURE_COLS + RAW_COLS_FOR_STD].copy()
    cont_source[SHOT_TYPE_COLS] = shot_imputed
    mu = cont_source[cont_cols].mean()
    sd = cont_source[cont_cols].std()
    return {
        "variant": "b",
        "column_order": col_names,
        "n_dims": len(col_names),
        "continuous_columns": cont_cols,
        "standardization_mu": mu.to_dict(),
        "standardization_sd": sd.to_dict(),
        "shot_type_imputation_means": impute_means.to_dict(),
        "n_train": len(df),
        "note": "Fitted on the full 273 anchors (2017-2025) - the deployment refit. "
                "standardization_mu/sd cover ALL continuous columns including the "
                "shot-type ones (the Phase 4 preprocessing.json omitted these; "
                "fixed in Phase 5 Step 1 for that file, built complete from the "
                "start here).",
    }


def main() -> None:
    DEPLOY_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(ANCHORS_PATH)
    assert len(df) == 273
    assert set(df["draft_year"].unique()) == set(range(2017, 2026))

    X_full, X_check, col_names = variant_b(df, df)
    assert np.allclose(X_full, X_check)
    assert X_full.shape == (273, 21)

    preprocessing = build_complete_preprocessing(df, col_names)
    # verify the hand-rebuilt preprocessing reproduces variant_b's real output exactly
    mu, sd = preprocessing["standardization_mu"], preprocessing["standardization_sd"]
    impute_means = preprocessing["shot_type_imputation_means"]
    shot_imputed = df[SHOT_TYPE_COLS].fillna(pd.Series(impute_means))
    cont_source = df[Z_FEATURE_COLS + RAW_COLS_FOR_STD].copy()
    cont_source[SHOT_TYPE_COLS] = shot_imputed
    cont_cols = preprocessing["continuous_columns"]
    check = ((cont_source[cont_cols] - pd.Series(mu)) / pd.Series(sd)).values
    assert np.allclose(check, X_full[:, :len(cont_cols)], atol=1e-10), \
        "hand-rebuilt preprocessing doesn't reproduce variant_b's real output - do not trust it"
    print("Preprocessing verified complete and self-consistent (atol=1e-10).")

    y_full, _ = build_targets(df, df)

    print(f"Fitting T1_b (deployment) on all {len(df)} anchors, seed={DEPLOY_SEED}...")
    _, diag = fit_t1_dirichlet(X_full, y_full, X_full, seed=DEPLOY_SEED, verbose=True)
    print(f"\nConverged at target_accept={diag['target_accept']}: "
          f"max_rhat={diag['max_rhat']:.4f}, min_ess={diag['min_ess']:.0f}, "
          f"n_divergent={diag['n_divergent']}/{diag['num_samples_total']}")
    assert diag["max_rhat"] < 1.01 and diag["min_ess"] >= 400 and diag["n_divergent"] == 0, \
        "deployment refit failed the same convergence gate every prior fit was held to - stop"

    # recover B, phi posterior samples directly (same approach as Phase 4's fit_final_model.py)
    import numpyro
    from numpyro.infer import MCMC, NUTS
    import jax
    import jax.numpy as jnp
    from dirichlet_model import _model, _add_intercept

    Xtr_aug = _add_intercept(X_full)
    kernel = NUTS(_model, target_accept_prob=diag["target_accept"])
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=NUM_CHAINS, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(DEPLOY_SEED), X=Xtr_aug, y=jnp.asarray(y_full, dtype=jnp.float32), k=K)
    samples = mcmc.get_samples()
    B_samples = np.asarray(samples["B"])  # (S, p_aug, K-1)
    phi_samples = np.asarray(samples["phi"])

    np.savez(DEPLOY_DIR / "posterior.npz", B=B_samples, phi=phi_samples)
    (TRANSLATOR_DIR / "deployment_preprocessing.json").write_text(json.dumps(preprocessing, indent=2, default=str))

    # --- 237-vs-273 coefficient agreement ---
    with np.load(TRANSLATOR_DIR / "chosen_model_posterior.npz") as z:
        B_237 = z["B"].mean(axis=0)  # (p_aug, K-1) posterior mean
    B_273 = B_samples.mean(axis=0)
    assert B_237.shape == B_273.shape
    flat_237, flat_273 = B_237.flatten(), B_273.flatten()
    corr = float(np.corrcoef(flat_237, flat_273)[0, 1])
    abs_shifts = np.abs(flat_237 - flat_273)
    largest_shift = float(abs_shifts.max())
    largest_shift_idx = int(abs_shifts.argmax())
    p_aug = B_237.shape[0]
    row_names = ["intercept"] + col_names
    shift_row = row_names[largest_shift_idx // B_237.shape[1]]
    shift_col = largest_shift_idx % B_237.shape[1]
    print(f"\n237-vs-273 coefficient agreement: correlation={corr:.4f}, "
          f"largest |shift|={largest_shift:.4f} (row='{shift_row}', archetype-col={shift_col})")
    material_disagreement = corr < 0.95 or largest_shift > 1.0
    if material_disagreement:
        print("** FLAGGED: correlation<0.95 or a shift>1.0 - this would be surprising with only "
              "36 rows added and should be investigated, not glossed over. **")

    manifest = {
        "chosen_cell": "T1_b", "model": "T1 - Dirichlet regression", "variant": "b",
        "fit_date": "2026-08-14", "n_train": len(df), "seed": DEPLOY_SEED,
        "git_hash": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                    capture_output=True, text=True).stdout.strip() or "uncommitted",
        "convergence": {k: v for k, v in diag.items() if k != "y_pred_samples"},
        "coefficient_agreement_vs_237fit": {
            "correlation": corr, "largest_abs_shift": largest_shift,
            "largest_shift_row": shift_row, "largest_shift_archetype_col": shift_col,
            "material_disagreement_flag": material_disagreement,
        },
        "posterior_file": "deployment/posterior.npz",
        "preprocessing_file": "deployment_preprocessing.json",
    }
    (DEPLOY_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nWrote {DEPLOY_DIR / 'posterior.npz'} (B: {B_samples.shape}, phi: {phi_samples.shape})")
    print(f"Wrote {TRANSLATOR_DIR / 'deployment_preprocessing.json'}")
    print(f"Wrote {DEPLOY_DIR / 'manifest.json'}")


if __name__ == "__main__":
    main()
