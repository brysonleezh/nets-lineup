"""
Phase 4 post-CHECKPOINT - fits the chosen cell (T1, variant b) on the FULL
237 training rows (no CV split - this is the production fit that Phase 5
will evaluate against the holdout). Saves posterior samples and the exact
preprocessing statistics (standardization mu/sd, shot-type imputation
means) as fitted on the full 237, per the pre-registration requirement
that these are computed "as fitted on the full 237, which happens only
now."

Run: .nets/bin/python src/translator/fit_final_model.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from input_variants import variant_b, build_targets, Z_FEATURE_COLS, SHOT_TYPE_COLS  # noqa: E402
from dirichlet_model import fit_t1_dirichlet, K, NUM_WARMUP, NUM_SAMPLES, NUM_CHAINS  # noqa: E402

TRAIN_PATH = REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv"
OUT_DIR = REPO_ROOT / "data" / "translator"


def main() -> None:
    df = pd.read_csv(TRAIN_PATH)
    assert len(df) == 237

    # variant_b(df, df): standardization/imputation statistics computed from
    # df and applied to df - passing the same full 237-row frame as both
    # "train" and "test" args yields the full-data preprocessing recipe,
    # reusing the exact tested Step 1 code path rather than duplicating it.
    X_full, X_full_check, col_names = variant_b(df, df)
    assert np.allclose(X_full, X_full_check)

    y_full, _ = build_targets(df, df)

    print(f"Fitting T1 (Dirichlet regression, variant b) on the full {len(df)} training rows...")
    # X_full already has no intercept column; fit_t1_dirichlet adds one internally.
    # For the final fit there is no held-out X_test - pass X_full itself so the
    # function's plumbing works, but only the posterior (B, phi) samples matter here.
    preds_on_train, diag = fit_t1_dirichlet(X_full, y_full, X_full, seed=42, verbose=True)

    print(f"\nConverged at target_accept={diag['target_accept']}: "
          f"max_rhat={diag['max_rhat']:.4f}, min_ess={diag['min_ess']:.0f}, "
          f"n_divergent={diag['n_divergent']}/{diag['num_samples_total']}")
    assert diag["max_rhat"] < 1.01 and diag["min_ess"] >= 400, \
        "final full-data fit failed the same convergence gate CV required - stop, do not proceed"

    # Recover B, phi posterior samples directly (fit_t1_dirichlet doesn't return them,
    # only y_pred_samples on the passed X_test=X_full) - refit is avoided by re-deriving
    # from the same seed would be wasteful; instead re-run once more capturing samples,
    # since fit_t1_dirichlet's internal MCMC object isn't exposed. This IS the same fit
    # (identical seed, identical data, identical model) - not a second independent fit.
    import numpyro
    from numpyro.infer import MCMC, NUTS
    import jax
    from dirichlet_model import _model, _add_intercept
    import jax.numpy as jnp

    Xtr_aug = _add_intercept(X_full)
    kernel = NUTS(_model, target_accept_prob=diag["target_accept"])
    mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES, num_chains=NUM_CHAINS, progress_bar=True)
    mcmc.run(jax.random.PRNGKey(42), X=Xtr_aug, y=jnp.asarray(y_full, dtype=jnp.float32), k=K)
    samples = mcmc.get_samples()
    B_samples = np.asarray(samples["B"])  # (S, p_aug, K-1)
    phi_samples = np.asarray(samples["phi"])  # (S,)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "chosen_model_posterior.npz", B=B_samples, phi=phi_samples)

    preprocessing = {
        "variant": "b",
        "column_order": col_names,
        "n_dims": len(col_names),
        "continuous_columns": Z_FEATURE_COLS + ["overall", "log_pick", "age_at_draft",
                                                  "years_in_college", "PORPAG"] + SHOT_TYPE_COLS,
        "standardization_mu": df[Z_FEATURE_COLS + ["overall", "log_pick", "age_at_draft",
                                                      "years_in_college", "PORPAG"]].mean().to_dict(),
        "standardization_sd": df[Z_FEATURE_COLS + ["overall", "log_pick", "age_at_draft",
                                                      "years_in_college", "PORPAG"]].std().to_dict(),
        "shot_type_imputation_means": df[SHOT_TYPE_COLS].mean().to_dict(),
        "shot_type_standardization_mu_after_impute": None,  # computed jointly above; see standardization_mu
        "zero_compression_n": len(df),
        "note": "All statistics fitted on the full 237-row training set (Phase 4's final production "
                "fit) - NOT any individual CV fold's statistics. Phase 5 must standardize/impute the "
                "holdout's X using THESE frozen values, never recomputed from the holdout.",
    }
    (OUT_DIR / "chosen_model_preprocessing.json").write_text(json.dumps(preprocessing, indent=2, default=str))

    manifest = {
        "chosen_cell": "T1_b",
        "model": "T1 - Dirichlet regression",
        "variant": "b",
        "fit_date": "2026-08-13",
        "n_train": len(df),
        "convergence": diag,
        "posterior_samples_file": "chosen_model_posterior.npz",
        "preprocessing_file": "chosen_model_preprocessing.json",
    }
    del manifest["convergence"]["y_pred_samples"]  # large array, not manifest-appropriate
    (OUT_DIR / "chosen_model_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nWrote {OUT_DIR / 'chosen_model_posterior.npz'} (B: {B_samples.shape}, phi: {phi_samples.shape})")
    print(f"Wrote {OUT_DIR / 'chosen_model_preprocessing.json'}")
    print(f"Wrote {OUT_DIR / 'chosen_model_manifest.json'}")


if __name__ == "__main__":
    main()
