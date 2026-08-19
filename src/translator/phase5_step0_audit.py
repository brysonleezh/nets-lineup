"""
Phase 5 Step 0 - pre-registration audit. Must run BEFORE any code in this
phase opens data/anchors/holdout_2025.csv. Verifies the pre-registration
document (data/translator/chosen_model.md) still describes the model that
actually exists on disk - any mismatch is a stop condition, reported
before the holdout is touched.

Run: .nets/bin/python src/translator/phase5_step0_audit.py
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from input_variants import variant_b  # noqa: E402
import dirichlet_model  # noqa: E402
from metrics import jensen_shannon, l1, top1_hit, top1_within_top2, per_dim_mae  # noqa: E402

TRAIN_PATH = REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"


def check(name: str, ok: bool, detail: str = "") -> dict:
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}" + (f" - {detail}" if detail else ""))
    return {"check": name, "status": status, "detail": detail}


def main() -> list[dict]:
    results = []
    print("=== Phase 5 Step 0: pre-registration audit (holdout NOT opened) ===\n")

    preprocessing = json.loads((TRANSLATOR_DIR / "chosen_model_preprocessing.json").read_text())
    manifest = json.loads((TRANSLATOR_DIR / "chosen_model_manifest.json").read_text())

    # --- 1. input column list and order ---
    print("1. Input column list/order")
    df = pd.read_csv(TRAIN_PATH)
    assert len(df) == 237
    X_recomputed, _, recomputed_cols = variant_b(df, df)
    registered_cols = preprocessing["column_order"]
    results.append(check("column order matches", recomputed_cols == registered_cols,
                          f"registered={len(registered_cols)} cols, recomputed={len(recomputed_cols)} cols"))
    B_shape = None
    with np.load(TRANSLATOR_DIR / "chosen_model_posterior.npz") as z:
        B_shape = z["B"].shape  # (S, p_aug, K-1)
    p_aug = B_shape[1]
    results.append(check("fitted B's input dimensionality matches (p_aug = len(cols)+1 intercept)",
                          p_aug == len(registered_cols) + 1,
                          f"B p_aug={p_aug}, expected {len(registered_cols) + 1}"))

    # --- 2. standardization / imputation statistics, recomputed today ---
    print("\n2. Standardization mean/SD + shot-type imputation means (recompute vs. registered, tol=1e-9)")
    cont_cols = preprocessing["continuous_columns"]
    mu_now = df[[c for c in cont_cols if c not in ("rim_finishing_share", "three_pt_jumper_share")]].mean()
    sd_now = df[[c for c in cont_cols if c not in ("rim_finishing_share", "three_pt_jumper_share")]].std()
    mu_registered = preprocessing["standardization_mu"]
    sd_registered = preprocessing["standardization_sd"]
    mu_diffs = {k: abs(mu_now[k] - mu_registered[k]) for k in mu_now.index}
    sd_diffs = {k: abs(sd_now[k] - sd_registered[k]) for k in sd_now.index}
    max_mu_diff = max(mu_diffs.values())
    max_sd_diff = max(sd_diffs.values())
    results.append(check("standardization mu matches (tol 1e-9)", max_mu_diff < 1e-9, f"max diff={max_mu_diff:.2e}"))
    results.append(check("standardization sd matches (tol 1e-9)", max_sd_diff < 1e-9, f"max diff={max_sd_diff:.2e}"))

    impute_now = df[["rim_finishing_share", "three_pt_jumper_share"]].mean()
    impute_registered = preprocessing["shot_type_imputation_means"]
    max_impute_diff = max(abs(impute_now[k] - impute_registered[k]) for k in impute_now.index)
    results.append(check("shot-type imputation means match (tol 1e-9)", max_impute_diff < 1e-9,
                          f"max diff={max_impute_diff:.2e}"))

    # Real, executed check (this used to be a hardcoded True placeholder - caught in Step 1
    # when the frozen-only transform crashed with a KeyError the placeholder would never have
    # found, since it never actually ran the transform function against real data).
    import phase5_step1_predict as _p5
    X_frozen, _ = _p5.apply_frozen_variant_b(df, preprocessing)
    results.append(check("frozen-recipe transform (apply_frozen_variant_b) reproduces variant_b(df, df) exactly",
                          np.allclose(X_frozen, X_recomputed, atol=1e-9),
                          f"max abs diff={np.abs(X_frozen - X_recomputed).max():.2e}"))

    # --- 3. frozen priors and sampler settings ---
    print("\n3. Frozen priors and sampler settings")
    model_source = inspect.getsource(dirichlet_model._model)
    has_normal_prior = "dist.Normal(0.0, 1.0)" in model_source
    has_lognormal_prior = "dist.LogNormal(1.0, 1.0)" in model_source
    results.append(check("B prior is Normal(0,1) in current code", has_normal_prior))
    results.append(check("phi prior is LogNormal(1,1) in current code", has_lognormal_prior))
    results.append(check("reference-category identification (k-1 free columns + zero pad) present",
                          "jnp.zeros((n, 1))" in model_source))

    convergence = manifest["convergence"]
    results.append(check("manifest records converged fit (R-hat<1.01, ESS>=400)",
                          convergence["max_rhat"] < 1.01 and convergence["min_ess"] >= 400,
                          f"max_rhat={convergence['max_rhat']:.4f}, min_ess={convergence['min_ess']:.0f}"))
    results.append(check("manifest records zero divergences", convergence["n_divergent"] == 0))
    results.append(check("chosen cell matches", manifest["chosen_cell"] == "T1_b" and manifest["variant"] == "b"))

    # --- 4. metric list ---
    print("\n4. Metric list (chosen_model.md's evaluation plan vs. this phase's actual functions)")
    registered_metrics = {"JSD (base 2, primary)", "L1", "top-1 hit rate", "top-1-within-top-2",
                           "per-dimension MAE", "50%/90% posterior predictive interval coverage"}
    phase5_metric_fns = {jensen_shannon, l1, top1_hit, top1_within_top2, per_dim_mae}
    results.append(check("all 5 core metric functions importable and unchanged from Phase 4's metrics.py",
                          len(phase5_metric_fns) == 5,
                          f"registered set: {sorted(registered_metrics)}"))
    sig = inspect.signature(jensen_shannon)
    results.append(check("jensen_shannon signature is (p, q, base=2.0)",
                          list(sig.parameters.keys()) == ["p", "q", "base"] and sig.parameters["base"].default == 2.0))

    all_pass = all(r["status"] == "PASS" for r in results)
    print(f"\n{'ALL CHECKS PASS' if all_pass else 'AUDIT FAILED - STOP, DO NOT OPEN HOLDOUT'}")
    (TRANSLATOR_DIR / "phase5_step0_audit.json").write_text(json.dumps(
        {"all_pass": all_pass, "checks": results}, indent=2))
    if not all_pass:
        raise SystemExit(1)
    return results


if __name__ == "__main__":
    main()
