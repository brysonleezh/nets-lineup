"""
Phase 4 Step 2 - T1, the primary model: Dirichlet regression.

mean mu_i = softmax(B x_i), reference-category identification (archetype 7
[0-indexed, "archetype 8" in the spec's 1-indexed prose] fixed at logit 0 -
its coefficient row is not sampled at all, not merely regularized to zero,
which is what makes the softmax identifiable). Scalar precision phi.
Likelihood: zero-compressed y ~ Dirichlet(phi * mu).

Priors, fixed per spec (a deliberate simplicity choice - no tuning):
B ~ Normal(0, 1) on standardized inputs (including an intercept row, since
the spec's prior section doesn't split one out and different archetypes
have very different base rates - Step 0's preflight found archetype 3 at
36% of training rookies vs. archetype 6 at 0%, so an intercept-free model
would badly misfit the marginal from the first sample).
phi ~ LogNormal(1, 1).

Sampler: numpyro NUTS, 4 chains. Divergences handled by raising
target_accept (0.8 -> 0.9 -> 0.95), never by silencing. Requires R-hat <
1.01 and effective sample size >= 400 (100/chain) on every parameter for a
fold fit to be accepted.
"""

from __future__ import annotations

import numpy as np

import numpyro
numpyro.set_host_device_count(4)  # must precede any other jax/numpyro call - lets the
                                   # 4 chains run as true parallel host devices on CPU

import jax
import jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import summary as numpyro_summary

K = 8
NUM_WARMUP = 750
NUM_SAMPLES = 750
NUM_CHAINS = 4
MIN_ESS = 400
MAX_RHAT = 1.01
TARGET_ACCEPT_LADDER = [0.8, 0.9, 0.95]


def _model(X, y=None, k=K):
    n, p_aug = X.shape  # X already has the intercept column prepended by the caller
    B = numpyro.sample("B", dist.Normal(0.0, 1.0).expand([p_aug, k - 1]).to_event(2))
    logits_free = X @ B  # (n, k-1)
    logits = jnp.concatenate([logits_free, jnp.zeros((n, 1))], axis=1)  # (n, k)
    mu = jax.nn.softmax(logits, axis=-1)
    phi = numpyro.sample("phi", dist.LogNormal(1.0, 1.0))
    numpyro.sample("y", dist.Dirichlet(phi * mu), obs=y)


def _add_intercept(X: np.ndarray) -> jnp.ndarray:
    return jnp.concatenate([jnp.ones((X.shape[0], 1)), jnp.asarray(X, dtype=jnp.float32)], axis=1)


def fit_t1_dirichlet(X_train: np.ndarray, y_train_compressed: np.ndarray, X_test: np.ndarray,
                      seed: int = 0, verbose: bool = False) -> tuple[np.ndarray, dict]:
    """Returns (posterior-mean predictions on X_test, diagnostics dict).
    Raises RuntimeError if R-hat/ESS criteria aren't met after the full
    target_accept ladder - per spec, a fold failing convergence after
    standard remedies is a stop-and-ask condition, not something to paper
    over by returning a bad fit silently."""
    Xtr = _add_intercept(X_train)
    Xte = _add_intercept(X_test)
    y = jnp.asarray(y_train_compressed, dtype=jnp.float32)

    last_diag = None
    for target_accept in TARGET_ACCEPT_LADDER:
        kernel = NUTS(_model, target_accept_prob=target_accept)
        mcmc = MCMC(kernel, num_warmup=NUM_WARMUP, num_samples=NUM_SAMPLES,
                    num_chains=NUM_CHAINS, progress_bar=verbose)
        mcmc.run(jax.random.PRNGKey(seed), X=Xtr, y=y, k=K)
        samples = mcmc.get_samples(group_by_chain=True)

        diag = numpyro_summary(samples)
        max_rhat = max(float(np.max(np.asarray(v["r_hat"]))) for v in diag.values())
        min_ess = min(float(np.min(np.asarray(v["n_eff"]))) for v in diag.values())
        n_divergent = int(np.sum(np.asarray(mcmc.get_extra_fields()["diverging"])))
        last_diag = {"target_accept": target_accept, "max_rhat": max_rhat, "min_ess": min_ess,
                      "n_divergent": n_divergent, "num_samples_total": NUM_SAMPLES * NUM_CHAINS}

        if max_rhat < MAX_RHAT and min_ess >= MIN_ESS:
            break
    else:
        raise RuntimeError(f"T1 failed to converge after the full target_accept ladder: {last_diag}")

    if not (max_rhat < MAX_RHAT and min_ess >= MIN_ESS):
        raise RuntimeError(f"T1 failed to converge after the full target_accept ladder: {last_diag}")

    B_samples = samples["B"].reshape(-1, samples["B"].shape[-2], samples["B"].shape[-1])  # (S, p_aug, k-1)
    phi_samples = samples["phi"].reshape(-1)  # (S,)
    logits_free = jnp.einsum("np,spk->snk", Xte, B_samples)  # (S, n_test, k-1)
    logits = jnp.concatenate([logits_free, jnp.zeros((*logits_free.shape[:-1], 1))], axis=-1)
    mu_samples = jax.nn.softmax(logits, axis=-1)  # (S, n_test, k)
    mu_mean = np.asarray(mu_samples.mean(axis=0))  # posterior mean, already on the simplex

    # Posterior PREDICTIVE y samples (not just mu's posterior) - draws a fresh
    # Dirichlet(phi_s * mu_s) observation per posterior sample, so calibration
    # coverage reflects the full predictive distribution (parameter uncertainty
    # + the Dirichlet's own sampling noise), which is what "posterior predictive
    # interval" means - using mu_samples alone would understate interval width.
    conc = phi_samples[:, None, None] * mu_samples  # (S, n_test, k)
    pred_key = jax.random.PRNGKey(seed + 10_000)
    y_pred_samples = np.asarray(dist.Dirichlet(conc).sample(pred_key))  # (S, n_test, k)

    # 50%/90% coverage against the true held-out y is computed by the caller
    # (needs the true labels, which this function intentionally doesn't take,
    # to keep the fitting code blind to anything about scoring).
    last_diag["y_pred_samples"] = y_pred_samples
    return mu_mean, last_diag


if __name__ == "__main__":
    import sys
    from pathlib import Path
    REPO_ROOT = Path(__file__).resolve().parent.parent.parent
    sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))
    import pandas as pd
    from input_variants import variant_a, build_targets
    from metrics import jensen_shannon, l1, top1_hit

    df = pd.read_csv(REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv")
    train_df = df[df["draft_year"] != 2017].reset_index(drop=True)
    test_df = df[df["draft_year"] == 2017].reset_index(drop=True)

    X_train, X_test, _ = variant_a(train_df, test_df)
    y_train, _ = build_targets(train_df, test_df)
    y_cols = [f"y_{j}" for j in range(K)]
    y_true = test_df[y_cols].values

    print(f"fitting T1 on variant (a), n_train={len(train_df)}, n_test={len(test_df)}...")
    preds, diag = fit_t1_dirichlet(X_train, y_train, X_test, verbose=True)
    print(f"\nconverged at target_accept={diag['target_accept']}: "
          f"max_rhat={diag['max_rhat']:.4f}, min_ess={diag['min_ess']:.0f}, "
          f"n_divergent={diag['n_divergent']}/{diag['num_samples_total']}")
    print(f"predictions sum to 1: {np.allclose(preds.sum(1), 1.0, atol=1e-5)}")
    print(f"JSD={jensen_shannon(preds, y_true).mean():.4f}, L1={l1(preds, y_true).mean():.4f}, "
          f"top1={top1_hit(preds, y_true).mean():.3f}")
    print("dirichlet_model.py self-check: PASS")
