"""
Phase 2 Step 1 - projection function. Fit pool != projection set: any
z-scored feature vector (in-pool or out-of-pool - short-minute COVID-era
seasons, the 2026 rookies) can be projected onto a fitted archetype
matrix without a refit, via simplex-constrained NNLS.

Same method as the NBA-side basis's own project() in
src/step1_archetype_model.py (augmented-row NNLS: min ||x - B.T @ p||^2
s.t. p >= 0, sum(p) = 1, approximated by appending a heavily-weighted
"sum to simplex_weight" row rather than a hard equality constraint, then
renormalizing) - reused for methodological consistency with the already-
validated NBA basis, not reimplemented from scratch.

Example
-------
>>> import numpy as np
>>> from college_model.project import project_recipe
>>> basis = np.load("data/college/model/basis.npz")
>>> z = np.array([0.5, -0.2, 1.1, 0.0, -0.8, 0.3, 0.1, -0.4, 0.2, 0.6, -0.1, 0.0])
>>> alpha = project_recipe(z, basis["basis"])
>>> alpha.sum()
1.0000000000000002
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import nnls


def project_recipe(z_features: np.ndarray, basis: np.ndarray, simplex_weight: float = 1e4) -> np.ndarray:
    """z_features: (p,) or (n, p) z-scored feature vector(s), same feature
    order as `basis`'s columns. basis: (k, p) fitted archetype matrix.
    Returns (k,) or (n, k) simplex weights (non-negative, sum to 1)."""
    B = np.asarray(basis, dtype=float)
    k, p = B.shape

    z = np.asarray(z_features, dtype=float)
    single = z.ndim == 1
    if single:
        z = z[None, :]
    if z.shape[1] != p:
        raise ValueError(f"z_features has {z.shape[1]} features, basis expects {p}")

    A = np.vstack([B.T, np.full((1, k), simplex_weight)])  # (p+1, k)

    P = np.empty((len(z), k))
    for i, x in enumerate(z):
        b = np.concatenate([x, [simplex_weight]])
        P[i], _ = nnls(A, b)

    row_sums = P.sum(axis=1, keepdims=True)
    if (row_sums < 1e-9).any():
        raise RuntimeError("degenerate projection: some rows got all-zero weights")
    P = P / row_sums

    return P[0] if single else P


def project_dataframe(df, basis: np.ndarray, feature_cols: list[str]) -> np.ndarray:
    """Convenience wrapper: project every row of a DataFrame's z-columns."""
    Z = df[feature_cols].astype(float).values
    return project_recipe(Z, basis)
