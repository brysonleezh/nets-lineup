"""
Phase 4 - compositional-data helpers: Smithson-Verkuilen zero-compression
and the isometric log-ratio (ILR) transform, K=8 -> 7 real dims.

ILR (not drop-one/ALR) chosen and fixed for this phase: ALR's log-ratio
against an arbitrary reference component is not isometric (distances in
ALR space don't match distances on the simplex, and the choice of which
component to drop is arbitrary); ILR uses an orthonormal basis so
Euclidean operations in the transformed space (ridge regression, k-NN
distances) correspond to a real geometry on the simplex instead of an
artifact of which component got dropped.

Run as a script to self-check the round-trip: python3 src/translator/compositional.py
"""

from __future__ import annotations

import numpy as np

K = 8


def zero_compress(Y: np.ndarray, n: int) -> np.ndarray:
    """Smithson-Verkuilen compression: y' = (y*(n-1) + 1/K) / n.

    n must be the TRAINING-FOLD size the caller is currently fitting on -
    never the full dataset size, or fold statistics leak into every other
    fold's transform.
    """
    Y = np.asarray(Y, dtype=float)
    k = Y.shape[-1]
    return (Y * (n - 1) + 1.0 / k) / n


def ilr_basis(k: int = K) -> np.ndarray:
    """(k, k-1) orthonormal Helmert-type ILR basis. Columns are orthonormal
    and each lies in the sum-to-zero hyperplane (so clr(x) @ basis is a
    valid change of basis, not merely a projection)."""
    V = np.zeros((k, k - 1))
    for j in range(k - 1):
        i = j + 1  # number of components below the pivot for this coordinate
        V[:i, j] = 1.0 / i
        V[i, j] = -1.0
        V[:i + 1, j] *= np.sqrt(i / (i + 1.0))
    # normalize each column to unit norm (the construction above already
    # gives orthogonal columns; this makes them orthonormal)
    V = V / np.linalg.norm(V, axis=0, keepdims=True)
    return V


def clr(Y: np.ndarray) -> np.ndarray:
    logY = np.log(Y)
    return logY - logY.mean(axis=-1, keepdims=True)


def ilr_transform(Y: np.ndarray, basis: np.ndarray | None = None) -> np.ndarray:
    """Y: (..., K) strictly-positive composition (already zero-compressed).
    Returns (..., K-1)."""
    if basis is None:
        basis = ilr_basis(Y.shape[-1])
    return clr(Y) @ basis


def ilr_inverse(Z: np.ndarray, basis: np.ndarray | None = None, k: int = K) -> np.ndarray:
    """Inverse ILR: (..., K-1) -> a valid K-part composition (rows sum to 1)."""
    if basis is None:
        basis = ilr_basis(k)
    clr_recovered = Z @ basis.T
    Y = np.exp(clr_recovered)
    return Y / Y.sum(axis=-1, keepdims=True)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    raw = rng.dirichlet(np.ones(K) * 0.5, size=200)
    # include some exact zeros/ones, the case zero_compress exists to handle
    raw[0] = np.eye(K)[0]
    raw[1, :] = 0
    raw[1, 3] = 1.0

    n = 237
    compressed = zero_compress(raw, n)
    assert (compressed > 0).all() and (compressed < 1).all(), "compression must land strictly inside the simplex"
    assert np.allclose(compressed.sum(axis=1), 1.0, atol=1e-10)

    basis = ilr_basis(K)
    assert basis.shape == (K, K - 1)
    assert np.allclose(basis.T @ basis, np.eye(K - 1), atol=1e-10), "basis must be orthonormal"
    assert np.allclose(basis.sum(axis=0), 0, atol=1e-10), "each basis column must lie in the sum-zero hyperplane"

    z = ilr_transform(compressed, basis)
    assert z.shape == (200, K - 1)
    back = ilr_inverse(z, basis, K)
    max_err = np.abs(back - compressed).max()
    print(f"round-trip max abs error: {max_err:.2e}")
    assert max_err < 1e-8, "ILR round-trip failed"
    print("compositional.py self-check: PASS")
