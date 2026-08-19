"""Phase 4 - CV metric functions: JSD (primary), L1, top-1 hit rate,
top-1-within-top-2, per-dimension MAE. Operate on raw (uncompressed)
K-simplex vectors - predictions and true rookie recipes both."""

from __future__ import annotations

import numpy as np

K = 8


def jensen_shannon(p: np.ndarray, q: np.ndarray, base: float = 2.0) -> np.ndarray:
    """Row-wise JS divergence between two (n, K) simplex matrices. base=2 -> bits, range [0,1]."""
    p = np.clip(p, 1e-12, None)
    q = np.clip(q, 1e-12, None)
    p = p / p.sum(axis=-1, keepdims=True)
    q = q / q.sum(axis=-1, keepdims=True)
    m = 0.5 * (p + q)

    def kl(a, b):
        return (a * (np.log(a) - np.log(b))).sum(axis=-1)

    js_nats = 0.5 * kl(p, m) + 0.5 * kl(q, m)
    return js_nats / np.log(base)


def l1(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return np.abs(p - q).sum(axis=-1)


def top1_hit(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    return (p.argmax(axis=-1) == q.argmax(axis=-1)).astype(float)


def top1_within_top2(p_true: np.ndarray, q_pred: np.ndarray) -> np.ndarray:
    """Is the TRUE top-1 archetype among the PREDICTED top-2?"""
    true_top1 = p_true.argmax(axis=-1)
    pred_top2 = np.argsort(-q_pred, axis=-1)[:, :2]
    return np.array([true_top1[i] in pred_top2[i] for i in range(len(true_top1))], dtype=float)


def per_dim_mae(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(K,) mean absolute error per archetype dimension."""
    return np.abs(p - q).mean(axis=0)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    p = rng.dirichlet(np.ones(K), size=50)
    assert np.allclose(jensen_shannon(p, p), 0, atol=1e-8), "JSD(p,p) must be 0"
    q = np.roll(p, 1, axis=1)
    js = jensen_shannon(p, q)
    assert (js >= 0).all() and (js <= 1 + 1e-9).all(), "base-2 JSD must be in [0,1]"

    a = np.array([[1.0, 0, 0], [0, 1.0, 0]])
    b = np.array([[1.0, 0, 0], [1.0, 0, 0]])
    assert np.allclose(l1(a, b), [0, 2.0])
    assert np.allclose(top1_hit(a, b), [1.0, 0.0])
    print("metrics.py self-check: PASS")
