"""
Phase 4 Step 2 - models T2-T5 (T1 Dirichlet regression lives in dirichlet_model.py,
gated on numpyro being installed).

T2: per-dimension ridge on raw y, clip-at-zero + renormalize. Reference model
    only - renormalization breaks the probabilistic reading.
T3: k-NN in standardized variant-(c) space. Prediction = mean of neighbors'
    raw rookie recipes (already a valid simplex - no clipping needed).
T4: Hungarian identity-mapping baseline - the bar the project must beat.
T5: naive floors - (i) training-fold global mean, (ii) height-tercile mean.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))
sys.path.insert(0, str(REPO_ROOT / "src"))

from input_variants import Z_FEATURE_COLS, variant_c  # noqa: E402

K = 8
C_COLS = [f"c_alpha_{j}" for j in range(K)]
Y_COLS = [f"y_{j}" for j in range(K)]
LAMBDA_GRID = np.logspace(-3, 3, 13)
KNN_GRID = [5, 8]


def _clip_renorm(P: np.ndarray) -> np.ndarray:
    P = np.clip(P, 0, None)
    row_sums = P.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < 1e-12, 1.0, row_sums)  # guard: all-zero row -> uniform below
    out = P / row_sums
    dead = (P.sum(axis=1) < 1e-12)
    if dead.any():
        out[dead] = 1.0 / K
    return out


def _inner_class_folds(train_df: pd.DataFrame, n_splits: int = 4, seed: int = 0):
    """Nested CV folds for hyperparameter selection, built from the OUTER
    training fold's own rows only (never touches the outer test class or
    the holdout) - a plain row-level KFold, not another leave-class-out
    split, since the inner selection just needs an unbiased λ/k estimate,
    not another class-generalization test."""
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(train_df))


# --- T2: per-dimension ridge -------------------------------------------

def fit_t2_ridge(train_df: pd.DataFrame, test_df: pd.DataFrame, variant_fn) -> tuple[np.ndarray, dict]:
    idx_pairs = _inner_class_folds(train_df)
    best_lambda, best_score = None, np.inf
    for lam in LAMBDA_GRID:
        errs = []
        for inner_tr_idx, inner_val_idx in idx_pairs:
            inner_tr = train_df.iloc[inner_tr_idx]
            inner_val = train_df.iloc[inner_val_idx]
            X_tr, X_val, _ = variant_fn(inner_tr, inner_val)
            y_tr = inner_tr[Y_COLS].values
            y_val = inner_val[Y_COLS].values
            preds = np.column_stack([
                Ridge(alpha=lam).fit(X_tr, y_tr[:, j]).predict(X_val) for j in range(K)
            ])
            preds = _clip_renorm(preds)
            errs.append(np.abs(preds - y_val).sum(axis=1).mean())
        score = float(np.mean(errs))
        if score < best_score:
            best_score, best_lambda = score, lam

    X_train, X_test, _ = variant_fn(train_df, test_df)
    y_train = train_df[Y_COLS].values
    preds = np.column_stack([
        Ridge(alpha=best_lambda).fit(X_train, y_train[:, j]).predict(X_test) for j in range(K)
    ])
    preds = _clip_renorm(preds)
    return preds, {"lambda": float(best_lambda), "inner_cv_l1": best_score}


# --- T3: k-NN ------------------------------------------------------------

def fit_t3_knn(train_df: pd.DataFrame, test_df: pd.DataFrame, variant_fn=variant_c,
                return_neighbors_for: list[int] | None = None) -> tuple[np.ndarray, dict, list]:
    """k-NN in the standardized space of whichever variant_fn is passed
    (defaults to variant (c) for standalone/ad-hoc calls outside the
    harness). Step 3's grid formula ("3 variants x {T1, T2, T3} + T4 + T5i
    + T5ii = 12 cells") only arithmetically resolves to 12 if T3 is run
    across all 3 variants like T1/T2 (3+3+1+1+1+1=10 otherwise, contra
    T2's own bullet's literal "in the standardized variant-(c) space"
    phrasing) - the CV harness always passes an explicit variant_fn."""
    idx_pairs = _inner_class_folds(train_df)
    best_k, best_score = None, np.inf
    for k in KNN_GRID:
        errs = []
        for inner_tr_idx, inner_val_idx in idx_pairs:
            inner_tr = train_df.iloc[inner_tr_idx]
            inner_val = train_df.iloc[inner_val_idx]
            X_tr, X_val, _ = variant_fn(inner_tr, inner_val)
            y_tr = inner_tr[Y_COLS].values
            y_val = inner_val[Y_COLS].values
            preds = _knn_predict(X_tr, y_tr, X_val, k)
            errs.append(np.abs(preds - y_val).sum(axis=1).mean())
        score = float(np.mean(errs))
        if score < best_score:
            best_score, best_k = score, k

    X_train, X_test, _ = variant_fn(train_df, test_df)
    y_train = train_df[Y_COLS].values
    preds = _knn_predict(X_train, y_train, X_test, best_k)

    neighbor_tables = []
    if return_neighbors_for:
        dists = np.linalg.norm(X_test[:, None, :] - X_train[None, :, :], axis=2)
        for i in return_neighbors_for:
            order = np.argsort(dists[i])[:best_k]
            neighbor_tables.append({
                "test_row": i,
                "neighbor_names": train_df.iloc[order]["player_name"].tolist(),
                "neighbor_draft_years": train_df.iloc[order]["draft_year"].tolist(),
                "distances": dists[i, order].tolist(),
            })
    return preds, {"k": best_k, "inner_cv_l1": best_score}, neighbor_tables


def _knn_predict(X_train, y_train, X_test, k):
    dists = np.linalg.norm(X_test[:, None, :] - X_train[None, :, :], axis=2)
    order = np.argsort(dists, axis=1)[:, :k]
    return np.array([y_train[order[i]].mean(axis=0) for i in range(len(X_test))])


# --- T4: Hungarian identity mapping --------------------------------------

_SHARED_RAW_NAMES = [c.replace("_z", "") for c in Z_FEATURE_COLS]


def hungarian_assignment() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (assignment array college_idx->nba_idx, cosine matrix (8,8),
    college archetype basis used)."""
    college_basis = np.load(REPO_ROOT / "data" / "college" / "model" / "k8_frozen_basis.npz")["basis"]  # (8,12)
    feature_order = __import__("json").loads(
        (REPO_ROOT / "data" / "basis_2025_26" / "feature_order.json").read_text())["feature_columns"]
    assert all(name in feature_order for name in _SHARED_RAW_NAMES), \
        "not all 12 shared coordinates exist in the NBA 29-feature order"
    idx = [feature_order.index(name) for name in _SHARED_RAW_NAMES]
    nba_basis_full = np.load(REPO_ROOT / "data" / "basis_2025_26" / "basis.npz")["basis"]  # (8,29)
    nba_basis_shared = nba_basis_full[:, idx]  # (8,12), same column order as college_basis

    def cos_sim_matrix(A, B):
        An = A / np.linalg.norm(A, axis=1, keepdims=True)
        Bn = B / np.linalg.norm(B, axis=1, keepdims=True)
        return An @ Bn.T

    cos = cos_sim_matrix(college_basis, nba_basis_shared)  # (8 college, 8 nba)
    row_idx, col_idx = linear_sum_assignment(-cos)  # maximize cosine -> minimize negative
    assignment = np.empty(K, dtype=int)
    assignment[row_idx] = col_idx
    return assignment, cos, college_basis


def fit_t4_hungarian(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      assignment: np.ndarray) -> np.ndarray:
    c = test_df[C_COLS].values  # (n, 8), college-archetype simplex
    preds = np.zeros_like(c)
    for college_j in range(K):
        preds[:, assignment[college_j]] = c[:, college_j]
    return preds


# --- T5: naive floors -----------------------------------------------------

def fit_t5_global_mean(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    mean_recipe = train_df[Y_COLS].values.mean(axis=0)
    mean_recipe = mean_recipe / mean_recipe.sum()
    return np.tile(mean_recipe, (len(test_df), 1))


def fit_t5_height_tercile(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    cutpoints = train_df["PLAYER_HEIGHT_INCHES"].quantile([1 / 3, 2 / 3]).values
    tercile_means = []
    for lo, hi in [(-np.inf, cutpoints[0]), (cutpoints[0], cutpoints[1]), (cutpoints[1], np.inf)]:
        mask = (train_df["PLAYER_HEIGHT_INCHES"] > lo) & (train_df["PLAYER_HEIGHT_INCHES"] <= hi)
        m = train_df.loc[mask, Y_COLS].values.mean(axis=0)
        tercile_means.append(m / m.sum())
    tercile_means = np.array(tercile_means)

    def which_tercile(h):
        if h <= cutpoints[0]:
            return 0
        if h <= cutpoints[1]:
            return 1
        return 2

    idx = test_df["PLAYER_HEIGHT_INCHES"].apply(which_tercile).values
    return tercile_means[idx]
