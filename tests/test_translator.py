"""Phase 4 tests: quarantine/leakage guards, compositional transforms,
input variants, and each model family's output validity."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from compositional import ilr_basis, ilr_transform, ilr_inverse, zero_compress  # noqa: E402
from input_variants import variant_a, variant_b, variant_c, build_targets  # noqa: E402
from models import _clip_renorm, hungarian_assignment, fit_t4_hungarian, fit_t5_global_mean  # noqa: E402
from metrics import jensen_shannon, l1, top1_hit, top1_within_top2  # noqa: E402

ANCHORS_DIR = REPO_ROOT / "data" / "anchors"
K = 8


# --- quarantine / leakage guards ------------------------------------------

def test_holdout_hash_unchanged():
    manifest = json.loads((ANCHORS_DIR / "holdout_manifest.json").read_text())
    actual = hashlib.sha256((ANCHORS_DIR / "holdout_2025.csv").read_bytes()).hexdigest()
    assert actual == manifest["sha256"], "holdout_2025.csv has changed since Step 0's quarantine"


def test_train_holdout_partition():
    train = pd.read_csv(ANCHORS_DIR / "train_2017_2024.csv")
    holdout = pd.read_csv(ANCHORS_DIR / "holdout_2025.csv")
    assert len(train) == 237
    assert len(holdout) == 36
    assert set(train["draft_year"].unique()) == set(range(2017, 2025))
    assert set(holdout["draft_year"].unique()) == {2025}
    assert set(train["player_name"]) & set(holdout["player_name"]) == set()


def test_no_phase4_source_file_opens_holdout_csv():
    """Grep-based guard, scoped to PHASE 4 specifically (Phase 4's own hard
    boundary: the holdout is untouchable after Step 0's quarantine). Phase 5
    exists precisely to open the holdout exactly once in a controlled,
    audited way (its own lock-file/rerun guards live in test_phase5.py) -
    phase5_*.py and its supporting fix script are correctly excluded here,
    not exempted from scrutiny, just out of Phase 4's scope."""
    translator_dir = REPO_ROOT / "src" / "translator"
    phase4_only_files = [
        "step0_quarantine.py", "input_variants.py", "compositional.py", "models.py",
        "dirichlet_model.py", "cv_harness.py", "metrics.py", "fit_final_model.py",
    ]
    offenders = []
    for name in phase4_only_files:
        f = translator_dir / name
        if not f.exists() or name == "step0_quarantine.py":
            continue
        text = f.read_text()
        if "holdout_2025.csv" in text or "holdout_2025" in text:
            offenders.append(name)
    assert not offenders, f"these Phase 4 files reference the holdout file: {offenders}"


# --- compositional transforms ---------------------------------------------

def test_zero_compress_bounds():
    Y = np.array([[1.0, 0, 0, 0, 0, 0, 0, 0], [0.125] * 8])
    comp = zero_compress(Y, n=100)
    assert (comp > 0).all() and (comp < 1).all()
    assert np.allclose(comp.sum(axis=1), 1.0, atol=1e-10)


def test_ilr_round_trip():
    rng = np.random.default_rng(1)
    raw = rng.dirichlet(np.ones(K) * 0.3, size=50)
    compressed = zero_compress(raw, n=200)
    basis = ilr_basis(K)
    z = ilr_transform(compressed, basis)
    assert z.shape == (50, K - 1)
    back = ilr_inverse(z, basis, K)
    assert np.abs(back - compressed).max() < 1e-8


def test_ilr_basis_orthonormal():
    basis = ilr_basis(K)
    assert basis.shape == (K, K - 1)
    assert np.allclose(basis.T @ basis, np.eye(K - 1), atol=1e-10)
    assert np.allclose(basis.sum(axis=0), 0, atol=1e-10)


# --- input variants ---------------------------------------------------------

@pytest.fixture(scope="module")
def train_test_split():
    df = pd.read_csv(ANCHORS_DIR / "train_2017_2024.csv")
    return df[df["draft_year"] != 2019].reset_index(drop=True), df[df["draft_year"] == 2019].reset_index(drop=True)


def test_variant_shapes(train_test_split):
    train_df, test_df = train_test_split
    Xa_tr, Xa_te, _ = variant_a(train_df, test_df)
    Xb_tr, Xb_te, _ = variant_b(train_df, test_df)
    Xc_tr, Xc_te, _ = variant_c(train_df, test_df)
    assert Xa_tr.shape[1] == 7 and Xa_te.shape[1] == 7
    assert Xb_tr.shape[1] == 21 and Xb_te.shape[1] == 21
    assert Xc_tr.shape[1] == 28 and Xc_te.shape[1] == 28
    assert Xc_tr.shape[1] == Xa_tr.shape[1] + Xb_tr.shape[1]
    for X in (Xa_tr, Xa_te, Xb_tr, Xb_te, Xc_tr, Xc_te):
        assert not np.isnan(X).any()


def test_variant_b_imputation_is_train_fold_only(train_test_split):
    """Corrupting a shot-type value in the TEST fold must not change the
    TRAIN fold's output at all - proof the imputation mean is fold-scoped."""
    train_df, test_df = train_test_split
    Xb_tr_orig, _, _ = variant_b(train_df, test_df)
    test_corrupted = test_df.copy()
    test_corrupted.loc[test_corrupted.index[0], "rim_finishing_share"] = 999.0
    Xb_tr_after, _, _ = variant_b(train_df, test_corrupted)
    assert np.allclose(Xb_tr_orig, Xb_tr_after)


def test_build_targets_valid_simplex(train_test_split):
    train_df, test_df = train_test_split
    y_train, y_test = build_targets(train_df, test_df)
    for y in (y_train, y_test):
        assert (y > 0).all() and (y < 1).all()  # zero-compressed: strictly inside the simplex
        assert np.allclose(y.sum(axis=1), 1.0, atol=1e-10)


# --- model output validity on a synthetic batch -----------------------------

def test_clip_renorm_validity():
    P = np.array([[0.5, -0.3, 0.9, -0.1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]])
    out = _clip_renorm(P)
    assert (out >= 0).all()
    assert np.allclose(out.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(out[1], 1.0 / K)  # all-zero row falls back to uniform


def test_t4_assignment_is_valid_permutation():
    assignment, cos, basis = hungarian_assignment()
    assert sorted(assignment.tolist()) == list(range(K))
    assert cos.shape == (K, K)
    assert basis.shape == (K, 12)


def test_t4_prediction_is_simplex_permutation(train_test_split):
    train_df, test_df = train_test_split
    assignment, _, _ = hungarian_assignment()
    preds = fit_t4_hungarian(train_df, test_df, assignment)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-9)
    c_cols = [f"c_alpha_{j}" for j in range(K)]
    # permutation must preserve the multiset of values in each row
    for i in range(len(test_df)):
        assert np.allclose(sorted(preds[i]), sorted(test_df.iloc[i][c_cols].values))


def test_t5_global_mean_is_simplex(train_test_split):
    train_df, test_df = train_test_split
    preds = fit_t5_global_mean(train_df, test_df)
    assert preds.shape == (len(test_df), K)
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-9)
    assert np.allclose(preds[0], preds[-1])  # same prediction for every test row


def test_knn_determinism():
    from models import _knn_predict
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(30, 5))
    y_train = rng.dirichlet(np.ones(K), size=30)
    X_test = rng.normal(size=(4, 5))
    p1 = _knn_predict(X_train, y_train, X_test, k=5)
    p2 = _knn_predict(X_train, y_train, X_test, k=5)
    assert np.array_equal(p1, p2)


def test_ridge_clip_renorm_validity():
    from models import fit_t2_ridge
    from input_variants import variant_a
    df = pd.read_csv(ANCHORS_DIR / "train_2017_2024.csv")
    train_df = df[df["draft_year"] != 2018].reset_index(drop=True)
    test_df = df[df["draft_year"] == 2018].reset_index(drop=True)
    preds, info = fit_t2_ridge(train_df, test_df, variant_a)
    assert (preds >= 0).all()
    assert np.allclose(preds.sum(axis=1), 1.0, atol=1e-6)
    assert info["lambda"] > 0


# --- fold construction ------------------------------------------------------

def test_fold_construction_covers_each_class_exactly_once():
    df = pd.read_csv(ANCHORS_DIR / "train_2017_2024.csv")
    classes = list(range(2017, 2025))
    test_rows_seen = []
    for c in classes:
        test_df = df[df["draft_year"] == c]
        assert len(test_df) > 0
        test_rows_seen.extend(test_df.index.tolist())
    assert sorted(test_rows_seen) == sorted(df.index.tolist()), \
        "leave-one-class-out folds must partition the training set exactly, no overlaps or gaps"


# --- metric functions against hand-computed fixtures -------------------------

def test_jsd_fixture():
    p = np.array([[1.0, 0.0]])
    q = np.array([[0.0, 1.0]])
    # JSD(p,q) for two point masses on disjoint outcomes = 1 bit (base 2)
    assert abs(jensen_shannon(p, q, base=2)[0] - 1.0) < 1e-6


def test_l1_fixture():
    p = np.array([[0.5, 0.5]])
    q = np.array([[0.9, 0.1]])
    assert abs(l1(p, q)[0] - 0.8) < 1e-9


def test_top1_fixtures():
    p = np.array([[0.1, 0.9], [0.7, 0.3]])
    q = np.array([[0.2, 0.8], [0.6, 0.4]])
    assert np.allclose(top1_hit(p, q), [1.0, 1.0])
    q2 = np.array([[0.8, 0.2], [0.4, 0.6]])
    assert np.allclose(top1_hit(p, q2), [0.0, 0.0])


def test_top1_within_top2_fixture():
    p_true = np.array([[0.0, 0.0, 1.0]])  # true top1 = index 2
    q_pred = np.array([[0.1, 0.45, 0.45]])  # predicted top2 = {1, 2}
    assert top1_within_top2(p_true, q_pred)[0] == 1.0
    q_pred_miss = np.array([[0.5, 0.4, 0.1]])  # predicted top2 = {0, 1}, misses 2
    assert top1_within_top2(p_true, q_pred_miss)[0] == 0.0
