"""
Phase 4 Step 1 - pre-registered input variants. No additions after this
step per the phase's own hard boundary on pre-registration discipline.

Variant (a): 8 c_alpha dims, zero-compressed then ILR-transformed -> 7 dims.
Variant (b): 12 shared z-features + pick_overall + log_pick + age_at_draft +
    years_in_college + conf_tier + PORPAG + rim_finishing_share +
    three_pt_jumper_share + 1 missing-indicator = 21 dims. Shot-type NaNs
    imputed with the TRAINING FOLD's mean (never the full dataset's).
Variant (c): concat(a, b) = 28 dims.

All continuous columns (everything except conf_tier and the missing
indicator) are standardized on training-fold mean/SD - fold-aware by
construction, since every function here takes a separate train_df/test_df
and fits statistics on train_df only.

Not included, per owner decision: position, college team/conference
identity beyond conf_tier.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from compositional import ilr_basis, ilr_transform, zero_compress  # noqa: E402

K = 8
C_COLS = [f"c_alpha_{j}" for j in range(K)]
Y_COLS = [f"y_{j}" for j in range(K)]

Z_FEATURE_COLS = ["PLAYER_HEIGHT_INCHES_z", "PTS_PER_100_z", "TS%_z", "USG%_z", "AST%_z",
                   "TOV%_z", "STL%_z", "BLK%_z", "TRB%_z", "FTr_z",
                   "% of FG Ast'd_2P_z", "% of FG Ast'd_3P_z"]
SHOT_TYPE_COLS = ["rim_finishing_share", "three_pt_jumper_share"]
VARIANT_B_RAW_COLS = (Z_FEATURE_COLS + ["overall", "log_pick", "age_at_draft",
                       "years_in_college", "PORPAG"] + SHOT_TYPE_COLS)
# 'overall' here plays the role of the spec's "pick_overall" (anchors.csv's own
# column for that is literally named 'overall' - see anchor_finalize.py's id_cols)


def variant_a(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    n = len(train_df)
    basis = ilr_basis(K)

    train_c = zero_compress(train_df[C_COLS].values, n)
    test_c = zero_compress(test_df[C_COLS].values, n)

    X_train = ilr_transform(train_c, basis)
    X_test = ilr_transform(test_c, basis)
    names = [f"ilr_c_{j}" for j in range(K - 1)]
    return X_train, X_test, names


def variant_b(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    train = train_df[VARIANT_B_RAW_COLS].copy()
    test = test_df[VARIANT_B_RAW_COLS].copy()

    # missing indicator (one shared column: 1 if EITHER shot-type feature is missing)
    train_missing = train[SHOT_TYPE_COLS].isna().any(axis=1).astype(float).values
    test_missing = test[SHOT_TYPE_COLS].isna().any(axis=1).astype(float).values

    # training-fold-mean imputation
    fold_means = train[SHOT_TYPE_COLS].mean()
    train[SHOT_TYPE_COLS] = train[SHOT_TYPE_COLS].fillna(fold_means)
    test[SHOT_TYPE_COLS] = test[SHOT_TYPE_COLS].fillna(fold_means)

    conf_tier_train = train_df["conf_tier"].astype(float).values
    conf_tier_test = test_df["conf_tier"].astype(float).values

    continuous_cols = [c for c in VARIANT_B_RAW_COLS]
    mu = train[continuous_cols].mean()
    sd = train[continuous_cols].std()
    sd = sd.replace(0, 1.0)  # guard against a degenerate fold with zero variance in some column

    train_z = ((train[continuous_cols] - mu) / sd).values
    test_z = ((test[continuous_cols] - mu) / sd).values

    X_train = np.column_stack([train_z, conf_tier_train, train_missing])
    X_test = np.column_stack([test_z, conf_tier_test, test_missing])
    names = continuous_cols + ["conf_tier", "shot_type_missing"]
    assert X_train.shape[1] == 21, f"expected 21 dims for variant b, got {X_train.shape[1]}"
    return X_train, X_test, names


def variant_c(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[str]]:
    a_train, a_test, a_names = variant_a(train_df, test_df)
    b_train, b_test, b_names = variant_b(train_df, test_df)
    X_train = np.column_stack([a_train, b_train])
    X_test = np.column_stack([a_test, b_test])
    assert X_train.shape[1] == 28, f"expected 28 dims for variant c, got {X_train.shape[1]}"
    return X_train, X_test, a_names + b_names


VARIANTS = {"a": variant_a, "b": variant_b, "c": variant_c}


def build_targets(train_df: pd.DataFrame, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Zero-compressed rookie recipes (the Dirichlet-likelihood-ready targets).
    Compressed with the TRAINING fold's n, applied identically to test rows -
    same coordinate system, no test information used in the compression itself."""
    n = len(train_df)
    y_train = zero_compress(train_df[Y_COLS].values, n)
    y_test = zero_compress(test_df[Y_COLS].values, n)
    return y_train, y_test


if __name__ == "__main__":
    df = pd.read_csv(REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv")
    train_df = df[df["draft_year"] != 2017]
    test_df = df[df["draft_year"] == 2017]
    for name, fn in VARIANTS.items():
        Xtr, Xte, cols = fn(train_df, test_df)
        print(f"variant {name}: train {Xtr.shape}, test {Xte.shape}, {len(cols)} named columns")
        assert not np.isnan(Xtr).any() and not np.isnan(Xte).any(), f"variant {name} produced NaNs"
    ytr, yte = build_targets(train_df, test_df)
    print(f"targets: train {ytr.shape}, test {yte.shape}, "
          f"row sums train min/max = {ytr.sum(1).min():.6f}/{ytr.sum(1).max():.6f}")
    print("input_variants.py self-check: PASS")
