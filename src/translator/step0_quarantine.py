"""
Phase 4 Step 0 - quarantine & preflight.

Splits data/anchors/anchors.csv into train_2017_2024.csv (237 rows) and
holdout_2025.csv (36 rows). Records the holdout file's sha256 so a later
test can assert it is byte-identical at phase end - the hard boundary is
that no Phase 4 code path opens holdout_2025.csv after this point.

Also runs the required preflight scans: Y argmax distribution (the naive
floor's implied top-1 hit rate), college-recipe argmax distribution,
age_at_draft x years_in_college joint plausibility scan, missingness table,
and a simplex validity assert on both c_alpha and y.

Run: python3 src/translator/step0_quarantine.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

ANCHORS_PATH = REPO_ROOT / "data" / "anchors" / "anchors.csv"
OUT_DIR = REPO_ROOT / "data" / "anchors"
K = 8


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> None:
    anchors = pd.read_csv(ANCHORS_PATH)
    assert len(anchors) == 273
    assert anchors["draft_year"].max() == 2025

    train = anchors[anchors["draft_year"] <= 2024].copy()
    holdout = anchors[anchors["draft_year"] == 2025].copy()
    assert len(train) == 237, f"expected 237 train rows, got {len(train)}"
    assert len(holdout) == 36, f"expected 36 holdout rows, got {len(holdout)}"

    train_path = OUT_DIR / "train_2017_2024.csv"
    holdout_path = OUT_DIR / "holdout_2025.csv"
    train.to_csv(train_path, index=False)
    holdout.to_csv(holdout_path, index=False)

    manifest = {
        "holdout_file": "holdout_2025.csv",
        "sha256": sha256_of(holdout_path),
        "n_rows": len(holdout),
        "created": "2026-08-13",
        "note": "No Phase 4 code path may open this file. A pytest asserts "
                "this hash is unchanged at phase end.",
    }
    (OUT_DIR / "holdout_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Quarantined: train={len(train)} rows -> {train_path.name}, "
          f"holdout={len(holdout)} rows -> {holdout_path.name}")
    print(f"holdout sha256: {manifest['sha256']}")

    # ---- preflight scans (train only, 237 rows) ----
    report = {}

    y_cols = [f"y_{j}" for j in range(K)]
    c_cols = [f"c_alpha_{j}" for j in range(K)]

    y_argmax_counts = train["y_argmax"].value_counts().sort_index()
    c_argmax_counts = train["c_argmax"].value_counts().sort_index()
    print("\nY argmax distribution (train, n=237):")
    print(y_argmax_counts.to_string())
    modal_y = y_argmax_counts.max() / len(train)
    print(f"Modal share (naive top-1 floor): {modal_y:.1%}")

    empty_y_archetypes = [j for j in range(K) if j not in y_argmax_counts.index]
    if empty_y_archetypes:
        print(f"  ** NBA archetype(s) {empty_y_archetypes} have ZERO training anchors with that "
              f"argmax. Top-1 hit rate for this class is structurally unmeasurable in CV on the "
              f"training folds. Whether this also holds on the holdout is NOT checked here - doing "
              f"so would mean reading 2025-class labels before Phase 5, which the quarantine forbids "
              f"even for a diagnostic. Treat as an open question, not a resolved one. **")

    print("\nCollege-recipe argmax distribution (train, n=237):")
    print(c_argmax_counts.to_string())
    modal_c = c_argmax_counts.max() / len(train)
    print(f"Modal share: {modal_c:.1%}")

    # age x years_in_college joint plausibility scan
    implausible = train[(train["age_at_draft"] >= 22.0) & (train["years_in_college"] <= 1)]
    print(f"\nage_at_draft >= 22.0 with years_in_college <= 1: {len(implausible)} rows")
    if len(implausible):
        print(implausible[["player_name", "draft_year", "age_at_draft", "years_in_college",
                             "college_match_method"]].to_string(index=False))

    # missingness table
    check_cols = (c_cols + y_cols +
                  ["PLAYER_HEIGHT_INCHES", "PTS_PER_100", "TS%", "USG%", "AST%", "TOV%",
                   "STL%", "BLK%", "TRB%", "FTr", "% of FG Ast'd_2P", "% of FG Ast'd_3P",
                   "PORPAG", "rim_finishing_share", "three_pt_jumper_share",
                   "overall", "log_pick", "age_at_draft", "years_in_college", "conf_tier"])
    miss = train[check_cols].isna().mean().sort_values(ascending=False)
    print("\nMissingness (train, fraction NaN):")
    print(miss[miss > 0].to_string() if (miss > 0).any() else "  none (all 0%)")
    print(f"  (shot-type mix specifically: rim={train['rim_finishing_share'].isna().mean():.1%}, "
          f"3pt={train['three_pt_jumper_share'].isna().mean():.1%})")

    # simplex validity
    c_vals = train[c_cols].values
    y_vals = train[y_cols].values
    assert (c_vals >= -1e-9).all() and np.abs(c_vals.sum(axis=1) - 1.0).max() < 1e-6
    assert (y_vals >= -1e-9).all() and np.abs(y_vals.sum(axis=1) - 1.0).max() < 1e-6
    print("\nSimplex validity: PASS (c_alpha and y both non-negative, sum to 1 within 1e-6)")

    report["empty_y_archetypes_train"] = empty_y_archetypes
    report["y_argmax_counts"] = y_argmax_counts.to_dict()
    report["c_argmax_counts"] = c_argmax_counts.to_dict()
    report["modal_y_share"] = float(modal_y)
    report["modal_c_share"] = float(modal_c)
    report["n_implausible_age_years"] = int(len(implausible))
    report["missingness"] = miss.to_dict()
    (OUT_DIR / "phase4_preflight.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nWrote {OUT_DIR / 'phase4_preflight.json'}")


if __name__ == "__main__":
    main()
