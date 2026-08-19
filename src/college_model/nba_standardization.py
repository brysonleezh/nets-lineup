"""
Phase 3 Step 1 - per-season NBA standardization artifacts (the Y-side
foundation). For each of the 9 seasons, population = full league MIN>=300
(flat threshold for the STANDARDIZATION pool, deliberately, even for
2020-21 - the eligibility threshold pro-ration in Step 2 is a separate,
later concern, not applied here). Persists mu/sd over the 29 basis
features in feature_order.json's exact order.

Hard assert: a from-scratch recompute of 2025-26 mu/sd must match the
already-persisted data/basis_2025_26/standardization.json element-wise
(1e-9) - proves this computation path is identical to the one that built
the basis.

**2026-08-12 finding**: that assert failed on the first run - not a bug in
this script. The live DB now has 434 2025-26 rows vs. the 433 the frozen
basis.npz/recipes.csv were built on (both dated 2026-07-28 23:28, exactly
matching each other). Diffed player_id sets against recipes.csv (frozen
alongside basis.npz) and found exactly one row added, none removed:
Ronald Holland II (DET, ~1550 min) - nowhere near the 300-minute threshold,
so this isn't a threshold-crossing case; his row must have been incomplete
(dropped by build_nba_side_tables's completeness filter) when the basis was
fit and has since been backfilled in the underlying tables. Root-caused via
timestamp/content cross-checks (nets_synergy.db's own mtime predates the
basis by ~23h and is reproducible run-to-run, ruling out query
non-determinism) before concluding it's real upstream data drift, not a
z-scoring bug. Owner decision (AskUserQuestion): reuse the frozen
standardization.json verbatim for 2025-26 rather than recompute, to stay
bit-identical to the basis.npz/recipes.csv the rest of the app already
depends on - Phase 2's frozen model is out of scope to touch in Phase 3.
The other 8 historical seasons have no frozen counterpart to diverge from,
so those are still freshly computed here. See reports/phase3_worklog.md.

Run: python3 src/college_model/nba_standardization.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from step1_archetype_model import load_population  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "nba_historical"
FROZEN_SEASON = "2025-26"
HISTORICAL_SEASONS = ["2017-18", "2018-19", "2019-20", "2020-21", "2021-22",
                       "2022-23", "2023-24", "2024-25"]
SEASONS = HISTORICAL_SEASONS + [FROZEN_SEASON]


def _verify_drift_is_the_known_holland_case(feature_cols: list[str]) -> None:
    """Recompute 2025-26 fresh and confirm the mismatch is exactly the known,
    already-investigated Ronald Holland II drift (1 extra row, small mu/sd
    delta) - not some new, un-investigated divergence. Does not write
    anything; this is a diagnostic re-check only, run every time so a
    *different* future drift doesn't get silently swallowed by the reuse path.
    """
    df = load_population(min_threshold=300, season_min=FROZEN_SEASON, season_max=FROZEN_SEASON)
    persisted = json.loads((REPO_ROOT / "data" / "basis_2025_26" / "standardization.json").read_text())

    n_diff = len(df) - persisted["n_train"]
    if n_diff == 0:
        print(f"{FROZEN_SEASON}: live population now matches frozen n_train "
              f"({persisted['n_train']}) exactly - Holland-case drift has resolved itself.")
        return

    X = df[feature_cols].astype(float)
    mu_diff = np.abs(X.mean().values - np.array(persisted["mu"])).max()
    sd_diff = np.abs(X.std().values - np.array(persisted["sd"])).max()
    print(f"{FROZEN_SEASON}: live n={len(df)} vs frozen n_train={persisted['n_train']} "
          f"(diff={n_diff:+d}), max|mu diff|={mu_diff:.2e}, max|sd diff|={sd_diff:.2e}")

    known_drift = (n_diff == 1 and mu_diff < 0.05 and sd_diff < 0.05)
    if not known_drift:
        raise AssertionError(
            f"{FROZEN_SEASON} drift no longer matches the investigated Ronald "
            f"Holland II case (n_diff={n_diff:+d}, max|mu diff|={mu_diff:.2e}, "
            f"max|sd diff|={sd_diff:.2e}) - this looks like a NEW divergence, "
            f"re-investigate before reusing the frozen standardization.json blindly."
        )
    print("  -> matches the known, owner-reviewed Holland-case drift magnitude; "
          "proceeding to reuse the frozen standardization.json unchanged.")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    feature_order = json.loads((REPO_ROOT / "data" / "basis_2025_26" / "feature_order.json").read_text())
    feature_cols = feature_order["feature_columns"]
    assert len(feature_cols) == 29, f"expected 29 basis features, got {len(feature_cols)}"

    for season in HISTORICAL_SEASONS:
        df = load_population(min_threshold=300, season_min=season, season_max=season)
        assert (df["SEASON"] == season).all()
        X = df[feature_cols].astype(float)
        mu = X.mean()
        sd = X.std()
        dead = sd[sd < 1e-12].index.tolist()
        if dead:
            raise ValueError(f"{season}: zero-variance features, z-scoring would divide by 0: {dead}")

        out = {
            "season": season,
            "feature_columns": feature_cols,
            "mu": mu.values.tolist(),
            "sd": sd.values.tolist(),
            "n": int(len(df)),
            "min_threshold": 300,
        }
        (OUT_DIR / f"standardization_{season}.json").write_text(json.dumps(out, indent=2))
        print(f"{season}: n={len(df)}, wrote standardization_{season}.json", flush=True)

    # 2025-26: owner decision (2026-08-12) is to reuse the frozen basis
    # standardization verbatim rather than recompute, to stay bit-identical
    # with basis.npz/recipes.csv - see module docstring for the Holland II
    # finding this resolves. Still re-check the drift shape every run so a
    # future, DIFFERENT divergence doesn't get silently masked.
    _verify_drift_is_the_known_holland_case(feature_cols)
    persisted = json.loads((REPO_ROOT / "data" / "basis_2025_26" / "standardization.json").read_text())
    frozen_out = {
        "season": FROZEN_SEASON,
        "feature_columns": persisted["feature_columns"],
        "mu": persisted["mu"],
        "sd": persisted["sd"],
        "n": persisted["n_train"],
        "min_threshold": 300,
        "source": "reused verbatim from data/basis_2025_26/standardization.json "
                   "(not recomputed) - live DB has drifted by +1 row (Ronald "
                   "Holland II) since the basis was frozen 2026-07-28; owner "
                   "decision 2026-08-12 was to preserve bit-identical continuity "
                   "with the frozen basis.npz/recipes.csv rather than track the "
                   "live DB. See reports/phase3_worklog.md.",
    }
    (OUT_DIR / f"standardization_{FROZEN_SEASON}.json").write_text(json.dumps(frozen_out, indent=2))
    print(f"{FROZEN_SEASON}: n={frozen_out['n']}, wrote standardization_{FROZEN_SEASON}.json (reused frozen values)")
    print("\nPASS: 2025-26 standardization artifact is bit-identical to the frozen basis (reused, not recomputed).")


if __name__ == "__main__":
    main()
