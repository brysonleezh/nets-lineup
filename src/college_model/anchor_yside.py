"""
Phase 3 Step 5 - Y-side computation (the core of the phase): project each
anchor's rookie NBA season onto the frozen 8x29 basis, using THAT season's
own persisted mu/sd (never 2025-26's, never recomputed from rookies only -
the locked convention from Step 1).

For each anchor: pull the full 29-feature raw rookie-season row, z-score
with the Step 1 standardization_{season}.json for his rookie_season_expected,
NNLS-project via the existing project() (mu/sd passed explicitly, never
recomputed) -> y_0..y_7, y_argmax, y_max.

Consistency asserts (all must pass):
  1. every y row non-negative, sums to 1 within 1e-6.
  2. 2025-class anchors' projected recipes match the app's existing stored
     data/basis_2025_26/recipes.csv for the same players (same humans, same
     basis, two computation paths) - reusing step1_archetype_model's own
     Part-D verification tolerance (atol=1e-3), never loosened.
  3. 6 sanity vignettes printed for the report (eyeball-plausibility only,
     no translator claims).

Run: python3 src/college_model/anchor_yside.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from step1_archetype_model import load_population, project  # noqa: E402

XSIDE_PATH = REPO_ROOT / "data" / "anchors" / "anchor_xside.csv"
LEDGER_PATH = REPO_ROOT / "data" / "anchors" / "anchor_ledger.csv"
BASIS_PATH = REPO_ROOT / "data" / "basis_2025_26" / "basis.npz"
FEATURE_ORDER_PATH = REPO_ROOT / "data" / "basis_2025_26" / "feature_order.json"
STANDARDIZATION_DIR = REPO_ROOT / "data" / "nba_historical"
APP_RECIPES_PATH = REPO_ROOT / "data" / "basis_2025_26" / "recipes.csv"
OUT_DIR = REPO_ROOT / "data" / "anchors"

K = 8
PART_D_ATOL = 1e-3  # step1_archetype_model.verify_projection's own tolerance - reused, not loosened

VIGNETTES = ["Trae Young", "Zion Williamson", "Ja Morant", "Zach Edey", "Stephon Castle", "Cooper Flagg"]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_csv(XSIDE_PATH)
    ledger = pd.read_csv(LEDGER_PATH)
    anchors = anchors.merge(
        ledger[["draft_year", "overall", "rookie_season_expected", "rookie_minutes"]],
        on=["draft_year", "overall"], how="left")
    assert anchors["rookie_season_expected"].notna().all()
    print(f"Y-side computation for {len(anchors)} anchors")

    basis = np.load(BASIS_PATH)["basis"]
    feature_columns = json.loads(FEATURE_ORDER_PATH.read_text())["feature_columns"]
    assert len(feature_columns) == 29

    # pull each anchor's raw rookie-season feature row from the full NBA population
    # (min_threshold=1, since anchors already cleared the real MIN gate in Step 2 -
    # this is just a row lookup, not a re-filter)
    full_pop = load_population(min_threshold=1)
    missing_cols = [c for c in feature_columns if c not in full_pop.columns]
    assert not missing_cols, f"basis features missing from NBA population: {missing_cols}"

    y_frames = []
    for season, group in anchors.groupby("rookie_season_expected"):
        std_path = STANDARDIZATION_DIR / f"standardization_{season}.json"
        std = json.loads(std_path.read_text())
        assert std["feature_columns"] == feature_columns, f"{season}: feature order mismatch"
        mu, sd = np.array(std["mu"]), np.array(std["sd"])

        rows = full_pop[(full_pop["SEASON"] == season) & (full_pop["PLAYER_ID"].isin(group["nba_player_id"]))]
        rows = rows.drop_duplicates("PLAYER_ID", keep="first")
        assert len(rows) == len(group), (
            f"{season}: expected {len(group)} rookie-season rows, found {len(rows)} - "
            f"a player in this group has no matching row in the full population despite "
            f"clearing Step 2's rookie MIN gate; missing player_ids: "
            f"{set(group['nba_player_id']) - set(rows['PLAYER_ID'])}")

        P = project(rows, basis, mu, sd, feature_columns)
        y_df = pd.DataFrame(P, columns=[f"y_{j}" for j in range(K)])
        y_df["nba_player_id"] = rows["PLAYER_ID"].values
        y_df["rookie_season_expected"] = season
        y_frames.append(y_df)

    y_all = pd.concat(y_frames, ignore_index=True)
    y_all["y_argmax"] = y_all[[f"y_{j}" for j in range(K)]].values.argmax(axis=1)
    y_all["y_max"] = y_all[[f"y_{j}" for j in range(K)]].values.max(axis=1)

    anchors = anchors.merge(y_all.drop(columns=["rookie_season_expected"]), on="nba_player_id", how="left")
    assert anchors[[f"y_{j}" for j in range(K)]].isna().any(axis=1).sum() == 0

    # --- assert 1: simplex validity ---
    y_vals = anchors[[f"y_{j}" for j in range(K)]].values
    assert (y_vals >= -1e-9).all(), "a rookie recipe has a negative y"
    row_sums = y_vals.sum(axis=1)
    max_dev = np.abs(row_sums - 1.0).max()
    print(f"\nAssert 1 (simplex validity): max |sum(y) - 1| = {max_dev:.2e} (must be < 1e-6)")
    assert max_dev < 1e-6, "rookie recipes don't sum to 1"
    print("  PASS")

    # --- assert 2: 2025-class consistency against the app's stored recipes ---
    app_recipes = pd.read_csv(APP_RECIPES_PATH)
    class2025 = anchors[anchors["draft_year"] == 2025].copy()
    check = class2025.merge(app_recipes, left_on="nba_player_id", right_on="PLAYER_ID", how="inner",
                             suffixes=("", "_app"))
    print(f"\nAssert 2 (2025-class consistency vs. app's stored 2025-26 recipes): "
          f"{len(check)}/{len(class2025)} 2025-class anchors matched by PLAYER_ID")
    assert len(check) == len(class2025), "not every 2025-class anchor found in the app's stored recipes"
    diffs = []
    for j in range(K):
        diffs.append(np.abs(check[f"y_{j}"].values - check[f"arch_{j}"].values))
    max_diff = np.max(diffs)
    print(f"  max |y_j - arch_j| across all 2025-class anchors, all 8 archetypes: {max_diff:.6f} "
          f"(Part-D tolerance: {PART_D_ATOL})")
    assert max_diff < PART_D_ATOL, (
        f"2025-class recipes disagree with the app's stored recipes by {max_diff:.4f} >= {PART_D_ATOL} - "
        f"STOP and diagnose (likely a mu/sd or column-order slip). Do not loosen this tolerance."
    )
    print("  PASS")

    # --- assert 3: sanity vignettes ---
    print("\nAssert 3 (sanity vignettes, eyeball-plausibility only):")
    vignette_rows = []
    for name in VIGNETTES:
        hit = anchors[anchors["player_name_raw"].str.contains(name, case=False, na=False)]
        if len(hit) == 0:
            print(f"  {name}: NOT FOUND among included anchors")
            continue
        r = hit.iloc[0]
        c_top3 = sorted(range(K), key=lambda j: -r[f"c_alpha_{j}"])[:3]
        y_top3 = sorted(range(K), key=lambda j: -r[f"y_{j}"])[:3]
        print(f"  {r['player_name_raw']} ({int(r['draft_year'])}): "
              f"college top-3 = {[(j, round(r[f'c_alpha_{j}'], 2)) for j in c_top3]}  "
              f"-> rookie NBA top-3 = {[(j, round(r[f'y_{j}'], 2)) for j in y_top3]}")
        vignette_rows.append({
            "player_name_raw": r["player_name_raw"], "draft_year": r["draft_year"],
            "college_top3": str([(j, round(r[f"c_alpha_{j}"], 3)) for j in c_top3]),
            "rookie_top3": str([(j, round(r[f"y_{j}"], 3)) for j in y_top3]),
        })
    pd.DataFrame(vignette_rows).to_csv(OUT_DIR / "sanity_vignettes.csv", index=False)

    anchors.to_csv(OUT_DIR / "anchor_yside.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'anchor_yside.csv'} ({len(anchors)} rows, {len(anchors.columns)} columns)")
    print(f"Wrote {OUT_DIR / 'sanity_vignettes.csv'}")


if __name__ == "__main__":
    main()
