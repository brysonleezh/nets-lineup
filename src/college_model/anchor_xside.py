"""
Phase 3 Step 4 - X-side assembly for the 273 included anchors.

College recipe: join Phase 2's data/college/recipes.csv on (matched_athlete_id,
final_college_season) -> c_alpha_0..c_alpha_7 (kept 0-indexed to match the
underlying archetype numbering used everywhere else in this repo -
recipes.csv's own columns are alpha_0..alpha_7, not 1..8; renaming to a
1-indexed c_alpha_1..8 as the spec's prose literally reads would introduce
an off-by-one mismatch against every other archetype reference in the repo,
so this phase treats that as shorthand for "the 8 alpha columns", not a
strict numbering mandate), c_argmax, c_alpha_max. Asserts every joined
recipe is non-negative and sums to 1 (1e-6).

Shared features: the 12 raw + 12 per-season-z columns from
shared_features.parquet. PORPAG kept as its own column, never merged with
BPM (the two are on different scales and mean different things - PORPAG is
possession-based, BPM is a box-score plus-minus model; conflating them
would be a silent unit error).

Step 4b (optional): college-native shot-type mix from the already-cached
shooting_*.json files (340 files, no additional network calls) - rim-
finishing share and 3PT-jumper share, joined by (athleteId, season).
College-only features (no NBA-side symmetry needed or attempted - these
are legitimate Phase 4 variant-(b) inputs only). Missing -> NaN, coverage
reported, no claim made about downstream use (Phase 4's call).

Run: python3 src/college_model/anchor_xside.py
"""

from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

COVARIATES_PATH = REPO_ROOT / "data" / "anchors" / "anchor_covariates.csv"
RECIPES_PATH = REPO_ROOT / "data" / "college" / "recipes.csv"
SHARED_FEATURES_PATH = REPO_ROOT / "data" / "college" / "shared_features.parquet"
RAW_CBBD_DIR = REPO_ROOT / "data" / "raw" / "cbbd"
OUT_DIR = REPO_ROOT / "data" / "anchors"

K = 8


def _load_shot_type_mix() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(RAW_CBBD_DIR / "shooting_*.json"))):
        for p in json.loads(Path(f).read_text()):
            tracked = p.get("trackedShots") or 0
            if tracked <= 0:
                continue
            ab = p.get("attemptsBreakdown", {})
            rim = (p["dunks"]["attempted"] + p["layups"]["attempted"] + p["tipIns"]["attempted"])
            rows.append({
                "athleteId": p["athleteId"], "season": p["season"],
                "trackedShots": tracked,
                "rim_finishing_share": rim / tracked,
                "three_pt_jumper_share": p["threePointJumpers"]["attempted"] / tracked,
            })
    df = pd.DataFrame(rows)
    # a few (athleteId, season) pairs appear in >1 conference file if a player
    # transferred mid-year and both old/new conference shooting files list him -
    # keep the row with more tracked shots (the fuller-season record)
    df = df.sort_values("trackedShots", ascending=False).drop_duplicates(["athleteId", "season"], keep="first")
    return df


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    anchors = pd.read_csv(COVARIATES_PATH)
    print(f"X-side assembly for {len(anchors)} anchors")

    # --- college recipe ---
    recipes = pd.read_csv(RECIPES_PATH)
    alpha_cols = [f"alpha_{j}" for j in range(K)]
    recipes_key = recipes[["player_id_source", "season"] + alpha_cols + ["argmax_archetype", "alpha_max"]].copy()
    recipes_key = recipes_key.rename(columns={f"alpha_{j}": f"c_alpha_{j}" for j in range(K)},)
    recipes_key = recipes_key.rename(columns={"argmax_archetype": "c_argmax", "alpha_max": "c_alpha_max"})

    before = len(anchors)
    anchors = anchors.merge(
        recipes_key, left_on=["matched_athlete_id", "final_college_season"],
        right_on=["player_id_source", "season"], how="left")
    assert len(anchors) == before, "recipe join changed row count - duplicate keys in recipes.csv"
    c_alpha = anchors[[f"c_alpha_{j}" for j in range(K)]]
    missing_recipe = c_alpha.isna().any(axis=1).sum()
    print(f"college recipe join: {missing_recipe} anchors missing a recipe (expect 0 - every "
          f"included anchor cleared the college_below_min gate in Step 2, which checked "
          f"exactly this presence)")
    assert missing_recipe == 0, "an included anchor has no college recipe - Step 2's gate should have caught this"
    assert (c_alpha.values >= -1e-9).all(), "a college recipe has a negative alpha"
    row_sums = c_alpha.sum(axis=1)
    assert (np.abs(row_sums - 1.0) < 1e-6).all(), f"college recipes don't sum to 1: max deviation {np.abs(row_sums-1).max():.2e}"
    print("college recipe simplex check: PASS (all non-negative, all sum to 1 within 1e-6)")
    anchors = anchors.drop(columns=["player_id_source", "season"])

    # --- shared features (12 raw + 12 z + PORPAG) ---
    sf = pd.read_parquet(SHARED_FEATURES_PATH)
    sf = sf.drop_duplicates(["player_id_source", "season"], keep="first")  # same A.J. Lawson dupe as Step 3
    feature_cols = [c for c in sf.columns if c not in
                     ("player_id_source", "player_name_raw", "season", "team", "league", "minutes", "games", "conference")]
    sf_key = sf[["player_id_source", "season"] + feature_cols].copy()

    before = len(anchors)
    anchors = anchors.merge(
        sf_key, left_on=["matched_athlete_id", "final_college_season"],
        right_on=["player_id_source", "season"], how="left", suffixes=("", "_sf"))
    assert len(anchors) == before, "shared_features join changed row count"
    missing_sf = anchors["PORPAG"].isna().sum()
    print(f"shared_features join: {missing_sf} anchors missing (expect 0, same reasoning as the recipe join)")
    assert missing_sf == 0, "an included anchor has no shared_features row"
    anchors = anchors.drop(columns=["player_id_source", "season"])

    # --- Step 4b: college-native shot-type mix (optional) ---
    shot_mix = _load_shot_type_mix()
    before = len(anchors)
    anchors = anchors.merge(
        shot_mix[["athleteId", "season", "rim_finishing_share", "three_pt_jumper_share"]],
        left_on=["matched_athlete_id", "final_college_season"], right_on=["athleteId", "season"], how="left")
    assert len(anchors) == before, "shot-type mix join changed row count"
    coverage = anchors["rim_finishing_share"].notna().mean()
    print(f"shot-type mix (Step 4b, optional) coverage: {coverage:.1%} - college-only features, "
          f"no claim made about downstream use")
    anchors = anchors.drop(columns=["athleteId", "season"])

    out_path = OUT_DIR / "anchor_xside.csv"
    anchors.to_csv(out_path, index=False)
    print(f"\nWrote {out_path} ({len(anchors)} rows, {len(anchors.columns)} columns)")


if __name__ == "__main__":
    main()
