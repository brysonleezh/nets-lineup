"""
Phase 3 Step 6 - final anchors.csv assembly.

Takes anchor_yside.csv (273 rows, the X+Y assembled table) and adds the two
remaining spec-required columns that weren't picked up by any earlier join
(college_team, college_minutes - dropped from the Step 4 shared_features
join on purpose since 'team'/'minutes'/'conference' were excluded from the
generic feature-column list there; conference was rebuilt separately in
Step 3, team/minutes were not), pulls college_match_method from the ledger,
renames to the spec's exact column names, and writes the final
data/anchors/anchors.csv with columns in the spec's stated order.

Run: python3 src/college_model/anchor_finalize.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

YSIDE_PATH = REPO_ROOT / "data" / "anchors" / "anchor_yside.csv"
LEDGER_PATH = REPO_ROOT / "data" / "anchors" / "anchor_ledger.csv"
SHARED_FEATURES_PATH = REPO_ROOT / "data" / "college" / "shared_features.parquet"
OUT_DIR = REPO_ROOT / "data" / "anchors"

K = 8

RAW_SHARED_COLS = ["PLAYER_HEIGHT_INCHES", "PTS_PER_100", "TS%", "USG%", "AST%", "TOV%",
                    "STL%", "BLK%", "TRB%", "FTr", "% of FG Ast'd_2P", "% of FG Ast'd_3P"]
Z_SHARED_COLS = [c + "_z" for c in RAW_SHARED_COLS]


def main() -> None:
    anchors = pd.read_csv(YSIDE_PATH)
    ledger = pd.read_csv(LEDGER_PATH)

    # college_team / college_minutes: dropped from Step 4's shared_features join on
    # purpose (excluded from the generic feature-column list there); pull them now.
    sf = pd.read_parquet(SHARED_FEATURES_PATH)
    sf = sf.drop_duplicates(["player_id_source", "season"], keep="first")
    team_min = sf[["player_id_source", "season", "team", "minutes"]].rename(
        columns={"team": "college_team", "minutes": "college_minutes"})
    before = len(anchors)
    anchors = anchors.merge(team_min, left_on=["matched_athlete_id", "final_college_season"],
                             right_on=["player_id_source", "season"], how="left")
    assert len(anchors) == before
    assert anchors["college_team"].notna().all(), "missing college_team for an included anchor"
    anchors = anchors.drop(columns=["player_id_source", "season"])

    # college_match_method: from the ledger (Step 2 recovery method for the 3 special cases,
    # direct_athlete_id / name_year_team for everyone else)
    anchors = anchors.merge(
        ledger[["draft_year", "overall", "match_method"]].rename(columns={"match_method": "college_match_method"}),
        on=["draft_year", "overall"], how="left")

    anchors = anchors.rename(columns={
        "matched_athlete_id": "cbbd_athlete_id",
        "player_name_raw": "player_name",
        "rookie_season_expected": "rookie_season",
    })

    id_cols = ["draft_year", "round", "pick", "overall", "nba_player_id", "cbbd_athlete_id", "player_name"]
    college_cols = (["final_college_season", "college_team", "conference", "college_minutes"]
                     + RAW_SHARED_COLS + Z_SHARED_COLS
                     + ["PORPAG", "rim_finishing_share", "three_pt_jumper_share"]
                     + [f"c_alpha_{j}" for j in range(K)] + ["c_argmax", "c_alpha_max"])
    covariate_cols = ["log_pick", "age_at_draft", "years_in_college", "position", "conf_tier"]
    y_cols = (["rookie_season", "rookie_minutes"] + [f"y_{j}" for j in range(K)] + ["y_argmax", "y_max"])
    bookkeeping_cols = ["college_match_method"]

    final_cols = id_cols + college_cols + covariate_cols + y_cols + bookkeeping_cols
    missing = [c for c in final_cols if c not in anchors.columns]
    assert not missing, f"columns missing from assembled anchors table: {missing}"

    final = anchors[final_cols].copy()
    assert (final["draft_year"] != 2026).all(), "a 2026 row leaked into anchors.csv"

    out_path = OUT_DIR / "anchors.csv"
    final.to_csv(out_path, index=False)
    print(f"Wrote {out_path}: {len(final)} rows, {len(final.columns)} columns")
    print(f"draft_year range: {final['draft_year'].min()}-{final['draft_year'].max()}")
    print(f"per-class counts:\n{final['draft_year'].value_counts().sort_index()}")


if __name__ == "__main__":
    main()
