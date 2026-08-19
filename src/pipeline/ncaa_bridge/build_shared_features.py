"""
Phase 1 Step 3 (data-only) - build shared_features.parquet on both sides,
using the 12 NBA archetype-basis columns confirmed buildable from CBBD
(see reports/feature_dictionary.md), plus each side's own non-equivalent
"value" column (NBA: BPM: college: PORPAG - kept separate, never merged
into one column, since they are not the same statistic).

Shared columns (identical name, identical definition, both sides):
  PLAYER_HEIGHT_INCHES, PTS_PER_100, TS%, USG%, AST%, TOV%, STL%, BLK%,
  TRB%, FTr, % of FG Ast'd_2P, % of FG Ast'd_3P

Per-season z-scores are computed independently on EACH side (each side's
own within-season mean/sd), per this project's own established
"role is relative to your own league-year" convention - NOT the fixed
2025-26 NBA basis mu/sd (that's a different, later-phase concern for
projecting onto the fixed Z; this file only produces raw+z shared
features for comparison/QA, no modeling).

Run: python3 src/pipeline/fit_harness/build_shared_features.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

SHARED_COLS = [
    "PLAYER_HEIGHT_INCHES", "PTS_PER_100", "TS%", "USG%", "AST%", "TOV%",
    "STL%", "BLK%", "TRB%", "FTr", "% of FG Ast'd_2P", "% of FG Ast'd_3P",
]


def per_season_zscore(df: pd.DataFrame, cols: list[str], season_col: str) -> pd.DataFrame:
    out = df.copy()
    for col in cols:
        out[f"{col}_z"] = out.groupby(season_col)[col].transform(lambda s: (s - s.mean()) / s.std())
    return out


def build_college_side() -> pd.DataFrame:
    df = pd.read_csv(REPO_ROOT / "data" / "college" / "features_complete.csv")
    df = df.rename(columns={"%FGAst'd_2P": "% of FG Ast'd_2P", "%FGAst'd_3P": "% of FG Ast'd_3P"})
    df["player_id_source"] = df["athleteId"]
    df["player_name_raw"] = df["name"]
    df["league"] = "NCAA"
    df["games"] = np.nan  # not carried in features_complete.csv today - noted as a gap, not fabricated
    df = per_season_zscore(df, SHARED_COLS, "SEASON")
    keep = ["player_id_source", "player_name_raw", "SEASON", "team", "league", "MIN", "games",
            "conference", "PORPAG"] + SHARED_COLS + [f"{c}_z" for c in SHARED_COLS]
    return df[keep].rename(columns={"SEASON": "season", "MIN": "minutes"})


def build_nba_side() -> pd.DataFrame:
    from step1_archetype_model import load_population
    df = load_population(min_threshold=300)
    df["player_id_source"] = df["PLAYER_ID"]
    df["player_name_raw"] = df["PLAYER_NAME"]
    df["league"] = "NBA"
    df = per_season_zscore(df, SHARED_COLS, "SEASON")
    keep = ["player_id_source", "player_name_raw", "SEASON", "TEAM_ABBREVIATION", "league", "MIN", "POSS",
            "BPM"] + SHARED_COLS + [f"{c}_z" for c in SHARED_COLS]
    return df[keep].rename(columns={"SEASON": "season", "TEAM_ABBREVIATION": "team",
                                     "MIN": "minutes", "POSS": "games"})


def main() -> None:
    college = build_college_side()
    nba = build_nba_side()

    college_out = REPO_ROOT / "data" / "college" / "shared_features.parquet"
    nba_out_dir = REPO_ROOT / "data" / "nba_historical"
    nba_out_dir.mkdir(parents=True, exist_ok=True)
    nba_out = nba_out_dir / "shared_features.parquet"

    college.to_parquet(college_out, index=False)
    nba.to_parquet(nba_out, index=False)

    print(f"College: {college.shape} -> {college_out}")
    print(f"NBA:     {nba.shape} -> {nba_out}")
    print()
    print("College per-season counts:")
    print(college.groupby("season").size().to_string())
    print()
    print("NBA per-season counts:")
    print(nba.groupby("season").size().to_string())
    print()
    print("Missingness on shared columns (college):")
    print(college[SHARED_COLS].isna().sum().to_string())
    print()
    print("Missingness on shared columns (NBA):")
    print(nba[SHARED_COLS].isna().sum().to_string())


if __name__ == "__main__":
    main()
