"""
Phase 2 Step 1 - fit pool definition.

Base pool: all complete-case player-seasons in the canonical table
(data/college/shared_features.parquet, 29,312 rows, 2016-17..2025-26,
MIN>=300), pooled across seasons into one fit (per-season z already makes
rows comparable - see build_shared_features.py).

restrict_to_draft_conferences (default ON): keep only player-seasons
whose OWN (season-specific) conference - already recorded per-row in the
canonical table, since it comes directly from that exact season's CBBD
data - is in the "picked-conference set": the set of conferences that
produced >=1 NCAA-path draft pick, where each pick's conference is looked
up in ITS OWN final_college_season (not today's realigned conference).
This correctly handles conference realignment on both sides: a fit-pool
row's conference is inherently season-accurate, and a pick's
"did this conference produce a pick" membership is evaluated at the
season it actually happened.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "raw" / "cbbd"
CANONICAL_TABLE = REPO_ROOT / "data" / "college" / "shared_features.parquet"

SHARED_COLS = [
    "PLAYER_HEIGHT_INCHES", "PTS_PER_100", "TS%", "USG%", "AST%", "TOV%",
    "STL%", "BLK%", "TRB%", "FTr", "% of FG Ast'd_2P", "% of FG Ast'd_3P",
]
Z_COLS = [f"{c}_z" for c in SHARED_COLS]


def load_canonical() -> pd.DataFrame:
    return pd.read_parquet(CANONICAL_TABLE)


def picked_conference_set() -> set[str]:
    cw = pd.read_csv(REPO_ROOT / "data" / "anchors" / "crosswalk_draft.csv")
    ncaa = cw[(cw["path"] == "NCAA") & (cw["matched_athlete_id"].notna())]

    team_season_cache: dict[int, list[dict]] = {}

    def teams_for_season(season: int) -> list[dict]:
        if season not in team_season_cache:
            f = CACHE_DIR / f"team_season_{season}.json"
            team_season_cache[season] = json.loads(f.read_text()) if f.exists() else []
        return team_season_cache[season]

    confs = set()
    for _, row in ncaa.iterrows():
        season = int(row["final_college_season"])
        school = row["source_team_location"]
        match = [t for t in teams_for_season(season) if t["team"] == school]
        if match:
            confs.add(match[0]["conference"])
    return confs


def build_fit_pool(df: pd.DataFrame, restrict: bool) -> pd.DataFrame:
    if not restrict:
        return df.copy()
    confs = picked_conference_set()
    return df[df["conference"].isin(confs)].copy()


def report_pool_sizes() -> dict:
    df = load_canonical()
    confs = picked_conference_set()
    pool_on = build_fit_pool(df, restrict=True)
    pool_off = build_fit_pool(df, restrict=False)
    return {
        "n_picked_conferences": len(confs),
        "picked_conferences": sorted(confs),
        "n_total_rows": len(df),
        "n_total_conferences": df["conference"].nunique(),
        "pool_on_rows": len(pool_on),
        "pool_on_conferences": pool_on["conference"].nunique(),
        "pool_off_rows": len(pool_off),
        "pool_off_conferences": pool_off["conference"].nunique(),
        "pool_on_by_season": pool_on.groupby("season").size().to_dict(),
        "pool_off_by_season": pool_off.groupby("season").size().to_dict(),
    }


if __name__ == "__main__":
    import json as _json
    report = report_pool_sizes()
    print(_json.dumps(report, indent=2, default=str))
