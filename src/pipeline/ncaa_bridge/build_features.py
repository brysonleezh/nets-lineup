"""
NCAA bridge - Step 1a: build the shared ~14-feature table from cached CBBD
raw responses, MIN>=300 filtered, one row per (season, athleteId).

Derivation notes (formulas are standard Dean-Oliver-style box-score rate
stats; CBBD gives some directly and the rest is derived from its own
teamStats/opponentStats blocks - confirmed live that opponentStats exists
and mirrors teamStats exactly, which is what makes STL%/BLK%/TRB%
computable at all):

  Direct from CBBD, normalized to a 0-1 fraction (CBBD mixes 0-1 and 0-100
  scales across fields in the SAME payload - trueShootingPct is 0-1,
  everything else percentage-like is 0-100 - normalized here so every
  feature in the output table is consistently 0-1):
    TS%   = trueShootingPct                        (already 0-1)
    USG%  = usage / 100
    ORB%  = offensiveReboundPct / 100
    FTr   = freeThrowRate / 100                     (= FTA/FGA, confirmed
                                                       against raw makes/
                                                       attempts on a sample
                                                       player)
    PORPAG = PORPAG as-is (substitutes BPM - no college BPM equivalent)

  Derived (needs team totalMinutes/possessions + the mirrored
  opponentStats block):
    AST%  = 100 * AST / (((MP/(TeamMP/5)) * TeamFGM) - FGM)
    TOV%  = 100 * TOV / (FGA + 0.44*FTA + TOV)       (player-level only)
    STL%  = 100 * STL * (TeamMP/5) / (MP * OppPoss)
    BLK%  = 100 * BLK * (TeamMP/5) / (MP * (OppFGA - Opp3PA))
    DRB%  = 100 * DRB * (TeamMP/5) / (MP * (TeamDRB + OppORB))
    TRB%  = 100 * TRB * (TeamMP/5) / (MP * (TeamTRB + OppTRB))
    PTS_PER_100 = 100 * PTS / (TeamPoss * (MP/TeamMP))
  (all four divided by 100 at the end to land on the same 0-1 fraction
  convention as the direct fields above)

  From /stats/player/shooting/season (looped per conference at ingest
  time), aggregated across CBBD's shot-TYPE categories (dunks/layups/
  tipIns/twoPointJumpers for the 2P side, threePointJumpers for the 3P
  side) rather than any single category's own assistedPct, since e.g.
  "twoPointJumpers.assistedPct" alone excludes dunks/layups/tip-ins and
  would understate true %FGAst'd_2P:
    %FGAst'd_2P = sum(assisted over the 4 twos categories) / sum(made)
    %FGAst'd_3P = threePointJumpers.assisted / threePointJumpers.made

  From /teams/roster (joined on athleteId; CBBD reports height already in
  inches, confirmed against Mikel Brown Jr.'s real listed height):
    PLAYER_HEIGHT_INCHES = height

Run: python3 src/pipeline/ncaa_bridge/build_features.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "ncaa_bridge_config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def _load_json(path: Path):
    return json.loads(path.read_text())


def build_season_features(season: int, cache_dir: Path) -> pd.DataFrame:
    players = _load_json(cache_dir / f"player_season_{season}.json")
    teams = _load_json(cache_dir / f"team_season_{season}.json")
    rosters = _load_json(cache_dir / f"roster_{season}.json")

    shooting_rows = []
    for f in cache_dir.glob(f"shooting_{season}_*.json"):
        shooting_rows.extend(_load_json(f))

    team_by_id = {t["teamId"]: t for t in teams}

    height_by_athlete: dict[int, float] = {}
    for t in rosters:
        for p in t["players"]:
            if p.get("height"):
                height_by_athlete[p["id"]] = p["height"]

    ast2p_num: dict[int, float] = {}
    ast2p_den: dict[int, float] = {}
    ast3p_num: dict[int, float] = {}
    ast3p_den: dict[int, float] = {}
    for row in shooting_rows:
        aid = row["athleteId"]
        two_cats = [row.get("dunks"), row.get("layups"), row.get("tipIns"), row.get("twoPointJumpers")]
        made_2p = sum((c or {}).get("made", 0) or 0 for c in two_cats)
        ast_2p = sum((c or {}).get("assisted", 0) or 0 for c in two_cats)
        three = row.get("threePointJumpers") or {}
        made_3p = three.get("made", 0) or 0
        ast_3p = three.get("assisted", 0) or 0
        ast2p_num[aid] = ast2p_num.get(aid, 0) + ast_2p
        ast2p_den[aid] = ast2p_den.get(aid, 0) + made_2p
        ast3p_num[aid] = ast3p_num.get(aid, 0) + ast_3p
        ast3p_den[aid] = ast3p_den.get(aid, 0) + made_3p

    records = []
    for p in players:
        min_played = p.get("minutes") or 0
        team = team_by_id.get(p["teamId"])
        if team is None:
            continue
        ts = team["teamStats"]
        os_ = team["opponentStats"]
        team_mp = team.get("totalMinutes") or 0
        if not team_mp or not min_played:
            continue

        fgm = (p.get("fieldGoals") or {}).get("made", 0) or 0
        fga = (p.get("fieldGoals") or {}).get("attempted", 0) or 0
        fta = (p.get("freeThrows") or {}).get("attempted", 0) or 0
        ast = p.get("assists") or 0
        tov = p.get("turnovers") or 0
        stl = p.get("steals") or 0
        blk = p.get("blocks") or 0
        drb = (p.get("rebounds") or {}).get("defensive", 0) or 0
        trb = (p.get("rebounds") or {}).get("total", 0) or 0
        pts = p.get("points") or 0

        team_fgm = ts["fieldGoals"]["made"]
        team_drb = ts["rebounds"]["defensive"]
        team_trb = ts["rebounds"]["total"]
        team_poss = ts["possessions"]
        opp_poss = os_["possessions"]
        opp_orb = os_["rebounds"]["offensive"]
        opp_trb = os_["rebounds"]["total"]
        opp_fga = os_["fieldGoals"]["attempted"]
        opp_3pa = os_["threePointFieldGoals"]["attempted"]

        # CBBD's team `totalMinutes` is already team GAME-clock minutes
        # (games * 40) - i.e. already equal to the standard formula's
        # "Tm MP / 5" (the traditional Tm MP convention sums all 5 players'
        # concurrent minutes, so Tm MP/5 = one player's worth of total
        # possible court time = games * 40). Confirmed by direct check: a
        # heavy-minutes starter who played ~75% of team totalMinutes came
        # out with a plausible ~75% MP/(TeamMP/5) ratio only when
        # totalMinutes is used AS Tm MP/5 directly, not divided by 5 again.
        team_poss_5 = team_mp

        ast_pct = 100 * ast / max(((min_played / team_poss_5) * team_fgm) - fgm, 1e-9)
        tov_pct = 100 * tov / max(fga + 0.44 * fta + tov, 1e-9)
        stl_pct = 100 * stl * team_poss_5 / max(min_played * opp_poss, 1e-9)
        blk_pct = 100 * blk * team_poss_5 / max(min_played * (opp_fga - opp_3pa), 1e-9)
        drb_pct = 100 * drb * team_poss_5 / max(min_played * (team_drb + opp_orb), 1e-9)
        trb_pct = 100 * trb * team_poss_5 / max(min_played * (team_trb + opp_trb), 1e-9)
        pts_per_100 = 100 * pts / max(team_poss * (min_played / team_mp), 1e-9)

        # Zero makes in a shot family (e.g. a true big who never shoots 3s)
        # makes "% of makes assisted" undefined, not missing - imputed to 0
        # (no assisted-shot signal in that family) rather than dropped, so
        # the MIN-filter pool doesn't disproportionately lose bigs.
        aid = p["athleteId"]
        ast2p_pct = (ast2p_num.get(aid, 0) / ast2p_den[aid]) if ast2p_den.get(aid) else 0.0
        ast3p_pct = (ast3p_num.get(aid, 0) / ast3p_den[aid]) if ast3p_den.get(aid) else 0.0

        records.append({
            "SEASON": season,
            "athleteId": aid,
            "name": p.get("name"),
            "team": p.get("team"),
            "conference": p.get("conference"),
            "MIN": min_played,
            "PLAYER_HEIGHT_INCHES": height_by_athlete.get(aid, np.nan),
            "PTS_PER_100": pts_per_100 / 100.0,
            "TS%": p.get("trueShootingPct"),
            "USG%": (p.get("usage") or np.nan) / 100.0,
            "AST%": ast_pct / 100.0,
            "TOV%": tov_pct / 100.0,
            "STL%": stl_pct / 100.0,
            "BLK%": blk_pct / 100.0,
            "ORB%": (p.get("offensiveReboundPct") or np.nan) / 100.0,
            "DRB%": drb_pct / 100.0,
            "TRB%": trb_pct / 100.0,
            "FTr": (p.get("freeThrowRate") or np.nan) / 100.0,
            "PORPAG": p.get("PORPAG"),
            "%FGAst'd_2P": ast2p_pct,
            "%FGAst'd_3P": ast3p_pct,
        })

    return pd.DataFrame.from_records(records)


def main() -> None:
    cfg = load_config()
    cache_dir = REPO_ROOT / cfg["cbbd"]["raw_cache_dir"]
    min_minutes = cfg["college_model"]["min_minutes"]
    feature_cols = cfg["college_model"]["features"]

    all_seasons = []
    for season in cfg["cbbd"]["seasons"]:
        df = build_season_features(season, cache_dir)
        all_seasons.append(df)
        print(f"season {season}: {len(df)} player-seasons before MIN filter", flush=True)

    full = pd.concat(all_seasons, ignore_index=True)
    full_filtered = full[full["MIN"] >= min_minutes].copy()

    n_missing = full_filtered[feature_cols].isna().sum()
    print("\nNull counts per feature (MIN-filtered pool):")
    print(n_missing.to_string())

    n_before = len(full_filtered)
    complete = full_filtered.dropna(subset=feature_cols)
    print(f"\n{n_before} rows MIN>={min_minutes}; {len(complete)} with all {len(feature_cols)} features present "
          f"({n_before - len(complete)} dropped for missing features)")

    out_dir = REPO_ROOT / "data" / "college"
    out_dir.mkdir(parents=True, exist_ok=True)
    full_filtered.to_csv(out_dir / "features_raw.csv", index=False)
    complete.to_csv(out_dir / "features_complete.csv", index=False)
    print(f"\nWrote {out_dir / 'features_raw.csv'} ({len(full_filtered)} rows, MIN-filtered, may have NaNs)")
    print(f"Wrote {out_dir / 'features_complete.csv'} ({len(complete)} rows, complete cases only)")

    print("\nPer-season complete-case counts:")
    print(complete.groupby("SEASON").size().to_string())


if __name__ == "__main__":
    main()
