"""
Phase 3 Step 3 - covariates for the 273 included anchors.

age_at_draft: the only network step this phase. Pulls commonplayerinfo
BIRTHDATE for included anchors' NBA ids only (not the full ~430-pick
candidate pool - the birthdate is only ever used downstream for anchors.csv
rows, so pulling it for excluded rows would be dead network traffic).
Cached immutably under data/raw/nba/commonplayerinfo/{player_id}.json; a
second run must be fully offline.

years_in_college: final_college_season - min(startSeason across all cached
CBBD roster entries for that athleteId) + 1. Validated against 12 known
cases (spec named 6; 6 more added here for a statistically firmer read) -
see the worklog for the Ja Morant investigation (1/12 = 8.3% disagreement,
under the 10% fallback trigger, so the pre-2016 player_season extension
fallback was NOT invoked; Morant is flagged individually instead).

position: from data/raw/cbbd/draft_picks_all.json (has it directly).
conference: final college season, verbatim string, from shared_features.parquet.
conf_tier: boolean high-major, proposed map below - owner reviews.
pick_overall, log_pick: from the ledger directly.

Run: python3 src/college_model/anchor_covariates.py
"""

from __future__ import annotations

import glob
import json
import sys
import time
from collections import defaultdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from nba_api.stats.endpoints import commonplayerinfo  # noqa: E402

LEDGER_PATH = REPO_ROOT / "data" / "anchors" / "anchor_ledger.csv"
SHARED_FEATURES_PATH = REPO_ROOT / "data" / "college" / "shared_features.parquet"
RAW_CBBD_DIR = REPO_ROOT / "data" / "raw" / "cbbd"
CPI_CACHE_DIR = REPO_ROOT / "data" / "raw" / "nba" / "commonplayerinfo"
OUT_DIR = REPO_ROOT / "data" / "anchors"

SLEEP_SECONDS = 0.8

# Draft dates - a few days' error is immaterial (<0.02y against a 365.25 denominator).
DRAFT_DATES = {
    2017: date(2017, 6, 22), 2018: date(2018, 6, 21), 2019: date(2019, 6, 20),
    2020: date(2020, 11, 18), 2021: date(2021, 7, 29), 2022: date(2022, 6, 23),
    2023: date(2023, 6, 22), 2024: date(2024, 6, 26), 2025: date(2025, 6, 25),
}

# Proposed high-major boolean map - owner reviews before Phase 4 uses it for anything
# beyond this bookkeeping table. Pac-12 counted high-major only through 2023-24 (CBBD
# season<=2024) - it stopped being a real power conference once realignment gutted it
# for 2024-25.
HIGH_MAJOR_ALWAYS = {"ACC", "Big Ten", "Big 12", "SEC", "Big East"}


def conf_tier(conference: str, season: int) -> bool:
    if conference in HIGH_MAJOR_ALWAYS:
        return True
    if conference == "Pac-12" and season <= 2024:
        return True
    return False


# --- years in college -------------------------------------------------

def _roster_min_start_by_athlete_id() -> dict[int, int]:
    out: dict[int, int] = {}
    for f in sorted(glob.glob(str(RAW_CBBD_DIR / "roster_*.json"))):
        for team in json.loads(Path(f).read_text()):
            for p in team["players"]:
                ss = p.get("startSeason")
                if ss is None:
                    continue
                aid = p["id"]
                if aid not in out or ss < out[aid]:
                    out[aid] = ss
    return out


VALIDATION_CASES = {
    # name: (athleteId, final_college_season, expected_years)
    "Frank Mason III": (52701, 2017, 4), "Trae Young": (48324, 2018, 1),
    "Jalen Brunson": (46040, 2018, 3), "Zach Edey": (15234, 2024, 4),
    "Ja Morant": (44342, 2019, 3), "Cooper Flagg": (205, 2025, 1),
    "Zion Williamson": (43429, 2019, 1), "Payton Pritchard": (34903, 2020, 4),
    "Anthony Edwards": (35495, 2020, 1), "RJ Barrett": (43431, 2019, 1),
    "Markelle Fultz": (53509, 2017, 1), "Luka Garza": (31694, 2021, 4),
}


def validate_years_in_college(roster_min_start: dict[int, int]) -> pd.DataFrame:
    rows = []
    for name, (aid, final_season, expected) in VALIDATION_CASES.items():
        min_start = roster_min_start.get(aid)
        computed = final_season - min_start + 1 if min_start is not None else None
        rows.append({"name": name, "athlete_id": aid, "final_college_season": final_season,
                      "min_start_season": min_start, "computed": computed, "expected": expected,
                      "match": computed == expected})
    return pd.DataFrame(rows)


# --- age at draft (network) --------------------------------------------

def _fetch_birthdate(nba_player_id: int) -> str | None:
    """Returns ISO birthdate string or None. Cached immutably - a second run
    for the same player_id never calls the network."""
    CPI_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CPI_CACHE_DIR / f"{nba_player_id}.json"
    if cache_path.exists():
        data = json.loads(cache_path.read_text())
    else:
        last_err = None
        data = None
        for attempt in range(1, 6):
            try:
                resp = commonplayerinfo.CommonPlayerInfo(player_id=nba_player_id, timeout=60)
                df = resp.get_data_frames()[0]
                data = {"BIRTHDATE": df.iloc[0]["BIRTHDATE"] if len(df) else None}
                break
            except Exception as e:
                last_err = e
                wait = min(30, 5 * attempt)
                print(f"    [retry {attempt}/5] player_id={nba_player_id}: {e} - sleeping {wait}s")
                time.sleep(wait)
        if data is None:
            raise last_err
        cache_path.write_text(json.dumps(data))
        time.sleep(SLEEP_SECONDS)
    bd = data.get("BIRTHDATE")
    return bd.split("T")[0] if bd else None


def pull_birthdates(player_ids: list[int]) -> dict[int, str | None]:
    out = {}
    n_cached = sum((CPI_CACHE_DIR / f"{pid}.json").exists() for pid in player_ids)
    print(f"Pulling birthdates for {len(player_ids)} players ({n_cached} already cached)...")
    for i, pid in enumerate(player_ids):
        out[pid] = _fetch_birthdate(int(pid))
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(player_ids)} done")
    return out


# --- position (from raw draft picks) ------------------------------------

def _position_by_athlete_id() -> dict[int, str]:
    picks = json.loads((RAW_CBBD_DIR / "draft_picks_all.json").read_text())
    return {p["athleteId"]: p.get("position") for p in picks}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger = pd.read_csv(LEDGER_PATH)
    anchors = ledger[ledger["status"] == "included"].copy()
    print(f"Building covariates for {len(anchors)} included anchors")

    # --- years in college ---
    roster_min_start = _roster_min_start_by_athlete_id()
    val = validate_years_in_college(roster_min_start)
    disagree_rate = 1 - val["match"].mean()
    print(f"\nyears_in_college validation: {val['match'].sum()}/{len(val)} match "
          f"({disagree_rate:.1%} disagreement, fallback triggers at >10%)")
    print(val.to_string(index=False))
    fallback_triggered = disagree_rate > 0.10
    print(f"Fallback triggered: {fallback_triggered}")

    anchors["min_start_season"] = anchors["matched_athlete_id"].map(roster_min_start)
    anchors["years_in_college"] = anchors["final_college_season"] - anchors["min_start_season"] + 1
    missing_start = anchors["min_start_season"].isna().sum()
    print(f"\nmatched_athlete_id with no roster startSeason found at all: {missing_start}")

    # --- position --- (keyed by matched_athlete_id, not the raw crosswalk athlete_id column -
    # Wesley Iwundu's raw athlete_id is still null even after Step 2's alias recovery, since
    # that recovery only ever touched matched_athlete_id; matched_athlete_id is guaranteed
    # non-null for every "included" row by construction of the eligibility chain's rule 2)
    pos_map = _position_by_athlete_id()
    anchors["position"] = anchors["matched_athlete_id"].map(pos_map)
    print(f"position coverage: {anchors['position'].notna().mean():.1%}")

    # --- conference + conf_tier ---
    sf = pd.read_parquet(SHARED_FEATURES_PATH)
    dupe_keys = sf.duplicated(["player_id_source", "season"], keep=False)
    if dupe_keys.any():
        print(f"\nshared_features.parquet has {dupe_keys.sum()} rows sharing a duplicate "
              f"(player_id_source, season) key (known Phase 1 data quirk, e.g. a mid-season "
              f"transfer double-counted) - keeping first occurrence for the conference lookup.")
    sf_dedup = sf.drop_duplicates(["player_id_source", "season"], keep="first")
    conf_lookup = sf_dedup.set_index(["player_id_source", "season"])["conference"]
    anchors["conference"] = anchors.apply(
        lambda r: conf_lookup.get((r["matched_athlete_id"], r["final_college_season"])), axis=1)
    anchors["conf_tier"] = anchors.apply(
        lambda r: conf_tier(r["conference"], int(r["final_college_season"])), axis=1)
    print(f"conference coverage: {anchors['conference'].notna().mean():.1%}")
    print("conf_tier True conferences present:",
          sorted(anchors.loc[anchors["conf_tier"], "conference"].unique()))
    print("conf_tier False conferences present:",
          sorted(anchors.loc[~anchors["conf_tier"], "conference"].unique()))

    # --- pick_overall, log_pick ---
    anchors["pick_overall"] = anchors["overall"]
    anchors["log_pick"] = np.log(anchors["pick_overall"])

    # --- age at draft (network) ---
    player_ids = anchors["nba_player_id"].dropna().astype(int).unique().tolist()
    birthdates = pull_birthdates(player_ids)
    anchors["birthdate"] = anchors["nba_player_id"].map(lambda pid: birthdates.get(int(pid)) if pd.notna(pid) else None)

    def _age(row):
        if pd.isna(row["birthdate"]) or row["birthdate"] is None:
            return np.nan
        bd = date.fromisoformat(row["birthdate"])
        dd = DRAFT_DATES[int(row["draft_year"])]
        return (dd - bd).days / 365.25

    anchors["age_at_draft"] = anchors.apply(_age, axis=1)
    coverage = anchors["age_at_draft"].notna().mean()
    print(f"\nbirthdate/age coverage: {coverage:.1%} (gate: stop if < 95%)")
    missing = anchors[anchors["age_at_draft"].isna()]
    if len(missing):
        print("Missing birthdate for:")
        print(missing[["draft_year", "player_name_raw", "nba_player_id"]].to_string(index=False))

    out_cols = ["draft_year", "round", "pick", "overall", "player_name_raw", "nba_player_id",
                "matched_athlete_id", "final_college_season", "years_in_college", "min_start_season",
                "position", "conference", "conf_tier", "pick_overall", "log_pick",
                "birthdate", "age_at_draft"]
    anchors[out_cols].to_csv(OUT_DIR / "anchor_covariates.csv", index=False)
    val.to_csv(OUT_DIR / "years_in_college_validation.csv", index=False)
    print(f"\nWrote {OUT_DIR / 'anchor_covariates.csv'} ({len(anchors)} rows)")
    print(f"Wrote {OUT_DIR / 'years_in_college_validation.csv'}")

    if coverage < 0.95:
        raise AssertionError(f"birthdate coverage {coverage:.1%} < 95% gate - stop and ask")


if __name__ == "__main__":
    main()
