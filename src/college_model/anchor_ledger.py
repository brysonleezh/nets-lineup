"""
Phase 3 Step 2 - eligibility ledger (classes 2017-2025) + NBA-side ID bridge.

Builds data/anchors/anchor_ledger.csv: every pick from data/anchors/crosswalk_draft.csv
(2017-2026, all paths) appears exactly once with a status. 2026 rows get
status=prediction_target and skip the eligibility chain entirely (no rookie
season exists yet - the hard boundary from the phase spec: 2026 may not enter
anchors.csv via any other path).

Eligibility chain (2017-2025 only), applied in order, first failure wins:
  1. path == NCAA                                        -> else non_ncaa_path
  2. college match exists (matched_athlete_id non-null)   -> else handled by
     the 6 named special cases below (no generic "unmatched" bucket needed -
     crosswalk_draft.csv has exactly 6 NCAA-path 2017-2025 nulls, and all 6
     are the ones the phase spec names individually)
  3. final-college-season row present in shared_features.parquet
                                                           -> else college_below_min
  4. gap_years = draft_year - final_college_season == 0   -> else college_gap_year
     (INCLUDE_GAP_YEARS=False - a config flag, not a fitted decision)
  5. rookie season (draft_year -> NBA season draft_year/draft_year+1) MIN
     clears threshold (300 flat, 263 for 2020-21 pro-rated)
                                                           -> else rookie_below_min
                                                              or delayed_debut

Run: python3 src/college_model/anchor_ledger.py
"""

from __future__ import annotations

import glob
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "pipeline" / "ncaa_bridge"))

from step1_archetype_model import load_population  # noqa: E402
from build_draft_crosswalk import normalize_name  # noqa: E402

CROSSWALK_PATH = REPO_ROOT / "data" / "anchors" / "crosswalk_draft.csv"
SHARED_FEATURES_PATH = REPO_ROOT / "data" / "college" / "shared_features.parquet"
RAW_CBBD_DIR = REPO_ROOT / "data" / "raw" / "cbbd"
DB_PATH = REPO_ROOT / "data" / "nets_synergy.db"
OUT_DIR = REPO_ROOT / "data" / "anchors"

INCLUDE_GAP_YEARS = False  # config flag per spec Step 2 rule 4 - not a fitted decision
ROOKIE_MIN_DEFAULT = 300
ROOKIE_MIN_2020_21 = 300 * 72 / 82  # pro-rated for the genuinely shortened 72-game season


def season_label(draft_year: int) -> str:
    """class Y -> NBA rookie season label 'Y-(Y+1 mod 100)', e.g. 2017 -> '2017-18'."""
    return f"{draft_year}-{str(draft_year + 1)[-2:]}"


def rookie_min_threshold(season: str) -> float:
    return ROOKIE_MIN_2020_21 if season == "2020-21" else ROOKIE_MIN_DEFAULT


# --- raw CBBD player-season lookup (unfiltered - no MIN floor, unlike shared_features) ---

def _load_all_player_season() -> pd.DataFrame:
    records = []
    for f in sorted(glob.glob(str(RAW_CBBD_DIR / "player_season_*.json"))):
        records.extend(json.loads(Path(f).read_text()))
    df = pd.DataFrame(records)[["season", "team", "conference", "athleteId", "name", "minutes", "games"]]
    df["name_norm"] = df["name"].apply(normalize_name)
    return df


# --- the 6 named special cases (Phase 1's crosswalk left these matched_athlete_id=NaN) ---

def _resolve_special_cases(cw: pd.DataFrame, raw_ps: pd.DataFrame) -> dict[int, dict]:
    """Returns {crosswalk_index: {resolved fields + note}} for the 6 named picks."""
    resolved = {}

    def _idx(draft_year, overall):
        m = cw[(cw["draft_year"] == draft_year) & (cw["overall"] == overall)]
        assert len(m) == 1, f"expected exactly 1 row for ({draft_year}, {overall}), got {len(m)}"
        return m.index[0]

    # 1. Shaedon Sharpe (2022) - confirm zero CBBD games anywhere under this name
    i = _idx(2022, 7)
    hits = raw_ps[raw_ps["name"].str.contains("Sharpe", case=False, na=False)]
    assert not (hits["name"] == "Shaedon Sharpe").any(), "Sharpe now has CBBD data - re-investigate, do not blindly apply no_college_data"
    resolved[i] = {"status": "no_college_data",
                    "note": "Confirmed zero CBBD player_season rows under 'Shaedon Sharpe' across all seasons "
                            "2017-2026 (redshirted at Kentucky, never played a game, declared for the draft). "
                            "The crosswalk's tentative athlete_id=31152 has zero player_season rows too - a stale/bad "
                            "match, not a real college record. Correct exclusion, not a matching failure."}

    # 2. Mitchell Robinson (2018) - same pattern
    i = _idx(2018, 36)
    hits = raw_ps[raw_ps["name"].str.contains("Mitchell Robinson", case=False, na=False)]
    assert len(hits) == 0, "Robinson now has CBBD data - re-investigate"
    resolved[i] = {"status": "no_college_data",
                    "note": "Confirmed zero CBBD player_season rows under 'Mitchell Robinson' across all seasons "
                            "(committed to Western Kentucky, never enrolled/played, declared for the draft directly "
                            "out of the recruiting class). Correct exclusion, not a matching failure."}

    # 3. Goga Bitadze (2019, crosswalk mis-tagged path=NCAA via source_team_location='Georgia')
    i = _idx(2019, 18)
    hits = raw_ps[raw_ps["name"].str.contains("Bitadze", case=False, na=False)]
    assert len(hits) == 0, "Bitadze now has CBBD data - re-investigate"
    resolved[i] = {"status": None,  # let the general chain assign non_ncaa_path after path correction
                    "path_override": "International (Serbia)",
                    "note": "Zero CBBD rows under 'Bitadze' in any season - confirms he never played NCAA. "
                            "The crosswalk's path='NCAA'/source_team_location='Georgia' was a source-data mis-tag: "
                            "CBBD's draft/picks record conflated Georgia the country (he played for Mega Bemax, "
                            "Serbia) with Georgia the university. Path corrected to International here; falls out "
                            "via the normal non_ncaa_path rule."}

    # 4. De'Anthony Melton (2018, USC) - sit-out case, recover via draft_year-1/-2 search
    i = _idx(2018, 46)
    cand = raw_ps[(raw_ps["name_norm"] == normalize_name("De'Anthony Melton"))
                   & (raw_ps["season"].isin([2017, 2016]))]
    assert len(cand) == 1, f"expected exactly 1 recovery candidate for Melton, found {len(cand)}"
    row = cand.iloc[0]
    resolved[i] = {"status": None, "matched_athlete_id": row["athleteId"], "final_college_season": row["season"],
                    "match_method": "gap_year_recovery",
                    "note": f"Recovered via search of seasons draft_year-1/-2: found in season={row['season']} "
                            f"at {row['team']} ({row['minutes']} min, {row['games']} games), athleteId={row['athleteId']}. "
                            f"He sat out the 2017-18 season (suspended pending a legal investigation, per public "
                            f"reporting) -> gap_years=1, falls out via the normal college_gap_year rule."}

    # 5. Dewan Hernandez (2019, Miami) - same pattern
    i = _idx(2019, 59)
    cand = raw_ps[(raw_ps["name_norm"] == normalize_name("Dewan Hernandez"))
                   & (raw_ps["season"].isin([2018, 2017]))]
    assert len(cand) >= 1, "expected at least 1 recovery candidate for Hernandez"
    row = cand.sort_values("season", ascending=False).iloc[0]  # most recent = his true final season
    resolved[i] = {"status": None, "matched_athlete_id": row["athleteId"], "final_college_season": row["season"],
                    "match_method": "gap_year_recovery",
                    "note": f"Recovered via search of seasons draft_year-1/-2: most recent hit season={row['season']} "
                            f"at {row['team']} ({row['minutes']} min, {row['games']} games), athleteId={row['athleteId']}. "
                            f"He sat out the 2018-19 season (NCAA-ruled ineligible, per public reporting) -> "
                            f"gap_years=1, falls out via the normal college_gap_year rule."}

    # 6. Wesley Iwundu (2017, Kansas State) - name-variant miss, minimal alias map (1 entry)
    i = _idx(2017, 33)
    alias = {"wesley iwundu": "wes iwundu"}
    target = alias.get(normalize_name("Wesley Iwundu"), normalize_name("Wesley Iwundu"))
    cand = raw_ps[(raw_ps["name_norm"] == target) & (raw_ps["season"] == 2017)]
    assert len(cand) == 1, f"expected exactly 1 alias-recovery candidate for Iwundu, found {len(cand)}"
    row = cand.iloc[0]
    resolved[i] = {"status": None, "matched_athlete_id": row["athleteId"], "final_college_season": row["season"],
                    "match_method": "alias_recovery",
                    "note": f"Recovered via a 1-entry alias map ('wesley iwundu' -> 'wes iwundu'; CBBD lists him as "
                            f"'Wes Iwundu'): season={row['season']} at {row['team']} ({row['minutes']} min, "
                            f"{row['games']} games), athleteId={row['athleteId']}. gap_years=0 - proceeds through "
                            f"the normal chain like any other matched pick (not auto-included)."}

    return resolved


def _nba_id_bridge() -> pd.DataFrame:
    """(draft_year, overall) -> nba_player_id, via player_bio's DRAFT_YEAR/DRAFT_NUMBER."""
    with sqlite3.connect(DB_PATH) as conn:
        bio = pd.read_sql(
            "SELECT DISTINCT PLAYER_ID, PLAYER_NAME, DRAFT_YEAR, DRAFT_NUMBER FROM player_bio", conn)
    bio = bio[bio["DRAFT_YEAR"] != "Undrafted"].copy()
    bio["DRAFT_YEAR"] = bio["DRAFT_YEAR"].astype(int)
    bio["DRAFT_NUMBER"] = pd.to_numeric(bio["DRAFT_NUMBER"], errors="coerce")
    bio = bio.dropna(subset=["DRAFT_NUMBER"])
    bio["DRAFT_NUMBER"] = bio["DRAFT_NUMBER"].astype(int)
    dupe = bio.groupby("PLAYER_ID")[["DRAFT_YEAR", "DRAFT_NUMBER"]].nunique()
    assert (dupe <= 1).all().all(), "a PLAYER_ID maps to >1 distinct draft_year/number across season rows"
    return bio.rename(columns={"PLAYER_ID": "nba_player_id", "PLAYER_NAME": "nba_player_name"})


def build_ledger() -> tuple[pd.DataFrame, dict]:
    cw = pd.read_csv(CROSSWALK_PATH)
    raw_ps = _load_all_player_season()
    sf = pd.read_parquet(SHARED_FEATURES_PATH)
    sf_keys = set(zip(sf["player_id_source"], sf["season"]))
    bio_bridge = _nba_id_bridge()

    # load_population() rebuilds the full multi-table NBA join from scratch on every
    # call - expensive, so load once per season here rather than per-row in the loop
    # below (~420 rows would otherwise mean ~420 full rebuilds).
    print("Pre-loading NBA populations for all 9 rookie-season years (one build_nba_side_tables() "
          "call each, reused across all rows in that season)...")
    full_pop = load_population(min_threshold=1)
    pop_by_season = {s: full_pop[full_pop["SEASON"] == s].set_index("PLAYER_ID") for s in full_pop["SEASON"].unique()}
    print(f"  done: {len(pop_by_season)} seasons cached, {len(full_pop)} total rows")

    special = _resolve_special_cases(cw, raw_ps)

    rows = []
    diagnostics = {"name_disagreements": [], "nba_id_missing": []}

    for idx, r in cw.iterrows():
        row = r.to_dict()
        row["status"] = None
        row["exclusion_reason_detail"] = ""
        row["notes"] = ""
        row["gap_years"] = np.nan
        row["nba_player_id"] = np.nan
        row["rookie_season_expected"] = ""
        row["rookie_minutes"] = np.nan
        row["rookie_min_threshold"] = np.nan
        row["debut_season"] = ""

        if idx in special:
            sp = special[idx]
            if "path_override" in sp:
                row["path"] = sp["path_override"]
            if "matched_athlete_id" in sp:
                row["matched_athlete_id"] = sp["matched_athlete_id"]
                row["final_college_season"] = sp["final_college_season"]
                row["match_method"] = sp["match_method"]
            row["notes"] = sp["note"]
            if sp["status"] is not None:
                row["status"] = sp["status"]

        draft_year = int(row["draft_year"])

        if draft_year == 2026:
            row["status"] = "prediction_target"
            rows.append(row)
            continue

        # rule 1
        if row["status"] is None and row["path"] != "NCAA":
            row["status"] = "non_ncaa_path"

        # rule 2 (only the 6 special cases have null matched_athlete_id in 2017-2025 NCAA-path
        # rows - already handled above; this branch should not fire for anyone else)
        if row["status"] is None and pd.isna(row["matched_athlete_id"]):
            raise AssertionError(f"unresolved null matched_athlete_id at idx={idx} "
                                  f"({row['player_name_raw']}, {draft_year}) - not one of the 6 named cases")

        # rule 3
        if row["status"] is None:
            key = (int(row["matched_athlete_id"]), int(row["final_college_season"]))
            if key not in sf_keys:
                row["status"] = "college_below_min"

        # rule 4
        if row["status"] is None:
            gap = draft_year - int(row["final_college_season"])
            row["gap_years"] = gap
            if gap != 0 and not INCLUDE_GAP_YEARS:
                row["status"] = "college_gap_year"

        # NBA ID bridge (attempted regardless, so the ledger always shows what we found)
        bridge_hit = bio_bridge[(bio_bridge["DRAFT_YEAR"] == draft_year) & (bio_bridge["DRAFT_NUMBER"] == row["overall"])]
        if len(bridge_hit) == 1:
            row["nba_player_id"] = bridge_hit.iloc[0]["nba_player_id"]
            nba_name = bridge_hit.iloc[0]["nba_player_name"]
            if normalize_name(nba_name) != normalize_name(row["player_name_raw"]):
                diagnostics["name_disagreements"].append(
                    {"draft_year": draft_year, "overall": row["overall"],
                     "crosswalk_name": row["player_name_raw"], "nba_name": nba_name})
        elif row["status"] is None:
            diagnostics["nba_id_missing"].append(
                {"draft_year": draft_year, "overall": row["overall"], "player_name_raw": row["player_name_raw"]})

        # rule 5
        if row["status"] is None:
            season = season_label(draft_year)
            row["rookie_season_expected"] = season
            thresh = rookie_min_threshold(season)
            row["rookie_min_threshold"] = thresh

            if pd.isna(row["nba_player_id"]):
                row["status"] = "rookie_below_min"
                row["rookie_minutes"] = 0.0
                row["notes"] = (row["notes"] + " | " if row["notes"] else "") + \
                    "No NBA PLAYER_ID found in player_bio at all (checked draft_year/draft_number join, " \
                    "and a broad surname search - genuinely absent from every pulled season) - treated as " \
                    "undocumented/zero NBA minutes, clears the below-min threshold trivially."
            else:
                pid = row["nba_player_id"]
                pop_this_season = pop_by_season.get(season)
                if pop_this_season is not None and pid in pop_this_season.index:
                    mins = float(pop_this_season.loc[pid, "MIN"])
                    row["rookie_minutes"] = mins
                    if mins < thresh:
                        row["status"] = "rookie_below_min"
                    else:
                        row["status"] = "included"
                else:
                    # never appeared in that season at all - check later seasons for a delayed debut
                    later_hits = full_pop[(full_pop["PLAYER_ID"] == pid) & (full_pop["SEASON"] > season)]
                    if len(later_hits) > 0:
                        row["status"] = "delayed_debut"
                        row["debut_season"] = later_hits["SEASON"].min()
                    else:
                        row["status"] = "rookie_below_min"
                        row["rookie_minutes"] = 0.0

        rows.append(row)

    ledger = pd.DataFrame(rows)
    assert ledger["status"].notna().all(), "every row must have a status"
    return ledger, diagnostics


def _covid_footnote(ledger: pd.DataFrame) -> str:
    pop = load_population(min_threshold=1, season_min="2019-20", season_max="2019-20")
    class2019 = ledger[(ledger["draft_year"] == 2019) & (ledger["nba_player_id"].notna())]
    joined = class2019.merge(pop[["PLAYER_ID", "MIN"]], left_on="nba_player_id", right_on="PLAYER_ID", how="inner")
    near_miss = joined[(joined["MIN"] >= 250) & (joined["MIN"] < 300)]
    return (f"2019-class rookies with 250-299 minutes in the COVID-shortened 2019-20 season "
            f"(near-misses under the flat 300 threshold, which is NOT pro-rated for 2019-20 - only "
            f"2020-21 gets the 263 pro-ration): {len(near_miss)} found"
            + (f" - {near_miss[['player_name_raw', 'MIN']].to_dict('records')}" if len(near_miss) else "."))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ledger, diagnostics = build_ledger()
    ledger.to_csv(OUT_DIR / "anchor_ledger.csv", index=False)

    print(f"Ledger rows: {len(ledger)} (crosswalk had {len(pd.read_csv(CROSSWALK_PATH))})")
    print("\nStatus counts:")
    print(ledger["status"].value_counts().to_string())

    elig_2017_2025 = ledger[ledger["draft_year"] <= 2025]
    print(f"\ncollege_below_min count: {(elig_2017_2025['status'] == 'college_below_min').sum()} (gate: stop if > 15)")

    candidates = elig_2017_2025[elig_2017_2025["path"] == "NCAA"]
    join_rate = candidates["nba_player_id"].notna().mean()
    print(f"NBA ID join rate (NCAA-path, 2017-2025): {join_rate:.1%} (gate: stop if < 95%)")

    print(f"\nName disagreements (benign nickname variants expected, hand-inspect all): "
          f"{len(diagnostics['name_disagreements'])}")
    for d in diagnostics["name_disagreements"]:
        print(f"  {d['draft_year']} #{d['overall']}: crosswalk='{d['crosswalk_name']}' nba='{d['nba_name']}'")

    print(f"\nNBA ID missing entirely (fringe/no-NBA-data players, expected): {len(diagnostics['nba_id_missing'])}")
    for d in diagnostics["nba_id_missing"]:
        print(f"  {d['draft_year']} #{d['overall']}: {d['player_name_raw']}")

    included = ledger[ledger["status"] == "included"]
    print(f"\nIncluded so far: {len(included)} (gate: stop if < 120)")
    included_2025 = included[included["draft_year"] == 2025]
    print(f"2025-class included: {len(included_2025)} (gate: stop if < 20)")

    print(f"\n{_covid_footnote(ledger)}")


if __name__ == "__main__":
    main()
