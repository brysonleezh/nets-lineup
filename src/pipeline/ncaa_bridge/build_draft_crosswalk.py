"""
Phase 1 Step 4 (data-only) - draft-pick <-> CBBD player crosswalk for
anchor-set construction. Draft classes 2017-2026, from the already-cached
data/raw/cbbd/draft_picks_all.json (no new API calls).

Path tagging: CBBD's own `sourceTeamCollegeId` field cleanly separates
NCAA-path picks (non-null) from everyone else (null); for the null group,
`sourceTeamLeagueAffiliation` is populated for every single row (verified:
non-null exactly where sourceTeamCollegeId is null, no ambiguous/blank
cases) and gives the specific tag - "G League Ignite", "Overtime Elite",
or a country/club name (International).

Matching, in priority order:
  1. DIRECT athleteId join against that draft year's player_season table.
     Not in the original spec's method (which assumed name-matching was
     necessary) - discovered live that CBBD's draft/picks and
     stats/player/season endpoints share the SAME athleteId space (same
     internal athlete record, not two sources needing reconciliation).
     Tested against all 486 NCAA-path picks: 461/486 = 94.9% direct hit
     rate on its own, already clearing the spec's 85% gate before any
     name-matching at all. Reported as a deviation from the spec's
     described method, not silently substituted.
  2. Name-normalized (name, year, team) match, for the direct-join misses.
  3. Name-normalized (name, year) match (team fallback), for method-2 misses.
  4. Remainder -> manual review.

Run: python3 src/pipeline/ncaa_bridge/build_draft_crosswalk.py
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CACHE_DIR = REPO_ROOT / "data" / "raw" / "cbbd"
ANCHORS_DIR = REPO_ROOT / "data" / "anchors"

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def normalize_name(name: str) -> str:
    """lowercase, strip punctuation, drop suffixes, transliterate diacritics."""
    if not name:
        return ""
    # transliterate: NFKD decompose then drop combining marks (é->e, etc.)
    # handles characters NFKD doesn't decompose (đ, ø) via an explicit map first
    manual_map = {"đ": "d", "Đ": "D", "ø": "o", "Ø": "O", "ł": "l", "Ł": "L"}
    for k, v in manual_map.items():
        name = name.replace(k, v)
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    name = name.lower()
    name = re.sub(r"[^a-z0-9\s]", "", name)  # strip punctuation
    tokens = [t for t in name.split() if t not in SUFFIXES]
    return " ".join(tokens)


def tag_path(pick: dict) -> str:
    if pick.get("sourceTeamCollegeId") is not None:
        return "NCAA"
    aff = pick.get("sourceTeamLeagueAffiliation")
    if aff == "G League Ignite":
        return "G-League Ignite"
    if aff == "Overtime Elite":
        return "OTE"
    if aff:
        return f"International ({aff})"
    return "Other/Unknown"


def load_season_players(season: int) -> list[dict]:
    f = CACHE_DIR / f"player_season_{season}.json"
    return json.loads(f.read_text()) if f.exists() else []


def main() -> None:
    ANCHORS_DIR.mkdir(parents=True, exist_ok=True)
    picks = json.loads((CACHE_DIR / "draft_picks_all.json").read_text())
    picks = [p for p in picks if 2017 <= p["year"] <= 2026]

    season_players: dict[int, list[dict]] = {yr: load_season_players(yr) for yr in range(2017, 2027)}
    by_athlete_id: dict[int, dict[int, dict]] = {
        yr: {row["athleteId"]: row for row in rows} for yr, rows in season_players.items()
    }
    # name-index for fallback matching: (norm_name, year) -> [rows], (norm_name, year, norm_team) -> [rows]
    name_team_index: dict[tuple, list[dict]] = {}
    name_index: dict[tuple, list[dict]] = {}
    for yr, rows in season_players.items():
        for row in rows:
            nn = normalize_name(row["name"])
            nt = normalize_name(row["team"])
            name_team_index.setdefault((nn, yr, nt), []).append(row)
            name_index.setdefault((nn, yr), []).append(row)

    results = []
    for p in picks:
        path = tag_path(p)
        row = {
            "draft_year": p["year"], "round": p["round"], "pick": p["pick"], "overall": p["overall"],
            "player_name_raw": p["name"], "path": path,
            "source_team_location": p.get("sourceTeamLocation"),
            "athlete_id": p["athleteId"], "matched_athlete_id": None,
            "final_college_season": None, "match_method": None,
        }
        if path == "NCAA":
            yr = p["year"]
            direct = by_athlete_id.get(yr, {}).get(p["athleteId"])
            if direct:
                row["matched_athlete_id"] = direct["athleteId"]
                row["final_college_season"] = yr
                row["match_method"] = "direct_athlete_id"
            else:
                nn = normalize_name(p["name"])
                nt = normalize_name(p.get("sourceTeamLocation", ""))
                cands = name_team_index.get((nn, yr, nt), [])
                if len(cands) == 1:
                    row["matched_athlete_id"] = cands[0]["athleteId"]
                    row["final_college_season"] = yr
                    row["match_method"] = "name_year_team"
                else:
                    cands2 = name_index.get((nn, yr), [])
                    if len(cands2) == 1:
                        row["matched_athlete_id"] = cands2[0]["athleteId"]
                        row["final_college_season"] = yr
                        row["match_method"] = "name_year_only"
                    else:
                        row["match_method"] = "unmatched_needs_review"
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(ANCHORS_DIR / "crosswalk_draft.csv", index=False)

    ncaa = df[df["path"] == "NCAA"]
    n_matched = ncaa["matched_athlete_id"].notna().sum()
    print(f"Total picks 2017-2026: {len(df)}")
    print(df["path"].value_counts().to_string())
    print()
    print(f"NCAA-path picks: {len(ncaa)}, matched: {n_matched} ({100*n_matched/len(ncaa):.1f}%)")
    print(ncaa["match_method"].value_counts(dropna=False).to_string())
    print()

    per_class = (ncaa.groupby("draft_year")
                 .agg(n_picks=("matched_athlete_id", "size"),
                      n_matched=("matched_athlete_id", lambda s: s.notna().sum()))
                 .reset_index())
    per_class["match_rate"] = per_class["n_matched"] / per_class["n_picks"]
    per_class.to_csv(ANCHORS_DIR / "crosswalk_match_rate_by_class.csv", index=False)
    print("Per-class match rate:")
    print(per_class.to_string(index=False))

    failures = ncaa[ncaa["match_method"] == "unmatched_needs_review"]
    failures.head(20).to_csv(ANCHORS_DIR / "crosswalk_failures_top20.csv", index=False)
    print(f"\n{len(failures)} unmatched NCAA-path picks (top 20 written to crosswalk_failures_top20.csv):")
    for _, r in failures.head(20).iterrows():
        print(f"  {r['player_name_raw']!r} ({r['draft_year']}, {r['source_team_location']})")

    overrides_path = ANCHORS_DIR / "name_overrides.csv"
    pd.DataFrame(columns=["player_name_raw", "draft_year", "override_athlete_id", "note"]).to_csv(
        overrides_path, index=False)
    print(f"\nWrote empty template: {overrides_path}")

    overall_rate = n_matched / len(ncaa)
    if overall_rate < 0.85:
        print(f"\n*** GATE FAILED: {100*overall_rate:.1f}% < 85% - stop and diagnose ***")
    else:
        print(f"\nGate cleared: {100*overall_rate:.1f}% >= 85%")


if __name__ == "__main__":
    main()
