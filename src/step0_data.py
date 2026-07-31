"""
Step 0 — data collection + processing. Combines what used to be two
separate files (data.py, features.py) into one, per request, since together
they're "collect the raw data, then process it into the feature table" —
one logical stage of the pipeline.

PART A (collection): pulls league-wide player and lineup data from NBA.com
(via nba_api) and Basketball-Reference, caches every result to SQLite
immediately, so a crash or rate-limit block never loses completed work.

PART B (processing): joins those raw tables into one player-season feature
table matching the "Scouting Anyone" paper's Table 3.3 NBA feature spec,
z-scored and ready for Step 1's ADA fit.

PART C (inspection): a rich-terminal (or --markdown, for README.md) viewer
over every raw table — name, description, shape, row preview, and a full
column dictionary (meaning/type/missing-count/example) per column, derived
from CLAUDE.md's feature spec, self-describing headers, and standard NBA/
Basketball-Reference stat terminology, not invented. Originally its own
file (inspect_data.py); folded in here since it's still part of "look at
what step 0 produced."

USAGE (Part C / inspection, the default when run with no special flags)
    python3 src/step0_data.py                 # display all tables
    python3 src/step0_data.py player_bio       # display just one table
    python3 src/step0_data.py --no-preview     # skip the row-preview panels
    python3 src/step0_data.py --markdown       # print as GitHub-flavored
                                                                # Markdown (for README.md)
    python3 src/step0_data.py --pull           # run Part A's data pull instead
                                                                # of inspecting (the heavy,
                                                                # deliberate action — not the
                                                                # default, so running this file
                                                                # to just look at data never
                                                                # accidentally triggers a pull)

nba_api is a thin wrapper around the stats.nba.com API, which is not officially supported by the NBA. It is subject to change or break at any time, and may
not be used for commercial purposes. See https://github.com/swar/nba_api

# Six functions to build up tables using SQLite database (Part A):
- pull_player_bio:
    Calls LeagueDashPlayerBioStats, one call per season. Returns league-wide height, weight, draft year/round, plus some rough per-game stats. Saved to player_bio.
- pull_player_advanced:
    Calls LeagueDashPlayerStats with measure_type="Advanced". Returns USG%, TS%, AST%, OFF/DEF/NET RATING, PACE, PIE, etc. Saved to player_advanced.
- pull_player_base:
    Calls LeagueDashPlayerStats, but measure_type="Base" — raw counting stats (FG/3PT/FT/REB/AST/STL/BLK/PTS). Saved to player_base.
- pull_bref_shooting:
    Doesn't use nba_api — uses requests to fetch the Basketball-Reference "shooting" page directly (_bref_get handles the request and rate-limit retries), then pd.read_html parses the <table id="shooting"> into a DataFrame. This is the one that gives the paper-exact shot-distance buckets, corner-3 rate, average shot distance, and %AST'd-2P/3P. Saved to player_shooting_bref.
- pull_play_type:
    Calls SynergyPlayTypes, one call per (season, play type) pair — that's why the runner loops over all 9 play types. Saved to player_playtype, with a PLAY_TYPE column marking which one.
- pull_lineups:
    Calls LeagueDashLineups for one team's lineup combos of a given size (2/3/4/5-man) in a given season. The runner loops over all 4 group sizes × 30 teams to dodge the 2000-row API cap. Saved to lineups.

Part B (processing) input: raw data tables (player_bio, player_base, player_advanced, player_advanced_bref,
       player_shooting_bref, player_playtype)
       - player_advanced_bref added: player_advanced (NBA.com) has no STL%, BLK%, or BPM.
       - lineups dropped: that table is for the synergy-matrix step later, not per-player
         archetype features, so it doesn't belong in this join.
Part B output and preprocessing: one dataframe, one row per player-season, columns = idenitity info + "Scouting Anyone: Probabilistic Player Archetypes for
Any League" paper table 3.3 features, filtered to >= 300 minutes, zero Nans left in it, cuz ADA can not handle NaN, plus a seperate z-scoring step.

A few points to note about the raw data tables (Part B):
- NBA.com tables key on PLAYER_ID; the two Basketball-Reference tables only have player names ->
  needs a normalized-name join, and the match rate needs to actually be checked.
- player_advanced_bref and player_advanced both have TS%/USG%/AST%/etc. under different column
  names/casing - pick one source per feature, don't silently mix.
- Traded players are represented differently in NBA.com vs. B-Ref tables - check
  a known traded player in both before joining.
- Not every MIN-like column is a season total - verify against a known player
  before trusting it for the >=300 filter.
- Player got traded mid-season
"""
import time
import sqlite3
import sys
from io import StringIO
from pathlib import Path

import pandas as pd
import requests
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashplayerbiostats,
    synergyplaytypes,
    leaguedashlineups,
    teamplayeronoffsummary,
)
from nba_api.stats.static import teams
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# AI-ASSISTED (Claude Code, chat) - Prompt: "if I want to deploy this
# application on streamlit... could you [build a filtered deploy db]" -
# nets_synergy.db is 211MB, 95% of it the `lineups` table's 3/4/5-man and
# pre-2025-26 rows the live app never queries (verified by tracing every
# live call site, not assumed - see src/pipeline/build_deploy_db.py's own
# docstring). That full file is intentionally gitignored (too large, and
# a straight `git push` of a >100MB file is rejected by GitHub outright).
# A from-scratch git clone (e.g. Streamlit Community Cloud, which has no
# separate file-upload channel - git IS the only deploy path) would
# therefore have no database at all unless something else provides one.
# Falls back to the small, filtered, git-trackable
# `nets_synergy_deploy.db` (built by build_deploy_db.py, ~13MB, same
# schema, same tables) ONLY when the full file is absent - a local dev
# environment or a self-hosted server with the real file copied on
# separately is completely unaffected by this fallback ever triggering.
# Not AI: the decision to solve this with a fallback path rather than
# committing the full 211MB db (impossible without Git LFS) or requiring
# a manual post-clone step - the user's own call, after asking to verify
# whether the full db was even needed live in the first place.
_DEPLOY_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nets_synergy_deploy.db"
DB_PATH = Path(__file__).resolve().parent.parent / "data" / "nets_synergy.db"
if not DB_PATH.exists() and _DEPLOY_DB_PATH.exists():
    DB_PATH = _DEPLOY_DB_PATH

"""
AI MODIFIED (Claude Code)
Prompt: Extend SEASONS back to 2017-18 to match the "Scouting Anyone" paper's
own NBA dataset window (2017-18 - 2024-25, per the paper excerpt), so the
K-selection sweep is comparable to the paper's rather than confounded by a
different era of the league. 2025-26 stays in the list too — it's the
intended holdout/validation season (archetype model trains on 2017-18-
2024-25, then 2025-26 gets scored against those archetypes via
score_all_players()'s transform-based approach, not included in training).
_pull_if_needed() means re-running this only pulls the 3 new seasons
(2017-18/2018-19/2019-20) — the 6 already in data/nets_synergy.db are
skipped, not re-fetched.
"""
SEASONS = [
    "2017-18", "2018-19", "2019-20",
    "2020-21", "2021-22", "2022-23", "2023-24", "2024-25", "2025-26",
]
PLAY_TYPES = [
    "Isolation", "PRBallHandler", "PRRollman", "Postup",
    "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound",
]
GROUP_QUANTITIES = (2, 3, 4, 5) # How many players are on the court together for a lineup. 2=duos, 3=trios, 4=quads, 5=full lineups.
TEAM_IDS = [t["id"] for t in teams.get_teams()]
SLEEP_SECONDS = 0.8
BREF_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
}
BREF_SLEEP_SECONDS = 3.0  # B-Ref rate-limits harder than stats.nba.com

# Part C (inspection) constants
PREVIEW_ROWS = 5
PREVIEW_MAX_COLS = 6  # wide tables (up to 80 cols) get capped here for readability;
                      # every column still appears in the column dictionary below
CONFIRMED = "confirmed"
INFERRED = "best-effort inference"
UNKNOWN = "Unknown / needs clarification"
console = Console()


# ============================================================================
# PART A — DATA COLLECTION
# ============================================================================

def _save(df: pd.DataFrame, table: str) -> None:
    """Append a pull result to SQLite, creating the table on first write."""
    if df.empty:
        print(f"  [skip] {table}: empty result")
        return
    with sqlite3.connect(DB_PATH) as conn:
        df.to_sql(table, conn, if_exists="append", index=False)
    print(f"  [saved] {table}: {len(df)} rows")


def _already_saved(table: str, **filters) -> bool:
    """Check whether this slice is already cached, so re-running the script
    after a crash/rate-limit block resumes instead of re-pulling (and
    duplicating) everything that already succeeded."""
    if not DB_PATH.exists():
        return False
    with sqlite3.connect(DB_PATH) as conn:
        cur = conn.cursor()
        try:
            where = " AND ".join(f'"{k}" = ?' for k in filters)
            cur.execute(f'SELECT 1 FROM "{table}" WHERE {where} LIMIT 1', list(filters.values()))
            return cur.fetchone() is not None
        except sqlite3.OperationalError:
            return False  # table doesn't exist yet


def _pull_if_needed(table: str, filters: dict, pull_fn, sleep_seconds: float = SLEEP_SECONDS) -> None:
    if _already_saved(table, **filters):
        print(f"  [cached] {table} {filters}")
        return
    _save(pull_fn(), table)
    time.sleep(sleep_seconds)


def _call_with_retry(endpoint_cls, *, retries=6, **kwargs):
    """stats.nba.com times out and rate-limits (empty body -> JSONDecodeError)
    often under sustained load — back off hard and retry before giving up."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return endpoint_cls(timeout=60, **kwargs)
        except Exception as e:
            last_err = e
            wait = min(60, 5 * attempt)
            print(f"  [retry {attempt}/{retries}] {endpoint_cls.__name__}: {e} — sleeping {wait}s")
            time.sleep(wait)
    raise last_err


def pull_player_bio(season: str) -> pd.DataFrame:
    resp = _call_with_retry(leaguedashplayerbiostats.LeagueDashPlayerBioStats, season=season)
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    return df


def pull_player_advanced(season: str) -> pd.DataFrame:
    resp = _call_with_retry(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
    )
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    return df


def pull_player_base(season: str) -> pd.DataFrame:
    resp = _call_with_retry(
        leaguedashplayerstats.LeagueDashPlayerStats,
        season=season,
        measure_type_detailed_defense="Base",
        per_mode_detailed="Totals",
    )
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    return df


def _season_to_bref_year(season: str) -> int:
    """Basketball-Reference URLs use the season's END year, e.g. '2024-25' -> 2025."""
    return int(season.split("-")[0]) + 1


def _bref_get(url: str, *, retries=3) -> str:
    last_err = None
    for attempt in range(1, retries + 1):
        resp = requests.get(url, headers=BREF_HEADERS, timeout=30)
        if resp.status_code == 200:
            resp.encoding = "utf-8"  # B-Ref doesn't send a charset header; requests defaults to
            return resp.text          # ISO-8859-1 and mangles accented names (e.g. Jokić, Dončić)
        if resp.status_code == 429:
            wait = int(resp.headers.get("Retry-After", 30))
            print(f"  [rate limited] sleeping {wait}s...")
            time.sleep(wait)
            continue
        last_err = f"HTTP {resp.status_code}"
        print(f"  [retry {attempt}/{retries}] {url}: {last_err}")
        time.sleep(3 * attempt)
    raise RuntimeError(f"failed to fetch {url}: {last_err}")


def pull_bref_shooting(season: str) -> pd.DataFrame:
    """Shot-distance buckets (0-3/3-10/10-16/16-3P), corner-3 rate, and
    %AST'd-2P/3P in one table — matches paper Table 3.3's NBA feature spec."""
    year = _season_to_bref_year(season)
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_shooting.html"
    html = _bref_get(url)
    tables = pd.read_html(StringIO(html), attrs={"id": "shooting"})
    df = tables[0]
    df.columns = [
        "_".join(c for c in col if "Unnamed" not in c).strip("_")
        for col in df.columns
    ]
    df = df[df["Rk"].notna()].copy()  # drop section-divider blank rows
    df["SEASON"] = season
    return df


def pull_bref_advanced(season: str) -> pd.DataFrame:
    """STL%, BLK%, and BPM — not available from NBA.com's Advanced measure type
    (it lacks opponent-possession data for STL%/BLK%). Also brings TRB%/AST%/
    TOV%/USG%/TS% as a cross-check against the NBA.com versions, and BPM, which
    Step 2 of the method needs as the individual-quality-control covariate."""
    year = _season_to_bref_year(season)
    url = f"https://www.basketball-reference.com/leagues/NBA_{year}_advanced.html"
    html = _bref_get(url)
    tables = pd.read_html(StringIO(html), attrs={"id": "advanced"})
    df = tables[0]
    df = df[df["Rk"].notna()].copy()  # drop section-divider blank rows
    df["SEASON"] = season
    return df


def pull_play_type(season: str, play_type: str) -> pd.DataFrame:
    resp = _call_with_retry(
        synergyplaytypes.SynergyPlayTypes,
        season=season,
        play_type_nullable=play_type,
        player_or_team_abbreviation="P",
        type_grouping_nullable="offensive",  # required — omitting this silently returns 0 rows
    )
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    df["PLAY_TYPE"] = play_type
    return df


def pull_lineups(season: str, group_quantity: int, team_id: int) -> pd.DataFrame:
    """Scoped to one team — a single league-wide call truncates at 2000 rows
    (sorted by MIN desc), which silently drops qualifying low-minute pairs/trios
    and biases the sample toward high-minute (star-heavy) combos. Combos only
    ever occur within one team, so looping per team avoids the cap entirely."""
    resp = _call_with_retry(
        leaguedashlineups.LeagueDashLineups,
        season=season,
        group_quantity=group_quantity,
        team_id_nullable=team_id,
        measure_type_detailed_defense="Advanced",
        per_mode_detailed="Totals",
    )
    df = resp.get_data_frames()[0]
    df["SEASON"] = season
    df["GROUP_QUANTITY"] = group_quantity
    return df


# AI-ASSISTED (Claude Code, chat)
# Prompt: the Diagnostic Analysis page's top metric row showed off-court net
# rating and the on/off differential as "-" since this project had never
# pulled an on/off minute split anywhere - asked to close that gap for the
# current roster's real-NBA-data players (scoped to one team's real minutes,
# not a league-wide pull this project doesn't otherwise need).
# Used: TeamPlayerOnOffSummary (Advanced measure type) returns 3 frames -
# team totals, an "On" split, and an "Off" split, each keyed by
# VS_PLAYER_ID/VS_PLAYER_NAME with its own COURT_STATUS already labeled
# 'On'/'Off' - just concatenate the On/Off frames (index 1, 2) rather than
# writing separate pull logic for each half.
# Not AI: scoping this to a single team (BKN) rather than pulling on/off
# for the whole league - this project's roster-level questions never need
# another team's on/off splits, so a league-wide pull would be pure waste.


def pull_team_onoff(season: str, team_id: int) -> pd.DataFrame:
    """Per-player on-court vs. off-court NET_RATING for one team-season."""
    resp = _call_with_retry(
        teamplayeronoffsummary.TeamPlayerOnOffSummary,
        team_id=team_id, season=season, measure_type_detailed_defense="Advanced",
    )
    frames = resp.get_data_frames()
    df = pd.concat([frames[1], frames[2]], ignore_index=True)
    df["SEASON"] = season
    return df


LINEUPS_SLEEP_SECONDS = 1.5  # this loop is 720 calls — got rate-limited at 0.8s


def run_day1_pull():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    for season in SEASONS:
        print(f"=== {season} ===")

        print("player bio (height/weight)...")
        _pull_if_needed("player_bio", {"SEASON": season}, lambda: pull_player_bio(season))

        print("player advanced (USG%, TS%, AST%, ...)...")
        _pull_if_needed("player_advanced", {"SEASON": season}, lambda: pull_player_advanced(season))

        print("player base (3PA, PTS, ...)...")
        _pull_if_needed("player_base", {"SEASON": season}, lambda: pull_player_base(season))

        print("B-Ref advanced (STL%, BLK%, BPM, ...)...")
        _pull_if_needed(
            "player_advanced_bref", {"SEASON": season}, lambda: pull_bref_advanced(season),
            sleep_seconds=BREF_SLEEP_SECONDS,
        )

        print("B-Ref shooting (distance buckets, corner-3, AST%)...")
        _pull_if_needed(
            "player_shooting_bref", {"SEASON": season}, lambda: pull_bref_shooting(season),
            sleep_seconds=BREF_SLEEP_SECONDS,
        )

        for pt in PLAY_TYPES:
            print(f"play type: {pt}...")
            _pull_if_needed(
                "player_playtype", {"SEASON": season, "PLAY_TYPE": pt},
                lambda pt=pt: pull_play_type(season, pt),
            )

        for gq in GROUP_QUANTITIES:
            for team_id in TEAM_IDS:
                _pull_if_needed(
                    "lineups", {"SEASON": season, "GROUP_QUANTITY": gq, "TEAM_ID": team_id},
                    lambda gq=gq, team_id=team_id: pull_lineups(season, gq, team_id),
                    sleep_seconds=LINEUPS_SLEEP_SECONDS,
                )
            print(f"lineups group_quantity={gq}: done all {len(TEAM_IDS)} teams")

    print("Day 1 pull complete.")


# ============================================================================
# PART B — FEATURE PROCESSING
# ============================================================================

def _load_table(table_name: str, db_path: Path = DB_PATH) -> pd.DataFrame:
    """Load a table from the SQLite database into a pandas DataFrame."""
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(f"SELECT * FROM {table_name}", conn)
    return df


"""
AI MODIFIED (Claude Code)
Prompt: 帮我查一下 Egor Dёmin 的数据 — investigating turned up that this
function's Cyrillic-homoglyph fix was lost during a rewrite. B-Ref's page
spells his name with a Cyrillic "ё" (U+0451) where NBA.com uses the Latin
"ë" (U+00EB) — visually identical, different codepoint. NFKD decomposition
doesn't cross scripts, so encode('ascii','ignore') silently drops the
Cyrillic character instead of transliterating it, which was dropping his
row from build_nba_side_tables() entirely (confirmed live: 1308 min in
player_base, MIN>=300, but 0 rows for him out of build_nba_side_tables()
before this fix) — a real 2025 Nets draft pick, not an edge case to ignore.
Map the common confusable Cyrillic letters to Latin before NFKD so this
doesn't reoccur for the next player with a similar name.
"""
_CYRILLIC_HOMOGLYPHS = str.maketrans({
    "а": "a", "е": "e", "ё": "e", "о": "o", "р": "p", "с": "c",
    "у": "y", "х": "x", "А": "A", "Е": "E", "О": "O", "Р": "P", "С": "C",
})


"""
AI MODIFIED (Claude Code)
Prompt: 帮我在 step0 解决 build_nba_side_tables() 的 "dropped 3 rows" name-
join miss. Investigated live and found JR Smith (2017-18) was one of the 3:
NBA.com writes "JR Smith" (no punctuation), B-Ref writes "J.R. Smith" (with
periods). The old regex order ran suffix-stripping (\\b(jr|sr|ii|...)\\b)
BEFORE punctuation removal: "jr smith" (already-clean NBA.com spelling) had
its leading "jr" stripped as if it were a Jr./Sr./II generational suffix,
producing "smith" - but "j.r. smith" (B-Ref, punctuated) didn't match the
same regex yet (the periods break up the literal "jr" substring), so it
only became "jr smith" once punctuation got stripped one step later. Same
normalizer, two different outputs for the same person.
Used: anchor the suffix regex to the END of the string ($) - Jr./Sr./II/
III/IV generational suffixes only ever appear after a last name, never as
a first name, so this correctly leaves "jr smith" (JR as a first name)
alone while still stripping "ronald holland ii" -> "ronald holland".
Not AI: NOT fixing Ronald Holland II here - that one is a genuine nickname
difference (B-Ref: "Ron Holland", NBA.com: "Ronald Holland II"), which no
amount of punctuation/suffix normalization can close; would need a manual
name-alias table, a real, separate decision to make, not a regex bug.
"""
"""
AI MODIFIED (Claude Code)
Prompt: Ronald Holland II 这个帮我解决一下, 既然他出现在2025-26了 - fix the
remaining name-join miss, now that he's relevant (2024-25 and 2025-26,
1266 and 1550 min respectively). This is a genuine nickname difference, not
a normalization-rule bug (B-Ref: "Ron Holland", NBA.com: "Ronald Holland
II") - no regex fixes this, it needs a manual, verified mapping.
Verified before adding: checked both raw tables for every "Holland" - "Ron
Holland" appears only in 2024-25/2025-26 on both sides and nowhere else,
"John Holland" (2017-18/2018-19) already matches cleanly on both sides and
needs no alias - so this mapping can't accidentally merge two different
real players.
Not AI: confirming this is really one person (not verifiable from string
matching alone - taken as given from context established earlier in this
project's own investigation of this player).
"""
_NAME_ALIASES = {
    "ron holland": "ronald holland",  # B-Ref's short form -> NBA.com's full form
}


def _normalize_name(name: str) -> str:
    import re
    import unicodedata

    name = str(name).translate(_CYRILLIC_HOMOGLYPHS)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")
    name = name.lower()
    name = re.sub(r"[^a-z ]", "", name)
    name = re.sub(r"\b(jr|sr|ii|iii|iv)$", "", name.strip())
    name = re.sub(r"\s+", " ", name).strip()
    return _NAME_ALIASES.get(name, name)


def _prefer_combined_bref_row(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["_is_combined"] = df["Team"].str.match(r"^\dTM$", na=False)
    df = df.sort_values("_is_combined", ascending=False)
    df = df.drop_duplicates(subset=["_norm_name", "SEASON"], keep="first")
    return df.drop(columns="_is_combined")


# build nba side tables: player_bio, player_base, player_advanced, player_advanced_bref, player_shooting_bref, player_playtype
def build_nba_side_tables() -> pd.DataFrame:
    """Join the relevant NBA.com and Basketball-Reference tables into a single DataFrame."""
    # Load the raw tables
    player_bio = _load_table("player_bio")
    player_base = _load_table("player_base")
    player_advanced = _load_table("player_advanced")
    player_advanced_bref = _load_table("player_advanced_bref")
    player_shooting_bref = _load_table("player_shooting_bref")
    player_playtype = _load_table("player_playtype")

    # --- NBA.com side: these three key cleanly on PLAYER_ID + SEASON.
    # Trim each to columns that don't collide with the others (gotcha #2) —
    # player_base is the identity/MIN source; player_bio and player_advanced
    # would otherwise re-introduce their own PLAYER_NAME/TEAM_ABBREVIATION/
    # AGE/MIN copies and pandas would silently suffix them _x/_y.
    base = player_base[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "SEASON", "MIN", "PTS"]]
    advanced = player_advanced[["PLAYER_ID", "SEASON", "POSS"]]
    bio = player_bio[["PLAYER_ID", "SEASON", "PLAYER_HEIGHT_INCHES"]]

    df = base.merge(advanced, on=["PLAYER_ID", "SEASON"], how="left")
    df = df.merge(bio, on=["PLAYER_ID", "SEASON"], how="left")
    df["_norm_name"] = df["PLAYER_NAME"].apply(_normalize_name)

    # Paper Table 3.3 wants pts-per-100-possessions, not raw PTS (which is
    # confounded by playing time, already handled separately via the MIN
    # filter) — derive it now that POSS is available.
    df["PTS_PER_100"] = df["PTS"] / df["POSS"] * 100

    # --- B-Ref side: no PLAYER_ID, so join on normalized name + SEASON instead.
    adv_bref = player_advanced_bref.copy()
    adv_bref["_norm_name"] = adv_bref["Player"].apply(_normalize_name)
    adv_bref = _prefer_combined_bref_row(adv_bref)
    adv_bref = adv_bref[["_norm_name", "SEASON", "TS%", "USG%", "AST%", "TOV%", "STL%", "BLK%", "TRB%", "FTr", "BPM"]]

    shot = player_shooting_bref.copy()
    shot["_norm_name"] = shot["Player"].apply(_normalize_name)
    shot = _prefer_combined_bref_row(shot)
    shot = shot[[
        "_norm_name", "SEASON", "Dist.", "% of FGA by Distance_0-3", "% of FGA by Distance_3-10",
        "% of FGA by Distance_10-16", "% of FGA by Distance_16-3P", "% of FGA by Distance_3P",
        "Corner 3s_%3PA", "% of FG Ast'd_2P", "% of FG Ast'd_3P",
    ]]

    df = df.merge(adv_bref, on=["_norm_name", "SEASON"], how="left")
    df = df.merge(shot, on=["_norm_name", "SEASON"], how="left")

    # --- play types: one row per player-season-PLAY_TYPE -> pivot to wide first,
    # or merging it directly on PLAYER_ID+SEASON would fan each row out up to 9x.
    playtype_wide = player_playtype.pivot_table(
        index=["PLAYER_ID", "SEASON"], columns="PLAY_TYPE", values="POSS_PCT", fill_value=0.0
    )
    playtype_wide.columns = [f"PLAYTYPE_{c.upper()}" for c in playtype_wide.columns]
    playtype_wide = playtype_wide.reset_index()
    df = df.merge(playtype_wide, on=["PLAYER_ID", "SEASON"], how="left")

    # Filter to players with >= 300 total minutes (player_base.MIN — confirmed
    # a true season total; player_advanced.MIN is a per-game average, gotcha #4).
    df = df[df["MIN"] >= 300]

    """
    AI MODIFIED (Claude Code)
    Prompt: Fill the two remaining kinds of structurally-missing values before
    dropping anything — "Corner 3s_%3PA" and "% of FG Ast'd_3P" are 0/0 (and
    therefore NaN) for players with ~zero 3-point attempts or makes, which is
    a legitimate 0, not missing data; the 9 PLAYTYPE_* columns are NaN when a
    player-season has no rows at all in player_playtype (zero possessions
    logged in every tracked play type), also a legitimate 0. Both must be
    filled before the dropna below, or real >=300-minute players get dropped
    over a stat that doesn't apply to them rather than missing data.
    """
    structural_zero_cols = ["Corner 3s_%3PA", "% of FG Ast'd_3P"] + [
        c for c in df.columns if c.startswith("PLAYTYPE_")
    ]
    df[structural_zero_cols] = df[structural_zero_cols].fillna(0.0)

    # Drop only rows still missing the B-Ref-sourced feature columns (a failed
    # name-join), not every column indiscriminately (gotcha #5) — Awards etc.
    # aren't feature columns and shouldn't drive what gets dropped.
    feature_cols = [
        "TS%", "USG%", "AST%", "TOV%", "STL%", "BLK%", "TRB%", "FTr", "BPM",
        "Dist.", "% of FGA by Distance_0-3", "% of FGA by Distance_3P",
    ]
    n_before = len(df)
    df = df.dropna(subset=feature_cols)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"dropped {n_dropped} row(s) with unresolved missing data (likely a name-join miss)")

    return df


# The actual archetype feature set — everything in the table EXCEPT identity
# columns (PLAYER_ID/PLAYER_NAME/TEAM_ABBREVIATION/SEASON/_norm_name), MIN
# (just the filter threshold, not a trait), and the raw PTS/POSS that
# PTS_PER_100 was derived from (keeping the raw versions in the feature
# vector too would double-count that signal).
FEATURE_COLUMNS = [
    "PTS_PER_100", "PLAYER_HEIGHT_INCHES",
    "TS%", "USG%", "AST%", "TOV%", "STL%", "BLK%", "TRB%", "FTr", "BPM",
    "Dist.", "% of FGA by Distance_0-3", "% of FGA by Distance_3-10",
    "% of FGA by Distance_10-16", "% of FGA by Distance_16-3P", "% of FGA by Distance_3P",
    "Corner 3s_%3PA", "% of FG Ast'd_2P", "% of FG Ast'd_3P",
    "PLAYTYPE_CUT", "PLAYTYPE_HANDOFF", "PLAYTYPE_ISOLATION", "PLAYTYPE_OFFREBOUND",
    "PLAYTYPE_OFFSCREEN", "PLAYTYPE_PRBALLHANDLER", "PLAYTYPE_PRROLLMAN",
    "PLAYTYPE_POSTUP", "PLAYTYPE_SPOTUP",
]


def zscore(df: pd.DataFrame, feature_columns: list = None) -> pd.DataFrame:
    feature_columns = feature_columns or FEATURE_COLUMNS
    out = df.copy()
    out[feature_columns] = (df[feature_columns] - df[feature_columns].mean()) / df[feature_columns].std()
    return out


# ============================================================================
# PART C — DATA INSPECTION
# ============================================================================
# SOURCE OF COLUMN DEFINITIONS
#   No data dictionary, README, or schema file exists anywhere in this repo
#   (README.md was empty, writeup/ is empty) — checked before writing this.
#   Definitions below come from three grounded sources instead:
#     1. This project's own CLAUDE.md (paper Table 3.3 feature spec).
#     2. Column headers that are self-describing (Basketball-Reference's
#        "% of FGA by Distance_0-3" etc. — the header *is* the definition).
#     3. Standard, publicly documented NBA.com/Basketball-Reference stat
#        glossary terms (USG%, TS%, PIE, EFG%, ...).
#   Columns whose exact computation NBA.com does not publish (the E_* and
#   sp_work_* internal variants) are marked "best-effort inference" rather
#   than "confirmed". One column (SUM_TIME_PLAYED) was resolved empirically
#   by checking its numeric ratio against MIN directly against the live
#   database rather than guessed — see get_column_meaning() for the math.

# --------------------------------------------------------------------------
# Table-level descriptions
# --------------------------------------------------------------------------

TABLE_DESCRIPTIONS = {
    "player_bio": (
        "One row per player per season: height, weight, draft background, and "
        "light per-game stats. Source: NBA.com LeagueDashPlayerBioStats."
    ),
    "player_advanced": (
        "One row per player per season: advanced rate stats (USG%, TS%, PACE, "
        "PIE, on/off ratings) used as archetype features per paper Table 3.3. "
        "Source: NBA.com LeagueDashPlayerStats, MeasureType=Advanced."
    ),
    "player_base": (
        "One row per player per season: raw counting stats (FG/3PT/FT/REB/AST/"
        "STL/BLK/PTS). Source: NBA.com LeagueDashPlayerStats, MeasureType=Base."
    ),
    "player_shooting_bref": (
        "One row per player per season per team (players traded mid-season get "
        "a row per team plus a combined multi-team row). Shot-distance buckets, "
        "corner-3 rate, and assisted-vs-unassisted shot splits — matches paper "
        "Table 3.3's shot-distribution and assisted-shot features. "
        "Source: Basketball-Reference 'shooting' page, scraped directly (no API)."
    ),
    "player_advanced_bref": (
        "One row per player per season per team. STL% and BLK% — not available "
        "from NBA.com's Advanced measure type — plus BPM (the individual-quality "
        "control covariate Step 2's synergy regression needs) and TRB%/AST%/TOV%/"
        "USG%/TS% as a cross-check against the NBA.com versions. "
        "Source: Basketball-Reference 'advanced' page, scraped directly (no API)."
    ),
    "player_playtype": (
        "One row per player, per offensive play type, per season (up to 9 rows "
        "per player-season — one per play type they logged possessions in). "
        "Synergy tracking: usage share and efficiency by play type. Matches "
        "paper Table 3.3's play-type features. Source: NBA.com SynergyPlayTypes."
    ),
    "lineups": (
        "One row per unique group of players who shared the court, per group "
        "size (2/3/4/5-man), per team, per season. Raw input for the synergy-"
        "matrix analysis (Step 2). Source: NBA.com LeagueDashLineups, pulled "
        "one team at a time to dodge the API's 2000-row-per-call cap. "
        "CAUTION: GROUP_NAME can collide for different players who abbreviate "
        "identically (e.g. Cody Martin vs Caleb Martin, both 'C. Martin', both "
        "on CHA in 2020-21) — use GROUP_ID for any join/dedup, never GROUP_NAME."
    ),
}


# --------------------------------------------------------------------------
# Column meanings
# --------------------------------------------------------------------------
# Each entry: column_name -> (meaning, confidence)
# Columns repeated verbatim across tables (PLAYER_ID, TEAM_ID, SEASON, ...)
# live in COMMON_COLUMNS and are checked before the table-specific dicts.

COMMON_COLUMNS = {
    "PLAYER_ID": ("NBA.com's unique numeric player identifier.", CONFIRMED),
    "PLAYER_NAME": ("Player's full name.", CONFIRMED),
    "NICKNAME": ("Player's display nickname (often just their first name).", CONFIRMED),
    "TEAM_ID": ("NBA.com's unique numeric team identifier.", CONFIRMED),
    "TEAM_ABBREVIATION": ("3-letter team code, e.g. LAL, BOS.", CONFIRMED),
    "TEAM_NAME": ("Full team name.", CONFIRMED),
    "AGE": ("Player's age as tracked by NBA.com for that season.", CONFIRMED),
    "SEASON": ("Season string 'YYYY-YY'. Added by this project's pull script, not part of the raw API response.", CONFIRMED),
    "GP": ("Games played.", CONFIRMED),
    "GS": ("Games started.", CONFIRMED),
    "MIN": ("Total minutes played (Totals per-mode). NOTE: player_advanced is the one exception — see its table-specific override below.", CONFIRMED),
    "W": ("Wins in games this player/lineup appeared in.", CONFIRMED),
    "L": ("Losses in games this player/lineup appeared in.", CONFIRMED),
    "W_PCT": ("Win percentage: W / (W + L).", CONFIRMED),
    "POSS": ("Number of possessions.", CONFIRMED),
    "TEAM_COUNT": ("Number of distinct teams this player played for in the season (>1 if traded).", CONFIRMED),
    "Rk": ("Row number as listed on the Basketball-Reference page — not a performance ranking.", CONFIRMED),
    "Player": ("Player's full name.", CONFIRMED),
    "Team": ("3-letter team code. 'TOT'/'2TM'/'3TM' rows are combined totals for players traded mid-season.", CONFIRMED),
    "Pos": ("Position.", CONFIRMED),
    "G": ("Games played.", CONFIRMED),
    "MP": ("Minutes played.", CONFIRMED),
    "Awards": ("Comma-separated season award abbreviations, e.g. 'MVP-8'=8th in MVP voting, 'AS'=All-Star, 'NBA1'/'NBA2'/'NBA3'=All-NBA team.", CONFIRMED),
    "FGM": ("Field goals made.", CONFIRMED),
    "FGA": ("Field goals attempted.", CONFIRMED),
    "FG_PCT": ("Field goal percentage: FGM / FGA.", CONFIRMED),
    "PTS": ("Total points.", CONFIRMED),
    "AST": ("Total assists.", CONFIRMED),
    "REB": ("Total rebounds (offensive + defensive).", CONFIRMED),
    "OREB_PCT": ("Offensive rebound percentage: share of available offensive rebounds grabbed while on court.", CONFIRMED),
    "DREB_PCT": ("Defensive rebound percentage: share of available defensive rebounds grabbed while on court.", CONFIRMED),
    "REB_PCT": ("Total rebound percentage: share of all available rebounds (off + def) grabbed while on court.", CONFIRMED),
    "USG_PCT": ("Usage percentage: share of team possessions this player used while on court.", CONFIRMED),
    "TS_PCT": ("True shooting percentage — scoring efficiency accounting for 2PT, 3PT, and FT value together.", CONFIRMED),
    "EFG_PCT": ("Effective field goal percentage — weights made 3-pointers 1.5x to reflect their extra value.", CONFIRMED),
    "AST_PCT": ("Percentage of teammate field goals this player assisted while on court.", CONFIRMED),
    "AST_TO": ("Assist-to-turnover ratio.", CONFIRMED),
    "AST_RATIO": ("Assists per 100 possessions used.", CONFIRMED),
    "TM_TOV_PCT": ("Turnover percentage: turnovers per 100 team plays used.", CONFIRMED),
    "PACE": ("Possessions per 48 minutes.", CONFIRMED),
    "PACE_PER40": ("Possessions per 40 minutes.", CONFIRMED),
    "PIE": ("Player Impact Estimate — NBA.com's box-score-based share of total game statistical production.", CONFIRMED),
    "OFF_RATING": ("Points scored per 100 possessions while on court.", CONFIRMED),
    "DEF_RATING": ("Points allowed per 100 possessions while on court.", CONFIRMED),
    "NET_RATING": ("OFF_RATING minus DEF_RATING.", CONFIRMED),
    "E_OFF_RATING": ("'Estimated' offensive rating — a simplified formula that doesn't require full play-by-play/lineup data. Same concept as OFF_RATING; exact formula difference not published by NBA.com.", INFERRED),
    "E_DEF_RATING": ("'Estimated' defensive rating, same caveat as E_OFF_RATING.", INFERRED),
    "E_NET_RATING": ("'Estimated' net rating, same caveat as E_OFF_RATING.", INFERRED),
    "E_PACE": ("'Estimated' pace, same caveat as E_OFF_RATING.", INFERRED),
    "E_USG_PCT": ("'Estimated' usage percentage, same caveat as E_OFF_RATING.", INFERRED),
    "E_TOV_PCT": ("'Estimated' turnover percentage, same caveat as E_OFF_RATING.", INFERRED),
    "sp_work_OFF_RATING": ("NBA.com internal alternate-pipeline offensive rating ('sp_work' prefix). Same underlying concept as OFF_RATING; NBA.com does not publicly document how this pipeline differs from the standard one.", INFERRED),
    "sp_work_DEF_RATING": ("NBA.com internal alternate-pipeline defensive rating, same caveat as sp_work_OFF_RATING.", INFERRED),
    "sp_work_NET_RATING": ("NBA.com internal alternate-pipeline net rating, same caveat as sp_work_OFF_RATING.", INFERRED),
    "sp_work_PACE": ("NBA.com internal alternate-pipeline pace, same caveat as sp_work_OFF_RATING.", INFERRED),
}


TABLE_COLUMNS = {
    "player_bio": {
        "PLAYER_HEIGHT": ("Height as a feet-inches string, e.g. '6-8'.", CONFIRMED),
        "PLAYER_HEIGHT_INCHES": ("Height converted to total inches.", CONFIRMED),
        "PLAYER_WEIGHT": ("Weight in pounds.", CONFIRMED),
        "COLLEGE": ("College attended (blank for players who didn't attend a US college).", CONFIRMED),
        "COUNTRY": ("Country of origin.", CONFIRMED),
        "DRAFT_YEAR": ("Year drafted ('Undrafted' if not drafted).", CONFIRMED),
        "DRAFT_ROUND": ("Draft round.", CONFIRMED),
        "DRAFT_NUMBER": ("Overall draft pick number.", CONFIRMED),
    },
    "player_advanced": {
        "MIN": (
            "Per-game minutes average — NOT a season total, despite this table being pulled "
            "with per_mode_detailed='Totals'. Empirically confirmed: Tyrese Maxey shows 37.7 here "
            "(2024-25) vs. 1960.2 in player_base.MIN for the same player-season (GP=52, "
            "52 x 37.7 = 1960.4, matching player_base). NBA.com's Advanced endpoint does not "
            "respect the Totals/PerGame toggle for this column. Use player_base.MIN (or "
            "player_advanced_bref.MP) for any total-minutes filter, e.g. the >=300 min cutoff.",
            CONFIRMED,
        ),
        "FGM_PG": ("Field goals made per game.", CONFIRMED),
        "FGA_PG": ("Field goals attempted per game.", CONFIRMED),
    },
    "player_base": {
        "FG3M": ("3-point field goals made.", CONFIRMED),
        "FG3A": ("3-point field goals attempted.", CONFIRMED),
        "FG3_PCT": ("3-point field goal percentage.", CONFIRMED),
        "FTM": ("Free throws made.", CONFIRMED),
        "FTA": ("Free throws attempted.", CONFIRMED),
        "FT_PCT": ("Free throw percentage.", CONFIRMED),
        "OREB": ("Offensive rebounds.", CONFIRMED),
        "DREB": ("Defensive rebounds.", CONFIRMED),
        "TOV": ("Turnovers.", CONFIRMED),
        "STL": ("Steals.", CONFIRMED),
        "BLK": ("Blocks (shots this player blocked).", CONFIRMED),
        "BLKA": ("Blocks against — times this player's shot was blocked by an opponent.", CONFIRMED),
        "PF": ("Personal fouls committed.", CONFIRMED),
        "PFD": ("Personal fouls drawn.", CONFIRMED),
        "PLUS_MINUS": ("Raw point differential while on court (not normalized per 100 possessions).", CONFIRMED),
        "NBA_FANTASY_PTS": ("Fantasy points under NBA.com's standard fantasy scoring formula.", CONFIRMED),
        "WNBA_FANTASY_PTS": ("Fantasy points under the WNBA fantasy scoring formula, computed for this (NBA) player. Likely a byproduct of NBA.com and WNBA.com sharing the same underlying stats API — not a meaningful analytic column for this project.", INFERRED),
        "DD2": ("Double-doubles: games with 10+ in two major statistical categories.", CONFIRMED),
        "TD3": ("Triple-doubles: games with 10+ in three major statistical categories.", CONFIRMED),
    },
    "player_shooting_bref": {
        "FG%": ("Overall field goal percentage.", CONFIRMED),
        "Dist.": ("Average distance (feet) of this player's field goal attempts.", CONFIRMED),
        "% of FGA by Distance_2P": ("Share of total FGA that were 2-point attempts.", CONFIRMED),
        "% of FGA by Distance_0-3": ("Share of total FGA taken from 0-3 feet.", CONFIRMED),
        "% of FGA by Distance_3-10": ("Share of total FGA taken from 3-10 feet.", CONFIRMED),
        "% of FGA by Distance_10-16": ("Share of total FGA taken from 10-16 feet.", CONFIRMED),
        "% of FGA by Distance_16-3P": ("Share of total FGA taken from 16 feet to the 3-point line.", CONFIRMED),
        "% of FGA by Distance_3P": ("Share of total FGA that were 3-point attempts.", CONFIRMED),
        "FG% by Distance_2P": ("Field goal percentage on 2-point attempts.", CONFIRMED),
        "FG% by Distance_0-3": ("Field goal percentage from 0-3 feet.", CONFIRMED),
        "FG% by Distance_3-10": ("Field goal percentage from 3-10 feet.", CONFIRMED),
        "FG% by Distance_10-16": ("Field goal percentage from 10-16 feet.", CONFIRMED),
        "FG% by Distance_16-3P": ("Field goal percentage from 16 feet to the 3-point line.", CONFIRMED),
        "FG% by Distance_3P": ("3-point field goal percentage.", CONFIRMED),
        "% of FG Ast'd_2P": ("Share of made 2-point shots that were assisted (vs. self-created).", CONFIRMED),
        "% of FG Ast'd_3P": ("Share of made 3-point shots that were assisted (vs. self-created).", CONFIRMED),
        "Dunks_%FGA": ("Share of total FGA that were dunks.", CONFIRMED),
        "Dunks_#": ("Count of made dunks.", CONFIRMED),
        "Corner 3s_%3PA": ("Share of total 3-point attempts taken from the corner (vs. above the break).", CONFIRMED),
        "Corner 3s_3P%": ("Field goal percentage on corner 3-point attempts specifically.", CONFIRMED),
        "1/2 Court_Att.": ("Half-court-or-longer shot attempts (heaves).", CONFIRMED),
        "1/2 Court_Md.": ("Half-court-or-longer shots made.", CONFIRMED),
    },
    "player_advanced_bref": {
        "PER": ("Player Efficiency Rating — Hollinger's per-minute overall rating, league average = 15.", CONFIRMED),
        "TS%": ("True shooting percentage.", CONFIRMED),
        "3PAr": ("Share of field goal attempts that were 3-pointers.", CONFIRMED),
        "FTr": ("Free throw attempts per field goal attempt.", CONFIRMED),
        "ORB%": ("Offensive rebound percentage.", CONFIRMED),
        "DRB%": ("Defensive rebound percentage.", CONFIRMED),
        "TRB%": ("Total rebound percentage.", CONFIRMED),
        "AST%": ("Percentage of teammate field goals this player assisted while on court.", CONFIRMED),
        "STL%": ("Percentage of opponent possessions (while this player was on court) that ended in a steal by this player. Not available from NBA.com's Advanced endpoint — this is why this table exists.", CONFIRMED),
        "BLK%": ("Percentage of opponent 2-point attempts (while this player was on court) blocked by this player. Not available from NBA.com's Advanced endpoint — this is why this table exists.", CONFIRMED),
        "TOV%": ("Turnovers per 100 plays used.", CONFIRMED),
        "USG%": ("Usage percentage.", CONFIRMED),
        "OWS": ("Offensive win shares — estimated wins contributed via offense.", CONFIRMED),
        "DWS": ("Defensive win shares — estimated wins contributed via defense.", CONFIRMED),
        "WS": ("Win shares — OWS + DWS.", CONFIRMED),
        "WS/48": ("Win shares per 48 minutes.", CONFIRMED),
        "OBPM": ("Offensive box plus/minus — points per 100 possessions above league average, offense only.", CONFIRMED),
        "DBPM": ("Defensive box plus/minus — points per 100 possessions above league average, defense only.", CONFIRMED),
        "BPM": ("Box plus/minus — OBPM + DBPM. This is the individual-quality control covariate Step 2's synergy regression needs, per CLAUDE.md.", CONFIRMED),
        "VORP": ("Value over replacement player.", CONFIRMED),
    },
    "player_playtype": {
        "SEASON_ID": ("NBA.com's internal season code, e.g. '22020' = 2020-21 regular season (leading '2' = regular season).", INFERRED),
        "PLAY_TYPE": ("Which of the 9 offensive play types this row covers (Isolation, PRBallHandler, PRRollman, Postup, Spotup, Handoff, Cut, OffScreen, OffRebound).", CONFIRMED),
        "TYPE_GROUPING": ("Offensive vs. defensive tracking side. Always 'offensive' in this dataset — filtered that way at pull time.", CONFIRMED),
        "PERCENTILE": ("Player's league percentile rank in scoring efficiency (PPP) for this play type.", CONFIRMED),
        "POSS_PCT": ("Share of the player's total offensive possessions that came from this play type.", CONFIRMED),
        "PPP": ("Points scored per possession within this play type.", CONFIRMED),
        "FT_POSS_PCT": ("Share of this play type's possessions that ended in a free-throw trip.", CONFIRMED),
        "TOV_POSS_PCT": ("Share of this play type's possessions that ended in a turnover.", CONFIRMED),
        "SF_POSS_PCT": ("Share of this play type's possessions that drew a shooting foul.", CONFIRMED),
        "PLUSONE_POSS_PCT": ("Share of this play type's possessions that resulted in an and-one.", CONFIRMED),
        "SCORE_POSS_PCT": ("Share of this play type's possessions that ended in a score.", CONFIRMED),
        "FGMX": ("Field goals missed within this play type (FGA - FGM).", CONFIRMED),
    },
    "lineups": {
        "GROUP_SET": ("NBA.com's internal label for the query grouping (constant value, not analytically meaningful).", INFERRED),
        "GROUP_ID": ("Encoded string of the actual player IDs in this combination. The true unique identifier for a lineup/combo — use this, not GROUP_NAME, for joins or dedup.", CONFIRMED),
        "GROUP_NAME": ("Human-readable abbreviated names (first initial + last name) of the players in this combination. NOT guaranteed unique — see the lineups table caution above.", CONFIRMED),
        "SUM_TIME_PLAYED": (
            "Empirically confirmed redundant with MIN: SUM_TIME_PLAYED / MIN ≈ 3000 consistently, "
            "regardless of GROUP_QUANTITY (checked directly against this database, not assumed). "
            "That constant is consistent with a base unit of 1/50th of a second "
            "(seconds = value/50, minutes = seconds/60 = value/3000), but NBA.com does not publish "
            "the official unit name, so the exact unit label is a best-effort inference.",
            INFERRED,
        ),
        "GROUP_QUANTITY": ("Number of players in this combination (2, 3, 4, or 5). Added by this project's pull script.", CONFIRMED),
    },
}


def get_column_meaning(table: str, column: str) -> tuple:
    """Look up (meaning, confidence) for a column: table-specific dict first,
    then the common dict, then the _RANK pattern, else Unknown."""
    if column in TABLE_COLUMNS.get(table, {}):
        return TABLE_COLUMNS[table][column]
    if column in COMMON_COLUMNS:
        return COMMON_COLUMNS[column]
    if column.endswith("_RANK"):
        base_col = column[: -len("_RANK")]
        base_meaning, base_conf = get_column_meaning(table, base_col)
        if base_conf != UNKNOWN:
            return (f"League rank (1 = best) for {base_col}: {base_meaning}", base_conf)
        return (f"League rank (1 = best) for {base_col}.", INFERRED)
    return ("", UNKNOWN)


def list_tables(conn: sqlite3.Connection) -> list:
    preferred = [
        "player_bio", "player_advanced", "player_advanced_bref", "player_base",
        "player_shooting_bref", "player_playtype", "lineups",
    ]
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    found = {row[0] for row in cur.fetchall()}
    ordered = [t for t in preferred if t in found]
    ordered += sorted(found - set(ordered))
    return ordered


def load_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    return pd.read_sql(f'SELECT * FROM "{table}"', conn)


def render_header(table: str, df: pd.DataFrame) -> None:
    description = TABLE_DESCRIPTIONS.get(table, "(no description available)")
    body = f"[bold]{description}[/bold]\n\nShape: {df.shape[0]:,} rows x {df.shape[1]} columns"
    console.print(Panel(body, title=f"[bold cyan]{table}[/bold cyan]", box=box.ROUNDED, expand=False))


def render_preview(table: str, df: pd.DataFrame, max_cols: int = PREVIEW_MAX_COLS, n_rows: int = PREVIEW_ROWS) -> None:
    cols = list(df.columns[:max_cols])
    hidden = df.shape[1] - len(cols)

    rich_table = Table(box=box.SIMPLE_HEAVY, header_style="bold magenta", show_lines=False)
    for c in cols:
        rich_table.add_column(c, overflow="ellipsis", no_wrap=True, max_width=14)

    for _, row in df.head(n_rows).iterrows():
        rich_table.add_row(*[str(row[c]) for c in cols])

    console.print(rich_table)
    if hidden > 0:
        console.print(f"[dim]... + {hidden} additional column(s) not shown here — see column dictionary below[/dim]")
    console.print()


def render_column_dictionary(table: str, df: pd.DataFrame) -> None:
    rich_table = Table(
        title="Column dictionary", box=box.SIMPLE_HEAVY,
        header_style="bold green", show_lines=False, title_justify="left",
    )
    rich_table.add_column("Column", style="bold")
    rich_table.add_column("Meaning", overflow="fold", max_width=50)
    rich_table.add_column("Type")
    rich_table.add_column("Missing", justify="right")
    rich_table.add_column("Example", overflow="fold", max_width=20)

    n = len(df)
    for col in df.columns:
        meaning, confidence = get_column_meaning(table, col)
        if confidence == UNKNOWN:
            meaning_display = f"[yellow]{UNKNOWN}[/yellow]"
        elif confidence == INFERRED:
            meaning_display = f"{meaning} [dim italic]({INFERRED})[/dim italic]"
        else:
            meaning_display = meaning

        dtype = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        pct_missing = (n_missing / n * 100) if n else 0.0
        missing_display = f"{n_missing:,} ({pct_missing:.1f}%)"

        non_null = df[col].dropna()
        example = str(non_null.iloc[0]) if len(non_null) else "(all null)"
        if len(example) > 20:
            example = example[:17] + "..."

        rich_table.add_row(col, meaning_display, dtype, missing_display, example)

    console.print(rich_table)


def render_table(table: str, df: pd.DataFrame, show_preview: bool = True) -> None:
    console.rule(style="cyan")
    render_header(table, df)
    console.print()
    if show_preview:
        console.print("[bold]Preview[/bold]")
        render_preview(table, df)
    render_column_dictionary(table, df)
    console.print()


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ").strip()


def render_table_markdown(table: str, df: pd.DataFrame, max_preview_cols: int = PREVIEW_MAX_COLS, n_preview_rows: int = PREVIEW_ROWS) -> str:
    lines = [f"### `{table}`", ""]
    lines.append(TABLE_DESCRIPTIONS.get(table, "(no description available)"))
    lines.append("")
    lines.append(f"**Shape:** {df.shape[0]:,} rows x {df.shape[1]} columns")
    lines.append("")

    cols = list(df.columns[:max_preview_cols])
    hidden = df.shape[1] - len(cols)
    lines.append("<details>")
    lines.append("<summary>Preview (first rows)</summary>")
    lines.append("")
    lines.append("| " + " | ".join(cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, row in df.head(n_preview_rows).iterrows():
        lines.append("| " + " | ".join(_md_escape(row[c]) for c in cols) + " |")
    if hidden > 0:
        lines.append("")
        lines.append(f"_+ {hidden} additional column(s) not shown here — see column dictionary below._")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    lines.append("<details>")
    lines.append(f"<summary>Column dictionary ({df.shape[1]} columns)</summary>")
    lines.append("")
    lines.append("| Column | Meaning | Type | Missing | Example |")
    lines.append("|---|---|---|---|---|")
    n = len(df)
    for col in df.columns:
        meaning, confidence = get_column_meaning(table, col)
        if confidence == UNKNOWN:
            meaning_display = f"**{UNKNOWN}**"
        elif confidence == INFERRED:
            meaning_display = f"{meaning} _({INFERRED})_"
        else:
            meaning_display = meaning

        dtype = str(df[col].dtype)
        n_missing = int(df[col].isna().sum())
        pct_missing = (n_missing / n * 100) if n else 0.0
        missing_display = f"{n_missing:,} ({pct_missing:.1f}%)"

        non_null = df[col].dropna()
        example = str(non_null.iloc[0]) if len(non_null) else "(all null)"
        if len(example) > 30:
            example = example[:27] + "..."

        lines.append(
            f"| `{_md_escape(col)}` | {_md_escape(meaning_display)} | {dtype} "
            f"| {missing_display} | {_md_escape(example)} |"
        )
    lines.append("")
    lines.append("</details>")
    lines.append("")
    return "\n".join(lines)


def render_all_markdown(conn: sqlite3.Connection) -> str:
    tables = list_tables(conn)
    parts = [
        "## Raw Data Tables",
        "",
        f"_Generated from `data/{DB_PATH.name}` via `python3 src/step0_data.py --markdown`. "
        "Column meanings are sourced from CLAUDE.md's feature spec, self-describing headers, "
        "and standard NBA/Basketball-Reference stat terminology — see the script's module "
        "docstring for details. Entries marked _(best-effort inference)_ are NBA.com internal "
        "variants whose exact computation isn't publicly documented._",
        "",
    ]
    for table in tables:
        df = load_table(conn, table)
        parts.append(render_table_markdown(table, df))
    return "\n".join(parts)


"""
AI MODIFIED (Claude Code)
Prompt: Fold inspect_data.py (the rich-terminal table/column-dictionary
viewer) into this file too, since it's still "look at what step 0
produced." Entry point now dispatches on argv: --pull runs Part A's data
pull (the heavy, deliberate action, so it's opt-in rather than triggered by
just running the file to inspect data); default with no special flags is
Part C's inspection display, matching inspect_data.py's original UX.
"""
def main() -> None:
    args = sys.argv[1:]

    if "--pull" in args:
        run_day1_pull()
        return

    if not DB_PATH.exists():
        console.print(f"[red]No database found at {DB_PATH} — run with --pull first.[/red]")
        sys.exit(1)

    show_preview = "--no-preview" not in args
    markdown_mode = "--markdown" in args
    requested_tables = [a for a in args if not a.startswith("--")]

    with sqlite3.connect(DB_PATH) as conn:
        all_tables = list_tables(conn)
        tables = requested_tables if requested_tables else all_tables

        unknown_tables = [t for t in tables if t not in all_tables]
        if unknown_tables:
            console.print(f"[red]Unknown table(s): {unknown_tables}. Available: {all_tables}[/red]")
            sys.exit(1)

        if markdown_mode:
            print(render_all_markdown(conn))
            return

        console.print(f"[bold]Found {len(all_tables)} tables in {DB_PATH.name}[/bold]: {all_tables}\n")
        for table in tables:
            df = load_table(conn, table)
            render_table(table, df, show_preview=show_preview)


if __name__ == "__main__":
    main()
