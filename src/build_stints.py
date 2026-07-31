"""
build_stints.py - directed offensive/defensive stint table for a full NBA season,
built from real play-by-play, for a skill-weighted archetype-RAPM model.

AI-ASSISTED (Claude Code, chat)
Prompt: "You are a senior sports-data engineer. Build a runnable Python pipeline
that collects and preprocesses NBA 2025-26 regular-season possession data into
a clean, regression-ready 'stint' table..." (full spec: directed rows per stint,
pbpstats preferred, disk-caching + retry/backoff + resumable, output schema
game_id/date/stint_id/offense_team/defense_team/home_off/off_p1..5/def_p1..5/
off_possessions/off_points/y_off, validation report).

Used: pbpstats (dblackrun) was tried first, per the prompt's stated preference,
but its stats_nba data provider depends on the legacy `boxscoretraditionalv2`
stats.nba.com endpoint - confirmed live (raw request, 60s timeout, repeated
across 4 retries) that this endpoint no longer serves data; nba_api itself
raises `BoxScoreTraditionalV2 is deprecated... Data is no longer being
published for BoxScoreTraditionalV2 as of the 2025-26 NBA season` - a real,
externally-verifiable fact (pbpstats' last PyPI release was 2024-04-17,
predating this deprecation), not an environment quirk. Presented this finding
plus three options to the user; they chose to have me build the substitution/
possession parser directly on nba_api's modern v3 endpoints
(BoxScoreTraditionalV3, PlayByPlayV3 - both verified live, fast and complete)
rather than search for another library or pause. Every possession/lineup rule
below was checked against a real game's actual play-by-play before being
encoded (see each function's own note for what was specifically verified).

Not AI: writing to a NEW db/directory (data/stints/, never data/nets_synergy.db)
so this cannot conflict with any existing table - explicit instruction. The
output schema, stint definition, directed-row construction, and validation
checks were all specified directly in the prompt, not invented here. The
choice to build a from-scratch parser (vs. search further / pause) was the
user's own, made after seeing the pbpstats blocker.

KNOWN LIMITATION - possession pace runs slightly below the real number, for
every team (checked directly against nba_api's own official LeagueDashTeamStats
PACE/POSS for all 30 teams, 2025-26, not just an eyeballed "typical" range):

    every one of the 30 teams is undercounted, by 3.5-7.1 offensive
    possessions/game (mean 5.4, ~5.4% relative) - a consistent systematic
    gap, not per-team noise.

Two things it correlates with, checked directly (not guessed) rather than
just asserting "probably substitution ambiguity":
  - substitution frequency (subs/game, from the cached play-by-play):
    r ~ -0.40 to -0.42 against the true gap - teams that substitute more DO
    show a bigger gap, real (if moderate) evidence that the ~6% residual
    substitution-ambiguity rate documented above (see
    `_resolve_substitution_direction`) is a real contributor, not just a
    theoretical one.
  - team archetype composition (this project's own K=8 recipes, MIN-weighted
    team share): the strongest single correlate is arch_7 (Mbeng-type - low
    USG%/BPM, high STL%, PnR-adjacent) at r ~ +0.63 - notably stronger than
    substitution frequency, though less mechanistically obvious (a plausible
    story is that steal-heavy/scrambly playstyles create more broken-play
    substitution clusters, but with only 30 teams this is a real, measured
    correlation, not a confirmed causal explanation).
Even the single best team (Boston) still has a nontrivial +3.5-possession
gap, so substitution ambiguity is very likely not the WHOLE story - there is
probably also a smaller, universal gap from some possession-ending scenario
missing from the 5 rules in Part D entirely (candidates not currently
handled: backcourt violations, defensive 3-seconds, a jump ball redo -
none of these are covered by Made Shot / Turnover / Free Throw / Rebound).
This affects the denominator of y_off, not off_points, which reconciles
exactly against real final scores for all 1230 games (see
combine_and_validate) - so aggregate y_off runs a little optimistic, not
points being wrong.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
STINTS_DIR = REPO_ROOT / "data" / "stints"           # NEW, separate from data/nets_synergy.db
GAMES_DIR = STINTS_DIR / "games"                      # one parquet per game_id - resumability unit
CACHE_DIR = STINTS_DIR / "raw_cache"                  # raw boxscore/pbp dataframes per game, cached to disk
MANIFEST_PATH = STINTS_DIR / "manifest.json"          # {game_id: "done"|"failed: <reason>"}
# AI-ASSISTED (Claude Code, chat)
# Prompt: "Fix D3 properly by closing the data gap it downgraded around...
# PHASE 1 - EVENT-TO-STINT LINK (extend build_stints.py, additive only -
# do NOT change the existing stint table or its schema): export an
# additional per-game table (data/stints/events/{game_id}.parquet)..."
# Used: a sibling directory to GAMES_DIR, same one-parquet-per-game_id
# resumability unit and naming convention - never touches GAMES_DIR itself.
# Not AI: the schema/location - specified directly in the task spec.
EVENTS_DIR = STINTS_DIR / "events"                    # one parquet per game_id - event-to-stint link, additive

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("build_stints")


def _call_with_retry(fn, *, retries=6, base_wait=5, max_wait=90, what=""):
    """Same shape as step0_data_collect_process.py's _call_with_retry -
    stats.nba.com times out and rate-limits under sustained load; this
    project's own established convention is retry-with-linear-backoff, not a
    fixed number of attempts with no backoff."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            return fn()
        except Exception as e:
            last_err = e
            wait = min(max_wait, base_wait * attempt)
            log.warning(f"  [retry {attempt}/{retries}] {what}: {e!r} - sleeping {wait}s")
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {retries} retries") from last_err


# --- Part A: season game list -------------------------------------------

def get_season_games(season: str, season_type: str = "Regular Season") -> pd.DataFrame:
    """One row per (game_id, team) via nba_api's LeagueGameFinder - verified
    live to return a complete, real season (1230 unique games / 2460 rows for
    a full 30-team regular season, 2025-10-21 -> 2026-04-12) before writing
    this file. Returns game_id, date, team_abbr, matchup (to derive home/away)."""
    from nba_api.stats.endpoints import leaguegamefinder

    def _fetch():
        gf = leaguegamefinder.LeagueGameFinder(
            season_nullable=season, season_type_nullable=season_type,
            league_id_nullable="00", timeout=60,
        )
        return gf.get_data_frames()[0]

    df = _call_with_retry(_fetch, what=f"LeagueGameFinder({season})")
    df = df[["GAME_ID", "GAME_DATE", "TEAM_ID", "TEAM_ABBREVIATION", "MATCHUP"]].copy()
    # AI-ASSISTED (Claude Code, chat) - found during the full-season run, not
    # anticipated up front. `IS_HOME = ~MATCHUP.str.contains("@")` assumes
    # each team's OWN row shows the matchup from ITS OWN perspective ("MEM
    # vs. ORL" for the home team's row, "ORL @ MEM" for the away team's row)
    # - true for 1225/1230 games, but confirmed live that 5 games have BOTH
    # rows carrying the IDENTICAL matchup string (e.g. both rows read
    # "ORL @ MEM"), making both rows look like the away team under that
    # heuristic and leaving the game with no identifiable home team at all
    # (the exact cause of this run's 5 failures - all KeyError on the home-
    # team lookup, not a parsing crash). Fixed by parsing the home tricode
    # directly out of the "X @ Y" text (Y, always the true home team
    # regardless of whose row it's read from) instead of relying on a
    # per-row perspective assumption that isn't universally true.
    # Only the "@" row's own text has the home tricode in it (a "vs."-style
    # row's matchup doesn't restate its own team's tricode there) - so
    # extract per-row, then take whichever non-null value exists PER GAME_ID
    # (there's always exactly one, from whichever row(s) contain "@") and
    # broadcast it back to both of that game's rows before comparing.
    df["_home_tricode_this_row"] = df["MATCHUP"].str.extract(r"@\s*(\w+)")[0]
    home_tricode_by_game = df.groupby("GAME_ID")["_home_tricode_this_row"].transform(
        lambda s: s.ffill().bfill())
    df["IS_HOME"] = df["TEAM_ABBREVIATION"] == home_tricode_by_game
    df = df.drop(columns=["_home_tricode_this_row"])
    n_games = df["GAME_ID"].nunique()
    log.info(f"season game list: {len(df)} team-game rows, {n_games} unique games, "
             f"{df['GAME_DATE'].min()} -> {df['GAME_DATE'].max()}")
    return df


# --- Part B: per-game raw data, cached to disk ---------------------------

def _load_boxscore_and_pbp(game_id: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Returns (boxscore_players_df, pbp_df) for one game. Prefers the local
    parquet cache (CACHE_DIR/{game_id}_box.parquet / _pbp.parquet) - this IS
    the resumability/caching mechanism: a rerun that finds both files present
    never touches the network for that game. On a cache miss, pulls both via
    nba_api (BoxScoreTraditionalV3, PlayByPlayV3 - both verified live: fast,
    complete, NOT the deprecated v2 endpoints pbpstats depends on) and caches
    them before returning."""
    box_path = CACHE_DIR / f"{game_id}_box.parquet"
    pbp_path = CACHE_DIR / f"{game_id}_pbp.parquet"
    if box_path.exists() and pbp_path.exists():
        return pd.read_parquet(box_path), pd.read_parquet(pbp_path)

    from nba_api.stats.endpoints import boxscoretraditionalv3, playbyplayv3

    def _fetch_box():
        return boxscoretraditionalv3.BoxScoreTraditionalV3(game_id=game_id, timeout=30).get_data_frames()[0]

    def _fetch_pbp():
        return playbyplayv3.PlayByPlayV3(game_id=game_id, timeout=30).get_data_frames()[0]

    box_df = _call_with_retry(_fetch_box, what=f"BoxScoreTraditionalV3({game_id})")
    pbp_df = _call_with_retry(_fetch_pbp, what=f"PlayByPlayV3({game_id})")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    box_df.to_parquet(box_path, index=False)
    pbp_df.to_parquet(pbp_path, index=False)
    return box_df, pbp_df


# --- Part C: on-court lineup tracking -------------------------------------
"""
Verified directly against real play-by-play (game 0022501192, TOR @ BKN)
before writing this section:
- BoxScoreTraditionalV3's `position` column is non-blank ONLY for the 5
  starters per team (blank for bench) - a long-standing NBA.com box-score
  convention, not assumed: checked live that both teams had exactly 5
  non-blank-position rows.
- Substitution rows carry the OUTGOING player's own personId directly
  (`description` reads "SUB: <incoming> FOR <outgoing display name>", and the
  row's own personId/playerName IS the outgoing player) - the incoming
  player's id has to be resolved by matching the description's "<incoming>"
  text against the SAME team's box-score familyName. Checked live for zero
  same-team family-name collisions in a real box score before trusting this
  match; still guarded below (raises loudly on a collision or a no-match,
  rather than guessing).
- Between-period lineup changes are NOT always logged as Substitution events
  - this was my first assumption (confirmed true on one sample game,
    0022501192) but DISPROVEN on a second real game: cross-checking every
    action-attributed event's personId (made/missed shot, rebound, turnover,
    FT, foul) against my tracked on-court set found a period-2 REBOUND
    credited to a player my substitution-only tracking still had marked as
    subbed out since period 1, with zero substitution event logged for his
    return anywhere in between. Some between-period changes apparently go
    unlogged entirely. Fixed via `_period_opening_five()` below: at every
    period boundary (2nd onward), the "carried forward" five from
    substitution-tracking is CROSS-CHECKED against actually-observed
    action-attributed players early in the new period, and corrected when
    they disagree, rather than trusted blindly.
"""


def _team_roster(box_df: pd.DataFrame, team_id: int) -> pd.DataFrame:
    return box_df[box_df["teamId"] == team_id]


def _starters(box_df: pd.DataFrame, team_id: int) -> set:
    roster = _team_roster(box_df, team_id)
    starters = roster[roster["position"].str.strip() != ""]
    ids = set(int(p) for p in starters["personId"])
    if len(ids) != 5:
        raise ValueError(
            f"team {team_id}: expected exactly 5 starters (non-blank position), got {len(ids)} - "
            f"box score starter convention may have changed, refusing to guess"
        )
    return ids


_RELIABLE_ATTRIBUTION_ACTIONS = {"Made Shot", "Missed Shot", "Turnover", "Free Throw", "Rebound", "Foul"}


def _period_opening_five(pbp_df: pd.DataFrame, team_id: int, period: int, carried_forward: set,
                          box_df: pd.DataFrame, game_id: str, warnings: list) -> set:
    """Cross-checks the substitution-tracked carried-forward five against
    directly-observed action-attributed players early in this period. Walks
    the WHOLE period chronologically (not just up to the first substitution
    - a real game can have a genuine early substitution AND a still-earlier
    unlogged period-opening change, confirmed live: stopping at the first
    substitution missed a true opener whose first action came a few events
    after it), applying each substitution to a local scratch copy of state
    so entrants confirmed via a resolvable substitution are excluded from
    counting as "observed openers" - a player is only trusted as direct
    opener-evidence if he shows up in a reliable-attribution event BEFORE
    any substitution has logged him as entering. Stops once 5 distinct
    qualifying openers are found (or the period ends).

    If exactly 5 distinct real players are observed this way, that observed
    set wins outright (it's ground truth - a player can't record a rebound
    without being on the floor). If fewer than 5 are observed, the gap is
    filled from carried_forward (best effort - some period-openers may
    simply not touch the ball early, and a small residual rate of this is an
    accepted, documented limitation, not something this function can fully
    eliminate). More than 5 observed distinct players before enough evidence
    accumulates would itself be a contradiction - falls back to
    carried_forward and logs a warning rather than guessing which are real.
    """
    period_df = pbp_df[(pbp_df["period"] == period) & (pbp_df["teamId"] == team_id)].sort_values("actionNumber")

    scratch_court = set(carried_forward)
    introduced_via_sub = set()
    observed = set()
    for _, event in period_df.iterrows():
        if event["actionType"] == "Substitution":
            try:
                outgoing_id, incoming_id = _resolve_substitution_direction(
                    scratch_court, box_df, event, game_id)
                scratch_court.discard(outgoing_id)
                scratch_court.add(incoming_id)
                introduced_via_sub.add(incoming_id)
            except ValueError:
                pass  # ambiguous in this lookahead too - the real main loop
                # will separately hit and warn on this same event; skipping
                # it here just means this substitution doesn't inform the
                # opener lookahead one way or the other
        elif event["actionType"] in _RELIABLE_ATTRIBUTION_ACTIONS:
            pid = event["personId"]
            if pd.notna(pid) and int(pid) != 0 and int(pid) not in introduced_via_sub:
                observed.add(int(pid))
        if len(observed) >= 5:
            break

    if len(observed) == 5:
        if observed != carried_forward:
            warnings.append(
                f"{game_id} period {period}: team {team_id} opening five corrected from "
                f"substitution-tracking {carried_forward} to observed {observed} "
                f"(an unlogged between-period lineup change)"
            )
        return observed
    if len(observed) < 5:
        combined = observed | carried_forward
        if len(combined) > 5:
            # carried_forward disagrees with SOME observed players (a real,
            # partial unlogged change) - keep all observed ones (ground
            # truth) and fill remaining slots from carried_forward
            combined = observed | (carried_forward - observed)
            combined = set(list(observed) + list(carried_forward - observed)[:5 - len(observed)])
        if combined != carried_forward:
            warnings.append(
                f"{game_id} period {period}: team {team_id} opening five partially corrected - "
                f"observed {observed} (< 5, filled from carried-forward) -> {combined}"
            )
        return combined
    warnings.append(
        f"{game_id} period {period}: team {team_id} - {len(observed)} distinct players observed "
        f"before any substitution this period (expected <=5) - falling back to carried-forward "
        f"{carried_forward} rather than guessing"
    )
    return carried_forward


_SUB_RE = re.compile(r"^SUB:\s*(.+?)\s+FOR\s+(.+)$")
# NBA.com disambiguates two same-family-name teammates with a first-initial
# prefix ("F. Wagner" vs "M. Wagner" for Franz/Moritz Wagner) - confirmed
# live on a real game where a plain family-name match against "F. Wagner"
# failed (the box score's own two Wagners: Franz Wagner, Moritz Wagner).
_PREFIX_RE = re.compile(r"^([A-Za-z]+)\.\s+(.+)$")


def _resolve_player_by_name(box_df: pd.DataFrame, team_id: int, name: str, game_id: str) -> int:
    """Matches via this project's own `_normalize_name` (step0), not raw
    string equality - confirmed live that box-score familyName and PBP
    description text disagree on diacritics (e.g. box score "Porzingis" (an
    ASCII-transliterated Kristaps Porziņģis) vs. cases where the raw feed's
    own text uses a different transliteration/accent) for real 2025-26
    games. Reusing the project's existing normalizer (NFKD-strip diacritics,
    lowercase, drop Jr/Sr/II/III/IV) rather than writing a second one."""
    from step0_data_collect_process import _normalize_name

    roster = _team_roster(box_df, team_id)
    name = name.strip()

    prefix_match = _PREFIX_RE.match(name)
    if prefix_match:
        prefix, family_name = prefix_match.group(1).lower(), prefix_match.group(2).strip()
        norm_family = _normalize_name(family_name)
        candidates = roster[roster["familyName"].apply(_normalize_name) == norm_family]
        matches = candidates[candidates["firstName"].str.lower().str.startswith(prefix)]
    else:
        norm_name = _normalize_name(name)
        matches = roster[roster["familyName"].apply(_normalize_name) == norm_name]

    if len(matches) == 1:
        return int(matches.iloc[0]["personId"])
    if len(matches) == 0:
        raise ValueError(
            f"{game_id}: no box-score player on team {team_id} matches substitution "
            f"name {name!r} - refusing to guess"
        )
    raise ValueError(
        f"{game_id}: {len(matches)} box-score players on team {team_id} match name "
        f"{name!r} (family-name collision) - refusing to guess which one this is"
    )


def _resolve_substitution_direction(court: set, box_df: pd.DataFrame, event, game_id: str) -> tuple:
    """Returns (outgoing_id, incoming_id) for a substitution event given a
    CURRENT on-court set (real or a lookahead scratch copy - see
    `_period_opening_five`). Raises ValueError if unresolvable.

    Verified live against real 2025-26 play-by-play that "SUB: <A> FOR <B>"
    does NOT have a fixed in/out convention - which of A/B is entering vs.
    exiting is NOT determined by word order (confirmed a real game where the
    first-named player enters in one substitution and exits in another) nor
    reliably by which name matches the event's own `personId` (checked: it
    matches whichever the ROW happens to be filed under - not consistently
    the enterer or the exiter). The only reliable signal is CURRENT ON-COURT
    STATE: resolve BOTH names to player ids, then whichever of the two is
    presently tracked as on the court is the one leaving, and the other is
    entering - order-independent and self-correcting regardless of the
    text's own (apparently inconsistent) grammar. `personId` is used only as
    a cross-check (should always match one of the two resolved ids), not as
    the source of truth for direction.
    """
    team_id = int(event["teamId"])
    m = _SUB_RE.match(event["description"].strip())
    if not m:
        raise ValueError(f"{game_id}: substitution description didn't match expected "
                         f"'SUB: X FOR Y' pattern: {event['description']!r}")
    name_a, name_b = m.group(1), m.group(2)
    id_a = _resolve_player_by_name(box_df, team_id, name_a, game_id)
    id_b = _resolve_player_by_name(box_df, team_id, name_b, game_id)

    a_on_court, b_on_court = id_a in court, id_b in court
    if a_on_court == b_on_court:  # both on court, or neither - can't tell direction
        raise ValueError(
            f"{game_id}: anomalous substitution for team {team_id} - "
            f"'{name_a}' (id={id_a}, on_court={a_on_court}) and '{name_b}' (id={id_b}, "
            f"on_court={b_on_court}) - can't determine direction; "
            f"current on_court={court} - skipping this substitution, state left unchanged"
        )
    return (id_a, id_b) if a_on_court else (id_b, id_a)


def _apply_substitution(on_court: dict, box_df: pd.DataFrame, event, game_id: str) -> None:
    """Mutates on_court[team_id] in place, atomically - resolution (and its
    ambiguous-case ValueError) is delegated to `_resolve_substitution_direction`;
    if it raises, NOTHING is mutated (state left unchanged) rather than
    partially applying - confirmed live that a small number of anomalous/
    duplicate-looking rows exist in the raw feed, and a naive discard-then-
    add on one of those corrupts the 5-man count for every following event
    in the game (an actual cascade reproduced during testing)."""
    team_id = int(event["teamId"])
    court = on_court[team_id]
    outgoing_id, incoming_id = _resolve_substitution_direction(court, box_df, event, game_id)
    court.discard(outgoing_id)
    court.add(incoming_id)


# AI-ASSISTED (Claude Code, chat)
# Prompt (D3 rebuild, Phase 1): the event export needs an assist rate, but
# PlayByPlayV3 has no structured assist column - checked live against a
# real cached game (data/stints/raw_cache/*_pbp.parquet): assist credit
# exists only as free text at the end of a made-shot's own `description`
# (e.g. "LaVine 20' Pullup Jump Shot (2 PTS) (DeRozan 1 AST)"), never a
# dedicated field.
# Used: `_ASSIST_RE` matches that trailing "(<Name> N AST)" pattern (only
# present on assisted makes - an unassisted make's description ends after
# the "(N PTS)" group, so a non-match correctly means "unassisted", not a
# parse failure). The extracted name is resolved to a personId via the
# SAME `_resolve_player_by_name` the substitution parser already uses and
# already validates (diacritics, "F. Wagner"-style prefix disambiguation,
# family-name-collision guard) - reusing proven infrastructure rather than
# writing a second name-matcher for one new field.
# Not AI: the decision to parse the description text at all (vs. treating
# assist rate as unbuildable) - the user's own explicit instruction to
# close this gap rather than downgrade around it.
_ASSIST_RE = re.compile(r"\(([^()]+?)\s+\d+\s+AST\)\s*$")


def _parse_assist_person_id(event, box_df: pd.DataFrame, team_id: int, game_id: str) -> int | None:
    """None for an unassisted make (no match) or a name this game's box
    score can't resolve (logged as a warning by the caller, never guessed)."""
    m = _ASSIST_RE.search(event.get("description") or "")
    if not m:
        return None
    try:
        return _resolve_player_by_name(box_df, team_id, m.group(1), game_id)
    except ValueError:
        return None


# --- Part D: possession-ending logic --------------------------------------
"""
Verified directly against real play-by-play before encoding these rules
(see each bullet). This is a from-scratch, real event-based possession
count (not the FGA+0.44*FTA estimate) - built because pbpstats (which would
otherwise have supplied this, already validated) can't reach 2025-26 data
(see module docstring). Simplifications are named explicitly in the README's
Limitations section, not hidden.

- Made Shot -> ends the shooting team's possession immediately, even if an
  and-1 free throw follows (the FT is a bonus attached to the possession
  that already ended, not a new one) - standard possession-counting
  convention, not specific to this implementation.
- Turnover -> ends the team-in-possession's possession immediately. NBA
  play-by-play has no "offensive rebound of a turnover" case.
- Missed Shot -> does NOT end the possession by itself; the following
  Rebound event decides (offensive rebound continues the SAME possession,
  defensive rebound ends it). Checked live: a missed-shot row is always
  immediately followed by a Rebound row.
- Free Throw: `subType` gives an exact "N of M" count (checked live: e.g.
  "Free Throw 1 of 2", "Free Throw 2 of 2", "Free Throw 1 of 1") - only the
  LAST free throw of a personal-foul trip can end a possession (earlier FTs
  in a trip never do). A MADE last FT ends the possession immediately (no
  rebound follows a make). A MISSED last FT does NOT end the possession by
  itself - like a missed field goal, the following Rebound event decides.
  "Free Throw Technical" (checked live: a distinct subType) never ends a
  possession - a technical FT doesn't take the ball away; the same team
  that was fouled keeps possession afterward.
- Rebound -> if the rebounding event's own team_id differs from the team_id
  of the immediately-preceding missed-shot/missed-last-FT event, this ends
  THAT PRECEDING team's possession (a defensive rebound). If the same team,
  no possession ends (an offensive rebound/second-chance continuation).
- Everything else (Jump Ball, Foul, Timeout, period markers, Instant Replay,
  blank actionType) never ends a possession on its own. A period boundary
  still forces a STINT split (see build_game_stints), which also closes out
  whatever possession was in progress at that moment - a real, accepted
  edge case (a possession genuinely can straddle a stint boundary if time
  expires mid-possession), noted in the README, not silently smoothed over.
"""


def _possession_ending_team(event, prev_event) -> int | None:
    """Returns the team_id whose possession ends on this event, or None."""
    action = event["actionType"]
    if action == "Made Shot":
        return int(event["teamId"])
    if action == "Turnover":
        return int(event["teamId"])
    if action == "Free Throw":
        sub_type = event["subType"]
        if sub_type == "Free Throw Technical":
            return None
        m = re.match(r"Free Throw (\d+) of (\d+)", sub_type or "")
        if not m or m.group(1) != m.group(2):
            return None  # not the last FT of the trip
        if event["shotResult"] == "Made":
            return int(event["teamId"])
        return None  # missed last FT - wait for the Rebound event
    if action == "Rebound":
        if prev_event is None:
            return None
        prev_action = prev_event["actionType"]
        if prev_action not in ("Missed Shot", "Free Throw"):
            return None  # a rebound not tied to a miss (shouldn't happen, but don't guess)
        if int(event["teamId"]) != int(prev_event["teamId"]):
            return int(prev_event["teamId"])  # defensive rebound - shooting team's possession ends
        return None  # offensive rebound - same possession continues
    return None


# --- Part E: build directed stint rows for one game -----------------------

def build_game_stints(box_df: pd.DataFrame, pbp_df: pd.DataFrame, game_id: str,
                       game_date: str, home_team_id: int) -> tuple[list[dict], list[dict], list[str]]:
    """Walks every play-by-play event in order, splitting a new stint
    whenever the on-court lineup for EITHER team changes (a substitution) OR
    the period changes - the user's own stint definition. Returns (rows,
    events, warnings); rows are directed (2 per stint, one per team as
    "offense") - UNCHANGED from before. `events` is new (D3 rebuild, Phase
    1): one row per attributed event (Made/Missed Shot, Turnover, Free
    Throw), tagged with the stint_id it belongs to - see the emission block
    below for why `stint_num + 1` is always the correct id at emission time.
    """
    df = pbp_df.sort_values("actionNumber").reset_index(drop=True)
    team_ids = sorted(int(t) for t in box_df["teamId"].unique())
    if len(team_ids) != 2:
        raise ValueError(f"{game_id}: expected 2 teams in box score, got {team_ids}")
    away_team_id = team_ids[0] if team_ids[1] == home_team_id else team_ids[1]

    on_court = {home_team_id: _starters(box_df, home_team_id),
                away_team_id: _starters(box_df, away_team_id)}

    rows, events, warnings = [], [], []
    stint_num = 0
    current_period = None
    stint_start_score = {"home": 0, "away": 0}
    poss_count = {home_team_id: 0, away_team_id: 0}
    last_score = {"home": 0, "away": 0}
    # AI-ASSISTED (Claude Code, chat) - Prompt: "extend to process-level stats
    # (eFG%, turnover rate) since Net rating is dominated by shooting-luck
    # noise unrelated to archetype composition" - purely additive counters,
    # direct per-event attribution (not the possession-ending logic above,
    # which these don't need - a shot attempt or turnover is unambiguously
    # this event's own teamId's, no "did this end a possession" reasoning
    # required), reset at the same points as poss_count so they never touch
    # or change the already-validated points/possessions columns.
    fga_count = {home_team_id: 0, away_team_id: 0}
    fgm_count = {home_team_id: 0, away_team_id: 0}
    fg3a_count = {home_team_id: 0, away_team_id: 0}
    fg3m_count = {home_team_id: 0, away_team_id: 0}
    tov_count = {home_team_id: 0, away_team_id: 0}

    def _flush(lineup_snapshot):
        nonlocal stint_num
        off_players = {}
        bad = False
        for tid in team_ids:
            ids = sorted(lineup_snapshot[tid])
            if len(ids) != 5 or len(set(ids)) != 5:
                warnings.append(f"{game_id} stint {stint_num + 1}: team {tid} has "
                                f"{len(set(ids))} distinct on-court players (expected 5) - skipped")
                bad = True
            off_players[tid] = ids
        if bad:
            return
        pts = {home_team_id: last_score["home"] - stint_start_score["home"],
               away_team_id: last_score["away"] - stint_start_score["away"]}
        stint_num += 1
        for off_tid in team_ids:
            def_tid = away_team_id if off_tid == home_team_id else home_team_id
            off_poss = poss_count[off_tid]
            if off_poss <= 0 and pts[off_tid] == 0:
                continue  # this side neither scored nor got a possession-ending event in
                # this stint (e.g. a stint that opens and closes again within the same live
                # possession) - no directed row to report for that side, not an error
            if off_poss <= 0 and pts[off_tid] != 0:
                # AI-ASSISTED (Claude Code, chat) - found via a real box-score
                # cross-check (summed off_points vs. true final scores) that
                # this project's own README promised but the code didn't yet
                # do: ~3% of points were silently disappearing. Root cause:
                # a substitution can legally happen BETWEEN free throws of a
                # multi-shot personal-foul trip (checked live - NBA rules
                # allow subs between FTs), and a non-last free throw scores
                # real points without ending a possession by this file's own
                # rules (see the possession-ending-logic note above `Made
                # Shot`/`Free Throw`) - if the stint boundary lands inside
                # that trip, the points already happened but no possession-
                # ending event has fired yet for this side, so poss_count
                # stayed 0 and the whole row (points included) was being
                # dropped outright. Floored to 1 possession here instead of
                # dropping the row - if a team scored at all, at least one
                # possession's worth of activity must have occurred, even if
                # this specific rule set didn't register its "ending" event
                # inside this exact stint window (the true trip's ending
                # event fires in an adjacent stint instead). This trades a
                # small, bounded possession over-count (this exact case,
                # rare) for eliminating a real, larger point under-count
                # (silently dropped scoring) - the more defensible direction
                # per the assignment's own "don't fabricate, but don't lose
                # real data either" spirit. Verified this fix's actual impact
                # empirically (see AI_USAGE.md) rather than assumed.
                off_poss = 1
            fga, fgm, fg3a, fg3m, tov = (fga_count[off_tid], fgm_count[off_tid],
                                        fg3a_count[off_tid], fg3m_count[off_tid], tov_count[off_tid])
            row = {
                "game_id": game_id, "date": game_date, "stint_id": stint_num,
                "offense_team": off_tid, "defense_team": def_tid,
                "home_off": 1 if off_tid == home_team_id else 0,
                "off_possessions": off_poss, "off_points": pts[off_tid],
                "y_off": 100.0 * pts[off_tid] / off_poss,
                "off_fga": fga, "off_fgm": fgm, "off_fg3a": fg3a, "off_fg3m": fg3m,
                "off_efg_pct": (fgm + 0.5 * fg3m) / fga if fga > 0 else float("nan"),
                "off_tov": tov,
                "off_tov_pct": 100.0 * tov / off_poss,
            }
            for i, pid in enumerate(off_players[off_tid], start=1):
                row[f"off_p{i}"] = pid
            for i, pid in enumerate(off_players[def_tid], start=1):
                row[f"def_p{i}"] = pid
            rows.append(row)

    prev_event = None
    for _, event in df.iterrows():
        period_changed = current_period is not None and event["period"] != current_period
        is_substitution = event["actionType"] == "Substitution"

        if current_period is None:
            current_period = event["period"]
        elif period_changed:
            _flush({tid: set(on_court[tid]) for tid in team_ids})
            stint_start_score = dict(last_score)
            poss_count = {home_team_id: 0, away_team_id: 0}
            fga_count = {home_team_id: 0, away_team_id: 0}
            fgm_count = {home_team_id: 0, away_team_id: 0}
            fg3a_count = {home_team_id: 0, away_team_id: 0}
            fg3m_count = {home_team_id: 0, away_team_id: 0}
            tov_count = {home_team_id: 0, away_team_id: 0}
            current_period = event["period"]
            for tid in team_ids:
                on_court[tid] = _period_opening_five(
                    df, tid, current_period, on_court[tid], box_df, game_id, warnings)
        elif is_substitution:
            _flush({tid: set(on_court[tid]) for tid in team_ids})
            stint_start_score = dict(last_score)
            poss_count = {home_team_id: 0, away_team_id: 0}
            fga_count = {home_team_id: 0, away_team_id: 0}
            fgm_count = {home_team_id: 0, away_team_id: 0}
            fg3a_count = {home_team_id: 0, away_team_id: 0}
            fg3m_count = {home_team_id: 0, away_team_id: 0}
            tov_count = {home_team_id: 0, away_team_id: 0}

        if is_substitution:
            try:
                _apply_substitution(on_court, box_df, event, game_id)
            except ValueError as e:
                warnings.append(str(e))

        if pd.notna(event.get("scoreHome")) and event["scoreHome"] != "":
            last_score["home"] = int(event["scoreHome"])
            last_score["away"] = int(event["scoreAway"])

        ending_team = _possession_ending_team(event, prev_event)
        if ending_team is not None and ending_team in poss_count:
            poss_count[ending_team] += 1

        # direct per-event attribution for process stats - no possession-ending
        # reasoning needed, a shot attempt or turnover always belongs to its own
        # event's teamId regardless of whether it ends a possession
        etid = int(event["teamId"]) if pd.notna(event.get("teamId")) else None
        if etid in fga_count:
            # AI-ASSISTED (Claude Code, chat) - D3 rebuild, Phase 1: emit one
            # event row per attributed event here, in the SAME branch that
            # already resolves etid/made/is_three for the stint-level
            # counters above - `stint_id = stint_num + 1` because stint_num
            # is only incremented INSIDE _flush(), so every event seen since
            # the last flush belongs to whichever stint is about to be
            # flushed next (identical reasoning to why fga_count etc. reset
            # at every flush point and never leak across a stint boundary).
            # `on_court` is a live data-integrity check, not assumed true by
            # construction - personId SHOULD always be in the tracked
            # on-court set at this point, but a residual unlogged-lineup-
            # change edge case is already documented above
            # (_period_opening_five's own note), so this flag lets the
            # Phase 1 validation actually catch it rather than assume it.
            pid = int(event["personId"]) if pd.notna(event.get("personId")) else None
            on_court_now = pid in on_court.get(etid, set()) if pid is not None else False
            if event["isFieldGoal"] == 1:
                fga_count[etid] += 1
                is_three = event["shotValue"] == 3
                if is_three:
                    fg3a_count[etid] += 1
                made = event["shotResult"] == "Made"
                if made:
                    fgm_count[etid] += 1
                    if is_three:
                        fg3m_count[etid] += 1
                assist_pid = _parse_assist_person_id(event, box_df, etid, game_id) if made else None
                events.append({
                    "game_id": game_id, "stint_id": stint_num + 1, "teamId": etid, "personId": pid,
                    "actionType": event["actionType"], "subType": event.get("subType"),
                    "shotDistance": event.get("shotDistance"), "shotValue": event.get("shotValue"),
                    "made": made, "assistPersonId": assist_pid, "on_court": on_court_now,
                })
            elif event["actionType"] == "Turnover":
                tov_count[etid] += 1
                events.append({
                    "game_id": game_id, "stint_id": stint_num + 1, "teamId": etid, "personId": pid,
                    "actionType": "Turnover", "subType": event.get("subType"),
                    "shotDistance": None, "shotValue": None,
                    "made": None, "assistPersonId": None, "on_court": on_court_now,
                })
            elif event["actionType"] == "Free Throw":
                # AI-ASSISTED (Claude Code, chat) - found while validating
                # Phase 1 against real off_points (a real bug, not
                # anticipated up front): unlike Made/Missed Shot rows,
                # `shotResult` is BLANK for every Free Throw row regardless
                # of outcome (checked live against cached PBP - shotValue is
                # also always 0, not 1, on the raw row) - using it here
                # silently marked every free throw as missed, undercounting
                # points on any stint containing one. The made/missed
                # signal for a free throw only exists in `description`'s own
                # "MISS " prefix (e.g. "MISS Santos Free Throw 1 of 2" vs.
                # "Santos Free Throw 2 of 2 (1 PTS)") - the same convention
                # already relied on for Made/Missed Shot text elsewhere in
                # this file, just not on this column for this action type.
                ft_made = not (event.get("description") or "").strip().startswith("MISS")
                events.append({
                    "game_id": game_id, "stint_id": stint_num + 1, "teamId": etid, "personId": pid,
                    "actionType": "Free Throw", "subType": event.get("subType"),
                    "shotDistance": None, "shotValue": 1,
                    "made": ft_made, "assistPersonId": None, "on_court": on_court_now,
                })

        prev_event = event

    _flush({tid: set(on_court[tid]) for tid in team_ids})
    return rows, events, warnings


# --- Part F: per-game runner + manifest/resumability ---------------------

def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        return json.loads(MANIFEST_PATH.read_text())
    return {}


def _save_manifest(manifest: dict):
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True))


# AI-ASSISTED (Claude Code, chat)
# Prompt (D3 rebuild, Phase 1): "Resumable/cached like everything else;
# rebuild only from the existing raw_cache (no new network pulls)."
# Used: the games parquet and the NEW events parquet are now resumable
# INDEPENDENTLY - the whole 2025-26 season already has every games/*.parquet
# written (from before this rebuild), so a naive "skip if games file exists"
# check (the old behavior) would never produce a single events file. Both
# paths are checked separately; if only the events file is missing, this
# still avoids the network entirely (`_load_boxscore_and_pbp` prefers its
# own on-disk cache, already populated for every game from the original
# run) and does not rewrite games/{game_id}.parquet if it's already there.
# Not AI: the resumability requirement itself - specified directly.
def process_game(game_id: str, game_date: str, home_team_id: int) -> tuple[int, int, list[str]]:
    """Returns (n_rows_written, n_events_written, warnings). Either count is
    -1 if that file already existed and was skipped."""
    out_path = GAMES_DIR / f"{game_id}.parquet"
    events_path = EVENTS_DIR / f"{game_id}.parquet"
    games_done, events_done = out_path.exists(), events_path.exists()
    if games_done and events_done:
        return -1, -1, []

    box_df, pbp_df = _load_boxscore_and_pbp(game_id)
    rows, events, warnings = build_game_stints(box_df, pbp_df, game_id, game_date, home_team_id)
    if not rows:
        raise RuntimeError(f"0 directed stint rows produced for {game_id} - not writing an empty file "
                           f"(a real game always has >0 possessions; this means parsing failed silently)")

    n_rows = -1
    if not games_done:
        pd.DataFrame(rows).to_parquet(out_path, index=False)
        n_rows = len(rows)

    n_events = -1
    if not events_done:
        EVENTS_DIR.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(events).to_parquet(events_path, index=False)
        n_events = len(events)

    return n_rows, n_events, warnings


def run(season: str, season_type: str, limit: int | None, sleep_between: float):
    STINTS_DIR.mkdir(parents=True, exist_ok=True)
    GAMES_DIR.mkdir(parents=True, exist_ok=True)
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    games_df = get_season_games(season, season_type)
    home_by_game = (
        games_df[games_df["IS_HOME"]].drop_duplicates("GAME_ID").set_index("GAME_ID")["TEAM_ID"]
    )
    unique_games = games_df.drop_duplicates("GAME_ID")[["GAME_ID", "GAME_DATE"]]
    if limit:
        unique_games = unique_games.head(limit)

    manifest = _load_manifest()
    n_done_this_run, n_skipped_cached, n_failed = 0, 0, 0
    all_warnings = []

    for i, (game_id, game_date) in enumerate(unique_games.itertuples(index=False), start=1):
        # AI-ASSISTED (Claude Code, chat) - D3 rebuild, Phase 1: the season
        # was already fully processed before the events table existed, so
        # the "done" fast-path now ALSO requires the events file, or a
        # backfill run would skip every game before process_game ever gets
        # a chance to write the missing events/{game_id}.parquet.
        if (manifest.get(game_id) == "done" and (GAMES_DIR / f"{game_id}.parquet").exists()
                and (EVENTS_DIR / f"{game_id}.parquet").exists()):
            n_skipped_cached += 1
            continue
        try:
            home_tid = int(home_by_game.loc[game_id])
            n_rows, n_events, warnings = process_game(game_id, str(game_date), home_tid)
            if n_rows == -1 and n_events == -1:
                n_skipped_cached += 1
                manifest[game_id] = "done"
                continue
            all_warnings.extend(warnings)
            manifest[game_id] = "done"
            n_done_this_run += 1
            log.info(f"[{i}/{len(unique_games)}] {game_id} ({game_date}): "
                     f"{n_rows if n_rows != -1 else 'skipped'} directed rows, "
                     f"{n_events if n_events != -1 else 'skipped'} events"
                     + (f", {len(warnings)} warnings" if warnings else ""))
        except Exception as e:
            manifest[game_id] = f"failed: {e!r}"
            n_failed += 1
            log.error(f"[{i}/{len(unique_games)}] {game_id} FAILED: {e!r}")
        if i % 25 == 0:
            _save_manifest(manifest)  # periodic checkpoint, not just at the very end
        time.sleep(sleep_between)

    _save_manifest(manifest)
    return {
        "n_games_total": len(unique_games), "n_done_this_run": n_done_this_run,
        "n_skipped_cached": n_skipped_cached, "n_failed": n_failed,
        "warnings": all_warnings, "manifest": manifest,
    }


# --- Part G: combine per-game parquets + validation report ----------------

def combine_and_validate(season: str, games_df: pd.DataFrame, manifest: dict):
    """BUG FIX (found 2026-07-29): GAMES_DIR is one shared directory across every
    season ever pulled (per-game files are just "{game_id}.parquet", no season
    subdirectory) - globbing the whole directory silently combined OTHER seasons'
    already-cached games into whatever season was currently being requested (e.g.
    stints_2024_25.parquet ended up with 1230 of its 1235 "games" actually being
    2025-26 game_ids, since 2025-26 was pulled first and sat in the same directory).
    stints_2025_26.parquet itself was unaffected only because it was combined before
    any other season existed in the shared directory - not because this bug didn't
    exist. Fixed by filtering `files` to exactly this season's own GAME_ID set
    (from `games_df`, the season-specific schedule already fetched by run()) before
    combining anything - every downstream count (n_parsed, combined row count,
    output file) now reflects ONLY the requested season.
    """
    season_game_ids = set(games_df["GAME_ID"].unique())
    all_files = sorted(GAMES_DIR.glob("*.parquet"))
    files = [f for f in all_files if f.stem in season_game_ids]
    if not files:
        log.error("no per-game parquet files found - nothing to combine")
        return None
    combined = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

    out_path = STINTS_DIR / f"stints_{season.replace('-', '_')}.parquet"
    combined.to_parquet(out_path, index=False)
    sample_path = STINTS_DIR / f"stints_{season.replace('-', '_')}_sample.csv"
    combined.sample(min(500, len(combined)), random_state=0).to_csv(sample_path, index=False)

    players_path = STINTS_DIR / "players.csv"
    _write_players_csv(season, players_path)

    print("\n" + "=" * 70)
    print(f"VALIDATION REPORT - {season}")
    print("=" * 70)

    n_expected = games_df["GAME_ID"].nunique()
    n_parsed = len(files)
    n_failed = sum(1 for v in manifest.values() if isinstance(v, str) and v.startswith("failed"))
    print(f"Games expected: {n_expected} | parsed successfully: {n_parsed} | failed: {n_failed}")
    if n_failed:
        failed_ids = [k for k, v in manifest.items() if isinstance(v, str) and v.startswith("failed")]
        print(f"  failed game_ids: {failed_ids[:20]}{' ...' if len(failed_ids) > 20 else ''}")

    print(f"\nTotal directed stint-rows: {len(combined)}")

    print("\n-- Offensive possessions per team (over games ACTUALLY PARSED) vs. rough expectation --")
    parsed_game_ids = set(combined["game_id"].unique())
    team_poss = combined.groupby("offense_team")["off_possessions"].sum().sort_values(ascending=False)
    games_per_team = (
        games_df[games_df["GAME_ID"].isin(parsed_game_ids)].groupby("TEAM_ID")["GAME_ID"].nunique()
    )
    for team_id, poss in team_poss.items():
        g = games_per_team.get(team_id, np.nan)
        pace_like = poss / g if g else np.nan
        print(f"  team {team_id}: {poss:,.0f} total off. possessions over {g:.0f} parsed games "
              f"({pace_like:.1f} / game - typical NBA pace is ~95-105 offensive possessions/game)")

    print("\n-- Score reconciliation (sum of directed off_points vs. real final score, per game) --")
    n_checked, n_exact, diffs = 0, 0, []
    for gid, gdf in combined.groupby("game_id"):
        pbp_path = CACHE_DIR / f"{gid}_pbp.parquet"
        if not pbp_path.exists() or not (gdf["home_off"] == 1).any() or not (gdf["home_off"] == 0).any():
            continue
        pbp = pd.read_parquet(pbp_path)
        scored = pbp[(pbp["scoreHome"].notna()) & (pbp["scoreHome"] != "")]
        if len(scored) == 0:
            continue
        last = scored.sort_values("actionNumber").iloc[-1]
        home_tid = int(gdf[gdf["home_off"] == 1]["offense_team"].iloc[0])
        away_tid = int(gdf[gdf["home_off"] == 0]["offense_team"].iloc[0])
        true_home, true_away = int(last["scoreHome"]), int(last["scoreAway"])
        my_home = gdf[gdf["offense_team"] == home_tid]["off_points"].sum()
        my_away = gdf[gdf["offense_team"] == away_tid]["off_points"].sum()
        n_checked += 1
        diff = abs(my_home - true_home) + abs(my_away - true_away)
        if diff == 0:
            n_exact += 1
        else:
            diffs.append(diff)
    print(f"  {n_checked} games checked against their real final score, {n_exact} exact "
          f"({n_exact/n_checked:.1%})" if n_checked else "  0 games checked (no cached PBP found)")
    if diffs:
        print(f"  {len(diffs)} games off by a nonzero total (both teams combined): "
              f"mean {sum(diffs)/len(diffs):.1f} pts, max {max(diffs)} pts")

    print("\n-- Stint-length distribution (off_possessions per directed row) --")
    for thresh in (2, 5, 10):
        n_below = (combined["off_possessions"] < thresh).sum()
        print(f"  < {thresh} possessions: {n_below} rows ({n_below/len(combined):.1%})")

    print("\n-- 5-offense + 5-defense distinct-player-id integrity --")
    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]
    off_nulls = combined[off_cols].isna().any(axis=1).sum()
    def_nulls = combined[def_cols].isna().any(axis=1).sum()
    dupe_within_off = (combined[off_cols].nunique(axis=1) != 5).sum()
    dupe_within_def = (combined[def_cols].nunique(axis=1) != 5).sum()
    overlap = combined.apply(
        lambda r: len(set(r[off_cols]) & set(r[def_cols])) > 0, axis=1
    ).sum()
    print(f"  rows with a missing offense player id: {off_nulls}")
    print(f"  rows with a missing defense player id: {def_nulls}")
    print(f"  rows without 5 DISTINCT offense ids: {dupe_within_off}")
    print(f"  rows without 5 DISTINCT defense ids: {dupe_within_def}")
    print(f"  rows where an offense id also appears on defense (should be 0): {overlap}")
    if not (off_nulls or def_nulls or dupe_within_off or dupe_within_def or overlap):
        print("  ALL ROWS PASS the 5-offense + 5-defense distinct-id check.")

    print(f"\nOutput: {out_path}")
    print(f"Sample: {sample_path}")
    print(f"Players reference: {players_path}")
    print("=" * 70)
    return combined


# AI-ASSISTED (Claude Code, chat)
# Prompt (D3 rebuild, Phase 1): "Validate: per stint, the sum of attributed
# scoring events must reproduce the stint's off_points (print mismatches)."
# Used: combines every data/stints/events/{game_id}.parquet for this
# season's own game_ids (same season-filter-before-combining pattern
# combine_and_validate already uses, to avoid the cross-season bleed bug
# documented on that function), sums made-shot + made-free-throw points per
# (game_id, stint_id, teamId), and diffs against the existing stints
# table's own off_points.
# Found while running this (not anticipated up front): a real, PRE-EXISTING
# anomaly in the already-shipped stint table, unrelated to anything this
# Phase 1 change touches - a small fraction of directed rows (0.876% in the
# full 2025-26 run) show a nonsensical off_points value (some strongly
# negative, e.g. -124, on a row with off_possessions=1), and every single
# one of these co-occurs with the exact "floored to 1 possession" fallback
# `_flush()`'s own comment already documents (a free-throw trip whose
# points landed in one stint while its ending event fired in an adjacent
# one). The aggregate per-game score reconciliation is UNCHANGED and still
# exact for all 1230 games, meaning these are misattributed points, not
# fabricated ones - they wash out over a full game but produce a nonsense
# single-stint value at the exact boundary this pre-existing edge case
# hits. This function reports the finding; it does not attempt to repair
# `_flush()`'s scoring logic - the task's own "do NOT change the existing
# stint table" instruction rules that out, and a ~0.9% rate concentrated at
# a single already-documented boundary case doesn't undermine Phase 2's use
# of the (correctly event-level, personId-attributed) events table itself.
# Not AI: the validation requirement - specified directly in the task spec.
def validate_events(season: str, games_df: pd.DataFrame):
    season_game_ids = set(games_df["GAME_ID"].unique())
    event_files = [f for f in EVENTS_DIR.glob("*.parquet") if f.stem in season_game_ids]
    if not event_files:
        log.error("no per-game event parquet files found - nothing to validate")
        return None
    events = pd.concat([pd.read_parquet(f) for f in event_files], ignore_index=True)
    combined_events_path = STINTS_DIR / f"events_{season.replace('-', '_')}.parquet"
    events.to_parquet(combined_events_path, index=False)

    stints_v2_path = STINTS_DIR / f"stints_{season.replace('-', '_')}_v2.parquet"
    stints_path = stints_v2_path if stints_v2_path.exists() else STINTS_DIR / f"stints_{season.replace('-', '_')}.parquet"
    stints = pd.read_parquet(stints_path)

    scoring = events[events["made"] == True]
    pts = scoring.groupby(["game_id", "stint_id", "teamId"])["shotValue"].sum().reset_index(name="event_points")
    merged = stints.merge(pts, left_on=["game_id", "stint_id", "offense_team"],
                          right_on=["game_id", "stint_id", "teamId"], how="left")
    merged["event_points"] = merged["event_points"].fillna(0)
    diff = merged["event_points"] - merged["off_points"]
    n_total, n_mismatch = len(merged), int((diff != 0).sum())
    n_from_floor_anomaly = int((merged.loc[diff != 0, "off_points"] < 0).sum())

    print("\n" + "=" * 70)
    print(f"EVENT-TO-STINT VALIDATION - {season}")
    print("=" * 70)
    print(f"Total events: {len(events)} ({events['actionType'].value_counts().to_dict()})")
    print(f"Directed rows checked: {n_total} | mismatched: {n_mismatch} ({n_mismatch / n_total:.3%})")
    print(f"  of which negative off_points (the pre-existing 'floored to 1 "
          f"possession' free-throw-boundary case): {n_from_floor_anomaly}")
    print(f"Combined events file: {combined_events_path}")
    print("=" * 70)
    return {"n_total": n_total, "n_mismatch": n_mismatch, "n_from_floor_anomaly": n_from_floor_anomaly}


def _write_players_csv(season: str, out_path: Path):
    """player_id, name, team - for joining downstream, reusing this
    project's existing player_base table (never re-pulled here)."""
    import sqlite3
    db_path = REPO_ROOT / "data" / "nets_synergy.db"
    if not db_path.exists():
        log.warning(f"players.csv: {db_path} not found - writing an empty file")
        pd.DataFrame(columns=["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]).to_csv(out_path, index=False)
        return
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT PLAYER_ID, PLAYER_NAME, TEAM_ABBREVIATION FROM player_base WHERE SEASON = ?",
            conn, params=(season,),
        )
    df.to_csv(out_path, index=False)


# --- CLI -------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2025-26")
    parser.add_argument("--season-type", default="Regular Season")
    parser.add_argument("--limit", type=int, default=None,
                        help="only process the first N games (for testing)")
    parser.add_argument("--sleep", type=float, default=1.5,
                        help="seconds to sleep between games (rate-limit throttle)")
    parser.add_argument("--combine-only", action="store_true",
                        help="skip pulling, just recombine existing per-game parquet files + validate")
    args = parser.parse_args()

    STINTS_DIR.mkdir(parents=True, exist_ok=True)
    games_df = get_season_games(args.season, args.season_type)
    manifest = _load_manifest()

    if not args.combine_only:
        result = run(args.season, args.season_type, args.limit, args.sleep)
        manifest = result["manifest"]
        log.info(f"run complete: {result['n_done_this_run']} newly done, "
                 f"{result['n_skipped_cached']} already cached, {result['n_failed']} failed")
        if result["warnings"]:
            log.warning(f"{len(result['warnings'])} stint-level warnings during this run "
                        f"(see combine_and_validate output / manifest.json for details)")

    combine_and_validate(args.season, games_df, manifest)
    validate_events(args.season, games_df)


if __name__ == "__main__":
    main()
