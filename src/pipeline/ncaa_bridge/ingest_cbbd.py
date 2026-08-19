"""
NCAA bridge - Step 0: CBBD raw data ingest.

Pulls everything the college archetype model + anchor set need from
collegebasketballdata.com and caches every raw response to disk under
ncaa_bridge_config.yaml's cbbd.raw_cache_dir, so every later step runs
fully offline and is reproducible without re-hitting the API.

Endpoints used (schema confirmed live against CBBD's own /api-docs.json
and by direct probe calls, not guessed):
  - /stats/player/season         (season only -> full D1, one call/season)
  - /stats/team/season           (season only -> full D1, one call/season;
                                   includes a full opponentStats mirror,
                                   needed for AST%/STL%/BLK%/TRB% formulas)
  - /teams/roster                (season only -> full D1, one call/season;
                                   source of PLAYER_HEIGHT_INCHES)
  - /stats/player/shooting/season (requires team OR conference per the
                                   endpoint's own param description -
                                   without either it silently returns a
                                   partial default set, not an error, not
                                   the full season - confirmed by a live
                                   probe where season-only calls returned
                                   exactly 400 rows regardless of season.
                                   Looped per conference: 34 conferences,
                                   far fewer calls than looping per team.)
  - /draft/picks                 (no params -> entire history in one call,
                                   2001-2026 confirmed live; filtered
                                   locally to the configured window)

Run: python3 src/pipeline/ncaa_bridge/ingest_cbbd.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import requests
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "ncaa_bridge_config.yaml"
BASE = "https://api.collegebasketballdata.com"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_api_key() -> str:
    env = {}
    with open(REPO_ROOT / ".env") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    return env["CBBD_API_KEY"]


def _get(path: str, params: dict, headers: dict, cfg: dict) -> list | dict:
    cbbd = cfg["cbbd"]
    last_exc = None
    for attempt in range(cbbd["retry_attempts"]):
        try:
            r = requests.get(
                f"{BASE}{path}", headers=headers, params=params,
                timeout=cbbd["request_timeout_s"],
            )
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001 - real network calls, real transient failures
            last_exc = e
            print(f"  retry {attempt + 1}/{cbbd['retry_attempts']} for {path} {params}: {e}", flush=True)
            time.sleep(cbbd["retry_sleep_s"])
    raise RuntimeError(f"Failed {path} {params} after {cbbd['retry_attempts']} attempts") from last_exc


def _cached_get(cache_path: Path, path: str, params: dict, headers: dict, cfg: dict) -> list | dict:
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    data = _get(path, params, headers, cfg)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(data))
    time.sleep(cfg["cbbd"]["inter_request_sleep_s"])
    return data


def pull_player_season(season: int, headers: dict, cfg: dict, cache_dir: Path) -> list:
    cache_path = cache_dir / f"player_season_{season}.json"
    data = _cached_get(cache_path, "/stats/player/season", {"season": season}, headers, cfg)
    print(f"  player/season {season}: {len(data)} rows", flush=True)
    return data


def pull_team_season(season: int, headers: dict, cfg: dict, cache_dir: Path) -> list:
    cache_path = cache_dir / f"team_season_{season}.json"
    data = _cached_get(cache_path, "/stats/team/season", {"season": season}, headers, cfg)
    print(f"  team/season {season}: {len(data)} rows", flush=True)
    return data


def pull_roster(season: int, headers: dict, cfg: dict, cache_dir: Path) -> list:
    cache_path = cache_dir / f"roster_{season}.json"
    data = _cached_get(cache_path, "/teams/roster", {"season": season}, headers, cfg)
    n_players = sum(len(t["players"]) for t in data)
    print(f"  roster {season}: {len(data)} teams, {n_players} player entries", flush=True)
    return data


def pull_conferences(headers: dict, cfg: dict, cache_dir: Path) -> list:
    cache_path = cache_dir / "conferences.json"
    data = _cached_get(cache_path, "/conferences", {}, headers, cfg)
    print(f"  conferences: {len(data)}", flush=True)
    return data


def pull_shooting_season(season: int, conferences: list[str], headers: dict, cfg: dict, cache_dir: Path) -> list:
    all_rows = []
    for conf in conferences:
        safe_conf = conf.replace("/", "-").replace(" ", "_")
        cache_path = cache_dir / f"shooting_{season}_{safe_conf}.json"
        data = _cached_get(
            cache_path, "/stats/player/shooting/season",
            {"season": season, "conference": conf}, headers, cfg,
        )
        all_rows.extend(data)
    print(f"  shooting/season {season}: {len(all_rows)} rows across {len(conferences)} conferences", flush=True)
    return all_rows


def pull_draft_picks(headers: dict, cfg: dict, cache_dir: Path) -> list:
    cache_path = cache_dir / "draft_picks_all.json"
    data = _cached_get(cache_path, "/draft/picks", {}, headers, cfg)
    print(f"  draft/picks (all history): {len(data)} rows", flush=True)
    return data


def main() -> None:
    cfg = load_config()
    headers = {"Authorization": f"Bearer {load_api_key()}"}
    cache_dir = REPO_ROOT / cfg["cbbd"]["raw_cache_dir"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    conferences = [c["abbreviation"] for c in pull_conferences(headers, cfg, cache_dir)]
    pull_draft_picks(headers, cfg, cache_dir)

    for season in cfg["cbbd"]["seasons"]:
        print(f"season {season}:", flush=True)
        pull_player_season(season, headers, cfg, cache_dir)
        pull_team_season(season, headers, cfg, cache_dir)
        pull_roster(season, headers, cfg, cache_dir)
        pull_shooting_season(season, conferences, headers, cfg, cache_dir)

    print("Done. All raw responses cached under", cache_dir)


if __name__ == "__main__":
    main()
