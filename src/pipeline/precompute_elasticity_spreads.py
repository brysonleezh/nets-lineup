"""
Precompute the league-wide role-elasticity spread distribution.

WHY: portal_shared.load_league_elasticity_spreads() loops every eligible player
and derives each one's usage-elasticity spread — ~54s for 433 players, to
produce 380 floats (3KB). Streamlit's cache makes that a once-per-container
cost, but a Community Cloud container is recycled often, so in practice real
visitors kept paying a minute-long spinner on the Player Breakdown page.

The values are a pure function of frozen inputs (the season's stint/event
tables and the fitted recipes), so there is no reason to derive them at
request time. This script computes them once; the portal reads the result and
falls back to computing if the file is absent, so a checkout that has not run
this still works — just slowly, exactly as before.

Run: python src/pipeline/precompute_elasticity_spreads.py
Output: data/league_elasticity_spreads_2025_26.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

import numpy as np  # noqa: E402

SEASON = "2025-26"
OUT = REPO_ROOT / "data" / f"league_elasticity_spreads_{SEASON.replace('-', '_')}.json"


def main() -> None:
    from portal_shared import load_static, compute_league_elasticity_spreads

    recipes, k, _labels, _oncourt = load_static()
    t0 = time.time()
    spreads, n = compute_league_elasticity_spreads(recipes, k, SEASON)
    dt = time.time() - t0

    payload = {
        "season": SEASON,
        "k": int(k),
        "n_players_with_elasticity": int(n),
        "n_players_considered": int(recipes["PLAYER_ID"].astype(int).nunique()),
        "spreads_pp": [float(x) for x in spreads],
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=1))
    print(f"computed {n} spreads in {dt:.1f}s -> {OUT} ({OUT.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
