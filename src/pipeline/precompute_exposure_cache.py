"""
Precompute load_exposure_cache()'s ~400-query result (portal_shared.py) to
disk, so a deployed copy of the portal never pays this cost live - not even
once. Same "precompute offline, load a file at runtime" convention already
used for the Intro page's K-slider bases (precompute_hull_bases.py).

AI-ASSISTED (Claude Code, chat)
Prompt: "如果我后续是要部署怎么解决这个问题" (if I'm going to deploy this
later, how do I solve the slow-first-load problem) - a follow-up to timing
load_exposure_cache() directly at 36.8s of a 38.7s cold-start critical path.
st.cache_data(persist="disk") (added to load_exposure_cache() in the same
session) already helps a long-running dev server survive restarts, but
isn't deployment-robust on its own: it only helps if the SAME filesystem
sticks around between runs, which a fresh container/redeploy won't
guarantee. Precomputing to a file that ships as part of the repo/deploy
artifact (like data/basis_2025_26/recipes.csv already does) removes the
live query cost entirely, regardless of hosting details.
Used: reuses compute_all_player_exposures() verbatim - no new computation,
just running it once offline and saving (pids, exposures) to an .npz file
in the same directory/naming convention as the existing basis.npz/
recipes.npy. load_exposure_cache() (portal_shared.py) was updated to load
this file directly when present, falling back to the live ~400-query path
(still cached via persist="disk") when it's missing - so nothing breaks
for a fresh clone that hasn't run this script yet, it's just slower once.
Not AI: the season/basis this precomputes for (2025-26, the app's one
production season) and the decision to keep the live-computation fallback
rather than making the file a hard requirement - both this project's own
existing "guard + actionable message, never a silent crash" convention.

Run: python3 src/pipeline/precompute_exposure_cache.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step1_archetype_model import DATA_DIR
from step2_diagnostic_analysis import compute_all_player_exposures

BASIS_DIR = DATA_DIR / "basis_2025_26"
SEASON = "2025-26"


def main():
    recipes = pd.read_csv(BASIS_DIR / "recipes.csv")
    k = sum(1 for c in recipes.columns if c.startswith("arch_"))

    print(f"computing exposures for every {SEASON} league player (k={k}) - "
          f"~400+ queries, this is the one-time cost being precomputed away...")
    pids, exposures = compute_all_player_exposures(recipes, k, season=SEASON)

    out_path = BASIS_DIR / f"exposure_cache_{SEASON}.npz"
    np.savez(out_path, pids=np.array(pids, dtype=np.int64), exposures=exposures, k=k)
    print(f"saved {len(pids)} players' exposures -> {out_path}")


if __name__ == "__main__":
    main()
