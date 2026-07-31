"""
Precompute ADA bases at K=4..10 for the Intro page's convex-hull scatter
(portal.py). Fitting ADA live inside the app is far too slow for an
interactive K slider - this script fits each K once, offline, and saves it
to disk in the exact format step1_archetypes_model's save_basis/load_basis
already use, so the portal can load any K's basis instantly at render time.

K=8 is NOT refit here - the app's authoritative K=8 basis already exists at
data/basis_2025_26 (the real production fit this whole project's write-up
and every other portal page reference) and is reused as-is by the portal,
so the K slider's K=8 position is guaranteed to match the rest of the app
rather than risk a second, possibly-different local optimum for the same K.

AI-ASSISTED (Claude Code, chat)
Prompt: build a K=4..10 slider for a convex-hull scatter on the Intro page,
with the explicit instruction "Refitting ADA live is far too slow, so
precompute bases at each K offline and cache them to disk first — write a
separate script for that, do not fit inside the app."
Used: reuses step1_archetypes_model's own load_population / fit_basis_best_of
/ save_basis / match_archetypes_to_players verbatim - no new fitting logic,
just looping the existing production fit function over K and writing to a
new hull_bases/ directory tree via the existing save_basis format.
Not AI: the K range (4-10), and the decision not to refit K=8 given an
authoritative fit already exists on disk - both the user's/project's own
prior choices, not new decisions made here.

Run: python3 src/precompute_hull_bases.py
"""

from __future__ import annotations

from step1_archetypes_model import (
    DATA_DIR,
    fit_basis_best_of,
    load_population,
    match_archetypes_to_players,
    save_basis,
)

HULL_BASES_DIR = DATA_DIR / "hull_bases"
CANDIDATE_KS = (4, 5, 6, 7, 9, 10)  # 8 already exists at data/basis_2025_26


def main():
    df = load_population(min_threshold=300, season_min="2025-26", season_max="2025-26")
    for k in CANDIDATE_KS:
        out_dir = HULL_BASES_DIR / f"k{k}"
        print(f"\n=== K={k} ===", flush=True)
        fit = fit_basis_best_of(df, k=k, seeds=(0, 1, 2), max_iter=200)
        assert fit["converged"], f"K={k} did not converge - raise max_iter before trusting this basis"
        save_basis(fit, out_dir)
        defs = match_archetypes_to_players(df, fit)
        defs.to_csv(out_dir / "archetype_definitions.csv", index=False)
        print(f"K={k} done -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
