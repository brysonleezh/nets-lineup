"""
Fit-model diagnostic harness - Phase 0: noise-ceiling measurement + volume
tables. Ground rule: measure the achievable out-of-sample correlation
ceiling BEFORE fitting anything, per the harness spec's Phase 0.

Lineup definition: a 5-man group, keyed by its sorted player-id tuple.
NET_RATING per lineup-season combines BOTH of its appearances in the
directed stint table - as the offense side (off_p1..5) contributing to
its own points-scored, and as the defense side (def_p1..5) contributing
to points-allowed - since a lineup's net rating is its scoring margin
across all the time it shares the floor, not just its own-offense rating.

Split-half reliability: for each lineup-season at a given possession
floor, its stints are randomly split into two halves (by possession-
count, not stint-count, so each half carries a comparable share of the
lineup's total sample), NET_RATING computed independently in each half,
Pearson r taken ACROSS lineups between half-A and half-B, then Spearman-
Brown corrected (r_full = 2r / (1+r)) to estimate the reliability of the
FULL (not half) sample - this bounds achievable out-of-sample correlation
at approximately sqrt(reliability).

Run: python3 src/pipeline/fit_harness/phase0_noise_ceiling.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from garbage_time import is_garbage_time

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "fit_harness_config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def load_stints(seasons: list[str]) -> pd.DataFrame:
    parts = []
    for s in seasons:
        path = REPO_ROOT / "data" / "stints" / f"stints_{s.replace('-', '_')}.parquet"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping season {s}")
            continue
        df = pd.read_parquet(path)
        df["season"] = s
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_lineup_stints(stints: pd.DataFrame) -> pd.DataFrame:
    """Two rows per directed stint row: one for the offense-side lineup
    (scoring `off_points` in `off_possessions`), one for the defense-side
    lineup (allowing `off_points` in `off_possessions`, i.e. its own
    defensive possessions faced). `lineup_key` is the sorted 5-tuple."""
    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]

    off_side = stints.copy()
    off_side["lineup_key"] = off_side[off_cols].apply(lambda r: tuple(sorted(r)), axis=1)
    off_side["points_for"] = off_side["off_points"]
    off_side["points_against"] = 0.0
    off_side["poss_off"] = off_side["off_possessions"]
    off_side["poss_def"] = 0.0

    def_side = stints.copy()
    def_side["lineup_key"] = def_side[def_cols].apply(lambda r: tuple(sorted(r)), axis=1)
    def_side["points_for"] = 0.0
    def_side["points_against"] = def_side["off_points"]
    def_side["poss_off"] = 0.0
    def_side["poss_def"] = def_side["off_possessions"]

    keep = ["season", "game_id", "stint_id", "lineup_key", "points_for", "points_against",
            "poss_off", "poss_def", "period", "stint_start_margin", "stint_start_clock"]
    return pd.concat([off_side[keep], def_side[keep]], ignore_index=True)


def _half_split_indices(n: int, poss: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Random possession-weighted half-split of n rows' own poss array.
    n=0 -> both empty. n=1 -> a coin flip decides which half gets it (NOT
    always A - an earlier version defaulted the odd/sub-2 case to A
    unconditionally, which is exactly the same class of directional bias
    already found and fixed for the off/def-side split itself, just at
    the single-row level - it showed up as a residual negative half_r
    that shrank but didn't vanish as the possession floor rose)."""
    if n == 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    if n == 1:
        return (np.array([0]), np.array([], dtype=int)) if rng.random() < 0.5 else \
               (np.array([], dtype=int), np.array([0]))
    perm = rng.permutation(n)
    cum = np.cumsum(poss[perm])
    if cum[-1] <= 0:
        return perm[:n // 2], perm[n // 2:]
    split_idx = np.searchsorted(cum, cum[-1] / 2.0)
    return perm[:split_idx + 1], perm[split_idx + 1:]


def _sb_correct(half_r: float) -> float:
    return 2 * half_r / (1 + half_r) if half_r is not None and half_r > -1 else np.nan


def split_half_reliability(lineup_stints: pd.DataFrame, min_poss: int, seed: int = 1) -> dict:
    """Reports OFFENSE-only, DEFENSE-only, and COMBINED (net rating) split-
    half reliability together, all from the same split draw.

    A first pass reporting only the combined metric gave a strongly
    NEGATIVE, seed-stable result at every possession bucket - implausible
    enough to investigate before trusting. Ruled out as a code bug three
    ways: (1) _half_split_indices independently verified to produce a
    disjoint, fully-covering partition over 2000 randomized trials: (2) run
    on pure single-rate i.i.d. synthetic noise (no off/def combination at
    all), it gives r=-0.02, correctly ~0; (3) confirmed on THIS repo's real
    data that OFFENSE-only and DEFENSE-only reliability are each small and
    POSITIVE (as expected), and only the COMBINED metric goes negative.

    The real mechanism: offense-side and defense-side rows are split
    independently, so each half ends up with a slightly different
    offense/defense POSSESSION-WEIGHT MIX purely by chance (e.g. half A
    randomly lands with a bit more of this lineup's own offensive
    possessions, half B a bit more of its defensive ones). Since a
    lineup's true offensive and defensive rates are rarely equal, that
    random mix-shift alone pulls the two halves' COMBINED net rating in
    opposite directions - a mechanical property of ratio-combining two
    independently-split components, not a sign either component is
    unreliable. This is direct empirical evidence for the harness spec's
    own Y-C design (model ORtg/DRtg separately rather than combined net
    rating) - not asserted, measured."""
    rng = np.random.default_rng(seed)
    grp = lineup_stints.groupby("lineup_key")
    total_poss = grp.apply(lambda g: g["poss_off"].sum() + g["poss_def"].sum(), include_groups=False)
    eligible = total_poss[total_poss >= min_poss].index

    off_a_l, off_b_l, def_a_l, def_b_l, comb_a_l, comb_b_l = [], [], [], [], [], []
    for key in eligible:
        g = lineup_stints[lineup_stints["lineup_key"] == key].reset_index(drop=True)
        off_rows = g.index[g["poss_off"] > 0].to_numpy()
        def_rows = g.index[g["poss_def"] > 0].to_numpy()

        oa, ob = _half_split_indices(len(off_rows), g.loc[off_rows, "poss_off"].values, rng)
        off_a, off_b = off_rows[oa], off_rows[ob]
        da, db = _half_split_indices(len(def_rows), g.loc[def_rows, "poss_def"].values, rng)
        def_a, def_b = def_rows[da], def_rows[db]

        def rate(rows, poss_col, pts_col):
            p = rows[poss_col].sum()
            return 100.0 * rows[pts_col].sum() / p if p > 0 else np.nan

        def net_rating(rows):
            poss_total = rows["poss_off"].sum() + rows["poss_def"].sum()
            margin = rows["points_for"].sum() - rows["points_against"].sum()
            return 100.0 * margin / poss_total if poss_total > 0 else np.nan

        oa_r, ob_r = rate(g.iloc[off_a], "poss_off", "points_for"), rate(g.iloc[off_b], "poss_off", "points_for")
        da_r, db_r = rate(g.iloc[def_a], "poss_def", "points_against"), rate(g.iloc[def_b], "poss_def", "points_against")
        idx_a, idx_b = np.concatenate([off_a, def_a]), np.concatenate([off_b, def_b])
        ca, cb = net_rating(g.iloc[idx_a]), net_rating(g.iloc[idx_b])

        if any(np.isnan(v) for v in (oa_r, ob_r, da_r, db_r, ca, cb)):
            continue
        off_a_l.append(oa_r); off_b_l.append(ob_r)
        def_a_l.append(da_r); def_b_l.append(db_r)
        comb_a_l.append(ca); comb_b_l.append(cb)

    n_lineups = len(comb_a_l)
    if n_lineups < 3:
        nan_row = {"half_r": np.nan, "full_r_sb": np.nan}
        return {"min_poss": min_poss, "n_lineups": n_lineups,
                "offense": nan_row, "defense": nan_row, "combined": nan_row}

    def summarize(a, b):
        r = float(np.corrcoef(a, b)[0, 1])
        return {"half_r": r, "full_r_sb": _sb_correct(r)}

    return {
        "min_poss": min_poss, "n_lineups": n_lineups,
        "offense": summarize(off_a_l, off_b_l),
        "defense": summarize(def_a_l, def_b_l),
        "combined": summarize(comb_a_l, comb_b_l),
    }


def volume_table(lineup_stints: pd.DataFrame, buckets: list[int]) -> pd.DataFrame:
    grp = lineup_stints.groupby("lineup_key")
    total_poss = grp.apply(lambda g: g["poss_off"].sum() + g["poss_def"].sum(), include_groups=False)
    league_total = total_poss.sum()
    rows = []
    for b in buckets:
        elig = total_poss[total_poss >= b]
        rows.append({
            "min_poss": b, "n_lineups": len(elig),
            "poss_covered": elig.sum(), "share_of_league_poss": elig.sum() / league_total,
        })
    return pd.DataFrame(rows)


def main() -> None:
    cfg = load_config()
    seasons = cfg["seasons"]["train"] + [cfg["seasons"]["holdout"]]
    print(f"Loading stints for seasons: {seasons}")
    stints = load_stints(seasons)
    print(f"Raw directed stint rows: {len(stints)}")

    stints["is_garbage"] = is_garbage_time(stints["period"], stints["stint_start_margin"], stints["stint_start_clock"])
    n_garbage = int(stints["is_garbage"].sum())
    print(f"Garbage-time rows: {n_garbage} ({100 * n_garbage / len(stints):.2f}%)")
    clean = stints[~stints["is_garbage"]].copy()

    total_poss_by_season = clean.groupby("season")["off_possessions"].sum()
    print("\nTotal (non-garbage) possessions by season:")
    print(total_poss_by_season.to_string())
    print(f"Total training (non-holdout) possessions: "
          f"{clean[clean['season'].isin(cfg['seasons']['train'])]['off_possessions'].sum():,.0f}")

    lineup_stints = build_lineup_stints(clean)

    buckets = cfg["phase0"]["possession_buckets"]
    vol = volume_table(lineup_stints, buckets)
    print("\nVolume table:")
    print(vol.to_string(index=False))

    print("\nSplit-half reliability (offense / defense / combined net rating):")
    raw_rel = [split_half_reliability(lineup_stints, b) for b in buckets]
    rel = pd.DataFrame([
        {
            "min_poss": r["min_poss"], "n_lineups": r["n_lineups"],
            "off_half_r": r["offense"]["half_r"], "off_full_r_sb": r["offense"]["full_r_sb"],
            "def_half_r": r["defense"]["half_r"], "def_full_r_sb": r["defense"]["full_r_sb"],
            "combined_half_r": r["combined"]["half_r"], "combined_full_r_sb": r["combined"]["full_r_sb"],
        }
        for r in raw_rel
    ])
    print(rel.to_string(index=False))

    out_dir = REPO_ROOT / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    vol.to_csv(REPO_ROOT / "data" / "fit_harness_volume_table.csv", index=False)
    rel.to_csv(REPO_ROOT / "data" / "fit_harness_split_half_reliability.csv", index=False)

    report = [
        "# Phase 0 — Noise Ceiling\n",
        f"Seasons: {seasons}\n",
        f"Raw directed stint rows: {len(stints)}\n",
        f"Garbage-time rows excluded: {n_garbage} ({100 * n_garbage / len(stints):.2f}%)\n",
        "\n## Possessions by season (non-garbage)\n",
        total_poss_by_season.to_frame("possessions").to_markdown(),
        "\n\n## Volume table\n",
        vol.to_markdown(index=False),
        "\n\n## Split-half reliability (Spearman-Brown corrected)\n",
        rel.to_markdown(index=False),
        "\n",
    ]
    (out_dir / "phase0_noise_ceiling.md").write_text("\n".join(report))
    print(f"\nWrote {out_dir / 'phase0_noise_ceiling.md'}")


if __name__ == "__main__":
    main()
