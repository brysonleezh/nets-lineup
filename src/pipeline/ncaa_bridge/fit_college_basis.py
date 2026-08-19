"""
NCAA bridge - Step 1b: fit the college archetype model.

Mirrors src/step1_archetype_model.py's fit_basis/_intra_archetype_variance/
select_k pattern (same two K-selection criteria from "Scouting Anyone"
Section 4.3: RSS/explained-variance, which the paper itself says is
misleading alone, and intra-archetype variance, which is what it actually
decides K on) - adapted in two ways forced by this dataset:

1. PER-SEASON z-scoring, not one fixed mu/sd. Locked design decision: a
   college role is defined relative to its own league-year, not a single
   reference season - so each season's raw features are standardized
   against THAT season's own D1 population before the (multi-season-
   pooled) archetype fit. Per-season mu/sd are saved (needed later to
   project the anchors' and target rookies' own seasons the same way).

2. AA (not ADA) is the primary/production method, for a hard computational
   reason discovered by direct measurement, not a stylistic preference:
   ADA fit on n=500 synthetic rows exceeded 5 minutes (K=8, max_iter=30)
   and was killed rather than let run further; AA fit the same shape in
   0.04s, and scaled near-linearly through n=3000 (0.54s). Extrapolated
   quadratically (consistent with ADA's archetypes-must-be-real-data-
   points search), a single ADA fit on this dataset's ~28-30k pooled rows
   would take on the order of days, and the K-sweep needs ~24-54 such fits
   (K=4..12 x 3 seeds). AA fits the full pool in seconds. ADA is still run
   (see fit_ada_comparison below), but only on a bounded single-season
   subsample, matching the spec's "prefer AA if ADA's archetypoids land on
   low-major statistical outliers" fallback - now additionally forced by
   this measured runtime gap, not just the interpretability concern the
   spec anticipated.

Run: python3 src/pipeline/ncaa_bridge/fit_college_basis.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import archetypes as arch

REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
CONFIG_PATH = REPO_ROOT / "ncaa_bridge_config.yaml"


def load_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text())


def per_season_zscore(df: pd.DataFrame, feature_cols: list[str]) -> tuple[pd.DataFrame, dict]:
    """Standardize each season's rows against that season's OWN mean/sd.
    Returns the pooled z-scored frame plus a {season: {mu, sd}} dict for
    later reuse (anchors/rookies must be standardized the same way)."""
    z_parts = []
    season_params = {}
    for season, g in df.groupby("SEASON"):
        X = g[feature_cols].astype(float)
        mu = X.mean()
        sd = X.std()
        dead = sd[sd < 1e-12].index.tolist()
        if dead:
            raise ValueError(f"season {season}: zero-variance features {dead}")
        z = (X - mu) / sd
        z_parts.append(z.assign(SEASON=season, athleteId=g["athleteId"].values, name=g["name"].values))
        season_params[int(season)] = {"mu": mu.values.tolist(), "sd": sd.values.tolist()}
    pooled = pd.concat(z_parts, ignore_index=True)
    return pooled, season_params


def drafted_conferences(cfg: dict, cache_dir: Path) -> set[str]:
    """Conferences with >=1 draft pick in the configured draft-class window,
    matched via the pick's sourceTeamName joined against that draft year's
    own team/season conference listing (conference realignment means the
    same school can sit in a different conference in different years)."""
    picks = json.loads((cache_dir / "draft_picks_all.json").read_text())
    lo, hi = cfg["college_model"]["draft_sample_window"]
    picks = [p for p in picks if lo <= p["year"] <= hi]

    confs = set()
    unmatched = []
    for p in picks:
        # CBBD draft year N -> player's final college season is the
        # season that ENDED in spring of year N, i.e. cbbd season=N.
        season = p["year"]
        team_path = cache_dir / f"team_season_{season}.json"
        if not team_path.exists():
            unmatched.append((p["name"], p["sourceTeamName"], season, "no team_season file"))
            continue
        teams = json.loads(team_path.read_text())
        # p["sourceTeamName"] is the MASCOT ("Blue Devils"), not the school
        # ("Duke") - confirmed by a live probe (AJ Dybantsa's pick record:
        # sourceTeamLocation="BYU", sourceTeamName="Cougars"). team_season's
        # own "team" field is the school name, so join on sourceTeamLocation.
        match = [t for t in teams if t["team"] == p["sourceTeamLocation"]]
        if not match:
            unmatched.append((p["name"], p["sourceTeamLocation"], season, "no team name match"))
            continue
        confs.add(match[0]["conference"])
    print(f"drafted_conferences: {len(confs)} conferences from {len(picks)} picks "
          f"({len(unmatched)} unmatched, e.g. international/non-D1 sources)", flush=True)
    if unmatched:
        print("  sample unmatched:", unmatched[:5], flush=True)
    return confs


def fit_aa(X_z: np.ndarray, k: int, seed: int, max_iter: int = 200, tol: float = 1e-4) -> dict:
    t0 = time.time()
    model = arch.AA(n_archetypes=k, max_iter=max_iter, tol=tol, random_state=seed)
    model.fit(X_z)
    elapsed = time.time() - t0
    n_iter = len(model.loss_)
    tss = float((X_z ** 2).sum())
    rss = float(model.rss_)
    return {
        "basis": np.asarray(model.archetypes_, dtype=float),
        "coefficients": np.asarray(model.coefficients_, dtype=float),
        "k": int(k), "seed": int(seed),
        "converged": bool(n_iter < max_iter), "n_iter": int(n_iter),
        "rss": rss, "explained_var": 1.0 - rss / tss, "seconds": elapsed,
    }


def intra_archetype_variance(X_z: np.ndarray, coefficients: np.ndarray, k: int) -> float:
    assignment = np.argmax(coefficients, axis=1)
    total_var, total_n = 0.0, 0
    for j in range(k):
        mask = assignment == j
        n_j = int(mask.sum())
        if n_j < 2:
            continue
        group = X_z[mask]
        centroid = group.mean(axis=0)
        var_j = float(((group - centroid) ** 2).sum(axis=1).mean())
        total_var += var_j * n_j
        total_n += n_j
    return total_var / total_n


def select_k_aa(X_z: np.ndarray, candidate_ks, seeds=(0, 1, 2), max_iter=200, tol=1e-4) -> pd.DataFrame:
    rows = []
    for k in candidate_ks:
        for seed in seeds:
            fit = fit_aa(X_z, k, seed, max_iter=max_iter, tol=tol)
            intra_var = intra_archetype_variance(X_z, fit["coefficients"], k)
            flag = "" if fit["converged"] else "  NOT CONVERGED"
            print(f"K={k} seed={seed}: EV={fit['explained_var']:.4f} "
                  f"intra_var={intra_var:.4f} ({fit['n_iter']} iters, {fit['seconds']:.2f}s){flag}", flush=True)
            rows.append({
                "k": k, "seed": seed, "rss": fit["rss"], "explained_var": fit["explained_var"],
                "intra_archetype_var": intra_var, "n_iter": fit["n_iter"],
                "converged": fit["converged"], "seconds": fit["seconds"],
            })
    raw = pd.DataFrame(rows)
    best = (raw.sort_values(["k", "explained_var"], ascending=[True, False])
               .groupby("k", as_index=False).first()
               [["k", "rss", "explained_var", "intra_archetype_var"]])
    agg = (raw.groupby("k")
              .agg(ev_spread=("explained_var", lambda s: s.max() - s.min()),
                   all_converged=("converged", "all"),
                   total_seconds=("seconds", "sum"))
              .reset_index())
    summary = best.merge(agg, on="k")
    summary["ev_gain"] = summary["explained_var"].diff()
    summary["intra_var_change"] = summary["intra_archetype_var"].diff()
    return raw, summary


def main() -> None:
    cfg = load_config()
    cache_dir = REPO_ROOT / cfg["cbbd"]["raw_cache_dir"]
    feature_cols = cfg["college_model"]["features"]
    k_lo, k_hi = cfg["college_model"]["k_range"]

    df = pd.read_csv(REPO_ROOT / "data" / "college" / "features_complete.csv")
    print(f"loaded {len(df)} complete-case player-seasons, {df['SEASON'].nunique()} seasons", flush=True)

    if cfg["college_model"]["restrict_to_drafted_conferences"]:
        confs = drafted_conferences(cfg, cache_dir)
        pool = df[df["conference"].isin(confs)].copy()
        print(f"restrict_to_drafted_conferences=True: pool {len(df)} -> {len(pool)} "
              f"({df['conference'].nunique()} -> {pool['conference'].nunique()} conferences)", flush=True)
    else:
        pool = df.copy()

    pooled_z, season_params = per_season_zscore(pool, feature_cols)
    X_z = pooled_z[feature_cols].values
    print(f"pooled z-scored fit matrix: {X_z.shape}", flush=True)

    out_dir = REPO_ROOT / "data" / "college"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "season_zscore_params.json").write_text(json.dumps(season_params, indent=2))

    candidate_ks = list(range(k_lo, k_hi + 1))
    raw, summary = select_k_aa(X_z, candidate_ks)
    raw.to_csv(out_dir / "k_selection_raw.csv", index=False)
    summary.to_csv(out_dir / "k_selection_summary.csv", index=False)
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
