"""
Step 1 - Player archetype recipes via Archetypoid Analysis (ADA).

The output of this file is three arrays: basis (k, p), mu (p,), sd (p,).
With those on disk, projecting any player onto the archetypes is a pure
function - no refit, no lookups. Everything downstream (Step 2 synergy
matrix, Step 3 lineup scoring, Step 4 slot query) consumes only the
resulting k-dim recipe vectors.

Part A  population loading + train/test window
Part B  fit the basis
Part C  project onto a fixed basis
Part D  verify the projection
Part E  save / load
Part F  K selection, criterion 1: explained variance
Part G  K selection, criterion 2: synergy-cell exposure
Part H  diagnostics
Part I  archetype identity and feature profile
Part J  phase runners
"""

from __future__ import annotations

import json
import sqlite3
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import archetypes as arch
from scipy.optimize import nnls

from step0_data import build_nba_side_tables, FEATURE_COLUMNS, DB_PATH, _normalize_name

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
ID_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "SEASON", "TEAM_ABBREVIATION", "MIN"]


"""
AI-ASSISTED (Claude Code, chat)
Prompt: phase2_build at MIN>=300 with 3 seeds is the longest run yet - asked
how to track status mid-run given the earlier archetypes_model.py episode
(no `verbose` output for a long time, unclear if it was hung or just slow).
Used: a timestamp-prefixed print so a log file is self-documenting hours
later (relative "elapsed=Ns" doesn't tell you what wall-clock time that
was); combined with running as `python3 -u ... > logfile 2>&1 &` so the
file actually updates in real time instead of sitting in a stdout buffer.
Not AI: not writing a heavier logging framework for a single long overnight
run - a timestamped print is enough for this.
"""
def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# AI-ASSISTED (Claude Code, chat) - relocated here during the src/
# restructuring. _parse_pair_ids/load_pairs originally lived in
# step2_synergy_matrix.py (CLAUDE.md's abandoned synergy-matrix attempt,
# now offline in src/pipeline/), but step2_diagnostic_analysis.py's live
# load_player_pairs() imports _parse_pair_ids directly, and this file's
# own K-exposure diagnostic (Part G) uses load_pairs - neither should
# depend on the abandoned/offline synergy-matrix file, so both functions
# moved to this always-loaded module instead. Bodies unchanged.
def _parse_pair_ids(group_id: str) -> tuple:
    """GROUP_ID looks like '-203471-1627752-': two player IDs joined by
    dashes, empty strings at both ends from the leading/trailing dash."""
    parts = [p for p in group_id.split("-") if p]
    if len(parts) != 2:
        raise ValueError(f"expected 2 player IDs in a 2-man GROUP_ID, got {parts}: {group_id!r}")
    return int(parts[0]), int(parts[1])


def load_pairs(min_threshold=100, db_path=None):
    """Load 2-man lineup combos with MIN >= min_threshold.

    One row per (player pair, season, team). MIN here is shared minutes -
    becomes the regression weight. NET_RATING is the regression target.
    """
    db_path = db_path or DB_PATH
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql(
            "SELECT GROUP_ID, TEAM_ID, TEAM_ABBREVIATION, SEASON, MIN, NET_RATING, POSS "
            "FROM lineups WHERE GROUP_QUANTITY = 2 AND MIN >= ?",
            conn, params=(min_threshold,),
        )
    ids = df["GROUP_ID"].apply(_parse_pair_ids)
    df["PLAYER_A_ID"] = [t[0] for t in ids]
    df["PLAYER_B_ID"] = [t[1] for t in ids]

    print(f"pairs: {len(df)} rows | MIN>={min_threshold} | "
          f"seasons {df['SEASON'].min()}..{df['SEASON'].max()}")
    return df


# --- Part A: population loading ---------------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: I need to fit the basis on seasons before 2025-26 so 2025-26 can be
a holdout for the Step 3a validation, but apply the basis to all
seasons. Asked where the season filter belongs so the training population is
never implicit.
Used: the argument that filtering belongs at the call site, not inside
fit_basis, so every fit records exactly which rows it saw.
Not AI: the 2025-26 holdout design, the MIN>=300 / MIN>=1000 thresholds.
"""


def load_population(min_threshold=300, season_max=None, season_min=None):
    """Load player-season-team rows with filters applied.

    season_max is the train/test switch: pass "2024-25" for the validation
    basis. SEASON strings sort correctly since the prefix is the start year.
    """
    df = build_nba_side_tables()
    df = df[df["MIN"] >= min_threshold]
    if season_min is not None:
        df = df[df["SEASON"] >= season_min]
    if season_max is not None:
        df = df[df["SEASON"] <= season_max]
    df = df.reset_index(drop=True)

    print(f"population: {len(df)} rows | MIN>={min_threshold} | "
          f"seasons {df['SEASON'].min()}..{df['SEASON'].max()}")
    return df


# --- Part B: fit the basis --------------------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: My previous version z-scored inside the fit and discarded the
mean/std. Asked what a fit function has to return so that a later projection
of held-out players cannot leak test information through standardization,
and how to tell whether an ADA fit actually converged rather than hitting
max_iter.
Used: (1) mu/sd must be returned with the basis and reused verbatim
downstream; (2) converged = len(model.loss_) < max_iter, since the package
breaks early on tol; (3) report explained variance rather than raw RSS so
the number is comparable across populations.
Not AI: max_iter=200, tol=1e-4, the winsorize escape hatch and when to use
it, the choice of ADA over AA.
"""


def fit_basis(df, k, seed=0, max_iter=200, tol=1e-4, winsorize=None, verbose=False):
    """Fit k archetypoids on df. df must already be filtered.

    winsorize: optional float, e.g. 0.01 clips each feature to its [1%, 99%]
    range first. Archetypoids sit on the convex hull, so one extreme row can
    become an "archetype" that is really an outlier. Use if Part I turns up a
    nonsense archetype.
    """
    X = df[FEATURE_COLUMNS].astype(float)

    if winsorize is not None:
        X = X.clip(lower=X.quantile(winsorize),
                   upper=X.quantile(1 - winsorize), axis=1)

    mu = X.mean()
    sd = X.std()

    dead = sd[sd < 1e-12].index.tolist()
    if dead:
        raise ValueError(f"zero-variance features, z-scoring would divide by 0: {dead}")

    X_z = ((X - mu) / sd).values

    t0 = time.time()
    model = arch.ADA(n_archetypes=k, max_iter=max_iter, tol=tol,
                     random_state=seed, verbose=verbose)
    model.fit(X_z)
    elapsed = time.time() - t0

    n_iter = len(model.loss_)
    converged = n_iter < max_iter
    tss = float((X_z ** 2).sum())   # already centered, so this is the TSS
    rss = float(model.rss_)

    fit = {
        "basis": np.asarray(model.archetypes_, dtype=float),
        "mu": mu.values,
        "sd": sd.values,
        "feature_columns": list(FEATURE_COLUMNS),
        "k": int(k),
        "seed": int(seed),
        "converged": bool(converged),
        "n_iter": int(n_iter),
        "max_iter": int(max_iter),
        "rss": rss,
        "explained_var": 1.0 - rss / tss,
        "n_train": int(len(df)),
        "season_range": [str(df["SEASON"].min()), str(df["SEASON"].max())],
        "min_threshold": float(df["MIN"].min()),
        "winsorize": winsorize,
        "seconds": elapsed,
        # kept for Part D only, not written to disk
        "_coefficients": np.asarray(model.coefficients_, dtype=float),
    }

    flag = "" if converged else "   NOT CONVERGED - hit max_iter"
    print(f"K={k} seed={seed}: EV={fit['explained_var']:.4f} "
          f"({n_iter} iters, {elapsed:.0f}s){flag}")
    return fit


"""
AI-ASSISTED (Claude, chat)
Prompt: ADA is initialization-sensitive. Asked whether a single seed is
enough for the K sweep and what the across-seed spread tells me.
Used: multi-restart best-of is the standard AA/ADA practice, and the spread
across seeds is itself a K-too-large signal - if the solution stops being
reproducible at some K, that caps K independently of the RSS curve.
Not AI: using seeds (0,1,2), and treating spread as a reporting column
rather than an automatic cutoff.
"""


def fit_basis_best_of(df, k, seeds=(0, 1, 2), **kw):
    """Fit several restarts, keep the best by explained variance."""
    kw.setdefault("verbose", True)  # else model.fit() is silent until the whole
                                     # seed converges - the ADA package's own
                                     # every-10th-iteration RSS print never fires
    _log(f"fit_basis_best_of: starting {len(seeds)} seed(s) {seeds} at k={k}, n={len(df)}")
    fits = []
    for i, s in enumerate(seeds):
        _log(f"  seed {s} ({i + 1}/{len(seeds)}) starting...")
        fits.append(fit_basis(df, k, seed=s, **kw))
        _log(f"  seed {s} done")
    evs = [f["explained_var"] for f in fits]
    best = fits[int(np.argmax(evs))]
    best["ev_spread"] = float(max(evs) - min(evs))
    best["all_seeds_converged"] = all(f["converged"] for f in fits)
    _log(f"best seed={best['seed']}, EV spread={best['ev_spread']:.4f}")
    return best


# --- Part C: project onto a fixed basis -------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: I need to score players who were NOT in the training population
(the MIN 300-999 band, and later 2025-26) against an already-fitted basis.
Asked how to write the simplex-constrained least squares directly so I can
inspect it, rather than calling the package's nnls_transform whose max_iter
argument I could not pin down from the source.
Used: augment the design matrix with one row of a large constant to turn
sum(p)=1 into a heavily weighted soft constraint, then solve with
scipy.optimize.nnls per row.
Not AI: simplex_weight=1e4 (chosen so the constraint residual is negligible
against the z-scored feature scale, verified against Part D), and the
decision to hand-roll this at all rather than trust the package.
"""


def project(df, basis, mu, sd, feature_columns, simplex_weight=1e4):
    """Project players onto a fixed basis. Returns (n, k) recipes.

    Solves per row:  min ||x - B.T @ p||^2  s.t.  p >= 0, sum(p) = 1

    mu and sd are arguments and are never recomputed from df. Recomputing
    them is how test-set information leaks in through standardization.
    """
    B = np.asarray(basis, dtype=float)      # (k, p)
    k = B.shape[0]

    X_z = ((df[list(feature_columns)].astype(float) - mu) / sd).values

    A = np.vstack([B.T, np.full((1, k), simplex_weight)])    # (p+1, k)

    P = np.empty((len(X_z), k))
    for i, x in enumerate(X_z):
        P[i], _ = nnls(A, np.concatenate([x, [simplex_weight]]))

    row_sums = P.sum(axis=1, keepdims=True)
    if (row_sums < 1e-9).any():
        raise RuntimeError("degenerate projection: some rows got all-zero weights")
    return P / row_sums


def recipes_frame(df, P, k):
    """Attach a projection matrix to identity columns."""
    out = df[[c for c in ID_COLUMNS if c in df.columns]].copy()
    for i in range(k):
        out[f"arch_{i}"] = P[:, i]
    return out


# --- Part D: verify the projection ------------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: Asked how to check my hand-written projection is correct, given I
deliberately did not use the package's own transform.
Used: on the training set, project() must reproduce model.coefficients_ to
numerical tolerance, since both solve the same constrained problem against
the same basis. Fails loudly rather than warning.
Not AI: atol=1e-3, and treating this as a hard gate before any other work.
"""


def verify_projection(df, fit, atol=1e-3):
    """project() must reproduce ADA's own coefficients on the training set."""
    P = project(df, fit["basis"], fit["mu"], fit["sd"], fit["feature_columns"])
    diff = np.abs(P - fit["_coefficients"])
    print(f"projection check: max diff={diff.max():.6f}, mean={diff.mean():.6f}")
    assert diff.max() < atol, (
        f"project() disagrees with ADA coefficients (max {diff.max():.4f}). "
        "Stop here - the projection is wrong and everything downstream inherits it."
    )
    print("  projection verified")
    return float(diff.max())


# --- Part E: save / load ----------------------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: My earlier version saved archetype identities to CSV and rebuilt the
basis later by looking players up on (PLAYER_NAME, SEASON). Asked whether
that key is safe at player-season-team granularity.
Used: it is not - a midseason-traded player produces two rows with identical
name and season, so the lookup silently picks one. Randle is exactly such a
case and is a core subject of Steps 3-4. Storing the matrix removes the
failure mode rather than guarding against it.
Not AI: npz over pickle, and splitting train/full into separate directories
instead of versioned filenames.
"""


def save_basis(fit, out_dir):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "basis.npz", basis=fit["basis"], mu=fit["mu"], sd=fit["sd"])
    meta = {key: val for key, val in fit.items()
            if not key.startswith("_") and key not in ("basis", "mu", "sd")}
    (out_dir / "basis_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved basis -> {out_dir}")


def load_basis(out_dir):
    out_dir = Path(out_dir)
    z = np.load(out_dir / "basis.npz")
    meta = json.loads((out_dir / "basis_meta.json").read_text())
    return {"basis": z["basis"], "mu": z["mu"], "sd": z["sd"], **meta}


# --- Part F: K selection, criterion 1 ---------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: My first select_k swept K on MIN>=1500 with one seed and
max_iter=50, while production fits MIN>=1000. Asked whether any of those
three choices bias the elbow.
Used: all three do. (1) A higher minute floor is a more homogeneous
population - starters only - which genuinely needs fewer archetypes, so
sweeping there biases toward small K. (2) ADA converges more slowly at
larger K, so a low max_iter inflates RSS at high K and shifts the apparent
elbow left. (3) Single-seed RSS moves the elbow on its own. Also: a step of
2 in the candidate grid cannot distinguish K=8 from K=9.
Not AI: the candidate range 5-12, and the decision to report explained
variance and its first difference rather than eyeballing a plot.

AI-ASSISTED (Claude Code, chat), later addition:
Prompt: "step1_archetypes_model没有用elbow method 我觉得我也应该在代码画一张这样
的图 像论文里这样" (pasted the paper's Figure 2), then, after re-reading the
paper's actual Section 4.3: "这里对K的数目进行了更详细的研究" - re-read 4.3
directly (pypdf) and found the RSS elbow alone is explicitly NOT what the
paper picks K on - the paper states the RSS elbow lands early (~4) and
calls that "clearly insufficient," introducing a second, separate metric
(intra-archetype variance, Figures 3/4) as the actual deciding criterion.
select_k only had the RSS/explained-variance side; added
_intra_archetype_variance() and threaded it through the same per-(k,seed)
loop (reuses the fit that was already computed - no extra fitting) so
both of the paper's own two criteria are computed together.
Not AI: which of the two metrics is authoritative (the paper is explicit:
RSS alone is misleading) - not a judgment call I introduced.
"""


def _intra_archetype_variance(df, fit):
    """Paper Section 4.3's second (deciding) K-selection criterion: hard-
    assign each player-season to its highest-membership archetype (argmax
    of the training-set alpha weights ADA already computed -
    fit["_coefficients"], no re-projection needed), then average each
    archetype group's own within-group variance (mean squared deviation
    from the GROUP's own centroid, in standardized feature space) -
    "how similar the players assigned to each archetype are to one
    another," per the paper's own description, not distance to the
    archetype/archetypoid vertex itself. Weighted by group size so one
    tiny group can't dominate the average.
    """
    X = df[fit["feature_columns"]].astype(float).values
    X_z = (X - fit["mu"]) / fit["sd"]
    assignment = np.argmax(fit["_coefficients"], axis=1)

    total_var, total_n = 0.0, 0
    for j in range(fit["k"]):
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


def select_k(df, candidate_ks=(5, 6, 7, 8, 9, 10, 11, 12), seeds=(0, 1, 2),
             max_iter=200, tol=1e-4, out_name="k_selection"):
    """Sweep over K with restarts, tracking both of the paper's Section 4.3
    criteria: RSS/explained variance (criterion 1 - the paper shows this
    alone is misleading) and intra-archetype variance (criterion 2 - what
    the paper actually decides K on)."""
    rows = []
    for k in candidate_ks:
        for seed in seeds:
            fit = fit_basis(df, k, seed=seed, max_iter=max_iter, tol=tol)
            intra_var = _intra_archetype_variance(df, fit)
            rows.append({
                "k": k, "seed": seed, "rss": fit["rss"],
                "explained_var": fit["explained_var"],
                "intra_archetype_var": intra_var,
                "n_iter": fit["n_iter"], "converged": fit["converged"],
                "seconds": fit["seconds"],
            })

    raw = pd.DataFrame(rows)

    # best seed per k (by explained variance) carries both metrics forward
    # together, rather than independently minimizing each column across
    # seeds - keeps the reported RSS and intra-var numbers from the same fit.
    best = (raw.sort_values(["k", "explained_var"], ascending=[True, False])
               .groupby("k", as_index=False).first()
               [["k", "rss", "explained_var", "intra_archetype_var"]])
    agg = (raw.groupby("k")
              .agg(ev_spread=("explained_var", lambda s: s.max() - s.min()),
                   all_converged=("converged", "all"),
                   max_n_iter=("n_iter", "max"),
                   total_seconds=("seconds", "sum"))
              .reset_index())
    summary = best.merge(agg, on="k")
    summary["ev_gain"] = summary["explained_var"].diff()
    summary["intra_var_change"] = summary["intra_archetype_var"].diff()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw.to_csv(DATA_DIR / f"{out_name}_raw.csv", index=False)
    summary.to_csv(DATA_DIR / f"{out_name}.csv", index=False)
    print("\n" + summary.to_string(index=False))

    if not summary["all_converged"].all():
        bad = summary.loc[~summary["all_converged"], "k"].tolist()
        print(f"\nK={bad} did not converge. Raise max_iter and re-run those "
              f"before reading the elbow - unconverged fits inflate RSS at "
              f"high K and pull the elbow left.")

    print("\nPer the paper (Section 4.3): the RSS elbow alone tends to land "
          "early and understate how many archetypes are needed - read "
          "intra_archetype_var instead (where it stops decreasing "
          "meaningfully, checking intra_var_change), not ev_gain. Then check "
          "ev_spread: a jump means the solution is unstable at that K, which "
          "caps K regardless of fit. Final K must also clear Part G.")
    return raw, summary


def plot_rss_scree(summary, out_path, title="Elbow Method for Archetypoid Analysis, NBA Data"):
    """Mirrors the paper's Figure 2 (Scree Plot on RSS). Plotted mainly to
    make the paper's own point visible in our data: this curve alone tends
    to make K look smaller than it should - see plot_intra_archetype_variance
    for the metric that actually decides K (Section 4.3)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["k"], summary["rss"], "-", color="#2563EB", linewidth=2, zorder=1)
    ax.scatter(summary["k"], summary["rss"], s=45, color="#7F1D1D", zorder=2)
    ax.set_xlabel("Number of Archetypes (K)")
    ax.set_ylabel("Residual Sum of Squares (RSS)")
    ax.set_title(title)
    ax.set_xticks(summary["k"])
    ax.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


def plot_intra_archetype_variance(summary, out_path,
                                  title="Intra-Archetype Variance per K, NBA Data (ADA)"):
    """Mirrors the paper's Figures 3/4. This is what Section 4.3 actually
    decides K on: read where the curve stops decreasing meaningfully
    (paper's own NBA/ADA result: minimum at K=9, but the K=8->9 gain was
    marginal and the 9th archetype was a subtle variant of an existing one,
    so they kept K=8 - a domain-knowledge call, not a strict-minimum rule)."""
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(summary["k"], summary["intra_archetype_var"], "-", color="#2563EB", linewidth=2, zorder=1)
    ax.scatter(summary["k"], summary["intra_archetype_var"], s=45, color="#7F1D1D", zorder=2)
    ax.set_xlabel("Number of Archetypes (K)")
    ax.set_ylabel("Intra-Archetype Variance")
    ax.set_title(title)
    ax.set_xticks(summary["k"])
    ax.grid(True, color="#E5E7EB", linewidth=0.6, zorder=0)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved -> {out_path}")


# --- Part G: K selection, criterion 2 ---------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: Asked whether the RSS elbow is a sufficient basis for choosing K
given that K also determines the size of my Step 2 synergy matrix.
Used: it is not. After symmetrization M has K(K+1)/2 free parameters -
quadratic in K - while the number of two-man lineups is fixed, so weighted
exposure per cell thins out fast. Past some K most cells are noise while RSS
is still falling. This constraint does not exist in the source paper, which
only needs K to be interpretable.
Not AI: starve_ratio=0.10 as the threshold for "estimable", and the choice
to define exposure as minutes-weighted rather than a raw pair count.
"""

def k_exposure_diagnostic(fit_df, lookup_df, pairs_df, candidate_ks=(6, 8, 10, 12),
                          seed=0, max_iter=200, starve_ratio=0.10):
    """Weighted exposure mass per synergy-matrix cell, at each candidate K.

    fit_df    : population the basis is fit on (small, fast)
    lookup_df : population projected for the pair lookup — must cover the
                players appearing in pairs_df, otherwise pairs get silently
                skipped and the exposure counts are biased toward starters
    pairs_df  : needs PLAYER_A_ID, PLAYER_B_ID, SEASON, MIN
    """
    key = {(int(r.PLAYER_ID), str(r.SEASON)): i
           for i, r in enumerate(lookup_df.itertuples())}

    out = []
    for k in candidate_ks:
        fit = fit_basis(fit_df, k, seed=seed, max_iter=max_iter)
        P = project(lookup_df, fit["basis"], fit["mu"], fit["sd"],
                    fit["feature_columns"])

        mass = np.zeros((k, k))
        matched = 0
        for r in pairs_df.itertuples():
            ia = key.get((int(r.PLAYER_A_ID), str(r.SEASON)))
            ib = key.get((int(r.PLAYER_B_ID), str(r.SEASON)))
            if ia is None or ib is None:
                continue
            pa, pb = P[ia], P[ib]
            mass += r.MIN * 0.5 * (np.outer(pa, pb) + np.outer(pb, pa))
            matched += 1

        iu, ju = np.triu_indices(k)
        cells = mass[iu, ju]
        n_cells = len(cells)
        threshold = starve_ratio * cells.sum() / n_cells
        effective = int((cells > threshold).sum())

        print(f"\nK={k}: matched {matched}/{len(pairs_df)} pairs "
              f"({matched / len(pairs_df):.0%})")
        if matched < 0.9 * len(pairs_df):
            print("  WARNING: many pairs skipped — recipe coverage has holes, "
                  "fix that before reading the numbers below")
        print(f"  {effective}/{n_cells} cells above threshold, "
              f"min={cells.min():.0f} min, median={np.median(cells):.0f} min")

        detail = pd.DataFrame({
            "cell": [f"{i}x{j}" for i, j in zip(iu, ju)],
            "mass_minutes": cells,
            "share": cells / cells.sum(),
        }).sort_values("mass_minutes")
        print("\n  8 thinnest cells:")
        print(detail.head(8).to_string(index=False))
        detail.to_csv(DATA_DIR / f"exposure_cells_k{k}.csv", index=False)

        out.append({
            "k": k, "n_cells": n_cells, "effective_cells": effective,
            "effective_share": effective / n_cells,
            "min_cell_mass": float(cells.min()),
            "median_cell_mass": float(np.median(cells)),
            "pairs_matched": matched,
        })

    res = pd.DataFrame(out)
    res.to_csv(DATA_DIR / "k_exposure.csv", index=False)
    return res

# --- Part H: diagnostics ----------------------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: Asked what can go wrong with the recipes themselves once the basis
is fit, specifically for the low-minute players (Demin, Traore, Saraf, Wolf)
who are excluded from training but needed in Step 3.
Used: archetypoids are vertices of the training population's convex hull, so
players outside it get projected onto the boundary and come out near
one-hot. That reads as "pure archetype" but is a projection failure, and it
hits exactly the rookies Step 3 depends on. Compare saturation across
minute bands to detect it. Separately, exp(entropy) gives an effective
component count that flags K too large (near 1) or too small (near K).
Not AI: the minute-band cutoffs, the 0.9/0.7 saturation thresholds, and the
2-3.5 healthy range for effective components.
"""


def check_simplex(P, atol=1e-6):
    """Recipes must sum to 1 and be non-negative.

    Step 2's collinearity treatment depends on the exposure fingerprint
    summing to 1, which follows from this.
    """
    s = P.sum(axis=1)
    ok = np.allclose(s, 1.0, atol=atol)
    print(f"simplex check: row sums in [{s.min():.8f}, {s.max():.8f}] "
          f"{'ok' if ok else 'FAIL'}")
    assert ok, "recipes do not sum to 1"
    assert (P >= -1e-9).all(), "negative weights present"


def check_saturation(recipes, k,
                     bins=((300, 500), (500, 1000), (1000, 2000), (2000, 10**9))):
    """Max recipe weight by minutes band. High max weight in the low bands
    means those players are being pinned to the hull boundary."""
    cols = [f"arch_{i}" for i in range(k)]
    max_w = recipes[cols].values.max(axis=1)

    rows = []
    for lo, hi in bins:
        m = ((recipes["MIN"] >= lo) & (recipes["MIN"] < hi)).values
        if m.sum() == 0:
            continue
        rows.append({
            "band": f"{lo}-{hi if hi < 10**9 else '+'}",
            "n": int(m.sum()),
            "pct_max_gt_0.9": 100 * float((max_w[m] > 0.9).mean()),
            "pct_max_gt_0.7": 100 * float((max_w[m] > 0.7).mean()),
            "median_max_weight": float(np.median(max_w[m])),
        })
    out = pd.DataFrame(rows)
    print("\nsaturation by minutes band:")
    print(out.to_string(index=False))
    return out


def check_mixture_health(P):
    """Effective number of components, exp(entropy). Reads as "this player is
    a blend of roughly N archetypes". Near 1 means K too large; near K means
    K too small or the features do not separate. Healthy median is 2-3.5."""
    ent = -(P * np.log(P + 1e-12)).sum(axis=1)
    eff = np.exp(ent)
    print(f"\neffective components: median={np.median(eff):.2f}, "
          f"p10={np.percentile(eff, 10):.2f}, p90={np.percentile(eff, 90):.2f}")
    return eff


# --- Part I: archetype identity and profile ---------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: Asked how to key the archetype-to-player match given the name+season
collision problem from Part E, and what output I need in order to name the
archetypes in the write-up and later explain M's high-value cells in
basketball terms.
Used: match by row index into the training frame; report the per-archetype
top and bottom features in z units so each archetype can be named from its
profile rather than from its exemplar alone.
Not AI: top_n=6, and the rule that an archetype which is a marginal player
with all-extreme features is a hull outlier to be fixed by winsorizing or
raising the minute floor.
"""


def match_archetypes_to_players(df, fit):
    """Which real player-season is each archetypoid? Matched by row index."""
    X_z = ((df[fit["feature_columns"]].astype(float) - fit["mu"]) / fit["sd"]).values

    rows = []
    for i, a in enumerate(fit["basis"]):
        d = np.linalg.norm(X_z - a, axis=1)
        j = int(np.argmin(d))
        rec = df.iloc[j]
        rows.append({
            "archetype": i, "row_index": j,
            "PLAYER_ID": rec["PLAYER_ID"], "PLAYER_NAME": rec["PLAYER_NAME"],
            "SEASON": rec["SEASON"], "TEAM": rec.get("TEAM_ABBREVIATION", ""),
            "MIN": rec["MIN"], "distance": float(d[j]),
        })
        print(f"  archetype {i}: {rec['PLAYER_NAME']} ({rec['SEASON']}, "
              f"{rec['MIN']:.0f} min) d={d[j]:.6f}")

    out = pd.DataFrame(rows)
    if (out["distance"] > 1e-6).any():
        print("  warning: some archetypes are not exact observations. That is AA "
              "behaviour, not ADA - check the model class.")
    return out


def describe_archetypes(fit, top_n=6):
    """Top and bottom features per archetype, in z units."""
    names = fit["feature_columns"]
    for i, row in enumerate(fit["basis"]):
        order = np.argsort(row)
        hi = [(names[j], row[j]) for j in order[::-1][:top_n]]
        lo = [(names[j], row[j]) for j in order[:top_n]]
        print(f"\narchetype {i}")
        print("  high: " + ", ".join(f"{n}({v:+.2f})" for n, v in hi))
        print("  low : " + ", ".join(f"{n}({v:+.2f})" for n, v in lo))


"""
AI-ASSISTED (Claude Code, chat)
Prompt: "我发现step1_archetypes_model没有做完 就是在选出了8个archetypes我想找到
这8个和paper或者现实中archtype的对应" - ran phase3_diagnose (already written
above, never actually executed against the final 2025-26 basis) to get
official per-archetype z-score profiles, then cross-referenced them
against the "Scouting Anyone" paper's own NBA ADA table (Section 5.2, 8
archetypoids: Inside Scoring Big, Shooting Specialist, Combo Guard,
Offensive Engine, Traditional Playmaker, 3&D Wing, Rim Protector/Roll
Man, Mobile Big - read directly from the paper's PDF via pypdf, not
guessed).
Used: this mapping is a judgment call, not a computed similarity score -
same as the paper's own methodology (Section 5.1: "we carried out a
detailed exploratory analysis to interpret the player types... reviewing
descriptive statistics, examining mean z-scores... boxplots"). Written as
a literal, reviewable table rather than an automated nearest-match,
since forcing a numeric score against the paper's qualitative
descriptions would manufacture false precision.
Not AI: which basketball concept each feature signature represents is
domain knowledge, not derivable from the z-scores alone - flagged 2 of 8
as partial/imperfect matches rather than forcing all 8 to look clean,
per this project's stated preference for stating limitations plainly.
"""

PAPER_NBA_ARCHETYPES = {
    "Inside Scoring Big": "Interior finisher and offensive rebounder - tallest, highest %post-up, high %roll man/putback, high USG%",
    "Shooting Specialist": "Movement or spot-up shooter - high off-screen & spot-up, low self-creation, low 2FGA%",
    "Combo Guard": "Scoring guard with creation ability - shorter, high PnR handler%/isolation%, moderate AST%",
    "Offensive Engine": "Primary initiator and focal scorer - very high USG%, low % of assisted shots, isolation/PnR-heavy",
    "Traditional Playmaker": "Pass-first facilitator - high AST%, higher TOV%, low shot volume, high PnR handler%",
    "3&D Wing": "Perimeter defender & floor spacer - average size, high 3PA%, moderate STL%/BLK%, minimal creation/rim presence",
    "Rim Protector / Roll Man": "Interior defender + PnR finisher - elite BLK%/OREB%, high roll man/putback frequency",
    "Mobile Big": "Stretch-capable or perimeter-mobile big - slightly shorter bigs, high corner-3 FGA%, high % assisted FGA, high % on cuts",
}

# archetype index (this project's basis) -> (paper label, match quality, note)
ARCHETYPE_TO_PAPER = {
    0: ("3&D Wing", "strong",
        "Batum: spot-up/corner-3 heavy, low USG%/FTr - low-usage floor-spacer profile"),
    1: ("Inside Scoring Big", "strong",
        "Valanciunas: post-up + TRB% + roll-man led, high scoring - close to the paper's own Vucevic exemplar"),
    2: ("Rim Protector / Roll Man", "strong",
        "Capela: OREB%/BLK%/roll-man led, no outside shot - matches the paper's Mitchell Robinson exemplar closely"),
    3: ("Combo Guard", "strong",
        "D. Russell: ball-handler + high USG%/AST% but low TS% - scoring guard w/ creation, not pure pass-first"),
    4: ("Play-Finishing Big", "data-fit",
        "Kalkbrenner: cut +3.7, TS% +3.4, rim-FGA +3.2, OREB/BLK led, NO outside shot (3PA%/Dist. bottomed out) - an efficient vertical finisher, not a stretch/mobile big. Relabelled from the paper's 'Mobile Big' to fit this fit's own z-profile (owner's call: labels describe this data, not the paper)."),
    5: ("Shooting Specialist", "strong",
        "Hardaway Jr.: off-screen/handoff led, low self-creation, high 3PA share - near-exact match to the paper's description"),
    6: ("Offensive Engine", "strong",
        "SGA: isolation + USG% + BPM led, low %assisted - same tier of player the paper itself names (Harden/Doncic)"),
    7: ("Perimeter Defender", "data-fit",
        "Mbeng: STL% +2.4 the one standout, USG%/PTS/BPM all low, AST% not distinguishing (rank 27/29) - a low-usage point-of-attack defender, not a facilitator. Relabelled from the paper's 'Traditional Playmaker' to fit this fit's own z-profile (its top loaders - Mashack, Andre Jackson Jr., Dru Smith - are all POA defenders, zero facilitators)."),
}


def label_archetypes_vs_paper(fit, defs):
    """Prints + returns the archetype -> paper-label mapping above, joined
    with this basis's own exemplar (defs, from match_archetypes_to_players)
    for a one-look comparison table. The mapping itself lives in
    ARCHETYPE_TO_PAPER - re-review it by hand if the basis is ever refit
    and the archetype indices/exemplars shift."""
    rows = []
    for i in range(fit["k"]):
        label, quality, note = ARCHETYPE_TO_PAPER.get(i, ("(unmapped)", "n/a", ""))
        exemplar = defs.loc[defs["archetype"] == i, "PLAYER_NAME"].iloc[0]
        rows.append({"archetype": i, "exemplar": exemplar, "paper_label": label,
                     "match_quality": quality, "note": note})
    out = pd.DataFrame(rows)
    print("\narchetype <-> paper (Scouting Anyone, Section 5.2 NBA table):")
    print(out.to_string(index=False))
    return out


# --- Part J: phase runners --------------------------------------------
"""
AI-ASSISTED (Claude, chat)
Prompt: Asked in what order these should run, given that choosing K requires
already being able to fit and project, so K cannot be chosen first.
Used: separate "does the pipeline work" from "what is the right K" - phase 0
fits at a throwaway K purely to check convergence and the projection
assertion, and only then does phase 1 sweep K. Phase 3 can send you back to
phase 2 if saturation or an outlier archetype turns up.
Not AI: the phase boundaries themselves and the two-basis (train / full)
output layout.
"""

# test code in smoke mode, no K selection, check return convergerd or not, no disk save
"""
Here is output from my run locally:
(.nets) (base) brycelee@MacBook-Pro-3 src % python step1_archetypes_model.py 
Using NumPy backend
dropped 4 row(s) with unresolved missing data (likely a name-join miss)
population: 771 rows | MIN>=2000 | seasons 2017-18..2025-26
K=3 seed=0: EV=0.3597 (6 iters, 42s)
projection check: max diff=0.000647, mean=0.000027
  projection verified
simplex check: row sums in [1.00000000, 1.00000000] ok

phase 0 done. If both checks passed, run phase 1.
"""
 
def phase0_smoke(k=3, min_threshold=2000):
    """Pipeline validation. K here is a placeholder and gets discarded; the
    only outputs that matter are `converged` and the projection check."""
    df = load_population(min_threshold=min_threshold)
    fit = fit_basis(df, k=k, max_iter=200, verbose=True)
    if not fit["converged"]:
        print("\ndid not converge - raise max_iter before anything else")
    verify_projection(df, fit)
    P = project(df, fit["basis"], fit["mu"], fit["sd"], fit["feature_columns"])
    check_simplex(P)
    print("\nphase 0 done. If both checks passed, run phase 1.")
    return fit


def phase1_select_k(min_threshold=1000, season_min=None, season_max=None,
                    candidate_ks=(2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
                    seeds=(0, 1, 2),
                    pairs_df=None, figures_dir=None, out_name="k_selection"):
    """season_min/season_max let this target the exact population the final
    basis is trained on (season_min=season_max="2025-26", min_threshold=300)
    rather than a different, stale population - the K=8 justification is
    only honest if it's checked against the same data phase2_build used.
    candidate_ks starts at 2 (not 5) so the RSS plot can actually show the
    paper's own point: the elbow lands deceptively early (~4)."""
    df = load_population(min_threshold=min_threshold, season_min=season_min, season_max=season_max)
    raw, summary = select_k(df, candidate_ks=candidate_ks, seeds=seeds, out_name=out_name)

    figures_dir = Path(figures_dir) if figures_dir else Path(__file__).resolve().parent.parent / "figures"
    plot_rss_scree(summary, figures_dir / f"{out_name}_rss_scree.png")
    plot_intra_archetype_variance(summary, figures_dir / f"{out_name}_intra_variance.png")

    if pairs_df is not None:
        candidates = summary.loc[summary["all_converged"], "k"].tolist()
        k_exposure_diagnostic(df, pairs_df, candidate_ks=candidates)
    else:
        print("\npairs_df not supplied - run k_exposure_diagnostic() once the "
              "two-man lineup table exists. Do not finalize K on the elbow alone.")
    return summary


"""
AI-ASSISTED (Claude Code, chat)
Prompt: 我有些思路上的转变 - direction change, quoting the intended write-up
language directly: "we trained our Archetypal Analysis model on NBA regular
season the 2025-26 season (player-season observations, minimum 300 minutes
played through Regular Season)." This replaces the multi-season-train /
2025-26-holdout design the Part A comment above describes - added
season_min so a single-season population (season_min=season_max="2025-26")
is expressible with load_population()'s existing filters, no new loader
needed. Threaded through the `everyone` projection call too, which
previously always pulled every season up to season_max regardless of
season_min - would have silently blended in prior seasons for this new
single-season use if left alone.
Not AI: the direction change itself, and MIN>=300 as the threshold.
"""


def phase2_build(k, min_threshold=1000, season_max=None, season_min=None, out_dir=None,
                 seeds=(0, 1, 2), winsorize=None):
    """Fit the production basis and project every eligible player.

    For the current design (train == project, single season): pass
    season_min=season_max="2025-26", min_threshold=300:
        phase2_build(k=K, min_threshold=300, season_min="2025-26", season_max="2025-26")

    The old multi-season-train / 2025-26-holdout split is still available
    (season_max="2024-25" for a train-only basis, season_max=None for a
    basis fit on everything through the present) if that comparison is
    ever wanted again - just don't pass season_min.
    """
    if out_dir is None:
        if season_min is not None and season_min == season_max:
            out_dir = DATA_DIR / f"basis_{season_min.replace('-', '_')}"
        else:
            out_dir = DATA_DIR / ("basis_train" if season_max else "basis_full")
    out_dir = Path(out_dir)

    _log(f"phase2_build starting: k={k}, min_threshold={min_threshold}, "
         f"season_min={season_min}, season_max={season_max}, out_dir={out_dir}")

    train = load_population(min_threshold=min_threshold, season_min=season_min, season_max=season_max)
    _log(f"population loaded: {len(train)} rows - starting fit_basis_best_of "
         f"(this is the long step)")
    fit = fit_basis_best_of(train, k=k, seeds=seeds, max_iter=200, winsorize=winsorize)
    assert fit["converged"], "production basis did not converge - raise max_iter"
    _log("fit_basis_best_of done")

    print("\narchetype -> real player:")
    defs = match_archetypes_to_players(train, fit)

    save_basis(fit, out_dir)
    defs.to_csv(out_dir / "archetype_definitions.csv", index=False)
    _log(f"basis + archetype definitions saved -> {out_dir}")

    # Project the full eligible population, including players below the
    # training floor, using the training population's mu/sd. Same season
    # window as training - see the prompt note above re: season_min.
    everyone = load_population(min_threshold=300, season_min=season_min, season_max=season_max)
    P = project(everyone, fit["basis"], fit["mu"], fit["sd"], fit["feature_columns"])
    check_simplex(P)

    recipes = recipes_frame(everyone, P, k)
    recipes.to_csv(out_dir / "recipes.csv", index=False)
    np.save(out_dir / "recipes.npy", P)
    _log(f"saved recipes -> {out_dir} ({len(recipes)} rows) - phase2_build complete")
    return fit, recipes


def phase3_diagnose(out_dir):
    """Run after phase 2. Either check here can send you back to phase 2."""
    out_dir = Path(out_dir)
    fit = load_basis(out_dir)
    recipes = pd.read_csv(out_dir / "recipes.csv")
    defs = pd.read_csv(out_dir / "archetype_definitions.csv")
    P = np.load(out_dir / "recipes.npy")

    check_simplex(P)
    sat = check_saturation(recipes, fit["k"])
    check_mixture_health(P)
    describe_archetypes(fit)
    labels = label_archetypes_vs_paper(fit, defs)
    labels.to_csv(out_dir / "archetype_labels.csv", index=False)

    print("\nManual review:")
    print("  1. Is every archetype a basketball-sensible role? A marginal "
          "player with all-extreme features is a hull outlier - refit with "
          "winsorize=0.01 or a higher minute floor.")
    print("  2. Does the 300-500 band saturate far more than 2000+? If so the "
          "rookie recipes are unreliable and Step 3 has to say so.")
    return sat


# AI-ASSISTED (Claude Code, chat) - Parts K-N below are the full,
# unmodified content of the former src/step3_nets_lineup_scoring.py
# (CLAUDE.md's "Step 3 - Nets lineup scoring"), absorbed into this file
# during the src/ restructuring since resolve_roster/NETS_ROSTER are
# roster-to-archetype-recipe resolution - an archetype-model concern
# that every step2/3/4 portal-tab file needs, not tab-specific UI code.
# Only the header docstring split below and the sys.path/import dance
# inside what's now phase_score() were removed (load_basis is a same-
# file call now); every other line, including Part C/D's scoring math
# (not currently wired into the live app - see DIAGNOSTICS_README.md),
# is unchanged.

# --- Part K: Nets lineup scoring (CLAUDE.md Step 3) ---------------------
"""
Five-man score = sum over the 10 pairs in a 5-man group, each pair scored
as p_x.T @ M @ p_y using step2's v1 (individual-quality-controlled) matrix.
This is an additivity assumption - a lineup's chemistry is treated as the
sum of its pairwise chemistries, not a genuine 5-way interaction.
CLAUDE.md's 3-part defense for this, stated here rather than left implicit:
  1. data reality - Day 1's own density check showed even 3-man combos
     don't survive MIN>=100 at a density that could support fitting a true
     interaction model past pairs.
  2. it's the standard first-order-approximation convention for exactly
     this reason in lineup-construction work.
  3. there's a stated upgrade path once real on-court tracking data
     (spacing, shot quality allowed, etc.) is available.

Part K.1  the 13 data-eligible Nets players (CLAUDE.md roster context)
Part K.2  resolve each to a PLAYER_ID + current-season archetype recipe
Part K.3  score every 5-man combo (sum of C(5,2)=10 pairwise terms)
Part K.4  rank + report

Explicitly out of scope here: Mikel Brown Jr., Joshua Jefferson, Tyler
Bilodeau (zero NBA data - CLAUDE.md routes them to Step 4's slot query,
which asks what archetype a 5th-man slot NEEDS rather than trying to score
a rookie who has no recipe to plug in).

STATUS: skeleton, written while step1's basis was still fitting. Part K.1/K.2's
roster-resolution logic is tested against the real database (doesn't need
the rest of step1); Part K.3/K.4's scoring math is tested against synthetic
recipes/M, not real ones yet - synergy_matrix_v1.npy (step2_synergy_matrix.py,
now offline in src/pipeline/) was never actually produced. Re-run
phase_score() if that changes.
"""

# --- Part K.1: the current Nets roster (corrected 2026-07-27) --------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: the previous version of this list ("full roster turnover") was
wrong - user pasted a roster screenshot/list showing MPJ, Randle, Terance
Mann, Day'Ron Sharpe, Noah Clowney, and Josh Minott still active, then
asked to verify against ESPN's live roster page
(https://www.espn.com/nba/team/roster/_/name/bkn/brooklyn-nets). Fetched
it directly: confirms all 6 are still on the roster, plus names never
before seen in this project (E.J. Liddell, Nolan Traore, Julius Randle,
Keon Ellis, Tyson Etienne, Moritz Wagner) - 19 total. The 9 names the old
list had instead (John Ukomadu, Aaron Scott, Ben Humrichous, Grant Nelson,
Dwight Murray Jr., Dion Brown, Duke Brennan, Dain Dainja, Hunter Sallis) do
not appear on ESPN's page and are dropped as unconfirmed.
Used: re-checked every one of the 19 names directly against
build_nba_side_tables() for season 2025-26 (not assumed from ESPN age/exp)
- 16 clear MIN>=300 and already have a real fitted recipe in
data/basis_2025_26/recipes.csv (the archetype model was fit league-wide,
so a recipe doesn't care which team a player's minutes were logged under -
this is exactly why Julius Randle/Keon Ellis/Moritz Wagner, whose
season-cumulative row still shows their previous team (MIN/CLE/ORL - they
were evidently traded to Brooklyn recently enough that the row hasn't
caught up), still resolve fine: the join is PLAYER_ID+SEASON only, no team
filter, so resolve_roster()'s existing logic needed zero code changes -
only the roster data itself was wrong). Mikel Brown Jr., Tyler Bilodeau,
and Joshua Jefferson return zero rows in player_base at ANY season
(confirmed via a direct LIKE query, not inferred) - true zero-data
rookies, same NCAA-bridge path as before.
Not AI: the roster itself (from ESPN, the user's source), which players
count as data-eligible (re-derived here directly from the DB per the
project's own "check, don't assume" convention), and the decision to defer
the NCAA-bridge mechanics for the 3 true rookies to a later message - all
the user's calls.
"""

NETS_ROSTER = [
    "Mikel Brown Jr.",
    "Danny Wolf",
    "Drake Powell",
    "Egor Dëmin",
    "E.J. Liddell",
    "Tyson Etienne",
    "Nolan Traore",
    "Terance Mann",
    "Michael Porter Jr.",
    "Day'Ron Sharpe",
    "Noah Clowney",
    "Julius Randle",
    "Chaney Johnson",
    "Tyler Bilodeau",
    "Ben Saraf",
    "Josh Minott",
    "Joshua Jefferson",
    "Keon Ellis",
    "Moritz Wagner",
]

# Real 2025-26 NBA data, confirmed MIN>=300 via build_nba_side_tables() -
# resolve_roster() below finds these directly, no bridge needed. Team label
# in the season-cumulative row may lag a recent trade (Randle=MIN,
# Ellis=CLE, Wagner=ORL as of this check) - doesn't matter, the recipe join
# is PLAYER_ID-based, not team-based.
NETS_ROSTER_NBA_DATA = [
    "Danny Wolf", "Drake Powell", "Egor Dëmin", "E.J. Liddell", "Nolan Traore",
    "Terance Mann", "Day'Ron Sharpe", "Noah Clowney", "Julius Randle",
    "Chaney Johnson", "Ben Saraf", "Josh Minott", "Michael Porter Jr.",
    "Keon Ellis", "Tyson Etienne", "Moritz Wagner",
]

# Zero real NBA data at any season (confirmed via a direct DB query, not
# inferred) - each needs step4's NCAA bridge. Bridge mechanics (CBBD school/
# season per player, ridge-regression mapping) deferred to a later message
# per explicit instruction - this list only marks who needs it.
NETS_ROSTER_NCAA_BRIDGE = [
    "Mikel Brown Jr.",
    "Tyler Bilodeau",
    "Joshua Jefferson",
]

assert len(NETS_ROSTER) == 19, "roster list drifted from the 2026-07-27 ESPN-verified correction"
assert len(NETS_ROSTER_NBA_DATA) + len(NETS_ROSTER_NCAA_BRIDGE) == len(NETS_ROSTER)


# --- Part K.2: resolve roster names -> recipes -----------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: 帮我把 step3 的框架写好. Asked how to look the roster up safely,
given this session's own history of name-matching bugs (the Egor Dёmin
Cyrillic-homoglyph miss, the Martin-twins GROUP_NAME collision). Checked
that reusing step0's own _normalize_name is possible (it's importable, not
duplicated here) so a fix to the join logic in one place fixes both.
Used: resolve-and-report rather than assume - fail loudly on 0 or >1
matches instead of silently taking the first row, since a silent wrong
match here would score a different real player without any visible error.
Not AI: MIN>=300 recipe floor already decided in step1/step0; treating
"MISSING" as a real, expected possibility rather than a bug (a rookie with
<300 minutes even across a full season is plausible, not a code fault).
"""


def resolve_roster(season="2025-26"):
    """Look up each NETS_ROSTER_NBA_DATA name's PLAYER_ID for the given
    season - the 16 current-roster players confirmed to have real NBA data
    at MIN>=300 (league-wide, regardless of which team logged those
    minutes - a recent trade doesn't invalidate the recipe join, since it's
    PLAYER_ID+SEASON, not team-scoped). The other 3 (NETS_ROSTER_NCAA_BRIDGE)
    are NOT looked up here at all - they have no real NBA row to find
    regardless of season, and belong to step4's NCAA-bridge path instead.

    Uses step0's own _normalize_name so this matches the exact same rules
    (accents, Cyrillic homoglyphs, suffixes, case) as the B-Ref join - not
    a second, possibly-inconsistent name-matching implementation.
    """
    pop = build_nba_side_tables()
    season_pop = pop[pop["SEASON"] == season].copy()
    season_pop["_norm"] = season_pop["PLAYER_NAME"].apply(_normalize_name)

    resolved, missing = [], []
    for name in NETS_ROSTER_NBA_DATA:
        norm = _normalize_name(name)
        match = season_pop[season_pop["_norm"] == norm]
        if len(match) == 0:
            missing.append(name)
            continue
        if len(match) > 1:
            raise ValueError(f"{name!r} matched {len(match)} rows in {season} - ambiguous, fix before scoring")
        row = match.iloc[0]
        resolved.append({
            "roster_name": name, "PLAYER_ID": int(row["PLAYER_ID"]),
            "PLAYER_NAME": row["PLAYER_NAME"], "MIN": float(row["MIN"]),
        })

    out = pd.DataFrame(resolved)
    print(f"resolved {len(out)}/{len(NETS_ROSTER_NBA_DATA)} NBA-data roster players in {season}")
    if missing:
        print(f"  MISSING (not in {season} at MIN>=300, or a name-spelling mismatch "
              f"- check against player_bio before assuming it's just low minutes): {missing}")
    return out


def attach_recipes(roster_df, recipes, k, season="2025-26"):
    """Join resolved roster PLAYER_IDs to their archetype recipe vectors."""
    arch_cols = [f"arch_{i}" for i in range(k)]
    season_recipes = recipes[recipes["SEASON"] == season][["PLAYER_ID"] + arch_cols]

    out = roster_df.merge(season_recipes, on="PLAYER_ID", how="left")
    n_before = len(out)
    out = out.dropna(subset=["arch_0"]).reset_index(drop=True)
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"attach_recipes: dropped {n_dropped}/{n_before} roster players "
              f"(resolved in build_nba_side_tables() but missing from recipes.csv - "
              f"check step1's projection floor)")
    if len(out) < 5:
        raise ValueError(f"only {len(out)} roster players have recipes - can't form a 5-man lineup")
    return out


# --- Part K.3: score every 5-man combo --------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: 帮我把 step3 的框架写好, the p_x.T @ M @ p_y sum-over-pairs scoring
CLAUDE.md specifies. Asked for a vectorized version once it's more than a
toy roster, since 13 players is only C(13,5)=1287 combos now but this
should not become the bottleneck if the roster grows.
Used: precompute all pairwise p_x.T @ M @ p_y scores once (C(13,2)=78 of
them, not 1287*10), then each 5-man lineup's score is just a sum over its
10 already-computed pair scores - avoids repeating the same matrix
multiply for the same pair across every combo that contains it.
Not AI: the additivity model itself (CLAUDE.md's, defended in the module
docstring above), which pairs count (all C(5,2)=10, no double-counting).
"""


def precompute_pair_scores(recipe_matrix, player_ids, M):
    """All pairwise p_x.T @ M @ p_y scores, once. recipe_matrix: (n, k),
    rows aligned with player_ids. Returns {(id_a, id_b): score} for a < b.
    """
    n = len(player_ids)
    scores = {}
    for a, b in combinations(range(n), 2):
        s = float(recipe_matrix[a] @ M @ recipe_matrix[b])
        key = tuple(sorted((player_ids[a], player_ids[b])))
        scores[key] = s
    return scores


def score_all_combos(roster_recipes, M):
    """roster_recipes: DataFrame with PLAYER_ID, PLAYER_NAME, arch_0..arch_{k-1}.
    Returns every 5-man combo from the roster, ranked by summed pair score.
    """
    ids = roster_recipes["PLAYER_ID"].tolist()
    names = dict(zip(roster_recipes["PLAYER_ID"], roster_recipes["PLAYER_NAME"]))
    arch_cols = [c for c in roster_recipes.columns if c.startswith("arch_")]
    recipe_matrix = roster_recipes[arch_cols].values

    pair_scores = precompute_pair_scores(recipe_matrix, ids, M)

    results = []
    for combo in combinations(ids, 5):
        total = sum(pair_scores[tuple(sorted(pair))] for pair in combinations(combo, 2))
        results.append({
            "players": " / ".join(names[i] for i in combo),
            "player_ids": combo,
            "score": total,
        })

    out = pd.DataFrame(results).sort_values("score", ascending=False).reset_index(drop=True)
    print(f"scored {len(out)} five-man combinations from {len(ids)} roster players "
          f"(C({len(ids)},5)={len(out)})")
    return out


# --- Part K.4: rank + report -------------------------------------------------

def report_top(ranked, n=10):
    print(f"\ntop {n} lineups:")
    print(ranked.head(n)[["players", "score"]].to_string(index=False))
    print(f"\nbottom {min(n, 3)} lineups (sanity check - should look like real bad-fit combos, not noise):")
    print(ranked.tail(min(n, 3))[["players", "score"]].to_string(index=False))


# --- Part L: Step 3 phase runner ------------------------------------------

def phase_score(basis_dir=None, matrix_path=None, season="2025-26"):
    basis_dir = Path(basis_dir) if basis_dir else DATA_DIR / "basis_full"
    matrix_path = Path(matrix_path) if matrix_path else DATA_DIR / "synergy_matrix_v1.npy"

    basis = load_basis(basis_dir)
    k = basis["k"]
    recipes = pd.read_csv(basis_dir / "recipes.csv")
    M = np.load(matrix_path)

    roster = resolve_roster(season=season)
    roster = attach_recipes(roster, recipes, k, season=season)

    ranked = score_all_combos(roster, M)
    ranked.to_csv(DATA_DIR / "nets_lineup_rankings.csv", index=False)
    report_top(ranked)
    return ranked


if __name__ == "__main__":
    # phase0_smoke()
    # phase1_select_k()
    # phase3_diagnose(DATA_DIR / "basis_full")
    # fit_basis(load_population(min_threshold=2000), k=8, max_iter=200)
    # we need to find the best k and min_threshold from 2000, 1000, and 300.
    # MIN>=2000
    # select_k(load_population(min_threshold=2000),
    #      candidate_ks=(5,6,7,8,9,10,11,12), seeds=(0,1,2))
        # all_converged: 5,6,7,8,9,10,11,12
    # k_exposure_diagnostic(
    # fit_df=load_population(min_threshold=2000),
    # lookup_df=load_population(min_threshold=300),   # 必须用宽人群查询
    # pairs_df=load_pairs(min_threshold=100),
    # candidate_ks=(8,),)

    # Direction change: train + project on 2025-26 regular season only, MIN>=300
    # (see the AI-ASSISTED note above phase2_build). 433 rows - much smaller
    # than the old multi-season population, should fit fast.
    phase2_build(k=8, min_threshold=300, season_min="2025-26", season_max="2025-26")


