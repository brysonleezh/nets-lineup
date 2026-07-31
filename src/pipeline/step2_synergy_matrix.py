"""
Step 2 - archetype-pair synergy matrix (my main extension; NOT in the paper).

Each 2-man lineup's net rating gets explained by an exposure vector: the
symmetrized outer product of the two players' archetype recipes, collapsed
to the K(K+1)/2 upper triangle since exposure(a,b) == exposure(b,a) - a
pair's synergy doesn't depend on which player GROUP_ID happened to list
first. Same symmetrization convention as step1's k_exposure_diagnostic
(Part G), so the K-selection diagnostic and this matrix describe the same
cells - that consistency matters, not just style.

Weighted regression of pair net rating on exposure, weights = shared
minutes, MIN >= 100 (CLAUDE.md Step 2). Reported twice per CLAUDE.md's
explicit requirement: v0 with no individual-quality control (expected to
degenerate toward "star x anything = positive" - this is the single most
likely silent failure mode named in CLAUDE.md's pitfalls list), v1
controlling for each side's own BPM. v1 is what everything downstream uses;
v0 is kept only as the comparison that justifies v1.

Part A  load 2-man lineup pairs
Part B  attach archetype recipes + quality covariate to each side
Part C  build the exposure design matrix
Part D  weighted regression (v0 uncontrolled, v1 controlled)
Part E  save / load the matrix
Part F  phase runner

STATUS: skeleton written while step1's K-selection sweep was still running,
so Parts A-C are tested against the real lineups table (they don't depend
on step1's output), but Parts D-F are untested against real recipes -
step1 hadn't produced basis_full/recipes.csv yet. Re-run phase_build() once
it has and fix whatever the first real run turns up.
"""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# AI-ASSISTED (Claude Code, chat) - path fix for the src/pipeline/ move:
# this file now sits one directory deeper than src/, so DATA_DIR needs an
# extra .parent and sys.path needs src/ added explicitly for the sibling
# imports below (both this module-level one and phase_build's local one).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step0_data import build_nba_side_tables

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"


# --- Part A: load 2-man lineup pairs -----------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: 帮我把 step2 的框架写好, using GROUP_ID from the lineups table.
Checked GROUP_ID's actual format live against the database first
(e.g. "-203471-1627752-") rather than assuming, since Step 0's own lineups
table caution note already flagged GROUP_NAME as unsafe for this and
GROUP_ID as the fix.
Not AI: MIN>=100 threshold (CLAUDE.md Step 2), which table/columns to pull.
"""


# AI-ASSISTED (Claude Code, chat) - during the src/ restructuring,
# _parse_pair_ids/load_pairs moved to step1_archetype_model.py (they're
# imported live by step2_diagnostic_analysis.py, which this file - now
# relocated to src/pipeline/ as abandoned/offline - shouldn't be a
# dependency of). Re-exported here so phase_build()'s own call below
# keeps working unchanged.
from step1_archetype_model import _parse_pair_ids, load_pairs  # noqa: E402,F401


# --- Part B: attach recipes + quality covariate -------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: 帮我把 step2 的框架写好. Asked to check whether recipes.csv can be
joined on (PLAYER_ID, SEASON) alone, given step1's own Part E note that
(PLAYER_NAME, SEASON) collides for a traded player. It can - recipes come
from build_nba_side_tables(), which already collapses a traded player's
in-season rows into one combined row via player_base's Totals per_mode
(the same reason a 2-man pair's SEASON-level recipe blends stats from
before/after a trade, rather than reflecting only the team the pair
happened on - a real approximation, not a bug, worth stating in the
write-up if it matters for a specific player like Randle).

Flagging, not resolving: BPM is already one of the 29 features the
archetype recipe itself was fit on, so using it again here as the
"individual quality" covariate is not fully independent of the exposure
vector it's supposed to be controlling against. CLAUDE.md's other named
option, each player's own on-court NET_RATING, would be more independent
but doesn't come out of build_nba_side_tables() today (it only pulls POSS
from player_advanced, not NET_RATING) - would need a small step0 addition
to use it instead. Using BPM for now since it's already at hand and step0
was mid-run (its K-selection sweep) when this was written, not touching it.
Not AI: which covariate to eventually pick is not decided here.
"""


def attach_recipes(pairs_df, recipes, k):
    """Attach each side's archetype-recipe vector via a vectorized merge
    (pairs_df can be tens of thousands of rows - a python-loop lookup
    would be slow). recipes must be keyed uniquely on (PLAYER_ID, SEASON).
    """
    arch_cols = [f"arch_{i}" for i in range(k)]
    recipe_slim = recipes[["PLAYER_ID", "SEASON"] + arch_cols]

    a = recipe_slim.rename(columns={"PLAYER_ID": "PLAYER_A_ID", **{c: f"A_{c}" for c in arch_cols}})
    b = recipe_slim.rename(columns={"PLAYER_ID": "PLAYER_B_ID", **{c: f"B_{c}" for c in arch_cols}})

    out = pairs_df.merge(a, on=["PLAYER_A_ID", "SEASON"], how="left")
    out = out.merge(b, on=["PLAYER_B_ID", "SEASON"], how="left")

    n_before = len(out)
    out = out.dropna(subset=["A_arch_0", "B_arch_0"]).reset_index(drop=True)
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"attach_recipes: dropped {n_dropped}/{n_before} pairs "
              f"(a side below the recipe floor, e.g. MIN<300 that season)")
    return out


def attach_quality_covariate(pairs_df, quality_col="BPM"):
    """Attach each side's own individual-quality stat (CLAUDE.md Step 2's
    required control) via build_nba_side_tables() - see the Part B prompt
    note above re: BPM vs. NET_RATING."""
    pop = build_nba_side_tables()[["PLAYER_ID", "SEASON", quality_col]]

    a = pop.rename(columns={"PLAYER_ID": "PLAYER_A_ID", quality_col: f"A_{quality_col}"})
    b = pop.rename(columns={"PLAYER_ID": "PLAYER_B_ID", quality_col: f"B_{quality_col}"})

    out = pairs_df.merge(a, on=["PLAYER_A_ID", "SEASON"], how="left")
    out = out.merge(b, on=["PLAYER_B_ID", "SEASON"], how="left")

    n_before = len(out)
    out = out.dropna(subset=[f"A_{quality_col}", f"B_{quality_col}"]).reset_index(drop=True)
    n_dropped = n_before - len(out)
    if n_dropped:
        print(f"attach_quality_covariate: dropped {n_dropped}/{n_before} pairs "
              f"(a side below build_nba_side_tables()'s MIN>=300 floor)")
    return out


# --- Part C: build the exposure design matrix ---------------------------

def build_exposure(pairs_df, k):
    """Symmetrized archetype-pair exposure per pair, upper triangle only
    (K(K+1)/2 cells) - same convention as step1's k_exposure_diagnostic.
    Fully vectorized (einsum over all pairs at once), not a per-row loop.
    """
    A = pairs_df[[f"A_arch_{i}" for i in range(k)]].values  # (n, k)
    B = pairs_df[[f"B_arch_{i}" for i in range(k)]].values  # (n, k)

    outer = 0.5 * (np.einsum("ni,nj->nij", A, B) + np.einsum("ni,nj->nij", B, A))  # (n, k, k)
    iu, ju = np.triu_indices(k)
    exposure = outer[:, iu, ju]  # (n, K(K+1)/2)
    return exposure, list(zip(iu.tolist(), ju.tolist()))


# --- Part D: weighted regression ----------------------------------------
"""
AI-ASSISTED (Claude Code, chat)
Prompt: 帮我把 step2 的框架写好. Asked what "weighted regression, weights =
shared minutes" concretely means as sklearn code, and how to turn the
K(K+1)/2 fitted coefficients back into a full symmetric K x K matrix M for
Step 3's quadratic form.
Used: LinearRegression(fit_intercept=True).fit(X, y, sample_weight=w) - the
project's own stack decision (sklearn, not statsmodels) rather than adding
a new dependency; scatter the fitted coefficients into upper AND lower
triangle so M is symmetric by construction, matching p_x.T @ M @ p_y being
order-independent.
Not AI: fit_intercept choice needs a real decision once real coefficients
exist (does a 0-exposure pair mean "no data", in which case no), and
whether v0 vs v1 gets compared by R^2, coefficient magnitude, or something
else - flagged below, not decided.
"""


def fit_synergy_matrix(exposure, net_rating, weights, k, cell_index,
                       extra_covariates=None, fit_intercept=True):
    """Weighted least squares: net_rating ~ exposure [+ extra_covariates].

    extra_covariates: None for v0 (uncontrolled), or an (n, 2) array of
    [A_quality, B_quality] for v1. Returns M (k, k) symmetric, plus the fit
    diagnostics needed to compare v0 vs v1 (CLAUDE.md wants both kept).
    """
    X = exposure if extra_covariates is None else np.hstack([exposure, extra_covariates])

    model = LinearRegression(fit_intercept=fit_intercept)
    model.fit(X, net_rating, sample_weight=weights)

    n_cells = exposure.shape[1]
    cell_coefs = model.coef_[:n_cells]
    covariate_coefs = model.coef_[n_cells:] if extra_covariates is not None else None

    M = np.zeros((k, k))
    for (i, j), c in zip(cell_index, cell_coefs):
        M[i, j] = c
        M[j, i] = c  # mirror onto the lower triangle so M is symmetric

    r2 = model.score(X, net_rating, sample_weight=weights)
    print(f"fit: R^2={r2:.4f} (weighted), n={len(net_rating)}, "
          f"n_cells={n_cells}{', +covariates' if extra_covariates is not None else ' (uncontrolled)'}")

    return {
        "M": M, "intercept": float(model.intercept_), "r2": r2,
        "covariate_coefs": None if covariate_coefs is None else covariate_coefs.tolist(),
        "k": int(k), "n_pairs": int(len(net_rating)),
    }


def compare_v0_v1(fit_v0, fit_v1):
    """CLAUDE.md's actual check: does v0 degenerate toward 'star x anything
    positive'? Report both fits' cell-value spread so that's inspectable,
    not just the two R^2 numbers - a v0 matrix that's uniformly positive
    and a v1 matrix with real structure is the expected, correct outcome,
    not a bug to fix."""
    print(f"\nv0 (uncontrolled): R^2={fit_v0['r2']:.4f}, "
          f"M range=[{fit_v0['M'].min():.3f}, {fit_v0['M'].max():.3f}], "
          f"frac positive={(fit_v0['M'] > 0).mean():.0%}")
    print(f"v1 (BPM-controlled): R^2={fit_v1['r2']:.4f}, "
          f"M range=[{fit_v1['M'].min():.3f}, {fit_v1['M'].max():.3f}], "
          f"frac positive={(fit_v1['M'] > 0).mean():.0%}")
    print("\nManual check: if v0 is ~uniformly positive and v1 has real "
          "sign variation across cells, that's CLAUDE.md's predicted "
          "degeneracy - confirms v1 is the one to use downstream, not a "
          "problem with the code.")


# --- Part E: save / load --------------------------------------------------

def save_matrix(fit, out_dir, name="synergy_matrix"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"{name}.npy", fit["M"])
    meta = {k: v for k, v in fit.items() if k != "M"}
    (out_dir / f"{name}_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"saved -> {out_dir / name}.npy")


def load_matrix(out_dir, name="synergy_matrix"):
    out_dir = Path(out_dir)
    M = np.load(out_dir / f"{name}.npy")
    meta = json.loads((out_dir / f"{name}_meta.json").read_text())
    return {"M": M, **meta}


# --- Part F: phase runner --------------------------------------------------

def phase_build(basis_dir, min_threshold=100, quality_col="BPM", out_dir=None):
    """End to end: load pairs -> attach recipes + quality -> fit v0 and v1
    -> save v1 (the one everything downstream should use).

    basis_dir: step1's output dir (data/basis_full, once it exists).
    """
    from step1_archetype_model import load_basis

    basis = load_basis(basis_dir)
    k = basis["k"]
    recipes = pd.read_csv(Path(basis_dir) / "recipes.csv")

    pairs = load_pairs(min_threshold=min_threshold)
    pairs = attach_recipes(pairs, recipes, k)
    pairs = attach_quality_covariate(pairs, quality_col=quality_col)

    exposure, cell_index = build_exposure(pairs, k)
    weights = pairs["MIN"].values
    y = pairs["NET_RATING"].values

    print("\n--- v0: uncontrolled ---")
    fit_v0 = fit_synergy_matrix(exposure, y, weights, k, cell_index)

    print("\n--- v1: controlled for individual quality ---")
    covariates = pairs[[f"A_{quality_col}", f"B_{quality_col}"]].values
    fit_v1 = fit_synergy_matrix(exposure, y, weights, k, cell_index, extra_covariates=covariates)

    compare_v0_v1(fit_v0, fit_v1)

    out_dir = Path(out_dir) if out_dir else DATA_DIR
    save_matrix(fit_v0, out_dir, name="synergy_matrix_v0")
    save_matrix(fit_v1, out_dir, name="synergy_matrix_v1")

    return fit_v0, fit_v1


if __name__ == "__main__":
    # phase_build(DATA_DIR / "basis_full")  # once step1 has produced it
    pass
