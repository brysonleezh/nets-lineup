"""
Pre-committed review of the 2026 Nets rookie predictions
(data/projections/predictions_frozen.json). Written at Phase 6
(2026-08-14), BEFORE the 2026-27 NBA season begins - run this ONCE real
2026-27 season data exists in the local DB.

FROZEN METHODOLOGY (do not modify after the season starts - any change
made after seeing real 2026-27 results would be exactly the kind of
post-hoc metric adjustment the whole project's discipline exists to
prevent):

1. Eligibility: for each of the 3 rookies (config.yaml), check actual
   2026-27 MIN >= threshold, threshold = 300 * games_in_season / 82
   (pro-rated only if the season was shortened - same convention Phase 3
   used for 2020-21; config.yaml's `games_in_season_2026_27` defaults to
   82, i.e. no pro-ration, override it only if the real 2026-27 season
   was actually shortened). Below threshold -> status 'unevaluable', NOT
   an error and NOT scored as a failure - this is exactly the "will he
   play" question the model never claimed to answer.
2. Build data/nba_historical/standardization_2026-27.json using the exact
   same method src/college_model/nba_standardization.py used for the
   other 9 seasons (full-league MIN>=300 population, the same 29 features
   in feature_order.json's order).
3. Project each qualifying rookie's REAL 2026-27 stat line onto the
   frozen 8x29 basis (data/basis_2025_26/basis.npz) via the existing,
   unmodified project() function, using the 2026-27 standardization from
   step 2 (never 2025-26's, never recomputed from rookies only - the same
   locked convention Phase 3 established).
4. Score with EXACTLY the Phase 5 metric list: JSD (base 2, primary), L1,
   top-1 hit, top-1-within-top-2, per-dimension MAE vs. per-dimension
   baseline (the training-set mean, same construction as Phase 5 Section
   D). No metric may be added, removed, or reweighted here.
5. Report actual vs. predicted top-3, whether any saturation caveat was
   borne out, and how each rookie's frozen k-NN comps' actual outcomes
   compare to his own real outcome.

Run: .nets/bin/python src/eval/review_2026_predictions.py
(requires season='2026-27' rows in the local NBA database - will report
which rookies are unevaluable rather than erroring if that data doesn't
exist yet or a rookie hasn't cleared the minutes threshold)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

from step1_archetype_model import load_population, project  # noqa: E402
from metrics import jensen_shannon, l1, top1_hit, top1_within_top2, per_dim_mae  # noqa: E402

K = 8
REVIEW_SEASON = "2026-27"
BASE_MIN_THRESHOLD = 300
CONFIG_PATH = REPO_ROOT / "config.yaml"
FROZEN_PATH = REPO_ROOT / "data" / "projections" / "predictions_frozen.json"
NETS_ROOKIES_PATH = REPO_ROOT / "data" / "projections" / "nets_rookies_2026.csv"
STD_DIR = REPO_ROOT / "data" / "nba_historical"
BASIS_DIR = REPO_ROOT / "data" / "basis_2025_26"
TRAIN_PATH = REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv"


# --- pure, synthetic-data-testable functions --------------------------------

def eligibility_threshold(games_in_season: int, base: int = BASE_MIN_THRESHOLD) -> float:
    """300 minutes, pro-rated only if the season is shorter than a normal
    82-game season - same convention as Phase 3's 2020-21 handling."""
    return base * games_in_season / 82.0


def check_eligibility(actual_minutes: float, games_in_season: int) -> tuple[bool, float]:
    threshold = eligibility_threshold(games_in_season)
    return actual_minutes >= threshold, threshold


def build_standardization_for_population(population_df: pd.DataFrame, feature_columns: list[str]) -> dict:
    """Pure function: given a population DataFrame (real or synthetic) and
    the 29 basis feature names, computes mu/sd exactly as
    src/college_model/nba_standardization.py does for every other season."""
    X = population_df[feature_columns].astype(float)
    mu = X.mean()
    sd = X.std()
    dead = sd[sd < 1e-12].index.tolist()
    if dead:
        raise ValueError(f"zero-variance features, z-scoring would divide by 0: {dead}")
    return {"season": REVIEW_SEASON, "feature_columns": feature_columns,
            "mu": mu.values.tolist(), "sd": sd.values.tolist(), "n": int(len(population_df))}


def score_actual_vs_predicted(y_actual: np.ndarray, y_predicted: np.ndarray, baseline: np.ndarray) -> dict:
    """Pure function: exactly the Phase 5 metric list, nothing added or
    removed. y_actual/y_predicted/baseline are (K,) simplex vectors for a
    single rookie (or (n,K) for a batch)."""
    y_actual = np.atleast_2d(y_actual)
    y_predicted = np.atleast_2d(y_predicted)
    baseline = np.atleast_2d(baseline)
    return {
        "jsd": float(jensen_shannon(y_predicted, y_actual)[0]),
        "l1": float(l1(y_predicted, y_actual)[0]),
        "top1_hit": float(top1_hit(y_predicted, y_actual)[0]),
        "top1_within_top2": float(top1_within_top2(y_actual, y_predicted)[0]),
        "per_dim_mae": per_dim_mae(y_predicted, y_actual).tolist(),
        "per_dim_baseline_mae": per_dim_mae(baseline, y_actual).tolist(),
    }


# --- real-data orchestration -------------------------------------------------

def load_2026_27_population(min_threshold: int = BASE_MIN_THRESHOLD) -> pd.DataFrame:
    return load_population(min_threshold=min_threshold, season_min=REVIEW_SEASON, season_max=REVIEW_SEASON)


def real_2026_27_data_available() -> bool:
    try:
        df = load_population(min_threshold=1, season_min=REVIEW_SEASON, season_max=REVIEW_SEASON)
        return len(df) > 0
    except Exception:
        return False


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    games_in_season = config.get("games_in_season_2026_27", 82)
    frozen = json.loads(FROZEN_PATH.read_text())
    nets_rookies = pd.read_csv(NETS_ROOKIES_PATH)

    if not real_2026_27_data_available():
        print(f"No {REVIEW_SEASON} data found in the local database yet - nothing to review. "
              f"This is expected if the season hasn't started/been pulled yet. Re-run this "
              f"script once season='{REVIEW_SEASON}' rows exist.")
        return

    feature_columns = json.loads((BASIS_DIR / "feature_order.json").read_text())["feature_columns"]
    population = load_2026_27_population(min_threshold=1)  # get everyone, filter per-rookie below
    std = build_standardization_for_population(
        population[population["MIN"] >= BASE_MIN_THRESHOLD], feature_columns)
    (STD_DIR / f"standardization_{REVIEW_SEASON}.json").write_text(json.dumps(std, indent=2))
    print(f"Wrote {STD_DIR / f'standardization_{REVIEW_SEASON}.json'} (n={std['n']})")

    basis = np.load(BASIS_DIR / "basis.npz")["basis"]
    train_df = pd.read_csv(TRAIN_PATH)
    y_cols = [f"y_{j}" for j in range(K)]
    baseline_recipe = train_df[y_cols].values.mean(axis=0)
    baseline_recipe = baseline_recipe / baseline_recipe.sum()

    results = []
    for rookie_cfg in config["rookies"]:
        name = rookie_cfg["display_name"]
        nba_pid = rookie_cfg["nba_player_id"]
        row = population[population["PLAYER_ID"] == nba_pid]

        if len(row) == 0:
            results.append({"display_name": name, "status": "unevaluable",
                             "reason": f"no {REVIEW_SEASON} row found at all (0 minutes recorded)"})
            continue

        actual_minutes = float(row.iloc[0]["MIN"])
        eligible, threshold = check_eligibility(actual_minutes, games_in_season)
        if not eligible:
            results.append({"display_name": name, "status": "unevaluable",
                             "reason": f"{actual_minutes:.0f} min < {threshold:.0f} min threshold "
                                       f"(the model answers 'what role if he plays', not 'will he play')"})
            continue

        y_actual = project(row, basis, np.array(std["mu"]), np.array(std["sd"]), feature_columns)[0]

        frozen_rookie = next(r for r in nets_rookies.to_dict("records") if r["display_name"] == name)
        y_predicted = np.array([frozen_rookie[f"y_pred_{j}"] for j in range(K)])

        scores = score_actual_vs_predicted(y_actual, y_predicted, baseline_recipe)

        import ast
        predicted_top3 = ast.literal_eval(frozen_rookie["y_top3"])
        actual_top3 = sorted(range(K), key=lambda j: -y_actual[j])[:3]
        saturated = bool(frozen_rookie["saturated"])
        saturation_borne_out = bool(y_actual.max() >= 0.85) if saturated else None

        comps = json.loads(frozen_rookie["comps"])
        comps_comparison = [{"name": c["name"], "his_actual_outcome": c["true_rookie_top2"],
                              "rookie_actual_top1": int(np.argmax(y_actual))} for c in comps]

        results.append({
            "display_name": name, "status": "evaluated", "actual_minutes": actual_minutes,
            "predicted_top3": predicted_top3, "actual_top3": actual_top3,
            "actual_argmax": int(np.argmax(y_actual)),
            "scores": scores, "saturated": saturated, "saturation_borne_out": saturation_borne_out,
            "comps_comparison": comps_comparison,
        })

    out_path = REPO_ROOT / "reports" / "phase6_2026_review_results.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nWrote {out_path}")
    for r in results:
        print(f"\n{r['display_name']}: {r['status']}")
        if r["status"] == "unevaluable":
            print(f"  {r['reason']}")
        else:
            print(f"  predicted top3={r['predicted_top3']}, actual top3={r['actual_top3']}")
            print(f"  JSD={r['scores']['jsd']:.4f}, top1_hit={r['scores']['top1_hit']}")


if __name__ == "__main__":
    main()
