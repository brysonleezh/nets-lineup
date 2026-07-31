"""
score_lineup.py - reusable scoring for a fitted skill-weighted archetype-RAPM
model (see fit_rapm.py / RAPM_README.md).

AI-ASSISTED (Claude Code, chat)
Prompt: part of the same fit_rapm.py spec - "src/score_lineup.py: a reusable
function that, given 5 offense player_ids and either 5 defense player_ids OR
'league-average', plus a coefficients json, returns predicted ORtg, DRtg
(lineup as defense vs league-avg offense), and Net = ORtg - DRtg. It must
rebuild features the same way and apply the stored scaler."

Used: reuses `fit_rapm.build_features_single` for feature construction (the
exact same code path Stage 1 uses to build the training design matrix) and
the coefficients JSON's own stored scaler_mean/scaler_std - never
re-derives either, so a lineup's predicted rating is always computed under
the identical transform the model was actually fit on. DRtg's definition is
fixed ("this lineup on defense vs. league-average offense") regardless of
what `defense` is passed for the ORtg side - matching the prompt's own
wording literally, not made configurable for something the prompt didn't ask for.

Not AI: the ORtg/DRtg/Net definition itself - specified directly.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import fit_rapm as fr

REPO_ROOT = Path(__file__).resolve().parent.parent


def _predict_y_off(x_raw: np.ndarray, model: dict) -> float:
    """Applies the model's OWN stored scaler + coefficients - never a
    freshly-computed mean/std, which would silently diverge from what the
    model was actually trained on if the underlying data changes at all."""
    mean = np.array(model["scaler_mean"])
    std = np.array(model["scaler_std"])
    xz = (x_raw - mean) / std
    k = model["k"]
    pairs = fr._pair_indices(k)
    beta = (
        [model["d_off"], model["d_sdef"], model["h_home"]]
        + model["Boff"] + model["Bdef"]
        + [model["Goff"][f"{i}_{j}"] for i, j in pairs]
        + [model["Gdef"][f"{i}_{j}"] for i, j in pairs]
    )
    return float(model["intercept"] + xz @ np.array(beta))


def _league_average_vectors(model: dict):
    """'league_average' isn't a real 5 players - it's the possession-
    weighted mean Eoff/Edef/Soff/Sdef vector saved alongside the model (see
    fit_rapm.save_model). Archetype shares can't be un-averaged back into 5
    synthetic individuals, so scoring "vs. league-average" means building
    the feature row directly from this saved vector, not from any 5 ids."""
    la = model["league_average"]
    return np.array(la["Soff"]), np.array(la["Sdef"]), np.array(la["Eoff"]), np.array(la["Edef"])


def score_lineup(off_ids, defense, model: dict, alpha: pd.DataFrame, skill: pd.DataFrame, k: int,
                 home_off: float = 0.5) -> dict:
    """
    off_ids: 5 player_ids - the lineup being scored.
    defense: either 5 player_ids, or the literal string "league_average" -
      governs the ORtg calculation only (DRtg is always vs. league-average
      offense, a fixed definition per the prompt's own wording).
    model: a loaded coefficients_*.json dict (see fit_rapm.save_model).
    home_off: court context for both predictions (default 0.5 - neutral,
      since a lineup has no fixed home/away identity of its own).

    Returns {"ORtg", "DRtg", "Net"}:
      ORtg = predicted y_off with `off_ids` on OFFENSE vs. `defense`
      DRtg = predicted y_off with `off_ids` on DEFENSE vs. league-average offense
             (points the average offense is expected to score against them)
      Net  = ORtg - DRtg
    """
    Soff_la, Sdef_la, Eoff_la, Edef_la = _league_average_vectors(model)
    pairs = fr._pair_indices(k)
    offpairs_la = np.array([Eoff_la[i] * Eoff_la[j] for i, j in pairs]) if pairs else np.zeros(0)
    defpairs_la = np.array([Edef_la[i] * Edef_la[j] for i, j in pairs]) if pairs else np.zeros(0)

    # One call gives both the offense-computed (skill_off-weighted) AND
    # defense-computed (skill_def-weighted) feature slices for THIS SAME 5
    # players, since build_features_single always derives Soff/Eoff/offpairs
    # from its first (offense) argument and Sdef/Edef/defpairs from its
    # second (defense) argument - passing off_ids into both slots is the
    # simplest way to get "this lineup, viewed both ways" in one call.
    x_self = fr.build_features_single(off_ids, off_ids, home_off, alpha, skill, k)

    # --- ORtg: off_ids on offense vs. `defense` ---
    if defense == "league_average":
        x_ortg = x_self.copy()
        x_ortg[1] = Sdef_la
        x_ortg[3 + k:3 + 2 * k] = Edef_la
        x_ortg[3 + 2 * k + len(pairs):] = defpairs_la
    else:
        x_ortg = fr.build_features_single(off_ids, defense, home_off, alpha, skill, k)
    ortg = _predict_y_off(x_ortg, model)

    # --- DRtg: off_ids on defense vs. league-average offense (fixed) ---
    x_drtg = x_self.copy()
    x_drtg[0] = Soff_la
    x_drtg[3:3 + k] = Eoff_la
    x_drtg[3 + 2 * k:3 + 2 * k + len(pairs)] = offpairs_la
    drtg = _predict_y_off(x_drtg, model)

    return {"ORtg": ortg, "DRtg": drtg, "Net": ortg - drtg}


def load_model(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found - run fit_rapm.py first")
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, required=True, help="path to a coefficients_*.json")
    p.add_argument("--off", nargs=5, type=int, required=True, help="5 offense player_ids")
    p.add_argument("--def", dest="defense", nargs="+", type=str, required=True,
                   help="5 defense player_ids, or the single word league_average")
    p.add_argument("--alpha-path", type=Path, default=fr.DEFAULT_ALPHA_PATH)
    p.add_argument("--skill-path", type=Path, default=fr.DEFAULT_SKILL_PATH,
                   help="build_skill.py's output - covers rookies, unlike the legacy "
                        "--skill-season DB-only lookup")
    p.add_argument("--skill-season", default=None,
                   help="LEGACY: if set, ignores --skill-path and uses the old "
                        "prior-season-BPM-only lookup instead")
    p.add_argument("--stint-season", default="2025-26")
    args = p.parse_args()

    model = load_model(args.model)
    alpha, k = fr.load_alpha(args.alpha_path)
    if args.skill_season is not None:
        skill = fr.load_skill(args.skill_season, {args.stint_season})
    else:
        skill = fr.load_skill_file(args.skill_path, {args.stint_season})

    defense = "league_average" if args.defense == ["league_average"] else [int(x) for x in args.defense]
    result = score_lineup(args.off, defense, model, alpha, skill, k)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
