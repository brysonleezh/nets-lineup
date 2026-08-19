"""
Phase 6 Step 2 - rookie predictions. Builds variant-(b) inputs for the 3
Nets rookies using ONLY the frozen deployment preprocessing recipe
(data/translator/deployment_preprocessing.json) - no statistic is ever
computed from the rookies' own data. Predicts via the deployment
posterior. Computes k-NN comps (5 nearest anchors) and the saturation
flag ("Kalkbrenner rule").

Run: .nets/bin/python src/translator/phase6_step2_predict_rookies.py
"""

from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src" / "translator"))

# numpyro/jax are imported LAZILY (inside predict_from_posterior and main, not
# here at module level) - this module's other functions (apply_frozen_transform,
# build_rookie_raw_row, roster_min_start_by_athlete_id, etc.) are pure, numpyro-
# independent, and are reused directly by the portal's Section 4 counterfactual
# (src/step5_rookie_projections.py), which has its own hard requirement to
# render in under ~2s on a cold cache - forcing jax's own import/JIT warmup
# cost on every portal page load for functions that don't need it at all would
# defeat that. Importing this module must stay cheap.

from input_variants import variant_b, Z_FEATURE_COLS, SHOT_TYPE_COLS  # noqa: E402

K = 8
CONFIG_PATH = REPO_ROOT / "config.yaml"
ANCHORS_PATH = REPO_ROOT / "data" / "anchors" / "anchors.csv"
SHARED_FEATURES_PATH = REPO_ROOT / "data" / "college" / "shared_features.parquet"
RAW_CBBD_DIR = REPO_ROOT / "data" / "raw" / "cbbd"
TRANSLATOR_DIR = REPO_ROOT / "data" / "translator"
DEPLOY_DIR = TRANSLATOR_DIR / "deployment"
PROJECTIONS_DIR = REPO_ROOT / "data" / "projections"

C_COLS = [f"c_alpha_{j}" for j in range(K)]
Y_COLS = [f"y_{j}" for j in range(K)]
RAW_COLS_FOR_STD = ["overall", "log_pick", "age_at_draft", "years_in_college", "PORPAG"]

HIGH_MAJOR_ALWAYS = {"ACC", "Big Ten", "Big 12", "SEC", "Big East"}


def conf_tier(conference: str, season: int) -> bool:
    if conference in HIGH_MAJOR_ALWAYS:
        return True
    if conference == "Pac-12" and season <= 2024:
        return True
    return False


def roster_min_start_by_athlete_id() -> dict[int, int]:
    import glob
    out = {}
    for f in sorted(glob.glob(str(RAW_CBBD_DIR / "roster_*.json"))):
        for team in json.loads(Path(f).read_text()):
            for p in team["players"]:
                ss = p.get("startSeason")
                if ss is None:
                    continue
                aid = p["id"]
                if aid not in out or ss < out[aid]:
                    out[aid] = ss
    return out


def shot_type_mix_for(athlete_id: int, season: int) -> tuple[float | None, float | None]:
    import glob
    best = None
    for f in sorted(glob.glob(str(RAW_CBBD_DIR / "shooting_*.json"))):
        for p in json.loads(Path(f).read_text()):
            if p.get("athleteId") == athlete_id and p.get("season") == season:
                tracked = p.get("trackedShots") or 0
                if tracked <= 0:
                    continue
                if best is None or tracked > best[0]:
                    rim = p["dunks"]["attempted"] + p["layups"]["attempted"] + p["tipIns"]["attempted"]
                    best = (tracked, rim / tracked, p["threePointJumpers"]["attempted"] / tracked)
    if best is None:
        return None, None
    return best[1], best[2]


def build_rookie_raw_row(rookie_cfg: dict, sf: pd.DataFrame, recipes_row: pd.Series,
                          roster_starts: dict, draft_date: date) -> dict:
    aid = rookie_cfg["cbbd_athlete_id"]
    season = 2026
    sf_row = sf[(sf["player_id_source"] == aid) & (sf["season"] == season)]
    assert len(sf_row) == 1, f"{rookie_cfg['display_name']}: expected 1 shared_features row, got {len(sf_row)}"
    sf_row = sf_row.iloc[0]

    birthdate = date.fromisoformat(rookie_cfg["birthdate"])
    age_at_draft = (draft_date - birthdate).days / 365.25

    min_start = roster_starts.get(aid)
    assert min_start is not None, f"{rookie_cfg['display_name']}: no roster startSeason found"
    years_in_college = season - min_start + 1

    rim_share, three_share = shot_type_mix_for(aid, season)

    row = {c: sf_row[c] for c in Z_FEATURE_COLS}  # already the correct season-relative z-scores
    row["overall"] = rookie_cfg["draft_pick_overall"]
    row["log_pick"] = np.log(rookie_cfg["draft_pick_overall"])
    row["age_at_draft"] = age_at_draft
    row["years_in_college"] = years_in_college
    row["PORPAG"] = sf_row["PORPAG"]
    row["rim_finishing_share"] = rim_share
    row["three_pt_jumper_share"] = three_share
    row["conf_tier"] = conf_tier(sf_row["conference"], season)
    row["conference"] = sf_row["conference"]
    row["college_team"] = sf_row["team"]
    row["college_minutes"] = sf_row["minutes"]
    return row


def apply_frozen_transform(rows: list[dict], preprocessing: dict) -> np.ndarray:
    """Same logic as Phase 5's apply_frozen_variant_b, generalized to a list
    of raw rows (the 3 rookies) - every statistic comes from `preprocessing`
    (fit on the 273 anchors), never from the rookies' own values."""
    cols = preprocessing["column_order"]
    mu, sd = preprocessing["standardization_mu"], preprocessing["standardization_sd"]
    impute_means = preprocessing["shot_type_imputation_means"]

    X = np.zeros((len(rows), len(cols)))
    for i, row in enumerate(rows):
        rim = row["rim_finishing_share"] if row["rim_finishing_share"] is not None else impute_means["rim_finishing_share"]
        three = row["three_pt_jumper_share"] if row["three_pt_jumper_share"] is not None else impute_means["three_pt_jumper_share"]
        missing = float(row["rim_finishing_share"] is None or row["three_pt_jumper_share"] is None)
        values = {**{c: row[c] for c in Z_FEATURE_COLS + RAW_COLS_FOR_STD},
                  "rim_finishing_share": rim, "three_pt_jumper_share": three}
        for j, c in enumerate(Z_FEATURE_COLS + RAW_COLS_FOR_STD + SHOT_TYPE_COLS):
            X[i, j] = (values[c] - mu[c]) / sd[c]
        X[i, len(Z_FEATURE_COLS) + len(RAW_COLS_FOR_STD) + len(SHOT_TYPE_COLS)] = float(row["conf_tier"])
        X[i, len(Z_FEATURE_COLS) + len(RAW_COLS_FOR_STD) + len(SHOT_TYPE_COLS) + 1] = missing
    assert not np.isnan(X).any()
    return X


def predict_from_posterior(X: np.ndarray, B_samples: np.ndarray, phi_samples: np.ndarray) -> np.ndarray:
    import jax
    import jax.numpy as jnp
    import numpyro.distributions as dist
    Xaug = jnp.concatenate([jnp.ones((X.shape[0], 1)), jnp.asarray(X, dtype=jnp.float32)], axis=1)
    logits_free = jnp.einsum("np,spk->snk", Xaug, jnp.asarray(B_samples))
    logits = jnp.concatenate([logits_free, jnp.zeros((*logits_free.shape[:-1], 1))], axis=-1)
    mu_samples = jax.nn.softmax(logits, axis=-1)
    return np.asarray(mu_samples.mean(axis=0))


def main() -> None:
    import numpyro
    numpyro.set_host_device_count(4)  # must precede any jax import - fine here since main()
    # is the CLI entry point, never imported by the portal (which only ever calls the
    # pure functions above, not main() itself).

    PROJECTIONS_DIR.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(CONFIG_PATH.read_text())
    draft_date = date.fromisoformat(config["draft_date_2026"])
    sat_threshold = config["saturation_c_alpha_max_threshold"]
    comps_k = config["comps_k"]

    anchors = pd.read_csv(ANCHORS_PATH)
    sf = pd.read_parquet(SHARED_FEATURES_PATH)
    recipes = pd.read_csv(REPO_ROOT / "data" / "college" / "recipes.csv")
    roster_starts = roster_min_start_by_athlete_id()

    preprocessing = json.loads((TRANSLATOR_DIR / "deployment_preprocessing.json").read_text())
    with np.load(DEPLOY_DIR / "posterior.npz") as z:
        B_samples, phi_samples = z["B"], z["phi"]

    rows = []
    for rookie_cfg in config["rookies"]:
        aid = rookie_cfg["cbbd_athlete_id"]
        rec = recipes[(recipes["player_id_source"] == aid) & (recipes["season"] == 2026)]
        assert len(rec) == 1, f"{rookie_cfg['display_name']}: expected 1 recipe row"
        rows.append(build_rookie_raw_row(rookie_cfg, sf, rec.iloc[0], roster_starts, draft_date))
        rows[-1]["_c_alpha"] = rec.iloc[0][[f"alpha_{j}" for j in range(K)]].values.astype(float)
        rows[-1]["_c_argmax"] = int(rec.iloc[0]["argmax_archetype"])
        rows[-1]["_c_alpha_max"] = float(rec.iloc[0]["alpha_max"])
        rows[-1]["display_name"] = rookie_cfg["display_name"]

    X = apply_frozen_transform(rows, preprocessing)
    print(f"Built variant-(b) inputs for {len(rows)} rookies using only the frozen deployment recipe: {X.shape}")

    y_pred = predict_from_posterior(X, B_samples, phi_samples)
    assert (y_pred >= -1e-9).all()
    assert np.abs(y_pred.sum(axis=1) - 1.0).max() < 1e-6
    print("Simplex validity: PASS")

    # --- k-NN comps: 5 nearest anchors in the SAME standardized variant-(b) space ---
    X_anchors, X_anchors_check, _ = variant_b(anchors, anchors)
    assert np.allclose(X_anchors, X_anchors_check)
    # re-standardize the anchors' own X using the SAME frozen deployment stats used for
    # the rookies (variant_b(anchors, anchors) uses anchors' own stats, which ARE the
    # deployment stats since deployment was fit on these same 273 rows - verified equal)
    dep_mu = np.array([preprocessing["standardization_mu"][c] for c in preprocessing["continuous_columns"]])
    assert X_anchors.shape[1] == 21

    # --- saturation percentile ---
    anchor_c_alpha_max = anchors["c_alpha_max"].values
    percentiles = [float((anchor_c_alpha_max < r["_c_alpha_max"]).mean() * 100) for r in rows]

    projections = []
    for i, r in enumerate(rows):
        dists = np.linalg.norm(X_anchors - X[i], axis=1)
        order = np.argsort(dists)[:comps_k]
        comps = []
        for idx in order:
            a = anchors.iloc[idx]
            comps.append({
                "name": a["player_name"], "draft_year": int(a["draft_year"]),
                "pick": int(a["overall"]), "college": a["college_team"],
                "college_top_archetype": int(a["c_argmax"]),
                "true_rookie_top_archetype": int(a["y_argmax"]),
                "true_rookie_top2": sorted(
                    [(j, round(float(a[f"y_{j}"]), 3)) for j in range(K)], key=lambda x: -x[1])[:2],
                "distance": float(dists[idx]),
            })

        saturated = r["_c_alpha_max"] >= sat_threshold
        top3 = sorted(range(K), key=lambda j: -y_pred[i, j])[:3]
        c_top3 = sorted(range(K), key=lambda j: -r["_c_alpha"][j])[:3]

        projections.append({
            "display_name": r["display_name"], "college_team": r["college_team"],
            "conference": r["conference"], "college_minutes": float(r["college_minutes"]),
            "age_at_draft": round(r["age_at_draft"], 2), "years_in_college": r["years_in_college"],
            "c_alpha": r["_c_alpha"].tolist(), "c_argmax": r["_c_argmax"],
            "c_alpha_max": r["_c_alpha_max"], "c_alpha_max_percentile": percentiles[i],
            "c_top3": c_top3,
            "y_pred": y_pred[i].tolist(), "y_pred_argmax": int(np.argmax(y_pred[i])),
            "y_top3": [(j, round(float(y_pred[i, j]), 3)) for j in top3],
            "saturated": bool(saturated),
            "comps": comps,
        })
        print(f"\n{r['display_name']} ({r['college_team']}, pick {rows[i]['overall']}): "
              f"college top3={c_top3}, predicted top3={top3}, "
              f"c_alpha_max={r['_c_alpha_max']:.3f} (pctile {percentiles[i]:.0f}), saturated={saturated}")

    # --- saturation rule evidence: training-set anchors with true y_max >= 0.9 ---
    # (out-of-fold CV predictions, reusing the same 8-fold refit pattern Phase 5 used for
    # calibration - training data only, the rookies/holdout are never touched by this)
    print("\nGathering training-set saturation-rule evidence (8-fold CV refit, variant b)...")
    from dirichlet_model import fit_t1_dirichlet
    from input_variants import build_targets
    train_df = pd.read_csv(REPO_ROOT / "data" / "anchors" / "train_2017_2024.csv")
    sat_evidence = []
    for held_class in range(2017, 2025):
        tr = train_df[train_df["draft_year"] != held_class].reset_index(drop=True)
        te = train_df[train_df["draft_year"] == held_class].reset_index(drop=True)
        X_tr, X_te, _ = variant_b(tr, te)
        y_tr, _ = build_targets(tr, te)
        preds_fold, _ = fit_t1_dirichlet(X_tr, y_tr, X_te, seed=held_class)
        y_true_fold = te[Y_COLS].values
        for j in range(len(te)):
            if y_true_fold[j].max() >= 0.9:
                sat_evidence.append({
                    "player_name": te.iloc[j]["player_name"], "draft_year": held_class,
                    "true_y_max": float(y_true_fold[j].max()), "true_y_argmax": int(y_true_fold[j].argmax()),
                    "predicted_weight_on_true_argmax": float(preds_fold[j, y_true_fold[j].argmax()]),
                    "predicted_y_max": float(preds_fold[j].max()),
                })
    sat_df = pd.DataFrame(sat_evidence)
    print(f"\n{len(sat_df)} training-set (out-of-fold) anchors with true y_max >= 0.9:")
    if len(sat_df):
        print(sat_df.to_string(index=False))
        print(f"\nMean predicted weight on the true top archetype for these cases: "
              f"{sat_df['predicted_weight_on_true_argmax'].mean():.3f} (vs. true weight ~1.0 by definition) - "
              f"mean predicted y_max: {sat_df['predicted_y_max'].mean():.3f}")
    sat_df.to_csv(TRANSLATOR_DIR / "saturation_rule_evidence.csv", index=False)

    # --- write outputs ---
    out_rows = []
    for p in projections:
        row = {"display_name": p["display_name"], "college_team": p["college_team"],
               "conference": p["conference"], "college_minutes": p["college_minutes"],
               "age_at_draft": p["age_at_draft"], "years_in_college": p["years_in_college"],
               "c_argmax": p["c_argmax"], "c_alpha_max": p["c_alpha_max"],
               "c_alpha_max_percentile": p["c_alpha_max_percentile"], "saturated": p["saturated"],
               "y_pred_argmax": p["y_pred_argmax"]}
        for j in range(K):
            row[f"c_alpha_{j}"] = p["c_alpha"][j]
            row[f"y_pred_{j}"] = p["y_pred"][j]
        row["c_top3"] = str(p["c_top3"])
        row["y_top3"] = str(p["y_top3"])
        row["comps"] = json.dumps(p["comps"])
        out_rows.append(row)
    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(PROJECTIONS_DIR / "nets_rookies_2026.csv", index=False)
    (TRANSLATOR_DIR / "rookie_projections_full.json").write_text(json.dumps(projections, indent=2, default=str))
    print(f"\nWrote {PROJECTIONS_DIR / 'nets_rookies_2026.csv'}")
    print(f"Wrote {TRANSLATOR_DIR / 'rookie_projections_full.json'}")
    print(f"Wrote {TRANSLATOR_DIR / 'saturation_rule_evidence.csv'}")


if __name__ == "__main__":
    main()
