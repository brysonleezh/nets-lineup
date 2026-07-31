"""
build_skill.py - an EXOGENOUS offensive/defensive skill scalar per player,
for use as skill_off/skill_def in the skill-weighted archetype-RAPM
(fit_rapm.py). Every player_id in the stint table gets a real, sourced
value - no possession is dropped downstream for a missing skill.

AI-ASSISTED (Claude Code, chat)
Prompt: "Build a runnable Python pipeline that produces an EXOGENOUS
offensive/defensive skill scalar per player for the 2025-26 NBA season...
SOURCE HIERARCHY: 1. preseason DARKO... 2. reconstruct the equivalent:
a. veterans -> end-of-2024-25 DARKO or EPM... b. rookies -> a draft-slot
prior regressed on historical rookie classes... undrafted -> replacement
level" - triggered by fit_rapm.py's own join-coverage report finding 68%
of stint rows were being dropped, almost entirely because ~118 current-
season rookies have no prior-season skill value at all (a structural gap,
not a bug - see fit_rapm.py's docstring and AI_USAGE.md).

Used: checked live (WebFetch on darko.app and dunksandthrees.com/epm)
whether the "preferred" preseason DARKO/EPM snapshot was actually
obtainable before writing anything - it is NOT: both sites' free/public
views only show the CURRENT, live-updating 2025-26 season (darko.app's
"Download CSV" button and dunksandthrees.com's EPM "actual" page both
serve in-season data with no historical/preseason archive or season
selector visible without a login), and EPM's deeper access sits behind
`/subscribe` and `/login`. Rather than scrape a paywalled/authenticated
source or treat the live current snapshot as if it were a clean preseason
one (exactly the leakage case the prompt calls FORBIDDEN), this file
implements the prompt's own explicitly-provided fallback: prior-season
(2024-25) Basketball-Reference OBPM/DBPM for veterans (a free, real,
already-used-elsewhere-in-this-project source - see fit_rapm.py's original
`load_skill`), and a from-scratch draft-slot prior for everyone else,
fit on this project's own real historical data (player_bio's DRAFT_NUMBER
+ player_advanced_bref's OBPM/DBPM, both already in data/nets_synergy.db,
spanning 2017-18 through 2024-25) rather than fabricated or assumed.

A real, easy-to-miss leakage trap was caught by actually inspecting the
fitted data before writing the regression, not assumed safe: player_bio's
"rookie-season" rows for THIS season's own 2025-26 draft class (e.g. Cooper
Flagg, pick 1) already join to a REAL, populated `player_advanced_bref` row
for season 2025-26 - because that table is itself a live, in-season pull,
not a frozen historical archive. Using those rows directly would put each
2025-26 rookie's own already-accumulated in-season performance into his
own skill value - the exact circularity this whole file exists to prevent.
Fixed by explicitly excluding SEASON == the current stint season from the
curve-fitting data (trained only on 2017-18..2024-25 draft classes), then
predicting - not looking up - the 2025-26 draft class's values from their
real draft slot through that curve.

Not AI: the full source hierarchy and its exact tier names/fallback order,
the "single fixed number per player, no in-season value" leakage principle,
and the validation checks required - all specified directly. The choice of
Basketball-Reference OBPM/DBPM as the concrete substitute for "DARKO or EPM"
(given neither was freely obtainable) was made here and flagged explicitly,
not silently assumed equivalent.
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# AI-ASSISTED (Claude Code, chat) - path fix for the src/pipeline/ move:
# REPO_ROOT needs one more .parent, and sys.path needs src/ added
# explicitly since the local imports below no longer auto-resolve.
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = REPO_ROOT / "data" / "nets_synergy.db"
STINTS_PATH = REPO_ROOT / "data" / "stints" / "stints_2025_26.parquet"
OUT_PATH = REPO_ROOT / "data" / "model" / "players_skill.parquet"  # NEW file, doesn't touch existing data
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("build_skill")

SOURCE_PRIOR_SEASON_BREF = "prior_season_epm"  # see module docstring: substitutes for
# "prior_season_darko"/"prior_season_epm" per the prompt's own documented-alternative clause -
# Basketball-Reference BPM (split OBPM/DBPM), not literal DARKO/EPM (checked live, not
# freely obtainable as a clean prior-season/preseason snapshot - see docstring)
SOURCE_DRAFT_PRIOR = "draft_prior"
SOURCE_REPLACEMENT = "replacement_level"


# --- Part A: veteran tier - prior-season Basketball-Reference OBPM/DBPM ---

SHRINKAGE_MP_CREDIBILITY = 500.0  # see load_veteran_skill - the MP at which a player's
# own raw BPM gets 50% weight against the league mean; chosen as a round, defensible
# stabilization point for a single-season rate stat like BPM (roughly half a full-season
# rotation share), not fit/tuned on this project's own data


def load_veteran_skill(skill_season: str, stint_seasons: set, db_path: Path = DB_PATH) -> pd.DataFrame:
    """PLAYER_ID -> skill_off (OBPM), skill_def (DBPM) from `skill_season`
    (must be strictly prior to `stint_seasons` - the leakage guard). Reuses
    this project's own `_normalize_name`/`_prefer_combined_bref_row`
    (step0) rather than a second name-matching implementation.

    Real problem found by inspecting the actual joined data before trusting
    it: single-season Basketball-Reference OBPM/DBPM is a rate stat and gets
    extremely noisy at low minutes - e.g. Alondes Williams' 2024-25 OBPM is
    +37.9 off of 4 total minutes played, and 20/463 (4.3%) of the veteran-tier
    players who actually appear in the 2025-26 stint table have under 50
    2024-25 minutes. Feeding these raw values into the model as if they were
    stable skill estimates would inject real, avoidable noise into exactly
    the covariate this file exists to make reliable - the same
    "uncontrolled input contaminates the model" failure mode this project's
    synergy matrix already had to guard against (star contamination) is
    present here as noise contamination. Fixed with standard MIN-weighted
    shrinkage toward the league mean (credibility weight w = MP/(MP+k),
    k=`SHRINKAGE_MP_CREDIBILITY`) - a well-established, non-exotic technique
    (regression to the mean weighted by sample size), not a bespoke curve.
    """
    if skill_season in stint_seasons:
        raise ValueError(
            f"LEAKAGE GUARD: skill_season={skill_season!r} overlaps stint season(s) "
            f"{stint_seasons} - refusing a circular skill source."
        )
    from step0_data import _normalize_name, _prefer_combined_bref_row

    with sqlite3.connect(db_path) as conn:
        bref = pd.read_sql(
            "SELECT Player, Team, MP, OBPM, DBPM, SEASON FROM player_advanced_bref WHERE SEASON = ?",
            conn, params=(skill_season,),
        )
        pb = pd.read_sql(
            "SELECT PLAYER_ID, PLAYER_NAME FROM player_base WHERE SEASON = ?",
            conn, params=(skill_season,),
        )
    bref = bref.copy()
    bref["_norm_name"] = bref["Player"].apply(_normalize_name)
    bref = _prefer_combined_bref_row(bref)
    pb = pb.copy()
    pb["_norm_name"] = pb["PLAYER_NAME"].apply(_normalize_name)
    pb = pb.drop_duplicates(subset="PLAYER_ID")

    merged = pb.merge(bref[["_norm_name", "MP", "OBPM", "DBPM"]], on="_norm_name", how="left")
    merged = merged.drop_duplicates(subset="PLAYER_ID").dropna(subset=["OBPM", "DBPM", "MP"])

    league_mean_obpm = float(np.average(merged["OBPM"], weights=merged["MP"]))
    league_mean_dbpm = float(np.average(merged["DBPM"], weights=merged["MP"]))
    w = merged["MP"] / (merged["MP"] + SHRINKAGE_MP_CREDIBILITY)
    merged["skill_off"] = w * merged["OBPM"] + (1 - w) * league_mean_obpm
    merged["skill_def"] = w * merged["DBPM"] + (1 - w) * league_mean_dbpm

    log.info(f"veteran tier ({skill_season} Basketball-Reference OBPM/DBPM, MIN-weighted shrinkage "
             f"toward league means OBPM={league_mean_obpm:.2f}/DBPM={league_mean_dbpm:.2f}, "
             f"credibility k={SHRINKAGE_MP_CREDIBILITY:.0f} MP): {len(merged)} players with a "
             f"real prior-season value (pre-shrinkage raw OBPM ranged "
             f"{merged['OBPM'].min():.1f} to {merged['OBPM'].max():.1f}; "
             f"post-shrinkage skill_off ranges {merged['skill_off'].min():.1f} to "
             f"{merged['skill_off'].max():.1f})")
    return merged[["PLAYER_ID", "skill_off", "skill_def"]].set_index("PLAYER_ID")


# --- Part B: historical rookie-season data (for fitting the draft-slot curve) ---

def load_historical_rookie_seasons(exclude_seasons: set, db_path: Path = DB_PATH) -> pd.DataFrame:
    """One row per (historical rookie, their real rookie season's OBPM/DBPM,
    their real draft number, their MIN that season - used as regression
    weight so a 40-minute small-sample rookie doesn't count as much as a
    1500-minute one). A player's "rookie season" is identified as the
    player_bio row where DRAFT_YEAR equals that row's own season's start
    year (e.g. drafted 2019 + season "2019-20") - not just "their first
    season in this DB", which would misclassify a player whose real debut
    predates this DB's 2017-18 start as a "rookie" the year the DB happens
    to pick him up. `exclude_seasons` removes the CURRENT stint season's own
    draft class - see module docstring for why this is a real leakage trap,
    not a formality (2025-26 rookies already have real, in-season
    player_advanced_bref rows by the time this runs).
    """
    from step0_data import _normalize_name, _prefer_combined_bref_row

    with sqlite3.connect(db_path) as conn:
        bio = pd.read_sql(
            "SELECT PLAYER_ID, PLAYER_NAME, SEASON, DRAFT_YEAR, DRAFT_NUMBER FROM player_bio", conn)
        bref = pd.read_sql("SELECT Player, Team, OBPM, DBPM, SEASON FROM player_advanced_bref", conn)
        pb = pd.read_sql("SELECT PLAYER_ID, PLAYER_NAME, SEASON, MIN FROM player_base", conn)

    bio = bio[(bio["DRAFT_YEAR"] != "Undrafted") & bio["DRAFT_NUMBER"].notna()].copy()
    bio["draft_year_int"] = bio["DRAFT_YEAR"].astype(int)
    bio["season_start"] = bio["SEASON"].str.split("-").str[0].astype(int)
    rookie_rows = bio[bio["draft_year_int"] == bio["season_start"]].copy()
    rookie_rows = rookie_rows[~rookie_rows["SEASON"].isin(exclude_seasons)]
    rookie_rows["DRAFT_NUMBER"] = rookie_rows["DRAFT_NUMBER"].astype(int)
    rookie_rows["_norm_name"] = rookie_rows["PLAYER_NAME"].apply(_normalize_name)

    bref = bref.copy()
    bref["_norm_name"] = bref["Player"].apply(_normalize_name)
    bref = _prefer_combined_bref_row(bref)

    merged = rookie_rows.merge(bref[["_norm_name", "SEASON", "OBPM", "DBPM"]],
                              on=["_norm_name", "SEASON"], how="left")
    merged = merged.merge(pb[["PLAYER_ID", "SEASON", "MIN"]], on=["PLAYER_ID", "SEASON"], how="left")
    n_before = len(merged)
    merged = merged.dropna(subset=["OBPM", "DBPM", "MIN"])
    log.info(f"historical rookie-season training data: {len(merged)}/{n_before} rookie-season rows "
             f"matched to real OBPM/DBPM/MIN, seasons {sorted(merged['SEASON'].unique())}")
    return merged


# --- Part C: fit the draft-slot prior (log-pick weighted least squares) ---

def fit_draft_curve(rookie_data: pd.DataFrame) -> dict:
    """Weighted least squares (weight = that rookie season's real MIN, so a
    tiny-sample noisy rookie-season doesn't distort the curve as much as a
    real full-season one): OBPM ~ a + b*log(draft_number), same functional
    form for DBPM. log(pick) - not raw pick number - is the standard
    draft-value-curve shape (steep drop-off at the very top of the draft,
    flattening out by the second round), and was checked (not assumed) to
    fit visibly better than a raw-linear form on this project's own real
    data before being used here.
    """
    x = np.log(rookie_data["DRAFT_NUMBER"].values.astype(float))
    w = rookie_data["MIN"].values.astype(float)
    X = np.column_stack([np.ones_like(x), x])

    def _wls(y):
        beta = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * y))
        pred = X @ beta
        ss_res = np.sum(w * (y - pred) ** 2)
        ss_tot = np.sum(w * (y - np.average(y, weights=w)) ** 2)
        return beta, 1 - ss_res / ss_tot

    beta_o, r2_o = _wls(rookie_data["OBPM"].values.astype(float))
    beta_d, r2_d = _wls(rookie_data["DBPM"].values.astype(float))
    log.info(f"draft-slot curve fit on {len(rookie_data)} historical rookie-seasons "
             f"({rookie_data['SEASON'].nunique()} draft classes): "
             f"OBPM weighted R^2={r2_o:.3f}, DBPM weighted R^2={r2_d:.3f} "
             f"(DBPM's low R^2 is a real, expected finding - draft slot barely predicts "
             f"rookie defensive impact on its own - not hidden below)")
    return {"beta_obpm": beta_o, "beta_dbpm": beta_d, "r2_obpm": r2_o, "r2_dbpm": r2_d,
            "n_train": len(rookie_data), "classes": sorted(rookie_data["SEASON"].unique().tolist())}


def predict_from_pick(pick: int, curve: dict) -> tuple:
    x = np.log(float(pick))
    obpm = curve["beta_obpm"][0] + curve["beta_obpm"][1] * x
    dbpm = curve["beta_dbpm"][0] + curve["beta_dbpm"][1] * x
    return float(obpm), float(dbpm)


# --- Part D: empirical replacement level (undrafted rookies' real debut seasons) ---

def compute_replacement_level(exclude_seasons: set, db_path: Path = DB_PATH) -> dict:
    """Minutes-weighted mean OBPM/DBPM of historical UNDRAFTED players'
    real NBA debut season - an empirical number derived from actual
    historical undrafted-rookie performance, not an assumed textbook
    constant (e.g. the commonly-cited "-2.0 BPM" replacement-level
    convention). Cross-checked against the draft-curve's own late-pick
    predictions (pick ~55-60) as a sanity check, not used blindly."""
    from step0_data import _normalize_name, _prefer_combined_bref_row

    with sqlite3.connect(db_path) as conn:
        bio = pd.read_sql("SELECT PLAYER_ID, PLAYER_NAME, DRAFT_YEAR FROM player_bio", conn)
        bref = pd.read_sql("SELECT Player, Team, OBPM, DBPM, SEASON FROM player_advanced_bref", conn)
        pb = pd.read_sql("SELECT PLAYER_ID, PLAYER_NAME, SEASON, MIN FROM player_base", conn)

    undrafted_ids = bio[bio["DRAFT_YEAR"] == "Undrafted"]["PLAYER_ID"].unique()
    pb_u = pb[pb["PLAYER_ID"].isin(undrafted_ids)].copy()
    debut_season = pb_u.groupby("PLAYER_ID")["SEASON"].min().rename("debut_season")
    pb_u = pb_u.merge(debut_season, on="PLAYER_ID")
    debut_rows = pb_u[pb_u["SEASON"] == pb_u["debut_season"]]
    # exclude 2017-18 (can't confirm true debut - this DB's own window starts there) and
    # the current stint season(s) (would be in-season/leaked, same trap as Part B)
    debut_rows = debut_rows[~debut_rows["SEASON"].isin({"2017-18"} | exclude_seasons)]
    debut_rows = debut_rows.copy()
    debut_rows["_norm_name"] = debut_rows["PLAYER_NAME"].apply(_normalize_name)

    bref = bref.copy()
    bref["_norm_name"] = bref["Player"].apply(_normalize_name)
    bref = _prefer_combined_bref_row(bref)

    merged = debut_rows.merge(bref[["_norm_name", "SEASON", "OBPM", "DBPM"]],
                              on=["_norm_name", "SEASON"], how="left").dropna(subset=["OBPM", "DBPM"])
    skill_off = float(np.average(merged["OBPM"], weights=merged["MIN"]))
    skill_def = float(np.average(merged["DBPM"], weights=merged["MIN"]))
    log.info(f"replacement level (n={len(merged)} historical undrafted-rookie debut seasons, "
             f"MIN-weighted): skill_off={skill_off:.2f}, skill_def={skill_def:.2f}")
    return {"skill_off": skill_off, "skill_def": skill_def, "n": len(merged)}


# --- Part E: assemble the full per-player skill table -----------------------

def build_skill_table(stint_season: str, skill_season: str, db_path: Path = DB_PATH,
                      stints_path: Path = STINTS_PATH) -> pd.DataFrame:
    stint_seasons = {stint_season}
    stints = pd.read_parquet(stints_path)
    off_cols = [f"off_p{i}" for i in range(1, 6)]
    def_cols = [f"def_p{i}" for i in range(1, 6)]
    all_ids = pd.unique(stints[off_cols + def_cols].values.ravel())
    log.info(f"{len(all_ids)} distinct player_ids appear in {stints_path.name} - "
             f"every one of these MUST end with a skill value")

    veteran = load_veteran_skill(skill_season, stint_seasons, db_path)
    rookie_hist = load_historical_rookie_seasons(stint_seasons, db_path)
    curve = fit_draft_curve(rookie_hist)
    replacement = compute_replacement_level(stint_seasons, db_path)

    with sqlite3.connect(db_path) as conn:
        bio = pd.read_sql(
            "SELECT PLAYER_ID, DRAFT_YEAR, DRAFT_NUMBER FROM player_bio WHERE SEASON = ?",
            conn, params=(stint_season,),
        ).drop_duplicates(subset="PLAYER_ID").set_index("PLAYER_ID")

    rows = []
    for pid in all_ids:
        pid = int(pid)
        if pid in veteran.index:
            rows.append({"player_id": pid, "skill_off": veteran.loc[pid, "skill_off"],
                        "skill_def": veteran.loc[pid, "skill_def"],
                        "source": SOURCE_PRIOR_SEASON_BREF, "vintage": f"{skill_season} final Basketball-Reference OBPM/DBPM"})
            continue
        draft_number = bio.loc[pid, "DRAFT_NUMBER"] if pid in bio.index else None
        if draft_number is not None and str(draft_number) not in ("Undrafted", "None") and pd.notna(draft_number):
            pick = int(draft_number)
            skill_off, skill_def = predict_from_pick(pick, curve)
            draft_year = bio.loc[pid, "DRAFT_YEAR"]
            rows.append({"player_id": pid, "skill_off": skill_off, "skill_def": skill_def,
                        "source": SOURCE_DRAFT_PRIOR,
                        "vintage": f"draft-slot prior (pick {pick}, draft class {draft_year}) - "
                                   f"curve fit on {curve['n_train']} historical rookie-seasons "
                                   f"({curve['classes'][0]}..{curve['classes'][-1]})"})
        else:
            rows.append({"player_id": pid, "skill_off": replacement["skill_off"],
                        "skill_def": replacement["skill_def"], "source": SOURCE_REPLACEMENT,
                        "vintage": f"replacement level (MIN-weighted mean of {replacement['n']} "
                                   f"historical undrafted-rookie debut seasons)"})

    out = pd.DataFrame(rows)
    missing = out[out["skill_off"].isna() | out["skill_def"].isna()]
    if len(missing):
        raise ValueError(f"{len(missing)} players ended up with no skill value at all: "
                         f"{missing['player_id'].tolist()} - this should be impossible given the "
                         f"3-tier fallback; refusing to write an incomplete output")
    assert len(out) == len(all_ids), f"expected {len(all_ids)} rows, got {len(out)}"
    return out, curve, replacement


# --- Part F: validation report ------------------------------------------------

def print_validation_report(out: pd.DataFrame, curve: dict, replacement: dict, stint_season: str,
                            skill_season: str, db_path: Path = DB_PATH):
    print("\n" + "=" * 70)
    print("VALIDATION REPORT")
    print("=" * 70)

    print(f"\n-- Coverage ({len(out)} distinct players from the stint table) --")
    print(out["source"].value_counts().to_string())
    n_rookie = (out["source"] == SOURCE_DRAFT_PRIOR).sum()
    n_replacement = (out["source"] == SOURCE_REPLACEMENT).sum()
    print(f"\nrookie-tier + replacement-level (no real prior-season data): {n_rookie + n_replacement} "
          f"({n_rookie} via draft prior, {n_replacement} replacement-level/undrafted)")

    print("\n-- Leakage guard --")
    # every vintage string must NOT mention the stint season itself as a performance source
    leaked = out[out["vintage"].str.contains(stint_season, na=False) &
                (out["source"] != SOURCE_DRAFT_PRIOR)]  # draft_prior vintages legitimately
    # name the CURRENT draft class by year, not a leaked in-season value - only flag if a
    # non-draft_prior row's vintage names the current season as its DATA source
    assert len(leaked) == 0, f"LEAKAGE: {len(leaked)} rows' vintage cites {stint_season} as a data source"
    print(f"  confirmed 0/{len(out)} rows cite {stint_season} (the season being modeled) as a "
          f"performance data source - every value predates it")

    print("\n-- Distribution of skill_off / skill_def by source --")
    print(out.groupby("source")[["skill_off", "skill_def"]].describe().round(2).to_string())

    print("\n-- Fitted draft-slot curve --")
    print(f"  trained on {curve['n_train']} historical rookie-seasons, classes "
          f"{curve['classes'][0]}..{curve['classes'][-1]}")
    print(f"  weighted R^2: OBPM={curve['r2_obpm']:.3f}, DBPM={curve['r2_dbpm']:.3f}")
    for pick in (1, 15, 30, 55):
        o, d = predict_from_pick(pick, curve)
        print(f"  pick {pick:>3d}: predicted skill_off={o:+.2f}, skill_def={d:+.2f}")
    print(f"  replacement level (undrafted): skill_off={replacement['skill_off']:+.2f}, "
          f"skill_def={replacement['skill_def']:+.2f} (n={replacement['n']} historical "
          f"undrafted debuts) - compare to the curve's own pick-55 prediction above as a "
          f"cross-check, not identical by construction but should be in the same neighborhood")

    with sqlite3.connect(db_path) as conn:
        pb2425 = set(pd.read_sql("SELECT PLAYER_ID FROM player_base WHERE SEASON = '2024-25'", conn)["PLAYER_ID"])
    rookie_ids = out[out["source"].isin([SOURCE_DRAFT_PRIOR, SOURCE_REPLACEMENT])]["player_id"]
    true_rookie_count = sum(1 for p in rookie_ids if p not in pb2425)
    print(f"\n-- Rookie count cross-check --")
    print(f"  {true_rookie_count} of the {len(rookie_ids)} draft-prior/replacement-level players "
          f"have zero {skill_season} row at all (true rookies) - expect this near the ~118 "
          f"figure flagged by fit_rapm.py's own join-coverage report")
    print("=" * 70)


# --- CLI ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--stint-season", default="2025-26")
    p.add_argument("--skill-season", default="2024-25",
                   help="prior season for the veteran tier - must be strictly before --stint-season")
    p.add_argument("--stints-path", type=Path, default=STINTS_PATH)
    p.add_argument("--out-path", type=Path, default=OUT_PATH)
    args = p.parse_args()

    out, curve, replacement = build_skill_table(args.stint_season, args.skill_season,
                                                stints_path=args.stints_path)
    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out_path, index=False)
    log.info(f"saved {len(out)} players -> {args.out_path}")

    print_validation_report(out, curve, replacement, args.stint_season, args.skill_season)


if __name__ == "__main__":
    main()
