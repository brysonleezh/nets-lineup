import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd

# AI-ASSISTED (Claude Code, chat) - path fix for the src/pipeline/ move:
# resolve src/ from this file's own location instead of assuming cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import step1_archetype_model as s1

REDUCED_FEATURES = [
    "PTS_PER_100", "PLAYER_HEIGHT_INCHES",
    "TS%", "USG%", "AST%", "TOV%", "STL%", "BLK%", "TRB%", "FTr", "BPM",
    "Dist.", "% of FGA by Distance_0-3", "% of FGA by Distance_3-10",
    "% of FGA by Distance_10-16", "% of FGA by Distance_16-3P", "% of FGA by Distance_3P",
    "Corner 3s_%3PA", "% of FG Ast'd_2P", "% of FG Ast'd_3P",
]

# --- population shot-TYPE baseline (real, n=qualifying D1 players with shot data) ---
shooting = json.load(open("/tmp/cbbd_all_shooting_by_team_2026.json"))
players = json.load(open("/tmp/cbbd_all_players_2026.json"))
qualified_ids = set(p["athleteId"] for p in players if (p.get("minutes") or 0) >= 300)

rows = []
for s in shooting:
    if s["athleteId"] not in qualified_ids:
        continue
    total_fga = (s["dunks"]["attempted"] + s["layups"]["attempted"] + s["tipIns"]["attempted"]
                + s["twoPointJumpers"]["attempted"] + s["threePointJumpers"]["attempted"])
    if total_fga < 20:
        continue
    made_rim = s["dunks"]["made"] + s["layups"]["made"] + s["tipIns"]["made"]
    ast_rim = s["dunks"]["assisted"] + s["layups"]["assisted"]
    made_mid = s["twoPointJumpers"]["made"]
    ast_mid = s["twoPointJumpers"]["assisted"]
    made_3 = s["threePointJumpers"]["made"]
    ast_3 = s["threePointJumpers"]["assisted"]
    made_2p = made_rim + made_mid
    rows.append({
        "pct_rim": 100 * (s["dunks"]["attempted"] + s["layups"]["attempted"] + s["tipIns"]["attempted"]) / total_fga,
        "pct_mid2": 100 * s["twoPointJumpers"]["attempted"] / total_fga,
        "pct_3": 100 * s["threePointJumpers"]["attempted"] / total_fga,
        "ast2": 100 * (ast_rim + ast_mid) / made_2p if made_2p > 0 else np.nan,
        "ast3": 100 * ast_3 / made_3 if made_3 > 0 else np.nan,
    })
shot_df = pd.DataFrame(rows).dropna()
print(f"population shot-type baseline: n={len(shot_df)}")
mu = shot_df.mean()
sd = shot_df.std()

brown_shot_dist = json.load(open("/tmp/brown_shot_dist_features.json"))
brown_assisted = json.load(open("/tmp/brown_assisted_features.json"))
brown_box_z = json.load(open("/tmp/brown_box_z.json"))

pct_rim_b = brown_shot_dist["% of FGA by Distance_0-3"] + brown_shot_dist["% of FGA by Distance_3-10"]
pct_mid_b = brown_shot_dist["% of FGA by Distance_10-16"] + brown_shot_dist["% of FGA by Distance_16-3P"]
pct_3_b = brown_shot_dist["% of FGA by Distance_3P"]

z_rim = (pct_rim_b - mu["pct_rim"]) / sd["pct_rim"]
z_mid = (pct_mid_b - mu["pct_mid2"]) / sd["pct_mid2"]
z_3 = (pct_3_b - mu["pct_3"]) / sd["pct_3"]
z_ast2 = (brown_assisted["% of FG Ast'd_2P"] - mu["ast2"]) / sd["ast2"]
z_ast3 = (brown_assisted["% of FG Ast'd_3P"] - mu["ast3"]) / sd["ast3"]

# NOTE (documented limitation, not hidden): CBBD's bulk shot-type data can't split
# "rim" into the NBA model's separate 0-3/3-10 buckets, or "mid2" into 10-16/16-3P,
# at population scale. Both sub-buckets in each pair get the SAME collapsed z-score
# rather than a fabricated split - simpler and more honest than inventing a
# proportional-weighting scheme with no real basis. See BUILD docs.
z_vec = {
    "PTS_PER_100": brown_box_z["PTS_PER_100"], "TS%": brown_box_z["TS%"], "USG%": brown_box_z["USG%"],
    "AST%": brown_box_z["AST%"], "TOV%": brown_box_z["TOV%"], "STL%": brown_box_z["STL%"],
    "BLK%": brown_box_z["BLK%"], "TRB%": brown_box_z["TRB%"], "FTr": brown_box_z["FTr"],
    "BPM": brown_box_z["BPM_proxy_PORPAG"],
    "% of FGA by Distance_0-3": z_rim, "% of FGA by Distance_3-10": z_rim,
    "% of FGA by Distance_10-16": z_mid, "% of FGA by Distance_16-3P": z_mid,
    "% of FGA by Distance_3P": z_3,
    "% of FG Ast'd_2P": z_ast2, "% of FG Ast'd_3P": z_ast3,
}

# PLAYER_HEIGHT_INCHES: league-invariant physical measurement - z-score against the
# NBA training population's OWN mu/sd (not a separate NCAA population), since a
# 6'3" guard is 6'3" in any league; this is the one feature where using the NBA
# basis's own scale directly is MORE correct, not less.
# Dist. (average shot distance in feet): genuinely league-scale-dependent (NCAA's
# closer 3PT line shifts this), but no population Dist. average is available from
# the bulk endpoints - z-scored against Brown's own ~15-player Louisville roster
# as the best available REAL (if small) reference, flagged as the weakest-support
# feature in this vector.
s1.FEATURE_COLUMNS = REDUCED_FEATURES
train = s1.load_population(min_threshold=300, season_min="2025-26", season_max="2025-26")
nba_mu = train["PLAYER_HEIGHT_INCHES"].mean()
nba_sd = train["PLAYER_HEIGHT_INCHES"].std()
z_vec["PLAYER_HEIGHT_INCHES"] = (75 - nba_mu) / nba_sd
print(f"height: 75in vs NBA pop mean={nba_mu:.1f}in sd={nba_sd:.2f} -> z={z_vec['PLAYER_HEIGHT_INCHES']:.3f}")

lou_dist_proxy = brown_shot_dist["Dist."]  # can't get a clean teammate Dist. average from
# the data pulled so far without another per-player pass - use Brown's own value with
# z=0 relative to himself is meaningless, so instead z-score Dist. against the NBA
# population directly (same reasoning as height: it's a physical measurement, and while
# it IS league-scale-dependent in principle, no better real reference population is
# available within this task's scope - documented as the second-weakest-support feature)
nba_dist_mu = train["Dist."].mean()
nba_dist_sd = train["Dist."].std()
z_vec["Dist."] = (brown_shot_dist["Dist."] - nba_dist_mu) / nba_dist_sd
print(f"Dist.: {brown_shot_dist['Dist.']:.2f}ft vs NBA pop mean={nba_dist_mu:.2f} sd={nba_dist_sd:.2f} -> z={z_vec['Dist.']:.3f}")

# Corner 3s_%3PA: same situation as Dist. - no population baseline from the bulk
# endpoints (would need per-player play-by-play at population scale). Z-scored
# against the NBA population's own mu/sd as the best available real fallback -
# the third and last feature using this fallback, all three flagged plainly.
nba_c3_mu = train["Corner 3s_%3PA"].mean()
nba_c3_sd = train["Corner 3s_%3PA"].std()
brown_c3_frac = brown_shot_dist["Corner 3s_%3PA"] / 100.0  # BUG FIX: player_shooting_bref
# stores this as a 0-1 fraction (verified directly: Luka Doncic's real row = 0.058, not 5.8),
# but my own coordinate-derived computation for Brown was in 0-100 percentage terms -
# caught by an absurd z=130 result, not assumed correct from a clean-looking number
z_vec["Corner 3s_%3PA"] = (brown_c3_frac - nba_c3_mu) / nba_c3_sd
print(f"Corner 3s_%3PA: {brown_c3_frac:.4f} (fraction) vs NBA pop mean={nba_c3_mu:.4f} sd={nba_c3_sd:.4f} -> z={z_vec['Corner 3s_%3PA']:.3f}")

print("\n=== FULL 20-feature z-vector for Mikel Brown Jr. ===")
for f in REDUCED_FEATURES:
    print(f"  {f}: {z_vec.get(f, 'MISSING')}")

assert all(f in z_vec for f in REDUCED_FEATURES), "missing features!"
json.dump(z_vec, open("/tmp/brown_z_vector_20.json", "w"))
print(f"\nsaved full 20-feature z-vector -> /tmp/brown_z_vector_20.json")
