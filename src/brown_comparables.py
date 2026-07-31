import json
import sys
import sqlite3
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, ".")
import step1_archetypes_model as s1

REDUCED_FEATURES = [
    "PTS_PER_100", "PLAYER_HEIGHT_INCHES",
    "TS%", "USG%", "AST%", "TOV%", "STL%", "BLK%", "TRB%", "FTr", "BPM",
    "Dist.", "% of FGA by Distance_0-3", "% of FGA by Distance_3-10",
    "% of FGA by Distance_10-16", "% of FGA by Distance_16-3P", "% of FGA by Distance_3P",
    "Corner 3s_%3PA", "% of FG Ast'd_2P", "% of FG Ast'd_3P",
]
s1.FEATURE_COLUMNS = REDUCED_FEATURES

basis_dir = Path("data/rc_multiseason/basis_reduced20")
fit = s1.load_basis(basis_dir)
k = fit["k"]

brown = json.load(open("data/brown/profile.json"))
brown_alpha = np.array([brown["alpha"][f"arch_{i}"] for i in range(k)])

# project all 3 seasons onto this SAME reduced basis, for an apples-to-apples comparable pool
all_recipes = []
recipes_2526 = pd.read_csv(basis_dir / "recipes_reduced20.csv")
recipes_2526["SEASON"] = "2025-26"
all_recipes.append(recipes_2526)
for season in ["2023-24", "2024-25"]:
    pop = s1.load_population(min_threshold=300, season_min=season, season_max=season)
    P = s1.project(pop, fit["basis"], fit["mu"], fit["sd"], fit["feature_columns"])
    rec = s1.recipes_frame(pop, P, k)
    all_recipes.append(rec)
    print(f"{season}: projected {len(rec)} players onto the reduced basis")

combined = pd.concat(all_recipes, ignore_index=True)
combined = combined.drop_duplicates(subset=["PLAYER_ID", "SEASON"])
print(f"total player-seasons in comparable pool: {len(combined)}")
combined.to_csv("data/rc_multiseason/basis_reduced20/recipes_reduced20_allseasons.csv", index=False)

arch_cols = [f"arch_{i}" for i in range(k)]
vecs = combined[arch_cols].values.astype(float)

def cosine_sim(a, B):
    return (B @ a) / (np.linalg.norm(B, axis=1) * np.linalg.norm(a) + 1e-12)

def euclidean_sim(a, B):
    d = np.linalg.norm(B - a, axis=1)
    return 1.0 / (1.0 + d)  # bounded similarity, 1=identical

cos = cosine_sim(brown_alpha, vecs)
euc_d = np.linalg.norm(vecs - brown_alpha, axis=1)
combined["cosine_sim"] = cos
combined["euclidean_dist"] = euc_d

# --- (a) all current (2025-26) NBA players ---
current = combined[combined["SEASON"] == "2025-26"].copy()
current_top = current.sort_values("cosine_sim", ascending=False).head(10)
print("\n=== Top 10 comps: ALL current (2025-26) NBA players ===")
print(current_top[["PLAYER_NAME", "TEAM_ABBREVIATION", "cosine_sim", "euclidean_dist"]].to_string(index=False))

# --- (b) rookie-season profiles across the 3 available seasons ---
db_path = "data/nets_synergy.db"
with sqlite3.connect(db_path) as conn:
    bio = pd.read_sql("SELECT PLAYER_ID, DRAFT_YEAR FROM player_bio", conn)
bio = bio[(bio["DRAFT_YEAR"] != "Undrafted") & bio["DRAFT_YEAR"].notna()].copy()
bio["draft_year_int"] = bio["DRAFT_YEAR"].astype(int)
combined["season_start"] = combined["SEASON"].str.split("-").str[0].astype(int)
rookie_pool = combined.merge(bio, on="PLAYER_ID", how="inner")
rookie_pool = rookie_pool[rookie_pool["draft_year_int"] == rookie_pool["season_start"]]
print(f"\nrookie-season profiles found across 3 seasons: {len(rookie_pool)}")
rookie_top = rookie_pool.sort_values("cosine_sim", ascending=False).head(10)
print("=== Top 10 comps: rookie-season profiles (2023-24, 2024-25, 2025-26 draft classes) ===")
print(rookie_top[["PLAYER_NAME", "SEASON", "TEAM_ABBREVIATION", "cosine_sim", "euclidean_dist"]].to_string(index=False))

# flag comps driven by residual-heavy dimensions (crude check: how much of the cosine
# similarity is coming from archetypes Brown himself has near-zero weight in - a comp
# that matches mostly on archetypes Brown DOESN'T strongly express is a weaker match)
low_weight_archs = [i for i in range(k) if brown_alpha[i] < 0.05]
print(f"\narchetypes Brown has <5% weight in (residual-heavy check): {low_weight_archs}")

out = {
    "current_nba_comps": current_top[["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "cosine_sim", "euclidean_dist"]].to_dict("records"),
    "rookie_season_comps": rookie_top[["PLAYER_ID", "PLAYER_NAME", "SEASON", "TEAM_ABBREVIATION", "cosine_sim", "euclidean_dist"]].to_dict("records"),
}
json.dump(out, open("data/brown/comparables.json", "w"), indent=2)
print("\nsaved -> data/brown/comparables.json")
