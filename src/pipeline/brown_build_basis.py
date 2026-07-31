import sys
from pathlib import Path
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
print(f"{len(REDUCED_FEATURES)} shared features (dropped 9 PLAYTYPE_* columns)")

s1.FEATURE_COLUMNS = REDUCED_FEATURES
train = s1.load_population(min_threshold=300, season_min="2025-26", season_max="2025-26")
print(f"population: {len(train)} rows")
fit = s1.fit_basis_best_of(train, k=8, seeds=(0, 1, 2), max_iter=200, verbose=True)
assert fit["converged"], "did not converge"

out_dir = Path("data/rc_multiseason/basis_reduced20")
out_dir.mkdir(parents=True, exist_ok=True)
s1.save_basis(fit, out_dir)
defs = s1.match_archetypes_to_players(train, fit)
defs.to_csv(out_dir / "archetype_definitions.csv", index=False)
print(defs[["archetype", "PLAYER_NAME", "MIN"]].to_string(index=False))

everyone = s1.load_population(min_threshold=300, season_min="2025-26", season_max="2025-26")
P = s1.project(everyone, fit["basis"], fit["mu"], fit["sd"], fit["feature_columns"])
s1.check_simplex(P)
recipes = s1.recipes_frame(everyone, P, 8)
recipes.to_csv(out_dir / "recipes_reduced20.csv", index=False)
print(f"saved {len(recipes)} recipes -> {out_dir}/recipes_reduced20.csv")
