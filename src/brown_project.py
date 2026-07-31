import json
import sys
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

basis_dir = Path("data/rc_multiseason/basis_reduced20")
fit = s1.load_basis(basis_dir)
k = fit["k"]
print(f"loaded reduced basis: k={k}, feature_columns match REDUCED_FEATURES: "
      f"{fit['feature_columns'] == REDUCED_FEATURES}")

z_vec = json.load(open("/tmp/brown_z_vector_20.json"))
x_z = np.array([z_vec[f] for f in REDUCED_FEATURES])  # already standardized (z-scores)

# project() expects RAW feature values and does its own (x-mu)/sd internally using the
# NBA basis's own mu/sd - but we deliberately built Brown's vector as an NCAA-population
# z-score, not an NBA-population raw value. Solving the simplex-constrained least squares
# DIRECTLY in z-space (bypassing project()'s own internal standardization step) is the
# correct way to combine "already cross-population-normalized" input with a basis that's
# itself expressed in its own z-space - the whole point of z-scoring separately per
# population before projecting.
from scipy.optimize import nnls

B = fit["basis"]  # (k, p) in the NBA basis's own z-space
simplex_weight = 1e4
A = np.vstack([B.T, np.full((1, k), simplex_weight)])
b = np.concatenate([x_z, [simplex_weight]])
alpha, residual = nnls(A, b)
alpha = alpha / alpha.sum()
print(f"\nraw NNLS residual: {residual:.4f}")

# reconstruction residual as a novelty flag: ||x_z - alpha @ B|| in the shared z-space,
# normalized by sqrt(p) for interpretability (average per-feature standardized error)
recon = alpha @ B
resid_vec = x_z - recon
novelty = float(np.linalg.norm(resid_vec) / np.sqrt(len(REDUCED_FEATURES)))
print(f"reconstruction residual (novelty flag, avg per-feature |z| gap): {novelty:.3f}")

defs = pd.read_csv(basis_dir / "archetype_definitions.csv")
print("\n=== Mikel Brown Jr. archetype vector (reduced 20-feature NBA basis) ===")
for i in range(k):
    exemplar = defs[defs["archetype"] == i]["PLAYER_NAME"].iloc[0]
    print(f"  archetype {i} ({exemplar}): {alpha[i]:.4f}")

top3 = np.argsort(-alpha)[:3]
print(f"\ntop archetype: {top3[0]} ({defs[defs['archetype']==top3[0]]['PLAYER_NAME'].iloc[0]}) = {alpha[top3[0]]:.1%}")

out = {
    "player": "Mikel Brown Jr.", "athleteId": 200182, "school": "Louisville",
    "season": "2025-26 (NCAA)", "draft_pick": 6,
    "alpha": {f"arch_{i}": float(alpha[i]) for i in range(k)},
    "reconstruction_residual_avg_abs_z_gap": novelty,
    "z_vector": z_vec,
    "raw_features": {
        "height_inches": 75, "minutes": 613, "games": 21,
    },
}
Path("data/brown").mkdir(parents=True, exist_ok=True)
json.dump(out, open("data/brown/profile.json", "w"), indent=2)
print("\nsaved -> data/brown/profile.json")
