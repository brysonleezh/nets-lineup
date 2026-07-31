import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# AI-ASSISTED (Claude Code, chat) - path fix for the src/pipeline/ move:
# resolve src/ from this file's own location instead of assuming cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step2_diagnostic_analysis import (
    compute_style_pool_by_vector, similarity_weighted_benchmark,
    load_player_oncourt_netrating, load_archetype_labels,
)

k = 8
brown = json.load(open("data/brown/profile.json"))
brown_alpha = np.array([brown["alpha"][f"arch_{i}"] for i in range(k)])

recipes = pd.read_csv("data/rc_multiseason/basis_reduced20/recipes_reduced20.csv")
recipes["SEASON"] = "2025-26"
labels_df = pd.read_csv("data/rc_multiseason/basis_reduced20/archetype_definitions.csv")
labels = {int(row["archetype"]): row["PLAYER_NAME"] for _, row in labels_df.iterrows()}

oncourt = load_player_oncourt_netrating(season="2025-26")

print("building similarity pool (this is real per-player SQL, ~400+ queries, one-time)...")
pool = compute_style_pool_by_vector(brown_alpha, recipes, k, season="2025-26")
result = similarity_weighted_benchmark(None, recipes, k, season="2025-26", oncourt=oncourt, power=4.0, pool=pool)

print(f"\nn_pool={result['n_pool']}, ess_all={result['ess_all']:.1f}")
print(f"n_pool_positive={result['n_pool_positive']}, ess_positive={result['ess_positive']}")

all_baseline = result["all_baseline"]
pos_baseline = result["positive_baseline"]

print("\n=== Teammate archetype exposure typically surrounding Brown's style comps (ALL) ===")
for i in range(k):
    print(f"  {labels.get(i, f'arch_{i}')}: {all_baseline[i]:.1%}")

if pos_baseline is not None:
    print("\n=== SAME, restricted to comps whose own on-court NET_RATING is positive ('worked') ===")
    for i in range(k):
        print(f"  {labels.get(i, f'arch_{i}')}: {pos_baseline[i]:.1%}")
    diff = pos_baseline - all_baseline
    print("\n=== Difference (positive-net environment minus unconditional) - the 'success pattern' ===")
    order = np.argsort(-diff)
    for i in order:
        print(f"  {labels.get(i, f'arch_{i}')}: {diff[i]:+.1%}")

out = {
    "n_pool": result["n_pool"], "ess_all": result["ess_all"],
    "n_pool_positive": result["n_pool_positive"], "ess_positive": result["ess_positive"],
    "all_baseline": {labels.get(i, f"arch_{i}"): float(all_baseline[i]) for i in range(k)},
    "positive_baseline": ({labels.get(i, f"arch_{i}"): float(pos_baseline[i]) for i in range(k)}
                          if pos_baseline is not None else None),
    "note": "DESCRIPTIVE / evidence synthesis, NOT causal inference - similarity-weighted "
            "pooling (Jensen-Shannon distance, power=4.0) across ALL 2025-26 NBA players "
            "weighted by stylistic closeness to Brown's own projected archetype vector, "
            "not a fixed top-10 comp average - a materially larger and more stable pooled "
            "sample than 10 discrete comps.",
}
Path("data/brown").mkdir(parents=True, exist_ok=True)
json.dump(out, open("data/brown/success_environment.json", "w"), indent=2)
print("\nsaved -> data/brown/success_environment.json")
