"""
Phase 2 Step 4 - interpretation deliverable: reports/college_archetypes.md.
Per archetype: mean z-profile (the AA archetype's own position - it IS a
point in z-space, not a group mean, but reported as such since that's
what it represents), distinguishing features (largest |z|), top-10
loading player-seasons (with conference/season/minutes so the owner can
judge outlier risk directly), and an empty label column.

Run: python3 src/college_model/build_interpretation.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pool import SHARED_COLS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
MODEL_DIR = REPO_ROOT / "data" / "college" / "model"
REPORTS_DIR = REPO_ROOT / "reports"
K = 8

# Paper's 8 European archetypes, for the orientation note (qualitative only)
PAPER_ARCHETYPES = [
    "High Usage Guard", "3PT Specialist", "Traditional Center", "Backup Big",
    "Defensive Specialist", "High Usage Forward", "Traditional Playmaker", "Role Guard",
]


def main() -> None:
    frozen = np.load(MODEL_DIR / "k8_frozen_basis.npz")
    basis = frozen["basis"]  # (K, 12) z-profile per archetype
    recipes = pd.read_csv(REPO_ROOT / "data" / "college" / "recipes.csv")

    lines = [
        "# College Archetype Interpretation (K=8, AA)\n",
        f"Frozen fit: `data/college/model/k8_frozen_basis.npz` (consensus-basin "
        "seed=2, see `reports/phase2_worklog.md` for the basin analysis that "
        "produced this choice). Method: AA (not ADA) - see CHECKPOINT 2 in the "
        "worklog for why (7/8 restricted-search archetypoids landed on low-major "
        "conference outliers).\n",
        "\n**`label` column is intentionally empty** - naming these requires "
        "basketball domain judgment, not an agent's call.\n",
    ]

    for j in range(K):
        z_profile = basis[j]
        order = np.argsort(-np.abs(z_profile))
        top_features = [(SHARED_COLS[i], z_profile[i]) for i in order[:5]]

        top10 = recipes.sort_values(f"alpha_{j}", ascending=False).head(10)[
            ["player_name_raw", "season", "team", "conference", "minutes", f"alpha_{j}"]
        ]

        lines.append(f"\n## Archetype {j}\n")
        lines.append("**label:** _____________________ (owner to fill)\n")
        lines.append("\n**Mean z-profile (all 12 features):**\n")
        lines.append("| Feature | z |")
        lines.append("|---|---|")
        for i, feat in enumerate(SHARED_COLS):
            lines.append(f"| {feat} | {z_profile[i]:+.2f} |")
        lines.append("\n**Distinguishing features (top 5 by \\|z\\|):**\n")
        for feat, val in top_features:
            lines.append(f"- {feat}: {val:+.2f}")
        lines.append(f"\n**Top-10 loading player-seasons (by alpha_{j}):**\n")
        lines.append("| Player | Season | Team | Conference | Minutes | alpha |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in top10.iterrows():
            lines.append(f"| {r['player_name_raw']} | {r['season']} | {r['team']} | "
                         f"{r['conference']} | {r['minutes']:.0f} | {r[f'alpha_{j}']:.3f} |")
        lines.append(f"\n**Apparent correspondence to paper's European archetypes:** "
                     f"_____________________ (owner to judge; candidates: {', '.join(PAPER_ARCHETYPES)})\n")

    (REPORTS_DIR / "college_archetypes.md").write_text("\n".join(lines))
    print(f"Wrote {REPORTS_DIR / 'college_archetypes.md'}")


if __name__ == "__main__":
    main()
