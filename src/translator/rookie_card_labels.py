"""Shared label lookups for Phase 6 rookie cards - parses
reports/college_archetypes.md programmatically (never hardcoded, so cards
can't silently drift from the finalized source of truth) and loads the
NBA-side archetype_labels.csv/archetype_definitions.csv the rest of the
app already uses.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
K = 8


def load_college_labels() -> dict[int, str]:
    text = (REPO_ROOT / "reports" / "college_archetypes.md").read_text()
    labels = {}
    for m in re.finditer(r"## Archetype (\d)\n\n\*\*label:\*\* (.+)", text):
        idx = int(m.group(1))
        label = m.group(2).strip()
        labels[idx] = label
    assert len(labels) == K, f"expected {K} college labels, parsed {len(labels)}"
    assert "(draft" not in "".join(labels.values()), \
        "college labels still contain a (draft...) marker - are they finalized?"
    return labels


def load_nba_labels() -> dict[int, dict]:
    df = pd.read_csv(REPO_ROOT / "data" / "basis_2025_26" / "archetype_labels.csv")
    out = {}
    for _, r in df.iterrows():
        out[int(r["archetype"])] = {"label": r["paper_label"], "exemplar": r["exemplar"],
                                     "match_quality": r["match_quality"], "note": r["note"]}
    assert len(out) == K
    return out


def load_nba_archetypoids() -> dict[int, dict]:
    """Real archetypoid (nearest real player-season) per NBA archetype,
    for the "closest to <player> <season>" card prose."""
    df = pd.read_csv(REPO_ROOT / "data" / "basis_2025_26" / "archetype_definitions.csv")
    out = {}
    for _, r in df.iterrows():
        out[int(r["archetype"])] = {"player": r["PLAYER_NAME"], "season": r["SEASON"], "team": r["TEAM"]}
    assert len(out) == K
    return out


if __name__ == "__main__":
    college = load_college_labels()
    nba = load_nba_labels()
    archetypoids = load_nba_archetypoids()
    for j in range(K):
        print(f"college {j}: {college[j]}")
    for j in range(K):
        print(f"nba {j}: {nba[j]['label']} (exemplar: {nba[j]['exemplar']}, "
              f"archetypoid: {archetypoids[j]['player']} {archetypoids[j]['season']})")
    print("rookie_card_labels.py self-check: PASS")
