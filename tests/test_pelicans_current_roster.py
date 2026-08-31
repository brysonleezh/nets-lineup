import numpy as np
import pandas as pd

from portal_shared import (
    DEFAULT_PELICANS_PLAYER_ID,
    PELICANS_CURRENT_ROSTER,
    current_roster_recipe_frame,
    load_pelicans_current_roster,
)


def _fixture_recipes(roster: pd.DataFrame) -> tuple[pd.DataFrame, set[int]]:
    eligible = roster.loc[~roster["roster_status"].isin({"two-way", "camp"})]
    rows = []
    for i, player in eligible.iterrows():
        weights = np.roll(
            np.array([0.40, 0.20, 0.10, 0.10, 0.07, 0.05, 0.04, 0.04]),
            i % 8,
        )
        row = {
            "PLAYER_ID": int(player["player_id"]),
            "PLAYER_NAME": player["player_name"],
            "TEAM_ABBREVIATION": "OLD",
            "MIN": 500 + i,
        }
        row.update({f"arch_{a}": float(weights[a]) for a in range(8)})
        rows.append(row)
    return pd.DataFrame(rows), set(eligible["player_id"].astype(int))


def test_curated_pelicans_roster_has_zion_and_expected_size():
    roster = load_pelicans_current_roster()

    assert len(roster) == 21
    assert DEFAULT_PELICANS_PLAYER_ID in set(roster["player_id"])
    assert PELICANS_CURRENT_ROSTER.startswith("Pelicans")


def test_current_roster_frame_is_player_breakdown_compatible():
    roster = load_pelicans_current_roster()
    recipes, eligible_ids = _fixture_recipes(roster)
    labels = {i: f"Type {i}" for i in range(8)}

    frame = current_roster_recipe_frame(recipes, 8, labels)

    assert set(frame["PLAYER_ID"]) == eligible_ids
    assert set(frame["TEAM_ABBREVIATION"]) == {"NOP"}
    assert set(frame["model_team"]) == {"OLD"}
    assert frame["role"].notna().all()
