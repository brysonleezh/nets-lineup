from step0_data_collect_process import build_nba_side_tables, FEATURE_COLUMNS


def load_population(min_threshold=300, season_max=None, season_min=None):
    """Load player-season-team rows with filters applied.

    season_max is the train/test switch: pass "2024-25" for the validation
    basis. SEASON strings sort correctly since the prefix is the start year.
    """
    df = build_nba_side_tables()
    df = df[df["MIN"] >= min_threshold]
    if season_min is not None:
        df = df[df["SEASON"] >= season_min]
    if season_max is not None:
        df = df[df["SEASON"] <= season_max]
    df = df.reset_index(drop=True)

    print(f"population: {len(df)} rows | MIN>={min_threshold} | "
          f"seasons {df['SEASON'].min()}..{df['SEASON'].max()}")
    return df


if __name__ == "__main__":
    df = load_population(min_threshold=100)
    bkn = df[(df["TEAM_ABBREVIATION"] == "BKN")]
    print(bkn[["PLAYER_NAME", "SEASON", "MIN"]].sort_values("MIN").to_string(index=False))