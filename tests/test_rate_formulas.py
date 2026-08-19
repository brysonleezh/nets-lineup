"""
Rate-formula regression tests against hand-checked fixtures.

Fixture values are the real, published Sports-Reference/CBB advanced stats
for 4 well-known players, fetched and cross-checked manually in
reports/rate_validation.md (curl + raw HTML parse - WebFetch itself was
blocked by the site's bot protection). This file codifies that manual
validation into an automated regression test, so a future formula change
that silently breaks AST%/STL%/BLK%/TRB%/TOV%/USG% would fail CI instead
of requiring another manual re-check.

Tolerance: ±0.5 percentage points, per the phase spec's own stated
tolerance for percentage stats. TS% is excluded from the tolerance
assertion - see reports/rate_validation.md: CBBD's own trueShootingPct
field (not something this pipeline derives) runs a small, consistent
+0.6 to +1.4pp higher than Sports-Reference's TS%, a source-level formula/
rounding difference, not a bug in this pipeline's formulas.
"""

from __future__ import annotations

import pytest

from build_features import build_season_features

TOLERANCE_PP = 0.5  # percentage points

FIXTURES = [
    # (name, season, {stat: published_percentage_points})
    ("Zach Edey", 2024, {
        "USG%": 33.4, "AST%": 14.6, "TOV%": 10.8, "STL%": 0.5, "BLK%": 6.9, "TRB%": 22.0,
    }),
    ("Brandon Miller", 2023, {
        "USG%": 26.2, "AST%": 12.9, "TOV%": 12.0, "STL%": 1.5, "BLK%": 2.4, "TRB%": 12.6,
    }),
    ("Stephon Castle", 2024, {
        "USG%": 22.0, "AST%": 18.4, "TOV%": 13.0, "STL%": 1.8, "BLK%": 2.1, "TRB%": 10.1,
    }),
    ("Cooper Flagg", 2025, {
        "USG%": 30.9, "AST%": 26.8, "TOV%": 11.5, "STL%": 2.8, "BLK%": 4.9, "TRB%": 14.1,
    }),
    # Phase 2 Step 0.2 - early-era backfill (2016-17..2018-19), incl. a
    # non-power-conference case (Ja Morant, Murray State/OVC), to rule out
    # schema drift across the full pooled fit window before fitting on it.
    ("Frank Mason III", 2017, {
        "USG%": 25.6, "AST%": 26.1, "TOV%": 12.4, "STL%": 2.0, "BLK%": 0.2, "TRB%": 6.4,
    }),
    ("Trae Young", 2018, {
        "USG%": 37.1, "AST%": 48.6, "TOV%": 18.2, "STL%": 2.5, "BLK%": 0.7, "TRB%": 5.8,
    }),
    ("Zion Williamson", 2019, {
        "USG%": 28.6, "AST%": 14.9, "TOV%": 12.8, "STL%": 3.9, "BLK%": 5.8, "TRB%": 15.5,
    }),
    ("Ja Morant", 2019, {
        "USG%": 33.3, "AST%": 51.8, "TOV%": 20.5, "STL%": 2.7, "BLK%": 2.3, "TRB%": 8.6,
    }),
]

KNOWN_HEIGHTS_IN = {
    ("Zach Edey", 2024): 88,
    ("Cooper Flagg", 2025): 81,
}


@pytest.fixture(scope="module")
def season_frames():
    seasons = {f[1] for f in FIXTURES}
    return {s: build_season_features(s, _cache_dir()) for s in seasons}


def _cache_dir():
    from pathlib import Path
    return Path(__file__).resolve().parent.parent / "data" / "raw" / "cbbd"


@pytest.mark.parametrize("name,season,published", FIXTURES)
def test_rate_stats_within_tolerance(season_frames, name, season, published):
    df = season_frames[season]
    row = df[df["name"] == name]
    assert len(row) == 1, f"expected exactly one row for {name} in season {season}, got {len(row)}"
    row = row.iloc[0]

    for stat, pub_value in published.items():
        computed_pp = row[stat] * 100.0
        diff = computed_pp - pub_value
        assert abs(diff) <= TOLERANCE_PP, (
            f"{name} {season} {stat}: computed={computed_pp:.2f}pp published={pub_value}pp "
            f"diff={diff:+.2f}pp exceeds ±{TOLERANCE_PP}pp tolerance"
        )


def test_ts_pct_has_known_small_positive_bias(season_frames):
    """TS% is a direct CBBD field, not derived here - documents (doesn't
    silently hide) the known small systematic offset vs Sports-Reference."""
    published_ts = {"Zach Edey": 65.9, "Brandon Miller": 58.3, "Stephon Castle": 55.1, "Cooper Flagg": 59.3}
    season_by_name = {"Zach Edey": 2024, "Brandon Miller": 2023, "Stephon Castle": 2024, "Cooper Flagg": 2025}
    for name, pub in published_ts.items():
        df = season_frames[season_by_name[name]]
        row = df[df["name"] == name].iloc[0]
        diff = row["TS%"] * 100.0 - pub
        assert 0.3 <= diff <= 1.8, f"{name}: TS% diff {diff:+.2f}pp outside the previously-observed 0.3-1.8pp band"


HEIGHT_CASES = [(name, season, inches) for (name, season), inches in KNOWN_HEIGHTS_IN.items()]


@pytest.mark.parametrize("name,season,expected_in", HEIGHT_CASES)
def test_height_matches_known_value(season_frames, name, season, expected_in):
    df = season_frames[season]
    row = df[df["name"] == name].iloc[0]
    assert row["PLAYER_HEIGHT_INCHES"] == expected_in
