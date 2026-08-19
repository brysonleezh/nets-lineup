"""
Name normalizer tests: suffixes (Jr/II/III/IV), diacritics, punctuation,
case, and the specific duplicate-name collision case that motivates
falling back from (name, year, team) to (name, year) matching in
build_draft_crosswalk.py.
"""

from __future__ import annotations

from build_draft_crosswalk import normalize_name


def test_lowercases():
    assert normalize_name("AJ Dybantsa") == "aj dybantsa"


def test_strips_suffix_jr():
    assert normalize_name("De'Anthony Melton Jr.") == normalize_name("De'Anthony Melton")


def test_strips_suffix_roman_numerals():
    assert normalize_name("Ron Holland II") == "ron holland"
    assert normalize_name("Tim Hardaway III") == "tim hardaway"
    assert normalize_name("Some Player IV") == "some player"


def test_strips_punctuation_and_apostrophe():
    assert normalize_name("De'Anthony Melton") == "deanthony melton"


def test_transliterates_diacritics_nfkd_decomposable():
    # e.g. accented Latin characters that NFKD decomposes cleanly
    assert normalize_name("Luka Dončić") == "luka doncic"
    assert normalize_name("José García") == "jose garcia"


def test_transliterates_diacritics_manual_map():
    # characters NFKD does NOT decompose on their own (đ, ø, ł)
    assert normalize_name("Đorđe Marković") == "dorde markovic"
    assert normalize_name("Kristaps Porziņģis") == normalize_name("Kristaps Porzingis")


def test_collapses_whitespace_and_is_stable():
    assert normalize_name("  Egor   Demin ") == "egor demin"


def test_two_players_same_normalized_name_are_distinguishable_only_by_year_team():
    """This is the real reason (name, year, team) is tried before falling
    back to (name, year) alone in the crosswalk: a bare normalized-name
    match can collide across different real players. The normalizer
    itself is correct here - collapsing "Jr."/"Jr"/no-suffix variants of
    the same person is the intended behavior, not a bug - the crosswalk's
    fallback ordering exists to handle exactly this kind of collision."""
    a = normalize_name("Michael Porter Jr.")
    b = normalize_name("Michael Porter")
    assert a == b  # by design - the crosswalk disambiguates via (year, team), not name alone


def test_empty_and_none_safe():
    assert normalize_name("") == ""
    assert normalize_name(None) == ""
