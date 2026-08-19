"""
Fit-model diagnostic harness - garbage-time classification.

Ground rule #3 (harness spec) requires garbage time excluded everywhere,
citing "pbpstats definitions." Checked directly: pbpstats (installed,
inspected its source) ships NO garbage-time/blowout filter at all - no
"garbage"/"blowout" string appears anywhere in the package. There is no
literal definition to import.

Used instead: a standard, widely-cited tiered score-margin / time-
remaining threshold (the convention behind most public "garbage time"
splits, e.g. Inpredictable-style tables) - not attributed to pbpstats,
since it doesn't have one. Documented here, not silently assumed:

    Periods 1-3:  garbage if |margin| > 25 (rare - an extreme blowout
                  well before the 4th quarter)
    Period 4+ (including OT), tiered by minutes remaining:
        > 9 min left :  |margin| > 25
        6-9 min left :  |margin| > 20
        3-6 min left :  |margin| > 15
        < 3 min left :  |margin| > 10

Operates on stint_start_margin/period/stint_start_clock - the score
context AT THE START of each stint (added to build_stints.py's output
specifically to support this; see that file's own comment at the
stint_start_clock variable).
"""

from __future__ import annotations

import re

import numpy as np
import pandas as pd

_CLOCK_RE = re.compile(r"PT(\d+)M([\d.]+)S")


def clock_to_minutes_remaining(clock: str) -> float:
    m = _CLOCK_RE.match(clock)
    if not m:
        return np.nan
    minutes, seconds = float(m.group(1)), float(m.group(2))
    return minutes + seconds / 60.0


def is_garbage_time(period: pd.Series, margin: pd.Series, clock: pd.Series) -> pd.Series:
    """margin: this row's OWN offense-team-relative margin at stint start
    (already signed that way in build_stints.py's output) - abs() here
    since a blowout is garbage time for BOTH sides' rows."""
    abs_margin = margin.abs()
    minutes_left = clock.map(clock_to_minutes_remaining)

    early = (period <= 3) & (abs_margin > 25)

    late = period >= 4
    thresh = pd.Series(np.select(
        [minutes_left > 9, minutes_left > 6, minutes_left > 3, minutes_left >= 0],
        [25, 20, 15, 10],
        default=np.nan,
    ), index=margin.index)
    late_garbage = late & (abs_margin > thresh)

    return (early | late_garbage).fillna(False)
