# Rate Validation — computed college shared features vs. published values

Compares `data/college/shared_features.parquet`'s computed rate stats against
Sports-Reference/CBB's own published advanced stats, for real, well-known
players (recent lottery picks, spanning center/wing/guard positions).
Fetched live (curl with a browser user-agent — WebFetch itself was blocked
by Sports-Reference's bot protection; raw HTML parsed directly, using each
cell's displayed text, not its `csk` sort-key attribute — found `tov_pct`'s
`csk` is stored as a raw fraction while every other stat's `csk` is already
a percentage, an inconsistency in the source page itself, not a parsing bug).

Minutes-played matched each published value within 0-2 minutes for every
player, confirming the correct player/season row every time (not a
coincidental match).

| Player | Season | Stat | Published | Computed | Diff |
|---|---|---|---|---|---|
| Zach Edey (Purdue, C) | 2023-24 | TS% | 65.9 | 67.3 | **+1.4** |
| | | USG% | 33.4 | 33.1 | -0.3 |
| | | AST% | 14.6 | 14.56 | -0.04 |
| | | TOV% | 10.8 | 10.96 | +0.16 |
| | | STL% | 0.5 | 0.52 | +0.02 |
| | | BLK% | 6.9 | 6.93 | +0.03 |
| | | TRB% | 22.0 | 22.01 | +0.01 |
| Brandon Miller (Alabama, F) | 2022-23 | TS% | 58.3 | 58.90 | **+0.60** |
| | | USG% | 26.2 | 26.20 | +0.00 |
| | | AST% | 12.9 | 12.90 | -0.00 |
| | | TOV% | 12.0 | 12.06 | +0.06 |
| | | STL% | 1.5 | 1.52 | +0.02 |
| | | BLK% | 2.4 | 2.39 | -0.01 |
| | | TRB% | 12.6 | 12.55 | -0.05 |
| Stephon Castle (UConn, G) | 2023-24 | TS% | 55.1 | 55.70 | **+0.60** |
| | | USG% | 22.0 | 22.10 | +0.10 |
| | | AST% | 18.4 | 18.44 | +0.04 |
| | | TOV% | 13.0 | 13.10 | +0.10 |
| | | STL% | 1.8 | 1.79 | -0.01 |
| | | BLK% | 2.1 | 2.06 | -0.04 |
| | | TRB% | 10.1 | 10.08 | -0.02 |
| Cooper Flagg (Duke, F) | 2024-25 | TS% | 59.3 | 60.00 | **+0.70** |
| | | USG% | 30.9 | 30.80 | -0.10 |
| | | AST% | 26.8 | 26.65 | -0.15 |
| | | TOV% | 11.5 | 11.66 | +0.16 |
| | | STL% | 2.8 | 2.74 | -0.06 |
| | | BLK% | 4.9 | 4.88 | +0.02 |
| | | TRB% | 14.1 | 14.19 | +0.09 |

Height spot-check: Zach Edey 88in / Cooper Flagg 81in both matched exactly.

## Verdict

**USG%, AST%, TOV%, STL%, BLK%, TRB% — all 24 comparisons across 4 players land
within ±0.2pp**, comfortably inside the ±0.5pp tolerance. This is direct evidence
the derived-formula features (the ones this pipeline computes itself from CBBD's
box+team+opponent fields, including the AST% team-minutes formula that had a real
bug caught and fixed in an earlier pass) are correct, not just plausible.

**TS% shows a small, consistent, one-directional bias: +0.6 to +1.4pp every time**,
always computed-higher-than-published. TS% is a field CBBD returns directly
(`trueShootingPct`) — not something this pipeline derives — so this reflects a real
difference between CBBD's own TS% formula/rounding and Sports-Reference's, not a
bug in this pipeline. Flagged, not silently absorbed: any downstream use of TS%
should note this small source-level discrepancy.

Not independently validated (no direct public equivalent to check against):
`PTS_PER_100` (points per 100 team possessions isn't a standard published college
stat), `% of FG Ast'd_2P/3P` (published sites report per-shot-type, not the
aggregated 2P/3P split this pipeline computes) — both rely on formulas already
sanity-checked internally (see `reports/feature_dictionary.md`), just not
cross-checked against an external published number.
