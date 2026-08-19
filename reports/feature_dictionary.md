# Feature Dictionary — NBA 29-feature basis vs. CBBD (NCAA) availability

Draft/reference version, produced ahead of formal Step 3, at the owner's request, to
ground the Step 3 shared-feature design in verified fact rather than assumption. All
CBBD field names below were checked directly against real cached responses in
`data/raw/cbbd/`, not recalled from memory.

Legend: **Direct** = CBBD returns this value (or an equivalent) directly.
**Derivable** = computable from CBBD's own player+team(+opponent) fields via a standard
formula. **Substitute** = CBBD has no equivalent stat, but offers a different metric
occupying a similar role. **None** = no CBBD data exists to build this at all (not a
coverage/sparsity issue — the field doesn't exist in the schema, in any season).

| NBA feature | Status | CBBD source |
|---|---|---|
| `TS%` | Direct | `/stats/player/season`: `trueShootingPct` |
| `USG%` | Direct | `/stats/player/season`: `usage` |
| `FTr` | Direct | `/stats/player/season`: `freeThrowRate` (= FTA/FGA, confirmed against raw makes/attempts) |
| `PLAYER_HEIGHT_INCHES` | Direct | `/teams/roster`: `height` (already inches) |
| `PTS_PER_100` | Derivable | `points` (player) ÷ `possessions` (team, `/stats/team/season`) |
| `AST%` | Derivable | `assists` (player) + team `fieldGoals.made` + team `totalMinutes` — standard formula; **known trap**: CBBD's team `totalMinutes` is already "Tm MP / 5" in the textbook formula's terms, not "Tm MP" itself — dividing by 5 again silently understates AST% ~5×, found and fixed in an earlier pass |
| `TOV%` | Derivable | `turnovers`, `fieldGoals.attempted`, `freeThrows.attempted` — player-level only, no team context needed |
| `STL%` | Derivable | `steals` (player) + team `totalMinutes` + `opponentStats.possessions` |
| `BLK%` | Derivable | `blocks` (player) + `opponentStats.fieldGoals.attempted` − `opponentStats.threePointFieldGoals.attempted` (opponent 2PA) |
| `TRB%` | Derivable | `rebounds.total` (player) + team `rebounds.total` + `opponentStats.rebounds.total` |
| `% of FG Ast'd_2P` | Derivable | `/stats/player/shooting/season`: sum `assisted`/`made` across `dunks`+`layups`+`tipIns`+`twoPointJumpers` (no single category alone equals "all 2-pointers") |
| `% of FG Ast'd_3P` | Derivable | `/stats/player/shooting/season`: `threePointJumpers.assisted` / `threePointJumpers.made` |
| `BPM` | Substitute | `/stats/player/season`: `PORPAG` — a different all-in-one value metric occupying BPM's role, not a formula-equivalent |
| `Dist.` (avg shot distance) | None | CBBD has no distance-in-feet data at all |
| `% of FGA by Distance_0-3` | None | same — CBBD tracks shot *type* (dunk/layup/tip-in/jumper), never distance-in-feet, in any season |
| `% of FGA by Distance_3-10` | None | same |
| `% of FGA by Distance_10-16` | None | same |
| `% of FGA by Distance_16-3P` | None | same |
| `% of FGA by Distance_3P` | None | same |
| `Corner 3s_%3PA` | None | no corner-vs-non-corner 3PT split anywhere in CBBD |
| `PLAYTYPE_CUT` | None | CBBD has no Synergy-style play-type tracking at all (9/9 play-type features) |
| `PLAYTYPE_HANDOFF` | None | " |
| `PLAYTYPE_ISOLATION` | None | " |
| `PLAYTYPE_OFFREBOUND` | None | " |
| `PLAYTYPE_OFFSCREEN` | None | " |
| `PLAYTYPE_PRBALLHANDLER` | None | " |
| `PLAYTYPE_PRROLLMAN` | None | " |
| `PLAYTYPE_POSTUP` | None | " |
| `PLAYTYPE_SPOTUP` | None | " |

## Summary

- **4 direct**, **8 derivable** (12 total buildable, matching NBA-side definitions closely)
- **1 substitute** (BPM → PORPAG, not equivalent, must be flagged wherever used)
- **16 not available at all** — 7 shot-location features (`Dist.` + 5 distance bins + corner-3),
  9 play-type features. All 16 are structural schema absences in CBBD, not a per-season
  coverage problem.

**Implication for Step 3**: the college-side archetype model can realistically carry at
most 12-13 of the NBA basis's 29 dimensions. The remaining 16-17 are not "missing data
to backfill" — they cannot be built from this source in any season. Step 3's shared-
feature set must be defined around this ceiling, not around trying to reach 29.
