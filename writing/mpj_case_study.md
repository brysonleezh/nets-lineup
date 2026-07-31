# Case Study: Diagnosing Player Combination Mismatch — Michael Porter Jr.

*Draft written by Claude Code from `src/step2_diagnostic_analysis.py`'s real output (2025-26 season data). Raw notes — expected to be edited before going into the final write-up.*

## Why this application, why this player

The *Scouting Anyone* paper (Section 6) names three practical applications of a probabilistic archetype framework: (1) diagnostic analysis of player-combination mismatches in an existing roster, (2) league-wide/cross-league scouting, (3) roster construction — balancing archetype distributions for tactical coherence. This case study follows application (1), mirroring the paper's own Section 6.1 structure (its Markus Howard / Baskonia case study) rather than the project's earlier synergy-matrix design (`src/pipeline/step2_synergy_matrix.py`, kept in the repo as visible prior work, not deleted — see `AI_USAGE.md` Entry 007 for why it was set aside: the additive "sum of 10 pairwise scores" assumption it rested on was hard to fully trust or explain).

**Player selection was data-driven, not assumed.** Checked every Nets data-eligible roster player's dominant-archetype concentration first; **Michael Porter Jr. has the highest single-archetype concentration on the roster — 66.6% archetype 5**. That's the same selection logic the paper used to pick Howard (72% Shooter Specialist, the highest concentration it observed in EuroLeague).

Archetype 5, per this project's own archetype-to-paper mapping (`data/basis_2025_26/archetype_labels.csv`, built in `src/step1_archetype_model.py`'s `label_archetypes_vs_paper`), corresponds to the paper's own **"Shooting Specialist"** archetype — a strong match: MPJ's top feature signature (off-screen/handoff usage, low self-creation, high 3PA share, low STL%/TRB%) lines up closely with the paper's own description ("movement or spot-up shooter... low self-creation, low 2FGA%"). MPJ's exemplar teammate-in-feature-space is Tim Hardaway Jr.

## The 8 archetypes, for reference

Fit via Archetypoid Analysis (ADA) on 2025-26 regular-season player-season rows, MIN≥300 (K=8, `data/basis_2025_26/`). Cross-referenced against the paper's own NBA archetype table (Section 5.2):

| # | exemplar | paper label | match |
|---|---|---|---|
| 0 | Nicolas Batum | 3&D Wing | strong |
| 1 | Jonas Valanciunas | Inside Scoring Big | strong |
| 2 | Clint Capela | Rim Protector / Roll Man | strong |
| 3 | D'Angelo Russell | Combo Guard | strong |
| 4 | Ryan Kalkbrenner | Mobile Big | partial — our data doesn't show the paper's "high corner-3%" signal |
| 5 | Tim Hardaway Jr. | **Shooting Specialist** | strong |
| 6 | Shai Gilgeous-Alexander | Offensive Engine | strong |
| 7 | Bez Mbeng | Traditional Playmaker | partial — reads more as a low-usage defensive guard |

K=8 itself: the paper's own deciding criterion (Section 4.3) is intra-archetype variance, not the plain RSS elbow (which the paper explicitly calls misleading — it lands too early). Re-run on our exact 2025-26/MIN≥300 population (`figures/k_selection_2025_26_intra_variance.png`), our data doesn't show as clean a minimum as the paper's (ours keeps slowly decreasing through K=12 rather than turning back up at K=9), but shows the same *diminishing-marginal-returns* shape the paper's reasoning relies on — most of the coherence gain is already captured by K=8. (Single-seed exploratory run; a 3-seed re-run would be more rigorous if this figure goes in the final write-up.)

## A. Macro view — archetype exposure

% of MPJ's real 2025-26 shared court time (weighted by shared minutes) spent with each archetype, computed from every real 2-man on-court pairing he was part of:

| archetype | exposure |
|---|---|
| Combo Guard | 31.3% |
| 3&D Wing | 16.2% |
| Rim Protector / Roll Man | 14.7% |
| Traditional Playmaker | 12.5% |
| Shooting Specialist (his own type) | 12.1% |
| Inside Scoring Big | 6.7% |
| Offensive Engine | 3.4% |
| Mobile Big | 3.0% |

MPJ himself is a Shooting Specialist, but only 12.1% of his shared minutes come alongside other Shooting Specialists — nearly a third of his floor time (31.3%) was spent with Combo Guards.

*(Caveat: this is a share of pairwise shared-minute weight, summing to 100% by construction — not corrected for double-counting when 3+ teammates share the floor simultaneously.)*

## B. Micro view — primary real teammates

MPJ's three most-played-with real teammates in 2025-26, and each one's own archetype makeup:

| teammate | shared min | % of MPJ's own season | archetype |
|---|---|---|---|
| Nic Claxton | 1264 | 74.9% | Rim Protector/Roll Man 36.2%, Inside Scoring Big 24.4% |
| Noah Clowney | 1146 | 67.9% | Shooting Specialist 35.4%, Combo Guard 23.5% |
| Egor Dëmin | 924 | 54.7% | Combo Guard 48.1%, 3&D Wing 30.7% |

MPJ's single most common partner (75% of his season) was Claxton — a non-shooting, paint-bound vertical big. *(Note: Claxton was traded away this past offseason — 2025-26 real data reflects last year's roster, not the current one. Doesn't invalidate this case study, since MPJ genuinely played these minutes, but matters for how directly this maps onto next season's roster.)*

## C. Real lineup performance

Real on-court net ratings for MPJ's actual 2-man and 3-man combinations with these three teammates:

| combo | MIN | ORtg | DRtg | NRtg |
|---|---|---|---|---|
| MPJ + Claxton | 1264 | 110.5 | 120.8 | **-10.2** |
| MPJ + Clowney | 1146 | 112.2 | 116.1 | -3.8 |
| MPJ + Dëmin | 924 | 112.8 | 117.4 | -4.5 |
| MPJ + Claxton + Clowney | 913 | 110.8 | 117.4 | -6.6 |
| MPJ + Claxton + Dëmin | 757 | 111.4 | 118.5 | -7.1 |
| MPJ + Clowney + Dëmin (no Claxton) | 775 | 114.1 | 115.0 | **-0.8** |

Every combination is net-negative (consistent with the paper's own Howard case: "every major pairing yields a negative net rating"), but there's a clear pattern within that: **every combo containing Claxton is meaningfully worse than every combo without him**, and MPJ's single best real trio (-0.8, nearly neutral) is the one that excludes Claxton entirely.

## D. League-wide benchmark

Top-20 league-wide players at ≥55% Shooting Specialist probability (MIN≥300), filtered to those with a positive real on-court net rating this season (the paper's own filter; its second, OR'd criterion — positive on/off differential — isn't computable from data this project has pulled, so this uses the on-court half only):

**11/20 qualified**: Tim Hardaway Jr., Duncan Robinson, Isaiah Joe, Luke Kennard, Sam Hauser, Devin Vassell, Sam Merrill, Donte DiVincenzo, Moses Moody, Stephen Curry, Cameron Johnson.

Averaging these 11 players' own macro archetype exposure gives a league "ideal ecosystem" baseline for a Shooting Specialist. Comparing MPJ's actual exposure against it:

| archetype | league baseline | MPJ actual | gap |
|---|---|---|---|
| Combo Guard | 15.6% | 31.3% | **+15.7pp** |
| Offensive Engine | 15.3% | 3.4% | **-11.8pp** |
| Rim Protector / Roll Man | 11.7% | 14.7% | +3.1pp |
| Inside Scoring Big | 10.6% | 6.7% | -3.9pp |
| Shooting Specialist | 15.1% | 12.1% | -3.0pp |
| Mobile Big | 5.4% | 3.0% | -2.4pp |
| Traditional Playmaker | 10.6% | 12.5% | +1.9pp |
| 3&D Wing | 15.7% | 16.2% | +0.5pp |

## Synthesis

Three independent angles on this roster's construction all point the same direction:

1. **Combo Guard oversupply** — flagged by (a) this project's earlier archetype-overlap pass across all real Nets pairs (six of six flagged "mismatch" pairs involved two Combo Guard-type players: Tyrese Martin, Drake Powell, Jalen Wilson, Terance Mann, Cam Thomas, Ben Saraf, Danny Wolf), (b) MPJ's own macro exposure (31.3% of his shared time, vs. 15.6% league-typical for his archetype), and (c) the benchmark comparison directly (+15.7pp, the single largest gap of any archetype).
2. **Offensive Engine scarcity** — the benchmark comparison's second-largest gap (-11.8pp). Successful Shooting Specialists league-wide typically share real minutes with a true lead creator; MPJ was largely without one in 2025-26. This is directly relevant to the open question of who initiates offense for this roster going forward.
3. A specific, real, named mismatch: **MPJ + Nic Claxton** was his most-played pairing (75% of his season) and his single worst 2-man combination (-10.2 NRtg); his best real combination excluded Claxton entirely.

## Limitations / adaptations from the paper's exact method (state plainly, don't overclaim)

- Single NBA season (2025-26, MIN≥300) — not the paper's own longer training window; K-selection and archetype fits are validated on this smaller population.
- Macro exposure uses each teammate's full soft archetype recipe (probability-weighted), not a hard argmax assignment — a modeling choice consistent with this project's probabilistic framing, not something the paper's text pins down explicitly either way.
- The league-benchmark "successful comparable" filter uses only the on-court-net-rating half of the paper's OR'd criterion (positive NRtg on court OR positive on/off differential) — no on/off split exists in this project's pulled data, so the differential half is left out rather than approximated.
- 2 of 8 archetypes (Mobile Big, Traditional Playmaker) are honest partial matches to the paper's NBA table, not forced clean ones.
- Nic Claxton was traded away this offseason — this case study's real-data teammates reflect the 2025-26 roster, which has since changed; the *analysis* of MPJ's fit issues stands, but doesn't map onto next season's roster 1:1.

## Next

Scope a portal so this same A→E pipeline can be run live against any player, not just MPJ as a frozen example.
