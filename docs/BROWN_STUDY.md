# Building Around Mikel Brown Jr. — a Scouting/Deployment Study

Brooklyn selected Mikel Brown Jr. (Louisville PG, one NCAA season, 2025-26)
at No. 6 in the 2026 draft. Research question: how should Brooklyn deploy
and staff lineups around him in 2026-27 — and which current players clash
with that plan? Real tension going in: Brooklyn's roster is already
guard-heavy (this project's own Roster Construction composition analysis
flags a Combo Guard surplus), and this adds another lead guard.

**Labeling convention** (per `docs/RESEARCH_FINDINGS.md`'s design
principle): every claim below is tagged **[validated-quantitative]**
(comes from the talent-aggregation engine, the one piece of this project's
modeling shown to have real out-of-sample predictive power — see
`RESEARCH_FINDINGS.md` §5) or **[descriptive]** (archetype composition,
similarity, or pooled-evidence language — informative, not predictive; no
numeric performance claim rides on it alone).

## Gate B0 — data feasibility (passed)

Verified live against a real CBBD API key: Brown's full 2025-26 season
(21 games, 613 minutes, real box score) is present; his own shot
coordinates are available at 97.1% coverage (265/273 real FGA, cross-checked
exactly against his season FGA/3PA totals). Of the NBA archetype model's 29
features, **20 are computable** for the college side (11 box-score/advanced
+ 9 shot-location, one — BPM — via a documented substitute, PORPAG); the 9
Synergy play-type features are confirmed unavailable (CBBD has no
Synergy-style play-type data, only generic play-by-play event types).

Because the feature set differs from the NBA model's original 29, a
**reduced, shared-feature K=8 basis was retrained** on the same 2025-26 NBA
population (434 players) using only the 20 shared features — Brown is never
projected onto axes the college side can't populate. Real exemplars:
Moussa Cisse, Neemias Queta, Matisse Thybulle, Kam Jones, Tony Bradley,
DeMar DeRozan, Luka Dončić, Doug McDermott.

## Phase B1 — his profile [descriptive]

Box-score features z-scored against a real NCAA D1 population (n=3,114,
MIN≥300 this season). Brown's z-scores: PTS_PER_100 **+2.10**, USG% **+2.24**,
AST% **+2.39** — elite scoring rate, usage, and playmaking, all near the top
of a 3,000+ player D1 population. TRB% **-0.76**, BLK% **-0.65** — below
average, expected for a 6'3" (75in, z=-1.13 vs. the *NBA* training
population) guard. Shot profile (from his own real coordinates, NCAA
3PT arc = 22.15ft, not the NBA's 23.75/22ft — documented, not silently
reused): 58.5% of his attempts are threes, 23.2% of those from the corner;
only 23.6% of his 2s and 50.9% of his 3s are assisted — he creates most of
his own offense even from three.

**Projection onto the reduced NBA basis**: **59.1% Luka Dončić-type**
(elite offensive engine — high usage/scoring/BPM/AST%, low assisted rate),
with secondary shares in archetype 3 (18.2%, Kam Jones-type — ball-handling
but low efficiency), archetype 7 (11.3%, Doug McDermott-type — shooter),
archetype 2 (8.8%, Matisse Thybulle-type — perimeter defense/steals).
Reconstruction residual (novelty flag): 0.698 average per-feature
standardized gap — a real, moderate figure, not treated as zero.

**Sanity check**: this profile — primary shot creator, high usage,
efficient, heavy 3PT rate, mostly self-created shots — matches his public
scouting reports as a shot-creating lead guard. It does not say
"rim-protecting big." Passed.

Three features in this vector have weaker population support than the
rest, flagged explicitly rather than hidden: `Dist.`, `Corner 3s_%3PA`, and
`PLAYER_HEIGHT_INCHES` were z-scored against the *NBA* training population
(not a real NCAA population) because no bulk NCAA population baseline was
available for them within this task's scope — height is league-invariant
so this is actually the *more* correct choice for that one feature; `Dist.`
and `Corner 3s_%3PA` are the two weakest-support features in the vector.

## Phase B2 — comparables [descriptive]

Cosine similarity + Euclidean distance (both reported, per spec — this
project's own archetype-similarity convention elsewhere uses
Jensen-Shannon distance; cosine/Euclidean were used here as explicitly
requested for this deliverable) between Brown's alpha vector and:

**(a) All current (2025-26) NBA players**, top 5 of 10: Jayson Tatum
(0.988), LaMelo Ball (0.986), Donovan Mitchell (0.984), James Harden
(0.971), Cade Cunningham (0.970). All ball-dominant shot-creators —
consistent with the projection.

**(b) Rookie-season profiles across the 3 available seasons** (2023-24,
2024-25, 2025-26 draft classes; 221 real rookie-season profiles found),
top comps: Javon Small (2025-26, 0.836), Jared McCain (2024-25, 0.812),
Cedric Coward (2025-26, 0.729), Tyrese Proctor (2025-26, 0.704), Stephon
Castle (2024-25, 0.698), Dylan Harper (2025-26, 0.692), Victor Wembanyama
(2023-24, 0.644 — a real, honestly-low outlier; his archetype match is far
weaker than the guards above, and shouldn't be over-read as a stylistic
comp despite being a famous rookie season).

No comp in either list is disproportionately driven by an archetype Brown
himself carries near-zero weight in (his lowest-weight archetypes — 0, 1,
4, 5, all big-man types — don't feature meaningfully among the comps
above, all of whom are guards/wings).

## Phase B3 — success-environment pattern [descriptive — evidence synthesis, NOT causal]

Reused this project's own existing similarity-weighted pooling machinery
(`compute_style_pool_by_vector` + `similarity_weighted_benchmark`,
Jensen-Shannon distance, power=4.0), applied to Brown's own projected
vector across all 2025-26 NBA players — a possession/pair-weighted pool
(n=434, effective sample size ≈104.5), not just an average of the 10 named
comps above, at the granularity where this project's broader investigation
(`RESEARCH_FINDINGS.md`) found real signal survives.

Comparing the unconditional teammate-archetype baseline against the same
pool restricted to players whose own on-court NET_RATING is positive
(n=207, ESS≈50.2) gives a real, ranked "success pattern":

| Teammate archetype | Δ (positive-net minus unconditional) | Roster supply |
|---|---|---|
| Luka Dončić-type (co-star creator) | **+1.6pp** | 2 (Randle, MPJ) |
| Neemias Queta-type (rim protection) | +1.0pp | 3 |
| Matisse Thybulle-type (perimeter D) | +0.6pp | **0** |
| Doug McDermott-type (shooting) | +0.3pp | 7 |
| Moussa Cisse-type | -0.0pp | 0 |
| Tony Bradley-type | -0.5pp | 0 |
| DeMar DeRozan-type | -0.6pp | 0 |
| Kam Jones-type (inefficient ball-handler) | **-2.5pp** | 4 |

**A real, specific finding, not the generic assumption**: the data does
*not* say "avoid a second guard" — pairing Brown with *another elite
creator* (Dončić-type) is the single strongest positive association. What's
negative is specifically the *low-efficiency* ball-handler archetype
(Kam Jones-type: high TOV%/AST% but low TS%/BPM/FTr) — a redundancy-of-
mediocrity problem, not a redundancy-of-role problem.

## Phase B4 — deployment on the current roster

**[validated-quantitative] numbers**: talent = possession-agnostic sum of
each player's exogenous `skill_off`/`skill_def` (`build_skill.py`'s
already-validated, prior-season, leakage-checked output — the same source
validated at the team-season level in `RESEARCH_FINDINGS.md`, not the
RAPM/archetype-synergy model, which failed its own validation gate and is
explicitly out of scope for numeric claims). Brown's own talent: draft-slot
prior at pick 6 → skill_off=-1.42, skill_def=-0.57 (a real, historically-
grounded floor for a top-6 pick's *rookie-year* expected impact, not his
ceiling).

**[descriptive] constraint flags**, grounded in Phase B3's real findings
(not generic archetype presence): needs rim protection (archetype 0 or 1
present), needs floor-spacing (archetype 7 present), second-guard
redundancy flag (archetype 3 — the specifically inefficient type — present).

Enumerated all C(16,4)=1,820 four-man complements from the 16
rotation-eligible (real 2025-26 recipe) Nets players. **364 pass every
constraint.** Top 3 by talent, all constraints passing:

1. Josh Minott, Julius Randle, Keon Ellis, Moritz Wagner (talent=3.48)
2. Day'Ron Sharpe, Julius Randle, Keon Ellis, Moritz Wagner (talent=3.28)
3. Day'Ron Sharpe, Josh Minott, Keon Ellis, Moritz Wagner (talent=3.00)

**Near-misses shown, not hidden**: (Day'Ron Sharpe, Julius Randle, Michael
Porter Jr., Moritz Wagner) — highest raw talent among all non-passing
groups (1.7) — violates "no floor-spacing shooter." (Julius Randle, Keon
Ellis, Moritz Wagner, Terance Mann) — violates the redundancy flag.

**Incompatibility list** (current players whose *dominant* archetype is
specifically the inefficient-ball-handler type, ≥30% share):

| Player | Archetype 3 share | Reason |
|---|---|---|
| Ben Saraf | 74% | Dominant type is the specific archetype Phase B3 found associated with worse outcomes alongside Brown's style |
| Nolan Traoré | 81% | Same |
| Drake Powell | 39% | Same |
| Terance Mann | 40% | Same |

This is a role-fit flag from the descriptive layer, not a trade or
roster-cut recommendation — no salary/contract modeling is part of this
project, per CLAUDE.md's own stated scope.

## Phase B5 — gap note [descriptive]

Success pattern (B3) minus roster supply (B4): the roster is **heaviest on
the smallest positive archetype** (McDermott-type shooting, +0.3pp, 7
players) and **completely missing the third-largest positive archetype**
(Thybulle-type perimeter defense, +0.6pp, 0 players). The single strongest
positive archetype (Dončić-type co-star creation, +1.6pp) is already
real-supplied by Randle and Porter Jr. — not a gap.

**Headline gap: a Matisse Thybulle-type perimeter defender/steals
specialist.** No target board is produced — no external candidate-pool file
was provided, and this project does not model salary/contract fit; per
CLAUDE.md's own scope, this is a fit shortlist framing, never trade advice.

## Summary

Brown projects as a real, efficient, high-usage shot-creating lead guard
(59.1% Dončić-type) — not a traditional table-setting point guard, and not
redundant with the roster's *existing* creator (Randle/MPJ pairing already
supplies the single most positively-associated archetype). The actual
incompatibility risk is narrower than "too many guards": four specific
players (Saraf, Traoré, Powell, Mann) carry the *inefficient* ball-handler
signature the pooled evidence associates with worse outcomes. 364 real
4-man units satisfy real constraint checks; the clearest remaining team
need is perimeter defense, not addressed by this roster at all.
