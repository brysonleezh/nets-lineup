# College Archetype Interpretation (K=8, AA)

Frozen fit: `data/college/model/k8_frozen_basis.npz` (consensus-basin seed=2, see `reports/phase2_worklog.md` for the basin analysis that produced this choice). Method: AA (not ADA) - see CHECKPOINT 2 in the worklog for why (7/8 restricted-search archetypoids landed on low-major conference outliers).


**Labels finalized 2026-08-14** (owner-confirmed, adopting the 2026-08-13 AI-drafted candidates as-is, grounded in each archetype's real z-profile and top-loading players - see `reports/phase3_worklog.md`'s labeling discussion). Archetypes 0, 1, and 7 remain flagged as weak/noisy rather than forced into a clean style - their top loaders are undifferentiated low-major bench players with no coherent basketball story, not a labeling gap.


## Archetype 0

**label:** Low-Minute Statistical Outlier (weak — see note above)


**Mean z-profile (all 12 features):**

| Feature | z |
|---|---|
| PLAYER_HEIGHT_INCHES | -0.84 |
| PTS_PER_100 | -2.39 |
| TS% | +0.10 |
| USG% | -2.45 |
| AST% | +1.37 |
| TOV% | +6.25 |
| STL% | -0.36 |
| BLK% | -0.75 |
| TRB% | -1.20 |
| FTr | +3.70 |
| % of FG Ast'd_2P | -1.81 |
| % of FG Ast'd_3P | +0.83 |

**Distinguishing features (top 5 by \|z\|):**

- TOV%: +6.25
- FTr: +3.70
- USG%: -2.45
- PTS_PER_100: -2.39
- % of FG Ast'd_2P: -1.81

**Top-10 loading player-seasons (by alpha_0):**

| Player | Season | Team | Conference | Minutes | alpha |
|---|---|---|---|---|---|
| Donnie Lewis | 2018 | Southeast Missouri State | OVC | 315 | 0.993 |
| Wiggy Ball | 2026 | UL Monroe | Sun Belt | 316 | 0.928 |
| Ben Knostman | 2022 | Lehigh | Patriot | 335 | 0.898 |
| Bryan Gee | 2017 | Longwood | Big South | 339 | 0.721 |
| Cameron Parker | 2019 | Sacred Heart | NEC | 886 | 0.710 |
| Bijan Cortes | 2022 | Oklahoma | Big 12 | 346 | 0.706 |
| Oliver Tot | 2017 | William & Mary | CAA | 312 | 0.695 |
| Daniel Peace | 2017 | Troy | Sun Belt | 542 | 0.677 |
| Drew King | 2026 | Youngstown State | Horizon | 557 | 0.675 |
| A.J. Patterson | 2026 | Air Force | Mountain West | 415 | 0.674 |

**Apparent correspondence to paper's European archetypes:** None cleanly - extreme TOV%+FTr combined with low USG%/scoring isn't a real basketball role; smallest archetype by population share (4.2%), all top loaders are undifferentiated 300-900 min low-major bench players. Reads as a small-sample noise bucket rather than a paper-archetype match.


## Archetype 1

**label:** Inefficient Low-Usage Reserve (weak — see note above)


**Mean z-profile (all 12 features):**

| Feature | z |
|---|---|
| PLAYER_HEIGHT_INCHES | -0.08 |
| PTS_PER_100 | -2.25 |
| TS% | -4.35 |
| USG% | -1.72 |
| AST% | -0.87 |
| TOV% | +0.43 |
| STL% | -0.72 |
| BLK% | -0.49 |
| TRB% | -0.65 |
| FTr | -0.95 |
| % of FG Ast'd_2P | +1.09 |
| % of FG Ast'd_3P | +0.75 |

**Distinguishing features (top 5 by \|z\|):**

- TS%: -4.35
- PTS_PER_100: -2.25
- USG%: -1.72
- % of FG Ast'd_2P: +1.09
- FTr: -0.95

**Top-10 loading player-seasons (by alpha_1):**

| Player | Season | Team | Conference | Minutes | alpha |
|---|---|---|---|---|---|
| Jack Webb | 2022 | Central Michigan | MAC | 353 | 0.979 |
| Ernest Minton | 2024 | Mississippi Valley State | SWAC | 383 | 0.968 |
| Levi Beckwith | 2026 | Maryland Eastern Shore | MEAC | 449 | 0.861 |
| Mason Miller | 2025 | Creighton | Big East | 376 | 0.860 |
| Chris Collins | 2019 | Texas A&M | SEC | 367 | 0.835 |
| Sione Lose | 2023 | UC Davis | Big West | 344 | 0.833 |
| Christian Terrell | 2017 | UC Santa Barbara | Big West | 496 | 0.832 |
| Justin Hinds | 2026 | Air Force | Mountain West | 333 | 0.829 |
| Garrett Gilkeson | 2017 | VMI | SoCon | 356 | 0.827 |
| Sam Bittner | 2017 | Fresno State | Mountain West | 341 | 0.821 |

**Apparent correspondence to paper's European archetypes:** Closest loose fit is **Role Guard** - low usage, low everything - but the -4.35 TS% is extreme and all top loaders are deep-bench low-major players with no other distinguishing stat. Second-largest weak/noisy archetype; treat this correspondence as tentative.


## Archetype 2

**label:** Rim-Protecting Big / Shot-Blocking Center


**Mean z-profile (all 12 features):**

| Feature | z |
|---|---|
| PLAYER_HEIGHT_INCHES | +2.44 |
| PTS_PER_100 | -0.87 |
| TS% | +1.33 |
| USG% | -1.33 |
| AST% | -1.32 |
| TOV% | +0.61 |
| STL% | -0.69 |
| BLK% | +3.63 |
| TRB% | +2.42 |
| FTr | +1.59 |
| % of FG Ast'd_2P | +1.52 |
| % of FG Ast'd_3P | -2.56 |

**Distinguishing features (top 5 by \|z\|):**

- BLK%: +3.63
- % of FG Ast'd_3P: -2.56
- PLAYER_HEIGHT_INCHES: +2.44
- TRB%: +2.42
- FTr: +1.59

**Top-10 loading player-seasons (by alpha_2):**

| Player | Season | Team | Conference | Minutes | alpha |
|---|---|---|---|---|---|
| Jamarion Sharp | 2022 | Western Kentucky | CUSA | 895 | 1.000 |
| Jamarion Sharp | 2023 | Western Kentucky | CUSA | 913 | 1.000 |
| Ike Obiagu | 2018 | Florida State | ACC | 364 | 1.000 |
| Malcolm Wilson | 2025 | Queens University | ASUN | 779 | 1.000 |
| Kaodirichi Akobundu-Ehiogu | 2021 | UT Arlington | Sun Belt | 393 | 1.000 |
| Jamarion Sharp | 2024 | Ole Miss | SEC | 491 | 1.000 |
| Kaodirichi Akobundu-Ehiogu | 2023 | Memphis | American | 318 | 1.000 |
| Ari Boya | 2020 | Bradley | MVC | 326 | 1.000 |
| Gabe Dynes | 2025 | Youngstown State | Horizon | 741 | 1.000 |
| Gabe Dynes | 2024 | Youngstown State | Horizon | 368 | 1.000 |

**Apparent correspondence to paper's European archetypes:** **Traditional Center** (high confidence) - one of the cleanest archetypes in the fit: extreme BLK%, height, and TRB% together, most top loaders at alpha≈1.00 (Jamarion Sharp, Ike Obiagu, Malcolm Wilson). Defensive Specialist is a plausible secondary label but Traditional Center fits the height+rebounding combination better.


## Archetype 3

**label:** Efficient Low-Usage Play-Finisher


**Mean z-profile (all 12 features):**

| Feature | z |
|---|---|
| PLAYER_HEIGHT_INCHES | +0.57 |
| PTS_PER_100 | -0.16 |
| TS% | +3.52 |
| USG% | -1.42 |
| AST% | -0.97 |
| TOV% | -1.74 |
| STL% | +0.17 |
| BLK% | -0.70 |
| TRB% | -0.18 |
| FTr | -1.30 |
| % of FG Ast'd_2P | +2.82 |
| % of FG Ast'd_3P | +0.71 |

**Distinguishing features (top 5 by \|z\|):**

- TS%: +3.52
- % of FG Ast'd_2P: +2.82
- TOV%: -1.74
- USG%: -1.42
- FTr: -1.30

**Top-10 loading player-seasons (by alpha_3):**

| Player | Season | Team | Conference | Minutes | alpha |
|---|---|---|---|---|---|
| Keller Boothby | 2022 | Cornell | Ivy | 600 | 1.000 |
| JR Hobbie | 2017 | Brown | Ivy | 486 | 0.859 |
| Chase Johnston | 2026 | High Point | Big South | 456 | 0.857 |
| Austin Loop | 2017 | Marshall | CUSA | 1092 | 0.822 |
| Destin Whitaker | 2022 | Fresno State | Mountain West | 370 | 0.773 |
| Tate Pierson | 2020 | Belmont | OVC | 353 | 0.733 |
| Devin Jensen | 2020 | Merrimack | NEC | 738 | 0.720 |
| Austin McCullough | 2021 | Campbell | Big South | 399 | 0.699 |
| Everett Duncan | 2018 | Vermont | Am. East | 990 | 0.691 |
| Matt Fox | 2018 | Bowling Green | MAC | 516 | 0.690 |

**Apparent correspondence to paper's European archetypes:** **Role Guard** - high TS%, heavily assisted 2P%, low usage, low turnovers: a low-usage player who finishes what's created for him efficiently. Note: this is the archetype flagged in the Phase 2 sensitivity check as NOT robust to the conference-restriction flag (cosine=-0.18 off) - treat any downstream use with that caveat in mind.


## Archetype 4

**label:** Ball-Hawking Defensive Guard


**Mean z-profile (all 12 features):**

| Feature | z |
|---|---|
| PLAYER_HEIGHT_INCHES | -1.42 |
| PTS_PER_100 | +0.30 |
| TS% | -1.05 |
| USG% | +0.92 |
| AST% | +1.25 |
| TOV% | -0.02 |
| STL% | +4.85 |
| BLK% | -0.36 |
| TRB% | -0.17 |
| FTr | -0.50 |
| % of FG Ast'd_2P | -1.15 |
| % of FG Ast'd_3P | +0.14 |

**Distinguishing features (top 5 by \|z\|):**

- STL%: +4.85
- PLAYER_HEIGHT_INCHES: -1.42
- AST%: +1.25
- % of FG Ast'd_2P: -1.15
- TS%: -1.05

**Top-10 loading player-seasons (by alpha_4):**

| Player | Season | Team | Conference | Minutes | alpha |
|---|---|---|---|---|---|
| Paris Collins | 2017 | Jackson State | SWAC | 773 | 1.000 |
| Nigel Ribeiro | 2017 | Grambling | SWAC | 849 | 1.000 |
| Ivy Smith Jr. | 2017 | Grambling | SWAC | 325 | 1.000 |
| Paris Collins | 2018 | Jackson State | SWAC | 728 | 1.000 |
| Jamall Gregory | 2019 | Jacksonville State | OVC | 673 | 0.947 |
| Ivy Smith Jr. | 2020 | Grambling | SWAC | 1042 | 0.938 |
| Diontae Jones | 2017 | Grambling | SWAC | 318 | 0.937 |
| Kameron Langley | 2021 | North Carolina A&T | MEAC | 593 | 0.934 |
| Fred Cleveland Jr. | 2021 | North Carolina A&T | MEAC | 317 | 0.928 |
| Ahmad Thomas | 2017 | UNC Asheville | Big South | 1051 | 0.913 |

**Apparent correspondence to paper's European archetypes:** **Defensive Specialist** (high confidence) - extreme STL% is by far the dominant feature, paired with below-average height and modest positive AST% (a smaller, ball-hawking perimeter defender rather than a scorer).


## Archetype 5

**label:** High-Usage Primary Ball-Handler


**Mean z-profile (all 12 features):**

| Feature | z |
|---|---|
| PLAYER_HEIGHT_INCHES | -2.01 |
| PTS_PER_100 | +2.69 |
| TS% | -0.01 |
| USG% | +3.22 |
| AST% | +3.62 |
| TOV% | -0.28 |
| STL% | +0.20 |
| BLK% | -0.67 |
| TRB% | -0.72 |
| FTr | -0.35 |
| % of FG Ast'd_2P | -1.66 |
| % of FG Ast'd_3P | -2.01 |

**Distinguishing features (top 5 by \|z\|):**

- AST%: +3.62
- USG%: +3.22
- PTS_PER_100: +2.69
- PLAYER_HEIGHT_INCHES: -2.01
- % of FG Ast'd_3P: -2.01

**Top-10 loading player-seasons (by alpha_5):**

| Player | Season | Team | Conference | Minutes | alpha |
|---|---|---|---|---|---|
| Gus Etchison | 2026 | Idaho State | Big Sky | 423 | 1.000 |
| Daeshun Ruffin | 2026 | Jackson State | SWAC | 906 | 0.950 |
| Trae Young | 2018 | Oklahoma | Big 12 | 1133 | 0.914 |
| Tavian Dunn-Martin | 2022 | Florida Gulf Coast | ASUN | 1109 | 0.883 |
| Marcus Keene | 2017 | Central Michigan | MAC | 1179 | 0.873 |
| Loren Cristian Jackson | 2021 | Akron | MAC | 812 | 0.837 |
| R.J. Cole | 2018 | Howard | MEAC | 1252 | 0.814 |
| Junior Robinson | 2018 | Mount St. Mary's | NEC | 1127 | 0.810 |
| Sharife Cooper | 2021 | Auburn | SEC | 397 | 0.808 |
| Jermaine Marrow | 2020 | Hampton | Big South | 1058 | 0.801 |

**Apparent correspondence to paper's European archetypes:** **Traditional Playmaker** / **High Usage Guard** hybrid (high confidence on the archetype being real, split on which paper label fits best) - AST% is the single most dominant feature, but USG% and PTS_PER_100 are nearly as extreme, so this reads as a ball-dominant, high-usage point guard rather than a pure distributor. Top loaders are elite, real names: **Trae Young (0.91)**, Marcus Keene, R.J. Cole, Sharife Cooper. Mikel Brown Jr.'s college recipe loads primarily here (0.57) - consistent with his "on-ball engine" scouting profile.


## Archetype 6

**label:** High-Usage Interior Scorer


**Mean z-profile (all 12 features):**

| Feature | z |
|---|---|
| PLAYER_HEIGHT_INCHES | +2.06 |
| PTS_PER_100 | +3.25 |
| TS% | +1.22 |
| USG% | +2.90 |
| AST% | -0.08 |
| TOV% | -0.68 |
| STL% | -1.08 |
| BLK% | +1.20 |
| TRB% | +2.59 |
| FTr | +1.60 |
| % of FG Ast'd_2P | +1.02 |
| % of FG Ast'd_3P | +0.80 |

**Distinguishing features (top 5 by \|z\|):**

- PTS_PER_100: +3.25
- USG%: +2.90
- TRB%: +2.59
- PLAYER_HEIGHT_INCHES: +2.06
- FTr: +1.60

**Top-10 loading player-seasons (by alpha_6):**

| Player | Season | Team | Conference | Minutes | alpha |
|---|---|---|---|---|---|
| Zach Edey | 2024 | Purdue | Big Ten | 1248 | 0.997 |
| Nathan Knight | 2020 | William & Mary | CAA | 948 | 0.886 |
| Kevin Obanor | 2019 | Oral Roberts | Summit | 627 | 0.871 |
| Michael Buchanan | 2017 | South Carolina Upstate | ASUN | 720 | 0.871 |
| Oscar Cluff | 2025 | South Dakota State | Summit | 830 | 0.849 |
| Luka Garza | 2020 | Iowa | Big Ten | 991 | 0.840 |
| Jock Landale | 2018 | Saint Mary's | WCC | 1199 | 0.835 |
| Filip Petrusev | 2020 | Gonzaga | WCC | 857 | 0.834 |
| Jordan Washington | 2017 | Iona | MAAC | 760 | 0.826 |
| Jordan Brown | 2023 | Louisiana | Sun Belt | 1052 | 0.816 |

**Apparent correspondence to paper's European archetypes:** **High Usage Forward** (high confidence) - PTS_PER_100, USG%, TRB%, and height are all strongly positive together: a tall, high-usage, high-volume scorer who also rebounds - the offensive focal point among bigs, distinct from Archetype 2's defense-first rim protector. Top loaders: **Zach Edey (0.997)**, Luka Garza, Jock Landale, Filip Petrusev.


## Archetype 7

**label:** Low-Event Floor Role Player (weak/catch-all — see note above)


**Mean z-profile (all 12 features):**

| Feature | z |
|---|---|
| PLAYER_HEIGHT_INCHES | -0.59 |
| PTS_PER_100 | -0.62 |
| TS% | +0.65 |
| USG% | -1.17 |
| AST% | -1.36 |
| TOV% | -2.15 |
| STL% | -1.78 |
| BLK% | -0.79 |
| TRB% | -1.79 |
| FTr | -1.84 |
| % of FG Ast'd_2P | -1.85 |
| % of FG Ast'd_3P | +0.74 |

**Distinguishing features (top 5 by \|z\|):**

- TOV%: -2.15
- % of FG Ast'd_2P: -1.85
- FTr: -1.84
- TRB%: -1.79
- STL%: -1.78

**Top-10 loading player-seasons (by alpha_7):**

| Player | Season | Team | Conference | Minutes | alpha |
|---|---|---|---|---|---|
| Jonah Jackson | 2020 | Drake | MVC | 727 | 0.997 |
| Chris Ashby | 2024 | Queens University | ASUN | 884 | 0.857 |
| Djordjije Mumin | 2018 | UCF | American | 338 | 0.814 |
| Bryan Trimble Jr. | 2018 | St. John's | Big East | 555 | 0.796 |
| Sam Hunt | 2018 | NC State | ACC | 530 | 0.785 |
| Carson Bischoff | 2022 | UT Arlington | Sun Belt | 407 | 0.784 |
| Jaden Schutt | 2026 | Virginia Tech | ACC | 792 | 0.782 |
| Kahlil Singleton | 2024 | Holy Cross | Patriot | 455 | 0.781 |
| Dillon Avare | 2018 | Eastern Kentucky | OVC | 560 | 0.778 |
| Connor Kern | 2017 | Arkansas State | Sun Belt | 545 | 0.773 |

**Apparent correspondence to paper's European archetypes:** No clean match - largest archetype by population share (18.7%) but no single feature stands out (all |z| < 2.2); reads as a catch-all "low mistakes, low impact" bucket rather than a distinct paper archetype. Closest loose fit would be **Role Guard**, but with much weaker signal than Archetype 1's version of that label.

## Sanity projections

Eyeball plausibility only - not a validation, not a translation claim. For the 3 rookies: does this college recipe look like the player, nothing about what it means in NBA-archetype space (that is a later phase).

| Player | Season | Team | Kind | Top-3 archetypes |
|---|---|---|---|---|
| Zach Edey | 2024 | Purdue | fixture | arch_6=1.00, arch_2=0.00, arch_0=0.00 |
| Brandon Miller | 2023 | Alabama | fixture | arch_6=0.47, arch_7=0.29, arch_5=0.12 |
| Stephon Castle | 2024 | UConn | fixture | arch_6=0.26, arch_5=0.19, arch_3=0.17 |
| Cooper Flagg | 2025 | Duke | fixture | arch_6=0.55, arch_5=0.21, arch_4=0.21 |
| Frank Mason III | 2017 | Kansas | fixture | arch_5=0.55, arch_3=0.22, arch_0=0.10 |
| Trae Young | 2018 | Oklahoma | fixture | arch_5=0.91, arch_0=0.07, arch_6=0.02 |
| Zion Williamson | 2019 | Duke | fixture | arch_6=0.53, arch_4=0.30, arch_3=0.13 |
| Ja Morant | 2019 | Murray State | fixture | arch_5=0.78, arch_0=0.12, arch_6=0.10 |
| Mikel Brown Jr. | 2026 | Louisville | rookie | arch_5=0.57, arch_6=0.18, arch_0=0.11 |
| Tyler Bilodeau | 2026 | UCLA | rookie | arch_6=0.52, arch_3=0.33, arch_7=0.11 |
| Joshua Jefferson | 2026 | Iowa State | rookie | arch_6=0.48, arch_4=0.35, arch_0=0.08 |

---

# Appendix — Model Provenance, Special Cases & Known Limitations

*Append to `reports/college_archetypes.md`. This appendix is the canonical register of what the college archetype model is, what it structurally cannot see, and every special case resolved during Phases 1–3. Downstream reports (Phase 5 validation, Phase 6 rookie cards) cite this document rather than restating it.*

---

## A1. What this model is

- **Unit of observation: player–season**, not player. A four-year college player has up to four recipes; they are different snapshots of how he played that year, not four estimates of one identity. (Follows the paper's player–season–team convention; NCAA transfers occur between seasons, so player–season ≈ player–season–team here, with one known exception — see A4.)
- **One pooled fit, not ten seasonal fits.** All seasons 2016-17 … 2025-26 were z-scored *within season* first ("how extreme relative to your own league-year"), then pooled into a single fit. This keeps every recipe in one common coordinate system across the decade, which the downstream translator requires; it mirrors the paper's pooled multi-season European model.
- **Fit pool vs. standardization pool are deliberately different.** Z-scores are computed on the full D1 population at MIN ≥ 300 (29,312 player-seasons, 33 conferences). The fit itself uses the conference-restricted pool (20,849 rows, the 22 conferences that produced ≥1 NCAA-path draft pick, conference membership resolved per season to handle realignment). A player's z always means "relative to all of D1 that season," including at projection time.
- **Fit pool ≠ projection set.** Recipes are produced for the entire canonical table (all 29,312 rows) by simplex-constrained NNLS projection onto the frozen basis (`src/college_model/project.py`). Any feature vector can be projected without refitting — including short-minute seasons, out-of-pool conferences, and future rookies.
- **K = 8**, frozen restart selected by the consensus-basin rule (see A3).

## A2. What the archetypes structurally cannot see

The model uses **12 features**: height, PTS/100, TS%, USG%, AST%, TOV%, STL%, BLK%, TRB%, FTr, %FGAst'd_2P, %FGAst'd_3P.

Of the NBA basis's 29 dimensions, **16 have no college counterpart at all** — not sparse coverage, but schema absence in CBBD across every season:

- **7 shot-location dimensions** (average shot distance, five distance bins, corner-3 share). CBBD tracks shot *type* (dunk / layup / tip-in / 2pt-jumper / 3pt-jumper), a different classification axis.
- **9 play-type dimensions** (PnR handler, PnR roll man, isolation, post-up, spot-up, off-screen, handoff, cut, putback). No Synergy-equivalent tracking exists in CBBD.

**Consequence for interpretation and labelling:** distinctions that live primarily in those 16 dimensions are invisible to this model and must not be asserted in an archetype's label. In particular the model cannot reliably separate:

- rim-protecting roll man vs. perimeter-mobile big (both read as "tall, high BLK%, high TRB%, low 3PA");
- spot-up shooter vs. movement/off-screen shooter (both read as "high 3PA share, high assisted-3P%, low usage");
- pick-and-roll handler vs. isolation scorer (both read as "high usage, high AST%, low assisted-2P%").

Labels should be written in terms the 12 features actually support (volume, efficiency, size, playmaking load, defensive events, shot mix by 2P/3P) and should avoid play-type vocabulary the data cannot back.

**Two further feature caveats:**

- **TS%** carries a small, consistent, one-directional offset (+0.6 to +1.4 pp) versus Sports-Reference. It is a field CBBD returns directly, not derived here — a source-level formula/rounding difference, verified across 8 hand-checked player-seasons. It shifts all players in the same direction, so relative (z-scored) position is largely unaffected, but any absolute TS% quoted from this pipeline should carry the caveat.
- **PORPAG is not BPM.** It is carried as a separate column, never merged with the NBA's BPM and never used in the archetype fit. The two are different all-in-one value metrics occupying a similar role, not formula equivalents.

## A3. K-selection record

- Swept K = 4…12, AA, 3 seeds each (27 fits, ~4.4 h). RSS/explained variance rose smoothly (0.61 → 0.87) with **no usable elbow** — exactly the failure mode the paper documents for RSS alone.
- The intra-archetype-variance curve initially appeared non-monotonic. **Root cause: a plotting-rule artifact, corrected with owner approval.** The rule plotted the best-RSS restart per K; at K = 8 that restart (seed 0) is the *outlier* basin (intra-var 0.602) while the consensus basin (seeds 1–2) sits at 0.564. Read by per-K median or by min-envelope, the curve is clean and monotone into a trough at **K = 8–9**, with the two separated by only 0.007–0.011 — the same marginal-difference situation the paper faced on NBA data (minimum at 9, adopted 8).
- K = 9's single best solution (0.557) was found by one seed and did not converge; its consensus basin (0.575) is **worse** than K = 8's consensus basin (0.564).
- **Decision: K = 8** (owner). Matches the NBA basis dimensionality, which also simplifies the Hungarian identity baseline. The residual risk knowingly accepted: if a genuine ninth college role exists, its signal is compressed — mitigated by the fact that the college recipe is an intermediate representation, and the 12 raw shared features remain available to the translator independently.
- **Convergence:** only 3 of 27 fits met the formal criterion within max_iter; the rest hit the iteration cap on a very flat loss tail. Evidence this is a flat basin rather than instability: independent seeds land in the same place (K = 5 explained variance 0.6680 / 0.6678 / 0.6674), two formally-converged K = 9 fits agree to four decimals, and doubling max_iter from 200 to 400 barely moved any result.
- **Frozen restart rule (Amendment A):** lowest RSS *within the consensus basin* — explicitly **not** lowest RSS overall, which would have selected the K = 8 outlier basin and built the entire downstream pipeline on the worst of three fits.
- No archetype anywhere in the sweep fell below the 30-player stability flag (worst case 261 assigned players at K = 12).
- **ADA at this scale (Amendment B):** full swap-based archetypoid search was measured at days per fit on ~20k rows and was not attempted. The Step-3 archetypoid solution uses restricted search — nearest-real-player initialization (candns) plus one limited swap pass over each archetype's 100 nearest real player-seasons. Documented as "restricted archetypoid search," not full ADA.

## A4. Population and coverage caveats

| Item | Detail |
|---|---|
| 2020-21 NCAA season | 2,460 qualifying player-seasons vs. ~2,900 typical — real COVID-shortened schedules, not a pipeline gap |
| Ja Morant 2016-17 | Missing entirely from CBBD (both roster and player-season files, though both exist and cover other players that year). An isolated source gap. Consequence: his `years_in_college` computes to 2 instead of 3. **Left as computed and flagged, not hand-corrected** — single-point manual surgery on a systematic pipeline is worse than a documented, bounded error |
| A.J. Lawson | The one duplicate (athleteId, season) key in `shared_features.parquet` — a mid-season-transfer double count carried over from Phase 1. Deduplicated at join time in Phase 3; the underlying quirk remains |
| Complete-case drops | height 39 rows, ORB% 37, FTr 8 (of 29,395) — dropped rather than imputed |
| CBBD birthdates | 94.9% missing. Irrelevant to anchors, whose ages come from the NBA side (100% coverage) |
| Draft-pick position | 96.3% coverage, gaps concentrated in the 2017 class. Position is **excluded from v1 translator covariates** (owner decision); height plus the recipe carry most of the same information |
| 2025-26 NBA population drift | A from-scratch recompute returned 434 players vs. the 433 the basis was frozen on: **Ronald Holland II**, whose row was backfilled upstream after the fit. Resolution (owner decision): reuse the frozen standardization verbatim to stay bit-identical with `basis.npz` and the app's stored recipes. A guard re-checks this drift shape on every run and raises if it changes |

## A5. Special-case ledger — draft crosswalk

| Player | Class | Finding | Disposition |
|---|---|---|---|
| Shaedon Sharpe | 2022 | Redshirted at Kentucky, never played a college game. The crosswalk's tentative athlete id has zero records — a stale candidate, not a real one | `no_college_data` |
| Mitchell Robinson | 2018 | Committed to Western Kentucky, never enrolled or played | `no_college_data` |
| Goga Bitadze | 2019 | Zero CBBD records under his name in any season; confirms the `path=NCAA` / "Georgia" tag was a source mis-tag — Georgia the *country* (he played for Mega Bemax, Serbia) conflated with Georgia the *university* | Re-tagged `International (Serbia)` |
| De'Anthony Melton | 2018 | Recovered at USC in season 2017 (972 min) via a −1/−2-season lookback; sat out 2017-18 | `college_gap_year`, excluded by default |
| Dewan Hernandez | 2019 | Recovered at Miami in season 2018 (825 min) the same way; sat out 2018-19 | `college_gap_year`, excluded by default |
| Wesley Iwundu | 2017 | A name-variant miss — CBBD lists him as "Wes Iwundu." Recovered via a one-entry alias map (no general nickname system was built); cleared every downstream gate | **Included as a full anchor** |

## A6. Selection bias — the most important limitation for downstream use

Three filters shape who becomes an anchor, and each biases the training set in a way that must be carried into any claim the translator makes.

**1. Elite prospects with truncated final college seasons are absent.** Seven NCAA-matched picks failed the MIN ≥ 300 college filter, and they are not a random seven: James Wiseman, Michael Porter Jr., Darius Garland, Bol Bol, Jarred Vanderbilt, Jalen Johnson, Cedric Coward — nearly all high-value prospects whose final season was cut short by injury, suspension, or opt-out. The training set therefore contains **no example of the "elite prospect, fragmentary college season" profile**, which is precisely the profile that recurs in real draft-room debates. Predictions for such a player are extrapolation, not interpolation, and should be discounted explicitly.

**2. Anchors must have earned a real rookie role.** The Y side requires ≥ 300 rookie minutes (pro-rated for 2020-21). Draftees who never got rotation minutes as rookies are excluded, so the translator learns the mapping *conditional on receiving a rookie role*. It answers "what role will he play if he plays," not "will he play."

**3. Gap-year players are excluded** (Melton, Hernandez). Players whose college and rookie seasons are separated by more than the standard one draft cycle have different developmental dynamics and are not represented.

These three statements must appear in Phase 5's "what the translator cannot see" section and in the limits footnote of every Phase 6 rookie card.

## A7. Open items

- **Archetype labels are still blank.** Labelling is a basketball-domain judgment reserved for the project owner and should be written under the A2 constraints (no play-type vocabulary the features cannot support). Required before the Hungarian matching table is interpreted and before any Phase 6 card ships.
- **conf_tier map approved as proposed:** high-major = {ACC, Big Ten, Big 12, SEC, Big East}, plus Pac-12 through season 2023-24 only (post-realignment).
- **Shot-type mix** (rim-finish share, 3PT-jumper share; 93.8% coverage) is a college-native feature family with no NBA counterpart. Approved for translator input variant (b) with training-fold mean imputation plus a missing indicator. It is not part of the archetype fit and does not affect any recipe in this document.
