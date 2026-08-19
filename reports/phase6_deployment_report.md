# Phase 6 Deployment Report — Rookie Cards & Prediction Freeze

Final deliverable for Phase 6, the last phase of the NCAA→NBA rookie
archetype translation pipeline. The deployed model (T1_b, pre-registered
in Phase 4, validated in Phase 5) was **not modified** anywhere in this
phase — the only fitting done was a mechanical refit of that same
specification on more data. See `reports/phase6_worklog.md` for the full
step-by-step narrative this report distills.

## A. Refit diagnostics + 237-vs-273 coefficient agreement

Refit T1_b (unchanged priors, reference-category identification, sampler
settings; seed=42, matching the pre-registered 237-fit's seed policy) on
all **273 anchors** (classes 2017–2025 — the 2025 holdout's role was
already served in Phase 5; the deployed model should use every real
anchor available).

**Convergence** (same standard held throughout the project): max R-hat =
1.0055, min ESS = 1466, **0 divergent transitions** out of 3000
post-warmup samples, converged at the first target_accept rung (0.8).

**237-vs-273 coefficient agreement**: correlation = **0.981**, largest
single coefficient shift = 0.227 (on `three_pt_jumper_share`'s effect on
archetype-column 0). Well under the flag thresholds (corr<0.95 or a
shift>1.0) set in advance — **no material disagreement**, exactly what
adding ~15% more real rows to an already-converged fit should look like.

**Preprocessing rebuilt complete this time**: `data/translator/
deployment_preprocessing.json` covers all 19 continuous columns, including
the two shot-type ones Phase 4's original file omitted (found and patched
in Phase 5). Verified the persisted mu/sd reproduce `variant_b`'s real
output to 1e-10 before trusting them.

## B. The three predictions

| Rookie | College (pick) | College top archetype | Predicted rookie top archetype | Confidence in top pick |
|---|---|---|---|---|
| **Mikel Brown Jr.** | Louisville (#6) | High-Usage Primary Ball-Handler (57%) | **Combo Guard** | 55% — concentrated |
| **Tyler Bilodeau** | UCLA (#43) | High-Usage Interior Scorer (52%) | **Shooting Specialist** | 35% — diffuse |
| **Joshua Jefferson** | Iowa State (#28) | High-Usage Interior Scorer (48%) | **3&D Wing** | 24% — nearly flat top-3 |

**Reasoning, not just numbers:**

- **Brown** is the model's most confident call. His college profile (a
  clean, high-usage lead-guard signature — the same college archetype
  Trae Young and Ja Morant load onto, per Phase 2's sanity vignettes)
  translates to a concentrated 55% prediction on Combo Guard, and **4 of
  his 5 nearest historical comps (Rob Dillingham, Dennis Smith, Dylan
  Harper, Keyonte George) all became that exact same NBA archetype as
  rookies**, with individual weights from 41% to 81%. Model confidence and
  comp-based evidence agree here.
- **Bilodeau** is a genuine toss-up. His predicted top archetype (Shooting
  Specialist, 35%) barely edges his second-place archetype (Combo Guard-
  adjacent "3&D Wing" territory at 30%), and his 5 comps split 2-vs-3
  between those two outcomes. The model isn't confused — the underlying
  historical pattern for his profile really is split.
- **Jefferson** is the least confident of the three: predicted top-3
  weights are nearly flat (24% / 23% / 20%), and his comps show no
  dominant pattern either. Read this as an honest "the model doesn't know
  yet" rather than a specific wrong-direction bet.

Full numeric detail in `data/projections/nets_rookies_2026.csv`.

## C. Comps tables

Per rookie, the 5 nearest anchors in the same standardized input space the
model itself uses (identical metric to Phase 4's T3), each shown with
what he actually became as an NBA rookie — this is the "range of
outcomes" role intervals would otherwise have played (Locked Decision 4).

**Mikel Brown Jr.**: Rob Dillingham (2024, Kentucky) → Combo Guard 81%;
Dennis Smith (2017, NC State) → Combo Guard 71%; Dylan Harper (2025,
Rutgers) → Combo Guard 41%; Keyonte George (2023, Baylor) → Combo Guard
78%; Jaden Ivey (2022, Purdue) → Combo Guard 72%. **All 5 became Combo
Guard as their top rookie archetype.**

**Tyler Bilodeau**: Antonio Reeves (2024, Kentucky) → Shooting Specialist
48%; Kessler Edwards (2021, Pepperdine) → 3&D Wing 53%; Semi Ojeleye
(2017, SMU) → 3&D Wing 64%; Keita Bates-Diop (2018, Ohio State) → 3&D Wing
28%; Corey Kispert (2021, Gonzaga) → Shooting Specialist 49%. **Split 2/5
Shooting Specialist, 3/5 3&D Wing.**

**Joshua Jefferson**: Chandler Hutchison (2018, Boise State) → Traditional
Playmaker 30%; Dillon Jones (2024, Weber State) → 3&D Wing 32%; Brooks
Barnhizer (2025, Northwestern) → Traditional Playmaker 44%; Nique Clifford
(2025, Colorado State) → Combo Guard 48%; Ayo Dosunmu (2021, Illinois) →
3&D Wing 33%. **No two comps share the same outcome** — the widest spread
of the three rookies, consistent with his own flat predicted distribution.

Full comp tables with distances in `data/translator/rookie_projections_full.json`
and each rookie's own card.

## D. Saturation analysis

**None of the 3 rookies triggered the flag** — college `c_alpha_max` is
0.572 (Brown, 85th percentile), 0.524 (Bilodeau, 76th percentile), 0.479
(Jefferson, 67th percentile), all under the 0.85 threshold. No card
carries the saturation caveat.

**The training-set evidence for the rule itself, reported honestly rather
than oversold**: searched all 237 training anchors via a fresh 8-fold
out-of-sample CV refit for cases where the true rookie recipe had
`y_max >= 0.9` (the Kalkbrenner threshold). **Found zero.** Softer
thresholds, for context only: `y_max>=0.85` → 1/237, `>=0.8` → 6/237,
`>=0.75` → 12/237. **The "Kalkbrenner rule" is grounded in exactly one
documented case** (the 2025 holdout, Phase 5: true recipe 1.0 on a single
archetype, model predicted only 17% weight there) — not a pattern
confirmed across many training examples. This is disclosed plainly in the
rule's own card text ("see the 2025 case of Ryan Kalkbrenner") rather than
implied to be well-established. Extreme concentration is genuinely rare
in this dataset, which is exactly why that one case stood out enough to
motivate a rule in the first place.

A related, unsolicited finding while building the cards: `data/basis_2025_26/
archetype_definitions.csv` shows **Ryan Kalkbrenner is the real
archetypoid exemplar for NBA archetype 4 "Mobile Big"** in the currently-
frozen basis — his rookie season was distinctive enough that he's
literally the nearest real point to that archetype's own centroid,
independently consistent with (not derived from) Phase 5's saturation
finding.

Full evidence table in `data/translator/saturation_rule_evidence.csv`.

## E. Freeze record

`data/projections/predictions_frozen.json`, written 2026-08-14, **before
the 2026-27 NBA season began**. Carries: sha256 of `nets_rookies_2026.csv`
(`a5ebf101...`), `config.yaml`, and the deployment manifest; git commit;
the Phase 5 holdout metrics at freeze time (JSD 0.169, top-1 hit 52.8%,
top-1-within-top-2 69.4%, n=36); all 3 rookies' predicted top-3.

The freeze guard was tested for real: a second run with no flag refused
before touching the file; `--refreeze` with a disclosed reason correctly
appended to `refreeze_log` while preserving the prior csv hash. One real
refreeze occurred during this phase (adding an unrelated config field
after the initial freeze, invalidating the recorded `config_hash` — not
the predictions themselves) — logged in the frozen file's own
`refreeze_log`, not hidden.

## F. Review-script description

`src/eval/review_2026_predictions.py`, written now, meant to run once
2026-27 season data exists. Frozen methodology (in the module's own
docstring, so it can't drift): eligibility at 300 minutes (pro-rated only
if the season is shortened); build a `standardization_2026-27.json` via
the exact method used for the other 9 seasons; project each qualifying
rookie's real stat line onto the same frozen 8×29 basis; score with
**exactly** the Phase 5 metric list. A rookie under the minutes threshold
is reported `unevaluable`, never an error or a failure. Core logic split
into pure functions and tested against synthetic 2026-27 inputs (9 tests,
including a full synthetic-population → standardization → projection →
scoring pipeline test using the real frozen basis) — proven runnable
today, per spec. Smoke-tested the real entry point against the current
(empty) database: correctly reports "nothing to review yet" rather than
erroring.

## G. Obstacles

1. Two real, disclosed precondition gaps at kickoff: no `config.yaml`
   existed, and CBBD's draft data showed Joshua Jefferson drafted by
   Minnesota, not Brooklyn. Resolved with live data (a `commonallplayers`
   query confirming his real 2026-27 team is BKN) rather than an
   assumption, before presenting a much smaller decision to the owner.
2. A column-naming mismatch (`recipes.csv` uses `alpha_0..7`, not
   `c_alpha_0..7`) caused an early crash in Step 2 — fixed immediately, no
   downstream impact.
3. The saturation-rule evidence search returning zero training-set cases
   at the target threshold was a real, slightly deflating finding —
   reported as-is (Section D) rather than loosened to manufacture support.
4. A card-readability bug (labels referencing a note that doesn't exist on
   a standalone card) was caught by re-reading the actual rendered output
   against the spec's own design constraint, not just checking the HTML
   was well-formed.
5. Playwright's chromium browser wasn't installed in this environment yet
   (one-time setup, per the project's own established convention) —
   installed and used to visually confirm card rendering, not just assume
   the template compiled.

## H. Open items

1. **Bilodeau and Jefferson's predictions are genuinely uncertain** — not
   a modeling failure, a real property of their statistical profiles
   (their comps disagree with each other). Nothing to fix; worth the
   Nets' scouting staff weighing this against film/interviews the model
   never sees.
2. **The saturation rule has n=1 supporting evidence.** If a future
   season's holdout produces another extreme-concentration case, revisit
   whether the rule generalizes or was specific to Kalkbrenner's profile.
3. **`src/eval/review_2026_predictions.py` is untested against real data**
   by construction (it doesn't exist yet) — the synthetic tests prove the
   methodology runs, not that it will produce a sensible verdict on real
   2026-27 numbers. Re-check its output sanity the first time it's
   actually run.
4. Per the project's standing convention: this report and
   `reports/project_summary.md` are the terminal deliverables of this
   pipeline. No further phase is planned; any future work (a portal
   showcase page, discussed and deliberately deferred earlier this
   session) starts as its own scoped task.
