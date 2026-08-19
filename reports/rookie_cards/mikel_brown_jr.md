# Mikel Brown Jr.

**Louisville** · G, 6'5" · 2026 draft pick #6

*Generated 2026-08-14 07:26 UTC · commit `429619e` · model T1_b, deployment refit 2026-08-14, n=273*

---

## 1. Predicted rookie-season archetype mix

Point estimates only — no uncertainty interval shown (see Section 5).

| Archetype | Predicted weight |
|---|---|
| Combo Guard | 55% |
| Traditional Playmaker | 11% |
| Shooting Specialist | 10% |
| Rim Protector / Roll Man | 6% |
| 3&D Wing | 6% |
| Offensive Engine | 5% |
| Mobile Big | 3% |
| Inside Scoring Big | 3% |


## 2. Top-3 predicted archetypes


- **1st: Combo Guard** (55%) — closest real comparison: D'Angelo Russell (2025-26)

- **2nd: Traditional Playmaker** (11%) — closest real comparison: Bez Mbeng (2025-26)

- **3rd: Shooting Specialist** (10%) — closest real comparison: Tim Hardaway Jr. (2025-26)


## 3. College profile (the model's input)

His actual 2025-26 college archetype mix — this is what went *into* the model, not a prediction.

| Archetype | Weight |
|---|---|
| High-Usage Primary Ball-Handler | 57% |
| High-Usage Interior Scorer | 18% |
| Low-Minute Statistical Outlier (a noisier, less distinct archetype in this model) | 11% |
| Ball-Hawking Defensive Guard | 7% |
| Efficient Low-Usage Play-Finisher | 6% |
| Inefficient Low-Usage Reserve (a noisier, less distinct archetype in this model) | 0% |
| Rim-Protecting Big / Shot-Blocking Center | 0% |
| Low-Event Floor Role Player (a noisier, less distinct archetype in this model) | 0% |


## 4. Comps: historically similar college profiles

These are the 5 draft-class anchors whose college statistical profile most closely resembles Mikel's (nearest neighbors in the model's own feature space), shown with what they actually became as NBA rookies. This is the honest answer to "how much could this vary": the spread across these 5 outcomes, not a synthetic confidence interval.

| Player | Class | Pick | College | His college top archetype | What he actually became as a rookie |
|---|---|---|---|---|---|
| Rob Dillingham | 2024 | #8 | Kentucky | High-Usage Primary Ball-Handler | Combo Guard (81%), then Traditional Playmaker (19%) |
| Dennis Smith | 2017 | #9 | NC State | High-Usage Primary Ball-Handler | Combo Guard (71%), then Offensive Engine (14%) |
| Dylan Harper | 2025 | #2 | Rutgers | High-Usage Primary Ball-Handler | Combo Guard (41%), then Offensive Engine (18%) |
| Keyonte George | 2023 | #16 | Baylor | High-Usage Primary Ball-Handler | Combo Guard (78%), then Shooting Specialist (14%) |
| Jaden Ivey | 2022 | #5 | Purdue | High-Usage Interior Scorer | Combo Guard (72%), then Traditional Playmaker (14%) |


## 5. How much should you trust this?

> On the 2025 draft class (n=36, the only class this model has been tested against that it never trained on), this model's predictions had a 52.8% chance of naming the correct top archetype, a 69.4% chance the correct archetype was in its top two, and an average Jensen-Shannon divergence of 0.169 between predicted and actual archetype mixes.


## 6. What this model cannot see


- This model is blind to shot-location and play-type data on the college side (16 of the NBA archetype basis's 29 dimensions have no college counterpart) — it cannot reliably separate certain role variants, e.g. a rim-protecting roll man from a perimeter-mobile big, or a spot-up shooter from a movement shooter.

- It was trained only on players who earned at least 300 NBA minutes as rookies — it answers "what role will he play if he plays," never "will he play."

- Elite prospects whose final college season was cut short by injury, suspension, or opt-out are absent from the training data — the model has never seen that profile, so a prediction for such a player would be extrapolation, not interpolation.

- Posterior uncertainty intervals from this model are not calibrated (validated on the 2025 holdout — see the confidence line above) and are deliberately not shown.

- Coaching decisions, scheme fit, and playing-time opportunity are not observable to any statistical model and are not represented here.
