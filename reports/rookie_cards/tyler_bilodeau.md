# Tyler Bilodeau

**UCLA** · F, 6'9" · 2026 draft pick #43

*Generated 2026-08-14 07:26 UTC · commit `429619e` · model T1_b, deployment refit 2026-08-14, n=273*

---

## 1. Predicted rookie-season archetype mix

Point estimates only — no uncertainty interval shown (see Section 5).

| Archetype | Predicted weight |
|---|---|
| Shooting Specialist | 35% |
| 3&D Wing | 30% |
| Traditional Playmaker | 8% |
| Inside Scoring Big | 8% |
| Mobile Big | 7% |
| Rim Protector / Roll Man | 6% |
| Combo Guard | 5% |
| Offensive Engine | 2% |


## 2. Top-3 predicted archetypes


- **1st: Shooting Specialist** (34%) — closest real comparison: Tim Hardaway Jr. (2025-26)

- **2nd: 3&D Wing** (30%) — closest real comparison: Nicolas Batum (2025-26)

- **3rd: Traditional Playmaker** (8%) — closest real comparison: Bez Mbeng (2025-26)


## 3. College profile (the model's input)

His actual 2025-26 college archetype mix — this is what went *into* the model, not a prediction.

| Archetype | Weight |
|---|---|
| High-Usage Interior Scorer | 52% |
| Efficient Low-Usage Play-Finisher | 33% |
| Low-Event Floor Role Player (a noisier, less distinct archetype in this model) | 11% |
| High-Usage Primary Ball-Handler | 3% |
| Low-Minute Statistical Outlier (a noisier, less distinct archetype in this model) | 0% |
| Inefficient Low-Usage Reserve (a noisier, less distinct archetype in this model) | 0% |
| Rim-Protecting Big / Shot-Blocking Center | 0% |
| Ball-Hawking Defensive Guard | 0% |


## 4. Comps: historically similar college profiles

These are the 5 draft-class anchors whose college statistical profile most closely resembles Tyler's (nearest neighbors in the model's own feature space), shown with what they actually became as NBA rookies. This is the honest answer to "how much could this vary": the spread across these 5 outcomes, not a synthetic confidence interval.

| Player | Class | Pick | College | His college top archetype | What he actually became as a rookie |
|---|---|---|---|---|---|
| Antonio  Reeves | 2024 | #47 | Kentucky | High-Usage Interior Scorer | Shooting Specialist (48%), then Combo Guard (22%) |
| Kessler Edwards | 2021 | #44 | Pepperdine | High-Usage Interior Scorer | 3&D Wing (53%), then Rim Protector / Roll Man (26%) |
| Semi Ojeleye | 2017 | #37 | SMU | High-Usage Interior Scorer | 3&D Wing (64%), then Traditional Playmaker (16%) |
| Keita Bates-Diop | 2018 | #48 | Ohio State | High-Usage Interior Scorer | 3&D Wing (28%), then Rim Protector / Roll Man (27%) |
| Corey Kispert | 2021 | #15 | Gonzaga | Efficient Low-Usage Play-Finisher | Shooting Specialist (49%), then 3&D Wing (33%) |


## 5. How much should you trust this?

> On the 2025 draft class (n=36, the only class this model has been tested against that it never trained on), this model's predictions had a 52.8% chance of naming the correct top archetype, a 69.4% chance the correct archetype was in its top two, and an average Jensen-Shannon divergence of 0.169 between predicted and actual archetype mixes.


## 6. What this model cannot see


- This model is blind to shot-location and play-type data on the college side (16 of the NBA archetype basis's 29 dimensions have no college counterpart) — it cannot reliably separate certain role variants, e.g. a rim-protecting roll man from a perimeter-mobile big, or a spot-up shooter from a movement shooter.

- It was trained only on players who earned at least 300 NBA minutes as rookies — it answers "what role will he play if he plays," never "will he play."

- Elite prospects whose final college season was cut short by injury, suspension, or opt-out are absent from the training data — the model has never seen that profile, so a prediction for such a player would be extrapolation, not interpolation.

- Posterior uncertainty intervals from this model are not calibrated (validated on the 2025 holdout — see the confidence line above) and are deliberately not shown.

- Coaching decisions, scheme fit, and playing-time opportunity are not observable to any statistical model and are not represented here.
