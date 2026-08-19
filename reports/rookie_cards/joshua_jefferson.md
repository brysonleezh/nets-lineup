# Joshua Jefferson

**Iowa State** · F, 6'9" · 2026 draft pick #28

*Generated 2026-08-14 07:26 UTC · commit `429619e` · model T1_b, deployment refit 2026-08-14, n=273*

---

## 1. Predicted rookie-season archetype mix

Point estimates only — no uncertainty interval shown (see Section 5).

| Archetype | Predicted weight |
|---|---|
| 3&D Wing | 24% |
| Rim Protector / Roll Man | 23% |
| Traditional Playmaker | 20% |
| Combo Guard | 11% |
| Inside Scoring Big | 8% |
| Shooting Specialist | 5% |
| Mobile Big | 5% |
| Offensive Engine | 4% |


## 2. Top-3 predicted archetypes


- **1st: 3&D Wing** (24%) — closest real comparison: Nicolas Batum (2025-26)

- **2nd: Rim Protector / Roll Man** (23%) — closest real comparison: Clint Capela (2025-26)

- **3rd: Traditional Playmaker** (20%) — closest real comparison: Bez Mbeng (2025-26)


## 3. College profile (the model's input)

His actual 2025-26 college archetype mix — this is what went *into* the model, not a prediction.

| Archetype | Weight |
|---|---|
| High-Usage Interior Scorer | 48% |
| Ball-Hawking Defensive Guard | 35% |
| Low-Minute Statistical Outlier (a noisier, less distinct archetype in this model) | 8% |
| High-Usage Primary Ball-Handler | 8% |
| Rim-Protecting Big / Shot-Blocking Center | 1% |
| Inefficient Low-Usage Reserve (a noisier, less distinct archetype in this model) | 0% |
| Efficient Low-Usage Play-Finisher | 0% |
| Low-Event Floor Role Player (a noisier, less distinct archetype in this model) | 0% |


## 4. Comps: historically similar college profiles

These are the 5 draft-class anchors whose college statistical profile most closely resembles Joshua's (nearest neighbors in the model's own feature space), shown with what they actually became as NBA rookies. This is the honest answer to "how much could this vary": the spread across these 5 outcomes, not a synthetic confidence interval.

| Player | Class | Pick | College | His college top archetype | What he actually became as a rookie |
|---|---|---|---|---|---|
| Chandler Hutchison | 2018 | #22 | Boise State | High-Usage Interior Scorer | Traditional Playmaker (30%), then Rim Protector / Roll Man (28%) |
| Dillon Jones | 2024 | #26 | Weber State | High-Usage Interior Scorer | 3&D Wing (32%), then Combo Guard (28%) |
| Brooks Barnhizer | 2025 | #44 | Northwestern | Ball-Hawking Defensive Guard | Traditional Playmaker (44%), then Combo Guard (24%) |
| Nique Clifford | 2025 | #24 | Colorado State | High-Usage Interior Scorer | Combo Guard (48%), then Traditional Playmaker (21%) |
| Ayo Dosunmu | 2021 | #38 | Illinois | High-Usage Primary Ball-Handler | 3&D Wing (33%), then Combo Guard (30%) |


## 5. How much should you trust this?

> On the 2025 draft class (n=36, the only class this model has been tested against that it never trained on), this model's predictions had a 52.8% chance of naming the correct top archetype, a 69.4% chance the correct archetype was in its top two, and an average Jensen-Shannon divergence of 0.169 between predicted and actual archetype mixes.


## 6. What this model cannot see


- This model is blind to shot-location and play-type data on the college side (16 of the NBA archetype basis's 29 dimensions have no college counterpart) — it cannot reliably separate certain role variants, e.g. a rim-protecting roll man from a perimeter-mobile big, or a spot-up shooter from a movement shooter.

- It was trained only on players who earned at least 300 NBA minutes as rookies — it answers "what role will he play if he plays," never "will he play."

- Elite prospects whose final college season was cut short by injury, suspension, or opt-out are absent from the training data — the model has never seen that profile, so a prediction for such a player would be extrapolation, not interpolation.

- Posterior uncertainty intervals from this model are not calibrated (validated on the 2025 holdout — see the confidence line above) and are deliberately not shown.

- Coaching decisions, scheme fit, and playing-time opportunity are not observable to any statistical model and are not represented here.
