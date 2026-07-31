import json
import sys
import sqlite3
from itertools import combinations
import numpy as np
import pandas as pd
from pathlib import Path

# AI-ASSISTED (Claude Code, chat) - path fix for the src/pipeline/ move:
# resolve src/ from this file's own location instead of assuming cwd.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from step0_data import _normalize_name, _prefer_combined_bref_row

k = 8
brown = json.load(open("data/brown/profile.json"))
brown_alpha = np.array([brown["alpha"][f"arch_{i}"] for i in range(k)])

recipes = pd.read_csv("data/rc_multiseason/basis_reduced20/recipes_reduced20.csv")
labels_df = pd.read_csv("data/rc_multiseason/basis_reduced20/archetype_definitions.csv")
labels = {int(row["archetype"]): row["PLAYER_NAME"] for _, row in labels_df.iterrows()}
arch_cols = [f"arch_{i}" for i in range(k)]

NETS_ROSTER_NBA_DATA = [
    "Danny Wolf", "Drake Powell", "Egor Dëmin", "E.J. Liddell", "Nolan Traore",
    "Terance Mann", "Day'Ron Sharpe", "Noah Clowney", "Julius Randle",
    "Chaney Johnson", "Ben Saraf", "Josh Minott", "Michael Porter Jr.",
    "Keon Ellis", "Tyson Etienne", "Moritz Wagner",
]
roster = recipes[recipes["PLAYER_NAME"].isin(NETS_ROSTER_NBA_DATA)].copy()
print(f"resolved {len(roster)}/{len(NETS_ROSTER_NBA_DATA)} rotation-eligible Nets players "
      f"on the reduced basis")
missing = set(NETS_ROSTER_NBA_DATA) - set(roster["PLAYER_NAME"])
if missing:
    print(f"MISSING from reduced-basis recipes (name mismatch?): {missing}")

# validated talent number: exogenous skill_off/skill_def (build_skill.py output) - the
# SAME real, prior-season, leakage-checked source already used and validated for the
# team-season talent-aggregation engine in Task A/RAPM_README.md - NOT the RAPM model's
# own archetype/synergy prediction, which failed GATE 2 and is explicitly out of scope
# for "numbers" per the design principle in docs/RESEARCH_FINDINGS.md
skill = pd.read_parquet("data/model/players_skill.parquet")
with sqlite3.connect("data/nets_synergy.db") as conn:
    pb = pd.read_sql("SELECT PLAYER_ID, PLAYER_NAME FROM player_base WHERE SEASON='2025-26'", conn)
roster = roster.merge(pb[["PLAYER_ID", "PLAYER_NAME"]], on="PLAYER_NAME", how="left", suffixes=("", "_pb"))
roster = roster.merge(skill, left_on="PLAYER_ID", right_on="player_id", how="left")
print(f"talent (skill_off/skill_def) join: {roster['skill_off'].notna().sum()}/{len(roster)} matched")

# Brown's own talent number: draft-slot prior at pick #6 (already-validated build_skill.py
# machinery, reused directly rather than re-derived)
# AI-ASSISTED (Claude Code, chat) - path fix for the src/pipeline/ move:
# build_skill.py is now a co-located sibling in src/pipeline/, not src/ -
# point at this file's own directory rather than the old "src" literal.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_skill as bsk
hist = bsk.load_historical_rookie_seasons({"2025-26"})
curve = bsk.fit_draft_curve(hist)
brown_skill_off, brown_skill_def = bsk.predict_from_pick(6, curve)
print(f"\nBrown draft-slot-prior skill: skill_off={brown_skill_off:.2f}, skill_def={brown_skill_def:.2f}")

# --- constraint flags, grounded in the REAL archetype profiles (describe_archetypes) ---
RIM_PROTECTION_ARCHS = [0, 1]  # Cisse/Queta-type: high BLK%/TRB%/rim-share
SHOOTING_ARCHS = [7]  # McDermott-type: high 3PT rate + high assisted rate (movement/spot-up)
INEFFICIENT_BALLHANDLER_ARCH = 3  # Kam Jones-type: high TOV%/AST% but LOW TS%/BPM/FTr -
# the archetype that concretely drove the NEGATIVE half of Phase B3's success pattern,
# not "any second guard" (Phase B3 found ANOTHER elite creator, archetype 6, trends
# POSITIVE - a real, specific finding, not the generic assumption)
PRESENCE_THRESH = 0.30  # a player counts as "supplying" an archetype if it's his largest share

roster["dominant_arch"] = roster[arch_cols].values.argmax(axis=1)
rim_protectors = roster[roster["dominant_arch"].isin(RIM_PROTECTION_ARCHS)]["PLAYER_NAME"].tolist()
shooters = roster[roster["dominant_arch"].isin(SHOOTING_ARCHS)]["PLAYER_NAME"].tolist()
redundant_guards = roster[roster["dominant_arch"] == INEFFICIENT_BALLHANDLER_ARCH]["PLAYER_NAME"].tolist()
print(f"\nrim-protector-dominant players on roster: {rim_protectors}")
print(f"shooting-dominant players on roster: {shooters}")
print(f"inefficient-ballhandler-dominant players on roster (redundancy risk): {redundant_guards}")

# --- incompatibility list: cosine similarity between Brown's OWN vector and each
# current player's vector, restricted to players who ALSO carry real weight in the
# inefficient-ballhandler archetype specifically (the one with a documented negative
# success-pattern association) - not flagging every guard indiscriminately
def cosine(a, b):
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

incompat = []
for _, r in roster.iterrows():
    vec = r[arch_cols].values.astype(float)
    sim_to_brown = cosine(brown_alpha, vec)
    ineff_share = vec[INEFFICIENT_BALLHANDLER_ARCH]
    if ineff_share >= PRESENCE_THRESH:
        incompat.append({
            "player": r["PLAYER_NAME"], "cosine_sim_to_brown": sim_to_brown,
            "inefficient_ballhandler_share": float(ineff_share),
            "reason": f"dominant archetype is the low-efficiency ball-handler type "
                      f"({ineff_share:.0%} share) that Phase B3's pooled evidence "
                      f"associated with WORSE outcomes when paired with Brown's own "
                      f"style, not merely 'another guard'",
        })
print(f"\n=== Incompatibility list ({len(incompat)} players) ===")
for r in incompat:
    print(f"  {r['player']}: {r['reason']}")

# --- enumerate every 4-man complement ---
records = []
for combo in combinations(roster.index, 4):
    sub = roster.loc[list(combo)]
    mix = sub[arch_cols].mean(axis=0).values
    dominant_archs = set(sub["dominant_arch"])
    has_rim = bool(dominant_archs & set(RIM_PROTECTION_ARCHS))
    has_shooting = bool(dominant_archs & set(SHOOTING_ARCHS))
    has_redundancy = INEFFICIENT_BALLHANDLER_ARCH in dominant_archs
    talent = sub["skill_off"].sum() + sub["skill_def"].sum() + brown_skill_off + brown_skill_def
    records.append({
        "players": tuple(sub["PLAYER_NAME"]),
        "talent_sum": float(talent),
        "has_rim_protection": has_rim, "has_shooting": has_shooting,
        "second_guard_redundancy_flag": has_redundancy,
        "passes_constraints": has_rim and has_shooting and not has_redundancy,
    })
combos_df = pd.DataFrame(records)
print(f"\n{len(combos_df)} total 4-man complements enumerated")
print(f"{combos_df['passes_constraints'].sum()} pass all constraints "
      f"(rim protection + shooting present, no redundancy flag)")

passing = combos_df[combos_df["passes_constraints"]].sort_values("talent_sum", ascending=False)
print("\n=== Top 10 recommended units (pass constraints, ranked by talent) ===")
print(passing.head(10)[["players", "talent_sum"]].to_string(index=False))

near_miss = combos_df[~combos_df["passes_constraints"]].sort_values("talent_sum", ascending=False).head(5)
print("\n=== Top 5 near-misses (highest talent, but violate a flag) ===")
for _, r in near_miss.iterrows():
    violations = []
    if not r["has_rim_protection"]:
        violations.append("no rim protection")
    if not r["has_shooting"]:
        violations.append("no floor-spacing shooter")
    if r["second_guard_redundancy_flag"]:
        violations.append("inefficient-ballhandler redundancy")
    print(f"  {r['players']} (talent={r['talent_sum']:.1f}): violates [{', '.join(violations)}]")

out = {
    "brown_skill_off": brown_skill_off, "brown_skill_def": brown_skill_def,
    "rim_protectors_on_roster": rim_protectors, "shooters_on_roster": shooters,
    "redundancy_risk_on_roster": redundant_guards,
    "incompatibility_list": incompat,
    "n_units_enumerated": len(combos_df), "n_passing_constraints": int(combos_df["passes_constraints"].sum()),
    "top_10_recommended": passing.head(10)[["players", "talent_sum"]].to_dict("records"),
    "near_misses": near_miss[["players", "talent_sum", "has_rim_protection", "has_shooting",
                              "second_guard_redundancy_flag"]].to_dict("records"),
}
Path("data/brown").mkdir(parents=True, exist_ok=True)
json.dump(out, open("data/brown/deployment.json", "w"), indent=2, default=str)
print("\nsaved -> data/brown/deployment.json")
