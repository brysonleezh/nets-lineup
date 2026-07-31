import json
import numpy as np

shots = json.load(open("/tmp/brown_shots.json"))
print(f"{len(shots)} real Brown FGA with coordinates")

# coordinate system confirmed empirically: tenths of a foot, full 94x50 court,
# two baskets (Louisville shoots at alternating ends across the game)
BASKET1 = (5.25 * 10, 25 * 10)
BASKET2 = ((94 - 5.25) * 10, 25 * 10)
NCAA_3PT_FT = 22.15  # men's D1 arc distance since the 2019-20 rule change - NOT the NBA
# distance (23.75 top / 22 corners) - using the real NCAA rule, not silently reusing NBA's

def dist_ft(x, y):
    d1 = np.hypot(x - BASKET1[0], y - BASKET1[1])
    d2 = np.hypot(x - BASKET2[0], y - BASKET2[1])
    return min(d1, d2) / 10.0

def nearest_basket_y_offset(x, y):
    # for corner-3 detection: lateral distance from the hoop's own centerline
    d1 = np.hypot(x - BASKET1[0], y - BASKET1[1])
    d2 = np.hypot(x - BASKET2[0], y - BASKET2[1])
    return abs(y - 25 * 10) / 10.0  # both baskets sit at y=25ft (mid-width) by construction

records = []
for p in shots:
    loc = p["shotInfo"]["location"]
    x, y = loc["x"], loc["y"]
    d = dist_ft(x, y)
    made = p["shotInfo"]["made"]
    is_three_cbbd = p["shotInfo"]["range"] == "three_pointer"
    records.append({"dist": d, "made": made, "is_three_cbbd": is_three_cbbd, "y_offset": nearest_basket_y_offset(x, y)})

dists = np.array([r["dist"] for r in records])
print(f"real distance range: {dists.min():.1f} - {dists.max():.1f} ft, mean={dists.mean():.2f}")

# cross-check: does my computed distance agree with CBBD's own 'range' label?
n_three_by_dist = (dists >= NCAA_3PT_FT).sum()
n_three_by_cbbd = sum(r["is_three_cbbd"] for r in records)
print(f"3PT attempts by my distance calc (>= {NCAA_3PT_FT}ft): {n_three_by_dist}")
print(f"3PT attempts per CBBD's own 'range' label: {n_three_by_cbbd} (season stat 3PA=160)")

buckets = {
    "% of FGA by Distance_0-3": (0, 3),
    "% of FGA by Distance_3-10": (3, 10),
    "% of FGA by Distance_10-16": (10, 16),
    "% of FGA by Distance_16-3P": (16, NCAA_3PT_FT),
    "% of FGA by Distance_3P": (NCAA_3PT_FT, 999),
}
n = len(dists)
shot_dist_features = {}
for name, (lo, hi) in buckets.items():
    shot_dist_features[name] = 100.0 * ((dists >= lo) & (dists < hi)).sum() / n
shot_dist_features["Dist."] = float(dists.mean())

# corner-3: a 3PT attempt with a small lateral offset from the basket's own centerline
# (standard corner-3 definition: near the sideline, not the top of the arc)
corner_3_thresh_ft = 8.0  # within 8ft of the hoop's own baseline-perpendicular centerline
threes = [r for r in records if r["dist"] >= NCAA_3PT_FT]
corner_3s = [r for r in threes if r["y_offset"] >= (25 - corner_3_thresh_ft)]
shot_dist_features["Corner 3s_%3PA"] = 100.0 * len(corner_3s) / len(threes) if threes else 0.0

print()
for k, v in shot_dist_features.items():
    print(f"{k}: {v:.2f}")

json.dump(shot_dist_features, open("/tmp/brown_shot_dist_features.json", "w"))
