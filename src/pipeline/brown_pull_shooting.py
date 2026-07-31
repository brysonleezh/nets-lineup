import json
import time
import requests

env = {}
with open(".env") as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"').strip("'")
key = env["CBBD_API_KEY"]
H = {"Authorization": f"Bearer {key}"}
BASE = "https://api.collegebasketballdata.com"

players = json.load(open("/tmp/cbbd_all_players_2026.json"))
qualified = [p for p in players if (p.get("minutes") or 0) >= 300]
teams = sorted(set(p.get("team") for p in qualified))
print(f"{len(teams)} teams to pull shooting/season for", flush=True)

all_rows = []
for i, team in enumerate(teams, start=1):
    for attempt in range(3):
        try:
            resp = requests.get(f"{BASE}/stats/player/shooting/season", headers=H,
                                params={"season": 2026, "team": team}, timeout=30)
            if resp.status_code == 200:
                all_rows.extend(resp.json())
                break
            else:
                time.sleep(2)
        except requests.exceptions.RequestException:
            time.sleep(2)
    if i % 25 == 0:
        print(f"[{i}/{len(teams)}] {team} done, {len(all_rows)} rows so far", flush=True)
    time.sleep(0.3)

json.dump(all_rows, open("/tmp/cbbd_all_shooting_by_team_2026.json", "w"))
print(f"DONE: {len(all_rows)} total player-shooting rows saved")
