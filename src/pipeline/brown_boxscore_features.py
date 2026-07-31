import json
import numpy as np
import pandas as pd

players = json.load(open("/tmp/cbbd_all_players_2026.json"))
teams = json.load(open("/tmp/cbbd_all_teams_2026.json"))

# team lookup: teamId -> team-season context (own + opponent)
team_by_id = {t["teamId"]: t for t in teams if t.get("season") == 2026}
print(f"{len(team_by_id)} team-season rows for 2026")

rows = []
for p in players:
    if (p.get("minutes") or 0) < 300:
        continue
    tid = p.get("teamId")
    t = team_by_id.get(tid)
    if t is None:
        continue
    tm = t.get("totalMinutes")
    ts = t.get("teamStats") or {}
    os_ = t.get("opponentStats") or {}
    if not tm or not ts or not os_:
        continue
    try:
        tm_fg = ts["fieldGoals"]["made"]
        tm_trb = ts["rebounds"]["total"]
        opp_trb = os_["rebounds"]["total"]
        opp_poss = os_["possessions"]
        opp_fga = os_["fieldGoals"]["attempted"]
        opp_3pa = os_["threePointFieldGoals"]["attempted"]
        team_poss = ts["possessions"]

        mp = p["minutes"]
        fga = p["fieldGoals"]["attempted"]
        fgm = p["fieldGoals"]["made"]
        fta = p["freeThrows"]["attempted"]
        tov = p["turnovers"]
        ast = p["assists"]
        stl = p["steals"]
        blk = p["blocks"]
        trb = p["rebounds"]["total"]
        pts = p["points"]

        minutes_share = mp / tm  # == MP / (TmMP/5)
        ast_pct = 100 * ast / ((minutes_share * tm_fg) - fgm) if (minutes_share * tm_fg - fgm) > 0 else np.nan
        trb_pct = 100 * (trb * tm) / (mp * (tm_trb + opp_trb)) if mp > 0 else np.nan
        stl_pct = 100 * (stl * tm) / (mp * opp_poss) if mp > 0 else np.nan
        blk_pct = 100 * (blk * tm) / (mp * (opp_fga - opp_3pa)) if mp > 0 else np.nan
        tov_pct = 100 * tov / (fga + 0.44 * fta + tov) if (fga + 0.44 * fta + tov) > 0 else np.nan
        pts_per_100 = 100 * pts / (mp * (team_poss / tm)) if mp > 0 else np.nan

        rows.append({
            "athleteId": p["athleteId"], "name": p["name"], "team": p["team"], "minutes": mp,
            "PTS_PER_100": pts_per_100,
            "TS%": p.get("trueShootingPct"),
            "USG%": p.get("usage"),
            "AST%": ast_pct, "TOV%": tov_pct, "STL%": stl_pct, "BLK%": blk_pct, "TRB%": trb_pct,
            "FTr": (p.get("freeThrowRate") or np.nan) / 100.0,  # CBBD gives 44.7-style, bref convention is 0.447
            "BPM_proxy_PORPAG": p.get("PORPAG"),
        })
    except (KeyError, TypeError, ZeroDivisionError):
        continue

df = pd.DataFrame(rows)
print(f"{len(df)} qualifying (MIN>=300) NCAA D1 players with complete box-score features")
df.to_csv("/tmp/cbbd_boxscore_features_2026.csv", index=False)

box_cols = ["PTS_PER_100", "TS%", "USG%", "AST%", "TOV%", "STL%", "BLK%", "TRB%", "FTr", "BPM_proxy_PORPAG"]
print(df[box_cols].describe().round(3).to_string())

brown = df[df["name"] == "Mikel Brown Jr."]
print("\nBrown's row:")
print(brown[["name", "team", "minutes"] + box_cols].to_string(index=False))

mu = df[box_cols].mean()
sd = df[box_cols].std()
brown_z = (brown[box_cols].iloc[0] - mu) / sd
print("\nBrown's z-scores (vs NCAA D1 MIN>=300 population, n={}):".format(len(df)))
print(brown_z.round(3).to_string())

mu.to_json("/tmp/cbbd_box_mu.json")
sd.to_json("/tmp/cbbd_box_sd.json")
brown_z.to_json("/tmp/brown_box_z.json")
