"""Travel and schedule features per team-game (all known before the game).

  tz_shift     : venue UTC offset now minus at the team's previous game (+ = travelled east)
  tz_big       : |tz_shift| >= 2 hours
  rest_days    : days since previous game, capped at 4 (0 never happens; 1 = back-to-back)
  games_last7  : games played in the previous 7 days
  road_trip    : consecutive road games including this one (0 at home)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def travel_features(games: pd.DataFrame) -> pd.DataFrame:
    g = games[["game_id", "game_date", "home_id", "away_id", "venue_utc_offset"]].copy()
    g["game_date"] = pd.to_datetime(g["game_date"])
    rows = pd.concat([
        g.rename(columns={"home_id": "team_id"}).assign(is_home=1)[["game_id", "game_date", "team_id", "is_home", "venue_utc_offset"]],
        g.rename(columns={"away_id": "team_id"}).assign(is_home=0)[["game_id", "game_date", "team_id", "is_home", "venue_utc_offset"]],
    ]).sort_values(["team_id", "game_date", "game_id"])
    out = []
    for _, t in rows.groupby("team_id", sort=False):
        t = t.copy()
        off = t.venue_utc_offset.ffill()
        t["tz_shift"] = (off - off.shift()).fillna(0.0)
        t["tz_big"] = (t.tz_shift.abs() >= 2).astype(int)
        t["rest_days"] = t.game_date.diff().dt.days.fillna(4).clip(upper=4)
        d = t.game_date.to_numpy()
        t["games_last7"] = [int(((d < x) & (d >= x - np.timedelta64(7, "D"))).sum()) for x in d]
        trip, run = [], 0
        for h in t.is_home:
            run = 0 if h else run + 1
            trip.append(run)
        t["road_trip"] = trip
        out.append(t)
    r = pd.concat(out)
    return r[["game_id", "team_id", "tz_shift", "tz_big", "rest_days", "games_last7", "road_trip"]]
