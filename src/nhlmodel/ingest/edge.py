"""NHL EDGE (puck/player tracking) team summaries per season.

Only season-level numbers are public (no game-by-game history), so the model uses the PRIOR
season's values, which keeps backtests free of look-ahead. Verified fields (Oct 2026):
skatingSpeed.burstsOver20/22, speedMax, shotSpeed.shotAttemptsOver90 / topShotSpeed,
distanceSkated.total, zoneTimeDetails.offensiveZonePctg / offensiveZoneEvPctg, team.gamesPlayed.
"""

from __future__ import annotations

import logging

import pandas as pd

from nhlmodel.ingest.http import fetch_many

log = logging.getLogger(__name__)
URL = "https://api-web.nhle.com/v1/edge/team-detail/{team}/{season}/2"


def parse_team(d: dict, team_id: int, season: int) -> dict | None:
    try:
        gp = d["team"].get("gamesPlayed") or 0
        if gp <= 0:
            return None
        sk, sh, z = d.get("skatingSpeed", {}), d.get("shotSpeed", {}), d.get("zoneTimeDetails", {})
        return {
            "team_id": team_id, "season": season, "games": gp,
            "bursts20_pg": sk["burstsOver20"]["value"] / gp,
            "bursts22_pg": sk["burstsOver22"]["value"] / gp,
            "speed_max": sk["speedMax"]["imperial"],
            "shots90_pg": sh["shotAttemptsOver90"]["value"] / gp,
            "top_shot_speed": sh["topShotSpeed"]["imperial"],
            "distance_pg": d["distanceSkated"]["total"]["imperial"] / gp,
            "oz_pct": z.get("offensiveZonePctg"), "oz_ev_pct": z.get("offensiveZoneEvPctg"),
            "dz_pct": z.get("defensiveZonePctg"),
        }
    except (KeyError, TypeError) as e:
        log.warning("EDGE parse failed team %s season %s: %s", team_id, season, e)
        return None


def fetch_seasons(team_ids: list[int], seasons: list[int]) -> pd.DataFrame:
    urls = {(t, s): URL.format(team=t, season=s) for t in team_ids for s in seasons}
    got = fetch_many(list(urls.values()), concurrency=6)
    rows = [parse_team(got[u], t, s) for (t, s), u in urls.items() if got.get(u)]
    return pd.DataFrame([r for r in rows if r])
