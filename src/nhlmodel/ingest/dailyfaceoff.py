"""Starting goalies from DailyFaceoff, with a fallback guess.

DailyFaceoff is unofficial: the page embeds its data as Next.js JSON (`__NEXT_DATA__`).
If the page changes or the request fails, we fall back to `guess_starter`, which picks the goalie
with the most recent starts and rotates on the 2nd night of a back-to-back.

Status values: "confirmed", "likely", "unconfirmed" (DailyFaceoff), "guessed" (our fallback).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date

import pandas as pd

from nhlmodel.ingest.http import get
from nhlmodel.ingest.nhl_api import name_key

log = logging.getLogger(__name__)
URL = "https://www.dailyfaceoff.com/starting-goalies/{d}"
STATUS = {"Confirmed": "confirmed", "Likely": "likely"}


def fetch(d: date) -> pd.DataFrame:
    """One row per team: team abbrev, goalie name, status. Empty frame if the scrape fails."""
    try:
        r = get(URL.format(d=d.isoformat()), retries=3)
        r.raise_for_status()
        m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text, re.S)
        if not m:
            log.warning("DailyFaceoff: no embedded data found")
            return pd.DataFrame()
        games = json.loads(m.group(1))["props"]["pageProps"].get("data") or []
    except Exception as e:  # scraping is best-effort
        log.warning("DailyFaceoff fetch failed: %s", e)
        return pd.DataFrame()
    rows = []
    for g in games:
        for side in ("home", "away"):
            name = g.get(f"{side}GoalieName")
            if not name:
                continue
            rows.append({
                "team": g.get(f"{side}TeamAbbreviation"),
                "goalie_name": name,
                "goalie_key": name_key(name),
                "status": STATUS.get(g.get(f"{side}NewsStrengthName"), "unconfirmed"),
                "news_at": g.get(f"{side}NewsCreatedAt"),
            })
    return pd.DataFrame(rows)


def guess_starter(goalie_games: pd.DataFrame, team_id: int, asof: date, b2b: bool,
                  roster_goalie_ids: set[int] | None = None) -> int | None:
    """Most frequent starter in the team's last 10 games; on a back-to-back, the other goalie
    if yesterday's starter is the usual starter."""
    gg = goalie_games[(goalie_games.team_id == team_id) & goalie_games.starter &
                      (pd.to_datetime(goalie_games.game_date).dt.date < asof)].sort_values("game_date")
    if roster_goalie_ids:
        gg = gg[gg.player_id.isin(roster_goalie_ids)]
    if gg.empty:
        return None
    last10 = gg.tail(10)
    counts = last10.player_id.value_counts()
    top = int(counts.index[0])
    if b2b and len(counts) > 1 and int(gg.iloc[-1].player_id) == top:
        return int(counts.index[1])
    return top
