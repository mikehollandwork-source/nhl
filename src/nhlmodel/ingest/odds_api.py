"""The Odds API client (https://the-odds-api.com, v4).

Verified from their docs (Oct 2026):
- /v4/sports/{sport}/odds costs 1 credit per market per region; empty responses are free.
- Player props are per event: /v4/sports/{sport}/events/{id}/odds, cost = markets returned x regions.
- Historical odds are paid plans only.
- Usage comes back in headers x-requests-remaining / x-requests-used / x-requests-last.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

import pandas as pd

from nhlmodel.config import env, settings
from nhlmodel.ingest.http import get
from nhlmodel.ingest.nhl_api import name_key

log = logging.getLogger(__name__)
BASE = "https://api.the-odds-api.com/v4/sports/icehockey_nhl"


class OddsUnavailable(RuntimeError):
    pass


def _key() -> str:
    k = env("ODDS_API_KEY")
    if not k:
        raise OddsUnavailable("ODDS_API_KEY not set")
    return k


def _usage(r) -> dict:
    h = r.headers
    return {"remaining": h.get("x-requests-remaining"), "used": h.get("x-requests-used"),
            "last": h.get("x-requests-last")}


def fetch_game_odds() -> tuple[list[dict], dict]:
    cfg = settings()["odds"]
    r = get(f"{BASE}/odds", params={"apiKey": _key(), "regions": cfg["regions"],
                                    "markets": ",".join(cfg["game_markets"]),
                                    "oddsFormat": "american", "dateFormat": "iso"})
    if r.status_code == 401:
        raise OddsUnavailable("Odds API rejected the key (401)")
    r.raise_for_status()
    return r.json(), _usage(r)


def fetch_event_props(event_id: str) -> tuple[dict, dict]:
    cfg = settings()["odds"]
    if not cfg.get("prop_markets"):
        return {}, {}
    r = get(f"{BASE}/events/{event_id}/odds", params={"apiKey": _key(), "regions": cfg["regions"],
                                                      "markets": ",".join(cfg["prop_markets"]),
                                                      "oddsFormat": "american", "dateFormat": "iso"})
    r.raise_for_status()
    return r.json(), _usage(r)


def team_matcher(teams: pd.DataFrame):
    """teams: columns abbrev, place, common. Returns fn(odds_api_name) -> abbrev or None.
    Matching on the nickname ("Canadiens", "Blues", "Mammoth") avoids accent / 'St.' differences."""
    keys = [(name_key(r.common), r.abbrev) for r in teams.itertuples()]

    def match(name: str) -> str | None:
        k = name_key(name)
        for common, abbrev in keys:
            if k.endswith(common):
                return abbrev
        return None
    return match


def normalize_game_odds(events: list[dict], match, snapshot_type: str, ts: str | None = None) -> pd.DataFrame:
    """Flatten to one row per (event, book, market, selection)."""
    ts = ts or datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for ev in events:
        home, away = match(ev["home_team"]), match(ev["away_team"])
        if not home or not away:
            log.warning("unmatched teams: %s @ %s", ev["away_team"], ev["home_team"])
            continue
        for bk in ev.get("bookmakers", []):
            for mk in bk.get("markets", []):
                for o in mk.get("outcomes", []):
                    if mk["key"] in ("h2h", "spreads"):
                        sel = "home" if o["name"] == ev["home_team"] else "away" if o["name"] == ev["away_team"] else "draw"
                    else:
                        sel = o["name"].lower()  # Over / Under
                    rows.append({
                        "ts": ts, "snapshot_type": snapshot_type, "event_id": ev["id"],
                        "commence_time": ev["commence_time"], "home": home, "away": away,
                        "book": bk["key"], "market": mk["key"], "selection": sel,
                        "line": o.get("point"), "price": o["price"],
                        "book_update": mk.get("last_update") or bk.get("last_update"),
                    })
    return pd.DataFrame(rows)


def normalize_props(event: dict, match, snapshot_type: str, ts: str | None = None) -> pd.DataFrame:
    ts = ts or datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    home, away = match(event.get("home_team", "")), match(event.get("away_team", ""))
    for bk in event.get("bookmakers", []):
        for mk in bk.get("markets", []):
            for o in mk.get("outcomes", []):
                rows.append({
                    "ts": ts, "snapshot_type": snapshot_type, "event_id": event["id"],
                    "commence_time": event.get("commence_time"), "home": home, "away": away,
                    "book": bk["key"], "market": mk["key"], "player": o.get("description"),
                    "player_key": name_key(o.get("description") or ""),
                    "selection": o["name"].lower(), "line": o.get("point"), "price": o["price"],
                })
    return pd.DataFrame(rows)
