"""Polymarket NHL game lines (free, no key): moneyline, totals and spreads.

Event slugs look like nhl-cal-edm-2026-10-28 (ET game date). Each market has two outcomes with
bestBid / bestAsk for the first outcome; the probability is the bid-ask mid. Rules checked
Oct 2026: the moneyline includes overtime and shootouts (same as the model's h2h market).
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date

import pandas as pd

from nhlmodel.ingest.http import get_json
from nhlmodel.ingest.nhl_api import name_key

log = logging.getLogger(__name__)
URL = "https://gamma-api.polymarket.com/events"
SOURCE = "polymarket"


def fetch_events() -> list[dict]:
    out, offset = [], 0
    while True:
        page = get_json(URL, {"tag_slug": "nhl", "closed": "false", "limit": 100, "offset": offset})
        if not page:
            break
        out += page
        if len(page) < 100:
            break
        offset += 100
    return out


def _mid(m: dict) -> tuple[float | None, float | None, float | None]:
    try:
        bid, ask = float(m.get("bestBid")), float(m.get("bestAsk"))
    except (TypeError, ValueError):
        return None, None, None
    if not (0 < bid <= ask < 1):
        return None, bid, ask
    return (bid + ask) / 2, bid, ask


def parse_events(events: list[dict], d: date, slate: list[tuple[str, str]], nick: dict[str, str],
                 max_spread: float = 0.10) -> pd.DataFrame:
    """slate: (home, away) abbrevs; nick: name_key(nickname) -> abbrev.
    Rows: home, away, market, selection, line, prob, bid, ask, liquidity, source."""
    pairs = {frozenset(p): p for p in slate}
    rows = []
    for ev in events:
        if not str(ev.get("slug", "")).endswith(d.isoformat()):
            continue
        for m in ev.get("markets", []):
            if m.get("closed") or not m.get("active", True):
                continue
            try:
                outcomes = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes")
            except (KeyError, ValueError):
                continue
            if not outcomes or len(outcomes) != 2:
                continue
            kind = m.get("sportsMarketType")
            p, bid, ask = _mid(m)
            if p is None or ask - bid > max_spread:
                continue
            base = {"bid": bid, "ask": ask, "liquidity": m.get("liquidityNum"), "source": SOURCE}
            flip = {**base, "bid": 1 - ask, "ask": 1 - bid}   # the other outcome's quotes
            if kind == "totals":
                teams = [nick.get(name_key(t)) for t in re.split(r"\s+vs\.?\s+", ev.get("title", ""))]
                pair = pairs.get(frozenset(teams)) if None not in teams else None
                if not pair or m.get("line") is None:
                    continue
                first = outcomes[0].lower()
                if first not in ("over", "under"):
                    continue
                other = "under" if first == "over" else "over"
                ln = float(m["line"])
                rows += [{"home": pair[0], "away": pair[1], "market": "totals", "selection": first, "line": ln,
                          "prob": p, **base},
                         {"home": pair[0], "away": pair[1], "market": "totals", "selection": other, "line": ln,
                          "prob": 1 - p, **flip}]
                continue
            teams = [nick.get(name_key(o)) for o in outcomes]
            pair = pairs.get(frozenset(teams)) if None not in teams else None
            if not pair:
                continue
            side = ["home" if t == pair[0] else "away" for t in teams]
            if kind == "moneyline":
                rows += [{"home": pair[0], "away": pair[1], "market": "h2h", "selection": side[0], "line": None,
                          "prob": p, **base},
                         {"home": pair[0], "away": pair[1], "market": "h2h", "selection": side[1], "line": None,
                          "prob": 1 - p, **flip}]
            elif kind == "spreads" and m.get("line") is not None:
                ln = float(m["line"])          # first outcome's handicap (negative = favourite)
                rows += [{"home": pair[0], "away": pair[1], "market": "spreads", "selection": side[0], "line": ln,
                          "prob": p, **base},
                         {"home": pair[0], "away": pair[1], "market": "spreads", "selection": side[1], "line": -ln,
                          "prob": 1 - p, **flip}]
    return pd.DataFrame(rows)


def game_lines(d: date, slate: list[tuple[str, str]], teams: pd.DataFrame, max_spread: float = 0.10) -> pd.DataFrame:
    """teams: abbrev + common (nickname), e.g. from daily.team_names()."""
    nick = {name_key(c): a for a, c in zip(teams.abbrev, teams.common, strict=True)}
    return parse_events(fetch_events(), d, slate, nick, max_spread)
