"""Kalshi NHL player props (free, no key): N+ goals, points and assists per player per game.

Market example: ticker KXNHLPTS-26OCT28CGYEDM-EDMTFREDERIC10-1, yes_sub_title "Trent Frederic: 1+",
yes_bid_dollars / yes_ask_dollars. "N+" is the over at line N - 0.5. The probability is the bid-ask
mid. Rules checked Oct 2026: a player who never enters the game settles at the last fair price
(effectively void), so the model's "if he plays" probability is the right comparison.
"""

from __future__ import annotations

import logging
import re
from datetime import date

import pandas as pd

from nhlmodel.ingest.http import get_json
from nhlmodel.ingest.nhl_api import name_key

log = logging.getLogger(__name__)
URL = "https://api.elections.kalshi.com/trade-api/v2/markets"
SERIES = {"KXNHLGOAL": "goals", "KXNHLPTS": "points", "KXNHLAST": "assists"}
SOURCE = "kalshi"


def date_code(d: date) -> str:
    return d.strftime("%y%b%d").upper()          # 2026-10-28 -> 26OCT28


def fetch_markets(series: str) -> list[dict]:
    out, cursor = [], None
    for _ in range(20):
        params = {"series_ticker": series, "status": "open", "limit": 1000}
        if cursor:
            params["cursor"] = cursor
        page = get_json(URL, params)
        out += page.get("markets", [])
        cursor = page.get("cursor")
        if not cursor:
            break
    return out


def _f(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_markets(markets: list[dict], market: str, d: date, players: pd.DataFrame,
                  max_spread: float = 0.10) -> tuple[pd.DataFrame, int]:
    """players: player_id, player, team for tonight. Returns (rows, number of unmatched players)."""
    code = date_code(d)
    pl = players.assign(key=players.player.map(name_key))
    by_key = {k: g for k, g in pl.groupby("key")}
    rows, unmatched = [], set()
    for m in markets:
        parts = m.get("ticker", "").split("-")
        if len(parts) < 4 or not parts[1].startswith(code):
            continue
        sub = m.get("yes_sub_title") or ""
        mm = re.match(r"^(.*):\s*(\d+)\+$", sub.strip())
        if not mm:
            continue
        name, n = mm.group(1), int(mm.group(2))
        cand = by_key.get(name_key(name))
        if cand is None:
            unmatched.add(name)
            continue
        if len(cand) > 1:                      # same name on both teams: the ticker starts with the team code
            cand = cand[[parts[2].startswith(t) for t in cand.team]]
            if len(cand) != 1:
                unmatched.add(name)
                continue
        bid, ask = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
        if bid is None or ask is None or not (0 < bid <= ask < 1) or ask - bid > max_spread:
            continue
        r = cand.iloc[0]
        rows.append({"player_id": int(r.player_id), "player": r.player, "team": r.team, "market": market,
                     "line": n - 0.5, "prob": (bid + ask) / 2, "bid": bid, "ask": ask,
                     "ticker": m["ticker"], "source": SOURCE})
    return pd.DataFrame(rows), len(unmatched)


def prop_lines(d: date, players: pd.DataFrame, max_spread: float = 0.10) -> pd.DataFrame:
    frames, miss = [], 0
    for series, market in SERIES.items():
        try:
            df, n = parse_markets(fetch_markets(series), market, d, players, max_spread)
        except Exception:
            log.exception("kalshi %s fetch failed", series)
            continue
        frames.append(df)
        miss += n
    if miss:
        log.info("kalshi: %d player names not matched to tonight's rosters", miss)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


# ---- game markets (winner incl. OT, totals, spreads) ----
# Tickers (checked Oct 2026): KXNHLGAME-26NOV01EDMNYR-NYR ("New York R wins"),
# KXNHLTOTAL-26OCT28CGYEDM-7 (strike 6.5, "Over 6.5 goals"), KXNHLSPREAD-26OCT28CGYEDM-EDM2
# ("Edmonton wins by over 1.5 goals"). The event code is away + home. Kalshi team codes are sometimes
# shorter than the NHL's (NJ for NJD), so codes are matched loosely.
GAME_SERIES = ("KXNHLGAME", "KXNHLTOTAL", "KXNHLSPREAD")


def _codes(abbrev: str) -> list[str]:
    return list(dict.fromkeys([abbrev, abbrev[:2]]))


def _pair_for(code: str, slate: list[tuple[str, str]]) -> tuple[str, str] | None:
    for h, a in slate:
        if any(code == ca + ch for ca in _codes(a) for ch in _codes(h)):
            return h, a
    return None


def _team_of(code: str, pair: tuple[str, str]) -> str | None:
    for t in pair:
        if code in _codes(t):
            return t
    return None


def parse_game_markets(markets: list[dict], d: date, slate: list[tuple[str, str]],
                       max_spread: float = 0.10) -> pd.DataFrame:
    """Rows: home, away, market, selection, line, ask, bid, source. One row per side; the 'No' side of a
    total is the under, the 'No' side of 'X wins by over N' is the opponent +N."""
    code = date_code(d)
    rows = []
    for m in markets:
        parts = m.get("ticker", "").split("-")
        if len(parts) < 3 or not parts[1].startswith(code):
            continue
        pair = _pair_for(parts[1][len(code):], slate)
        if pair is None:
            continue
        home, away = pair
        yb, ya = _f(m.get("yes_bid_dollars")), _f(m.get("yes_ask_dollars"))
        nb, na = _f(m.get("no_bid_dollars")), _f(m.get("no_ask_dollars"))
        if None in (yb, ya, nb, na) or not (0 < yb <= ya < 1) or ya - yb > max_spread:
            continue
        series = parts[0]
        base = {"home": home, "away": away, "source": SOURCE}
        if series == "KXNHLGAME":
            t = _team_of(parts[2], pair)
            if t is None:
                continue
            sel = "home" if t == home else "away"
            other = "away" if sel == "home" else "home"
            rows += [{**base, "market": "h2h", "selection": sel, "line": None, "ask": ya, "bid": yb},
                     {**base, "market": "h2h", "selection": other, "line": None, "ask": na, "bid": nb}]
        elif series == "KXNHLTOTAL":
            ln = _f(m.get("floor_strike"))
            if ln is None:
                mm = re.search(r"Over ([\d.]+)", m.get("title", ""))
                ln = float(mm.group(1)) if mm else None
            if ln is None:
                continue
            rows += [{**base, "market": "totals", "selection": "over", "line": ln, "ask": ya, "bid": yb},
                     {**base, "market": "totals", "selection": "under", "line": ln, "ask": na, "bid": nb}]
        elif series == "KXNHLSPREAD":
            mm = re.match(r"^([A-Z]+)(\d+)$", parts[2])
            if not mm:
                continue
            t = _team_of(mm.group(1), pair)
            if t is None:
                continue
            ln = int(mm.group(2)) - 0.5            # "wins by over 1.5" = -1.5
            sel = "home" if t == home else "away"
            other = "away" if sel == "home" else "home"
            rows += [{**base, "market": "spreads", "selection": sel, "line": -ln, "ask": ya, "bid": yb},
                     {**base, "market": "spreads", "selection": other, "line": ln, "ask": na, "bid": nb}]
    return pd.DataFrame(rows)


def game_lines(d: date, slate: list[tuple[str, str]], max_spread: float = 0.10) -> pd.DataFrame:
    frames = []
    for s in GAME_SERIES:
        try:
            frames.append(parse_game_markets(fetch_markets(s), d, slate, max_spread))
        except Exception:
            log.exception("kalshi %s fetch failed", s)
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
