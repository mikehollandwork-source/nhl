"""Pre-game prices for past NHL games, for backtesting betting rules (free, no keys).

Kalshi: settled markets of the current season (winner, totals ladder, puck line) with hourly
bid/ask candles. Polymarket: closed game events back to Oct 2024 with price history (mid price only,
so the buy price is taken as mid + POLY_HALF_SPREAD).

The stored quote for each selection is the last one at least LEAD_MIN minutes before puck drop.
Rows: season, game_id, game_date, home, away, source, market, selection, line, bid, ask, quote_ts.
"""

from __future__ import annotations

import json
import logging
import re
import time
from datetime import datetime, timedelta

import httpx
import pandas as pd

from nhlmodel.config import tz
from nhlmodel.ingest import kalshi as KS
from nhlmodel.ingest.http import TIMEOUT, UA, fetch_many
from nhlmodel.ingest.nhl_api import name_key

log = logging.getLogger(__name__)
K = "https://api.elections.kalshi.com/trade-api/v2"
GAMMA = "https://gamma-api.polymarket.com/events"
CLOB = "https://clob.polymarket.com/prices-history"
LEAD_MIN = 30
POLY_HALF_SPREAD = 0.01
POLY_TOTAL_LINES = (5.5, 6.5)
SLUG = re.compile(r"^nhl-[a-z]+-[a-z]+-(\d{4}-\d{2}-\d{2})$")


def _get(url: str, params: dict, tries: int = 8) -> dict | list:
    """GET with patient backoff on 429 (Kalshi and Polymarket rate-limit bursts)."""
    for i in range(tries):
        r = httpx.get(url, params=params, timeout=TIMEOUT, headers={"User-Agent": UA}, follow_redirects=True)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(min(2 ** i, 30))
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"GET {url} kept failing")


def slate_by_date(games: pd.DataFrame) -> dict:
    """ET date -> list of (home, away, game_id, start_ts)."""
    g = games.copy()
    start = pd.to_datetime(g.start_utc, utc=True)
    g["d"] = start.dt.tz_convert(tz()).dt.date
    g["ts"] = start.map(lambda x: int(x.timestamp()))
    out: dict = {}
    for r in g.itertuples():
        out.setdefault(r.d, []).append((r.home_abbrev, r.away_abbrev, int(r.game_id), int(r.ts)))
    return out


def pick_quote(points: list[tuple[int, float, float]], start_ts: int, lead_min: int = LEAD_MIN):
    """points: (ts, bid, ask). The last quote at least lead_min before puck drop, or None."""
    ok = [p for p in points if p[0] <= start_ts - lead_min * 60 and 0 < p[1] <= p[2] < 1]
    return max(ok, key=lambda p: p[0]) if ok else None


# ----------------------------------------------------------------------------- Kalshi

def kalshi_settled(series: str) -> list[dict]:
    out, cur = [], None
    for _ in range(200):
        p = {"series_ticker": series, "status": "settled", "limit": 1000}
        if cur:
            p["cursor"] = cur
        d = _get(K + "/markets", p)
        out += d.get("markets", [])
        cur = d.get("cursor")
        if not cur or not d.get("markets"):
            break
        time.sleep(0.4)
    return out


def kalshi_history(games: pd.DataFrame, season: int) -> pd.DataFrame:
    slate = slate_by_date(games)
    by_code = {KS.date_code(d): d for d in slate}
    rows = []
    for series in KS.GAME_SERIES:
        mk = kalshi_settled(series)
        log.info("kalshi %s: %d settled markets", series, len(mk))
        groups: dict = {}
        for m in mk:
            code = m["ticker"].split("-")[1][:7]
            if code in by_code:
                groups.setdefault(by_code[code], []).append(m)
        for d, ms in sorted(groups.items()):
            games_d = slate[d]
            starts = {(h, a): (gid, ts) for h, a, gid, ts in games_d}
            pairs = [(h, a) for h, a, _, _ in games_d]
            lo, hi = min(ts for *_, ts in games_d) - 8 * 3600, max(ts for *_, ts in games_d)
            quotes: dict[str, list] = {}
            ticks = [m["ticker"] for m in ms]
            for i in range(0, len(ticks), 60):
                d2 = _get(K + "/markets/candlesticks", {"market_tickers": ",".join(ticks[i:i + 60]), "start_ts": lo,
                                                        "end_ts": hi, "period_interval": 60})
                for mm in d2.get("markets", []):
                    pts = []
                    for c in mm.get("candlesticks", []):
                        try:
                            pts.append((int(c["end_period_ts"]), float(c["yes_bid"]["close_dollars"]),
                                        float(c["yes_ask"]["close_dollars"])))
                        except (KeyError, TypeError, ValueError):
                            continue
                    quotes[mm.get("market_ticker") or mm.get("ticker")] = pts
                time.sleep(0.25)
            for m in ms:
                pair = KS._pair_for(m["ticker"].split("-")[1][7:], pairs)
                if pair is None:
                    continue
                gid, start = starts[pair]
                q = pick_quote(quotes.get(m["ticker"], []), start)
                if q is None:
                    continue
                fake = {**m, "yes_bid_dollars": q[1], "yes_ask_dollars": q[2], "no_bid_dollars": 1 - q[2],
                        "no_ask_dollars": 1 - q[1]}
                df = KS.parse_game_markets([fake], d, [pair], max_spread=1.0)
                for r in df.itertuples():
                    rows.append({"season": season, "game_id": gid, "game_date": d.isoformat(), "home": r.home,
                                 "away": r.away, "source": "kalshi", "market": r.market, "selection": r.selection,
                                 "line": r.line, "bid": r.bid, "ask": r.ask, "quote_ts": q[0]})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- Polymarket

def poly_closed_events(first: str, last: str) -> list[dict]:
    """Closed NHL game events with slug dates in [first, last] (ISO dates). The API pages 100 at a time."""
    out, off = [], 0
    lo = (datetime.fromisoformat(first) - timedelta(days=2)).strftime("%Y-%m-%dT00:00:00Z")
    hi = (datetime.fromisoformat(last) + timedelta(days=30)).strftime("%Y-%m-%dT00:00:00Z")
    while True:
        e = _get(GAMMA, {"tag_slug": "nhl", "closed": "true", "limit": 100, "offset": off,
                         "end_date_min": lo, "end_date_max": hi})
        if not e:
            break
        for v in e:
            m = SLUG.match(v.get("slug", ""))
            if m and first <= m.group(1) <= last:
                out.append(v)
        if len(e) < 100:
            break
        off += 100
        time.sleep(0.2)
    return out


def poly_markets(events: list[dict], slate: dict, nick: dict[str, str]) -> list[dict]:
    """One entry per usable market: game, market type, first outcome's selection/line, token id."""
    out = []
    for ev in events:
        d = datetime.fromisoformat(SLUG.match(ev["slug"]).group(1)).date()
        games_d = slate.get(d, [])
        if not games_d:
            continue
        pairs = {frozenset((h, a)): (h, a, gid, ts) for h, a, gid, ts in games_d}
        for m in ev.get("markets", []):
            kind = m.get("sportsMarketType")
            try:
                outcomes = json.loads(m["outcomes"]) if isinstance(m.get("outcomes"), str) else m.get("outcomes")
                tok = json.loads(m["clobTokenIds"])[0] if isinstance(m.get("clobTokenIds"), str) else m["clobTokenIds"][0]
            except (KeyError, ValueError, TypeError, IndexError):
                continue
            if not outcomes or len(outcomes) != 2:
                continue
            if kind == "totals":
                if m.get("line") is None or float(m["line"]) not in POLY_TOTAL_LINES or outcomes[0].lower() != "over":
                    continue
                teams = [nick.get(name_key(t)) for t in re.split(r"\s+vs\.?\s+", ev.get("title", ""))]
                g = pairs.get(frozenset(teams)) if None not in teams else None
                if g:
                    out.append({"g": g, "market": "totals", "sel": "over", "other": "under", "line": float(m["line"]),
                                "oline": float(m["line"]), "token": tok})
                continue
            teams = [nick.get(name_key(o)) for o in outcomes]
            g = pairs.get(frozenset(teams)) if None not in teams else None
            if not g:
                continue
            sel = "home" if teams[0] == g[0] else "away"
            other = "away" if sel == "home" else "home"
            if kind == "moneyline":
                out.append({"g": g, "market": "h2h", "sel": sel, "other": other, "line": None, "oline": None, "token": tok})
            elif kind == "spreads":
                # the handicap is read from the text ("Spread: Oilers (-1.5)"): the `line` field had the
                # wrong sign in early 2025-26
                hc = re.search(r"\(([-+]?\d+(?:\.\d+)?)\)", m.get("question") or "")
                if not hc or abs(float(hc.group(1))) != 1.5:
                    continue
                ln = float(hc.group(1))
                out.append({"g": g, "market": "spreads", "sel": sel, "other": other, "line": ln, "oline": -ln, "token": tok})
    return out


def polymarket_history(games: pd.DataFrame, season: int, nick: dict[str, str]) -> pd.DataFrame:
    slate = slate_by_date(games)
    days = sorted(slate)
    events = poly_closed_events(days[0].isoformat(), (days[-1] + timedelta(days=1)).isoformat())
    mk = poly_markets(events, slate, nick)
    log.info("polymarket %s: %d events, %d markets to price", season, len(events), len(mk))
    urls = {}
    for i, m in enumerate(mk):
        start = m["g"][3]
        urls[i] = f"{CLOB}?market={m['token']}&startTs={start - 8 * 3600}&endTs={start}&fidelity=10"
    rows = []
    idx = list(urls)
    for j in range(0, len(idx), 400):
        got = fetch_many([urls[i] for i in idx[j:j + 400]], concurrency=6)
        for i in idx[j:j + 400]:
            h = (got.get(urls[i]) or {}).get("history") or []
            m = mk[i]
            hs, a, gid, start = m["g"]
            q = pick_quote([(int(p["t"]), float(p["p"]) - POLY_HALF_SPREAD, float(p["p"]) + POLY_HALF_SPREAD)
                            for p in h if 0.02 < float(p["p"]) < 0.98], start)
            if q is None:
                continue
            d = datetime.fromtimestamp(start, tz()).date().isoformat()
            base = {"season": season, "game_id": gid, "game_date": d, "home": hs, "away": a, "source": "polymarket",
                    "market": m["market"], "quote_ts": q[0]}
            rows.append({**base, "selection": m["sel"], "line": m["line"], "bid": q[1], "ask": q[2]})
            rows.append({**base, "selection": m["other"], "line": m["oline"], "bid": 1 - q[2], "ask": 1 - q[1]})
        log.info("polymarket %s: %d/%d markets priced", season, min(j + 400, len(idx)), len(idx))
    return pd.DataFrame(rows)


def team_nicknames(t: pd.DataFrame) -> dict[str, str]:
    """Nickname -> abbrev, plus the names Polymarket uses for Utah."""
    nick = {name_key(c): a for a, c in zip(t.abbrev, t.common, strict=True)}
    nick.update({"utah": "UTA", "utahhockeyclub": "UTA"})
    return nick


def backfill(seasons: list[int], current: int) -> dict[str, int]:
    """Download and store pre-game price history (table market_history, one partition per source+season)."""
    from nhlmodel.pipelines.daily import team_names
    from nhlmodel.store import db
    t = team_names()
    nick = team_nicknames(t)
    out = {}
    for s in seasons:
        games = db.read("games", partitions=[str(s)])
        games = games[games.game_type == 2].drop_duplicates("game_id", keep="last")
        pm = polymarket_history(games, s, nick)
        if not pm.empty:
            db.write("market_history", pm, f"polymarket_{s}")
        out[f"polymarket_{s}"] = len(pm)
        if s == current:
            ks = kalshi_history(games, s)
            if not ks.empty:
                db.write("market_history", ks, f"kalshi_{s}")
            out[f"kalshi_{s}"] = len(ks)
    return out
