"""Game lines from the prediction-market exchanges (Polymarket, Kalshi), as sportsbook-style odds.

No API keys. Each exchange is treated as a "book" whose price for a selection is its BUY price
(the ask, what you would actually pay), converted from cents to American odds (38c -> +163).
The output has the same shape the rest of the pipeline uses for odds snapshots, so de-vigging,
consensus, disagreements, grading, closing-line value and the weekly line report work unchanged.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime

import pandas as pd

from nhlmodel.config import settings
from nhlmodel.ingest import kalshi, polymarket
from nhlmodel.market.devig import cents_to_american

log = logging.getLogger(__name__)
COLS = ["ts", "snapshot_type", "event_id", "commence_time", "home", "away", "book", "market", "selection",
        "line", "price", "ask", "bid", "book_update"]


def to_odds(lines: pd.DataFrame, book: str, starts: dict[tuple[str, str], str], d: date, snapshot_type: str,
            ts: str) -> pd.DataFrame:
    """lines: home, away, market, selection, line, ask, bid. Keeps the best (lowest) ask per selection."""
    if lines is None or lines.empty:
        return pd.DataFrame(columns=COLS)
    x = lines.copy()
    x["line_key"] = x.line.astype(float).fillna(0.0).round(2)
    x = x.sort_values("ask").drop_duplicates(["home", "away", "market", "selection", "line_key"], keep="first")
    x = x[(x.ask > 0) & (x.ask < 1)]
    x["price"] = x.ask.map(cents_to_american)
    x["book"] = book
    x["event_id"] = [f"{d.isoformat()}-{a}-{h}" for h, a in zip(x.home, x.away, strict=True)]
    x["commence_time"] = [starts.get((h, a)) for h, a in zip(x.home, x.away, strict=True)]
    x["ts"], x["snapshot_type"], x["book_update"] = ts, snapshot_type, ts
    x["line"] = x.line.where(x.market != "h2h", None)
    return x[COLS].reset_index(drop=True)


def game_odds(d: date, slate: list[tuple[str, str, str]], teams: pd.DataFrame, snapshot_type: str) -> pd.DataFrame:
    """slate: (home, away, start_utc). teams: abbrev + common (nickname)."""
    cfg = settings().get("lines", {})
    spread = cfg.get("max_spread", 0.10)
    pairs = [(h, a) for h, a, _ in slate]
    starts = {(h, a): s for h, a, s in slate}
    ts = datetime.now(UTC).isoformat(timespec="seconds")
    frames = []
    if cfg.get("polymarket", True):
        try:
            frames.append(to_odds(polymarket.game_lines(d, pairs, teams, spread), "polymarket", starts, d,
                                  snapshot_type, ts))
        except Exception:
            log.exception("polymarket game lines failed")
    if cfg.get("kalshi", True):
        try:
            frames.append(to_odds(kalshi.game_lines(d, pairs, spread), "kalshi", starts, d, snapshot_type, ts))
        except Exception:
            log.exception("kalshi game lines failed")
    frames = [f for f in frames if not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=COLS)
