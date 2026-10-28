"""Compare model probabilities with de-vigged book probabilities.

For each game / market / line / selection:
  book_prob   = median across books of that book's de-vigged probability (multiplicative)
  best_price  = best American price on offer for that selection (on an exchange: the buy price)
  price_prob  = probability implied by best_price (what you actually pay)
  edge        = model_prob - price_prob  (so a flagged pick beats a price you can actually get)
  fair_edge   = model_prob - book_prob
  ev          = model_prob * decimal(best_price) - 1
  flagged     = |edge| >= threshold for the market, and the flagged pick is the side with edge > 0
Every row is stored (flagged or not) so performance can be measured by size of disagreement.
The model probability is the raw model: it is never blended with the market here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nhlmodel.config import settings
from nhlmodel.market import devig as D

PAIRS = {"h2h": ("home", "away"), "spreads": ("home", "away"), "totals": ("over", "under")}


def per_book_fair(odds: pd.DataFrame, method: str = "multiplicative") -> pd.DataFrame:
    """De-vig each book's two-way market. Spreads pair home line L with away line -L."""
    out = []
    for (eid, book, market), g in odds.groupby(["event_id", "book", "market"]):
        if market not in PAIRS:
            continue
        a, b = PAIRS[market]
        ga, gb = g[g.selection == a], g[g.selection == b]
        for ra in ga.itertuples():
            if market == "h2h":
                rb = gb
            elif market == "spreads":
                rb = gb[np.isclose(gb.line, -ra.line)]
            else:
                rb = gb[np.isclose(gb.line, ra.line)]
            if rb.empty:
                continue
            rb = rb.iloc[0]
            fair = D.devig([ra.price, rb.price], method)
            base = {"event_id": eid, "book": book, "market": market, "home": ra.home, "away": ra.away,
                    "commence_time": ra.commence_time, "overround": D.overround([ra.price, rb.price])}
            out.append({**base, "selection": a, "line": ra.line, "price": ra.price, "fair": fair[0]})
            out.append({**base, "selection": b, "line": rb.line, "price": rb.price, "fair": fair[1]})
    return pd.DataFrame(out)


def consensus(fair: pd.DataFrame) -> pd.DataFrame:
    """Median fair prob and best price across books for each (event, market, selection, line)."""
    if fair.empty:
        return fair
    fair = fair.copy()
    fair["line_key"] = fair["line"].fillna(0.0)
    fair["dec"] = fair["price"].map(D.american_to_decimal)
    g = fair.groupby(["event_id", "home", "away", "commence_time", "market", "selection", "line_key"])
    agg = g.agg(book_prob=("fair", "median"), n_books=("book", "nunique"), best_dec=("dec", "max"),
                avg_overround=("overround", "mean")).reset_index()
    best_book = fair.loc[g["dec"].idxmax(), ["event_id", "market", "selection", "line_key", "book", "price"]]
    agg = agg.merge(best_book.rename(columns={"book": "best_book", "price": "best_price"}),
                    on=["event_id", "market", "selection", "line_key"])
    return agg.rename(columns={"line_key": "line"})


def compare(cons: pd.DataFrame, model_prob_fn) -> pd.DataFrame:
    """model_prob_fn(home, away, market, selection, line) -> model probability or None."""
    th = settings()["disagreement"]["thresholds"]
    w = settings()["disagreement"]["blend_model_weight"]
    rows = []
    for r in cons.itertuples(index=False):
        line = None if r.market == "h2h" else r.line
        p = model_prob_fn(r.home, r.away, r.market, r.selection, line)
        if p is None:
            continue
        price_prob = 1.0 / r.best_dec
        edge = p - price_prob
        rows.append({**r._asdict(), "line": line, "model_prob": p, "edge": edge, "price_prob": price_prob,
                     "fair_edge": p - r.book_prob,
                     "blended_prob": w * p + (1 - w) * r.book_prob,
                     "ev": p * r.best_dec - 1,
                     "flagged": bool(edge >= th.get(r.market, 0.04))})
    return pd.DataFrame(rows)
