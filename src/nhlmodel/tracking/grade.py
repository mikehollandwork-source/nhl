"""Grade logged disagreements once games are final.

For each logged row:
  outcome        1 win / 0 loss / None push
  model_logloss  vs  book_logloss (book = opening consensus de-vigged prob)
  close_prob     consensus de-vigged prob in the last snapshot before puck drop (same line)
  clv_prob       close_prob - book_prob (positive = market moved toward the selection)
  clv_price      best opening decimal x close_prob - 1 (did we beat the closing price?)
  units          flat 1-unit result at the best opening price (flagged picks only)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nhlmodel.market.devig import log_loss


def settle(market: str, selection: str, line: float | None, home_score: int, away_score: int) -> float | None:
    if market == "h2h":
        home_won = home_score > away_score
        return float(home_won if selection == "home" else not home_won)
    if market == "spreads":
        margin = (home_score - away_score) if selection == "home" else (away_score - home_score)
        adj = margin + float(line)
        return None if adj == 0 else float(adj > 0)
    if market == "totals":
        t = home_score + away_score
        if t == float(line):
            return None
        return float(t > line) if selection == "over" else float(t < line)
    raise ValueError(market)


def closing_consensus(cons_snapshots: pd.DataFrame) -> pd.DataFrame:
    """Latest consensus per (event, market, selection, line) taken before commence_time."""
    c = cons_snapshots.copy()
    c = c[pd.to_datetime(c["ts"], utc=True) < pd.to_datetime(c["commence_time"], utc=True)]
    c = c.sort_values("ts").groupby(["event_id", "market", "selection", "line"], dropna=False).tail(1)
    return c[["event_id", "market", "selection", "line", "book_prob", "ts"]].rename(
        columns={"book_prob": "close_prob", "ts": "close_ts"})


def grade(disagreements: pd.DataFrame, results: pd.DataFrame, cons_snapshots: pd.DataFrame) -> pd.DataFrame:
    """disagreements: logged rows (opening). results: game_date, home, away, home_score, away_score."""
    d = disagreements.merge(results, on=["game_date", "home", "away"], how="inner")
    if d.empty:
        return d
    d["line_m"] = d["line"].fillna(0.0)
    if not cons_snapshots.empty:
        close = closing_consensus(cons_snapshots).assign(line_m=lambda x: x["line"].fillna(0.0)).drop(columns="line")
        d = d.merge(close, on=["event_id", "market", "selection", "line_m"], how="left")
        # no later snapshot than the one we logged -> no closing-line information
        same = pd.to_datetime(d["close_ts"], utc=True) <= pd.to_datetime(d["ts"], utc=True)
        d.loc[same, "close_prob"] = np.nan
        d = d.drop(columns="close_ts")
    else:
        d["close_prob"] = np.nan
    d["outcome"] = [settle(r.market, r.selection, r.line, r.home_score, r.away_score) for r in d.itertuples()]
    d["outcome"] = d["outcome"].astype(float)
    has = d["outcome"].notna()
    for col, src in (("model_logloss", "model_prob"), ("book_logloss", "book_prob")):
        d[col] = [log_loss(p, int(o)) if pd.notna(o) else np.nan for p, o in zip(d[src], d.outcome, strict=True)]
    d["clv_prob"] = d["close_prob"] - d["book_prob"]
    d["clv_price"] = d["best_dec"] * d["close_prob"] - 1
    d["units"] = np.where(d.flagged & has, np.where(d.outcome == 1, d.best_dec - 1, -1.0),
                          np.where(d.flagged, 0.0, np.nan))
    return d.drop(columns=["line_m"])


def summarize(g: pd.DataFrame) -> dict:
    """Headline stats: whose probabilities were better, and how flagged picks did."""
    if g.empty:
        return {}
    graded = g[g.outcome.notna()]
    f = graded[graded.flagged]
    out = {
        "rows": int(len(graded)),
        "model_logloss": float(graded.model_logloss.mean()),
        "book_logloss": float(graded.book_logloss.mean()),
        "flagged": int(len(f)),
        "flagged_win_rate": float(f.outcome.mean()) if len(f) else None,
        "flagged_units": float(f.units.sum()) if len(f) else 0.0,
        "flagged_roi": float(f.units.sum() / len(f)) if len(f) else None,
        "flagged_model_logloss": float(f.model_logloss.mean()) if len(f) else None,
        "flagged_book_logloss": float(f.book_logloss.mean()) if len(f) else None,
        "flagged_clv_moved_our_way": float((f.clv_prob > 0).mean()) if f.clv_prob.notna().any() else None,
        "flagged_avg_clv_price": float(f.clv_price.mean()) if f.clv_price.notna().any() else None,
    }
    return out


def breakdown(g: pd.DataFrame, by: str) -> pd.DataFrame:
    graded = g[g.outcome.notna() & g.flagged]
    if graded.empty:
        return pd.DataFrame()
    return graded.groupby(by).agg(n=("outcome", "size"), win_rate=("outcome", "mean"),
                                  units=("units", "sum"), avg_edge=("edge", "mean"),
                                  clv_prob=("clv_prob", "mean")).reset_index()


def edge_bucket(edge: pd.Series) -> pd.Series:
    return pd.cut(edge, [-1, 0.02, 0.04, 0.06, 0.08, 1], labels=["<2%", "2-4%", "4-6%", "6-8%", "8%+"])


# ---- player props vs Kalshi ----

def grade_props(lines: pd.DataFrame, player_games: pd.DataFrame, flag_edge: float = 0.04) -> pd.DataFrame:
    """lines: posted Kalshi comparisons (player_id, game_id, market, line, model_prob, kalshi_prob, edge).
    A player who did not play is void (dropped), matching Kalshi's rules. Flagged pick = the side the model
    prefers when |edge| >= flag_edge, priced at the Kalshi mid."""
    if lines.empty or player_games.empty:
        return pd.DataFrame()
    x = lines.sort_values("ts").drop_duplicates(["game_id", "player_id", "market", "line"], keep="last")
    pg = player_games[["game_id", "player_id", "goals", "assists", "points"]]
    x = x.merge(pg, on=["game_id", "player_id"], how="inner")
    if x.empty:
        return x
    stat = np.select([x.market == "goals", x.market == "assists"], [x.goals, x.assists], x.points)
    x["outcome"] = (stat > x.line).astype(float)
    eps = 1e-6
    for col, name in (("model_prob", "model_ll"), ("kalshi_prob", "kalshi_ll")):
        p = x[col].clip(eps, 1 - eps)
        x[name] = -(x.outcome * np.log(p) + (1 - x.outcome) * np.log(1 - p))
    x["flagged"] = x.edge.abs() >= flag_edge
    over = x.edge > 0
    price = np.where(over, x.kalshi_prob, 1 - x.kalshi_prob)
    won = np.where(over, x.outcome == 1, x.outcome == 0)
    x["units"] = np.where(x.flagged, np.where(won, (1 - price) / price, -1.0), 0.0)
    x["won"] = won
    return x


def summarize_props(g: pd.DataFrame) -> dict | None:
    if g is None or g.empty:
        return None
    f = g[g.flagged]
    return {"n": len(g), "model_logloss": float(g.model_ll.mean()), "kalshi_logloss": float(g.kalshi_ll.mean()),
            "flagged": int(len(f)), "flagged_win_rate": float(f.won.mean()) if len(f) else None,
            "flagged_units": float(f.units.sum())}
