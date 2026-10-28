"""Team "macro" stats from play-by-play: faceoffs, zone starts, giveaways, takeaways, hits.

These match the official NHL stats API's team pages (summary / realtime / faceoffpercentages /
percentages), but are built game by game from play-by-play so every rating uses only earlier games.
Ratings are recency-weighted and shrunk toward the league rate; priors are in faceoffs or minutes.
"""

from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from nhlmodel.config import settings

# (rating, numerator column, denominator column, prior denominator)
SPECS = [
    ("fo_pct", "fo_wins", "fo_total", 400.0),
    ("fo_oz_pct", "fo_oz_wins", "fo_oz", 200.0),
    ("fo_dz_pct", "fo_dz_wins", "fo_dz", 200.0),
    ("fo_pp_pct", "fo_pp_wins", "fo_pp", 80.0),
    ("oz_start_pct", "oz_starts_5v5", "zs_total", 300.0),
    ("giveaways60", "giveaways", "secs", 10 * 3600.0),
    ("takeaways60", "takeaways", "secs", 10 * 3600.0),
    ("hits60", "hits", "secs", 10 * 3600.0),
]


def prepare(te: pd.DataFrame) -> pd.DataFrame:
    t = te.copy()
    t["zs_total"] = t.oz_starts_5v5 + t.dz_starts_5v5
    t["secs"] = 3600.0
    t["game_date"] = pd.to_datetime(t.game_date).dt.date
    return t.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)


def macro_ratings(te: pd.DataFrame, window_days: int = 365) -> pd.DataFrame:
    """Pre-game ratings (m_<name>) and league rates (lgm_<name>) per team-game, plus game_id -1
    'next game' rows per team."""
    t = prepare(te)
    cfg = settings()["ratings"]
    decay = 0.5 ** (1.0 / cfg["ewm_halflife_games"])
    carry = cfg["carryover"]
    names = [s[0] for s in SPECS]
    P = np.array([s[3] for s in SPECS])
    # league rate per date from earlier games only (fallback for the very first dates of the data)
    cols = list(dict.fromkeys(c for s in SPECS for c in (s[1], s[2])))   # unique, order kept
    daily = t.groupby("game_date")[cols].sum()
    dates = list(daily.index)
    cs = daily.cumsum()
    fallback = {s[0]: t[s[1]].sum() / max(t[s[2]].sum(), 1) for s in SPECS}
    lg = {}
    for i, d in enumerate(dates):
        lo = np.searchsorted(dates, d - timedelta(days=window_days)) - 1
        if i == 0:
            lg[d] = np.array([fallback[n] for n in names])
            continue
        tot = cs.iloc[i - 1] - (cs.iloc[lo] if lo >= 0 else 0)
        lg[d] = np.array([tot[s[1]] / tot[s[2]] if tot[s[2]] > 0 else fallback[s[0]] for s in SPECS])
    out = []
    for team, g in t.groupby("team_id", sort=False):
        num = np.zeros(len(SPECS))
        den = np.zeros(len(SPECS))
        season, last, carried = None, None, None
        recs = []
        for row in g.itertuples(index=False):
            lgv = lg[row.game_date]
            if row.season != season:
                carried = last
                num[:] = 0
                den[:] = 0
                season = row.season
            prior = lgv if carried is None else carry * carried + (1 - carry) * lgv
            recs.append(np.r_[(num + P * prior) / (den + P), lgv])
            num = num * decay + np.array([getattr(row, s[1]) for s in SPECS], float)
            den = den * decay + np.array([getattr(row, s[2]) for s in SPECS], float)
            last = (num + P * prior) / (den + P)
        cols = [f"m_{n}" for n in names] + [f"lgm_{n}" for n in names]
        out.append(pd.concat([g[["game_id", "team_id"]].reset_index(drop=True), pd.DataFrame(recs, columns=cols)], axis=1))
        out.append(pd.DataFrame([{"game_id": -1, "team_id": team, **{f"m_{n}": v for n, v in zip(names, last, strict=True)},
                                  **{f"lgm_{n}": v for n, v in zip(names, lg[dates[-1]], strict=True)}}]))
    return pd.concat(out, ignore_index=True)


def attach(rows: pd.DataFrame, ratings: pd.DataFrame) -> pd.DataFrame:
    """Add team (m_*) and opponent (om_*) macro ratings to assembled team-game rows."""
    r = ratings[ratings.game_id > 0]
    mcols = [c for c in r.columns if c.startswith(("m_", "lgm_"))]
    rows = rows.merge(r[["game_id", "team_id", *mcols]], on=["game_id", "team_id"], how="left")
    opp = r[["game_id", "team_id"] + [c for c in r.columns if c.startswith("m_")]].rename(
        columns={"team_id": "opp_id", **{c: "o" + c for c in r.columns if c.startswith("m_")}})
    return rows.merge(opp, on=["game_id", "opp_id"], how="left")
