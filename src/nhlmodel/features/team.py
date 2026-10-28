"""Point-in-time team and goalie ratings.

Every rating for a game on date D uses only games played before D (no look-ahead).

Team rates are per regulation game, counting only time with both goalies in net:
  sogf / soga : shots on goal for / against
  xgf / xga   : expected goals for / against (unblocked shots, from the xG model)
  gf / ga     : actual goals for / against
They are exponentially weighted toward recent games and shrunk toward the league average
(and, early in a season, toward last season's rating).

Goalie factor = shrunk (goals allowed / xG faced). 1.0 is average, < 1 is better than average.
  career = (GA + K) / (xGA + K)        (shrunk to 1.0)
  recent = (GA_r + K*career) / (xGA_r + K)   (recent EW sums, shrunk to career)
This is the PDF's w = N / (N + K) shrinkage, with N = xG faced.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from nhlmodel.config import settings

RATED = ["sogf", "soga", "xgf", "xga", "gf", "ga"]


# ----------------------------------------------------------------------------- team-game table

def team_games(games: pd.DataFrame, shots: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game with regulation counts (goalies in), plus rest flags."""
    s = shots[(shots.period <= 3) & ~shots.empty_net & ~shots.own_goalie_pulled]
    on_goal = s.event.isin(["sog", "goal"])
    agg = (s.assign(sog=on_goal.astype(int), g=s.is_goal.astype(int))
           .groupby(["game_id", "team_id"]).agg(sog=("sog", "sum"), xg=("xg", "sum"), goals=("g", "sum"))
           .reset_index())
    rows = []
    for side, opp in (("home", "away"), ("away", "home")):
        t = games[["game_id", "season", "game_type", "game_date", f"{side}_id", f"{opp}_id"]].rename(
            columns={f"{side}_id": "team_id", f"{opp}_id": "opp_id"})
        t["is_home"] = int(side == "home")
        rows.append(t)
    tg = pd.concat(rows, ignore_index=True)
    tg = tg.merge(agg.rename(columns={"sog": "sogf", "xg": "xgf", "goals": "gf"}), on=["game_id", "team_id"], how="left")
    tg = tg.merge(agg.rename(columns={"team_id": "opp_id", "sog": "soga", "xg": "xga", "goals": "ga"}),
                  on=["game_id", "opp_id"], how="left")
    tg[RATED] = tg[RATED].fillna(0.0)
    tg["game_date"] = pd.to_datetime(tg["game_date"]).dt.date
    tg = tg.sort_values(["team_id", "game_date", "game_id"]).reset_index(drop=True)
    tg = add_rest_flags(tg)
    return tg


def add_rest_flags(tg: pd.DataFrame) -> pd.DataFrame:
    """b2b: team also played yesterday. three_in_four: this is at least the 3rd game in 4 days."""
    tg = tg.sort_values(["team_id", "game_date"]).copy()
    d = pd.to_datetime(tg["game_date"])
    prev1 = d.groupby(tg["team_id"]).shift(1)
    prev2 = d.groupby(tg["team_id"]).shift(2)
    tg["b2b"] = ((d - prev1).dt.days == 1).astype(int)
    tg["three_in_four"] = ((d - prev2).dt.days <= 3).astype(int)
    return tg


# ----------------------------------------------------------------------------- team ratings

def team_ratings(tg: pd.DataFrame) -> pd.DataFrame:
    """Pre-game ratings for every team-game row (columns r_<metric>) plus league averages lg_<metric>.

    Also returns one extra 'next game' row per team (game_id = -1) holding the latest rating,
    used for projecting games that have not been played yet.
    """
    cfg = settings()["ratings"]
    half = cfg["ewm_halflife_games"]
    prior_n = cfg["prior_games"]
    carry = cfg["carryover"]
    decay = 0.5 ** (1.0 / half)

    tg = tg.sort_values(["game_date", "game_id"]).reset_index(drop=True)
    lg = league_averages(tg)
    tg = tg.merge(lg, on="game_date", how="left")

    out = []
    for team_id, g in tg.groupby("team_id", sort=False):
        g = g.sort_values(["game_date", "game_id"])
        num = np.zeros(len(RATED))
        den = 0.0
        season = None
        last_rating = None
        recs = []
        for row in g.itertuples(index=False):
            lgv = np.array([getattr(row, f"lg_{m}") for m in RATED])
            if row.season != season:
                # new season: restart sums, prior = blend of last season's rating and league
                prior = lgv if last_rating is None else carry * last_rating + (1 - carry) * lgv
                num = np.zeros(len(RATED))
                den = 0.0
                season = row.season
            rating = (num + prior_n * prior) / (den + prior_n)
            recs.append(rating)
            vals = np.array([getattr(row, m) for m in RATED])
            num = num * decay + vals
            den = den * decay + 1.0
            last_rating = (num + prior_n * prior) / (den + prior_n)
        r = pd.DataFrame(recs, columns=[f"r_{m}" for m in RATED], index=g.index)
        out.append(pd.concat([g, r], axis=1))
        nxt = {"team_id": team_id, "game_id": -1, "season": season,
               **{f"r_{m}": v for m, v in zip(RATED, last_rating, strict=True)}}
        out.append(pd.DataFrame([nxt]))
    return pd.concat(out, ignore_index=True)


def league_averages(tg: pd.DataFrame, window_days: int = 365) -> pd.DataFrame:
    """For each date D: average per-team-game values over team-games in [D - window, D)."""
    daily = tg.groupby("game_date")[RATED].sum()
    counts = tg.groupby("game_date").size()
    dates = sorted(set(tg["game_date"]))
    cs = daily.cumsum()
    cc = counts.cumsum()
    idx = pd.Index(daily.index)
    rows = []
    for d in dates:
        hi = idx.searchsorted(d, side="left") - 1
        lo = idx.searchsorted(d - timedelta(days=window_days), side="left") - 1
        if hi < 0:
            rows.append({"game_date": d, **{f"lg_{m}": np.nan for m in RATED}})
            continue
        s = cs.iloc[hi] - (cs.iloc[lo] if lo >= 0 else 0)
        n = cc.iloc[hi] - (cc.iloc[lo] if lo >= 0 else 0)
        rows.append({"game_date": d, **{f"lg_{m}": s[m] / n for m in RATED}})
    lg = pd.DataFrame(rows).bfill()  # first dates of history borrow the first available average
    return lg


def latest_league_average(tg: pd.DataFrame, asof: date, window_days: int = 365) -> dict[str, float]:
    m = (tg["game_date"] < asof) & (tg["game_date"] >= asof - timedelta(days=window_days))
    x = tg.loc[m, RATED].mean()
    return {f"lg_{k}": float(v) for k, v in x.items()}


# ----------------------------------------------------------------------------- goalie ratings

def goalie_games_xg(shots: pd.DataFrame) -> pd.DataFrame:
    """xG faced and goals allowed per goalie per game (goalie in net, no shootout)."""
    s = shots[shots.goalie_id.notna() & ~shots.empty_net & shots.event.isin(["sog", "goal", "miss"])]
    g = (s.assign(g=s.is_goal.astype(int))
         .groupby(["game_id", "goalie_id", "game_date"]).agg(xga=("xg", "sum"), ga=("g", "sum")).reset_index())
    g["goalie_id"] = g["goalie_id"].astype(int)
    g["game_date"] = pd.to_datetime(g["game_date"]).dt.date
    return g


def goalie_ratings(gx: pd.DataFrame, recent_halflife: float = 30.0) -> pd.DataFrame:
    """Pre-game goalie factor for each goalie-game, plus a latest row (game_id = -1) per goalie."""
    K = settings()["ratings"]["goalie_prior_xg"]
    decay = 0.5 ** (1.0 / recent_halflife)
    out = []
    for gid, g in gx.sort_values(["game_date", "game_id"]).groupby("goalie_id", sort=False):
        c_ga = c_xga = r_ga = r_xga = 0.0
        facs = []
        for row in g.itertuples(index=False):
            career = (c_ga + K) / (c_xga + K)
            facs.append(((r_ga + K * career) / (r_xga + K), c_xga))
            c_ga += row.ga
            c_xga += row.xga
            r_ga = r_ga * decay + row.ga
            r_xga = r_xga * decay + row.xga
        gg = g[["game_id", "goalie_id", "game_date"]].copy()
        gg["goalie_factor"] = [f for f, _ in facs]
        gg["career_xga"] = [n for _, n in facs]
        out.append(gg)
        career = (c_ga + K) / (c_xga + K)
        out.append(pd.DataFrame([{"game_id": -1, "goalie_id": gid, "game_date": None,
                                  "goalie_factor": (r_ga + K * career) / (r_xga + K), "career_xga": c_xga}]))
    return pd.concat(out, ignore_index=True)


def latest_goalie_factor(gr: pd.DataFrame, goalie_id: int | None) -> tuple[float, float]:
    """(factor, career xG faced) for a goalie's next game. Unknown goalie -> league average (1.0)."""
    if goalie_id is None:
        return 1.0, 0.0
    r = gr[(gr.goalie_id == goalie_id) & (gr.game_id == -1)]
    if r.empty:
        return 1.0, 0.0
    return float(r.goalie_factor.iloc[0]), float(r.career_xga.iloc[0])


# ----------------------------------------------------------------------------- goalie detail

DANGER_EDGES = (0.05, 0.15)  # xG per shot: low < 0.05 <= mid < 0.15 <= high


def goalie_detail(shots: pd.DataFrame, goalie_games: pd.DataFrame, k_xg: float = 40.0,
                  k_reb: float = 300.0, halflife: float = 40.0) -> pd.DataFrame:
    """Pre-game goalie detail per goalie-game (+ game_id -1 'next game' rows):

    g_high   : goals allowed / xG on high-danger shots (xG >= 0.15), shrunk to 1
    g_lowmid : same for shots below 0.15
    g_rebound: rebound shots allowed per save, relative to league, shrunk to 1
    starts7  : starts in the previous 7 days;  g_b2b: started the previous day too
    """
    s = shots[shots.goalie_id.notna() & ~shots.empty_net & shots.event.isin(["sog", "goal", "miss"])].copy()
    s["goalie_id"] = s.goalie_id.astype(int)
    s["high"] = s.xg >= DANGER_EDGES[1]
    s["g"] = s.is_goal.astype(int)
    s["save"] = (s.event == "sog").astype(int)
    s["reb"] = s.rebound.astype(int)
    per = s.groupby(["game_id", "goalie_id", "high"]).agg(xg=("xg", "sum"), ga=("g", "sum")).unstack("high").fillna(0)
    per.columns = [f"{c}_{'hi' if h else 'lo'}" for c, h in per.columns]
    per = per.join(s.groupby(["game_id", "goalie_id"]).agg(saves=("save", "sum"), reb=("reb", "sum"))).reset_index()
    for c in ("xg_hi", "xg_lo", "ga_hi", "ga_lo"):
        if c not in per:
            per[c] = 0.0
    dates = goalie_games[["game_id", "game_date"]].drop_duplicates("game_id")
    per = per.merge(dates, on="game_id")
    per["game_date"] = pd.to_datetime(per["game_date"]).dt.date
    lg_reb = per.reb.sum() / max(per.saves.sum(), 1)

    starts = goalie_games[goalie_games.starter][["player_id", "game_date"]].copy()
    starts["game_date"] = pd.to_datetime(starts.game_date).dt.date
    start_dates = starts.groupby("player_id")["game_date"].apply(lambda x: sorted(x)).to_dict()

    decay = 0.5 ** (1.0 / halflife)
    out = []
    for gid, g in per.sort_values(["game_date", "game_id"]).groupby("goalie_id", sort=False):
        acc = np.zeros(6)  # xg_hi, ga_hi, xg_lo, ga_lo, saves, reb
        recs = []
        sd = start_dates.get(gid, [])
        for row in g.itertuples(index=False):
            recs.append(_detail(acc, k_xg, k_reb, lg_reb) + _workload(sd, row.game_date))
            acc = acc * decay + np.array([row.xg_hi, row.ga_hi, row.xg_lo, row.ga_lo, row.saves, row.reb])
        cols = ["g_high", "g_lowmid", "g_rebound", "starts7", "g_b2b"]
        d = pd.DataFrame(recs, columns=cols)
        d.insert(0, "game_id", g.game_id.to_numpy())
        d.insert(1, "goalie_id", gid)
        out.append(d)
        out.append(pd.DataFrame([{"game_id": -1, "goalie_id": gid,
                                  **dict(zip(cols[:3], _detail(acc, k_xg, k_reb, lg_reb), strict=True)),
                                  "starts7": 0, "g_b2b": 0}]))
    return pd.concat(out, ignore_index=True)


def _detail(acc: np.ndarray, k_xg: float, k_reb: float, lg_reb: float) -> tuple[float, float, float]:
    xh, gh, xl, gl, sv, rb = acc
    return ((gh + k_xg) / (xh + k_xg), (gl + k_xg) / (xl + k_xg),
            ((rb + k_reb * lg_reb) / (sv + k_reb)) / lg_reb if lg_reb > 0 else 1.0)


def _workload(start_dates: list, d) -> tuple[int, int]:
    prior = [x for x in start_dates if x < d]
    s7 = sum(1 for x in prior if (d - x).days <= 7)
    b2b = int(bool(prior) and (d - prior[-1]).days == 1)
    return s7, b2b


def workload_for(goalie_games: pd.DataFrame, goalie_id: int | None, d) -> tuple[int, int]:
    """Live: (starts in last 7 days, started yesterday) for tonight's starter."""
    if goalie_id is None:
        return 0, 0
    g = goalie_games[(goalie_games.player_id == goalie_id) & goalie_games.starter]
    return _workload(sorted(pd.to_datetime(g.game_date).dt.date), d)
