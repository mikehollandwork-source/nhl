"""Team rates split by strength state, measured per second actually spent in each state.

From stints (who is on the ice) and shots (situation at the shot), per team-game in regulation
with both goalies in:
  ev_*  : equal strength (5v5, 4v4, 3v3)
  pp_*  : team has more skaters (power play)
  sh_*  : team has fewer skaters (shorthanded)
  pk_starts : times the team went shorthanded (= penalties that actually produced a power play)

Ratings are exposure-weighted, recency-weighted and shrunk toward the league rate:
  rate = (sum_w counts + P * league_rate) / (sum_w exposure + P)
with P a prior exposure in seconds. Every rating for a game uses only earlier games.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from nhlmodel.config import settings

STATES = ("ev", "pp", "sh")
COUNTS = ("sogf", "xgf", "gf", "soga", "xga", "ga")
# (rating name, count column, exposure column, prior exposure seconds)
RATES = [
    ("ev_sogf", "ev_sogf", "ev_secs", 6 * 3000), ("ev_soga", "ev_soga", "ev_secs", 6 * 3000),
    ("ev_xgf", "ev_xgf", "ev_secs", 10 * 3000), ("ev_xga", "ev_xga", "ev_secs", 10 * 3000),
    ("ev_gf", "ev_gf", "ev_secs", 30 * 3000), ("ev_ga", "ev_ga", "ev_secs", 30 * 3000),
    ("pp_sogf", "pp_sogf", "pp_secs", 20 * 400), ("pp_xgf", "pp_xgf", "pp_secs", 25 * 400),
    ("pp_gf", "pp_gf", "pp_secs", 60 * 400),
    ("sh_soga", "sh_soga", "sh_secs", 20 * 400), ("sh_xga", "sh_xga", "sh_secs", 25 * 400),
    ("sh_ga", "sh_ga", "sh_secs", 60 * 400),
    ("sh_sogf", "sh_sogf", "sh_secs", 60 * 400), ("sh_xgf", "sh_xgf", "sh_secs", 80 * 400),
    ("pen_taken", "pk_starts", "reg_secs", 15 * 3600), ("pen_drawn", "pp_starts", "reg_secs", 15 * 3600),
]


def _state(n_own: pd.Series, n_opp: pd.Series) -> np.ndarray:
    return np.where(n_own == n_opp, "ev", np.where(n_own > n_opp, "pp", "sh"))


def team_strength_games(games: pd.DataFrame, shots: pd.DataFrame, stints: pd.DataFrame) -> pd.DataFrame:
    """One row per team per game: exposure seconds and counts by strength state, plus PP/PK starts."""
    g = games[["game_id", "season", "game_date", "home_id", "away_id"]]
    st = stints[(stints.period <= 3) & stints.home_goalie_in & stints.away_goalie_in].merge(g, on="game_id")
    st = st.sort_values(["game_id", "t0"])
    rows = []
    for side, opp_side in (("home", "away"), ("away", "home")):
        n_own, n_opp = st[f"n_{side}"], st[f"n_{opp_side}"]
        s = _state(n_own, n_opp)
        d = pd.DataFrame({"game_id": st.game_id.to_numpy(), "team_id": st[f"{side}_id"].to_numpy(),
                          "state": s, "secs": st.secs.to_numpy()})
        prev = pd.Series(s).groupby(st.game_id.to_numpy()).shift().to_numpy()
        d["pk_start"] = (s == "sh") & (prev != "sh")
        d["pp_start"] = (s == "pp") & (prev != "pp")
        rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    expo = d.pivot_table(index=["game_id", "team_id"], columns="state", values="secs", aggfunc="sum", fill_value=0)
    expo = expo.reindex(columns=list(STATES), fill_value=0).add_suffix("_secs")
    starts = d.groupby(["game_id", "team_id"])[["pk_start", "pp_start"]].sum().rename(
        columns={"pk_start": "pk_starts", "pp_start": "pp_starts"})
    tg = expo.join(starts).reset_index()
    tg["reg_secs"] = tg[["ev_secs", "pp_secs", "sh_secs"]].sum(axis=1)

    s = shots[(shots.period <= 3) & ~shots.empty_net & ~shots.own_goalie_pulled &
              shots.event.isin(["sog", "goal", "miss"])].copy()
    s["state"] = _state(s.sk_for, s.sk_against)
    s["sog"] = s.event.isin(["sog", "goal"]).astype(int)
    s["g"] = s.is_goal.astype(int)
    f = s.groupby(["game_id", "team_id", "state"]).agg(sogf=("sog", "sum"), xgf=("xg", "sum"), gf=("g", "sum"))
    f = f.unstack("state").fillna(0)
    f.columns = [f"{st_}_{c}" for c, st_ in f.columns]
    f = f.reset_index()
    tg = tg.merge(f, on=["game_id", "team_id"], how="left")
    gm = g.melt(id_vars=["game_id", "season", "game_date"], value_vars=["home_id", "away_id"],
                var_name="side", value_name="team_id")
    gm["opp_id"] = np.where(gm.side == "home_id", g.set_index("game_id").loc[gm.game_id, "away_id"].to_numpy(),
                            g.set_index("game_id").loc[gm.game_id, "home_id"].to_numpy())
    gm["is_home"] = (gm.side == "home_id").astype(int)
    tg = gm.drop(columns="side").merge(tg, on=["game_id", "team_id"], how="left")
    # "against" counts are the opponent's "for" counts in the mirrored state
    opp = tg[["game_id", "team_id"] + [c for c in tg.columns if c.endswith(("_sogf", "_xgf", "_gf"))]].copy()
    mirror = {"ev": "ev", "pp": "sh", "sh": "pp"}
    ren = {}
    for st_ in STATES:
        for c in ("sogf", "xgf", "gf"):
            ren[f"{st_}_{c}"] = f"{mirror[st_]}_{c[:-1]}a"
    opp = opp.rename(columns={"team_id": "opp_id", **ren})
    tg = tg.merge(opp, on=["game_id", "opp_id"], how="left")
    num = [c for c in tg.columns if c not in ("game_id", "season", "game_date", "team_id", "opp_id", "is_home")]
    tg[num] = tg[num].fillna(0.0)
    tg["game_date"] = pd.to_datetime(tg["game_date"]).dt.date
    # shift charts are occasionally missing or broken: keep only games whose reconstructed regulation
    # time is ~60 minutes with a plausible amount of special-teams time
    tg["valid"] = tg.reg_secs.between(3300, 3700) & (tg.pp_secs <= 1200) & (tg.sh_secs <= 1200)
    tg["valid"] = tg.groupby("game_id").valid.transform("all")
    return tg.sort_values(["game_date", "game_id", "team_id"]).reset_index(drop=True)


def league_rates(tg: pd.DataFrame, asof: date, window_days: int = 365) -> dict[str, float]:
    m = (tg.game_date < asof) & (tg.game_date >= asof - timedelta(days=window_days)) & tg.valid
    x = tg[m]
    return {name: float(x[c].sum() / max(x[e].sum(), 1.0)) for name, c, e, _ in RATES}


def strength_ratings(tg: pd.DataFrame, window_days: int = 365) -> pd.DataFrame:
    """Pre-game strength ratings (per second) for every team-game, plus one 'next game' row per team
    (game_id = -1). Columns: s_<rating> and lgs_<rating> (league rate at that date)."""
    cfg = settings()["ratings"]
    decay = 0.5 ** (1.0 / cfg["ewm_halflife_games"])
    carry = cfg["carryover"]
    dates = sorted(set(tg.game_date))
    names = [r[0] for r in RATES]
    valid_rows = tg[tg.valid]
    # fallback for the first dates of the data, which have no earlier games (2021 training rows only)
    fallback = {name: float(valid_rows[c].sum() / max(valid_rows[e].sum(), 1.0)) for name, c, e, _ in RATES}
    lg_by_date = {}
    for d in dates:
        lg = league_rates(tg, d, window_days)
        lg_by_date[d] = {k: (val if val > 0 else fallback[k]) for k, val in lg.items()}
    P = np.array([r[3] for r in RATES], float)
    out = []
    for team_id, g in tg.groupby("team_id", sort=False):
        g = g.sort_values(["game_date", "game_id"])
        num = np.zeros(len(RATES))
        den = np.zeros(len(RATES))
        season = None
        last = None          # latest rating
        carried = None       # rating at the end of the previous season
        recs = []
        for row in g.itertuples(index=False):
            lgv = np.array([lg_by_date[row.game_date][n] for n in names])
            if row.season != season:
                carried = last
                num[:] = 0
                den[:] = 0
                season = row.season
            prior = lgv if carried is None else carry * carried + (1 - carry) * lgv
            rate = (num + P * prior) / (den + P)
            recs.append(np.r_[rate, lgv])
            w = float(row.valid)
            c = np.array([getattr(row, r[1]) for r in RATES], float) * w
            e = np.array([getattr(row, r[2]) for r in RATES], float) * w
            num = num * decay + c
            den = den * decay + e
            last = (num + P * prior) / (den + P)
        cols = [f"s_{n}" for n in names] + [f"lgs_{n}" for n in names]
        out.append(pd.concat([g[["game_id", "team_id", "season", "game_date"]].reset_index(drop=True),
                              pd.DataFrame(recs, columns=cols)], axis=1))
        nxt = {"game_id": -1, "team_id": team_id, "season": season, "game_date": None,
               **{f"s_{n}": v for n, v in zip(names, last, strict=True)}}
        out.append(pd.DataFrame([nxt]))
    return pd.concat(out, ignore_index=True)
