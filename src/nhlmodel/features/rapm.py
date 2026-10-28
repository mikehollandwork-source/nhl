"""Player impact ratings (regularized adjusted plus-minus, "RAPM") from 5-on-5 stints.

For every 5v5 stint (both goalies in) we know the 10 skaters on the ice and the shots / xG each
team produced. Each stint gives two rows:
    rate_for(team A) = intercept + home_adv*[A is home] + sum(offense of A's skaters)
                                                          + sum(defense of B's skaters)
weighted by stint length. Ridge regression shrinks every player toward 0 (league average), so
players with little ice time stay near 0. Older seasons get less weight.

Output per player: off_xg / def_xg (xG per 60 added / allowed) and off_sog / def_sog (shots per 60).
Positive def_* = the player's team ALLOWS more when he is on the ice (bad).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge

SK = 6


def stint_events(stints: pd.DataFrame, shots: pd.DataFrame, games: pd.DataFrame) -> pd.DataFrame:
    """5v5 stints with home/away shots on goal and xG assigned by time."""
    st = stints[(stints.n_home == 5) & (stints.n_away == 5) & stints.home_goalie_in & stints.away_goalie_in &
                (stints.secs > 0)].copy()
    st = st.merge(games[["game_id", "season", "home_id"]], on="game_id")
    st = st.sort_values(["t0"]).reset_index(drop=True)
    st["sid"] = np.arange(len(st))
    s = shots[shots.event.isin(["sog", "goal", "miss"]) & ~shots.empty_net & (shots.sk_for == 5) &
              (shots.sk_against == 5)][["game_id", "t", "is_home", "event", "xg"]].copy()
    s["sog"] = s.event.isin(["sog", "goal"]).astype(int)
    s = s.sort_values("t")
    m = pd.merge_asof(s, st[["game_id", "t0", "t1", "sid"]].sort_values("t0"), left_on="t", right_on="t0",
                      by="game_id", direction="backward", allow_exact_matches=False)
    m = m[m.sid.notna() & (m.t <= m.t1)]
    m["sid"] = m.sid.astype(int)
    agg = m.groupby(["sid", "is_home"]).agg(xg=("xg", "sum"), sog=("sog", "sum")).unstack("is_home").fillna(0)
    agg.columns = [f"{c}_{'home' if h else 'away'}" for c, h in agg.columns]
    st = st.join(agg, on="sid")
    for c in ("xg_home", "xg_away", "sog_home", "sog_away"):
        if c not in st:
            st[c] = 0.0
        st[c] = st[c].fillna(0.0)
    return st


def fit(st: pd.DataFrame, target: str = "xg", alpha: float = 2000.0, season_decay: float = 0.6,
        latest_season: int | None = None) -> pd.DataFrame:
    """Ridge RAPM for one target ('xg' or 'sog'). Returns player_id, off_<t>, def_<t>, toi_s."""
    if st.empty:
        return pd.DataFrame(columns=["player_id", f"off_{target}", f"def_{target}", "toi_s"])
    hcols = [f"h{k + 1}" for k in range(SK)]
    acols = [f"a{k + 1}" for k in range(SK)]
    players = np.unique(st[hcols + acols].to_numpy())
    players = players[players > 0]
    idx = {p: i for i, p in enumerate(players)}
    n, P = len(st), len(players)

    def onehots(cols):
        r, c = [], []
        arr = st[cols].to_numpy()
        for k in range(arr.shape[1]):
            v = arr[:, k]
            m = v > 0
            r.append(np.nonzero(m)[0])
            c.append(np.array([idx[x] for x in v[m]]))
        r, c = np.concatenate(r), np.concatenate(c)
        return sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(n, P))

    H, A = onehots(hcols), onehots(acols)
    # rows 0..n-1: home attacking; rows n..2n-1: away attacking. Columns: [offense P | defense P | home]
    X = sparse.vstack([sparse.hstack([H, A, np.ones((n, 1))]), sparse.hstack([A, H, np.zeros((n, 1))])]).tocsr()
    secs = st.secs.to_numpy().astype(float)
    y = np.r_[st[f"{target}_home"].to_numpy() / secs, st[f"{target}_away"].to_numpy() / secs] * 3600.0
    latest = latest_season or int(st.season.max())
    age = (latest // 10000 - st.season.to_numpy() // 10000).astype(float)
    w = np.r_[secs, secs] * np.r_[season_decay ** age, season_decay ** age]
    model = Ridge(alpha=alpha, fit_intercept=True, solver="sparse_cg", max_iter=2000, tol=1e-6)
    model.fit(X, y, sample_weight=w)
    coef = model.coef_
    toi = np.asarray((H + A).T @ secs).ravel()
    return pd.DataFrame({"player_id": players.astype(int), f"off_{target}": coef[:P], f"def_{target}": coef[P:2 * P],
                         "toi_s": toi})


def fit_all(st: pd.DataFrame, alpha_xg: float = 2000.0, alpha_sog: float = 2000.0,
            latest_season: int | None = None) -> pd.DataFrame:
    a = fit(st, "xg", alpha_xg, latest_season=latest_season)
    b = fit(st, "sog", alpha_sog, latest_season=latest_season)
    return a.merge(b.drop(columns="toi_s"), on="player_id", how="outer").fillna(0.0)


def choose_alpha(train: pd.DataFrame, valid: pd.DataFrame, target: str, grid=(250, 500, 1000, 2000, 4000, 8000)) -> float:
    """Pick the ridge strength that best predicts a later season's 5v5 stints (time-weighted MSE)."""
    best, best_a = np.inf, grid[0]
    for a in grid:
        r = fit(train, target, a)
        off = dict(zip(r.player_id, r[f"off_{target}"], strict=True))
        de = dict(zip(r.player_id, r[f"def_{target}"], strict=True))
        base = (train[f"{target}_home"].sum() + train[f"{target}_away"].sum()) / (2 * train.secs.sum()) * 3600
        err = 0.0
        for side, other in (("h", "a"), ("a", "h")):
            pred = base + sum(valid[f"{side}{k + 1}"].map(off).fillna(0) for k in range(5)) + \
                sum(valid[f"{other}{k + 1}"].map(de).fillna(0) for k in range(5))
            col = f"{target}_{'home' if side == 'h' else 'away'}"
            err += float((valid.secs * (valid[col] / valid.secs * 3600 - pred) ** 2).sum())
        if err < best:
            best, best_a = err, a
    return float(best_a)
