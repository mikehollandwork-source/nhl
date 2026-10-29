"""Lineup strength: how tonight's skaters compare with the team's usual lineup.

Three parts, all computed from games BEFORE the one being rated:
  1. 5v5 impact (features/rapm.py): 5 * sum_i rating_i * toi_i / sum_i toi_i for offense / defense,
     shots and xG. Players with no rating get REPLACEMENT level (the average of low-minute players),
     not league average, because call-ups are usually below average.
  2. Finishing: the lineup's shooting skill (goals vs expected goals, heavily shrunk), weighted by
     each player's share of the team's expected goals.
  3. Power play: the five skaters with the most projected PP time; their combined PP scoring rate
     (xG per 60 x finishing). Losing a PP1 shooter lowers it.
The model uses DELTAS vs the team's previous TYPICAL_GAMES games, since team ratings already
reflect the usual lineup.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import lfilter

IMPACTS = ("off_xg", "def_xg", "off_sog", "def_sog")
EXTRA = ("pp", "fin")
ALL = IMPACTS + EXTRA
DEFAULT_TOI = {"F": 12.5 * 60, "D": 17.0 * 60}   # 5v5 seconds; used only for players with no history
TYPICAL_GAMES = 10
DECAY = 0.5 ** (1 / 30)       # per-game decay for rate history
TOI_DECAY = 0.5 ** (1 / 8)    # per-game decay for ice-time projection
PP_PRIOR_MIN = 60.0
EV_PRIOR_MIN = 300.0
FIN_PRIOR_XG = 25.0


def replacement(rapm: pd.DataFrame, quantile: float = 0.3) -> dict[str, float]:
    """Average impact of low-minute players (bottom 30% of 5v5 ice time) = replacement level."""
    if rapm.empty:
        return {c: 0.0 for c in IMPACTS}
    low = rapm[rapm.toi_s <= rapm.toi_s.quantile(quantile)]
    return {c: float(low[c].mean()) for c in IMPACTS}


def _pre_sum(x: np.ndarray, decay: float) -> np.ndarray:
    """Decayed sum of PREVIOUS values: s_t = decay * s_{t-1} + x_{t-1}."""
    y = lfilter([1.0], [1.0, -decay], x)
    return np.r_[0.0, y[:-1]]


def player_form(player_games: pd.DataFrame, player_toi: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Pre-game form for every player-game: projected 5v5 seconds and PP minutes, shrunk EV and PP
    xG rates per 60, and finishing. Also returns a 'latest' set (including the last game) via
    latest_form()."""
    pg = player_games[["game_id", "player_id", "team_id", "game_date", "position"]].merge(
        player_toi[["game_id", "player_id", "toi_5v5_s", "toi_evo_s", "toi_pp_s"]], on=["game_id", "player_id"], how="left")
    ev = events.copy()
    for c in ("ev_ixg", "pp_ixg", "sh_ixg", "ev_goals", "pp_goals", "sh_goals"):
        if c not in ev:
            ev[c] = 0.0
    pg = pg.merge(ev[["game_id", "player_id", "ev_ixg", "pp_ixg", "sh_ixg", "ev_goals", "pp_goals", "sh_goals"]],
                  on=["game_id", "player_id"], how="left").fillna(0.0)
    pg = pg.sort_values(["player_id", "game_date", "game_id"]).reset_index(drop=True)
    pg["ev_min"] = (pg.toi_5v5_s + pg.toi_evo_s) / 60
    pg["pp_min"] = pg.toi_pp_s / 60
    pg["ixg"] = pg.ev_ixg + pg.pp_ixg + pg.sh_ixg
    pg["goals"] = pg.ev_goals + pg.pp_goals + pg.sh_goals
    lg_ev = pg.ev_ixg.sum() / max(pg.ev_min.sum(), 1)
    lg_pp = pg.pp_ixg.sum() / max(pg.pp_min.sum(), 1)

    cols = {k: np.zeros(len(pg)) for k in ("s_toi", "s_toiw", "s_ppm", "s_ppw", "s_evx", "s_evm", "s_ppx", "s_ppmin",
                                           "s_g", "s_x")}
    for _, idx in pg.groupby("player_id").indices.items():
        g = pg.iloc[idx]
        ones = np.ones(len(idx))
        cols["s_toi"][idx] = _pre_sum(g.toi_5v5_s.to_numpy(), TOI_DECAY)
        cols["s_toiw"][idx] = _pre_sum(ones, TOI_DECAY)
        cols["s_ppm"][idx] = _pre_sum(g.pp_min.to_numpy(), TOI_DECAY)
        cols["s_ppw"][idx] = _pre_sum(ones, TOI_DECAY)
        cols["s_evx"][idx] = _pre_sum(g.ev_ixg.to_numpy(), DECAY)
        cols["s_evm"][idx] = _pre_sum(g.ev_min.to_numpy(), DECAY)
        cols["s_ppx"][idx] = _pre_sum(g.pp_ixg.to_numpy(), DECAY)
        cols["s_ppmin"][idx] = _pre_sum(g.pp_min.to_numpy(), DECAY)
        cols["s_g"][idx] = _pre_sum(g.goals.to_numpy(), DECAY)
        cols["s_x"][idx] = _pre_sum(g.ixg.to_numpy(), DECAY)
    return _finish(pg, cols, lg_ev, lg_pp)


def _finish(pg: pd.DataFrame, s: dict, lg_ev: float, lg_pp: float) -> pd.DataFrame:
    pos = np.where(pg.position == "D", "D", "F")
    default = pd.Series(pos, index=pg.index).map(DEFAULT_TOI).to_numpy()
    out = pg[["game_id", "player_id", "team_id", "game_date", "position"]].copy()
    with np.errstate(invalid="ignore", divide="ignore"):
        out["proj_toi"] = np.where(s["s_toiw"] > 0, s["s_toi"] / s["s_toiw"], default)
        out["proj_pp_min"] = np.where(s["s_ppw"] > 0, s["s_ppm"] / s["s_ppw"], 0.0)
    out["ev_xg60"] = (s["s_evx"] + lg_ev * EV_PRIOR_MIN) / (s["s_evm"] + EV_PRIOR_MIN) * 60
    out["pp_xg60"] = (s["s_ppx"] + lg_pp * PP_PRIOR_MIN) / (s["s_ppmin"] + PP_PRIOR_MIN) * 60
    out["finishing"] = (s["s_g"] + FIN_PRIOR_XG) / (s["s_x"] + FIN_PRIOR_XG)
    out.attrs["lg_ev60"] = lg_ev * 60
    out.attrs["lg_pp60"] = lg_pp * 60
    return out


def latest_form(player_games: pd.DataFrame, player_toi: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    """Form as of today (includes each player's most recent game): append a dummy future game per
    player and read its pre-game values."""
    last = player_games.sort_values("game_date").groupby("player_id").tail(1).copy()
    last["game_id"] = -last.player_id
    last["game_date"] = "2999-01-01"
    f = player_form(pd.concat([player_games, last], ignore_index=True), player_toi, events)
    return f[f.game_id < 0].drop(columns=["game_id", "game_date"]).reset_index(drop=True)


def impact_of(players: pd.DataFrame, rapm: pd.DataFrame, repl: dict | None = None) -> dict[str, float]:
    """players: player_id, proj_toi (+ optional p_play, proj_pp_min, pp_xg60, ev_xg60, finishing)."""
    repl = repl or {c: 0.0 for c in IMPACTS}
    p = players.merge(rapm[["player_id", *IMPACTS]], on="player_id", how="left")
    for c in IMPACTS:
        p[c] = p[c].fillna(repl[c])
    pw = p.get("p_play", pd.Series(1.0, index=p.index)).to_numpy()
    w = p["proj_toi"].to_numpy() * pw
    out = {c: (float(5 * (p[c].to_numpy() * w).sum() / w.sum()) if w.sum() > 0 else 0.0) for c in IMPACTS}
    out.update(_extra(p, pw))
    return out


def _extra(p: pd.DataFrame, pw: np.ndarray) -> dict[str, float]:
    if "pp_xg60" not in p:
        return {"pp": 0.0, "fin": 1.0}
    fin = p.finishing.fillna(1.0).to_numpy()
    wf = p.ev_xg60.fillna(p.ev_xg60.mean()).to_numpy() * p.proj_toi.to_numpy() * pw
    fin_idx = float((wf * fin).sum() / wf.sum()) if wf.sum() > 0 else 1.0
    ppm = p.proj_pp_min.fillna(0).to_numpy() * pw
    top = np.argsort(-ppm)[:5]
    pp_idx = float(np.sum(p.pp_xg60.fillna(0).to_numpy()[top] * fin[top] * np.minimum(pw[top], 1.0)) / 5)
    return {"pp": pp_idx, "fin": fin_idx}


def lineup_history(form: pd.DataFrame, rapm_by_season: dict[int, pd.DataFrame], seasons: dict[int, int],
                   use_replacement: bool = True) -> pd.DataFrame:
    """Per team-game lineup indices (ratings fit on PRIOR seasons only) and deltas vs the team's
    previous TYPICAL_GAMES games. seasons: game_id -> season."""
    form = form.assign(season=form.game_id.map(seasons))
    recs = []
    for season, fs in form.groupby("season"):
        r = rapm_by_season.get(int(season))
        if r is None:
            continue
        repl = replacement(r) if use_replacement else {c: 0.0 for c in IMPACTS}
        m = fs.merge(r[["player_id", *IMPACTS]], on="player_id", how="left")
        for c in IMPACTS:
            m[c] = m[c].fillna(repl[c])
        w = m.proj_toi.to_numpy()
        for c in IMPACTS:
            m[f"w_{c}"] = m[c] * w
        m["wf"] = m.ev_xg60 * m.proj_toi
        m["wf_fin"] = m.wf * m.finishing
        m["pp_val"] = m.pp_xg60 * m.finishing
        m["pp_rank"] = m.groupby(["game_id", "team_id"]).proj_pp_min.rank(ascending=False, method="first")
        m["pp_top"] = np.where(m.pp_rank <= 5, m.pp_val, 0.0)
        agg = m.groupby(["game_id", "team_id", "game_date"]).agg(
            wsum=("proj_toi", "sum"), wf=("wf", "sum"), wf_fin=("wf_fin", "sum"), pp_top=("pp_top", "sum"),
            **{f"s_{c}": (f"w_{c}", "sum") for c in IMPACTS}).reset_index()
        for c in IMPACTS:
            agg[f"lu_{c}"] = 5 * agg[f"s_{c}"] / agg.wsum
        agg["lu_fin"] = agg.wf_fin / agg.wf
        agg["lu_pp"] = agg.pp_top / 5
        recs.append(agg[["game_id", "team_id", "game_date"] + [f"lu_{c}" for c in ALL]])
    if not recs:
        return pd.DataFrame()
    lu = pd.concat(recs).sort_values(["team_id", "game_date", "game_id"])
    for c in ALL:
        typical = lu.groupby("team_id")[f"lu_{c}"].transform(
            lambda x: x.shift().rolling(TYPICAL_GAMES, min_periods=3).mean())
        lu[f"d_{c}"] = (lu[f"lu_{c}"] - typical).fillna(0.0)
    return lu


def typical_impact(lu_hist: pd.DataFrame, team_id: int) -> dict[str, float]:
    t = lu_hist[lu_hist.team_id == team_id].sort_values("game_date").tail(TYPICAL_GAMES)
    return {c: float(t[f"lu_{c}"].mean()) if len(t) and f"lu_{c}" in t else 0.0 for c in ALL}
