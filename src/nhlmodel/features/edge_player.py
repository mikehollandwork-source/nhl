"""Per-player NHL EDGE stats (skating speed, bursts, offensive-zone time) rolled up to tonight's lineup.

EDGE only publishes season totals, so a game in season S uses each player's season S-1 values
(z-scored within that season). Players with no prior-season row get a replacement-level value
(30th percentile), the same rule the RAPM lineup ratings use. Each team-game gets the ice-time
weighted lineup average and its difference from the team's typical lineup (previous 10 games):
the model only sees the difference, i.e. what tonight's lineup changes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from nhlmodel.features.lineup import TYPICAL_GAMES

# feature name -> edge_skater column
COLS = {"e_speed": "speed_max", "e_bursts": "bursts20_pg", "e_oz": "oz_ev_pct"}
REPL_Q = 0.3


def prior_season_z(edge_skater: pd.DataFrame) -> pd.DataFrame:
    """player_id, season (the season the values are USED in), e_* z-scores."""
    if edge_skater is None or edge_skater.empty:
        return pd.DataFrame(columns=["player_id", "season", *COLS])
    e = edge_skater.copy()
    for name, src in COLS.items():
        grp = e.groupby("season")[src]
        e[name] = (e[src] - grp.transform("mean")) / grp.transform("std")
    e["season"] = e.season + 10001   # 20242025 -> 20252026
    return e[["player_id", "season", *COLS]]


def replacement(z: pd.DataFrame) -> pd.DataFrame:
    """Replacement-level value per season (season, e_*)."""
    return z.groupby("season")[list(COLS)].quantile(REPL_Q).reset_index()


def lineup_values(players: pd.DataFrame, z: pd.DataFrame, season: int) -> dict[str, float]:
    """players: player_id, proj_toi (+ optional p_play). Lineup e_* for one game in `season`."""
    zs = z[z.season == season]
    rep = zs[list(COLS)].quantile(REPL_Q) if len(zs) else pd.Series(0.0, index=list(COLS))
    p = players.merge(zs[["player_id", *COLS]], on="player_id", how="left")
    w = p.proj_toi.to_numpy() * p.get("p_play", pd.Series(1.0, index=p.index)).to_numpy()
    out = {}
    for c in COLS:
        v = p[c].fillna(rep[c]).to_numpy()
        out[c] = float((v * w).sum() / w.sum()) if w.sum() > 0 else 0.0
    return out


def lineup_history(form: pd.DataFrame, z: pd.DataFrame, seasons: dict[int, int]) -> pd.DataFrame:
    """Per team-game lineup e_* and d_e_* (difference from the previous TYPICAL_GAMES games).
    form: game_id, team_id, player_id, game_date, proj_toi. seasons: game_id -> season."""
    if z.empty or form.empty:
        return pd.DataFrame()
    f = form[["game_id", "team_id", "player_id", "game_date", "proj_toi"]].copy()
    f["season"] = f.game_id.map(seasons)
    f = f.merge(z, on=["player_id", "season"], how="left")
    rep = replacement(z).set_index("season")
    for c in COLS:
        f[c] = f[c].fillna(f.season.map(rep[c])).fillna(0.0)
        f[f"w_{c}"] = f[c] * f.proj_toi
    agg = f.groupby(["game_id", "team_id", "game_date", "season"]).agg(
        wsum=("proj_toi", "sum"), **{f"s_{c}": (f"w_{c}", "sum") for c in COLS}).reset_index()
    for c in COLS:
        agg[c] = np.where(agg.wsum > 0, agg[f"s_{c}"] / agg.wsum, 0.0)
    lu = agg.sort_values(["team_id", "game_date", "game_id"])
    for c in COLS:
        # typical = this season's earlier games only (z-scores change source at the season boundary)
        typical = lu.groupby(["team_id", "season"])[c].transform(lambda x: x.shift().rolling(TYPICAL_GAMES, min_periods=3).mean())
        lu[f"d_{c}"] = (lu[c] - typical).fillna(0.0)
    return lu[["game_id", "team_id", "game_date", *COLS, *[f"d_{c}" for c in COLS]]]


def typical(lu_hist: pd.DataFrame, team_id: int) -> dict[str, float]:
    t = lu_hist[lu_hist.team_id == team_id].sort_values("game_date").tail(TYPICAL_GAMES)
    return {c: float(t[c].mean()) if len(t) else 0.0 for c in COLS}
