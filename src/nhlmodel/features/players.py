"""Player rates for props, shrunk toward position averages.

Change from the PDF (section 8): the PDF uses raw last-20-game shares, which are very noisy.
Here each per-minute rate is
    (recent-weighted events + prior_rate * P) / (recent-weighted minutes + P)
with prior_rate = the position's league rate and P = `player_prior_toi_min` minutes.
Finishing (goals above xG) is shrunk much harder because it is mostly noise.
"""

from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

from nhlmodel.config import settings

STATS = ["sog", "goals", "assists", "blocks", "ppp", "ixg"]
FINISHING_PRIOR_XG = 25.0      # xG worth of shrinkage on finishing skill (goals / xG)
GAME_HALFLIFE = 30.0           # games
TOI_HALFLIFE = 5.0             # games, for projected ice time


def pos_group(pos: str | None) -> str:
    return "D" if pos == "D" else "F"


def player_ixg(shots: pd.DataFrame) -> pd.DataFrame:
    s = shots[shots.shooter_id.notna() & shots.event.isin(["sog", "goal", "miss"])]
    out = s.groupby(["game_id", "shooter_id"])["xg"].sum().reset_index()
    out.columns = ["game_id", "player_id", "ixg"]
    out["player_id"] = out["player_id"].astype(int)
    return out


def league_assist_rates(shots: pd.DataFrame) -> dict[str, float]:
    g = shots[shots.is_goal & (shots.period <= 4)]
    a1 = g.assist1_id.notna().mean()
    a2_given_a1 = g.loc[g.assist1_id.notna(), "assist2_id"].notna().mean()
    pp = (g.sk_for > g.sk_against) & ~g.empty_net
    pp_pts_per_goal = 1 + g.loc[pp, "assist1_id"].notna().mean() + g.loc[pp, "assist2_id"].notna().mean()
    return {"p_a1": float(a1), "p_a2_given_a1": float(a2_given_a1),
            "pp_goal_frac": float(pp.sum() / max(len(g), 1)), "pp_pts_per_goal": float(pp_pts_per_goal)}


def player_rates(pg: pd.DataFrame, ixg: pd.DataFrame, asof: date, player_ids: list[int]) -> pd.DataFrame:
    """Shrunk per-minute rates and projected TOI for the given players, using games before `asof`."""
    P = settings()["ratings"]["player_prior_toi_min"]
    hist = pg[pd.to_datetime(pg.game_date).dt.date < asof].merge(ixg, on=["game_id", "player_id"], how="left")
    hist["ixg"] = hist["ixg"].fillna(0.0)
    hist["grp"] = hist["position"].map(pos_group)
    recent = hist[pd.to_datetime(hist.game_date).dt.date >= asof - timedelta(days=365)]
    prior = {g: {s: d[s].sum() / max(d["toi"].sum(), 1) for s in STATS} for g, d in recent.groupby("grp")}

    rows = []
    for pid in player_ids:
        h = hist[hist.player_id == pid].sort_values("game_date")
        grp = pos_group(h["position"].iloc[-1]) if len(h) else "F"
        pr = prior.get(grp, {s: 0.0 for s in STATS})
        if h.empty:
            rec = {"player_id": pid, "grp": grp, "toi_proj": 12.0 if grp == "F" else 16.0, "n_games": 0,
                   **{f"rate_{s}": pr[s] for s in STATS}, "finishing": 1.0}
            rows.append(rec)
            continue
        k = np.arange(len(h))[::-1]
        w = 0.5 ** (k / GAME_HALFLIFE)
        toi = h["toi"].to_numpy()
        wt = (w * toi).sum()
        rec = {"player_id": pid, "grp": grp, "n_games": int(len(h))}
        for s in STATS:
            rec[f"rate_{s}"] = ((w * h[s].to_numpy()).sum() + pr[s] * P) / (wt + P)
        g_sum = (w * h["goals"].to_numpy()).sum()
        x_sum = (w * h["ixg"].to_numpy()).sum()
        rec["finishing"] = (g_sum + FINISHING_PRIOR_XG) / (x_sum + FINISHING_PRIOR_XG)
        wt_toi = 0.5 ** (k / TOI_HALFLIFE)
        last = h.tail(10)
        wl = wt_toi[-len(last):]
        rec["toi_proj"] = float((wl * last["toi"].to_numpy()).sum() / wl.sum())
        rows.append(rec)
    df = pd.DataFrame(rows)
    df["w_goal"] = df["rate_ixg"] * df["finishing"] * df["toi_proj"]
    df["w_sog"] = df["rate_sog"] * df["toi_proj"]
    df["w_ast"] = df["rate_assists"] * df["toi_proj"]
    df["w_blk"] = df["rate_blocks"] * df["toi_proj"]
    df["w_ppp"] = df["rate_ppp"] * df["toi_proj"]
    return df


def team_block_ratio(pg: pd.DataFrame, games: pd.DataFrame, team_id: int, asof: date, prior_n: float = 10) -> float:
    """Team blocked shots per opponent shot on goal, recent-weighted and shrunk to league."""
    d = pd.to_datetime(pg.game_date).dt.date
    recent = pg[(d < asof) & (d >= asof - timedelta(days=365))]
    blocks = recent.groupby(["game_id", "team_id"])["blocks"].sum().reset_index()
    g = games[["game_id", "home_id", "away_id", "home_sog", "away_sog"]]
    blocks = blocks.merge(g, on="game_id")
    blocks["opp_sog"] = np.where(blocks.team_id == blocks.home_id, blocks.away_sog, blocks.home_sog)
    lg = blocks["blocks"].sum() / max(blocks["opp_sog"].sum(), 1)
    t = blocks[blocks.team_id == team_id].tail(40)
    return float((t["blocks"].sum() + lg * prior_n * 30) / (t["opp_sog"].sum() + prior_n * 30))


def team_pp_goal_frac(shots: pd.DataFrame, team_id: int, asof: date, lg_frac: float, prior_goals: float = 30) -> float:
    d = pd.to_datetime(shots.game_date).dt.date
    s = shots[(shots.team_id == team_id) & shots.is_goal & ~shots.empty_net & (d < asof) &
              (d >= asof - timedelta(days=365))]
    pp = (s.sk_for > s.sk_against).sum()
    return float((pp + lg_frac * prior_goals) / (len(s) + prior_goals))
