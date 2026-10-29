"""Build today's model inputs (one row per team) with the same features the backtest uses.

Everything here is "as of today": latest team strength ratings, latest goalie ratings, tonight's
projected lineup (DailyFaceoff when available, else the team's last lineup), travel since the last
game, prior-season EDGE values, and the referee crew factor when referees are known (warmups only).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from nhlmodel.config import settings
from nhlmodel.features import assemble as A
from nhlmodel.features import lineup as LU
from nhlmodel.features import macro as MC
from nhlmodel.features import officials as OF
from nhlmodel.features import players as PL
from nhlmodel.features import special_teams as ST
from nhlmodel.features import team as T
from nhlmodel.features.history import History

log = logging.getLogger(__name__)


@dataclass
class LiveContext:
    h: History
    ex: A.Extras
    lgs: dict
    strength_next: pd.DataFrame          # strength ratings 'next game' rows (game_id = -1)
    goalie_detail_next: pd.DataFrame
    rapm: pd.DataFrame
    form: pd.DataFrame                    # latest per-player form (proj_toi, proj_pp_min, pp_xg60, ...)
    positions: pd.Series                  # player_id -> position code
    repl: dict                            # replacement-level 5v5 impacts
    lineup_hist: pd.DataFrame
    lineup_hist0: pd.DataFrame            # same, with unrated players as league average ("basic" variant)
    edge: pd.DataFrame
    ref_latest: dict = field(default_factory=dict)
    macro_next: pd.DataFrame | None = None


def build_context(h: History, ex: A.Extras, rapm: pd.DataFrame, d: date) -> LiveContext:
    tg = ST.team_strength_games(h.games, h.shots, ex.stints)
    sr = ST.strength_ratings(tg)
    lgs = {f"lgs_{k}": v for k, v in ST.league_rates(tg, d).items()}
    gd = T.goalie_detail(h.shots, h.goalie_games)
    events = PL.player_state_events(h.shots)
    form_latest = LU.latest_form(h.player_games, ex.player_toi, events)
    positions = h.player_games.sort_values("game_date").groupby("player_id").position.last()
    season = int(h.games.season.max())
    cur = h.player_games[h.player_games.game_id.isin(h.games[h.games.season == season].game_id)]
    lu_hist = lu_hist0 = pd.DataFrame()
    if len(cur):
        form_cur = LU.player_form(cur, ex.player_toi, events)
        lu_hist = LU.lineup_history(form_cur, {season: rapm}, h.games.set_index("game_id").season.to_dict())
        lu_hist0 = LU.lineup_history(form_cur, {season: rapm}, h.games.set_index("game_id").season.to_dict(),
                                     use_replacement=False)
    ref_latest: dict = {}
    if not ex.officials.empty:
        pp = tg.groupby("game_id").pk_starts.sum().rename("pp_total").reset_index()
        _, ref_latest = OF.ref_factors(ex.officials, pp, h.games)
    macro_next = None
    if ex.team_events is not None and not ex.team_events.empty:
        mr = MC.macro_ratings(ex.team_events)
        macro_next = mr[mr.game_id == -1]
    return LiveContext(h, ex, lgs, sr[sr.game_id == -1], gd[gd.game_id == -1], rapm, form_latest, positions,
                       LU.replacement(rapm), lu_hist, lu_hist0, A.edge_features(ex.edge), ref_latest, macro_next)


def _strength(ctx: LiveContext, team_id: int, prefix: str) -> dict:
    r = ctx.strength_next[ctx.strength_next.team_id == team_id]
    out = {}
    for name, *_ in ST.RATES:
        v = float(r[f"s_{name}"].iloc[0]) if len(r) else ctx.lgs[f"lgs_{name}"]
        out[f"{prefix}{name}"] = v
    return out


def goalie_features(ctx: LiveContext, goalie_id: int | None, d: date) -> dict:
    gf, _ = T.latest_goalie_factor(ctx.h.goalie_ratings, goalie_id)
    det = ctx.goalie_detail_next[ctx.goalie_detail_next.goalie_id == goalie_id] if goalie_id else pd.DataFrame()
    s7, b2b = T.workload_for(ctx.h.goalie_games, goalie_id, d)
    return {"opp_goalie_factor": gf,
            "opp_g_high": float(det.g_high.iloc[0]) if len(det) else 1.0,
            "opp_g_lowmid": float(det.g_lowmid.iloc[0]) if len(det) else 1.0,
            "opp_g_rebound": float(det.g_rebound.iloc[0]) if len(det) else 1.0,
            "opp_g_starts7": s7, "opp_g_b2b": b2b}


def player_inputs(ctx: LiveContext, players: pd.DataFrame) -> pd.DataFrame:
    """Attach latest form to tonight's players; unknown players get position defaults."""
    p = players[["player_id"] + [c for c in ("p_play",) if c in players]].merge(
        ctx.form[["player_id", "proj_toi", "proj_pp_min", "pp_xg60", "ev_xg60", "finishing"]], on="player_id", how="left")
    pos = p.player_id.map(ctx.positions)
    default = np.where(pos == "D", LU.DEFAULT_TOI["D"], LU.DEFAULT_TOI["F"])
    p["proj_toi"] = p.proj_toi.fillna(pd.Series(default, index=p.index))
    p["proj_pp_min"] = p.proj_pp_min.fillna(0.0)
    p["pp_xg60"] = p.pp_xg60.fillna(ctx.form.pp_xg60.quantile(0.25))
    p["ev_xg60"] = p.ev_xg60.fillna(ctx.form.ev_xg60.quantile(0.25))
    p["finishing"] = p.finishing.fillna(1.0)
    return p


def lineup_delta(ctx: LiveContext, team_id: int, players: pd.DataFrame) -> dict:
    """players: player_id (+ optional p_play). Returns d_* for every lineup index."""
    if players.empty or ctx.rapm.empty:
        return {**{f"d_{c}": 0.0 for c in LU.ALL}, **{f"d0_{c}": 0.0 for c in LU.IMPACTS}}
    inp = player_inputs(ctx, players)
    now = LU.impact_of(inp, ctx.rapm, ctx.repl)
    typ = LU.typical_impact(ctx.lineup_hist, team_id) if not ctx.lineup_hist.empty else now
    out = {f"d_{c}": now[c] - typ[c] for c in LU.ALL}
    now0 = LU.impact_of(inp, ctx.rapm, None)
    typ0 = LU.typical_impact(ctx.lineup_hist0, team_id) if not ctx.lineup_hist0.empty else now0
    out.update({f"d0_{c}": now0[c] - typ0[c] for c in LU.IMPACTS})
    return out


def travel_today(ctx: LiveContext, team_id: int, d: date, venue_offset: float | None, is_home: bool) -> dict:
    g = ctx.h.games
    tg = g[(g.home_id == team_id) | (g.away_id == team_id)].copy()
    tg["gd"] = pd.to_datetime(tg.game_date).dt.date
    tg = tg[tg.gd < d].sort_values("gd")
    last_off = tg.venue_utc_offset.dropna().iloc[-1] if "venue_utc_offset" in tg and tg.venue_utc_offset.notna().any() else None
    shift = (venue_offset - last_off) if (venue_offset is not None and last_off is not None) else 0.0
    last7 = int((tg.gd >= d - timedelta(days=7)).sum())
    trip = 0
    if not is_home:
        trip = 1
        for _, row in tg.iloc[::-1].iterrows():
            if row.away_id == team_id:
                trip += 1
            else:
                break
    return {"tz_shift": shift, "tz_big": int(abs(shift) >= 2), "games_last7": last7, "road_trip": trip}


def team_row(ctx: LiveContext, team_id: int, opp_id: int, is_home: bool, d: date, rest: dict, opp_rest: dict,
             opp_goalie_id: int | None, players: pd.DataFrame, opp_players: pd.DataFrame,
             venue_offset: float | None, ref_factor: float = 1.0) -> pd.Series:
    season = int(ctx.h.games.season.max())
    row = {"team_id": team_id, "opp_id": opp_id, "season": season, "game_date": d, "is_home": int(is_home),
           "b2b": rest["b2b"], "three_in_four": rest["three_in_four"],
           "opp_b2b": opp_rest["b2b"], "opp_three_in_four": opp_rest["three_in_four"], "ref_factor": ref_factor}
    row.update(ctx.lgs)
    row.update(_strength(ctx, team_id, "s_"))
    row.update(_strength(ctx, opp_id, "o_"))
    row.update(goalie_features(ctx, opp_goalie_id, d))
    me = lineup_delta(ctx, team_id, players)
    op = lineup_delta(ctx, opp_id, opp_players)
    row.update(me)
    row.update({"o_d_def_xg": op["d_def_xg"], "o_d_def_sog": op["d_def_sog"],
                "o_d0_def_xg": op["d0_def_xg"], "o_d0_def_sog": op["d0_def_sog"]})
    tv = travel_today(ctx, team_id, d, venue_offset, is_home)
    otv = travel_today(ctx, opp_id, d, venue_offset, not is_home)
    row.update(tv)
    row.update({"opp_tz_big": otv["tz_big"], "opp_games_last7": otv["games_last7"]})
    if ctx.macro_next is not None:
        me_m = ctx.macro_next[ctx.macro_next.team_id == team_id]
        op_m = ctx.macro_next[ctx.macro_next.team_id == opp_id]
        for c in ctx.macro_next.columns:
            if c.startswith(("m_", "lgm_")) and len(me_m):
                row[c] = float(me_m[c].iloc[0])
            if c.startswith("m_") and len(op_m):
                row["o" + c] = float(op_m[c].iloc[0])
    e = ctx.edge[(ctx.edge.team_id == team_id) & (ctx.edge.season == season)]
    oe = ctx.edge[(ctx.edge.team_id == opp_id) & (ctx.edge.season == season)]
    for c in ("edge_bursts", "edge_oz", "edge_shots90"):
        row[c] = float(e[c].iloc[0]) if len(e) and pd.notna(e[c].iloc[0]) else 0.0
    row["opp_edge_bursts"] = float(oe.edge_bursts.iloc[0]) if len(oe) and pd.notna(oe.edge_bursts.iloc[0]) else 0.0
    return pd.Series(row)


def gtd_play_prob() -> float:
    return float(settings().get("lineups", {}).get("gtd_play_prob", 0.5))
