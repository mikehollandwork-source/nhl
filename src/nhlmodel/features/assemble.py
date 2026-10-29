"""Assemble point-in-time team-game rows with every phase-2 feature group.

Every value attached to a game uses only information available before that game:
strength ratings and goalie ratings are pre-game; player impacts (RAPM) for season S are fit on
seasons before S; EDGE uses the previous season; referee factors use earlier games.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from nhlmodel.features import edge_player as EP
from nhlmodel.features import lineup as LU
from nhlmodel.features import macro as MC
from nhlmodel.features import officials as OF
from nhlmodel.features import players as PL
from nhlmodel.features import rapm as RP
from nhlmodel.features import special_teams as ST
from nhlmodel.features import team as T
from nhlmodel.features import travel as TR
from nhlmodel.features.history import History
from nhlmodel.store import db

log = logging.getLogger(__name__)
EDGE_COLS = {"bursts20_pg": "edge_bursts", "oz_ev_pct": "edge_oz", "shots90_pg": "edge_shots90"}


@dataclass
class Extras:
    stints: pd.DataFrame
    penalties: pd.DataFrame
    player_toi: pd.DataFrame
    officials: pd.DataFrame
    edge: pd.DataFrame
    team_events: pd.DataFrame | None = None
    edge_skater: pd.DataFrame | None = None


def load_extras(seasons: list[int] | None = None) -> Extras:
    parts = [str(s) for s in seasons] if seasons else None
    return Extras(db.read("stints", partitions=parts), db.read("penalties", partitions=parts),
                  db.read("player_toi", partitions=parts), db.read("officials", partitions=parts),
                  db.read("edge_team"), db.read("team_events", partitions=parts), db.read("edge_skater"))


def rapm_by_season(h: History, ex: Extras, seasons: list[int]) -> tuple[dict[int, pd.DataFrame], pd.DataFrame]:
    """Player impacts for each season S fit on seasons < S. Also returns the stint table used."""
    st = RP.stint_events(ex.stints, h.shots, h.games)
    out = {}
    for s in seasons:
        prior = st[st.season < s]
        if prior.empty:
            continue
        out[s] = RP.fit_all(prior, latest_season=int(prior.season.max()))
        log.info("RAPM for %s: %d players from %d stints", s, len(out[s]), len(prior))
    return out, st


def edge_features(edge: pd.DataFrame) -> pd.DataFrame:
    """Prior-season EDGE values, z-scored within season, keyed to the NEXT season."""
    if edge.empty:
        return pd.DataFrame(columns=["team_id", "season", *EDGE_COLS.values()])
    e = edge.copy()
    for src in EDGE_COLS:
        e[src] = (e[src] - e.groupby("season")[src].transform("mean")) / e.groupby("season")[src].transform("std")
    e["season"] = e.season + 10001  # 20242025 -> 20252026
    return e[["team_id", "season", *EDGE_COLS]].rename(columns=EDGE_COLS)


def assemble(h: History, ex: Extras, rapm: dict[int, pd.DataFrame] | None = None) -> pd.DataFrame:
    games = h.games
    tg = ST.team_strength_games(games, h.shots, ex.stints)
    sr = ST.strength_ratings(tg)
    ratings = sr[sr.game_id > 0].drop(columns=["season", "game_date"])
    rows = tg.merge(ratings, on=["game_id", "team_id"], how="left")
    opp = ratings[["game_id", "team_id"] + [c for c in ratings.columns if c.startswith("s_")]].rename(
        columns={"team_id": "opp_id", **{c: "o_" + c[2:] for c in ratings.columns if c.startswith("s_")}})
    rows = rows.merge(opp, on=["game_id", "opp_id"], how="left")

    rest = h.team_games[["game_id", "team_id", "b2b", "three_in_four"]]
    rows = rows.merge(rest, on=["game_id", "team_id"], how="left")
    rows = rows.merge(rest.rename(columns={"team_id": "opp_id", "b2b": "opp_b2b", "three_in_four": "opp_three_in_four"}),
                      on=["game_id", "opp_id"], how="left")

    # opponent starting goalie: overall factor (phase 1) + detail
    starters = h.goalie_games[h.goalie_games.starter][["game_id", "team_id", "player_id"]].rename(
        columns={"team_id": "opp_id", "player_id": "opp_goalie_id"})
    rows = rows.merge(starters.drop_duplicates(["game_id", "opp_id"]), on=["game_id", "opp_id"], how="left")
    gf = h.goalie_ratings[h.goalie_ratings.game_id > 0][["game_id", "goalie_id", "goalie_factor"]].rename(
        columns={"goalie_id": "opp_goalie_id", "goalie_factor": "opp_goalie_factor"})
    rows = rows.merge(gf, on=["game_id", "opp_goalie_id"], how="left")
    gd = T.goalie_detail(h.shots, h.goalie_games)
    gd = gd[gd.game_id > 0].rename(columns={"goalie_id": "opp_goalie_id", "g_high": "opp_g_high",
                                             "g_lowmid": "opp_g_lowmid", "g_rebound": "opp_g_rebound",
                                             "starts7": "opp_g_starts7", "g_b2b": "opp_g_b2b"})
    rows = rows.merge(gd, on=["game_id", "opp_goalie_id"], how="left")

    if "venue_utc_offset" in games:
        tv = TR.travel_features(games)
        rows = rows.merge(tv, on=["game_id", "team_id"], how="left")
        rows = rows.merge(tv[["game_id", "team_id", "tz_big", "games_last7"]].rename(
            columns={"team_id": "opp_id", "tz_big": "opp_tz_big", "games_last7": "opp_games_last7"}),
            on=["game_id", "opp_id"], how="left")

    if not ex.officials.empty:
        pp = tg.groupby("game_id").pk_starts.sum().rename("pp_total").reset_index()
        rf, _ = OF.ref_factors(ex.officials, pp, games)
        rows = rows.merge(rf, on="game_id", how="left")

    if rapm:
        form = LU.player_form(h.player_games, ex.player_toi, PL.player_state_events(h.shots))
        seasons = games.set_index("game_id").season.to_dict()
        lu = LU.lineup_history(form, rapm, seasons)
        if not lu.empty:
            d_cols = [f"d_{c}" for c in LU.ALL]
            rows = rows.merge(lu[["game_id", "team_id", *d_cols]], on=["game_id", "team_id"], how="left")
            rows = rows.merge(lu[["game_id", "team_id", "d_def_xg", "d_def_sog"]].rename(
                columns={"team_id": "opp_id", "d_def_xg": "o_d_def_xg", "d_def_sog": "o_d_def_sog"}),
                on=["game_id", "opp_id"], how="left")
            # "basic" variant: players without a rating count as league average (0) instead of replacement
            lu0 = LU.lineup_history(form, rapm, seasons, use_replacement=False)
            rows = rows.merge(lu0[["game_id", "team_id", "d_off_xg", "d_off_sog"]].rename(
                columns={"d_off_xg": "d0_off_xg", "d_off_sog": "d0_off_sog"}), on=["game_id", "team_id"], how="left")
            rows = rows.merge(lu0[["game_id", "team_id", "d_def_xg", "d_def_sog"]].rename(
                columns={"team_id": "opp_id", "d_def_xg": "o_d0_def_xg", "d_def_sog": "o_d0_def_sog"}),
                on=["game_id", "opp_id"], how="left")
        if ex.edge_skater is not None and not ex.edge_skater.empty:
            el = EP.lineup_history(form, EP.prior_season_z(ex.edge_skater), seasons)
            if not el.empty:
                dcols = [f"d_{c}" for c in EP.COLS]
                rows = rows.merge(el[["game_id", "team_id", *dcols]], on=["game_id", "team_id"], how="left")
                rows = rows.merge(el[["game_id", "team_id", *dcols]].rename(
                    columns={"team_id": "opp_id", **{c: "o_" + c for c in dcols}}), on=["game_id", "opp_id"], how="left")

    ef = edge_features(ex.edge)
    if not ef.empty:
        rows = rows.merge(ef, on=["team_id", "season"], how="left")
        rows = rows.merge(ef[["team_id", "season", "edge_bursts"]].rename(
            columns={"team_id": "opp_id", "edge_bursts": "opp_edge_bursts"}), on=["opp_id", "season"], how="left")

    if ex.team_events is not None and not ex.team_events.empty:
        rows = MC.attach(rows, MC.macro_ratings(ex.team_events))

    rows = rows.merge(games[["game_id", "game_type"]], on="game_id", how="left")
    num = rows.select_dtypes(include=[np.number]).columns
    rows[num] = rows[num].replace([np.inf, -np.inf], np.nan)
    return rows
