"""Load stored history and build point-in-time rating tables (shared by train, backtest and daily runs)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nhlmodel.features import team as T
from nhlmodel.features.xg import XGModel
from nhlmodel.sim.params import attach_opponent, exposure
from nhlmodel.store import db


@dataclass
class History:
    games: pd.DataFrame
    shots: pd.DataFrame          # with xg column
    states: pd.DataFrame
    player_games: pd.DataFrame
    goalie_games: pd.DataFrame
    team_games: pd.DataFrame
    team_ratings: pd.DataFrame   # pre-game ratings per team-game (+ game_id -1 'next game' rows)
    goalie_ratings: pd.DataFrame


def load(xg: XGModel, seasons: list[int] | None = None) -> History:
    parts = [str(s) for s in seasons] if seasons else None
    games = db.read("games", partitions=parts)
    games = games[games.game_type.isin([2, 3])].copy()
    shots = db.read("shots", partitions=parts)
    shots = shots[shots.game_id.isin(games.game_id)].copy()
    shots["xg"] = xg.predict(shots)
    states = db.read("states", partitions=parts)
    pg = db.read("player_games", partitions=parts)
    gg = db.read("goalie_games", partitions=parts)
    tg = T.team_games(games, shots)
    tr = T.team_ratings(tg)
    gr = T.goalie_ratings(T.goalie_games_xg(shots))
    return History(games, shots, states, pg, gg, tg, tr, gr)


def training_rows(h: History) -> pd.DataFrame:
    """Team-game rows with pre-game ratings, opponent ratings, opponent starter's goalie factor,
    exposure and observed outcomes. Used to fit and backtest the game model."""
    tr = h.team_ratings[h.team_ratings.game_id > 0].copy()
    rows = attach_opponent(tr)
    starters = h.goalie_games[h.goalie_games.starter][["game_id", "team_id", "player_id"]].rename(
        columns={"team_id": "opp_id", "player_id": "opp_goalie_id"})
    rows = rows.merge(starters, on=["game_id", "opp_id"], how="left")
    gf = h.goalie_ratings[h.goalie_ratings.game_id > 0][["game_id", "goalie_id", "goalie_factor"]].rename(
        columns={"goalie_id": "opp_goalie_id", "goalie_factor": "opp_goalie_factor"})
    rows = rows.merge(gf, on=["game_id", "opp_goalie_id"], how="left")
    rows = rows.merge(exposure(h.states), on="game_id", how="left")
    return rows
