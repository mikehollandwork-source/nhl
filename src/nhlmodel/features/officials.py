"""Referee crew penalty factor.

For each referee: power plays called in his games vs the league average per game, shrunk:
    factor = (observed + K * avg) / (games * avg + K * avg)
K = prior games. Crew factor = mean of the two referees. Uses only games before the date.
Referees are posted only at warmups (verified), so live runs before then use 1.0.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

K_GAMES = 40.0


def ref_factors(officials: pd.DataFrame, pp_per_game: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """pp_per_game: game_id, pp_total (power plays awarded, both teams).

    Returns (per-game pre-game crew factor, latest factor per referee name)."""
    g = games[["game_id", "game_date"]].merge(officials[["game_id", "ref1", "ref2"]], on="game_id")
    g = g.merge(pp_per_game, on="game_id").sort_values(["game_date", "game_id"])
    avg = float(g.pp_total.mean()) if len(g) else 7.0
    obs: dict[str, float] = {}
    n: dict[str, float] = {}

    def factor(r):
        if not isinstance(r, str):
            return 1.0
        return (obs.get(r, 0.0) + K_GAMES * avg) / (n.get(r, 0.0) * avg + K_GAMES * avg)

    out = []
    for _date, day in g.groupby("game_date", sort=True):
        # all of a day's games are rated with information from earlier days only
        for row in day.itertuples(index=False):
            out.append({"game_id": row.game_id, "ref_factor": float(np.mean([factor(row.ref1), factor(row.ref2)]))})
        for row in day.itertuples(index=False):
            for r in (row.ref1, row.ref2):
                if isinstance(r, str):
                    obs[r] = obs.get(r, 0.0) + row.pp_total  # each referee is credited with the game's total
                    n[r] = n.get(r, 0.0) + 1.0
    latest = {r: (obs[r] + K_GAMES * avg) / (n[r] * avg + K_GAMES * avg) for r in obs}
    return pd.DataFrame(out), latest


def crew_factor(latest: dict, ref1: str | None, ref2: str | None) -> float:
    vals = [latest.get(r, 1.0) for r in (ref1, ref2) if isinstance(r, str) and r]
    return float(np.mean(vals)) if vals else 1.0
