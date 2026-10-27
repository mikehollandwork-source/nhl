from datetime import date

import numpy as np
import pandas as pd
import pytest

from nhlmodel.features import team as T
from nhlmodel.ingest.nhl_api import parse_situation, shot_geometry
from nhlmodel.publish.telegram import _balance_pre, chunks
from nhlmodel.sim.engine import GameInput, SimConstants, simulate
from nhlmodel.sim.props import TeamProps, team_player_sims


def _players(n=12):
    return pd.DataFrame({"player_id": range(n), "name": [f"p{i}" for i in range(n)], "team": "TOR",
                         "w_goal": np.linspace(0.05, 0.3, n), "w_sog": np.linspace(1.0, 3.5, n),
                         "w_ast": np.linspace(0.1, 0.5, n), "w_blk": np.linspace(0.2, 1.5, n),
                         "w_ppp": np.linspace(0.01, 0.2, n)})


def test_props_consistent_with_team_totals():
    s = simulate(GameInput(30, 29, 0.095, 0.093), SimConstants(), n=20_000, step=10, rng=np.random.default_rng(3))
    tp = TeamProps(_players(), p_a1=0.9, p_a2_given_a1=0.75, pp_goal_frac=0.2, pp_pts_per_goal=2.7, block_ratio=0.5)
    out = team_player_sims(s.goals_h, s.en_h, s.sog_h, s.sog_a, tp, np.random.default_rng(4))
    assert (out["goals"].sum(axis=1) == s.goals_h).all()          # every goal has exactly one scorer
    assert (out["sog"].sum(axis=1) == s.sog_h).all()              # player shots add up to team shots
    assert (out["sog"] >= out["goals"]).all()
    # assists: at most 2 per goal, never more than goals * 2
    assert (out["assists"].sum(axis=1) <= 2 * s.goals_h).all()
    assert out["assists"].sum() / s.goals_h.sum() == pytest.approx(0.9 * 1.75, rel=0.05)
    assert (out["points"] == out["goals"] + out["assists"]).all()


def test_no_self_assist():
    """With one dominant scorer/assister, scorer must never also get an assist on the same goal."""
    p = _players(3)
    p["w_goal"] = [1.0, 1e-6, 1e-6]
    p["w_ast"] = [1.0, 0.5, 0.5]
    goals = np.ones(5000, int)
    tp = TeamProps(p, p_a1=1.0, p_a2_given_a1=0.0, pp_goal_frac=0.0, pp_pts_per_goal=2.7, block_ratio=0.5)
    out = team_player_sims(goals, np.zeros(5000, int), goals * 3, goals * 30, tp, np.random.default_rng(1))
    assert out["assists"][:, 0].sum() == 0


def test_situation_and_geometry():
    assert parse_situation("1551") == (1, 5, 5, 1)
    assert parse_situation("1560") == (1, 5, 6, 0)          # home goalie pulled, 6 skaters
    assert parse_situation(None) == (1, 5, 5, 1)
    d, a = shot_geometry(79, 0, +1)                          # 10 ft straight out from right net
    assert d == pytest.approx(10) and a == pytest.approx(0)
    d, a = shot_geometry(-79, 0, -1)                         # mirrored on left net
    assert d == pytest.approx(10)
    _, a = shot_geometry(95, 3, +1)                          # behind the net
    assert a > 90


def test_ratings_are_point_in_time():
    """A team's rating for game k must not change when game k's own result changes."""
    tg = pd.DataFrame({
        "game_id": [1, 2, 3, 4], "season": 20252026, "game_type": 2,
        "game_date": [date(2025, 10, d) for d in (8, 10, 12, 14)],
        "team_id": 1, "opp_id": 2, "is_home": 1,
        "sogf": [30, 30, 30, 30], "soga": 28, "xgf": 3.0, "xga": 2.8, "gf": 3, "ga": 2, "b2b": 0, "three_in_four": 0,
    })
    base = T.team_ratings(tg)
    tg2 = tg.copy()
    tg2.loc[tg2.game_id == 3, "sogf"] = 90
    alt = T.team_ratings(tg2)
    r = lambda df, g: float(df.loc[df.game_id == g, "r_sogf"].iloc[0])  # noqa: E731
    assert r(base, 3) == r(alt, 3)          # game 3's own result is not in its pre-game rating
    assert r(alt, 4) > r(base, 4)           # but it does feed the next game


def test_telegram_chunking_keeps_pre_balanced():
    text = "<pre>" + "\n".join("x" * 50 for _ in range(200)) + "</pre>"
    parts = _balance_pre(chunks(text, limit=1000))
    assert len(parts) > 1
    for p in parts:
        assert p.count("<pre>") == p.count("</pre>")
        assert len(p) <= 1100
