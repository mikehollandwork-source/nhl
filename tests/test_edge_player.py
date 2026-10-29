import pandas as pd
import pytest

from nhlmodel.features import edge_player as EP
from nhlmodel.ingest import edge


def _payload(gp=67):
    return {"player": {"gamesPlayed": gp},
            "skatingSpeed": {"speedMax": {"imperial": 23.969}, "burstsOver20": {"value": 442}},
            "totalDistanceSkated": {"imperial": 256.1},
            "topShotSpeed": {"imperial": 88.0},
            "zoneTimeDetails": {"offensiveZoneEvPctg": 0.466}}


def test_parse_skater():
    r = edge.parse_skater(_payload(), 8478402, 20242025)
    assert r["speed_max"] == pytest.approx(23.969)
    assert r["bursts20_pg"] == pytest.approx(442 / 67)
    assert edge.parse_skater(_payload(gp=5), 1, 20242025) is None   # too few games


def _edge():
    return pd.DataFrame({"player_id": [1, 2, 3, 4], "season": [20232024] * 4,
                         "speed_max": [21.0, 22.0, 23.0, 24.0], "bursts20_pg": [1.0, 2.0, 3.0, 4.0],
                         "oz_ev_pct": [0.40, 0.42, 0.44, 0.46]})


def test_prior_season_mapping():
    z = EP.prior_season_z(_edge())
    assert set(z.season) == {20242025}               # 2023-24 values are used in 2024-25
    assert z.e_speed.mean() == pytest.approx(0.0)


def test_unrated_player_gets_replacement():
    z = EP.prior_season_z(_edge())
    rep = z.e_speed.quantile(EP.REPL_Q)
    v = EP.lineup_values(pd.DataFrame({"player_id": [99], "proj_toi": [600.0]}), z, 20242025)
    assert v["e_speed"] == pytest.approx(rep)


def test_delta_zero_for_unchanged_lineup():
    z = EP.prior_season_z(_edge())
    rows = []
    for g in range(6):
        for p in (1, 2, 3):
            rows.append({"game_id": g, "team_id": 10, "player_id": p, "game_date": f"2024-10-{g + 10}", "proj_toi": 600.0})
    lu = EP.lineup_history(pd.DataFrame(rows), z, {g: 20242025 for g in range(6)})
    assert (lu.d_e_speed.abs() < 1e-12).all()
    # swap the slowest player for the fastest in the last game -> positive speed delta
    rows[-3]["player_id"] = 4
    lu = EP.lineup_history(pd.DataFrame(rows), z, {g: 20242025 for g in range(6)})
    assert lu.sort_values("game_id").d_e_speed.iloc[-1] > 0
