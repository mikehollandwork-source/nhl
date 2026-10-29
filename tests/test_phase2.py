"""Tests for phase-2 inputs: stints, player impacts, special teams, penalties, props by strength,
DailyFaceoff parsing, referee factors and travel features."""

import json
from datetime import date

import numpy as np
import pandas as pd
import pytest

from nhlmodel.features import officials as OF
from nhlmodel.features import rapm as RP
from nhlmodel.features import special_teams as ST
from nhlmodel.features.travel import travel_features
from nhlmodel.ingest import lineups as LN
from nhlmodel.ingest.shifts import parse_right_rail, parse_shifts, strength_toi
from nhlmodel.sim import props as P
from nhlmodel.sim.engine import GameInput, SimConstants, simulate


def _shift(pid, team, period, start, end):
    return {"playerId": pid, "teamId": team, "period": period, "startTime": start, "endTime": end, "typeCode": 517}


def test_stints_and_strength_toi():
    home, away = 1, 2
    rows = [_shift(100, home, 1, "00:00", "20:00"), _shift(200, away, 1, "00:00", "20:00")]  # goalies
    rows += [_shift(p, home, 1, "00:00", "20:00") for p in (11, 12, 13, 14)]
    rows += [_shift(15, home, 1, "00:00", "05:00"), _shift(16, home, 1, "07:00", "20:00")]  # home short 05:00-07:00
    rows += [_shift(p, away, 1, "00:00", "20:00") for p in (21, 22, 23, 24, 25)]
    rows.append({"typeCode": 505, "playerId": 11, "teamId": home, "period": 1, "startTime": "03:00", "endTime": "03:00"})
    st = parse_shifts({"data": rows}, 1, 2, home, {100, 200})
    assert st.secs.sum() == 1200
    pk = st[(st.n_home == 4) & (st.n_away == 5)]
    assert pk.secs.sum() == 120
    assert (st[st.n_home == 5].n_away == 5).all()
    toi = strength_toi(st, home, away).set_index("player_id")
    assert toi.loc[11, "toi_sh_s"] == 120 and toi.loc[21, "toi_pp_s"] == 120
    assert toi.loc[15, "toi_5v5_s"] == 300 and toi.loc[16, "toi_5v5_s"] == 780


def test_right_rail_before_warmups_is_empty():
    off, scr = parse_right_rail({"gameInfo": {"referees": [], "linesmen": [], "homeTeam": {}, "awayTeam": {}}}, 5)
    assert off["ref1"] is None and scr == []


def test_rapm_recovers_known_impacts():
    rng = np.random.default_rng(0)
    n = 6000
    pool_h = np.arange(1, 21)
    pool_a = np.arange(21, 41)
    true_off = {p: 0.0 for p in range(1, 41)}
    true_off[3] = 1.5     # strong offensive player
    true_off[30] = -1.0   # weak one
    recs = []
    for _ in range(n):
        h = rng.choice(pool_h, 5, replace=False)
        a = rng.choice(pool_a, 5, replace=False)
        secs = 60
        lam_h = (2.5 + sum(true_off[p] for p in h)) * secs / 3600
        lam_a = (2.5 + sum(true_off[p] for p in a)) * secs / 3600
        rec = {"season": 20252026, "secs": secs, "xg_home": rng.poisson(lam_h * 20) / 20,
               "xg_away": rng.poisson(lam_a * 20) / 20, "sog_home": 0, "sog_away": 0}
        for k in range(6):
            rec[f"h{k + 1}"] = int(h[k]) if k < 5 else 0
            rec[f"a{k + 1}"] = int(a[k]) if k < 5 else 0
        recs.append(rec)
    r = RP.fit(pd.DataFrame(recs), "xg", alpha=50).set_index("player_id")
    assert r.loc[3, "off_xg"] == r.off_xg.max()
    assert r.loc[30, "off_xg"] == r.off_xg.min()


def test_strength_ratings_point_in_time():
    base = {"season": 20252026, "opp_id": 2, "is_home": 1, "valid": True, "reg_secs": 3600, "ev_secs": 3000,
            "pp_secs": 300, "sh_secs": 300, "pk_starts": 2, "pp_starts": 2}
    cnt = {f"{s}_{c}": 1.0 for s in ("ev", "pp", "sh") for c in ("sogf", "xgf", "gf", "soga", "xga", "ga")}
    tg = pd.DataFrame([{**base, **cnt, "game_id": i, "team_id": 1, "game_date": date(2025, 10, i + 1), "ev_sogf": 20.0}
                       for i in range(1, 5)])
    a = ST.strength_ratings(tg)
    tg2 = tg.copy()
    tg2.loc[tg2.game_id == 3, "ev_sogf"] = 80.0
    b = ST.strength_ratings(tg2)
    r = lambda df, g: float(df.loc[df.game_id == g, "s_ev_sogf"].iloc[0])  # noqa: E731
    assert r(a, 3) == r(b, 3) and r(b, 4) > r(a, 4)


def test_penalties_and_power_plays():
    gi = GameInput(sog_h=28, sog_a=28, gp_h=0.09, gp_a=0.09, pen_h=3.0, pen_a=3.0, pp_sog_h=50, pp_sog_a=50,
                   pp_gp_h=0.15, pp_gp_a=0.15, sh_sog_h=10, sh_sog_a=10, sh_gp_h=0.08, sh_gp_a=0.08)
    c = SimConstants(pull_prob={1: 0, 2: 0})
    s = simulate(gi, c, n=20000, rng=np.random.default_rng(1))
    assert 5.0 < (s.pp_count_h + s.pp_count_a).mean() < 6.2          # ~3 per team, minus stacking
    assert (s.pp_goals_h <= s.pp_count_h).all()                        # minors end on a PP goal (no majors)
    assert (s.pp_goals_h + s.sh_goals_h <= s.goals_h).all()
    assert (s.pp_sog_h >= s.pp_goals_h).all()
    off = simulate(GameInput(28, 28, 0.09, 0.09), c, n=20000, rng=np.random.default_rng(1))
    assert off.pp_count_h is None                                      # phase-1 path unchanged


def test_props_v2_consistency():
    gi = GameInput(sog_h=30, sog_a=28, gp_h=0.095, gp_a=0.09, pen_h=3, pen_a=3, pp_sog_h=50, pp_sog_a=50,
                   pp_gp_h=0.15, pp_gp_a=0.15, sh_sog_h=10, sh_sog_a=10, sh_gp_h=0.08, sh_gp_a=0.08)
    s = simulate(gi, SimConstants(), n=10000, rng=np.random.default_rng(2))
    n = 18
    pl = pd.DataFrame({"player_id": range(n), "w_blk": 1.0})
    for st in ("ev", "pp", "sh"):
        on = np.ones(n) if st == "ev" else (np.arange(n) < 5).astype(float) + 1e-3
        pl[f"w_{st}_goal"] = 0.1 * on
        pl[f"w_{st}_sog"] = 1.0 * on
        pl[f"w_{st}_a1"] = 0.2 * on
        pl[f"w_{st}_a2"] = 0.2 * on
    tp = P.TeamPropsV2(pl, {"ev": (0.9, 0.7), "pp": (0.95, 0.85), "sh": (0.8, 0.6)}, 0.5)
    o = P.team_player_sims_v2(s, "home", tp, np.random.default_rng(3))
    assert (o["goals"].sum(1) == s.goals_h).all()
    assert (o["sog"].sum(1) == s.sog_h).all()
    assert (o["ppp"].sum(1) <= 3 * s.pp_goals_h).all()
    # power-play points go almost entirely to the 5 PP players
    assert o["ppp"][:, :5].sum() > 0.95 * o["ppp"].sum()


def test_dailyfaceoff_parse():
    data = {"props": {"pageProps": {"combinations": {"updatedAt": "2026-10-27T14:00:00Z", "players": [
        {"name": "Auston Matthews", "groupIdentifier": "f1", "positionIdentifier": "c", "injuryStatus": None, "gameTimeDecision": False},
        {"name": "Auston Matthews", "groupIdentifier": "pp1", "positionIdentifier": "sk1", "injuryStatus": None, "gameTimeDecision": False},
        {"name": "Kirill Marchenko", "groupIdentifier": "f2", "positionIdentifier": "lw", "injuryStatus": None, "gameTimeDecision": True},
        {"name": "Kirill Marchenko", "groupIdentifier": "ir", "positionIdentifier": "ir1", "injuryStatus": "dtd", "gameTimeDecision": False},
        {"name": "John Tavares", "groupIdentifier": "ir", "positionIdentifier": "ir2", "injuryStatus": "out", "gameTimeDecision": False},
    ]}}}}
    html = f'<script id="__NEXT_DATA__" type="application/json">{json.dumps(data)}</script>'
    lines, upd = LN.parse_page(html)
    s = LN.summarize(lines, pd.DataFrame()).set_index("name")
    assert upd.startswith("2026-10-27")
    assert s.loc["Auston Matthews", "pp_unit"] == 1 and s.loc["Auston Matthews", "dressed"]
    assert s.loc["Kirill Marchenko", "injury"] == "dtd" and s.loc["Kirill Marchenko", "gtd"]
    assert not s.loc["John Tavares", "dressed"]
    assert LN.team_slug("Montréal", "Canadiens") == "montreal-canadiens"
    assert LN.team_slug("St. Louis", "Blues") == "st-louis-blues"


def test_referee_factor_uses_only_earlier_games():
    games = pd.DataFrame({"game_id": [1, 2, 3], "game_date": [date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3)]})
    off = pd.DataFrame({"game_id": [1, 2, 3], "ref1": ["A", "A", "D"], "ref2": ["B", "C", "B"]})
    pp = pd.DataFrame({"game_id": [1, 2, 3], "pp_total": [12, 12, 4]})
    f, latest = OF.ref_factors(off, pp, games)
    f = f.set_index("game_id").ref_factor
    assert f.loc[1] == pytest.approx(1.0)        # no history yet
    assert f.loc[2] > 1.0                        # A called a lot in game 1
    assert latest["A"] > 1.0 and OF.crew_factor(latest, "A", None) == latest["A"]
    assert OF.crew_factor(latest, None, None) == 1.0


def test_travel_features():
    games = pd.DataFrame({
        "game_id": [1, 2, 3, 4], "game_date": ["2025-10-01", "2025-10-02", "2025-10-04", "2025-10-06"],
        "home_id": [9, 7, 8, 1], "away_id": [1, 1, 1, 5], "venue_utc_offset": [-7.0, -7.0, -4.0, -4.0]})
    t = travel_features(games)
    t1 = t[t.team_id == 1].set_index("game_id")
    assert t1.loc[3, "tz_shift"] == 3.0 and t1.loc[3, "tz_big"] == 1
    assert t1.loc[2, "rest_days"] == 1 and t1.loc[3, "road_trip"] == 3 and t1.loc[4, "road_trip"] == 0
    assert t1.loc[4, "games_last7"] == 3


def test_lineup_replacement_and_pre_sums():
    from nhlmodel.features import lineup as LU
    # decayed sums never include the current game
    s = LU._pre_sum(np.array([1.0, 1.0, 1.0]), 0.5)
    assert list(s) == [0.0, 1.0, 1.5]
    rapm = pd.DataFrame({"player_id": [1, 2, 3, 4], "toi_s": [50000, 60000, 100, 200],
                         "off_xg": [0.5, 0.1, -0.3, -0.2], "def_xg": [0.0] * 4, "off_sog": [0.0] * 4, "def_sog": [0.0] * 4})
    repl = LU.replacement(rapm, quantile=0.5)
    assert repl["off_xg"] == pytest.approx(-0.25)
    star = pd.DataFrame({"player_id": [1, 2], "proj_toi": [900.0, 900.0]})
    callup = pd.DataFrame({"player_id": [99, 2], "proj_toi": [900.0, 900.0]})   # 99 has no rating
    assert LU.impact_of(callup, rapm, repl)["off_xg"] < LU.impact_of(star, rapm, repl)["off_xg"]
    # an unrated player counts as replacement level, not league average (0)
    assert LU.impact_of(callup, rapm, repl)["off_xg"] == pytest.approx(5 * (-0.25 + 0.1) / 2)


def test_lineup_history_delta_zero_for_unchanged_lineup():
    from nhlmodel.features import lineup as LU
    rows = []
    for g in range(1, 8):
        for p in (1, 2, 3):
            rows.append({"game_id": g, "player_id": p, "team_id": 10, "game_date": date(2025, 10, g), "position": "C",
                         "proj_toi": 800.0, "proj_pp_min": 2.0, "ev_xg60": 0.6, "pp_xg60": 1.5, "finishing": 1.0})
    form = pd.DataFrame(rows)
    rapm = pd.DataFrame({"player_id": [1, 2, 3], "toi_s": [1e5] * 3, "off_xg": [0.2, 0.0, -0.1], "def_xg": [0.0] * 3,
                         "off_sog": [0.0] * 3, "def_sog": [0.0] * 3})
    lu = LU.lineup_history(form, {20252026: rapm}, {g: 20252026 for g in range(1, 8)})
    assert np.allclose(lu.d_off_xg, 0.0) and np.allclose(lu.d_pp, 0.0)


def test_rowwise_multinomial_and_volume_noise():
    rng = np.random.default_rng(0)
    p = np.tile(np.array([0.5, 0.3, 0.2]), (20000, 1))
    counts = rng.poisson(10, 20000)
    out = P._multinomial_rows(counts, p, rng)
    assert (out.sum(1) == counts).all()
    assert np.allclose(out.mean(0) / counts.mean(), [0.5, 0.3, 0.2], atol=0.01)
    # saved-shot volume noise widens shots on goal but leaves goals unchanged
    gi = GameInput(30, 30, 0.09, 0.09)
    a = simulate(gi, SimConstants(), n=20000, rng=np.random.default_rng(5))
    b = simulate(gi, SimConstants(sog_var=0.03), n=20000, rng=np.random.default_rng(5))
    assert b.sog_h.var() > a.sog_h.var() * 1.2
    assert abs(b.goals_h.mean() - a.goals_h.mean()) < 0.05


def test_stretch_keeps_totals():
    from nhlmodel.sim.engine import stretch
    gi = GameInput(sog_h=33, sog_a=27, gp_h=0.10, gp_a=0.085, pen_h=2.5, pen_a=3.5, pp_sog_h=52, pp_sog_a=46,
                   pp_gp_h=0.16, pp_gp_a=0.14)
    s = stretch(gi, 1.6)
    assert s.sog_h + s.sog_a == pytest.approx(60)
    assert s.sog_h * s.gp_h + s.sog_a * s.gp_a == pytest.approx(33 * 0.10 + 27 * 0.085)
    assert s.pen_h + s.pen_a == pytest.approx(6.0)
    assert (s.sog_h * s.gp_h) / (s.sog_a * s.gp_a) > (33 * 0.10) / (27 * 0.085)   # gap widened
    assert stretch(gi, 1.0) is gi


def test_macro_ratings_point_in_time_and_bounds():
    from nhlmodel.features import macro as MC
    base = {"season": 20252026, "opp_id": 2, "fo_oz_wins": 5, "fo_oz": 10, "fo_dz_wins": 5, "fo_dz": 10,
            "fo_pp_wins": 2, "fo_pp": 4, "oz_starts_5v5": 8, "dz_starts_5v5": 8, "hits": 20, "hits_against": 20,
            "giveaways": 8, "takeaways": 6}
    te = pd.DataFrame([{**base, "game_id": i, "team_id": 1, "game_date": date(2025, 10, i), "fo_wins": 30, "fo_total": 60}
                       for i in range(1, 6)])
    a = MC.macro_ratings(te)
    te2 = te.copy()
    te2.loc[te2.game_id == 3, "fo_wins"] = 60
    b = MC.macro_ratings(te2)
    r = lambda df, g: float(df.loc[df.game_id == g, "m_fo_pct"].iloc[0])  # noqa: E731
    assert r(a, 3) == r(b, 3) and r(b, 4) > r(a, 4)
    assert a.m_oz_start_pct.between(0, 1).all() and a.m_fo_pct.between(0, 1).all()
