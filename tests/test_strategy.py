import numpy as np
import pandas as pd
import pytest

from nhlmodel.ingest import history_prices as HP
from nhlmodel.pipelines import strategy_backtest as SB


def test_pick_quote_never_after_puck_drop():
    start = 10_000
    pts = [(start - 7200, 0.40, 0.42), (start - 1900, 0.45, 0.47), (start - 600, 0.60, 0.62), (start + 60, 0.9, 0.95)]
    q = HP.pick_quote(pts, start)
    assert q == (start - 1900, 0.45, 0.47)                      # last quote at least 30 min before
    assert HP.pick_quote([(start - 60, 0.5, 0.52)], start) is None
    assert HP.pick_quote([(start - 4000, 0.0, 0.52)], start) is None   # one-sided book ignored


def _cands(n=600, seed=0, model_sign=1.0):
    """Market is fair; the model adds a signal (model_sign > 0) or an anti-signal (< 0)."""
    rng = np.random.default_rng(seed)
    true = rng.uniform(0.35, 0.65, n)
    noise = rng.normal(0, 0.06, n)
    mkt = np.clip(true - noise, 0.05, 0.95)                     # market misses `noise`
    m = np.clip(mkt + model_sign * noise, 0.05, 0.95)           # the model sees it (or sees the opposite)
    y = (rng.random(n) < true).astype(int)
    d = pd.to_datetime("2025-10-01") + pd.to_timedelta(np.arange(n) // 6, unit="D")
    return pd.DataFrame({"game_id": np.arange(n), "season": 1, "game_date": d.strftime("%Y-%m-%d"), "market": "h2h",
                         "line": 0.0, "ask_a": mkt + 0.01, "ask_b": 1 - mkt + 0.01, "m_a": m, "y_a": y,
                         "mkt_a": mkt})


def test_place_and_settle():
    c = pd.DataFrame({"game_id": [1, 2], "season": 1, "game_date": "2025-10-01", "market": "totals", "line": 6.5,
                      "ask_a": [0.50, 0.50], "ask_b": [0.52, 0.52], "m_a": [0.60, 0.40], "y_a": [1, 1],
                      "mkt_a": 0.49, "p_corr": np.nan, "p_blend": np.nan})
    f = SB.place(c, "follow", 0.05)
    assert list(f.side) == ["a", "b"] and list(f.won) == [True, False]
    assert f.units.iloc[0] == pytest.approx(1.0) and f.units.iloc[1] == -1.0
    fade = SB.place(c, "fade", 0.05)
    assert list(fade.side) == ["b", "a"]                         # the opposite side of each follow bet
    assert SB.place(c, "follow", 0.2).empty


def test_blend_weight_sign():
    good = SB.walk_forward(_cands(model_sign=1.0)).w_blend.dropna().iloc[-1]
    bad = SB.walk_forward(_cands(model_sign=-1.0)).w_blend.dropna().iloc[-1]
    assert good > 0.3 and bad < -0.3                             # negative weight = fade the model


def test_walk_forward_uses_only_the_past():
    c = _cands()
    a = SB.walk_forward(c)
    cut = c.game_date.unique()[60]
    c2 = c.copy()
    late = c2.game_date > cut
    c2.loc[late, "y_a"] = 1 - c2.loc[late, "y_a"]                # change every future outcome
    b = SB.walk_forward(c2)
    early = c.game_date <= cut
    assert np.allclose(a.p_blend[early], b.p_blend[early], equal_nan=True)
    assert np.allclose(a.p_corr[early], b.p_corr[early], equal_nan=True)


def test_score_bootstrap_by_day():
    b = pd.DataFrame({"game_date": ["d1"] * 3 + ["d2"] * 3, "units": [1.0, 1.0, -1.0, 1.0, -1.0, 1.0],
                      "won": [True, True, False, True, False, True], "price": 0.5})
    s = SB.score(b)
    assert s["n"] == 6 and s["roi"] == pytest.approx(1 / 3) and s["lo90"] <= s["roi"] <= s["hi90"]
    assert SB.score(b.iloc[0:0]) == {"n": 0}


def test_candidates_main_total_line():
    preds = pd.DataFrame({"game_id": [7], "season": [1], "p_home": [0.6], "p_o45": [0.8], "p_o55": [0.6], "p_o65": [0.45],
                          "p_o75": [0.3], "p_o85": [0.2], "p_home_m15": [0.4], "p_away_m15": [0.2]})
    games = pd.DataFrame({"game_id": [7], "game_date": ["2025-10-08"], "home_score": [4], "away_score": [3]})
    rows = []
    for sel, ln, ask in (("over", 5.5, 0.64), ("under", 5.5, 0.38), ("over", 6.5, 0.49), ("under", 6.5, 0.53),
                         ("home", None, 0.62), ("away", None, 0.40), ("home", -1.5, 0.39), ("away", 1.5, 0.63)):
        mk = "totals" if sel in ("over", "under") else "h2h" if ln is None else "spreads"
        rows.append({"game_id": 7, "market": mk, "selection": sel, "line": ln, "ask": ask, "source": "x"})
    c = SB.candidates(preds, pd.DataFrame(rows), games).set_index("market")
    assert c.loc["totals", "line"] == 6.5 and c.loc["totals", "y_a"] == 1 and c.loc["totals", "m_a"] == 0.45
    assert c.loc["h2h", "y_a"] == 1
    assert c.loc["spreads", "line"] == -1.5 and c.loc["spreads", "y_a"] == 0     # won by 1: -1.5 loses


def test_main_lines_one_per_game_market():
    from nhlmodel.tracking import report as RP
    g = pd.DataFrame({"game_date": "d", "home": "A", "away": "B", "market": "totals",
                      "selection": ["under", "over", "under", "over", "under"], "line": [4.5, 6.5, 6.5, 5.5, 5.5],
                      "book_prob": [0.18, 0.49, 0.51, 0.62, 0.38], "flagged": True, "outcome": 1.0, "units": 1.0})
    m = RP.main_lines(g)
    assert set(m.line) == {6.5} and len(m) == 2            # only the line closest to 50/50 is kept


def test_strategy_message_and_plays_note():
    from nhlmodel.publish import discord as DC
    from nhlmodel.publish import format as F
    v = {"n": 300, "win_rate": 0.55, "roi": 0.08, "lo90": -0.02, "hi90": 0.17}
    rep = {"selection_season": 20252026, "test_season": 20262027, "selection": {"fade 5 totals": v},
           "test": {"fade 5 totals": {**v, "n": 117, "roi": -0.05}}, "test_verdict": {}, "props_posted": {}}
    msg = F.strategy_message(rep)
    assert "No betting rule has passed" in msg and "fade totals" in msg
    assert all(len(p) <= 2000 for p in DC.chunks(DC.to_markdown(msg)))
    assert "No proven plays" in F.plays_note(rep)
    assert "Proven rules" in F.plays_note({"test_verdict": {"fade 5 totals": "pass"}})
    assert F.plays_note(None) == ""
