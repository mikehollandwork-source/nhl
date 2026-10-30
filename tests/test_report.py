from datetime import date, timedelta

import numpy as np
import pandas as pd

from nhlmodel.publish import discord as DC
from nhlmodel.publish import format as F
from nhlmodel.tracking import report as RP

DAY = date(2026, 12, 31)


def _games(n, over_proj=0.0, seed=1):
    rng = np.random.default_rng(seed)
    rows, res = [], []
    for i in range(n):
        gd = (DAY - timedelta(days=i // 6)).isoformat()
        h, a = f"H{i}", f"A{i}"
        hs, as_ = int(rng.poisson(3.1)), int(rng.poisson(2.9))
        for sel, v in (("p_home", 0.55), ("exp_goals_home", 3.1 + over_proj / 2), ("exp_goals_away", 2.9 + over_proj / 2)):
            rows.append({"run_type": "morning", "market": "summary", "ts": "t", "game_date": gd, "home": h,
                         "away": a, "selection": sel, "model_prob": v})
        res.append({"game_date": gd, "home": h, "away": a, "home_score": hs, "away_score": as_})
    return pd.DataFrame(rows), pd.DataFrame(res)


def _graded(n, model_better: bool, seed=2):
    """Graded moneyline rows where the model's probability is closer to the outcome than the book's."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n).astype(float)
    book = np.full(n, 0.5)
    model = np.where(y == 1, 0.62, 0.38) if model_better else np.where(y == 1, 0.40, 0.60)
    ll = lambda p, o: -(o * np.log(p) + (1 - o) * np.log(1 - p))  # noqa: E731
    gd = [(DAY - timedelta(days=i // 6)).isoformat() for i in range(n)]
    return pd.DataFrame({"game_date": gd, "home": [f"H{i}" for i in range(n)], "away": [f"A{i}" for i in range(n)],
                         "market": "h2h", "selection": "home", "outcome": y, "model_logloss": ll(model, y),
                         "book_logloss": ll(book, y), "flagged": False, "units": np.nan, "edge": model - book})


def test_strength_needs_sample_and_margin():
    proj, res = _games(150)
    rep = RP.build(DAY, proj, res, _graded(150, True), pd.DataFrame(), pd.DataFrame())
    assert any(f["label"] == "Moneyline prices" for f in rep["strengths"])
    rep = RP.build(DAY, proj, res, _graded(20, True), pd.DataFrame(), pd.DataFrame())
    ml = [f for f in rep["leaning"] + rep["strengths"] if f["label"] == "Moneyline prices"]
    assert ml and ml[0]["verdict"] == "leaning"                  # 20 games is not proof


def test_weakness_when_model_worse():
    proj, res = _games(150)
    rep = RP.build(DAY, proj, res, _graded(150, False), pd.DataFrame(), pd.DataFrame())
    assert any(f["label"] == "Moneyline prices" for f in rep["weaknesses"])


def test_scoring_bias_flagged():
    proj, res = _games(150, over_proj=1.0)
    rep = RP.build(DAY, proj, res, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
    bias = [f for f in rep["weaknesses"] if f["label"] == "Scoring bias"]
    assert bias and "high" in bias[0]["text"]


def test_star_tier_bias_named():
    n = 200
    lines = pd.DataFrame({"ts": "t", "game_date": "2026-12-30", "game_id": range(n), "player_id": 7,
                          "market": "points", "line": 0.5, "model_prob": 0.50, "kalshi_prob": 0.6, "edge": -0.1})
    pg = pd.DataFrame({"game_id": range(n), "player_id": 7, "goals": 0, "assists": 1, "points": 1,
                       "game_date": "2026-12-30"})
    hist = pd.DataFrame({"game_id": range(-30, 0), "player_id": 7, "goals": 1, "assists": 1, "points": 2,
                         "game_date": "2026-11-01"})
    rep = RP.build(DAY, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), lines, pd.concat([pg, hist]))
    stars = [f for f in rep["weaknesses"] + rep["leaning"] if f["label"] == "Stars props"]
    assert stars and "under-predict" in stars[0]["text"]


def test_message_sections_and_length():
    proj, res = _games(150, over_proj=1.0)
    rep = RP.build(DAY, proj, res, _graded(150, True), pd.DataFrame(), pd.DataFrame())
    msg = F.system_report_message(rep)
    assert "Strengths" in msg and "Weaknesses" in msg
    assert all(len(p) <= 2000 for p in DC.chunks(DC.to_markdown(msg)))
    early = F.system_report_message({"date": "x", "games": 3, "prop_lines": 0, "strengths": [], "weaknesses": [],
                                     "leaning": [], "trends": []})
    assert "No proven strengths" in early
