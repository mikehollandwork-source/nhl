import numpy as np
import pandas as pd
import pytest

from nhlmodel.market import implied as IM
from nhlmodel.market.tells import bh
from nhlmodel.tracking import disagree as DG
from nhlmodel.tracking import grade as GR


def test_settle():
    assert GR.settle("h2h", "home", None, 3, 2) == 1.0
    assert GR.settle("h2h", "away", None, 3, 2) == 0.0
    assert GR.settle("spreads", "home", -1.5, 3, 2) == 0.0
    assert GR.settle("spreads", "away", 1.5, 3, 2) == 1.0
    assert GR.settle("spreads", "home", -1.0, 3, 2) is None     # push
    assert GR.settle("totals", "over", 5.5, 3, 3) == 1.0
    assert GR.settle("totals", "under", 6.0, 3, 3) is None


def _odds():
    rows = []
    for book, ph, pa, po, pu in (("dk", -150, 130, -110, -110), ("fd", -145, 125, -105, -115)):
        base = {"event_id": "e1", "commence_time": "2026-10-28T23:00:00Z", "home": "TOR", "away": "MTL",
                "book": book, "ts": "2026-10-28T15:00:00Z"}
        rows += [{**base, "market": "h2h", "selection": "home", "line": None, "price": ph},
                 {**base, "market": "h2h", "selection": "away", "line": None, "price": pa},
                 {**base, "market": "totals", "selection": "over", "line": 6.5, "price": po},
                 {**base, "market": "totals", "selection": "under", "line": 6.5, "price": pu}]
    return pd.DataFrame(rows)


def test_consensus_and_compare():
    fair = DG.per_book_fair(_odds())
    sums = fair.groupby(["book", "market"]).fair.sum()
    assert np.allclose(sums, 1.0)
    cons = DG.consensus(fair)
    home = cons[(cons.market == "h2h") & (cons.selection == "home")].iloc[0]
    assert home.best_price == -145 and home.best_book == "fd"
    dis = DG.compare(cons, lambda h, a, m, s, line: {"home": 0.65, "away": 0.35, "over": 0.45, "under": 0.55}[s])
    flagged = dis[dis.flagged]
    assert set(flagged.selection) == {"home", "under"}
    assert (flagged.edge > 0).all()


def test_grade_flow():
    dis = DG.compare(DG.consensus(DG.per_book_fair(_odds())),
                     lambda h, a, m, s, line: {"home": 0.65, "away": 0.35, "over": 0.45, "under": 0.55}[s])
    dis["game_date"] = "2026-10-28"
    dis["ts"] = "2026-10-28T15:00:00Z"
    results = pd.DataFrame([{"game_date": "2026-10-28", "home": "TOR", "away": "MTL", "home_score": 4, "away_score": 2}])
    cons = DG.consensus(DG.per_book_fair(_odds())).assign(ts="2026-10-28T22:00:00Z")
    g = GR.grade(dis, results, cons)
    home = g[(g.market == "h2h") & (g.selection == "home")].iloc[0]
    assert home.outcome == 1.0 and home.units == pytest.approx(home.best_dec - 1)
    under = g[(g.market == "totals") & (g.selection == "under")].iloc[0]
    assert under.outcome == 1.0
    s = GR.summarize(g)
    assert s["flagged"] == 2 and s["flagged_win_rate"] == 1.0
    assert g.close_prob.notna().all()   # a later snapshot exists, so closing info is kept


def test_implied_rates_roundtrip():
    lh, la = 3.3, 2.7
    p_home = IM.probs_from_rates(lh, la)["p_home"]
    p_over = IM.p_over(lh, la, 6.5)
    slh, sla, err = IM.solve_rates(p_home, 6.5, p_over)
    assert err < 1e-6
    assert slh == pytest.approx(lh, rel=1e-3) and sla == pytest.approx(la, rel=1e-3)


def test_implied_spread_complement():
    a = IM.p_home_spread(3.2, 2.8, -1.5)
    b = IM.p_home_spread(2.8, 3.2, 1.5)   # same game from the away side
    assert a + b == pytest.approx(1.0, abs=1e-7)


def test_benjamini_hochberg():
    assert bh([0.001, 0.2, 0.03, 0.5], 0.10) == [True, False, True, False]
    assert bh([], 0.1) == []
