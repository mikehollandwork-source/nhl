import numpy as np
import pytest

from nhlmodel.market import devig as D


@pytest.mark.parametrize("method", D.METHODS)
@pytest.mark.parametrize("prices", [[-110, -110], [-150, 130], [-300, 240], [180, -220], [-120, 100, 400]])
def test_devig_sums_to_one(method, prices):
    p = D.devig(prices, method)
    assert p.sum() == pytest.approx(1.0, abs=1e-9)
    assert (p > 0).all()


def test_even_market_is_half():
    for m in D.METHODS:
        assert D.devig([-110, -110], m) == pytest.approx([0.5, 0.5])


def test_longshot_methods_take_more_from_dog():
    prices = [-300, 240]
    mult = D.devig(prices, "multiplicative")
    for m in ("power", "shin"):
        p = D.devig(prices, m)
        assert p[1] < mult[1]  # underdog fair prob lower than multiplicative


def test_odds_conversions_roundtrip():
    for a in (-250, -110, 100, 150, 400):
        assert D.decimal_to_american(D.american_to_decimal(a)) == pytest.approx(a, abs=1)
    assert D.american_to_decimal(-110) == pytest.approx(1.90909, rel=1e-4)
    assert D.prob_to_american(0.5) == pytest.approx(100, abs=1) or D.prob_to_american(0.5) == pytest.approx(-100, abs=1)


def test_ev_and_logloss():
    assert D.ev(0.5, 100) == pytest.approx(0.0)
    assert D.ev(0.55, -110) == pytest.approx(0.55 * 1.909090909 - 1)
    assert D.log_loss(0.7, 1) == pytest.approx(-np.log(0.7))
    assert D.log_loss(0.7, 0) == pytest.approx(-np.log(0.3))
