import numpy as np
import pytest

from nhlmodel.sim import markets as M
from nhlmodel.sim.baseline import bivariate_matrix
from nhlmodel.sim.engine import GameInput, SimConstants, SimResult, simulate

AVG = GameInput(sog_h=30.0, sog_a=29.0, gp_h=0.095, gp_a=0.093)


def run(gi=AVG, c=None, n=40_000, seed=1):
    return simulate(gi, c or SimConstants(), n=n, step=10, rng=np.random.default_rng(seed))


def test_bivariate_poisson_sums_to_one():
    for l3 in (0.0, 0.03, 0.05):
        assert bivariate_matrix(3.1, 2.8, l3).sum() == pytest.approx(1.0, abs=1e-6)


def test_bivariate_reduces_to_independent():
    from scipy.stats import poisson
    m = bivariate_matrix(3.0, 2.5, 0.0, 12)
    assert m[2, 3] == pytest.approx(poisson.pmf(2, 3.0) * poisson.pmf(3, 2.5))


def test_basic_invariants():
    s = run()
    assert (s.sog_h >= s.goals_h).all() and (s.sog_a >= s.goals_a).all()
    assert (s.final_h != s.final_a).all()                       # someone always wins
    assert ((s.home_win) == (s.final_h > s.final_a)).all()
    assert (s.went_so <= s.went_ot).all()
    assert ((s.reg_h == s.reg_a) == s.went_ot).all()
    # regular season: at most one OT goal, shootout winner gets +1 in the settlement score
    assert ((s.goals_h + s.goals_a) - (s.reg_h + s.reg_a) <= 1).all()
    assert (s.final_h + s.final_a - (s.goals_h + s.goals_a) == s.went_so.astype(int)).all()


def test_sog_and_goal_means_match_inputs():
    c = SimConstants(sog_mult_by_lead=[1] * 5, gp_mult_by_lead=[1] * 5, pull_prob={1: 0, 2: 0})
    s = run(c=c)
    reg_sog_h = s.sog_h[~s.went_ot].mean()
    assert reg_sog_h == pytest.approx(30.0, rel=0.03)
    assert s.reg_h.mean() == pytest.approx(30 * 0.095, rel=0.03)


def test_empty_net_for_both_teams():
    """The PDF's pseudocode only pulled the home goalie. Ours must pull whichever team trails."""
    c = SimConstants(pull_prob={1: 1.0, 2: 1.0}, pull_remaining_s={1: 300, 2: 300}, en_goal_rate=0.01)
    strong_home = GameInput(sog_h=40, sog_a=20, gp_h=0.12, gp_a=0.07)
    strong_away = GameInput(sog_h=20, sog_a=40, gp_h=0.07, gp_a=0.12)
    s1, s2 = run(strong_home, c), run(strong_away, c)
    assert s1.en_h.sum() > 0 and s2.en_a.sum() > 0      # leading team scores into empty net
    assert s1.en_h.mean() > s1.en_a.mean()
    assert s2.en_a.mean() > s2.en_h.mean()


def test_no_pulls_no_empty_net_goals():
    c = SimConstants(pull_prob={1: 0.0, 2: 0.0})
    s = run(c=c)
    assert s.en_h.sum() == 0 and s.en_a.sum() == 0


def test_playoff_overtime_has_no_shootout():
    gi = GameInput(sog_h=30, sog_a=30, gp_h=0.09, gp_a=0.09, game_type=3)
    s = run(gi)
    assert not s.went_so.any()
    assert (s.final_h != s.final_a).all()


def _fake(final_h, final_a, reg_h=None, reg_a=None):
    final_h, final_a = np.array(final_h), np.array(final_a)
    reg_h = final_h if reg_h is None else np.array(reg_h)
    reg_a = final_a if reg_a is None else np.array(reg_a)
    z = np.zeros(len(final_h), int)
    return SimResult(goals_h=final_h, goals_a=final_a, reg_h=reg_h, reg_a=reg_a, sog_h=z, sog_a=z, en_h=z, en_a=z,
                     went_ot=reg_h == reg_a, went_so=np.zeros(len(z), bool), home_win=final_h > final_a,
                     first_goal=z, final_h=final_h, final_a=final_a)


def test_settlement_edge_cases():
    # game 1: 3-2 home in OT (2-2 after regulation); game 2: 4-1 home; game 3: 1-3 away
    s = _fake([3, 4, 1], [2, 1, 3], reg_h=[2, 4, 1], reg_a=[2, 1, 3])
    assert M.h2h(s)["home"] == pytest.approx(2 / 3)
    r3 = M.regulation_3way(s)
    assert r3["draw"] == pytest.approx(1 / 3) and r3["home"] == pytest.approx(1 / 3)
    # home -1.5 covers only the 4-1 game; OT win by one does not cover
    assert M.spread(s, "home", -1.5) == (pytest.approx(1 / 3), 0.0)
    assert M.spread(s, "away", 1.5)[0] == pytest.approx(2 / 3)
    # totals 5.0: 5 (push), 5 (push), 4 (under)
    win, push = M.total(s, "over", 5.0)
    assert win == 0 and push == pytest.approx(2 / 3)
    assert M.no_push_prob(*M.total(s, "under", 5.0)) == pytest.approx(1.0)


def test_market_probs_complement():
    s = run()
    assert M.market_prob(s, "h2h", "home", None) + M.market_prob(s, "h2h", "away", None) == pytest.approx(1.0)
    assert M.market_prob(s, "totals", "over", 5.5) + M.market_prob(s, "totals", "under", 5.5) == pytest.approx(1.0)
    assert M.market_prob(s, "spreads", "home", -1.5) + M.market_prob(s, "spreads", "away", 1.5) == pytest.approx(1.0)
