"""Turn simulated games into market probabilities.

Settlement rules (standard US books, The Odds API NHL markets):
- h2h (moneyline): 2-way, includes OT and shootout.
- h2h_3_way / regulation: home / draw / away after 60 minutes.
- spreads (puck line): final score incl. OT, shootout winner credited +1 goal.
- totals: same final score as puck line (shootout winner +1).
Integer lines can push; probabilities returned are (win, push, lose) where needed.
"""

from __future__ import annotations

import numpy as np

from nhlmodel.sim.engine import SimResult


def h2h(sim: SimResult) -> dict[str, float]:
    p = float(sim.home_win.mean())
    return {"home": p, "away": 1 - p}


def regulation_3way(sim: SimResult) -> dict[str, float]:
    return {"home": float((sim.reg_h > sim.reg_a).mean()),
            "draw": float((sim.reg_h == sim.reg_a).mean()),
            "away": float((sim.reg_h < sim.reg_a).mean())}


def spread(sim: SimResult, side: str, line: float) -> tuple[float, float]:
    """(P(win), P(push)) for `side` ('home'/'away') getting `line` goals (e.g. -1.5, +1.5)."""
    margin = (sim.final_h - sim.final_a) if side == "home" else (sim.final_a - sim.final_h)
    adj = margin + line
    return float((adj > 0).mean()), float((adj == 0).mean())


def total(sim: SimResult, side: str, line: float) -> tuple[float, float]:
    """(P(win), P(push)) for 'over'/'under' the line on the settlement total."""
    t = sim.final_h + sim.final_a
    win = (t > line) if side == "over" else (t < line)
    return float(win.mean()), float((t == line).mean())


def team_total(sim: SimResult, team: str, side: str, line: float) -> tuple[float, float]:
    t = sim.final_h if team == "home" else sim.final_a
    win = (t > line) if side == "over" else (t < line)
    return float(win.mean()), float((t == line).mean())


def no_push_prob(win: float, push: float) -> float:
    """Probability of winning given the bet does not push (how books price integer lines)."""
    return win / (1 - push) if push < 1 else 0.5


def market_prob(sim: SimResult, market: str, selection: str, line: float | None) -> float:
    """Model probability for a book outcome, conditioned on no push. selection: home/away/over/under/draw."""
    if market == "h2h":
        return h2h(sim)[selection]
    if market in ("h2h_3_way", "regulation"):
        return regulation_3way(sim)[selection]
    if market == "spreads":
        return no_push_prob(*spread(sim, selection, float(line)))
    if market == "totals":
        return no_push_prob(*total(sim, selection, float(line)))
    raise ValueError(f"unsupported market {market}")


def summary(sim: SimResult) -> dict[str, float]:
    """Headline numbers for a game card."""
    ml = h2h(sim)
    r3 = regulation_3way(sim)
    pl_h, _ = spread(sim, "home", -1.5)
    pl_a, _ = spread(sim, "away", -1.5)
    t_tot = sim.final_h + sim.final_a
    return {
        "exp_goals_home": float(sim.final_h.mean()), "exp_goals_away": float(sim.final_a.mean()),
        "exp_sog_home": float(sim.sog_h.mean()), "exp_sog_away": float(sim.sog_a.mean()),
        "p_home": ml["home"], "p_away": ml["away"],
        "p_reg_home": r3["home"], "p_reg_draw": r3["draw"], "p_reg_away": r3["away"],
        "p_ot": float(sim.went_ot.mean()), "p_so": float(sim.went_so.mean()),
        "p_home_m15": pl_h, "p_away_m15": pl_a,
        "p_over_5_5": float((t_tot > 5.5).mean()), "p_over_6_5": float((t_tot > 6.5).mean()),
        "median_total": float(np.median(t_tot)),
    }


def score_matrix(sim: SimResult, max_goals: int = 8) -> np.ndarray:
    """P(final_h = i, final_a = j), capped at max_goals."""
    h = np.minimum(sim.final_h, max_goals)
    a = np.minimum(sim.final_a, max_goals)
    m = np.zeros((max_goals + 1, max_goals + 1))
    np.add.at(m, (h, a), 1)
    return m / sim.n
