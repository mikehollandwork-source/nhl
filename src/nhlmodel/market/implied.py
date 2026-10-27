"""Back out a book's implied scoring rates from its own prices.

Assumption under test (from the PDF): books price from two Poisson-like scoring rates.
Given the book's de-vigged home moneyline probability and its de-vigged P(over L), solve for
(lambda_home, lambda_away) under independent Poisson regulation scoring, with:
  - ties going to OT/SO, won by home with prob lambda_home / (lambda_home + lambda_away);
  - the settlement total adding 1 goal to tied games (OT goal or shootout winner).
With the book's own rates in hand we can price its OTHER markets (puck line, alt totals,
3-way) and look for internal inconsistencies.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import poisson

MAXG = 20


def _matrix(lh: float, la: float) -> np.ndarray:
    k = np.arange(MAXG)
    return np.outer(poisson.pmf(k, lh), poisson.pmf(k, la))


def probs_from_rates(lh: float, la: float) -> dict:
    m = _matrix(lh, la)
    i, j = np.indices(m.shape)
    tie = np.trace(m)
    share = lh / (lh + la)
    total = np.where(i == j, i + j + 1, i + j)
    final_margin = np.where(i == j, 0, i - j)  # ties resolved below
    out = {
        "p_home": float(m[i > j].sum() + tie * share),
        "p_reg_home": float(m[i > j].sum()), "p_reg_draw": float(tie), "p_reg_away": float(m[i < j].sum()),
        "_m": m, "_total": total, "_margin": final_margin, "_tie": tie, "_share": share,
    }
    return out


def p_over(lh: float, la: float, line: float) -> float:
    d = probs_from_rates(lh, la)
    m, total = d["_m"], d["_total"]
    win = m[total > line].sum()
    push = m[total == line].sum()
    return float(win / (1 - push)) if push < 1 else 0.5


def p_home_spread(lh: float, la: float, line: float) -> float:
    """P(home covers `line`) with OT/SO winner credited +1 (ties: home wins with prob share)."""
    d = probs_from_rates(lh, la)
    m, margin = d["_m"], d["_margin"]
    i, j = np.indices(m.shape)
    non_tie = i != j
    win = m[non_tie & (margin + line > 0)].sum()
    push = m[non_tie & (margin + line == 0)].sum()
    tie = d["_tie"]
    # tied after regulation: final margin is +1 (home win) or -1 (away win)
    win += tie * (d["_share"] * (1 + line > 0) + (1 - d["_share"]) * (-1 + line > 0))
    push += tie * (d["_share"] * (1 + line == 0) + (1 - d["_share"]) * (-1 + line == 0))
    return float(win / (1 - push)) if push < 1 else 0.5


def solve_rates(p_home: float, line: float, p_over_line: float) -> tuple[float, float, float]:
    """(lambda_home, lambda_away, residual) matching the book's ML and total."""
    def f(x):
        lh, la = np.exp(x)
        return [probs_from_rates(lh, la)["p_home"] - p_home, p_over(lh, la, line) - p_over_line]
    res = least_squares(f, x0=np.log([3.1, 2.9]), bounds=(np.log(0.5), np.log(8.0)))
    lh, la = np.exp(res.x)
    return float(lh), float(la), float(np.abs(res.fun).max())
