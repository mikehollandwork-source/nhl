"""Odds conversions and vig removal.

Four standard de-vig methods. Each takes the book's implied probabilities for one market
(which sum to > 1 because of the margin) and returns fair probabilities that sum to 1.

- multiplicative: scale every outcome down by the same factor.
- additive: subtract the same amount from every outcome.
- power: raise to a common exponent k (removes more from longshots).
- shin: Shin (1993) insider-trading model; also removes more from longshots.

Which method best explains a book's prices is itself a "tell" (see market/tells.py).
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from scipy.optimize import brentq

METHODS = ("multiplicative", "additive", "power", "shin")


def american_to_decimal(a: float) -> float:
    a = float(a)
    if a == 0:
        raise ValueError("American odds cannot be 0")
    return 1 + (a / 100.0 if a > 0 else 100.0 / -a)


def decimal_to_american(d: float) -> float:
    if d <= 1:
        raise ValueError("decimal odds must be > 1")
    return round((d - 1) * 100) if d >= 2 else round(-100 / (d - 1))


def prob_to_american(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return decimal_to_american(1 / p)


def implied(prices_american: Sequence[float]) -> np.ndarray:
    return np.array([1 / american_to_decimal(a) for a in prices_american])


def overround(prices_american: Sequence[float]) -> float:
    return float(implied(prices_american).sum())


def multiplicative(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    return q / q.sum()


def additive(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    p = q - (q.sum() - 1) / len(q)
    if (p <= 0).any():  # additive can go negative on big longshots; fall back
        return multiplicative(q)
    return p


def power(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, float)
    if abs(q.sum() - 1) < 1e-12:
        return q.copy()
    k = brentq(lambda k: (q ** k).sum() - 1, 0.01, 100.0)
    return q ** k


def shin(q: np.ndarray) -> np.ndarray:
    """Shin's method. z is the estimated share of insider money."""
    q = np.asarray(q, float)
    s = q.sum()
    if abs(s - 1) < 1e-12:
        return q.copy()

    def probs(z: float) -> np.ndarray:
        return (np.sqrt(z ** 2 + 4 * (1 - z) * q ** 2 / s) - z) / (2 * (1 - z))

    z = brentq(lambda z: probs(z).sum() - 1, 0.0, 0.999)
    return probs(z)


def devig(prices_american: Sequence[float], method: str = "multiplicative") -> np.ndarray:
    q = implied(prices_american)
    fn = {"multiplicative": multiplicative, "additive": additive, "power": power, "shin": shin}[method]
    return fn(q)


def ev(prob: float, price_american: float) -> float:
    """Expected profit per 1 unit staked."""
    return prob * american_to_decimal(price_american) - 1


def log_loss(p: float, outcome: int) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    return -math.log(p if outcome else 1 - p)
