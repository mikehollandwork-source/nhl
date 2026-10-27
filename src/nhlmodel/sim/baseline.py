"""Bivariate Poisson baseline (PDF section 6).

P(X=x, Y=y) = exp(-(l1+l2+l3)) * l1^x/x! * l2^y/y! * sum_k C(x,k) C(y,k) k! (l3/(l1 l2))^k

l3 is the shared (covariance) component. Measured on 2021-26 data, the correlation between
the two teams' non-empty-net goals is small (0.00 to 0.08 by season), so l3 is near zero and
this baseline is close to independent Poisson. Kept for comparison and tests.
"""

from __future__ import annotations

from math import comb, exp, factorial

import numpy as np


def bivariate_poisson_pmf(x: int, y: int, l1: float, l2: float, l3: float) -> float:
    s = sum(comb(x, k) * comb(y, k) * factorial(k) * (l3 / (l1 * l2)) ** k for k in range(min(x, y) + 1))
    return exp(-(l1 + l2 + l3)) * l1 ** x / factorial(x) * l2 ** y / factorial(y) * s


def bivariate_matrix(l1: float, l2: float, l3: float, max_goals: int = 20) -> np.ndarray:
    return np.array([[bivariate_poisson_pmf(i, j, l1, l2, l3) for j in range(max_goals)] for i in range(max_goals)])
