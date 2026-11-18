"""Portfolio weight estimators: minimum variance and CVaR (Section 3.1/3.2).

The mean-variance program is solved with ``scipy.optimize.minimize`` (SLSQP);
no portfolio-optimization library (skfolio, pyportfolioopt) exposes the exact
Rockafellar-Uryasev CVaR linear program of Section 3.2, so that program is
implemented from scratch with ``cvxpy`` as a linear program, following the
formulation given in the paper.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

import cvxpy as cp

from src import config


def minimum_variance_weights(returns_window: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    """Long-only or long-short minimum-variance weights (Section 3.1).

    Solves ``min w' Sigma w`` subject to ``1'w = 1`` and ``lo <= w_i <= hi``,
    where ``Sigma`` is the sample covariance matrix of ``returns_window``
    (T x N, T observations, N assets).
    """
    T, N = returns_window.shape
    sigma = np.atleast_2d(np.cov(returns_window, rowvar=False))
    w0 = np.full(N, 1.0 / N)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
    result = minimize(
        lambda w: w @ sigma @ w,
        w0,
        jac=lambda w: 2 * sigma @ w,
        method="SLSQP",
        bounds=[bounds] * N,
        constraints=constraints,
        options={"maxiter": config.SLSQP_MAXITER, "ftol": config.SLSQP_TOL},
    )
    if not result.success:
        raise RuntimeError(f"MVP optimization failed to converge: {result.message}")
    w = result.x
    w = w / w.sum()  # renormalize against small equality-constraint slack
    return w


def cvar_weights(returns_window: np.ndarray, alpha: float, bounds: tuple[float, float]) -> np.ndarray:
    """CVaR-minimizing weights via the Rockafellar-Uryasev linear program (Section 3.2).

    ``min_{w,nu} nu + (1/((1-alpha)*T)) * sum_t max(0, -w'r_t - nu)``
    subject to ``1'w = 1`` and ``lo <= w_i <= hi``.
    """
    T, N = returns_window.shape
    w = cp.Variable(N)
    nu = cp.Variable()
    losses = -returns_window @ w - nu
    cvar_term = nu + cp.sum(cp.pos(losses)) / ((1 - alpha) * T)
    constraints = [cp.sum(w) == 1, w >= bounds[0], w <= bounds[1]]
    problem = cp.Problem(cp.Minimize(cvar_term), constraints)
    problem.solve(solver=config.CVXPY_SOLVER)
    if w.value is None:
        raise RuntimeError(f"CVaR optimization failed: status={problem.status}")
    return np.asarray(w.value).flatten()


def equal_weights(n_assets: int) -> np.ndarray:
    """Equally weighted portfolio w_i = 1/N (Section 3.3)."""
    return np.full(n_assets, 1.0 / n_assets)
