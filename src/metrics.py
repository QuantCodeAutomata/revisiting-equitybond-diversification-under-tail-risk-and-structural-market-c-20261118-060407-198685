"""Out-of-sample risk-return performance metrics (Section 4.1/4.2).

Formulas follow the paper exactly; no library offers the Rachev ratio or the
paper's empirical VaR/CVaR convention, so these are implemented from scratch.
Standard moments use ``scipy.stats``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def sharpe_ratio(excess_returns: np.ndarray) -> float:
    """Sharpe ratio = mean(excess returns) / std(excess returns) (Section 4.1)."""
    return float(np.mean(excess_returns) / np.std(excess_returns, ddof=1))


def sortino_ratio(excess_returns: np.ndarray) -> float:
    """Sortino ratio = mean(excess returns) / downside semi-deviation (Section 4.1)."""
    downside = np.minimum(0.0, excess_returns)
    dsd = np.sqrt(np.mean(downside ** 2))
    return float(np.mean(excess_returns) / dsd)


def rachev_ratio(excess_returns: np.ndarray, beta: float = 0.95, gamma: float = 0.95) -> float:
    """Rachev ratio with beta=gamma=0.95 (Section 4.1):

    mean of the top (1-beta) fraction of excess returns divided by the
    absolute mean of the bottom (1-gamma) fraction.
    """
    sorted_returns = np.sort(excess_returns)
    n = len(sorted_returns)
    n_tail = max(1, int(round((1 - beta) * n)))
    top_mean = np.mean(sorted_returns[-n_tail:])
    bottom_mean = np.mean(sorted_returns[:n_tail])
    return float(top_mean / abs(bottom_mean))


def max_drawdown_pct(wealth: np.ndarray) -> float:
    """Maximum drawdown in percent: 100 * max((M_t - V_t)/M_t) (Section 4.2)."""
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    return float(100 * np.max(drawdown))


def value_at_risk(returns: np.ndarray, level: float) -> float:
    """Empirical VaR at confidence ``level``, reported as a positive loss measure.

    VaR_alpha = -F^{-1}(1-alpha); e.g. level=0.95 uses the 5th percentile.
    """
    pct = 100 * (1 - level)
    return float(-np.percentile(returns, pct))


def conditional_value_at_risk(returns: np.ndarray, level: float) -> float:
    """Empirical CVaR at confidence ``level``, reported as a positive loss measure.

    CVaR_alpha = -(1/(1-alpha)) * E[r | r <= -VaR_alpha] (Section 4.2).
    """
    pct = 100 * (1 - level)
    threshold = np.percentile(returns, pct)
    tail = returns[returns <= threshold]
    return float(-np.mean(tail))


def daily_volatility(returns: np.ndarray) -> float:
    """Standard deviation of daily portfolio returns (Section 4.2)."""
    return float(np.std(returns, ddof=1))


def gross_exposure(weights: np.ndarray) -> float:
    """Gross exposure G_t = sum |w_i| (Section 3.6)."""
    return float(np.sum(np.abs(weights)))


def short_exposure(weights: np.ndarray) -> float:
    """Short exposure S_t = sum max(-w_i, 0) (Section 3.6)."""
    return float(np.sum(np.maximum(-weights, 0.0)))


def daily_turnover(
    weights_t: np.ndarray, weights_tm1: np.ndarray, asset_returns_tm1: np.ndarray, port_return_tm1: float
) -> float:
    """Daily turnover T_t = (1/2) sum |w_t - drift(w_{t-1})| (Section 3.6).

    ``drift(w_{t-1})_k = w_{t-1,k} * (1+R_{k,t-1}) / (1+R_{p,t-1})`` is the
    pre-rebalancing weight resulting from asset-level drift between t-1 and t.
    """
    drifted = weights_tm1 * (1 + asset_returns_tm1) / (1 + port_return_tm1)
    return float(0.5 * np.sum(np.abs(weights_t - drifted)))


def performance_summary(
    port_returns: pd.Series, risk_free_daily: pd.Series, initial_wealth: float = 100.0
) -> dict:
    """Full set of Section 4.1/4.2 metrics for one out-of-sample return series."""
    rf = risk_free_daily.reindex(port_returns.index)
    excess = (port_returns - rf).to_numpy()
    returns = port_returns.to_numpy()
    wealth = initial_wealth * np.cumprod(1 + returns)
    return {
        "sharpe": sharpe_ratio(excess),
        "sortino": sortino_ratio(excess),
        "rachev": rachev_ratio(excess),
        "daily_volatility": daily_volatility(returns),
        "max_drawdown_pct": max_drawdown_pct(wealth),
        "var_95": value_at_risk(returns, 0.95),
        "var_99": value_at_risk(returns, 0.99),
        "cvar_95": conditional_value_at_risk(returns, 0.95),
        "cvar_99": conditional_value_at_risk(returns, 0.99),
        "terminal_wealth": float(wealth[-1]),
    }
