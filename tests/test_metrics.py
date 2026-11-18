"""Unit tests for src/metrics.py."""

import numpy as np
import pandas as pd
import pytest

from src import metrics


def test_sharpe_ratio_sign_and_scale():
    excess = np.array([0.01, 0.02, -0.01, 0.015, -0.005])
    sharpe = metrics.sharpe_ratio(excess)
    assert np.isclose(sharpe, np.mean(excess) / np.std(excess, ddof=1))
    # All-positive excess returns must give a positive Sharpe ratio
    assert metrics.sharpe_ratio(np.array([0.01, 0.02, 0.015])) > 0
    # All-negative excess returns must give a negative Sharpe ratio
    assert metrics.sharpe_ratio(np.array([-0.01, -0.02, -0.015])) < 0


def test_sortino_ratio_only_penalizes_downside():
    excess_upside_only = np.array([0.01, 0.02, 0.03])
    # with no downside observations, DSD = 0, Sortino -> +inf; guard separately
    excess_mixed = np.array([0.01, -0.02, 0.015, -0.03])
    sortino = metrics.sortino_ratio(excess_mixed)
    downside = np.minimum(0.0, excess_mixed)
    dsd = np.sqrt(np.mean(downside ** 2))
    assert np.isclose(sortino, np.mean(excess_mixed) / dsd)


def test_rachev_ratio_symmetric_distribution_near_one():
    rng = np.random.default_rng(0)
    symmetric = rng.normal(0, 1, size=100000)
    rachev = metrics.rachev_ratio(symmetric, beta=0.95, gamma=0.95)
    assert 0.9 < rachev < 1.1


def test_max_drawdown_known_path():
    wealth = np.array([100, 110, 90, 95, 120, 80])
    mdd = metrics.max_drawdown_pct(wealth)
    # running max before the trough of 80 is 120 -> drawdown = (120-80)/120*100
    assert np.isclose(mdd, (120 - 80) / 120 * 100)


def test_var_cvar_positive_loss_convention():
    rng = np.random.default_rng(2)
    returns = rng.normal(0.0005, 0.01, size=5000)
    var95 = metrics.value_at_risk(returns, 0.95)
    var99 = metrics.value_at_risk(returns, 0.99)
    cvar95 = metrics.conditional_value_at_risk(returns, 0.95)
    cvar99 = metrics.conditional_value_at_risk(returns, 0.99)
    # CVaR must be at least as large as VaR at the same confidence level (it
    # averages losses beyond the VaR threshold)
    assert cvar95 >= var95
    assert cvar99 >= var99
    # A higher confidence level must give a larger (more conservative) VaR/CVaR
    assert var99 >= var95
    assert cvar99 >= cvar95


def test_gross_and_short_exposure():
    w_long_only = np.array([0.5, 0.3, 0.2])
    assert np.isclose(metrics.gross_exposure(w_long_only), 1.0)
    assert np.isclose(metrics.short_exposure(w_long_only), 0.0)

    w_long_short = np.array([1.2, -0.3, 0.1])
    assert np.isclose(metrics.gross_exposure(w_long_short), 1.6)
    assert np.isclose(metrics.short_exposure(w_long_short), 0.3)


def test_turnover_zero_when_weights_match_drift():
    w_prev = np.array([0.5, 0.5])
    r_prev = np.array([0.01, -0.01])
    port_r_prev = float(w_prev @ r_prev)
    drifted = w_prev * (1 + r_prev) / (1 + port_r_prev)
    turnover = metrics.daily_turnover(drifted, w_prev, r_prev, port_r_prev)
    assert np.isclose(turnover, 0.0, atol=1e-10)


def test_performance_summary_keys_and_terminal_wealth():
    dates = pd.bdate_range("2023-01-02", periods=100)
    returns = pd.Series(np.full(100, 0.0005), index=dates)
    rf = pd.Series(0.0, index=dates)
    summary = metrics.performance_summary(returns, rf, initial_wealth=100.0)
    expected_keys = {
        "sharpe", "sortino", "rachev", "daily_volatility", "max_drawdown_pct",
        "var_95", "var_99", "cvar_95", "cvar_99", "terminal_wealth",
    }
    assert expected_keys == set(summary.keys())
    assert np.isclose(summary["terminal_wealth"], 100.0 * (1.0005) ** 100)
    assert summary["max_drawdown_pct"] == 0.0  # strictly increasing wealth path
