"""Unit tests for src/backtest.py on a small synthetic panel (reduced window)."""

import numpy as np
import pandas as pd
import pytest

from src.backtest import STRATEGY_SPECS, portfolio_wealth, run_rolling_backtest


SMALL_STRATEGIES = {
    "LO-MVP": STRATEGY_SPECS["LO-MVP"],
    "LS-MVP": STRATEGY_SPECS["LS-MVP"],
    "LO-C95": STRATEGY_SPECS["LO-C95"],
    "EWP": STRATEGY_SPECS["EWP"],
}


def test_run_rolling_backtest_shapes_and_no_lookahead(synthetic_returns):
    window = 100
    results = run_rolling_backtest(synthetic_returns, window=window, strategies=SMALL_STRATEGIES)
    n_oos = len(synthetic_returns) - window
    for name, res in results.items():
        assert res["weights"].shape == (n_oos, synthetic_returns.shape[1])
        assert len(res["returns"]) == n_oos
        # weights index must equal the out-of-sample dates, strictly after the window
        assert res["weights"].index[0] == synthetic_returns.index[window]
        assert res["weights"].index[-1] == synthetic_returns.index[-1]


def test_ewp_weights_are_constant_and_equal(synthetic_returns):
    results = run_rolling_backtest(synthetic_returns, window=100, strategies={"EWP": STRATEGY_SPECS["EWP"]})
    weights = results["EWP"]["weights"]
    n_assets = synthetic_returns.shape[1]
    assert np.allclose(weights.to_numpy(), 1.0 / n_assets)


def test_turnover_first_day_is_nan_then_finite(synthetic_returns):
    results = run_rolling_backtest(synthetic_returns, window=100, strategies={"LO-MVP": STRATEGY_SPECS["LO-MVP"]})
    turnover = results["LO-MVP"]["turnover"]
    assert np.isnan(turnover.iloc[0])
    assert np.isfinite(turnover.iloc[1:]).all()


def test_weight_estimation_uses_only_past_returns(synthetic_returns):
    """Perturbing a single future return must not change weights estimated
    for an earlier out-of-sample date (no look-ahead)."""
    window = 100
    results_a = run_rolling_backtest(
        synthetic_returns, window=window, strategies={"LO-MVP": STRATEGY_SPECS["LO-MVP"]}
    )
    perturbed = synthetic_returns.copy()
    perturbed.iloc[-1, 0] += 10.0  # drastic change to the very last observation
    results_b = run_rolling_backtest(
        perturbed, window=window, strategies={"LO-MVP": STRATEGY_SPECS["LO-MVP"]}
    )
    w_a = results_a["LO-MVP"]["weights"].iloc[:-1]
    w_b = results_b["LO-MVP"]["weights"].iloc[:-1]
    assert np.allclose(w_a.to_numpy(), w_b.to_numpy(), atol=1e-6)


def test_portfolio_wealth_compounding():
    returns = pd.Series([0.01, -0.01, 0.02])
    wealth = portfolio_wealth(returns, initial_wealth=100.0)
    expected = 100.0 * np.cumprod(1 + returns.to_numpy())
    assert np.allclose(wealth.to_numpy(), expected)
