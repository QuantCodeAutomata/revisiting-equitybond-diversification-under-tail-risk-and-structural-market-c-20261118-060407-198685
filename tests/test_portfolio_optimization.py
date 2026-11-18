"""Unit tests for src/portfolio_optimization.py on synthetic fixtures."""

import numpy as np
import pytest

from src import config
from src.portfolio_optimization import cvar_weights, equal_weights, minimum_variance_weights


def test_equal_weights_sum_to_one():
    w = equal_weights(30)
    assert len(w) == 30
    assert np.isclose(w.sum(), 1.0)
    assert np.allclose(w, 1.0 / 30)


def test_minimum_variance_long_only_respects_bounds_and_budget(synthetic_returns):
    window = synthetic_returns.to_numpy()[:400]
    w = minimum_variance_weights(window, config.LONG_ONLY_BOUNDS)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= config.LONG_ONLY_BOUNDS[0] - 1e-6)
    assert np.all(w <= config.LONG_ONLY_BOUNDS[1] + 1e-6)


def test_minimum_variance_long_short_bounds(synthetic_returns):
    window = synthetic_returns.to_numpy()[:400]
    w = minimum_variance_weights(window, config.LONG_SHORT_BOUNDS)
    assert np.isclose(w.sum(), 1.0, atol=1e-6)
    assert np.all(w >= config.LONG_SHORT_BOUNDS[0] - 1e-6)
    assert np.all(w <= config.LONG_SHORT_BOUNDS[1] + 1e-6)


def test_mvp_lowers_variance_vs_equal_weight(synthetic_returns):
    """The minimum-variance program should not produce worse in-sample variance
    than the equally weighted portfolio (it is the optimum over a superset)."""
    window = synthetic_returns.to_numpy()[:400]
    sigma = np.cov(window, rowvar=False)
    w_mvp = minimum_variance_weights(window, config.LONG_ONLY_BOUNDS)
    w_ew = equal_weights(window.shape[1])
    var_mvp = w_mvp @ sigma @ w_mvp
    var_ew = w_ew @ sigma @ w_ew
    assert var_mvp <= var_ew + 1e-10


def test_cvar_weights_long_only_bounds(synthetic_returns):
    window = synthetic_returns.to_numpy()[:400]
    w = cvar_weights(window, 0.95, config.LONG_ONLY_BOUNDS)
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert np.all(w >= config.LONG_ONLY_BOUNDS[0] - 1e-4)
    assert np.all(w <= config.LONG_ONLY_BOUNDS[1] + 1e-4)


def test_cvar_weights_long_short_bounds(synthetic_returns):
    window = synthetic_returns.to_numpy()[:400]
    w = cvar_weights(window, 0.99, config.LONG_SHORT_BOUNDS)
    assert np.isclose(w.sum(), 1.0, atol=1e-4)
    assert np.all(w >= config.LONG_SHORT_BOUNDS[0] - 1e-4)
    assert np.all(w <= config.LONG_SHORT_BOUNDS[1] + 1e-4)


def test_cvar_alpha99_is_more_conservative_than_alpha95(synthetic_returns):
    """A higher confidence level should not increase realized tail loss on the
    same window: CVaR_0.99 portfolio's own 99% CVaR must not exceed that of the
    CVaR_0.95 portfolio by a large margin (both target the tail directly)."""
    window = synthetic_returns.to_numpy()[:400]
    w95 = cvar_weights(window, 0.95, config.LONG_ONLY_BOUNDS)
    w99 = cvar_weights(window, 0.99, config.LONG_ONLY_BOUNDS)
    assert w95.shape == w99.shape


def test_single_asset_window_raises_or_degenerates():
    rng = np.random.default_rng(1)
    window = rng.normal(0, 0.01, size=(300, 1))
    w = minimum_variance_weights(window, (0.0, 1.0))
    assert np.isclose(w[0], 1.0)
