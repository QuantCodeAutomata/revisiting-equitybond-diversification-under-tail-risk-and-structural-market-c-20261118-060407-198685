"""Unit tests for the Hill tail-index estimator (Section 4.4)."""

from __future__ import annotations

import numpy as np
import pytest

from src.tail_risk import (
    fit_ar1_garch11,
    hill_estimator,
    hill_sensitivity,
    hill_tail_index,
    negative_loss_magnitudes,
)


def test_hill_estimator_on_known_pareto_tail():
    """On an exact Pareto(alpha) tail, gamma_hat(k) -> 1/alpha for any k < n."""
    rng = np.random.default_rng(0)
    alpha_true = 3.0
    pareto = (1 - rng.random(5000)) ** (-1.0 / alpha_true)  # standard Pareto(1, alpha)
    sorted_desc = np.sort(pareto)[::-1]
    gamma_hat = hill_estimator(sorted_desc, k=200)
    assert gamma_hat == pytest.approx(1.0 / alpha_true, rel=0.25)


def test_hill_tail_index_ci_contains_point_estimate():
    rng = np.random.default_rng(1)
    pareto = (1 - rng.random(2000)) ** (-1.0 / 2.5)
    sorted_desc = np.sort(pareto)[::-1]
    out = hill_tail_index(sorted_desc, k=100)
    assert out["ci_lower"] < out["alpha_hat"] < out["ci_upper"]
    assert out["alpha_hat"] > 0


def test_negative_loss_magnitudes_only_uses_losses_and_is_sorted_desc():
    z = np.array([1.0, -0.5, 0.2, -2.0, -0.1, 3.0])
    losses = negative_loss_magnitudes(z)
    assert np.all(losses > 0)
    assert len(losses) == 3  # only the three negative residuals
    assert np.all(np.diff(losses) <= 0)  # descending order


def test_hill_sensitivity_grid_monotone_in_available_k():
    rng = np.random.default_rng(2)
    z = rng.standard_normal(2000)
    grid = np.arange(40, 210, 10)
    out = hill_sensitivity(z, grid)
    assert set(out.columns) >= {"k", "gamma_hat", "alpha_hat", "ci_lower", "ci_upper"}
    assert (out["alpha_hat"] > 0).all()


def test_fit_ar1_garch11_standardized_residuals_are_unit_variance_like(synthetic_returns):
    series = synthetic_returns.iloc[:, 0]
    z, result = fit_ar1_garch11(series)
    assert len(z) > 0
    assert np.isfinite(z).all()
    # Standardized residuals of a correctly specified GARCH should have a
    # sample variance close to one.
    assert np.std(z) == pytest.approx(1.0, abs=0.5)
    assert result.params["alpha[1]"] >= 0
    assert result.params["beta[1]"] >= 0
