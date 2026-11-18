"""Regression tests for the NIG Levy option-pricing module (paper Section 6)."""

import numpy as np

from src.nig_pricing import no_arbitrage_violations


def test_no_arbitrage_violations_detects_monotonicity_break():
    """A call-price curve that is not non-increasing in strike must be flagged."""
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    s0, r_daily, maturity_days, tol = 100.0, 0.0001, 30, 1e-8
    decreasing_calls = np.array([20.0, 11.0, 5.0, 2.0, 0.5])
    clean = no_arbitrage_violations(strikes, decreasing_calls, s0, r_daily, maturity_days, tol)
    assert clean["n_monotonicity_violations"] == 0

    broken_calls = decreasing_calls.copy()
    broken_calls[2] = 12.0  # introduces a local increase in K
    broken = no_arbitrage_violations(strikes, broken_calls, s0, r_daily, maturity_days, tol)
    assert broken["n_monotonicity_violations"] >= 1


def test_no_arbitrage_violations_respects_lower_and_upper_bounds():
    strikes = np.array([80.0, 100.0, 120.0])
    s0, r_daily, maturity_days, tol = 100.0, 0.0, 30, 1e-8
    # Deep ITM call priced below its intrinsic lower bound: must be flagged.
    calls = np.array([5.0, 3.0, 1.0])
    result = no_arbitrage_violations(strikes, calls, s0, r_daily, maturity_days, tol)
    assert result["n_lower_violations"] >= 1
    assert result["n_upper_violations"] == 0
