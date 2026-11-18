"""Tests of the real-data path through ``data/data_loader.py``.

These use the session-scoped ``market_data_slice`` fixture (five ETFs, a
reduced window) defined in ``tests/conftest.py``. They check structural
properties of the loaded panel (shape, ordering, currency conversion, no
look-ahead in the FX/rate alignment), not the paper's reported numbers,
which depend on the full 30-ETF, 2020-2024 sample.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.mark.market_data
def test_market_data_slice_shape_and_tickers(market_data_slice):
    prices, risk_free = market_data_slice
    expected_tickers = {"AOA", "AOK", "VBAL.TO", "VDBA.AX"}
    assert expected_tickers.issubset(set(prices.columns))
    assert len(prices) > 300  # a 2.5-year daily slice has well over 300 rows
    assert prices.index.is_monotonic_increasing
    assert not prices.index.has_duplicates


@pytest.mark.market_data
def test_market_data_slice_prices_are_positive_and_filled(market_data_slice):
    prices, _ = market_data_slice
    assert (prices > 0).all().all()
    assert not prices.isna().any().any()  # forward-filled, as in src.config


@pytest.mark.market_data
def test_market_data_slice_risk_free_is_small_daily_rate(market_data_slice):
    """The daily-equivalent 3-month T-bill yield is a small positive rate,
    consistent with ``R_f^daily = (1+y/100)^(1/252) - 1`` (Section 2.2)."""
    _, risk_free = market_data_slice
    risk_free = risk_free.dropna()
    assert len(risk_free) > 0
    assert (risk_free > -0.001).all()
    assert (risk_free < 0.01).all()


@pytest.mark.market_data
def test_market_data_slice_no_lookahead_in_fx_alignment(market_data_slice):
    """The FX/rate series are forward-filled onto the price index (most
    recent preceding observation), never filled from a later date."""
    from data.data_loader import load_fred_series
    from src import config as cfg

    prices, _ = market_data_slice
    fred = load_fred_series(
        [cfg.FX_CAD_PER_USD_SERIES], "2022-01-03", "2024-06-28"
    ).reindex(prices.index)
    raw_known_dates = fred[cfg.FX_CAD_PER_USD_SERIES].dropna().index
    filled = fred[cfg.FX_CAD_PER_USD_SERIES].ffill()
    for date in prices.index[:: max(1, len(prices.index) // 20)]:
        preceding = raw_known_dates[raw_known_dates <= date]
        if len(preceding) == 0:
            continue
        last_known_date = preceding[-1]
        assert filled.loc[date] == pytest.approx(fred.loc[last_known_date, cfg.FX_CAD_PER_USD_SERIES])
