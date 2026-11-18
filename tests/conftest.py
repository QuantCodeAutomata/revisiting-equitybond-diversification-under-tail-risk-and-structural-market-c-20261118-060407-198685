"""Shared pytest fixtures: synthetic return panels and a cached market-data slice."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: tests that load a reduced slice of real market data"
    )


@pytest.fixture
def synthetic_returns() -> pd.DataFrame:
    """A small synthetic daily-return panel: 60 assets x 700 days (reproducible)."""
    rng = np.random.default_rng(0)
    n_days, n_assets = 700, 8
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    cov = 0.0001 * (0.3 * np.ones((n_assets, n_assets)) + 0.7 * np.eye(n_assets))
    returns = rng.multivariate_normal(np.full(n_assets, 0.0002), cov, size=n_days)
    return pd.DataFrame(returns, index=dates, columns=[f"A{i}" for i in range(n_assets)])


@pytest.fixture
def synthetic_risk_free(synthetic_returns) -> pd.Series:
    return pd.Series(0.00005, index=synthetic_returns.index, name="risk_free_daily")


@pytest.fixture(scope="session")
def market_data_slice(pytestconfig):
    """A reduced real-data slice (5 ETFs, ~2.5 years) loaded and cached once per session."""
    if not os.environ.get("DATALOADER_API_KEY"):
        pytest.skip("DATALOADER_API_KEY not set: market-data tests are skipped")
    cache_dir = pytestconfig.cache.mkdir("market_data_slice")
    prices_path = cache_dir / "prices.csv"
    rf_path = cache_dir / "risk_free.csv"
    if prices_path.exists() and rf_path.exists():
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        rf = pd.read_csv(rf_path, index_col=0, parse_dates=True).iloc[:, 0]
        return prices, rf

    from data.data_loader import load_ohlcv, load_fred_series
    from src import config as cfg

    tickers = ["AOA", "AOK", "VBAL.TO", "VDBA.AX", cfg.DJIA_PROXY_TICKER]
    bars, _ = load_ohlcv(tickers, "2022-01-03", "2024-06-28", "1d", identity_date="2024-06-28")
    raw = bars.pivot(index="date", columns="ticker", values="close").sort_index()
    fred = load_fred_series(
        [cfg.FX_CAD_PER_USD_SERIES, cfg.FX_USD_PER_AUD_SERIES, cfg.RISK_FREE_SERIES],
        "2022-01-03", "2024-06-28",
    ).reindex(raw.index).ffill()
    usd = raw.copy()
    usd["VBAL.TO"] = raw["VBAL.TO"] / fred[cfg.FX_CAD_PER_USD_SERIES]
    usd["VDBA.AX"] = raw["VDBA.AX"] * fred[cfg.FX_USD_PER_AUD_SERIES]
    usd = usd.ffill().dropna()
    rf_daily = (1 + fred[cfg.RISK_FREE_SERIES] / 100) ** (1 / cfg.TRADING_DAYS_PER_YEAR) - 1
    rf_daily = rf_daily.reindex(usd.index)

    usd.to_csv(prices_path)
    rf_daily.to_csv(rf_path)
    return usd, rf_daily
