"""Rolling-window portfolio construction for the seven strategies (Section 3).

For each out-of-sample date t, weights are estimated from the preceding
``ROLLING_WINDOW`` daily returns (t-504 .. t-1) and applied to the returns
realized on date t (Section 3, paragraph on the rolling-window procedure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config, metrics
from src.portfolio_optimization import cvar_weights, equal_weights, minimum_variance_weights

STRATEGY_SPECS = {
    "LO-MVP": {"kind": "mvp", "bounds": config.LONG_ONLY_BOUNDS},
    "LO-C95": {"kind": "cvar", "alpha": config.CVAR_ALPHA_95, "bounds": config.LONG_ONLY_BOUNDS},
    "LO-C99": {"kind": "cvar", "alpha": config.CVAR_ALPHA_99, "bounds": config.LONG_ONLY_BOUNDS},
    "LS-MVP": {"kind": "mvp", "bounds": config.LONG_SHORT_BOUNDS},
    "LS-C95": {"kind": "cvar", "alpha": config.CVAR_ALPHA_95, "bounds": config.LONG_SHORT_BOUNDS},
    "LS-C99": {"kind": "cvar", "alpha": config.CVAR_ALPHA_99, "bounds": config.LONG_SHORT_BOUNDS},
    "EWP": {"kind": "ewp"},
}


def _estimate_weights(spec: dict, window_returns: np.ndarray, n_assets: int) -> np.ndarray:
    if spec["kind"] == "ewp":
        return equal_weights(n_assets)
    if spec["kind"] == "mvp":
        return minimum_variance_weights(window_returns, spec["bounds"])
    if spec["kind"] == "cvar":
        return cvar_weights(window_returns, spec["alpha"], spec["bounds"])
    raise ValueError(f"unknown strategy kind {spec['kind']}")


def run_rolling_backtest(
    returns: pd.DataFrame,
    window: int = config.ROLLING_WINDOW,
    strategies: dict | None = None,
) -> dict[str, dict[str, pd.DataFrame | pd.Series]]:
    """Run the rolling-window backtest for every strategy in ``strategies``.

    Parameters
    ----------
    returns : DataFrame (T x N)
        Daily simple returns of the N assets, date-indexed, no missing values.
    window : int
        Estimation window length (Section 3): ``ROLLING_WINDOW`` = 504.
    strategies : dict
        Mapping strategy name -> spec; defaults to :data:`STRATEGY_SPECS`.

    Returns
    -------
    dict mapping strategy name to a dict with keys:
        ``weights`` (DataFrame, out-of-sample dates x assets),
        ``returns`` (Series, out-of-sample portfolio returns),
        ``gross_exposure`` (Series), ``short_exposure`` (Series),
        ``turnover`` (Series).
    """
    strategies = strategies or STRATEGY_SPECS
    assets = returns.columns.to_list()
    n_assets = len(assets)
    values = returns.to_numpy()
    dates = returns.index
    T = len(dates)
    out_of_sample_dates = dates[window:]

    results: dict[str, dict] = {}
    for name, spec in strategies.items():
        weight_rows = []
        port_returns = []
        gross = []
        short = []
        turnover = []
        prev_weights = None
        prev_port_return = None
        for t in range(window, T):
            window_returns = values[t - window:t, :]
            w = _estimate_weights(spec, window_returns, n_assets)
            r_t = values[t, :]
            port_r = float(np.dot(w, r_t))
            weight_rows.append(w)
            port_returns.append(port_r)
            gross.append(metrics.gross_exposure(w))
            short.append(metrics.short_exposure(w))
            if prev_weights is None:
                turnover.append(np.nan)  # no pre-existing portfolio on the first day
            else:
                r_tm1 = values[t - 1, :]
                turnover.append(
                    metrics.daily_turnover(w, prev_weights, r_tm1, prev_port_return)
                )
            prev_weights = w
            prev_port_return = port_r

        weights_df = pd.DataFrame(weight_rows, index=out_of_sample_dates, columns=assets)
        returns_sr = pd.Series(port_returns, index=out_of_sample_dates, name=name)
        results[name] = {
            "weights": weights_df,
            "returns": returns_sr,
            "gross_exposure": pd.Series(gross, index=out_of_sample_dates, name=name),
            "short_exposure": pd.Series(short, index=out_of_sample_dates, name=name),
            "turnover": pd.Series(turnover, index=out_of_sample_dates, name=name),
        }
    return results


def portfolio_wealth(port_returns: pd.Series, initial_wealth: float = config.INITIAL_WEALTH) -> pd.Series:
    """Cumulative wealth V_t = V_{t-1}(1+R_t), V_0 given (Section 3.5)."""
    return initial_wealth * (1 + port_returns).cumprod()


def asset_simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily simple returns R_{k,t} = P_{k,t}/P_{k,t-1} - 1 for every price column
    (Section 2.2); the first date has no return and is dropped."""
    return prices.pct_change().iloc[1:]


def run_full_pipeline(
    use_cache: bool = True, strategies: dict | None = None
) -> dict[str, object]:
    """The one shared data-to-backtest path used by every experiment script.

    Loads the 30-ETF/DJIA-proxy USD price panel and the FRED risk-free rate
    (``data.data_loader.load_etf_prices_usd``), builds daily simple returns
    (Section 2.2), and runs the Section 3 rolling-window backtest for the
    seven strategies in :data:`STRATEGY_SPECS` (or ``strategies`` if given).

    Returns a dict with keys:
        ``prices`` (DataFrame, USD-converted levels incl. the DJIA proxy),
        ``returns`` (DataFrame, 30-ETF daily simple returns),
        ``djia_returns`` (Series, DJIA-proxy daily simple returns, aligned to
            the out-of-sample dates of the backtest),
        ``risk_free_daily`` (Series),
        ``results`` (dict strategy -> {weights, returns, gross_exposure,
            short_exposure, turnover}, as returned by :func:`run_rolling_backtest`),
        ``meta`` (dict, data-loading metadata).
    """
    from data.data_loader import load_etf_prices_usd

    prices, risk_free_daily, meta = load_etf_prices_usd(use_cache=use_cache)
    all_returns = asset_simple_returns(prices)
    returns = all_returns[config.ALL_TICKERS]
    results = run_rolling_backtest(returns, strategies=strategies)
    out_of_sample_dates = next(iter(results.values()))["returns"].index
    djia_returns = all_returns[config.DJIA_PROXY_TICKER].reindex(out_of_sample_dates)
    return {
        "prices": prices,
        "returns": returns,
        "djia_returns": djia_returns,
        "risk_free_daily": risk_free_daily,
        "results": results,
        "meta": meta,
    }
