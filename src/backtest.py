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
