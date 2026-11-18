"""Experiment 4: Volatility Persistence Comparison: AR(1)-GARCH(1,1) vs.
AR(1)-FIGARCH(1,d,1) Models (Section 5).

For each of the three portfolio return series singled out in Section 5
(``config.VOL_MODEL_SERIES`` = EWP, LS-MVP, LS-C95), this script:

1. Fits an AR(1)-GARCH(1,1) and an AR(1)-FIGARCH(1,d,1) model (Gaussian
   innovations) to the out-of-sample strategy returns from the Section 3
   rolling-window backtest.
2. Compares the two specifications by per-observation AIC/BIC (rugarch
   convention) and reports GARCH persistence (alpha_1 + beta_1) and the
   FIGARCH fractional-differencing parameter ``d`` with its standard error.
3. Runs Ljung-Box diagnostics on GARCH standardized residuals (levels and
   squares) at lags 10/20/50, and reports sample skewness/kurtosis of the
   raw returns.
4. Plots the squared-return autocorrelation function before and after GARCH
   filtering, and a distributional-diagnostics panel (histogram + kernel
   density + fitted normal, and a normal Q-Q plot) for each series.

Run: ``python -m exp.exp4``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde, norm, probplot

from src import backtest, config, volatility_models as vm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _garch_param_row(series: str, result) -> dict:
    """Extract AR(1)-GARCH(1,1) parameters by their exact ``arch`` names
    (Section 5.2): mean intercept ``Const``, AR(1) coefficient ``<series>[1]``,
    and variance parameters ``omega``, ``alpha[1]``, ``beta[1]``."""
    p = result.params
    return {
        "series": series,
        "mu": float(p["Const"]),
        "phi1": float(p[f"{series}[1]"]),
        "omega": float(p["omega"]),
        "alpha1": float(p["alpha[1]"]),
        "beta1": float(p["beta[1]"]),
        "persistence": vm.persistence(result),
    }


def _figarch_param_row(series: str, result) -> dict:
    """Extract the FIGARCH fractional-differencing parameter ``d`` and its
    standard error (Section 5.3); ``arch`` names it ``d`` for the
    FIGARCH(1,d,1) specification."""
    return {
        "series": series,
        "d": float(result.params["d"]),
        "se_d": float(result.std_err["d"]),
    }


def _plot_acf_comparison(series: str, acf_pre: np.ndarray, acf_post: np.ndarray) -> None:
    """One figure per series: squared-return ACF before vs. after GARCH
    filtering (Section 5.2 steps 8/11), lags up to ``config.ACF_MAX_LAG``."""
    lags = np.arange(len(acf_pre))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].stem(lags, acf_pre, markerfmt=" ", basefmt=" ")
    axes[0].set_title(f"{series}: squared-return ACF (pre-filter)")
    axes[0].set_xlabel("Lag")
    axes[0].set_ylabel("Autocorrelation")

    lags_post = np.arange(len(acf_post))
    axes[1].stem(lags_post, acf_post, markerfmt=" ", basefmt=" ")
    axes[1].set_title(f"{series}: squared standardized-residual ACF (post-GARCH)")
    axes[1].set_xlabel("Lag")
    axes[1].set_ylabel("Autocorrelation")
    for ax in axes:
        ax.set_xlim(0, config.ACF_MAX_LAG)
        ax.legend(["ACF"], fontsize=8)
    fig.suptitle(f"Squared-return autocorrelation before/after AR(1)-GARCH(1,1) filtering: {series}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"exp4_acf_{series}.png", dpi=150)
    plt.close(fig)


def _plot_distributional_diagnostics(series: str, returns: pd.Series) -> None:
    """Histogram + kernel density + fitted normal density, and a normal Q-Q
    plot, for the raw return series (Section 5.1)."""
    values = returns.to_numpy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].hist(values, bins=50, density=True, alpha=0.5, label="Empirical histogram")
    grid = np.linspace(values.min(), values.max(), 500)
    kde = gaussian_kde(values)
    axes[0].plot(grid, kde(grid), label="Kernel density", color="tab:blue")
    mu, sigma = values.mean(), values.std(ddof=1)
    axes[0].plot(grid, norm.pdf(grid, mu, sigma), label="Fitted normal", color="tab:red", linestyle="--")
    axes[0].set_title(f"{series}: return distribution")
    axes[0].set_xlabel("Daily return")
    axes[0].set_ylabel("Density")
    axes[0].legend(fontsize=8)

    probplot(values, dist="norm", plot=axes[1])
    axes[1].set_title(f"{series}: normal Q-Q plot")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / f"exp4_diagnostics_{series}.png", dpi=150)
    plt.close(fig)


def run(use_cache: bool = True) -> dict:
    """Fit AR(1)-GARCH(1,1) and AR(1)-FIGARCH(1,d,1) for each series in
    ``config.VOL_MODEL_SERIES`` and write all Section 5 comparison outputs."""
    logger.info("Loading data and running the Section 3 rolling-window backtest...")
    pipeline = backtest.run_full_pipeline(use_cache=use_cache)
    results = pipeline["results"]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    ic_rows: list[dict] = []
    garch_param_rows: list[dict] = []
    figarch_param_rows: list[dict] = []
    lb_rows: list[dict] = []
    diag_rows: list[dict] = []
    run_log: dict[str, dict] = {}

    for series in config.VOL_MODEL_SERIES:
        logger.info("Fitting AR(1)-GARCH(1,1) and AR(1)-FIGARCH(1,d,1) for %s...", series)
        returns = results[series]["returns"]

        garch_result = vm.fit_ar1_garch11(returns)
        figarch_result = vm.fit_ar1_figarch(returns)

        garch_ic = vm.per_observation_ic(garch_result)
        figarch_ic = vm.per_observation_ic(figarch_result)
        ic_rows.append({"series": series, "model": "GARCH", **garch_ic})
        ic_rows.append({"series": series, "model": "FIGARCH", **figarch_ic})

        garch_param_rows.append(_garch_param_row(series, garch_result))
        figarch_param_rows.append(_figarch_param_row(series, figarch_result))

        std_resid = garch_result.resid / garch_result.conditional_volatility
        lb_df = vm.ljung_box_diagnostics(std_resid, config.LJUNG_BOX_LAGS)
        lb_df.insert(0, "series", series)
        lb_rows.append(lb_df)

        diag = vm.distributional_diagnostics(returns)
        diag_rows.append({"series": series, **diag})

        acf_pre = vm.squared_returns_acf(returns, config.ACF_MAX_LAG)
        acf_post = vm.squared_standardized_residual_acf(std_resid, config.ACF_MAX_LAG)
        _plot_acf_comparison(series, acf_pre, acf_post)
        _plot_distributional_diagnostics(series, returns)

        run_log[series] = {
            "nobs": int(garch_result.nobs),
            "garch_aic_per_obs": garch_ic["aic_per_obs"],
            "garch_bic_per_obs": garch_ic["bic_per_obs"],
            "figarch_aic_per_obs": figarch_ic["aic_per_obs"],
            "figarch_bic_per_obs": figarch_ic["bic_per_obs"],
            "garch_persistence": vm.persistence(garch_result),
            "figarch_d": float(figarch_result.params["d"]),
        }

    ic_table = pd.DataFrame(ic_rows)
    garch_params_table = pd.DataFrame(garch_param_rows)
    figarch_params_table = pd.DataFrame(figarch_param_rows)
    lb_table = pd.concat(lb_rows, ignore_index=True)
    diag_table = pd.DataFrame(diag_rows)

    ic_table.to_csv(RESULTS_DIR / "exp4_garch_figarch_ic.csv", index=False)
    garch_params_table.to_csv(RESULTS_DIR / "exp4_garch_params.csv", index=False)
    figarch_params_table.to_csv(RESULTS_DIR / "exp4_figarch_params.csv", index=False)
    lb_table.to_csv(RESULTS_DIR / "exp4_ljung_box.csv", index=False)
    diag_table.to_csv(RESULTS_DIR / "exp4_distributional_diagnostics.csv", index=False)
    with open(RESULTS_DIR / "exp4_run_log.json", "w") as f:
        json.dump(run_log, f, indent=2, default=str)

    logger.info("AIC/BIC per observation (GARCH vs. FIGARCH):\n%s", ic_table.round(4).to_string())
    logger.info(
        "GARCH persistence / FIGARCH d:\n%s",
        pd.DataFrame(run_log).T[["garch_persistence", "figarch_d"]].round(4).to_string(),
    )

    return {
        "ic_table": ic_table,
        "garch_params": garch_params_table,
        "figarch_params": figarch_params_table,
        "ljung_box": lb_table,
        "diagnostics": diag_table,
        "run_log": run_log,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="bypass the on-disk data cache")
    args = parser.parse_args()
    run(use_cache=not args.no_cache)
