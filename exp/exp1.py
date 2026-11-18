"""Experiment 1: Rolling-Window Portfolio Optimization and Out-of-Sample
Risk-Return Performance Evaluation (Section 3, Section 4.1/4.2).

Compares the seven Section 3 strategies (LO-MVP, LO-C95, LO-C99, LS-MVP,
LS-C95, LS-C99, EWP) on the 30-ETF global equity-bond universe using a
504-day rolling window, and reports the Section 4.1/4.2 out-of-sample
risk-return metrics.

Run: ``python -m exp.exp1``
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src import backtest, config, metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def compute_exposure_turnover_summary(results: dict) -> pd.DataFrame:
    """Mean gross exposure, mean short exposure, and mean daily turnover (%)
    per strategy, out-of-sample (Section 3.6)."""
    rows = {}
    for name, res in results.items():
        rows[name] = {
            "gross_exposure_mean": res["gross_exposure"].mean(),
            "short_exposure_mean": res["short_exposure"].mean(),
            "turnover_mean_pct": 100 * res["turnover"].dropna().mean(),
        }
    return pd.DataFrame(rows).T


def plot_cumulative_wealth(results: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res in results.items():
        wealth = backtest.portfolio_wealth(res["returns"])
        ax.plot(wealth.index, wealth.to_numpy(), label=name)
    ax.set_title("Out-of-sample cumulative wealth, V0=$100 (Section 3.5)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio wealth ($)")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def run(use_cache: bool = True) -> dict:
    logger.info("Loading data and running the Section 3 rolling-window backtest...")
    pipeline = backtest.run_full_pipeline(use_cache=use_cache)
    prices, returns, risk_free, results, meta = (
        pipeline["prices"], pipeline["returns"], pipeline["risk_free_daily"],
        pipeline["results"], pipeline["meta"],
    )
    out_of_sample_dates = next(iter(results.values()))["returns"].index
    logger.info(
        "Realized data: %d price rows (%s to %s), %d-asset universe, "
        "%d out-of-sample dates (%s to %s)",
        prices.shape[0], prices.index.min().date(), prices.index.max().date(),
        returns.shape[1], len(out_of_sample_dates),
        out_of_sample_dates.min().date(), out_of_sample_dates.max().date(),
    )

    summary = {
        name: metrics.performance_summary(res["returns"], risk_free)
        for name, res in results.items()
    }
    summary_df = pd.DataFrame(summary).T[
        ["sharpe", "sortino", "rachev", "daily_volatility", "max_drawdown_pct",
         "var_95", "cvar_95", "var_99", "cvar_99", "terminal_wealth"]
    ]
    exposure_df = compute_exposure_turnover_summary(results)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(RESULTS_DIR / "exp1_performance_summary.csv")
    exposure_df.to_csv(RESULTS_DIR / "exp1_exposure_turnover.csv")
    plot_cumulative_wealth(results, RESULTS_DIR / "exp1_cumulative_wealth.png")

    log = {
        "n_price_rows": int(prices.shape[0]),
        "date_range": [str(prices.index.min().date()), str(prices.index.max().date())],
        "universe_size": int(returns.shape[1]),
        "rolling_window": config.ROLLING_WINDOW,
        "n_out_of_sample_obs": int(len(out_of_sample_dates)),
        "out_of_sample_range": [str(out_of_sample_dates.min().date()), str(out_of_sample_dates.max().date())],
        "non_massive_source": meta.get("non_massive_source"),
        "n_forward_filled": meta.get("n_forward_filled"),
        "warnings": meta.get("warnings"),
    }
    with open(RESULTS_DIR / "exp1_run_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    logger.info("Performance summary:\n%s", summary_df.round(6).to_string())
    logger.info("Exposure/turnover summary:\n%s", exposure_df.round(4).to_string())
    return {"summary": summary_df, "exposure": exposure_df, "log": log}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="bypass the on-disk data cache")
    args = parser.parse_args()
    run(use_cache=not args.no_cache)
