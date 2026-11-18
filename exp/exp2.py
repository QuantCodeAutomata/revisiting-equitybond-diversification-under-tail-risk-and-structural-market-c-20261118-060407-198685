"""Experiment 2: Transaction-Cost Sensitivity Analysis of Portfolio Risk
Measures (Section 4.3).

For each of the seven Section 3 strategies, the out-of-sample weights from
the 504-day rolling-window backtest are held fixed, and a proportional
transaction cost (0, 5, 10, 20 bps) is applied as a daily drag proportional
to turnover: R_net_t = R_gross_t - 2*c*T_t (Section 4.3). Terminal wealth,
daily volatility, maximum drawdown, VaR and CVaR are recomputed from the net
return series at each cost level.

Run: ``python -m exp.exp2``
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

from src import backtest, config, metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def net_returns(gross_returns: pd.Series, turnover: pd.Series, cost_bps: float) -> pd.Series:
    """R_net = R_gross - 2*c*T, c in decimal (Section 4.3 step 4).

    The factor 2 accounts for both purchases and sales during rebalancing.
    Turnover is NaN on the first out-of-sample date (no pre-existing
    portfolio); the cost drag there is treated as zero (no net position change
    recorded yet on that date), matching the "no reoptimization, weights
    held fixed" convention of Section 4.3.
    """
    cost_decimal = cost_bps / 10_000.0
    drag = 2.0 * cost_decimal * turnover.fillna(0.0)
    return gross_returns - drag


def net_metrics(net_ret: pd.Series, risk_free_daily: pd.Series) -> dict:
    """Terminal wealth, daily volatility, max drawdown, VaR, CVaR from a net
    return series (Section 4.3 step 6); Sharpe/Sortino/Rachev are not part of
    this experiment's reported outputs but are included for completeness."""
    summary = metrics.performance_summary(net_ret, risk_free_daily)
    return summary


def run(use_cache: bool = True) -> dict:
    logger.info("Loading data and running the Section 3 rolling-window backtest...")
    pipeline = backtest.run_full_pipeline(use_cache=use_cache)
    results, risk_free = pipeline["results"], pipeline["risk_free_daily"]

    rows = []
    net_return_series: dict[tuple[str, int], pd.Series] = {}
    for strategy, res in results.items():
        for cost_bps in config.TRANSACTION_COST_BPS:
            net_ret = net_returns(res["returns"], res["turnover"], cost_bps)
            net_return_series[(strategy, cost_bps)] = net_ret
            summary = net_metrics(net_ret, risk_free)
            rows.append({"strategy": strategy, "cost_bps": cost_bps, **summary})

    table = pd.DataFrame(rows)
    pivot_wealth = table.pivot(index="strategy", columns="cost_bps", values="terminal_wealth")
    pivot_vol = table.pivot(index="strategy", columns="cost_bps", values="daily_volatility")
    pivot_mdd = table.pivot(index="strategy", columns="cost_bps", values="max_drawdown_pct")
    pivot_var95 = table.pivot(index="strategy", columns="cost_bps", values="var_95")
    pivot_cvar95 = table.pivot(index="strategy", columns="cost_bps", values="cvar_95")

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "exp2_transaction_cost_full.csv", index=False)
    pivot_wealth.to_csv(RESULTS_DIR / "exp2_terminal_wealth_by_cost.csv")
    pivot_vol.to_csv(RESULTS_DIR / "exp2_volatility_by_cost.csv")
    pivot_mdd.to_csv(RESULTS_DIR / "exp2_max_drawdown_by_cost.csv")
    pivot_var95.to_csv(RESULTS_DIR / "exp2_var95_by_cost.csv")
    pivot_cvar95.to_csv(RESULTS_DIR / "exp2_cvar95_by_cost.csv")

    fig, ax = plt.subplots(figsize=(9, 6))
    for strategy in results:
        ax.plot(config.TRANSACTION_COST_BPS, pivot_wealth.loc[strategy], marker="o", label=strategy)
    ax.set_title("Terminal wealth vs. proportional transaction cost (Section 4.3)")
    ax.set_xlabel("Transaction cost (basis points)")
    ax.set_ylabel("Terminal wealth ($), V0=$100")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp2_terminal_wealth_vs_cost.png", dpi=150)
    plt.close(fig)

    log = {
        "cost_levels_bps": list(config.TRANSACTION_COST_BPS),
        "n_out_of_sample_obs": int(len(next(iter(results.values()))["returns"])),
        "strategies": list(results.keys()),
    }
    with open(RESULTS_DIR / "exp2_run_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    logger.info("Terminal wealth by cost level:\n%s", pivot_wealth.round(4).to_string())
    logger.info("Max drawdown (%%) by cost level:\n%s", pivot_mdd.round(2).to_string())
    return {"table": table, "terminal_wealth": pivot_wealth, "max_drawdown": pivot_mdd}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="bypass the on-disk data cache")
    args = parser.parse_args()
    run(use_cache=not args.no_cache)
