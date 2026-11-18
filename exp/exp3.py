"""Experiment 3: Extreme-Tail Risk Estimation via Hill Tail-Index Estimator
with AR(1)-GARCH(1,1) Filtering (Section 4.4).

For each of the four Section 4.4 series (the DJIA proxy, the equal-weighted
portfolio, and the long-short MVP and 95%-confidence CVaR-optimal strategies),
an AR(1)-GARCH(1,1) model with Gaussian innovations is fitted to the daily
out-of-sample returns. The standardized residuals are filtered to the negative
(loss) side, and the Hill tail-index estimator alpha_hat(k) is computed at the
reference order-statistic count k=100 together with a Wald-type 95% confidence
interval, and across a sensitivity grid k in {40, 50, ..., 200}.

Run: ``python -m exp.exp3``
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

from src import backtest, config, tail_risk

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _series_returns(series_name: str, pipeline: dict) -> pd.Series:
    """Extract the out-of-sample daily return series for a Section 4.4 series name."""
    if series_name == "DJIA":
        return pipeline["djia_returns"]
    return pipeline["results"][series_name]["returns"]


def run(use_cache: bool = True) -> dict:
    """Run the Section 4.4 Hill tail-index experiment end-to-end.

    Returns a dict with the reference table (k=100) and the long-format
    sensitivity table (k grid) as pandas DataFrames.
    """
    logger.info("Loading data and running the Section 3 rolling-window backtest...")
    pipeline = backtest.run_full_pipeline(use_cache=use_cache)

    k_grid = np.arange(config.HILL_K_GRID_MIN, config.HILL_K_GRID_MAX + 1, config.HILL_K_GRID_STEP)

    reference_rows = []
    sensitivity_frames = []
    garch_log: dict[str, dict] = {}

    for series_name in config.TAIL_RISK_SERIES:
        logger.info("Fitting AR(1)-GARCH(1,1) for series '%s'...", series_name)
        returns = _series_returns(series_name, pipeline).dropna()
        standardized_residuals, arch_result = tail_risk.fit_ar1_garch11(returns)
        losses_sorted_desc = tail_risk.negative_loss_magnitudes(standardized_residuals)

        reference = tail_risk.hill_tail_index(
            losses_sorted_desc, k=config.HILL_K_REFERENCE, z=config.HILL_CI_Z
        )
        reference_rows.append(
            {
                "series": series_name,
                "n_obs": int(len(returns)),
                "n_negative_residuals": int(len(losses_sorted_desc)),
                **reference,
            }
        )

        sensitivity = tail_risk.hill_sensitivity(standardized_residuals, k_grid, z=config.HILL_CI_Z)
        sensitivity.insert(0, "series", series_name)
        sensitivity_frames.append(sensitivity)

        params = arch_result.params
        persistence = float(params.get("alpha[1]", np.nan) + params.get("beta[1]", np.nan))
        garch_log[series_name] = {
            "n_obs": int(len(returns)),
            "n_negative_residuals": int(len(losses_sorted_desc)),
            "garch_alpha1": round(float(params.get("alpha[1]", np.nan)), 6),
            "garch_beta1": round(float(params.get("beta[1]", np.nan)), 6),
            "garch_persistence": round(persistence, 6),
        }

    reference_table = pd.DataFrame(reference_rows)[
        ["series", "n_obs", "n_negative_residuals", "k", "gamma_hat", "alpha_hat", "ci_lower", "ci_upper"]
    ]
    sensitivity_table = pd.concat(sensitivity_frames, ignore_index=True)[
        ["series", "k", "gamma_hat", "alpha_hat", "ci_lower", "ci_upper"]
    ]

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    reference_table.to_csv(RESULTS_DIR / "exp3_hill_tail_index_k100.csv", index=False)
    sensitivity_table.to_csv(RESULTS_DIR / "exp3_hill_sensitivity.csv", index=False)

    fig, ax = plt.subplots(figsize=(9, 6))
    for series_name in config.TAIL_RISK_SERIES:
        sub = sensitivity_table[sensitivity_table["series"] == series_name]
        ax.fill_between(sub["k"], sub["ci_lower"], sub["ci_upper"], alpha=0.2)
        ax.plot(sub["k"], sub["alpha_hat"], label=series_name)
    ax.set_title("Hill tail-index alpha_hat(k) with 95% Wald CI (Section 4.4)")
    ax.set_xlabel("k (number of upper order statistics)")
    ax.set_ylabel("alpha_hat(k)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "exp3_hill_sensitivity.png", dpi=150)
    plt.close(fig)

    log = {
        "k_reference": int(config.HILL_K_REFERENCE),
        "k_grid": [int(k) for k in k_grid],
        "ci_z": float(config.HILL_CI_Z),
        "series": list(config.TAIL_RISK_SERIES),
        "garch_fits": garch_log,
    }
    with open(RESULTS_DIR / "exp3_run_log.json", "w") as f:
        json.dump(log, f, indent=2, default=str)

    logger.info("Hill tail index at k=100:\n%s", reference_table.round(4).to_string())
    return {"reference": reference_table, "sensitivity": sensitivity_table}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-cache", action="store_true", help="bypass the on-disk data cache")
    args = parser.parse_args()
    run(use_cache=not args.no_cache)
