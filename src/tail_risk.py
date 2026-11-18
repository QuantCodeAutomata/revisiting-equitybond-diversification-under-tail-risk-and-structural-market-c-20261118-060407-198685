"""Hill tail-index estimator on AR(1)-GARCH(1,1) standardized residuals (Section 4.4).

No library in the project's dependency list implements the Hill estimator with
the paper's Wald-type confidence interval, so it is implemented from scratch
following the formulas of Section 4.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model


def fit_ar1_garch11(returns: pd.Series) -> tuple[np.ndarray, "arch.univariate.base.ARCHModelResult"]:
    """Fit an AR(1)-GARCH(1,1) model with Gaussian innovations (Section 4.4 step 2).

    Conditional mean: r_t = mu + phi_1 r_{t-1} + eps_t.
    Conditional variance: h_t = omega + alpha_1 eps_{t-1}^2 + beta_1 h_{t-1}.

    Returns the standardized residuals z_t = eps_t / sqrt(h_t) (as a numpy
    array, aligned to the fitted sample) and the fitted ``arch`` result.
    """
    am = arch_model(returns * 100.0, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
    result = am.fit(disp="off")
    standardized_residuals = (result.resid / result.conditional_volatility).dropna().to_numpy()
    return standardized_residuals, result


def hill_estimator(losses_sorted_desc: np.ndarray, k: int) -> float:
    """Hill estimator of the extreme-value index gamma(k) (Section 4.4 Eq.):

    gamma_hat(k) = (1/k) * sum_{i=1}^{k} (ln L_(i) - ln L_(k+1)),

    where ``losses_sorted_desc`` are positive loss magnitudes L_(1) >= L_(2)
    >= ... >= L_(n), and ``k`` is the number of upper order statistics used.
    """
    if k >= len(losses_sorted_desc):
        raise ValueError(f"k={k} must be smaller than the number of losses ({len(losses_sorted_desc)})")
    top_k = losses_sorted_desc[:k]
    threshold = losses_sorted_desc[k]
    gamma_hat = float(np.mean(np.log(top_k) - np.log(threshold)))
    return gamma_hat


def hill_tail_index(losses_sorted_desc: np.ndarray, k: int, z: float = 1.96) -> dict[str, float]:
    """Hill tail index alpha_hat(k) = 1/gamma_hat(k) with a Wald-type 95% CI (Section 4.4).

    SE(gamma_hat) = gamma_hat / sqrt(k) (asymptotic standard error of the Hill
    estimator). The CI for gamma is transformed into a CI for alpha = 1/gamma.
    """
    gamma_hat = hill_estimator(losses_sorted_desc, k)
    se_gamma = gamma_hat / np.sqrt(k)
    gamma_lower, gamma_upper = gamma_hat - z * se_gamma, gamma_hat + z * se_gamma
    alpha_hat = 1.0 / gamma_hat
    # A larger gamma gives a smaller alpha, so the alpha-CI bounds are swapped.
    ci_lower = 1.0 / gamma_upper if gamma_upper > 0 else np.nan
    ci_upper = 1.0 / gamma_lower if gamma_lower > 0 else np.nan
    return {
        "k": k,
        "gamma_hat": gamma_hat,
        "alpha_hat": alpha_hat,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
    }


def negative_loss_magnitudes(standardized_residuals: np.ndarray) -> np.ndarray:
    """Positive loss magnitudes L = -min(0, z_t) from negative standardized residuals,
    sorted in descending order (Section 4.4 steps 4-5)."""
    negative = standardized_residuals[standardized_residuals < 0]
    losses = -negative
    return np.sort(losses)[::-1]


def hill_sensitivity(
    standardized_residuals: np.ndarray, k_grid: np.ndarray, z: float = 1.96
) -> pd.DataFrame:
    """Hill tail-index estimates and CIs across a grid of thresholds ``k`` (Section 4.4 step 9)."""
    losses_sorted = negative_loss_magnitudes(standardized_residuals)
    rows = [hill_tail_index(losses_sorted, int(k), z) for k in k_grid if k < len(losses_sorted)]
    return pd.DataFrame(rows)
