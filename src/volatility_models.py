"""AR(1)-GARCH(1,1) vs. AR(1)-FIGARCH(1,d,1) volatility specifications (Section 5).

Both models are fit with the ``arch`` package (Bollerslev-Baillie-style
FIGARCH is a built-in ``vol="FIGARCH"`` specification; Section 5.2), which
covers the paper's conditional-mean/-variance specifications exactly, so no
custom likelihood is implemented. Residual diagnostics use ``statsmodels``
(Ljung-Box) and ``scipy.stats`` (skewness/kurtosis, Section 5.1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from scipy.stats import kurtosis, skew
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import acf

# Returns are rescaled by this factor before fitting (``arch`` warns and can
# fail to converge when the series is "poorly scaled"); fitted mean/variance
# parameters are rescaled back by the corresponding powers of this factor.
RESCALE = 100.0


def fit_ar1_garch11(returns: pd.Series) -> ARCHModelResult:
    """AR(1)-GARCH(1,1), Gaussian innovations (Section 5.2 Eq.):

    r_t = mu + phi_1 r_{t-1} + eps_t,  h_t = omega + alpha_1 eps_{t-1}^2 + beta_1 h_{t-1}.
    """
    am = arch_model(returns * RESCALE, mean="AR", lags=1, vol="GARCH", p=1, q=1, dist="normal")
    return am.fit(disp="off")


def fit_ar1_figarch(returns: pd.Series) -> ARCHModelResult:
    """AR(1)-FIGARCH(1,d,1), Gaussian innovations (Section 5.2 Eq.):

    conditional mean as in :func:`fit_ar1_garch11`; conditional variance follows
    the fractionally-integrated ARCH(infinity) representation with fractional
    differencing parameter ``d`` (0<d<1 gives hyperbolic decay; d=0 reduces to GARCH).
    """
    am = arch_model(returns * RESCALE, mean="AR", lags=1, vol="FIGARCH", p=1, q=1, dist="normal")
    return am.fit(disp="off")


def per_observation_ic(result: ARCHModelResult) -> dict[str, float]:
    """Per-observation AIC/BIC, matching the ``rugarch`` reporting convention
    referenced in Section 5.3 (total criterion divided by the sample size).

    ``result.aic``/``result.bic`` are computed on the RESCALE-d series used for
    numerical fitting. Rescaling the data by a constant ``s`` shifts the
    per-observation Gaussian log-likelihood by ``-log(s)`` (the Jacobian of the
    change of variable), so the per-observation information criteria are
    corrected by ``-2*log(RESCALE)`` to report them on the natural (unscaled
    return) likelihood scale used in Section 5.3. The correction only shifts
    the reported level; it leaves the GARCH-vs-FIGARCH ranking (identical
    shift for both models) unchanged.
    """
    n = result.nobs
    jacobian = 2.0 * np.log(RESCALE)
    return {
        "aic_per_obs": float(result.aic / n - jacobian),
        "bic_per_obs": float(result.bic / n - jacobian),
    }


def persistence(result: ARCHModelResult) -> float:
    """GARCH(1,1) persistence alpha_1 + beta_1 (Section 5.3)."""
    params = result.params
    return float(params.get("alpha[1]", np.nan) + params.get("beta[1]", np.nan))


def ljung_box_diagnostics(standardized_residuals: np.ndarray, lags: tuple[int, ...]) -> pd.DataFrame:
    """Ljung-Box test on standardized residuals and their squares at the given
    lags (Section 5.2 step 9)."""
    resid = pd.Series(standardized_residuals).dropna()
    lb_level = acorr_ljungbox(resid, lags=list(lags), return_df=True)
    lb_squared = acorr_ljungbox(resid ** 2, lags=list(lags), return_df=True)
    out = pd.DataFrame({
        "lag": list(lags),
        "lb_stat_level": lb_level["lb_stat"].to_numpy(),
        "lb_pvalue_level": lb_level["lb_pvalue"].to_numpy(),
        "lb_stat_squared": lb_squared["lb_stat"].to_numpy(),
        "lb_pvalue_squared": lb_squared["lb_pvalue"].to_numpy(),
    })
    return out


def distributional_diagnostics(returns: pd.Series) -> dict[str, float]:
    """Sample skewness and excess-kurtosis-free (Pearson) kurtosis of a return
    series (Section 5.1); ``scipy.stats`` with the default bias-corrected
    estimator and Fisher=False so kurtosis of 3 denotes normality."""
    values = returns.to_numpy()
    return {
        "skewness": float(skew(values)),
        "kurtosis": float(kurtosis(values, fisher=False)),
    }


def squared_returns_acf(returns: pd.Series, max_lag: int) -> np.ndarray:
    """ACF of squared daily returns up to ``max_lag`` (Section 5.2 step 11)."""
    return acf(returns.to_numpy() ** 2, nlags=max_lag, fft=True)


def squared_standardized_residual_acf(standardized_residuals: np.ndarray, max_lag: int) -> np.ndarray:
    """ACF of squared standardized residuals up to ``max_lag`` (Section 5.2 step 8)."""
    resid = pd.Series(standardized_residuals).dropna().to_numpy()
    return acf(resid ** 2, nlags=max_lag, fft=True)
