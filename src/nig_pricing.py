"""Normal Inverse Gaussian (NIG) Levy option pricing (Section 6).

No library in the project's dependency list implements the paper's combined
moment/characteristic-function NIG calibration, its martingale correction, or
its Carr-Madan FFT pricing grid, so Sections 6.2-6.6 are implemented from
scratch with ``numpy``/``scipy``. Black-Scholes implied-volatility inversion
uses ``scipy.optimize.brentq``; ``scipy.stats.norm`` supplies the Gaussian cdf.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import NonlinearConstraint, brentq, minimize
from scipy.stats import kurtosis as _kurtosis
from scipy.stats import norm
from scipy.stats import skew as _skew


@dataclass(frozen=True)
class NIGParams:
    alpha: float
    beta: float
    delta: float
    mu: float


# ---------------------------------------------------------------------------
# NIG cumulant/characteristic function and theoretical moments (Section 6.2)
# ---------------------------------------------------------------------------

def nig_cgf(w: np.ndarray, params: NIGParams) -> np.ndarray:
    """NIG cumulant-generating function (Section 6.2):

    K(w) = mu*w + delta*(sqrt(alpha^2-beta^2) - sqrt(alpha^2-(beta+w)^2)).
    """
    alpha, beta, delta, mu = params.alpha, params.beta, params.delta, params.mu
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    return mu * w + delta * (gamma - np.sqrt(alpha ** 2 - (beta + w) ** 2 + 0j))


def nig_characteristic_function(u: np.ndarray, params: NIGParams, t: float = 1.0) -> np.ndarray:
    """phi_X(u;t) = exp{t * K(iu)} (Section 6.2/6.3)."""
    psi = nig_cgf(1j * u, params)
    return np.exp(t * psi)


def nig_theoretical_moments(params: NIGParams) -> np.ndarray:
    """Theoretical mean, variance, skewness, excess kurtosis of the NIG law
    (standard Barndorff-Nielsen NIG moment formulas, used in the Section 6.2
    moment-matching term of the calibration objective)."""
    alpha, beta, delta, mu = params.alpha, params.beta, params.delta, params.mu
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    mean = mu + delta * beta / gamma
    variance = delta * alpha ** 2 / gamma ** 3
    skewness = 3 * beta / (alpha * np.sqrt(delta * gamma))
    excess_kurtosis = 3 * (1 + 4 * (beta / alpha) ** 2) / (delta * gamma)
    return np.array([mean, variance, skewness, excess_kurtosis])


def empirical_moments(returns: np.ndarray) -> np.ndarray:
    """Empirical mean, variance, skewness, excess kurtosis (Section 6.2)."""
    return np.array([
        float(np.mean(returns)),
        float(np.var(returns, ddof=0)),
        float(_skew(returns, bias=True)),
        float(_kurtosis(returns, fisher=True, bias=True)),
    ])


def empirical_characteristic_function(returns: np.ndarray, u_grid: np.ndarray) -> np.ndarray:
    """phi_hat(u) = (1/n) sum_j exp(i*u*r_j) (Section 6.2)."""
    phases = np.exp(1j * np.outer(u_grid, returns))
    return phases.mean(axis=1)


def nig_calibration_objective(theta: np.ndarray, m_hat: np.ndarray, phi_hat: np.ndarray, u_grid: np.ndarray) -> float:
    """J(Theta) of Section 6.2:

    sum_{q=1}^{4} (1 - m_q(Theta)/m_hat_q)^2
      + [sum_{j=1}^{1000} |1 - phi_Theta(u_j)/phi_hat(u_j)|]^2.
    """
    params = NIGParams(*theta)
    m_theta = nig_theoretical_moments(params)
    moment_term = np.sum((1 - m_theta / m_hat) ** 2)
    phi_theta = nig_characteristic_function(u_grid, params, t=1.0)
    cf_term = np.sum(np.abs(1 - phi_theta / phi_hat)) ** 2
    return float(moment_term + cf_term)


def fit_nig(
    returns: np.ndarray,
    bounds: tuple[tuple[float, float], ...],
    beta_alpha_slack: float,
    ecf_grid_points: int,
    ecf_grid_range: tuple[float, float],
    x0: tuple[float, float, float, float] | None = None,
) -> tuple[NIGParams, float, float]:
    """Estimate Theta=(alpha,beta,delta,mu) minimizing ``nig_calibration_objective``
    subject to the bounds and the constraint beta^2 - alpha^2 + slack <= 0 (Section 6.2).

    Returns ``(params, objective_value, max_cf_discrepancy)``.
    """
    m_hat = empirical_moments(returns)
    u_grid = np.linspace(ecf_grid_range[0], ecf_grid_range[1], ecf_grid_points)
    phi_hat = empirical_characteristic_function(returns, u_grid)

    if x0 is None:
        # A numerically safe starting point (Section 6.2 gives no explicit
        # starting value): a moderately peaked, near-symmetric NIG law scaled
        # to the sample standard deviation, with beta well inside the
        # alpha > |beta| feasible region so sqrt(alpha^2 - beta^2) is finite.
        std = float(np.std(returns, ddof=0))
        sk = float(_skew(returns, bias=True))
        shape = 10.0  # dimensionless shape constant for the initial guess
        delta0 = max(std, 1e-6)
        alpha0 = shape / delta0
        beta0 = np.clip(0.1 * alpha0 * np.sign(sk), -0.5 * alpha0, 0.5 * alpha0)
        gamma0 = np.sqrt(alpha0 ** 2 - beta0 ** 2)
        mu0 = float(np.mean(returns)) - delta0 * beta0 / gamma0
        x0 = (
            float(np.clip(alpha0, bounds[0][0], bounds[0][1])),
            float(np.clip(beta0, bounds[1][0], bounds[1][1])),
            float(np.clip(delta0, bounds[2][0], bounds[2][1])),
            float(np.clip(mu0, bounds[3][0], bounds[3][1])),
        )

    def constraint_fun(theta):
        alpha, beta = theta[0], theta[1]
        return beta ** 2 - alpha ** 2 + beta_alpha_slack

    constraint = NonlinearConstraint(constraint_fun, -np.inf, 0.0)
    result = minimize(
        nig_calibration_objective,
        x0,
        args=(m_hat, phi_hat, u_grid),
        method="SLSQP",
        bounds=bounds,
        constraints=[constraint],
        options={"maxiter": 1000, "ftol": 1e-14},
    )
    params = NIGParams(*result.x)
    phi_theta = nig_characteristic_function(u_grid, params, t=1.0)
    max_discrepancy = float(np.max(np.abs(1 - phi_theta / phi_hat)))
    return params, float(result.fun), max_discrepancy


# ---------------------------------------------------------------------------
# Martingale correction and Carr-Madan FFT pricing (Section 6.3/6.4)
# ---------------------------------------------------------------------------

def martingale_drift(params: NIGParams, r_daily: float) -> float:
    """c = r - K(1), the martingale-correcting drift (Section 6.3)."""
    k1 = float(np.real(nig_cgf(np.array([1.0]), params))[0])
    return r_daily - k1


def log_price_characteristic_function(
    u: np.ndarray, params: NIGParams, s0: float, r_daily: float, maturity_days: int
) -> np.ndarray:
    """phi_T(u) = exp{iu[ln(S0) + (r-K(1))T] + T*K(iu)} (Section 6.3 step 7)."""
    c = martingale_drift(params, r_daily)
    drift = np.log(s0) + c * maturity_days
    psi = nig_cgf(1j * u, params)
    return np.exp(1j * u * drift + maturity_days * psi)


def carr_madan_fft_calls(
    params: NIGParams,
    s0: float,
    r_daily: float,
    maturity_days: int,
    n_fft: int,
    v_max: float,
    alpha_d: float,
) -> tuple[np.ndarray, np.ndarray]:
    """European call prices across a log-strike grid via the Carr-Madan FFT
    method (Section 6.4 steps 8-9), using the rectangular (midpoint) quadrature
    rule on the damped characteristic function.

    Returns ``(strikes, call_prices)`` for ``n_fft`` log-strikes centered
    around ``s0``.
    """
    dv = v_max / n_fft
    v = np.arange(n_fft) * dv
    lam = 2 * np.pi / (n_fft * dv)
    b = n_fft * lam / 2
    k_offset = np.log(s0)

    phi_t = log_price_characteristic_function(v - 1j * (alpha_d + 1), params, s0, r_daily, maturity_days)
    psi_t = np.exp(-r_daily * maturity_days) * phi_t / ((alpha_d + 1j * v) * (alpha_d + 1 + 1j * v))

    y = np.exp(1j * v * (b - k_offset)) * psi_t * dv
    x = np.fft.fft(y)
    u_idx = np.arange(n_fft)
    k_u = k_offset - b + lam * u_idx
    call_prices = (np.exp(-alpha_d * k_u) / np.pi) * np.real(x)
    strikes = np.exp(k_u)
    order = np.argsort(strikes)
    return strikes[order], call_prices[order]


def interpolate_to_moneyness(
    strikes: np.ndarray, call_prices: np.ndarray, s0: float, moneyness_grid: np.ndarray
) -> np.ndarray:
    """Interpolate the FFT call-price grid onto a target moneyness grid
    K/S0 (Section 6.4 step 12)."""
    target_strikes = moneyness_grid * s0
    return np.interp(target_strikes, strikes, call_prices)


def put_from_call_parity(call_price: np.ndarray, strike: np.ndarray, s0: float, r_daily: float, maturity_days: int) -> np.ndarray:
    """Put-call parity with no dividends (Section 6.4 step 11):
    P(K,T) = C(K,T) + K*exp(-rT) - S0."""
    return call_price + strike * np.exp(-r_daily * maturity_days) - s0


# ---------------------------------------------------------------------------
# No-arbitrage diagnostics (Section 6.4 step 13)
# ---------------------------------------------------------------------------

def no_arbitrage_violations(
    strikes: np.ndarray, call_prices: np.ndarray, s0: float, r_daily: float, maturity_days: int, tol: float
) -> dict[str, float]:
    """Lower/upper no-arbitrage bound, monotonicity, and convexity violations
    of a call-price curve at one maturity (Section 6.4 step 13)."""
    discount = np.exp(-r_daily * maturity_days)
    lower_bound = np.maximum(s0 * discount - strikes * discount, 0.0)
    upper_bound = s0
    lower_violation = np.maximum(lower_bound - call_prices - tol, 0.0)
    upper_violation = np.maximum(call_prices - upper_bound - tol, 0.0)
    diffs = np.diff(call_prices)
    monotonicity_violation = np.maximum(diffs + tol, 0.0)  # calls must be non-increasing in K
    second_diff = np.diff(call_prices, n=2)
    convexity_violation = np.maximum(-second_diff - tol, 0.0)  # calls must be convex in K
    return {
        "n_lower_violations": int(np.sum(lower_violation > 0)),
        "max_lower_violation": float(np.max(lower_violation)),
        "n_upper_violations": int(np.sum(upper_violation > 0)),
        "max_upper_violation": float(np.max(upper_violation)) if upper_violation.size else 0.0,
        "n_monotonicity_violations": int(np.sum(monotonicity_violation > 0)),
        "n_convexity_violations": int(np.sum(convexity_violation > 0)),
        "max_convexity_violation": float(np.max(convexity_violation)) if convexity_violation.size else 0.0,
    }


# ---------------------------------------------------------------------------
# Black-Scholes implied volatility inversion (Section 6.6)
# ---------------------------------------------------------------------------

def bs_call_price(s0: float, strike: float, r_daily: float, maturity_days: int, sigma_daily: float) -> float:
    """Black-Scholes call price with inputs on a daily-rate/daily-vol basis."""
    if sigma_daily <= 0 or maturity_days <= 0:
        return max(s0 - strike * np.exp(-r_daily * maturity_days), 0.0)
    sqrt_t = np.sqrt(maturity_days)
    d1 = (np.log(s0 / strike) + (r_daily + 0.5 * sigma_daily ** 2) * maturity_days) / (sigma_daily * sqrt_t)
    d2 = d1 - sigma_daily * sqrt_t
    return s0 * norm.cdf(d1) - strike * np.exp(-r_daily * maturity_days) * norm.cdf(d2)


def implied_volatility(
    call_price: float, s0: float, strike: float, r_daily: float, maturity_days: int,
    sigma_bounds: tuple[float, float] = (1e-6, 5.0),
) -> float:
    """Invert the Black-Scholes call price for sigma via ``scipy.optimize.brentq``
    (Section 6.6); returns NaN when no sigma in ``sigma_bounds`` reproduces the
    observed price (the no-arbitrage bounds are violated)."""
    lo, hi = sigma_bounds

    def objective(sigma):
        return bs_call_price(s0, strike, r_daily, maturity_days, sigma) - call_price

    f_lo, f_hi = objective(lo), objective(hi)
    if np.isnan(f_lo) or np.isnan(f_hi) or f_lo * f_hi > 0:
        return float("nan")
    try:
        return float(brentq(objective, lo, hi, xtol=1e-10, rtol=1e-10, maxiter=200))
    except (ValueError, RuntimeError):
        return float("nan")
