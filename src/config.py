"""Paper-derived parameters for every experiment (module-level constants only).

Section references are to "Revisiting Equity-Bond Diversification under Tail
Risk and Structural Market Change: Evidence from Global ETFs and Model-Implied
Option Prices".
"""

# ---------------------------------------------------------------------------
# Universe (Section 2.1, Table 1)
# ---------------------------------------------------------------------------
US_TICKERS = [
    "AOA", "AOR", "AOM", "AOK", "GAL", "MDIV", "IYLD", "CVY", "RLY", "PWS",
    "GAA", "TRTY", "OCIO", "GDMA", "NTSX", "RPAR",
]  # 16 U.S. balanced / multi-asset ETFs (Table 1)
CA_TICKERS = [
    "VBAL.TO", "XBAL.TO", "ZBAL.TO", "VGRO.TO", "XGRO.TO", "ZGRO.TO",
    "VCNS.TO", "VCIP.TO", "ZCON.TO", "XINC.TO",
]  # 10 Canadian balanced ETFs (Table 1), quoted in CAD
AU_TICKERS = [
    "VDBA.AX", "VDGR.AX", "DBBF.AX", "DGGF.AX",
]  # 4 Australian balanced ETFs (Table 1), quoted in AUD
ALL_TICKERS = US_TICKERS + CA_TICKERS + AU_TICKERS  # 30 ETFs (Section 2.1)
N_ASSETS = len(ALL_TICKERS)  # 30, source: Section 2.1
assert N_ASSETS == 30

DJIA_PROXY_TICKER = "DIA"  # DJIA-tracking ETF; Dataloader has no DJIA index feed,
# so the broad-equity benchmark is proxied by the SPDR Dow Jones Industrial
# Average ETF (DIA), see "Deviations from the paper" in results/RESULTS.md.

FX_CAD_PER_USD_SERIES = "DEXCAUS"  # FRED series, CAD per USD (Section 2.1)
FX_USD_PER_AUD_SERIES = "DEXUSAL"  # FRED series, USD per AUD (Section 2.1)
RISK_FREE_SERIES = "DGS3MO"  # FRED 3-month Treasury constant maturity yield, annualized % (Section 2.2)

# ---------------------------------------------------------------------------
# Sample period (Section 2.1)
# ---------------------------------------------------------------------------
SAMPLE_START = "2020-01-02"  # first sample date, source: Section 2.1
SAMPLE_END = "2024-12-31"  # last sample date, source: Section 2.1

# ---------------------------------------------------------------------------
# Rolling-window optimization (Section 3, exp_1 / exp_2 / exp_3 / exp_4)
# ---------------------------------------------------------------------------
ROLLING_WINDOW = 504  # trading days (~2y) of history used to form weights, Section 3
OUT_OF_SAMPLE_OBS_PAPER = 754  # paper's reported out-of-sample daily observation count, Section 3
TRADING_DAYS_PER_YEAR = 252  # daily<->annual conversion factor, Section 2.2

LONG_ONLY_BOUNDS = (0.0, 1.0)  # 0<=w_i<=1, Section 3 step 8
LONG_SHORT_BOUNDS = (-0.3, 1.3)  # -0.3<=w_i<=1.3, Section 3 step 8
CVAR_ALPHA_95 = 0.95  # CVaR confidence level, Section 3.2
CVAR_ALPHA_99 = 0.99  # CVaR confidence level, Section 3.2

# Optimizer settings: tolerance/iterations not specified in the paper (uncertainties).
SLSQP_MAXITER = 500  # not specified in the paper; conservative SLSQP cap
SLSQP_TOL = 1e-9  # not specified in the paper
CVXPY_SOLVER = "CLARABEL"  # not specified in the paper; default LP/SOCP solver for CVaR

INITIAL_WEALTH = 100.0  # V_0, Section 3.5
EWP_VALUE = 1.0 / N_ASSETS  # equally weighted portfolio, Section 3.3

# ---------------------------------------------------------------------------
# Performance evaluation (Section 4.1 / 4.2)
# ---------------------------------------------------------------------------
RACHEV_BETA = 0.95  # top/bottom tail fraction for the Rachev ratio, Section 4.1
RACHEV_GAMMA = 0.95
VAR_CVAR_LEVELS = (0.95, 0.99)  # confidence levels for VaR / CVaR, Section 4.2

# ---------------------------------------------------------------------------
# Transaction-cost sensitivity (Section 4.3, exp_2)
# ---------------------------------------------------------------------------
TRANSACTION_COST_BPS = (0, 5, 10, 20)  # proportional cost levels, basis points, Section 4.3

# ---------------------------------------------------------------------------
# Hill tail-index estimation (Section 4.4, exp_3)
# ---------------------------------------------------------------------------
HILL_K_REFERENCE = 100  # reference number of upper order statistics, Section 4.4
HILL_K_GRID_MIN = 40  # sensitivity grid lower bound, Section 4.4
HILL_K_GRID_MAX = 200  # sensitivity grid upper bound, Section 4.4
HILL_K_GRID_STEP = 10  # sensitivity grid step, Section 4.4
HILL_CI_Z = 1.96  # 95% Wald critical value, Section 4.4
TAIL_RISK_SERIES = ["DJIA", "EWP", "LS-MVP", "LS-C95"]  # series analyzed, Section 4.4

# ---------------------------------------------------------------------------
# GARCH / FIGARCH volatility models (Section 5, exp_4)
# ---------------------------------------------------------------------------
VOL_MODEL_SERIES = ["EWP", "LS-MVP", "LS-C95"]  # series analyzed, Section 5
LJUNG_BOX_LAGS = (10, 20, 50)  # Section 5.2 step 9
ACF_MAX_LAG = 200  # Section 5.2 step 8/11

# ---------------------------------------------------------------------------
# NIG Levy option pricing (Section 6, exp_5)
# ---------------------------------------------------------------------------
NIG_MOMENT_ORDER = 4  # mean, variance, skewness, kurtosis matched, Section 6.2
NIG_ECF_GRID_POINTS = 1000  # points for the empirical characteristic function match, Section 6.2
NIG_ECF_GRID_RANGE = (-1.5, 1.5)  # evaluation interval for the ECF match, Section 6.2
NIG_ALPHA_BOUNDS = (1e-6, 500.0)  # Section 6.2
NIG_BETA_BOUNDS = (-500.0, 500.0)  # Section 6.2
NIG_DELTA_BOUNDS = (1e-6, 1.0)  # Section 6.2
NIG_MU_BOUNDS = (-1.0, 1.0)  # Section 6.2
NIG_BETA_ALPHA_SLACK = 1e-8  # nonlinear constraint beta^2 - alpha^2 + slack <= 0, Section 6.2

OPTION_S0 = 100.0  # normalized initial underlying price, Section 6.3
OPTION_MONEYNESS_RANGE = (0.8, 1.2)  # K/S0 grid range, Section 6.4
OPTION_MONEYNESS_POINTS = 100  # grid density; exact spacing not specified in the paper (uncertainties)
OPTION_MATURITIES_DAYS = list(range(1, 201))  # T = 1..200 trading days, Section 6.4

FFT_N_BASELINE = 2 ** 10  # baseline FFT grid size, Section 6.4
FFT_N_CONVERGENCE = [2 ** 10, 2 ** 11, 2 ** 12, 2 ** 13]  # convergence diagnostic grids, Section 6.4
FFT_V_MAX = 575.0  # upper truncation of the integration grid, Section 6.4
FFT_DAMPING_ALPHA = 0.45  # Carr-Madan damping parameter alpha_d, Section 6.4
NO_ARBITRAGE_TOLERANCE = 1e-8  # Section 6.4 step 13
IV_SMOOTHING_BANDWIDTH = None  # Gaussian-kernel bandwidth not specified in the paper (uncertainties);
# scipy.stats.gaussian_kde default (Scott's rule) is used.

RANDOM_SEED = 42  # not specified in the paper; used only for reproducible numerical fallbacks
