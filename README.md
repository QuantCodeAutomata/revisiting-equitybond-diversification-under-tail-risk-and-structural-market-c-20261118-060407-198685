# Revisiting Equity–Bond Diversification under Tail Risk and Structural Market Change: Evidence from Global ETFs and Model-Implied Option Prices

Reproduction of five experiments from the paper "Revisiting Equity–Bond
Diversification under Tail Risk and Structural Market Change: Evidence from
Global ETFs and Model-Implied Option Prices".

## What this repository implements

1. **Rolling-window portfolio optimization** — minimum-variance (MVP) and
   CVaR portfolios under long-only and long-short constraints, benchmarked
   against an equally weighted portfolio (EWP), on a 30-ETF global
   equity-bond universe with a 504-day rolling estimation window
   (Section 3, Section 4.1/4.2).
2. **Transaction-cost sensitivity** of the same seven strategies to
   proportional costs of 0, 5, 10 and 20 basis points applied to the
   turnover-based daily drag (Section 4.3).
3. **Extreme-tail risk estimation** via the Hill tail-index estimator applied
   to AR(1)-GARCH(1,1) standardized residuals of DJIA, EWP, LS-MVP and
   LS-C95 (Section 4.4).
4. **Volatility persistence**: AR(1)-GARCH(1,1) vs. AR(1)-FIGARCH(1,d,1) for
   EWP, LS-MVP and LS-C95, compared by AIC/BIC, persistence parameters and
   Ljung-Box diagnostics (Section 5).
5. **NIG Lévy option pricing**: Normal Inverse Gaussian calibration to EWP
   log returns, martingale-corrected Carr-Madan FFT pricing of European
   calls/puts, and the model-implied volatility surface (Section 6).

## Repository layout

```
src/        shared library: data-independent portfolio, metric, and model code
exp/        one script per experiment, calling only the src/ API
data/       data/data_loader.py downloads and assembles the ETF/FX/rate panel
tests/      pytest unit tests (synthetic fixtures) and market-data tests
results/    tables, plots and RESULTS.md written by the experiment scripts
```

## Data

Daily OHLCV bars for the 30 ETFs and the DJIA proxy (DIA, used because the
QuantX Dataloader and Massive do not serve a DJIA index feed — see
`results/RESULTS.md`) are loaded through the QuantX Dataloader REST API
(dividend-adjusted). The CAD/USD and AUD/USD exchange rates and the 3-month
US Treasury yield are loaded directly from FRED, since the Dataloader and
Massive do not serve these macroeconomic series (Section 2.1).

Copy `.env.example` to `.env` and fill in:

```
DATALOADER_API_KEY=...   # https://quantx.limex.pro/
MASSIVE_API_KEY=...      # https://massive.com/
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Tests

```bash
python -m pytest -m "not market_data"   # synthetic-fixture unit tests
python -m pytest                         # also runs market-data tests (needs API keys)
```

## Running the experiments

```bash
python -m exp.exp1   # rolling-window portfolio optimization (Section 4.1/4.2)
python -m exp.exp2   # transaction-cost sensitivity analysis (Section 4.3)
python -m exp.exp3   # Hill tail-index estimation (Section 4.4)
python -m exp.exp4   # GARCH vs. FIGARCH volatility persistence (Section 5)
python -m exp.exp5   # NIG Levy option pricing and implied-volatility surface (Section 6)
```

Each script writes its tables, plots and logs directly to `results/`. See
`results/RESULTS.md` for the realized configuration, the comparison with the
paper's reported values, and every deviation from the paper.

Full-paper-scale runs (where the executed configuration was reduced to fit
the replication's time budget) are selected with `--full`; see
`results/RESULTS.md` for which experiments this applies to and the estimated
runtime of the full configuration.

## Reproduce end to end

```bash
bash reproduce.sh
```
