# AGENTS.md

Guide for coding agents and contributors working in this repository.

## What this repository implements

A reproduction of five experiments from Afroz, Shirvani, Appiah, Rachev, and
Fabozzi, *"Revisiting Equity–Bond Diversification under Tail Risk and
Structural Market Change: Evidence from Global ETFs and Model-Implied Option
Prices"* (arXiv:2611.21897): rolling-window minimum-variance and CVaR
portfolio optimization on 30 global equity-bond ETFs (Sections 3-4.2),
transaction-cost sensitivity (Section 4.3), Hill tail-index estimation under
AR(1)-GARCH(1,1) filtering (Section 4.4), AR(1)-GARCH(1,1) vs.
AR(1)-FIGARCH(1,d,1) volatility persistence (Section 5), and Normal Inverse
Gaussian Lévy option pricing via the Carr-Madan FFT method (Section 6).

## Layout

```
src/        shared library: portfolio optimization, metrics, tail-risk,
            volatility models, NIG/FFT option pricing, and config.py
exp/        one script per experiment (exp1.py..exp5.py), calling only the
            src/ public functions
data/       data_loader.py loads ETF OHLCV bars through the QuantX Dataloader
            REST API and FX/Treasury-yield series from FRED; no synthetic
            data is used
tests/      pytest unit tests on small/synthetic fixtures, plus market-data
            tests gated on API keys
results/    tables, plots, run logs and RESULTS.md written by the experiment
            scripts (overwritten on every rerun)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # synthetic-fixture unit tests, no API keys needed
python -m pytest                         # also runs the market-data tests (needs API keys)
```

## Experiments

```bash
python -m exp.exp1   # rolling-window portfolio optimization (Section 4.1/4.2)
python -m exp.exp2   # transaction-cost sensitivity analysis (Section 4.3)
python -m exp.exp3   # Hill tail-index estimation (Section 4.4)
python -m exp.exp4   # GARCH vs. FIGARCH volatility persistence (Section 5)
python -m exp.exp5   # NIG Levy option pricing and implied-volatility surface (Section 6)
```

All five run at the paper's full configuration (the complete 30-ETF
universe, the full 2020-2024 sample, the full 504-day rolling window); no
reduced-scale variant is needed.

## Conventions

* All experiment parameters live in `src/config.py` as module-level
  UPPER_CASE variables, each with a one-line comment giving its meaning,
  unit, and source (a paper section, or "not specified in the paper"). No
  classes or parameter dicts.
* Data is loaded only through `data/data_loader.py` (QuantX Dataloader for
  OHLCV bars, FRED for FX rates and the Treasury yield). No loader or
  generator code lives in `src/` or `exp/`, and no market data is committed;
  `data/cache/` is ignored by Git.
* `src/` modules are flat and shared across every experiment; `exp/` scripts
  call their public functions by the documented argument names and return
  keys and do not invent private shortcuts.
* API keys (`DATALOADER_API_KEY`, `MASSIVE_API_KEY`) are read from the
  environment via `.env`, which is never committed; `.env.example` holds only
  empty placeholders.
* After rerunning an experiment, its command overwrites the corresponding
  files in `results/`. The numbers and verdicts in `results/RESULTS.md` are
  not regenerated automatically — check them against the new files and
  update them by hand.
