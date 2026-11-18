#!/usr/bin/env bash
# Runs the test suite and every experiment in the order described in README.md.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1_rolling_portfolios
python -m exp.exp2_transaction_costs
python -m exp.exp3_hill_tail_index
python -m exp.exp4_garch_figarch
python -m exp.exp5_nig_option_pricing
