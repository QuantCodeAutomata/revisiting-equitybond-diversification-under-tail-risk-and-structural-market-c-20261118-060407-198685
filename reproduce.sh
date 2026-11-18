#!/usr/bin/env bash
# Runs the test suite and every experiment in the order described in README.md.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp1
python -m exp.exp2
python -m exp.exp3
python -m exp.exp4
python -m exp.exp5
