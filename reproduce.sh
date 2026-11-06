#!/usr/bin/env bash
# Runs the test suite and every experiment of this replication in sequence.
set -uo pipefail

python -m pytest -m "not market_data"

python -m exp.exp_1
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4

echo "Done. See results/RESULTS.md for the comparison against the paper."
