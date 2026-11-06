# Information-Theoretic Model for the Growth Optimal Portfolio of Stocks

Replication of selected experiments from Platen, E. and Fergusson, K. (2026),
"Information-Theoretic Model for the Growth Optimal Portfolio of Stocks",
arXiv:2611.07457v1 [q-fin.PR].

This repository reproduces four experiments from the paper:

1. **Construction of Equal-Weighted Indices (EWIs)** as proxies for the growth
   optimal portfolio (GOP), Sec. 4.1-4.2.
2. **Activity time estimation** via the Minimal Market Model (MMM)
   parametrization and trendline fitting, Sec. 2.1.
3. **Simulation of the normalized activity model** with overreactions in
   trading behavior, Sec. 3.1-3.2.
4. **Log-return density estimation and likelihood ratio testing** for the
   symmetric generalized hyperbolic (SGH) density and its nested special
   cases, Sec. 5.

## Data

Daily OHLCV bars are loaded through the QuantX Dataloader REST API
(`POST /fetch-ohlcvs`, `data/data_loader.py`), which falls back to Massive and
then Yahoo Finance per ticker. The paper's own data source -- a proprietary
Thomson-Reuters panel of market-capitalisation-weighted GICS sub-indices for
67 countries, 1973-2024 -- is not obtainable through the Dataloader or the
Massive REST API. This repository substitutes exchange-traded index funds
that are themselves cap-weighted indices (single-country MSCI ETFs, the 11
GICS Select Sector SPDR ETFs, and GICS industry-level SPDR ETFs) as a reduced,
genuinely available universe. See `results/RESULTS.md`, "Deviations from the
paper", for the full justification, the tickers used, and the realized sample
period.

Set `DATALOADER_API_KEY` and `MASSIVE_API_KEY` in a local `.env` file (see
`.env.example`); these are never committed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY
```

## Tests

```bash
python -m pytest -m "not market_data"   # unit tests on fixtures/synthetic data
python -m pytest                        # full suite, including a reduced live data slice
```

## Running the experiments

Each experiment runs from the repository root as a module and writes its
artifacts directly into `results/`. The default (reduced) configuration uses
the realized sample window `2011-01-01` to `2024-07-31`, described in
`results/RESULTS.md`; `--full` selects the paper's `1973-01-01` to
`2024-07-31` window (not obtainable with the substitute ETF universe, see
"Deviations from the paper").

```bash
python -m exp.exp_1            # EWI construction and MSCI comparison (Sec. 4.1-4.2)
python -m exp.exp_2            # Activity time estimation (Sec. 2.1)
python -m exp.exp_3            # Normalized activity simulation (Sec. 3.1-3.2)
python -m exp.exp_4            # SGH density estimation and LR testing (Sec. 5)
python -m exp.exp_1 --full     # paper's 1973-2024 window (ETF universe still substitutes GICS indices)
```

`reproduce.sh` runs the full test suite (excluding the live-data marker) and
every experiment in sequence with the commands above.

## Repository layout

```
src/            EWI construction, activity time, CIR simulation, SGH density -- shared library code
exp/            exp_1.py .. exp_4.py, one script per experiment, calling only src/ and data/
data/           data_loader.py (QuantX Dataloader / Massive / Yahoo Finance loader)
tests/          pytest unit tests (synthetic fixtures) and market_data-marked integration tests
results/        plots, CSV/JSON metrics saved by the experiment scripts, and RESULTS.md
```

See `AGENTS.md` for conventions and commands for contributors and coding agents.
