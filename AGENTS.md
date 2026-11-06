# AGENTS.md

Replication of Platen, E. and Fergusson, K. (2026), "Information-Theoretic
Model for the Growth Optimal Portfolio of Stocks", arXiv:2611.07457v1
[q-fin.PR]: four experiments covering equal-weighted index construction
(Sec. 4), activity time estimation under the Minimal Market Model (Sec. 2.1),
simulation of the normalized activity process with overreactions (Sec. 3),
and SGH log-return density estimation with likelihood-ratio testing (Sec. 5).

## Layout

```
src/            Shared library code: ewi.py, activity_time.py, cir_simulation.py,
                density.py, config.py (all experiment parameters, as
                module-level UPPER_CASE constants with a source comment each)
exp/            exp_1.py .. exp_4.py; call only src/ and data/, write to results/
data/           data_loader.py (QuantX Dataloader / Massive / Yahoo Finance)
tests/          pytest unit tests; market_data-marked tests use a cached slice
results/        plots, CSV/JSON metrics, RESULTS.md (the paper comparison)
```

## Commands

```bash
pip install -r requirements.txt
cp .env.example .env   # fill in DATALOADER_API_KEY and MASSIVE_API_KEY

python -m pytest -m "not market_data"   # unit tests, no network access
python -m pytest                        # full suite, incl. a cached live-data slice

python -m exp.exp_1            # reduced (realized) configuration
python -m exp.exp_1 --full     # paper's 1973-2024 window (exp_1 and exp_4 support --full)
python -m exp.exp_2
python -m exp.exp_3
python -m exp.exp_4

./reproduce.sh                 # tests (excluding market_data) + all four experiments
```

## Conventions

- All experiment parameters live only in `src/config.py` as module-level
  constants with a comment stating their meaning, unit and the paper section
  (or "not specified in the paper"); no parameter is hardcoded in `exp/`.
- Market data is loaded only through `data/data_loader.py`, which wraps the
  QuantX Dataloader REST API (`POST /fetch-ohlcvs`) with a Massive and then
  Yahoo Finance fallback per ticker. No other module calls a market-data API.
- `DATALOADER_API_KEY` and `MASSIVE_API_KEY` are read from an uncommitted
  `.env` file (see `.env.example`); they must never appear in code, logs, or
  results.
- Downloaded bars are cached under `data/cache/` (ignored by Git) and are
  never committed; only derived metrics, tables and plots go into `results/`.
- `tests/` contains unit tests on synthetic fixtures (always run) and tests
  marked `market_data` that use a small cached real-data slice; they are
  skipped automatically when the API keys are not set.
- After re-running an experiment, its command overwrites the corresponding
  files in `results/`. The numbers and verdicts in `results/RESULTS.md` are
  not regenerated automatically: check them against the new files and update
  them by hand.
- `results/RESULTS.md` records every deviation from the paper (data source,
  universe, sample period, and any identified cause of a numerical gap) and
  lists, under "Not reproduced", the paper results this repository does not
  recover, with the reason.
