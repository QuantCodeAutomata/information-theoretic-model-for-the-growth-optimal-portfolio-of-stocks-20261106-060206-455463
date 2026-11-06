"""Experiment 1: Construction of Equal-Weighted Indices (EWIs) as proxies for
the growth optimal portfolio.

Reproduces Platen & Fergusson (2026), Section 4.1 ("Construction of
Equal-Weighted Indices") and Section 4.2 ("Comparison of EWIs and MSCI"):
global equal-weighted indices are built at several GICS classification
depths d and compared, by their average annualised growth rate, against the
MSCI total return world stock index.

The paper's input is a Thomson-Reuters panel of daily market-capitalisation
-weighted total-return index values for 67 countries at four GICS
classification depths (d in {11, 25, 74, 163}), 1973-2024. That proprietary
panel cannot be obtained through the QuantX Dataloader or the Massive REST
API (see data/data_loader.py docstring); this experiment instead builds the
EWI recursion of Eq. (4.1)-(4.3) on three classification resolutions
assembled from exchange-traded index funds that are themselves
cap-weighted indices:

  d=1   -- COUNTRY_ETFS: one cap-weighted MSCI single-country index ETF per
           country, equally weighted across countries (no sector split).
  d=11  -- SECTOR_ETFS: the 11 GICS Select Sector SPDR ETFs for the United
           States, equally weighted across sectors (single country).
  d=H   -- INDUSTRY_ETFS: GICS industry-level SPDR ETFs for the United
           States, the highest classification resolution available from
           exchange-traded funds (single country).

This substitution and its consequences for the d-monotonicity hypothesis are
recorded in results/RESULTS.md, "Deviations from the paper".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from data.data_loader import (
    COUNTRY_ETFS,
    INDUSTRY_ETFS,
    MSCI_WORLD_PROXY,
    SECTOR_ETFS,
    load_close_panel,
)
from src.config import (
    SAMPLE_END_PAPER,
    SAMPLE_END_RUN,
    SAMPLE_START_PAPER,
    SAMPLE_START_RUN,
    TRADING_DAYS_PER_YEAR,
    TRANSACTION_COST_KBAR,
)
from src.ewi import annualized_volatility, average_annualized_growth_rate, construct_ewi

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def _single_country_panel(close_panel: pd.DataFrame) -> pd.DataFrame:
    """d=1: one group ("Global"), every column a country -- Eq. (4.1)-(4.3)."""
    cols = pd.MultiIndex.from_tuples(
        [("Global", c) for c in close_panel.columns], names=["group", "country"]
    )
    out = close_panel.copy()
    out.columns = cols
    return out


def _single_group_country_panel(close_panel: pd.DataFrame, country_label: str) -> pd.DataFrame:
    """d=H: one country, every column a classification group."""
    cols = pd.MultiIndex.from_tuples(
        [(g, country_label) for g in close_panel.columns], names=["group", "country"]
    )
    out = close_panel.copy()
    out.columns = cols
    return out


def build_all_ewis(start_date: str, end_date: str, use_cache: bool = True) -> dict:
    """Load the three ETF panels and the MSCI proxy, construct the EWIs."""
    country_panel, country_meta = load_close_panel(COUNTRY_ETFS, start_date, end_date, use_cache)
    sector_panel, sector_meta = load_close_panel(SECTOR_ETFS, start_date, end_date, use_cache)
    industry_panel, industry_meta = load_close_panel(INDUSTRY_ETFS, start_date, end_date, use_cache)
    msci_panel, msci_meta = load_close_panel({MSCI_WORLD_PROXY: "MSCI"}, start_date, end_date, use_cache)

    ewi1 = construct_ewi(_single_country_panel(country_panel), TRANSACTION_COST_KBAR)
    ewi11 = construct_ewi(_single_group_country_panel(sector_panel, "US"), TRANSACTION_COST_KBAR)
    ewiH = construct_ewi(_single_group_country_panel(industry_panel, "US"), TRANSACTION_COST_KBAR)
    msci = msci_panel["MSCI"].dropna()

    meta = {
        "country_meta": country_meta, "sector_meta": sector_meta,
        "industry_meta": industry_meta, "msci_meta": msci_meta,
        "n_countries": country_panel.shape[1], "n_sectors": sector_panel.shape[1],
        "n_industries": industry_panel.shape[1],
    }
    return {"ewi1": ewi1, "ewi11": ewi11, "ewiH": ewiH, "msci": msci, "meta": meta}


def summarize(ewis: dict) -> pd.DataFrame:
    """Average annualised growth rate and volatility of every EWI and the MSCI proxy."""
    rows = []
    labels = {
        "ewi1": f"EWI1 (d=1, {ewis['meta']['n_countries']} countries)",
        "ewi11": f"EWI11 (d=11 US sectors)",
        "ewiH": f"EWI{ewis['meta']['n_industries']} (d={ewis['meta']['n_industries']} US industries)",
        "msci": "MSCI proxy (ACWI)",
    }
    for key, label in labels.items():
        series = ewis[key]["ewi"] if key != "msci" else ewis[key]
        g = average_annualized_growth_rate(series, TRADING_DAYS_PER_YEAR)
        vol = annualized_volatility(series, TRADING_DAYS_PER_YEAR)
        rows.append({"index": label, "g_annualized": g, "sigma_annualized": vol, "n_obs": len(series)})
    return pd.DataFrame(rows)


def plot_log_indices(ewis: dict, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    for key, color, label in [
        ("ewi1", "tab:blue", "log EWI1"),
        ("ewi11", "tab:green", "log EWI11"),
        ("ewiH", "tab:red", f"log EWI{ewis['meta']['n_industries']}"),
    ]:
        series = ewis[key]["ewi"]
        ax.plot(series.index, np.log(series / series.iloc[0]), color=color, label=label)
    msci = ewis["msci"]
    ax.plot(msci.index, np.log(msci / msci.iloc[0]), color="black", linestyle="--", label="log MSCI proxy")
    ax.set_title("Logarithms of the equal-weighted indices and the MSCI proxy")
    ax.set_xlabel("Date")
    ax.set_ylabel("log(index / index at t0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(full: bool = False) -> None:
    start_date = SAMPLE_START_PAPER if full else SAMPLE_START_RUN
    end_date = SAMPLE_END_PAPER if full else SAMPLE_END_RUN
    print(f"exp_1: sample window {start_date} to {end_date} (full={full})")

    ewis = build_all_ewis(start_date, end_date)
    print(f"Loaded {ewis['meta']['n_countries']} country ETFs, "
          f"{ewis['meta']['n_sectors']} sector ETFs, {ewis['meta']['n_industries']} industry ETFs")

    table = summarize(ewis)
    print(table.to_string(index=False))

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "exp1_growth_rates.csv", index=False)
    plot_log_indices(ewis, RESULTS_DIR / "exp1_ewi_comparison.png")

    log_meta = {
        "start_date": start_date, "end_date": end_date,
        "n_countries": ewis["meta"]["n_countries"], "n_sectors": ewis["meta"]["n_sectors"],
        "n_industries": ewis["meta"]["n_industries"],
        "country_failed": ewis["meta"]["country_meta"]["failed"],
        "sector_failed": ewis["meta"]["sector_meta"]["failed"],
        "industry_failed": ewis["meta"]["industry_meta"]["failed"],
        "non_massive_source": {
            **ewis["meta"]["country_meta"]["non_massive_source"],
            **ewis["meta"]["sector_meta"]["non_massive_source"],
            **ewis["meta"]["industry_meta"]["non_massive_source"],
            **ewis["meta"]["msci_meta"]["non_massive_source"],
        },
    }
    (RESULTS_DIR / "exp1_meta.json").write_text(json.dumps(log_meta, indent=2, default=str))
    print(f"Saved results/exp1_growth_rates.csv, exp1_ewi_comparison.png, exp1_meta.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's 1973-2024 sample window")
    args = parser.parse_args()
    main(full=args.full)
