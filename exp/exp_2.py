"""Experiment 2: Activity time estimation via MMM parametrization and
trendline fitting.

Reproduces Platen & Fergusson (2026), Section 2.1 ("Minimal Market Model
Parametrization"): the activity time tau_t of a benchmark proxy is computed
from the quadratic variation of the square root of the proxy, Eq. (2.1), its
initial value tau_{t0} is estimated by maximising the R^2 of a linear
trendline fitted to the activity-time trajectory, and the activity
a_t = d(tau_t)/dt is computed in both calendar time and activity time.

The paper applies this to the MSCI total return world stock index and the
EWI163 (Sec. 4.1). Both are substituted here by the series built in exp_1
(the MSCI proxy ACWI, and the highest-resolution EWI the available
exchange-traded index funds allow -- see exp_1 and results/RESULTS.md,
"Deviations from the paper").
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

from exp.exp_1 import build_all_ewis
from src.activity_time import activity_in_calendar_time, estimate_initial_activity_time
from src.config import (
    SAMPLE_END_PAPER,
    SAMPLE_END_RUN,
    SAMPLE_START_PAPER,
    SAMPLE_START_RUN,
    TAU_T0_GRID_MAX,
    TAU_T0_GRID_MIN,
    TAU_T0_GRID_STEP,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def run_for_series(name: str, series: pd.Series) -> dict:
    """Grid-search tau_t0, fit the trendline, and compute activity (Sec. 2.1)."""
    series = series.dropna()
    series = series[series > 0]
    grid = np.arange(TAU_T0_GRID_MIN, TAU_T0_GRID_MAX + TAU_T0_GRID_STEP, TAU_T0_GRID_STEP)
    fit = estimate_initial_activity_time(series, grid)
    activity_cal = activity_in_calendar_time(fit["trajectory"])
    return {
        "name": name,
        "tau_t0": fit["tau_t0"],
        "slope": fit["slope"],
        "intercept": fit["intercept"],
        "r2": fit["r2"],
        "trajectory": fit["trajectory"],
        "activity_calendar_time": activity_cal,
    }


def plot_activity_time(result: dict, out_path: Path) -> None:
    tau = result["trajectory"]
    years = (tau.index - tau.index[0]).days / 365.25
    trend = result["intercept"] + result["slope"] * years
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(tau.index, tau.to_numpy(), label=r"activity time $\tau_t$", color="tab:blue")
    ax.plot(tau.index, trend, label="linear trendline", color="tab:red", linestyle="--")
    ax.set_title(f"Activity time of {result['name']} (R$^2$={result['r2']:.3f})")
    ax.set_xlabel("Date")
    ax.set_ylabel(r"$\tau_t$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_activity(result: dict, out_path: Path) -> None:
    activity_cal = result["activity_calendar_time"]
    tau = result["trajectory"]
    activity_in_tau = pd.Series(activity_cal.to_numpy(), index=tau.loc[activity_cal.index].to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(activity_cal.index, activity_cal.to_numpy(), color="tab:green", linewidth=0.6)
    axes[0].set_title(f"Activity $a_t$ in calendar time -- {result['name']}")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel(r"$a_t = d\tau_t/dt$")

    axes[1].plot(activity_in_tau.index, activity_in_tau.to_numpy(), color="tab:purple", linewidth=0.6)
    axes[1].set_title(f"Activity in activity time -- {result['name']}")
    axes[1].set_xlabel(r"$\tau_t$")
    axes[1].set_ylabel(r"$a_t$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(full: bool = False) -> None:
    start_date = SAMPLE_START_PAPER if full else SAMPLE_START_RUN
    end_date = SAMPLE_END_PAPER if full else SAMPLE_END_RUN
    print(f"exp_2: sample window {start_date} to {end_date} (full={full})")

    ewis = build_all_ewis(start_date, end_date)
    msci = ewis["msci"]
    ewi_high = ewis["ewiH"]["ewi"]
    n_industries = ewis["meta"]["n_industries"]

    results = {
        "MSCI": run_for_series("MSCI proxy", msci),
        "EWI_high": run_for_series(f"EWI{n_industries} proxy", ewi_high),
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table_rows = []
    for key, res in results.items():
        plot_activity_time(res, RESULTS_DIR / f"exp2_activity_time_{key}.png")
        plot_activity(res, RESULTS_DIR / f"exp2_activity_{key}.png")
        table_rows.append({
            "series": res["name"], "tau_t0": res["tau_t0"],
            "trendline_slope": res["slope"], "r2": res["r2"],
        })
    table = pd.DataFrame(table_rows)
    print(table.to_string(index=False))
    table.to_csv(RESULTS_DIR / "exp2_activity_time_fit.csv", index=False)

    meta = {"start_date": start_date, "end_date": end_date, "n_industries": n_industries}
    (RESULTS_DIR / "exp2_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("Saved results/exp2_activity_time_fit.csv and exp2_* plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's 1973-2024 sample window")
    args = parser.parse_args()
    main(full=args.full)
