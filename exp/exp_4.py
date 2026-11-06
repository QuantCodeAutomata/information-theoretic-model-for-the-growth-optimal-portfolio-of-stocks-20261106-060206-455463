"""Experiment 4: Log-return density estimation and likelihood ratio testing
for benchmark proxies.

Reproduces Platen & Fergusson (2026), Section 5 ("Characterisation and
Estimation of Log-Return Densities"): the symmetric generalized hyperbolic
(SGH) density and its nested special cases (variance-gamma, hyperbolic,
normal-inverse Gaussian, Student t, normal) are fitted by maximum likelihood
to the daily log-returns of each benchmark proxy, and a likelihood ratio
test, Eq. (5.3) with statistic L = -2 ln(Lambda) (Sec. 5.3), checks whether
the Student t density can be rejected against the four-parameter SGH density
at the 1% significance level.

The indices fitted are the MSCI proxy and the three equal-weighted indices
built in exp_1 (EWI1, EWI11 and the highest-resolution EWI available from
exchange-traded index funds), in place of the paper's MSCI and EWI1/11/25/74
/163 built from a 67-country Thomson-Reuters GICS panel (see exp_1 and
results/RESULTS.md, "Deviations from the paper").
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
from src.config import (
    CHI2_CRITICAL_99_DF1,
    CHI2_DF_NORMAL_VS_SGH,
    LR_TEST_SIGNIFICANCE_LEVEL,
    MLE_MAX_ITER,
    MLE_OPTIMIZER_METHOD,
    SAMPLE_END_PAPER,
    SAMPLE_END_RUN,
    SAMPLE_START_PAPER,
    SAMPLE_START_RUN,
)
from src.density import fit_all_densities, likelihood_ratio_statistic

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def log_returns(series: pd.Series) -> np.ndarray:
    """Daily log-returns x_i = ln(S_{t_i}/S_{t_{i-1})), Sec. 5.1."""
    series = series.dropna()
    series = series[series > 0]
    r = np.log(series).diff().dropna()
    return r.to_numpy()


def analyze_series(name: str, x: np.ndarray) -> dict:
    """Fit every density, run the likelihood-ratio test against SGH (Sec. 5.3)."""
    fits = fit_all_densities(x, MLE_OPTIMIZER_METHOD, MLE_MAX_ITER)
    ll_sgh = fits["SGH"].log_likelihood

    rows = []
    for key, fit in fits.items():
        df = CHI2_DF_NORMAL_VS_SGH if key == "normal" else 1
        if key == "SGH":
            lr_stat, critical, rejected = np.nan, np.nan, False
        else:
            lr_stat = likelihood_ratio_statistic(fit.log_likelihood, ll_sgh)
            critical = CHI2_CRITICAL_99_DF1 if df == 1 else None
            rejected = (lr_stat > critical) if critical is not None else None
        rows.append({
            "index": name, "density": key, "n_params": fit.n_params,
            "log_likelihood": fit.log_likelihood, "lr_statistic": lr_stat,
            "chi2_critical_99": critical,
            "rejected_at_1pct": rejected,
            "params": json.dumps(fit.params),
        })
    return {"rows": rows, "fits": fits, "x": x}


def plot_density_fit(name: str, x: np.ndarray, fits: dict, out_path: Path) -> None:
    from src.density import sgh_logpdf

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.hist(x, bins=200, density=True, color="lightgray", label="empirical log-returns")
    grid = np.linspace(np.percentile(x, 0.1), np.percentile(x, 99.9), 1000)

    sgh = fits["SGH"].params
    pdf_sgh = np.exp(sgh_logpdf(grid, sgh["g"], sgh["delta"], sgh["alpha_bar"], sgh["lambda"]))
    ax.plot(grid, pdf_sgh, color="tab:red", label="fitted SGH")

    t = fits["Student t"].params
    from scipy import stats
    scale = t["sigma"] * np.sqrt((t["nu"] - 2.0) / t["nu"])
    pdf_t = stats.t.pdf(grid, df=t["nu"], loc=t["g"], scale=scale)
    ax.plot(grid, pdf_t, color="tab:blue", linestyle="--", label=f"fitted Student t (nu={t['nu']:.2f})")

    ax.set_yscale("log")
    ax.set_title(f"Log-return density fit -- {name}")
    ax.set_xlabel("daily log-return")
    ax.set_ylabel("density (log scale)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main(full: bool = False) -> None:
    start_date = SAMPLE_START_PAPER if full else SAMPLE_START_RUN
    end_date = SAMPLE_END_PAPER if full else SAMPLE_END_RUN
    print(f"exp_4: sample window {start_date} to {end_date} (full={full})")

    ewis = build_all_ewis(start_date, end_date)
    n_industries = ewis["meta"]["n_industries"]
    series_map = {
        "MSCI proxy": ewis["msci"],
        "EWI1": ewis["ewi1"]["ewi"],
        "EWI11": ewis["ewi11"]["ewi"],
        f"EWI{n_industries}": ewis["ewiH"]["ewi"],
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for name, series in series_map.items():
        x = log_returns(series)
        print(f"{name}: {len(x)} daily log-returns, mean={x.mean():.6f}, std={x.std():.6f}")
        result = analyze_series(name, x)
        all_rows.extend(result["rows"])
        safe_name = name.replace(" ", "_")
        plot_density_fit(name, x, result["fits"], RESULTS_DIR / f"exp4_density_{safe_name}.png")

    table = pd.DataFrame(all_rows)
    print(table[["index", "density", "n_params", "log_likelihood", "lr_statistic", "rejected_at_1pct"]]
          .to_string(index=False))
    table.to_csv(RESULTS_DIR / "exp4_density_fits.csv", index=False)

    student_t_rows = table[table["density"] == "Student t"]
    summary = student_t_rows[["index", "lr_statistic", "chi2_critical_99", "rejected_at_1pct"]].copy()
    summary["significance_level"] = LR_TEST_SIGNIFICANCE_LEVEL
    summary.to_csv(RESULTS_DIR / "exp4_student_t_lr_test.csv", index=False)
    print(summary.to_string(index=False))

    meta = {"start_date": start_date, "end_date": end_date, "n_industries": n_industries}
    (RESULTS_DIR / "exp4_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print("Saved results/exp4_density_fits.csv, exp4_student_t_lr_test.csv, exp4_density_*.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's 1973-2024 sample window")
    args = parser.parse_args()
    main(full=args.full)
