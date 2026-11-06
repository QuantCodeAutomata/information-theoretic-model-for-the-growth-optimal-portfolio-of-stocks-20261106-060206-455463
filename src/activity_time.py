"""Activity time estimation under the MMM parametrization.

Custom implementation of Platen & Fergusson (2026), Section 2.1, Eq. (2.1):
tau_t = ln([sqrt(S*)]_{t0,t} + exp(tau_t0)), with the quadratic variation
approximated by the cumulative sum of squared increments of sqrt(S*_t)
(Sec. 2.1, p.5-7). No existing library computes this notion of "activity
time"; it is implemented directly from the paper's formula.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import linregress


def quadratic_variation_sqrt(benchmark: pd.Series) -> pd.Series:
    """[sqrt(S*)]_{t0,t}, the cumulative sum of squared increments of sqrt(S*_t).

    Parameters
    ----------
    benchmark:
        Benchmark proxy values S*_t (savings-account/US-Dollar denomination),
        strictly positive, indexed by date in ascending order.

    Returns
    -------
    Series of the cumulative quadratic variation, with value 0 at t_0.
    """
    if (benchmark <= 0).any():
        raise ValueError("benchmark values must be strictly positive")
    sqrt_s = np.sqrt(benchmark)
    increments_sq = sqrt_s.diff().pow(2)
    increments_sq.iloc[0] = 0.0
    return increments_sq.cumsum()


def activity_time_trajectory(benchmark: pd.Series, tau_t0: float) -> pd.Series:
    """tau_t = ln([sqrt(S*)]_{t0,t} + exp(tau_t0)), Eq. (2.1)."""
    qv = quadratic_variation_sqrt(benchmark)
    return np.log(qv + np.exp(tau_t0))


def fit_trendline(tau: pd.Series) -> tuple[float, float, float]:
    """Linear regression of tau_t on calendar time (in years), Sec. 2.1.

    Returns (slope per year, intercept, R^2).
    """
    years = (tau.index - tau.index[0]).days / 365.25
    result = linregress(years.to_numpy(), tau.to_numpy())
    return float(result.slope), float(result.intercept), float(result.rvalue ** 2)


def estimate_initial_activity_time(
    benchmark: pd.Series, tau_t0_grid: np.ndarray
) -> dict:
    """Grid search over tau_t0 maximising the R^2 of the activity-time trendline.

    Sec. 2.1: "the initial activity time is estimated by maximizing the R^2
    score of a linear regression fit to the activity time trajectory."

    Returns a dict with the best tau_t0, slope, intercept, R^2 and the full
    activity-time trajectory at the optimum.
    """
    best = {"tau_t0": None, "slope": None, "intercept": None, "r2": -np.inf}
    for tau_t0 in tau_t0_grid:
        tau = activity_time_trajectory(benchmark, tau_t0)
        slope, intercept, r2 = fit_trendline(tau)
        if r2 > best["r2"]:
            best = {"tau_t0": float(tau_t0), "slope": slope, "intercept": intercept, "r2": r2}
    best["trajectory"] = activity_time_trajectory(benchmark, best["tau_t0"])
    return best


def activity_in_calendar_time(tau: pd.Series) -> pd.Series:
    """a_t = d(tau_t)/dt, numerically differentiated in calendar time (years).

    Finite-difference (forward-difference) derivative; the exact scheme is
    not specified in the paper beyond "activity ... as the derivative of
    the activity time" (Sec. 2.1).
    """
    years = (tau.index - tau.index[0]).days / 365.25
    dt = np.diff(years)
    dtau = np.diff(tau.to_numpy())
    dt = np.where(dt == 0, np.nan, dt)
    activity = dtau / dt
    return pd.Series(activity, index=tau.index[1:])
