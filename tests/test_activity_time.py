import numpy as np
import pandas as pd
import pytest

from src.activity_time import (
    activity_in_calendar_time,
    activity_time_trajectory,
    estimate_initial_activity_time,
    fit_trendline,
    quadratic_variation_sqrt,
)


def _make_benchmark(n=1000, mu=0.0002, sigma=0.01, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    log_ret = mu + sigma * rng.standard_normal(n)
    log_ret[0] = 0.0
    values = np.exp(np.cumsum(log_ret))
    return pd.Series(values, index=dates)


def test_quadratic_variation_is_nondecreasing_and_starts_at_zero():
    s = _make_benchmark()
    qv = quadratic_variation_sqrt(s)
    assert qv.iloc[0] == 0.0
    assert (qv.diff().dropna() >= 0).all()


def test_quadratic_variation_rejects_nonpositive_values():
    s = pd.Series([1.0, -1.0, 2.0], index=pd.date_range("2020-01-01", periods=3))
    with pytest.raises(ValueError):
        quadratic_variation_sqrt(s)


def test_activity_time_trajectory_monotone_in_tau_t0():
    s = _make_benchmark()
    tau_low = activity_time_trajectory(s, tau_t0=-10.0)
    tau_high = activity_time_trajectory(s, tau_t0=-1.0)
    # A higher initial activity time shifts the whole trajectory up (Eq. 2.1).
    assert (tau_high.to_numpy() >= tau_low.to_numpy() - 1e-12).all()


def test_fit_trendline_recovers_known_linear_series():
    dates = pd.date_range("2000-01-01", periods=500, freq="D")
    years = (dates - dates[0]).days / 365.25
    tau = pd.Series(2.0 + 0.1 * years, index=dates)
    slope, intercept, r2 = fit_trendline(tau)
    assert slope == pytest.approx(0.1, rel=1e-6)
    assert intercept == pytest.approx(2.0, rel=1e-6)
    assert r2 == pytest.approx(1.0, abs=1e-9)


def test_estimate_initial_activity_time_r2_is_bounded_and_high_for_smooth_benchmark():
    s = _make_benchmark(n=3000, seed=1)
    grid = np.arange(-10.0, 0.0, 0.1)
    best = estimate_initial_activity_time(s, grid)
    assert 0.0 <= best["r2"] <= 1.0
    # A roughly geometric-Brownian benchmark should give an activity time that
    # is close to linear in calendar time (Sec. 2.1 prediction).
    assert best["r2"] > 0.8
    assert best["tau_t0"] == grid[np.argmin(np.abs(grid - best["tau_t0"]))]


def test_activity_in_calendar_time_matches_manual_finite_difference():
    s = _make_benchmark(n=200, seed=2)
    tau = activity_time_trajectory(s, tau_t0=-5.0)
    activity = activity_in_calendar_time(tau)
    years = (tau.index - tau.index[0]).days / 365.25
    manual = np.diff(tau.to_numpy()) / np.diff(years)
    np.testing.assert_allclose(activity.to_numpy(), manual)
