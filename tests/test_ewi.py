import numpy as np
import pandas as pd
import pytest

from src.ewi import (
    annualized_volatility,
    average_annualized_growth_rate,
    compute_weights,
    construct_ewi,
)


def test_compute_weights_time_varying_group_count(synthetic_level_panel):
    weights = compute_weights(synthetic_level_panel)
    # Day 0: group B absent in JP -> N_sector=2, N_sector_country(A)=2, N_sector_country(B)=1
    day0 = weights.iloc[0]
    assert day0[("A", "US")] == pytest.approx(1.0 / (2 * 2))
    assert day0[("A", "JP")] == pytest.approx(1.0 / (2 * 2))
    assert day0[("B", "US")] == pytest.approx(1.0 / (2 * 1))
    assert day0[("B", "JP")] == 0.0
    # Day 5 onward: group B present in both countries -> N_sector_country(B)=2
    day5 = weights.iloc[5]
    assert day5[("B", "US")] == pytest.approx(1.0 / (2 * 2))
    assert day5[("B", "JP")] == pytest.approx(1.0 / (2 * 2))
    # weights within a day across the whole grid sum to <= 1 for each group's
    # equal split and are always non-negative
    assert (weights.to_numpy() >= 0).all()


def test_compute_weights_rejects_non_multiindex():
    df = pd.DataFrame({"a": [1.0, 2.0]})
    with pytest.raises(ValueError):
        compute_weights(df)


def test_construct_ewi_zero_transaction_cost_matches_weighted_return(synthetic_level_panel):
    ewi = construct_ewi(synthetic_level_panel, transaction_cost_kbar=0.0)
    assert ewi["ewi"].iloc[0] == 1.0
    # With zero transaction cost EWI == EWI pre-cost exactly.
    np.testing.assert_allclose(ewi["ewi"].to_numpy(), ewi["ewi_pre_cost"].to_numpy())
    assert (ewi["turnover"] >= 0).all()


def test_construct_ewi_positive_transaction_cost_reduces_value(synthetic_level_panel):
    ewi_free = construct_ewi(synthetic_level_panel, transaction_cost_kbar=0.0)
    ewi_cost = construct_ewi(synthetic_level_panel, transaction_cost_kbar=0.004)
    # Turnover scales with the previous period's value (Eq. 4.1-4.2), so the
    # *relative* turnover TO(t_i)/EWId(t_{i-1}) is identical with and without
    # cost (weights depend only on presence/absence, not on the EWI level).
    rel_turnover_free = ewi_free["turnover"] / ewi_free["ewi"].shift(1)
    rel_turnover_cost = ewi_cost["turnover"] / ewi_cost["ewi"].shift(1)
    np.testing.assert_allclose(
        rel_turnover_free.dropna().to_numpy(), rel_turnover_cost.dropna().to_numpy(), rtol=1e-9
    )
    assert (ewi_cost["ewi"].to_numpy()[1:] <= ewi_free["ewi"].to_numpy()[1:] + 1e-12).all()


def test_construct_ewi_single_constant_group_has_zero_turnover():
    dates = pd.date_range("2021-01-01", periods=5, freq="D")
    cols = pd.MultiIndex.from_tuples([("A", "US")], names=["group", "country"])
    values = pd.DataFrame({("A", "US"): [1.0, 1.02, 1.01, 1.05, 1.04]}, index=dates)
    values.columns = cols
    ewi = construct_ewi(values, transaction_cost_kbar=0.004)
    # A single always-present series is never rebalanced, so turnover is 0
    # and the EWI equals the raw index exactly.
    np.testing.assert_allclose(ewi["turnover"].to_numpy(), 0.0, atol=1e-12)
    np.testing.assert_allclose(ewi["ewi"].to_numpy(), values[("A", "US")].to_numpy())


def test_average_annualized_growth_rate_known_value():
    # Deterministic daily gross return of exp(0.001) gives known annualised growth.
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    values = pd.Series(np.exp(0.001 * np.arange(100)), index=dates)
    g = average_annualized_growth_rate(values, trading_days_per_year=262)
    assert g == pytest.approx(0.001 * 262, rel=1e-6)


def test_annualized_volatility_nonnegative(synthetic_level_panel):
    ewi = construct_ewi(synthetic_level_panel, transaction_cost_kbar=0.004)
    vol = annualized_volatility(ewi["ewi"], trading_days_per_year=262)
    assert vol >= 0


def test_higher_resolution_grid_can_increase_growth_rate():
    """Synthetic check of the paper's qualitative hypothesis (Sec. 4): equally
    weighting across more independent groups of idiosyncratic-noise series
    raises the average growth rate relative to a single aggregate series,
    holding the common drift fixed (Sec. 2.1, Theorem 3.6 in Platen 2026b)."""
    rng = np.random.default_rng(42)
    n = 2000
    dates = pd.date_range("2000-01-01", periods=n, freq="D")
    common_drift = 0.0002
    common_vol = 0.01

    def make_panel(n_groups):
        cols = pd.MultiIndex.from_tuples(
            [(f"G{j}", "US") for j in range(n_groups)], names=["group", "country"]
        )
        idio_vol = 0.03
        shocks = rng.normal(0, 1, size=(n, n_groups))
        common_shock = rng.normal(0, 1, size=n)
        log_ret = common_drift + common_vol * common_shock[:, None] + idio_vol * shocks
        log_ret[0, :] = 0.0
        values = np.exp(np.cumsum(log_ret, axis=0))
        return pd.DataFrame(values, index=dates, columns=cols)

    g_low = average_annualized_growth_rate(
        construct_ewi(make_panel(1), 0.0)["ewi"], trading_days_per_year=262
    )
    g_high = average_annualized_growth_rate(
        construct_ewi(make_panel(20), 0.0)["ewi"], trading_days_per_year=262
    )
    assert g_high > g_low
