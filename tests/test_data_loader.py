"""Tests of the shared data-loading path against a reduced live slice.

These tests only run when DATALOADER_API_KEY and MASSIVE_API_KEY are set; they
load a small, cached slice of the sector-ETF universe used by exp_1/exp_4 and
check the properties the pipeline relies on, not the paper's reported numbers.
"""

import numpy as np
import pandas as pd
import pytest


@pytest.mark.market_data
def test_market_data_slice_has_expected_columns_and_range(market_data_slice: pd.DataFrame) -> None:
    assert set(market_data_slice.columns) == {"Information Technology", "Energy", "Financials"}
    assert isinstance(market_data_slice.index, pd.DatetimeIndex)
    assert market_data_slice.index.is_monotonic_increasing
    assert market_data_slice.index.min() >= pd.Timestamp("2022-01-01")
    assert market_data_slice.index.max() <= pd.Timestamp("2023-12-31")


@pytest.mark.market_data
def test_market_data_slice_prices_are_positive_and_finite(market_data_slice: pd.DataFrame) -> None:
    values = market_data_slice.dropna(how="all").values
    assert np.isfinite(values[~np.isnan(values)]).all()
    assert (values[~np.isnan(values)] > 0).all()
    # Real daily data has day-to-day moves, not a flat line.
    returns = np.log(market_data_slice.dropna()).diff().dropna()
    assert returns.abs().sum().sum() > 0


@pytest.mark.market_data
def test_market_data_slice_feeds_ewi_construction(market_data_slice: pd.DataFrame) -> None:
    """The live slice, reshaped into a one-country (group, country) panel,
    runs through the shared EWI construction without look-ahead or NaNs."""
    from src.ewi import construct_ewi

    panel = market_data_slice.dropna()
    panel.columns = pd.MultiIndex.from_tuples(
        [(name, "US") for name in panel.columns], names=["group", "country"]
    )
    result = construct_ewi(panel, transaction_cost_kbar=0.004)
    assert len(result) == len(panel)
    assert result["ewi"].iloc[0] == pytest.approx(1.0)
    assert np.isfinite(result["ewi"].values).all()
    assert (result["ewi"].values > 0).all()
    assert (result["turnover"].values >= 0).all()
