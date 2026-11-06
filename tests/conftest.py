import os

import numpy as np
import pandas as pd
import pytest


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "market_data: tests that load a reduced real market-data slice"
    )


@pytest.fixture(scope="session")
def has_api_keys() -> bool:
    return bool(os.environ.get("DATALOADER_API_KEY")) and bool(os.environ.get("MASSIVE_API_KEY"))


@pytest.fixture(scope="session")
def market_data_slice(request, has_api_keys):
    """A few ETFs, two years, loaded once per session through data.data_loader."""
    if not has_api_keys:
        pytest.skip("DATALOADER_API_KEY / MASSIVE_API_KEY not set")
    cache = request.config.cache
    cached = cache.get("market_data_slice/panel", None)
    if cached is not None:
        return pd.DataFrame(cached).assign(date=lambda d: pd.to_datetime(d["date"])).set_index("date")

    from data.data_loader import load_close_panel

    tickers = {"XLK": "Information Technology", "XLE": "Energy", "XLF": "Financials"}
    panel, _meta = load_close_panel(tickers, "2022-01-01", "2023-12-31")
    cache.set("market_data_slice/panel", panel.reset_index().assign(date=lambda d: d["date"].astype(str)).to_dict("list"))
    return panel


@pytest.fixture
def synthetic_level_panel():
    """Synthetic (group, country) panel of cap-weighted index values with a
    known answer: two groups, two countries, one group present in only one
    country starting midway, to exercise the time-varying weight formula."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2020-01-01", periods=10, freq="D")
    cols = pd.MultiIndex.from_tuples(
        [("A", "US"), ("A", "JP"), ("B", "US"), ("B", "JP")], names=["group", "country"]
    )
    data = pd.DataFrame(1.0, index=dates, columns=cols)
    growth = 1.0 + rng.normal(0, 0.001, size=(10, 4))
    growth[0, :] = 1.0
    data.loc[:, :] = np.cumprod(growth, axis=0)
    # Group B absent in Japan for the first 5 days (not yet listed).
    data.loc[dates[:5], ("B", "JP")] = 0.0
    return data
