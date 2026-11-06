"""Construction of Equal-Weighted Indices (EWIs).

Custom implementation of the recursive EWI formulae in Platen & Fergusson
(2026), Section 4.1 ("Construction of Equal-Weighted Indices"), Eq. (4.1)-
(4.3). This is a bespoke accounting recursion for a transaction-cost-aware
equal-weighted rebalancing scheme; no portfolio-optimization library
implements it, so it is implemented directly from the paper's formulae.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_weights(level_values: pd.DataFrame) -> pd.DataFrame:
    """Equal weights pi^{j,k}(t_i) of Sec. 4.1, Eq. (4.3).

    Parameters
    ----------
    level_values:
        Wide DataFrame indexed by date, one column per (classification
        group j, country k) series S^{j,k}(t). A value of 0 or NaN means
        the group/country pair had no index value at that date (the
        country or classification group was not present).

    Returns
    -------
    DataFrame of the same shape with pi^{j,k}(t_i) = 1/(N_sector(t_i) *
    N_sector_country^j(t_i)) where S^{j,k}(t_i) > 0, else 0. Columns must
    be a MultiIndex with levels ("group", "country") so that the sector
    count and the sector-country count can be computed.
    """
    if not isinstance(level_values.columns, pd.MultiIndex):
        raise ValueError("level_values.columns must be a MultiIndex (group, country)")
    present = level_values.fillna(0.0) > 0.0

    groups = level_values.columns.get_level_values("group")
    unique_groups = pd.unique(groups)

    # N_sector(t_i): number of classification groups present in ANY country.
    group_present = pd.DataFrame(
        {g: present.loc[:, groups == g].any(axis=1) for g in unique_groups}
    )
    n_sector = group_present.sum(axis=1)

    # N_sector_country^j(t_i): number of countries present in group j.
    n_sector_country = pd.DataFrame(
        {g: present.loc[:, groups == g].sum(axis=1) for g in unique_groups}
    )

    weights = pd.DataFrame(0.0, index=level_values.index, columns=level_values.columns)
    for g in unique_groups:
        cols = level_values.columns[groups == g]
        denom = n_sector.mul(n_sector_country[g], axis=0).replace(0.0, np.nan)
        w = (1.0 / denom).fillna(0.0)
        for c in cols:
            weights[c] = np.where(present[c].to_numpy(), w.to_numpy(), 0.0)
    return weights


def construct_ewi(
    level_values: pd.DataFrame,
    transaction_cost_kbar: float,
) -> pd.DataFrame:
    """Recursive EWI construction, Sec. 4.1, Eq. (4.1)-(4.3).

    Parameters
    ----------
    level_values:
        Wide DataFrame indexed by date (ascending), MultiIndex columns
        (group, country), of market-cap-weighted index values S^{j,k}(t)
        in US Dollars. Missing/zero entries mean the group-country pair
        is absent at that date.
    transaction_cost_kbar:
        Proportional transaction cost kappa-bar per US Dollar of turnover.

    Returns
    -------
    DataFrame with columns ``ewi`` (EWId(t_i), initial value 1 at t_0),
    ``turnover`` (TO(t_i)) and ``ewi_pre_cost`` (EWId(t_i^-)).
    """
    weights = compute_weights(level_values)
    dates = level_values.index
    n = len(dates)
    if n < 2:
        raise ValueError("construct_ewi requires at least two observation dates")

    ret = (level_values / level_values.shift(1)).to_numpy()  # S(ti)/S(ti-1)
    w = weights.to_numpy()
    ret = np.where(np.isfinite(ret), ret, 0.0)

    ewi = np.empty(n)
    ewi_pre_cost = np.empty(n)
    turnover = np.zeros(n)
    ewi[0] = 1.0
    ewi_pre_cost[0] = 1.0

    for i in range(1, n):
        w_prev = w[i - 1]
        w_curr = w[i]
        gross_return = np.sum(w_prev * ret[i])
        pre_cost = ewi[i - 1] * gross_return
        drifted_prev_value = w_prev * ewi[i - 1] * ret[i]
        target_value = w_curr * pre_cost
        turnover[i] = np.sum(np.abs(target_value - drifted_prev_value))
        ewi_pre_cost[i] = pre_cost
        ewi[i] = pre_cost - transaction_cost_kbar * turnover[i]

    return pd.DataFrame(
        {"ewi": ewi, "ewi_pre_cost": ewi_pre_cost, "turnover": turnover}, index=dates
    )


def average_annualized_growth_rate(index_values: pd.Series, trading_days_per_year: int) -> float:
    """Average annualised growth rate g = mean(log-return) * trading days/year.

    The growth rate of a portfolio value process is the drift of its
    logarithm; the paper reports growth rates "p.a." (Sec. 4.2, Table 5.8),
    i.e. the mean daily log-return scaled by the trading-day count.
    """
    log_returns = np.log(index_values / index_values.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("index_values must contain at least two observations")
    return float(log_returns.mean() * trading_days_per_year)


def annualized_volatility(index_values: pd.Series, trading_days_per_year: int) -> float:
    """Annualised volatility: std(log-return) * sqrt(trading days/year)."""
    log_returns = np.log(index_values / index_values.shift(1)).dropna()
    if log_returns.empty:
        raise ValueError("index_values must contain at least two observations")
    return float(log_returns.std(ddof=1) * np.sqrt(trading_days_per_year))
