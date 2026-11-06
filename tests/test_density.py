import numpy as np
import pytest
from scipy import stats

from src.density import (
    fit_all_densities,
    fit_normal,
    fit_student_t,
    likelihood_ratio_statistic,
    sgh_logpdf,
)


def test_sgh_logpdf_integrates_to_one_numerically():
    """sgh_logpdf must be a proper density (Sec. 5.1 normal variance-mixture)."""
    grid = np.linspace(-20, 20, 400001)
    log_f = sgh_logpdf(grid, g=0.0, delta=1.0, alpha_bar=0.5, lam=-1.5)
    integral = np.trapz(np.exp(log_f), grid)
    assert integral == pytest.approx(1.0, abs=1e-2)


def test_fit_normal_recovers_known_mean_and_std():
    rng = np.random.default_rng(0)
    x = rng.normal(loc=0.0005, scale=0.01, size=20000)
    fit = fit_normal(x)
    assert fit.params["g"] == pytest.approx(0.0005, abs=2e-4)
    assert fit.params["sigma"] == pytest.approx(0.01, rel=0.05)


def test_fit_student_t_recovers_known_degrees_of_freedom():
    rng = np.random.default_rng(1)
    nu_true = 4.0
    scale_true = 0.01 * np.sqrt((nu_true - 2.0) / nu_true)
    x = stats.t.rvs(df=nu_true, loc=0.0, scale=scale_true, size=30000, random_state=rng)
    fit = fit_student_t(x, method="Nelder-Mead", max_iter=2000)
    assert fit.params["nu"] == pytest.approx(nu_true, rel=0.25)
    assert fit.params["sigma"] == pytest.approx(0.01, rel=0.15)


def test_sgh_log_likelihood_is_at_least_as_high_as_every_nested_case():
    """The 4-parameter SGH nests all 3-parameter special cases, so its
    maximized log-likelihood cannot be lower (Sec. 5.1-5.3)."""
    rng = np.random.default_rng(2)
    x = stats.t.rvs(df=3.5, size=5000, random_state=rng) * 0.01
    fits = fit_all_densities(x, method="Nelder-Mead", max_iter=3000)
    sgh_ll = fits["SGH"].log_likelihood
    for name, fit in fits.items():
        if name != "SGH":
            assert sgh_ll >= fit.log_likelihood - 1e-3, f"{name} beat SGH"


def test_likelihood_ratio_statistic_nonnegative_and_matches_definition():
    assert likelihood_ratio_statistic(ll_nested=100.0, ll_full=101.0) == pytest.approx(2.0)
    assert likelihood_ratio_statistic(ll_nested=100.0, ll_full=100.0) == pytest.approx(0.0)


def test_likelihood_ratio_rejects_normal_but_not_student_t_for_fat_tailed_data():
    """Regression test matching the paper's qualitative finding (Sec. 5.3-5.4):
    a Student t sample's LR statistic against SGH stays far below the 1%
    chi-squared critical value (6.635, 1 d.o.f.), while the normal density's
    LR statistic exceeds it."""
    rng = np.random.default_rng(3)
    x = stats.t.rvs(df=3.3, size=8000, random_state=rng) * 0.008
    fits = fit_all_densities(x, method="Nelder-Mead", max_iter=3000)
    sgh_ll = fits["SGH"].log_likelihood
    l_t = likelihood_ratio_statistic(fits["Student t"].log_likelihood, sgh_ll)
    l_normal = likelihood_ratio_statistic(fits["normal"].log_likelihood, sgh_ll)
    assert l_t < 6.635
    assert l_normal > 6.635
