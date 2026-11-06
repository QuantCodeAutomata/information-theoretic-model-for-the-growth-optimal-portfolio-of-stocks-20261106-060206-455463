"""Symmetric generalized hyperbolic (SGH) log-return density estimation.

Custom implementation of Platen & Fergusson (2026), Section 5.1
("Characterisation of Log-Return Densities"), using the normal
variance-mixture representation X = g + sqrt(V) * Z, V ~ GIG(alpha, delta,
lambda), Z ~ N(0,1), and its nested special cases (variance-gamma,
hyperbolic, normal-inverse Gaussian, Student t, normal).

SciPy provides ``scipy.stats.genhyperbolic``, but in a different ("4th",
Prause 1999) parametrization (shapes p, a, b) that is not delta/alpha-bar
based, so using it would still require re-deriving the paper's (g, sigma,
alpha_bar, lambda) parametrization and its sigma-delta conversion (Sec. 5.1)
on top of it. The density and its nested cases are therefore implemented
directly from the paper's formulae, using
``scipy.special.kv`` (SciPy) for the modified Bessel function of the second
kind and ``scipy.optimize.minimize`` for maximum likelihood estimation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln, kv


def _log_kv(order: float, z: np.ndarray) -> np.ndarray:
    """log K_order(z) for z > 0, stable for the ranges used here."""
    value = kv(order, z)
    value = np.clip(value, a_min=1e-300, a_max=None)
    return np.log(value)


def sgh_logpdf(x: np.ndarray, g: float, delta: float, alpha_bar: float, lam: float) -> np.ndarray:
    """log-density of the general SGH family, Sec. 5.1 (delta != 0, alpha_bar > 0).

    f_X(x) = (1/(delta K_lam(alpha_bar))) * sqrt(alpha_bar/(2 pi))
             * (1 + (x-g)^2/delta^2)^{(lam-1/2)/2}
             * K_{lam-1/2}(alpha_bar * sqrt(1 + (x-g)^2/delta^2))
    """
    z = (x - g) / delta
    q = np.sqrt(1.0 + z**2)
    log_norm = -np.log(delta) - _log_kv(lam, np.asarray(alpha_bar)) + 0.5 * np.log(alpha_bar / (2 * np.pi))
    log_term = ((lam - 0.5) / 2.0) * np.log(q**2) + _log_kv(lam - 0.5, alpha_bar * q)
    return log_norm + log_term


def variance_gamma_logpdf(x: np.ndarray, g: float, sigma: float, lam: float) -> np.ndarray:
    """log-density of the symmetric variance-gamma special case (delta=0, lambda>0).

    Obtained as the delta -> 0 limit of ``sgh_logpdf`` with alpha = sqrt(2*lam)/sigma
    held fixed (Sec. 5.1):
    f(x) = alpha^{lam+1/2} |x-g|^{lam-1/2} K_{lam-1/2}(alpha |x-g|) / (Gamma(lam) 2^{lam-1} sqrt(2 pi))
    """
    alpha = np.sqrt(2.0 * lam) / sigma
    absz = np.abs(x - g)
    absz = np.where(absz == 0, 1e-12, absz)
    log_f = (
        (lam + 0.5) * np.log(alpha)
        + (lam - 0.5) * np.log(absz)
        + _log_kv(lam - 0.5, alpha * absz)
        - gammaln(lam)
        - (lam - 1.0) * np.log(2.0)
        - 0.5 * np.log(2 * np.pi)
    )
    return log_f


def sgh_sigma_from_delta(delta: float, alpha_bar: float, lam: float) -> float:
    """sigma = delta^2 K_{lam+1}(alpha_bar) / (alpha_bar K_lam(alpha_bar)), Sec. 5.1."""
    return float(delta**2 * kv(lam + 1.0, alpha_bar) / (alpha_bar * kv(lam, alpha_bar)))


@dataclass
class DensityFit:
    name: str
    n_params: int
    params: dict
    log_likelihood: float


def _neg_loglik(logpdf_values: np.ndarray) -> float:
    if not np.all(np.isfinite(logpdf_values)):
        return 1e12
    return -float(np.sum(logpdf_values))


def fit_sgh(x: np.ndarray, method: str, max_iter: int) -> DensityFit:
    """Fit the full 4-parameter SGH density (g, sigma, alpha_bar, lambda)."""
    g0, sigma0 = float(np.mean(x)), float(np.std(x))

    def objective(theta):
        g, log_delta, log_alpha_bar, lam = theta
        delta, alpha_bar = np.exp(log_delta), np.exp(log_alpha_bar)
        return _neg_loglik(sgh_logpdf(x, g, delta, alpha_bar, lam))

    x0 = np.array([g0, np.log(sigma0), np.log(0.3), -1.5])
    res = minimize(objective, x0, method=method, options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-8})
    g, log_delta, log_alpha_bar, lam = res.x
    delta, alpha_bar = np.exp(log_delta), np.exp(log_alpha_bar)
    sigma = sgh_sigma_from_delta(delta, alpha_bar, lam)
    ll = -res.fun
    return DensityFit(
        "SGH", 4, {"g": g, "sigma": sigma, "alpha_bar": alpha_bar, "lambda": lam, "delta": delta}, ll
    )


def fit_restricted_sgh(x: np.ndarray, lam_fixed: float, name: str, method: str, max_iter: int) -> DensityFit:
    """Fit SGH with lambda fixed (hyperbolic: lam=1, NIG: lam=-0.5); 3 free params."""
    g0, sigma0 = float(np.mean(x)), float(np.std(x))

    def objective(theta):
        g, log_delta, log_alpha_bar = theta
        delta, alpha_bar = np.exp(log_delta), np.exp(log_alpha_bar)
        return _neg_loglik(sgh_logpdf(x, g, delta, alpha_bar, lam_fixed))

    x0 = np.array([g0, np.log(sigma0), np.log(0.3)])
    res = minimize(objective, x0, method=method, options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-8})
    g, log_delta, log_alpha_bar = res.x
    delta, alpha_bar = np.exp(log_delta), np.exp(log_alpha_bar)
    sigma = sgh_sigma_from_delta(delta, alpha_bar, lam_fixed)
    ll = -res.fun
    return DensityFit(
        name, 3, {"g": g, "sigma": sigma, "alpha_bar": alpha_bar, "lambda": lam_fixed, "delta": delta}, ll
    )


def fit_variance_gamma(x: np.ndarray, method: str, max_iter: int) -> DensityFit:
    """Fit the variance-gamma special case (g, sigma, lambda), delta=0, lambda>0."""
    g0, sigma0 = float(np.mean(x)), float(np.std(x))

    def objective(theta):
        g, log_sigma, log_lam = theta
        sigma, lam = np.exp(log_sigma), np.exp(log_lam)
        return _neg_loglik(variance_gamma_logpdf(x, g, sigma, lam))

    x0 = np.array([g0, np.log(sigma0), np.log(1.5)])
    res = minimize(objective, x0, method=method, options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-8})
    g, log_sigma, log_lam = res.x
    sigma, lam = np.exp(log_sigma), np.exp(log_lam)
    ll = -res.fun
    return DensityFit("variance-gamma", 3, {"g": g, "sigma": sigma, "lambda": lam, "alpha_bar": 0.0}, ll)


def fit_student_t(x: np.ndarray, method: str, max_iter: int) -> DensityFit:
    """Fit the Student t special case (g, sigma, nu), alpha_bar=0, lambda=-nu/2.

    Uses scipy.stats.t with scale = sigma * sqrt((nu-2)/nu) so that the
    fitted variable has standard deviation sigma, matching the paper's
    (g, sigma) convention (Sec. 5.1).
    """
    g0, sigma0 = float(np.mean(x)), float(np.std(x))

    def objective(theta):
        g, log_sigma, log_nu_minus_2 = theta
        sigma = np.exp(log_sigma)
        nu = 2.0 + np.exp(log_nu_minus_2)
        scale = sigma * np.sqrt((nu - 2.0) / nu)
        return _neg_loglik(stats.t.logpdf(x, df=nu, loc=g, scale=scale))

    x0 = np.array([g0, np.log(sigma0), np.log(2.0)])
    res = minimize(objective, x0, method=method, options={"maxiter": max_iter, "xatol": 1e-8, "fatol": 1e-8})
    g, log_sigma, log_nu_minus_2 = res.x
    sigma = np.exp(log_sigma)
    nu = 2.0 + np.exp(log_nu_minus_2)
    ll = -res.fun
    return DensityFit("Student t", 3, {"g": g, "sigma": sigma, "nu": nu, "lambda": -nu / 2.0, "alpha_bar": 0.0}, ll)


def fit_normal(x: np.ndarray) -> DensityFit:
    """Fit the normal special case (g, sigma) via its closed-form MLE."""
    g, sigma = float(np.mean(x)), float(np.std(x))
    ll = float(np.sum(stats.norm.logpdf(x, loc=g, scale=sigma)))
    return DensityFit("normal", 2, {"g": g, "sigma": sigma}, ll)


def fit_all_densities(x: np.ndarray, method: str, max_iter: int) -> dict[str, DensityFit]:
    """Fit the SGH density and all five nested special cases (Sec. 5.1-5.2)."""
    x = np.asarray(x, dtype=float)
    fits = {
        "SGH": fit_sgh(x, method, max_iter),
        "variance-gamma": fit_variance_gamma(x, method, max_iter),
        "hyperbolic": fit_restricted_sgh(x, 1.0, "hyperbolic", method, max_iter),
        "NIG": fit_restricted_sgh(x, -0.5, "NIG", method, max_iter),
        "Student t": fit_student_t(x, method, max_iter),
        "normal": fit_normal(x),
    }
    return fits


def likelihood_ratio_statistic(ll_nested: float, ll_full: float) -> float:
    """L = -2 ln(Lambda) = -2 (ll_nested - ll_full), Sec. 5.3."""
    return float(-2.0 * (ll_nested - ll_full))
