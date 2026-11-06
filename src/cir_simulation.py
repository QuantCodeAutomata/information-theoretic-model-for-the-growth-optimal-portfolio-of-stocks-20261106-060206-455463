"""Simulation of the normalized activity as an ergodic scalar diffusion.

Custom implementation of Platen & Fergusson (2026), Section 3.1-3.2,
Eq. (3.18), (3.21), (3.25)-(3.27): the moving-average overreaction model and
its CIR-process limit for the normalized activity. ``hmmlearn``/``arch`` do
not implement this specific predictor-corrector moving-average construction,
so it is implemented directly from the paper's recursion.
"""

from __future__ import annotations

import numpy as np


def simulate_moving_average(
    n_steps: int, kappa: float, a_bar: float, delta: float, seed: int
) -> dict:
    """Simulate W (benchmark Brownian motion) and its moving average U.

    Eq. (3.18): U(tau_ti) = kappa * W(tau_ti) + (1 - kappa) * U(tau_t{i-1}),
    with U(tau_t0) = W(tau_t0). The normalized activity time increment is
    constant for constant average activity: tau_ti - tau_t{i-1} = a_bar *
    delta (Sec. 3.1, implementation note).

    Returns a dict with arrays ``tau`` (normalized activity time grid),
    ``w`` (Brownian motion), ``u`` (moving average), ``sgn`` (sign function
    of Eq. (3.26)) and ``brownian_increments`` (the underlying dW draws).
    """
    if not (0 < kappa <= 2):
        raise ValueError("kappa must lie in (0, 2]; the paper studies kappa in (1, 2)")
    rng = np.random.default_rng(seed)
    dtau = a_bar * delta
    tau = np.arange(n_steps + 1) * dtau

    dw = rng.standard_normal(n_steps) * np.sqrt(dtau)
    w = np.empty(n_steps + 1)
    w[0] = 0.0
    w[1:] = np.cumsum(dw)

    u = np.empty(n_steps + 1)
    u[0] = w[0]
    for i in range(1, n_steps + 1):
        u[i] = kappa * w[i] + (1.0 - kappa) * u[i - 1]

    diff = w - u
    sgn = np.sign(diff)  # Eq. (3.26): sgn(tau_t) = (W-U)/|W-U|, 0 if W=U (np.sign(0)=0)

    return {"tau": tau, "w": w, "u": u, "sgn": sgn, "dtau": dtau}


def normalized_activity_from_moving_average(u: np.ndarray, tau: np.ndarray, xi: float) -> np.ndarray:
    """a(tau_ti) = xi * ((U(tau_ti) - U(tau_t{i-1})) / (tau_ti - tau_t{i-1}))^2.

    Methodology step 6 of exp_3 (equivalent to differentiating Eq. (3.21)).
    """
    du = np.diff(u)
    dtau = np.diff(tau)
    return xi * (du / dtau) ** 2


def simulate_cir_via_ou(
    n_steps: int, kappa: float, xi: float, a_bar: float, delta: float, seed: int
) -> dict:
    """Simulate the normalized activity as the square of an OU process.

    Eq. (3.27): db(tau_t) = -kappa_Delta b(tau_t) d tau_t + sqrt(xi) kappa_Delta dW,
    a(tau_t) = b(tau_t)^2, kappa_Delta = kappa / (a_bar * Delta).

    b is a mean-zero OU process with stationary variance
    (sqrt(xi)*kappa_Delta)^2 / (2*kappa_Delta) = xi*kappa_Delta/2, so
    a = b^2 is a CIR process of dimension 1 with stationary mean
    xi*kappa_Delta/2 (Sec. 3.2).

    Because kappa_Delta * dtau = kappa (close to 2, not small -- that is the
    point of the overreaction parametrization), a plain Euler-Maruyama step
    does not reproduce this stationary variance: it discretizes the mean
    -reversion term as (1 - kappa_Delta*dtau) = (1-kappa), which is only a
    good approximation of exp(-kappa_Delta*dtau) when kappa_Delta*dtau is
    small. Instead this uses the exact Ornstein-Uhlenbeck transition kernel
    (Kloeden & Platen, "Numerical Solution of SDEs", the exact scheme for a
    linear mean-reverting SDE), which reproduces the SDE's stationary
    variance xi*kappa_Delta/2 for any step size:
        b_i = phi * b_{i-1} + sqrt(stationary_var * (1 - phi^2)) * Z_i,
        phi = exp(-kappa_Delta * dtau).
    """
    kappa_delta = kappa / (a_bar * delta)
    rng = np.random.default_rng(seed)
    dtau = a_bar * delta
    tau = np.arange(n_steps + 1) * dtau
    stationary_var = xi * kappa_delta / 2.0
    phi = np.exp(-kappa_delta * dtau)
    innovation_std = np.sqrt(stationary_var * (1.0 - phi**2))
    z = rng.standard_normal(n_steps)
    b = np.empty(n_steps + 1)
    b[0] = 0.0
    for i in range(1, n_steps + 1):
        b[i] = phi * b[i - 1] + innovation_std * z[i - 1]
    a = b**2
    return {"tau": tau, "b": b, "a": a, "kappa_delta": kappa_delta, "stationary_var": stationary_var}
