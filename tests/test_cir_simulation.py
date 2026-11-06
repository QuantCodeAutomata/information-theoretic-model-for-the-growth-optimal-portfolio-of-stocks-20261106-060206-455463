import numpy as np
import pytest

from src.cir_simulation import (
    normalized_activity_from_moving_average,
    simulate_cir_via_ou,
    simulate_moving_average,
)


def test_simulate_moving_average_rejects_kappa_out_of_range():
    with pytest.raises(ValueError):
        simulate_moving_average(10, kappa=2.5, a_bar=0.15, delta=1 / 262, seed=0)


def test_simulate_moving_average_initial_condition_and_shapes():
    out = simulate_moving_average(1000, kappa=1.97, a_bar=0.15, delta=1 / 262, seed=1)
    assert out["u"][0] == out["w"][0] == 0.0
    assert len(out["w"]) == len(out["u"]) == len(out["tau"]) == 1001
    assert np.all(np.diff(out["tau"]) > 0)


def test_simulate_moving_average_kappa_one_reduces_to_brownian_motion():
    # kappa=1 => U(tau_i) = W(tau_i) exactly (Eq. 3.18 with no overreaction).
    out = simulate_moving_average(500, kappa=1.0, a_bar=0.15, delta=1 / 262, seed=2)
    np.testing.assert_allclose(out["u"], out["w"])
    assert np.all(out["sgn"] == 0.0)


def test_sign_function_takes_values_in_minus1_0_1():
    out = simulate_moving_average(2000, kappa=1.97, a_bar=0.15, delta=1 / 262, seed=3)
    assert set(np.unique(out["sgn"])).issubset({-1.0, 0.0, 1.0})


def test_normalized_activity_is_nonnegative():
    out = simulate_moving_average(2000, kappa=1.97, a_bar=0.15, delta=1 / 262, seed=4)
    activity = normalized_activity_from_moving_average(out["u"], out["tau"], xi=0.0042)
    assert (activity >= 0).all()
    assert len(activity) == len(out["u"]) - 1


def test_normalized_activity_mean_matches_cir_stationary_mean():
    """a(tau_t) = b(tau_t)^2 is a CIR process of dimension 1 with stationary
    mean xi*kappa_Delta/2 (Sec. 3.2), reproduced by the exact
    Ornstein-Uhlenbeck transition kernel regardless of the step size."""
    kappa, a_bar, delta, xi = 1.97, 0.15, 1 / 262, 0.0042
    kappa_delta = kappa / (a_bar * delta)
    theoretical_mean = xi * kappa_delta / 2.0
    out = simulate_cir_via_ou(60000, kappa=kappa, xi=xi, a_bar=a_bar, delta=delta, seed=5)
    burn_in = 5000
    empirical_mean = np.mean(out["a"][burn_in:])
    assert empirical_mean == pytest.approx(theoretical_mean, rel=0.1)
    assert out["stationary_var"] == pytest.approx(theoretical_mean)


def test_simulate_cir_via_ou_reaches_stationary_mean_from_different_seeds():
    """The exact OU transition kernel converges to the same stationary mean
    xi*kappa_Delta/2 regardless of the random seed, unlike a plain
    Euler-Maruyama discretization of the same SDE whose discrete-time
    stationary variance depends on kappa_Delta*dtau not being small."""
    kappa, a_bar, delta, xi = 1.97, 0.15, 1 / 262, 0.0042
    theoretical_mean = xi * (kappa / (a_bar * delta)) / 2.0
    means = [
        np.mean(simulate_cir_via_ou(60000, kappa=kappa, xi=xi, a_bar=a_bar, delta=delta, seed=s)["a"][5000:])
        for s in (11, 12, 13)
    ]
    for m in means:
        assert m == pytest.approx(theoretical_mean, rel=0.15)


def test_cir_process_is_nonnegative_by_construction():
    out = simulate_cir_via_ou(5000, kappa=1.97, xi=0.0042, a_bar=0.15, delta=1 / 262, seed=6)
    assert (out["a"] >= 0).all()
