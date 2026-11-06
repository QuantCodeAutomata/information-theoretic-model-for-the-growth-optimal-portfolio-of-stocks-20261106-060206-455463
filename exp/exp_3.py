"""Experiment 3: Simulation of the normalized activity model with
overreactions in trading behaviour.

Reproduces Platen & Fergusson (2026), Section 3.1 ("Overreactions in Trading
Behaviour") and Section 3.2 ("Normalized Activity as Ergodic Scalar
Diffusion"): a moving average U of a Brownian motion W is simulated with
smoothing parameter kappa in (1, 2) to capture overreactions (Eq. 3.18), the
normalized activity a is derived from U's increments (Eq. 3.21, a squared,
high-frequency-differenced estimator that is inherently rough), and its
smooth continuous-time counterpart is simulated as the square of an
Ornstein-Uhlenbeck process (Eq. 3.27), which is exactly a CIR process of
dimension 1 with stationary gamma(1) density (Sec. 3.2).

No external data is required; every parameter is the paper's EWI163 estimate
(Sec. 3.1-3.2): kappa=1.97, xi=0.0042, a_bar=0.15, Delta=1/262.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.cir_simulation import (
    normalized_activity_from_moving_average,
    simulate_cir_via_ou,
    simulate_moving_average,
)
from src.config import (
    CIR_A_BAR,
    CIR_DELTA,
    CIR_KAPPA,
    CIR_RANDOM_SEED,
    CIR_SIMULATION_STEPS_PAPER,
    CIR_SIMULATION_STEPS_RUN,
    CIR_XI,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def plot_overreaction(sim: dict, out_path: Path, n_show: int = 2000) -> None:
    """Panel (a): Brownian motion and its moving average, overreaction effect."""
    idx = slice(0, n_show)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sim["tau"][idx], sim["w"][idx], label=r"$W_{\bar\tau_t}$ (benchmark)", color="tab:blue")
    ax.plot(sim["tau"][idx], sim["u"][idx], label=r"$U(\bar\tau_t)$ (moving average)", color="tab:orange")
    ax.set_title(f"Overreaction effect: moving average vs. Brownian motion (kappa={CIR_KAPPA})")
    ax.set_xlabel(r"normalized activity time $\bar\tau_t$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_driving_brownian_motions(sim: dict, out_path: Path, n_show: int = 2000) -> None:
    """Panel (b): W_tau vs. the cumulative integral of sgn(tau) dW_tau."""
    idx = slice(0, n_show)
    dw = np.diff(sim["w"])
    sgn_dw_integral = np.concatenate([[0.0], np.cumsum(sim["sgn"][:-1] * dw)])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(sim["tau"][idx], sim["w"][idx], label=r"$W_{\bar\tau_t}$", color="tab:blue")
    ax.plot(sim["tau"][idx], sgn_dw_integral[idx], label=r"$\int \mathrm{sgn}(\bar\tau_s)\,dW_{\bar\tau_s}$", color="tab:red")
    ax.set_title("Brownian motions driving the benchmark and the normalized activity")
    ax.set_xlabel(r"normalized activity time $\bar\tau_t$")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return sgn_dw_integral


def plot_activity_calendar_and_tau(a_rough: np.ndarray, cir: dict, delta: float, out_path: Path) -> None:
    """Panels (c)-(d): the rough moving-average estimator in calendar time,
    and its smooth CIR-process counterpart in normalized activity time."""
    years = np.arange(len(a_rough)) * delta
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(years, a_rough, color="tab:green", linewidth=0.5)
    axes[0].set_title("Normalized activity estimator in calendar time (rough, spiky)")
    axes[0].set_xlabel("calendar time (years)")
    axes[0].set_ylabel(r"$a(\bar\tau_{t_i})$")

    axes[1].plot(cir["tau"], cir["a"], color="tab:purple", linewidth=0.6)
    axes[1].set_title("CIR-process counterpart in normalized activity time (smooth)")
    axes[1].set_xlabel(r"normalized activity time $\bar\tau_t$")
    axes[1].set_ylabel(r"$a(\bar\tau_t)=b(\bar\tau_t)^2$")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def check_stationary_gamma(cir: dict, kappa: float, a_bar: float, delta: float, xi: float) -> dict:
    """Stationary density of a=b^2 (OU square) is gamma(1) with mean xi*kappa_Delta/2
    (Sec. 3.2); fit a gamma distribution to the simulated path (after burn-in)."""
    kappa_delta = kappa / (a_bar * delta)
    theoretical_mean = xi * kappa_delta / 2.0
    burn_in = len(cir["a"]) // 10
    sample = cir["a"][burn_in:]
    shape_hat, loc_hat, scale_hat = stats.gamma.fit(sample, floc=0)
    ks_stat, ks_pvalue = stats.kstest(sample, "gamma", args=(shape_hat, loc_hat, scale_hat))
    return {
        "theoretical_mean": theoretical_mean,
        "sample_mean": float(np.mean(sample)),
        "fitted_gamma_shape": shape_hat,
        "fitted_gamma_scale": scale_hat,
        "ks_statistic": ks_stat,
        "ks_pvalue": ks_pvalue,
    }


def main(full: bool = False) -> None:
    n_steps = CIR_SIMULATION_STEPS_PAPER if full else CIR_SIMULATION_STEPS_RUN
    print(f"exp_3: simulating {n_steps} steps (kappa={CIR_KAPPA}, xi={CIR_XI}, "
          f"a_bar={CIR_A_BAR}, delta={CIR_DELTA:.6f}), full={full}")

    sim = simulate_moving_average(n_steps, CIR_KAPPA, CIR_A_BAR, CIR_DELTA, CIR_RANDOM_SEED)
    a_rough = normalized_activity_from_moving_average(sim["u"], sim["tau"], CIR_XI)
    cir = simulate_cir_via_ou(n_steps, CIR_KAPPA, CIR_XI, CIR_A_BAR, CIR_DELTA, CIR_RANDOM_SEED + 1)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    plot_overreaction(sim, RESULTS_DIR / "exp3_overreaction.png")
    sgn_dw_integral = plot_driving_brownian_motions(sim, RESULTS_DIR / "exp3_driving_brownian_motions.png")
    plot_activity_calendar_and_tau(a_rough, cir, CIR_DELTA, RESULTS_DIR / "exp3_activity_calendar_vs_tau.png")

    gamma_check = check_stationary_gamma(cir, CIR_KAPPA, CIR_A_BAR, CIR_DELTA, CIR_XI)

    # Covariation <W, int sgn dW> average over the path: the mean sign should
    # be close to zero because overreactions switch sign symmetrically.
    mean_sgn = float(np.mean(sim["sgn"]))
    dw = np.diff(sim["w"])
    sample_covariation = float(np.mean(dw * (sim["sgn"][:-1] * dw)) / CIR_DELTA)

    summary = {
        "n_steps": n_steps,
        "kappa": CIR_KAPPA,
        "xi": CIR_XI,
        "a_bar": CIR_A_BAR,
        "delta": CIR_DELTA,
        "mean_sgn": mean_sgn,
        "sample_covariation_rate": sample_covariation,
        **gamma_check,
    }
    print(json.dumps(summary, indent=2))
    (RESULTS_DIR / "exp3_summary.json").write_text(json.dumps(summary, indent=2))
    print("Saved results/exp3_summary.json and exp3_* plots")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true", help="Use the paper's full ~50-year simulation length")
    args = parser.parse_args()
    main(full=args.full)
