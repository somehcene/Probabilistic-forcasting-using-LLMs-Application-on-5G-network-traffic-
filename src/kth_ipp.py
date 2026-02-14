# src/kth_ipp.py
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import KTHParams, AugmentationReport


@dataclass
class SlotParams:
    lam: float        # arrivals per second during ON
    psi_mean: float   # mean amount per arrival (Mbits)


def stationary_p_on(tau: float, zeta: float) -> float:
    denom = tau + zeta
    return tau / denom if denom > 0 else 0.0


# ============================================================
# STABLE MOMENT MATCHING (MEAN ONLY)
# ============================================================

def solve_lambda_and_psi_mean(E_rate_mbps: float, p: KTHParams):
    """
    Stable solver:

    - Fix λ globally (controls burstiness)
    - Solve ψ_mean from first moment constraint only

    We preserve:
        E[ coarse rate ] exactly.

    No second-moment estimation (not supported by dataset structure).
    """

    if E_rate_mbps <= 0:
        return SlotParams(lam=p.lambda_min, psi_mean=p.psi_mean_min)

    tau, zeta, T = p.tau, p.zeta, p.T

    # ---- FIX λ (burstiness control parameter) ----
    lam = p.lambda_fixed  # arrivals/sec during ON

    # Expected number of arrivals in slot
    P_on = stationary_p_on(tau, zeta)
    E_U = P_on * lam * T

    # We want:
    # coarse_rate = (E_U * psi_mean) / T
    # => psi_mean = coarse_rate * T / E_U

    psi_mean = (E_rate_mbps * T) / E_U

    return SlotParams(lam=float(lam), psi_mean=float(psi_mean))


# ============================================================
# EXACT EVENT-DRIVEN IPP SIMULATION
# ============================================================

def simulate_ipp_slot(slot_params: SlotParams, p: KTHParams, rng: np.random.Generator) -> np.ndarray:
    """
    Event-driven CTMC IPP.

    ON duration ~ Exp(rate=zeta)
    OFF duration ~ Exp(rate=tau)

    During ON:
        arrivals ~ Poisson(lam * duration)
        each arrival amount ~ Exp(psi_mean)

    Output:
        instantaneous rate in Mbps (amount/dt)
    """

    steps = int(round(p.T / p.dt))
    lam = slot_params.lam
    psi_mean = slot_params.psi_mean

    bin_amount = np.zeros(steps, dtype=float)

    on = rng.random() < stationary_p_on(p.tau, p.zeta)

    t = 0.0
    while t < p.T:
        if on:
            dur = rng.exponential(1.0 / p.zeta)
            t_end = min(t + dur, p.T)
            seg_len = t_end - t

            n = rng.poisson(lam * seg_len)

            if n > 0:
                arr_times = t + rng.random(n) * seg_len
                idx = np.minimum((arr_times / p.dt).astype(int), steps - 1)

                amounts = rng.exponential(scale=psi_mean, size=n)
                bin_amount += np.bincount(idx, weights=amounts, minlength=steps)

            t = t_end
            on = False
        else:
            dur = rng.exponential(1.0 / p.tau)
            t = min(t + dur, p.T)
            on = True

    # Convert amount per bin → Mbps
    return bin_amount / p.dt


# ============================================================
# MAIN GENERATOR
# ============================================================

def generate_fine_series_from_coarse(
    coarse_mbps: np.ndarray,
    coarse_var_unused: np.ndarray,   # kept for API compatibility
    p: KTHParams,
    seed: int = 42
):
    rng = np.random.default_rng(seed)

    coarse_mbps = np.asarray(coarse_mbps, dtype=float)

    steps = int(round(p.T / p.dt))
    N = coarse_mbps.size

    fine = np.zeros(N * steps)
    recon = np.zeros(N)

    for t in range(N):
        sp = solve_lambda_and_psi_mean(coarse_mbps[t], p)

        slot_rate = simulate_ipp_slot(sp, p, rng)

        fine[t*steps:(t+1)*steps] = slot_rate
        recon[t] = slot_rate.mean()

    report = AugmentationReport(
        total_slots=int(N),
        infeasible_lambda_slots=0,
        used_clamp_slots=0,
    )

    return fine, report, recon
