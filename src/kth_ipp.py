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
    Event-driven CTMC IPP with session spreading.

    ON duration  ~ Exp(rate=zeta)
    OFF duration ~ Exp(rate=tau)

    During ON:
        arrivals ~ Poisson(lam * duration)
        arrival times uniform on the ON interval
        each arrival amount ~ Exp(psi_mean)   (Mbits)

    Each arrival generates a session of duration D ~ Exp(mean=session_mean_s),
    and its amount is spread uniformly over covered dt-bins.

    Output:
        instantaneous rate in Mbps (Mbits/s) sampled every dt seconds.
    """
    T = float(p.T)
    dt = float(p.dt)
    steps = int(round(T / dt))

    tau = float(p.tau)
    zeta = float(p.zeta)
    lam = float(slot_params.lam)
    psi_mean = float(slot_params.psi_mean)

    # --- 1) Simulate CTMC ON/OFF and generate arrival times ---
    t = 0.0
    state_on = False  # start OFF (choice doesn't matter much)
    arr_times_list = []

    while t < T:
        if state_on:
            # ON duration
            if zeta > 0:
                dur = rng.exponential(scale=1.0 / zeta)
            else:
                dur = T - t  # never turns off
        else:
            # OFF duration
            if tau > 0:
                dur = rng.exponential(scale=1.0 / tau)
            else:
                dur = T - t  # never turns on

        t_next = min(T, t + dur)

        if state_on:
            on_len = t_next - t
            if on_len > 0 and lam > 0:
                n = rng.poisson(lam * on_len)
                if n > 0:
                    # uniform arrival times over [t, t_next)
                    arr_times = t + rng.random(n) * on_len
                    arr_times_list.append(arr_times)

        # flip state, move time
        state_on = not state_on
        t = t_next

    if len(arr_times_list) == 0:
        return np.zeros(steps, dtype=float)

    arr_times = np.concatenate(arr_times_list)
    n = arr_times.size

    # --- 2) Draw amounts (Mbits) ---
    # Note: psi_mean might be tiny/huge depending on parameters; keep it >= small eps
    psi_mean_eff = max(psi_mean, 1e-12)
    amounts = rng.exponential(scale=psi_mean_eff, size=n)

    # --- 3) Spread each arrival over a session duration ---
    bin_amount = np.zeros(steps, dtype=float)

    start_idx = np.minimum((arr_times / dt).astype(int), steps - 1)

    # NEW param: p.session_mean_s must exist in KTHParams
    sess_mean = float(getattr(p, "session_mean_s", 30.0))
    sess_mean = max(sess_mean, dt)  # avoid < dt weirdness

    D = rng.exponential(scale=sess_mean, size=n)  # seconds
    end_times = np.minimum(arr_times + D, T)
    end_idx = np.minimum((end_times / dt).astype(int), steps - 1)

    for j in range(n):
        i0 = int(start_idx[j])
        i1 = int(end_idx[j])
        if i1 < i0:
            i1 = i0
        L = (i1 - i0 + 1)
        bin_amount[i0:i1+1] += amounts[j] / L

    # --- 4) Convert Mbits per bin -> Mbps (Mbits/s) ---
    slot_rate = bin_amount / dt
    return slot_rate



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

        target = coarse_mbps[t]   # the original 5-min value (Mbps)
        m = float(slot_rate.mean())

        if target <= 0:
            slot_rate[:] = 0.0
        else:
            if m <= 0:
                # Rare edge case: simulation produced all zeros but target > 0
                # Fallback: constant fill (still preserves exact mean)
                slot_rate[:] = target
            else:
                slot_rate *= (target / m)  # <-- exact conservation step


        fine[t*steps:(t+1)*steps] = slot_rate
        recon[t] = slot_rate.mean()

    report = AugmentationReport(
        total_slots=int(N),
        infeasible_lambda_slots=0,
        used_clamp_slots=0,
    )

    return fine, report, recon
