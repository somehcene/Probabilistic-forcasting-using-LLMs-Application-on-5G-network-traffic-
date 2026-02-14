# src/config.py
from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class KTHParams:
    """
    Parameters for the KTH IPP-based augmentation.

    Notes
    -----
    - Coarse slot duration is T seconds (teacher: treat each Bouygues sample as a 5-min average => T=300).
    - Fine step is dt seconds (e.g., 5s => 60 fine steps per coarse slot).
    - ON/OFF CTMC:
        OFF -> ON rate: tau
        ON  -> OFF rate: zeta
      Steady-state:
        P(ON) = tau / (tau + zeta)
    """
    T: float = 300.0
    dt: float = 5.0

    # ON/OFF CTMC parameters
    tau: float = 1/30     # OFF -> ON rate
    zeta: float = 1/10    # ON -> OFF rate

    # Fixed arrival rate during ON
    lambda_fixed: float = 0.5

    # safety floors
    lambda_min: float = 1e-6
    psi_mean_min: float = 1e-9
    


@dataclass(frozen=True)
class AugmentationReport:
    """Simple counters for sanity / grading writeup."""
    total_slots: int
    infeasible_lambda_slots: int
    used_clamp_slots: int
