# src/utils.py
from __future__ import annotations
import numpy as np
import pandas as pd


def rolling_variance_proxy(coarse_mean, k, var_floor_ratio, *, tau, zeta, T, eps=1e-6):
    x = np.asarray(coarse_mean, dtype=float)
    s = pd.Series(x)

    v = s.rolling(window=k, min_periods=max(3, k // 4)).var(ddof=0).to_numpy()
    global_v = float(np.nanvar(x)) if np.isfinite(np.nanvar(x)) else 0.0
    v = np.where(np.isfinite(v), v, global_v)

    # generic floor
    floor_generic = (var_floor_ratio * np.maximum(x, 0.0)) ** 2
    v = np.maximum(v, floor_generic)

    # KTH feasibility floor (Eq.11 condition)
    correction = (2.0 * zeta) / (tau * T * (tau + zeta))
    disp_min = correction + eps
    floor_kth = disp_min * (np.maximum(x, 0.0) ** 2)
    v = np.maximum(v, floor_kth)

    return v



def coarsen_fine_to_slots(fine: np.ndarray, steps_per_slot: int) -> np.ndarray:
    """
    Sum fine samples inside each coarse slot.
    Used to build Fig.6(a) orange curve.
    """
    fine = np.asarray(fine, dtype=float)
    if fine.size % steps_per_slot != 0:
        raise ValueError("fine length must be multiple of steps_per_slot.")
    return fine.reshape(-1, steps_per_slot).sum(axis=1)


def summary_errors(a: np.ndarray, b: np.ndarray) -> dict:
    """
    Simple summary stats comparing two coarse series (trend sanity check).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    eps = 1e-12

    mae = float(np.mean(np.abs(a - b)))
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    mape = float(np.mean(np.abs(a - b) / (np.abs(a) + eps))) * 100.0

    return {"mae": mae, "rmse": rmse, "mape_%": mape}
