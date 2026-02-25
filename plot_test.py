import numpy as np
import matplotlib.pyplot as plt
from src.config import KTHParams
from src.kth_ipp import solve_lambda_and_psi_mean, simulate_ipp_slot

def generate_slot(target, use_scaling=True, seed=None):
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
        
    p = KTHParams()
    # Assume arbitrary variance (e.g. 5x target mean for spike tests)
    sp = solve_lambda_and_psi_mean(target, target * 100, p)
    
    slot_rate = simulate_ipp_slot(sp, p, rng)
    m = float(slot_rate.mean())
    
    if target <= 0:
        return np.zeros_like(slot_rate)
        
    if use_scaling:
        if m > 0:
            slot_rate *= (target / m)
        else:
            slot_rate[:] = target
            
    return slot_rate

# Generate 5 slots of target=50 Mbps
target = 50.0
fine_scaled = []
fine_unscaled = []

# fix seed to compare exactly the same arrivals
rng = np.random.default_rng(42)
for _ in range(10):
    seed = rng.integers(0, 10000)
    fine_s = generate_slot(target, use_scaling=True, seed=seed)
    fine_u = generate_slot(target, use_scaling=False, seed=seed)
    fine_scaled.extend(fine_s)
    fine_unscaled.extend(fine_u)

print("Max scaled:", np.max(fine_scaled))
print("Max unscaled:", np.max(fine_unscaled))
