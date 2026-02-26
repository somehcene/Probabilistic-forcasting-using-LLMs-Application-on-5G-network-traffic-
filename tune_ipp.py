import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.config import KTHParams
from src.utils import rolling_variance_proxy
from src.kth_ipp import generate_fine_series_from_coarse
from src.data_loader import load_and_aggregate_data

def test_ipp_parameters(df_agg, tau, zeta, title_suffix=""):
    coarse_agg = df_agg["trafic_mbps"].to_numpy()[:500] # Use only 500 samples for faster tuning
    
    # Set up the IPP parameters representing the aggregate model
    p_macro = KTHParams(
        T=300.0,       # 5 minutes coarse duration
        tau=tau,       # OFF -> ON rate
        zeta=zeta,     # ON -> OFF rate
        lambda_fixed=0.5, 
        dt=60.0        # Granularité 60s
    )

    # Generate the fine synthetic traffic
    fine_macro, report_macro, recon_macro = generate_fine_series_from_coarse(
        coarse_agg,
        coarse_var=rolling_variance_proxy(coarse_agg, k=6, var_floor_ratio=0.01, tau=p_macro.tau, zeta=p_macro.zeta, T=p_macro.T),
        p=p_macro,
        seed=42
    )

    # Prepare for plotting
    steps = int(p_macro.T / p_macro.dt)
    coarse_step = np.repeat(coarse_agg, steps)
    fine_series = fine_macro[:len(coarse_step)]

    # Slice a small segment to see bursts more clearly (e.g., first 50 coarse steps = 250 fine steps)
    plot_len = min(250, len(coarse_step))
    
    plt.figure(figsize=(14, 5))
    plt.plot(coarse_step[:plot_len], label="Coarse (Real Aggregated)", alpha=0.5, color='tab:blue', linewidth=2)
    plt.plot(fine_series[:plot_len], label=f"Simulated (tau={tau:.3f}, zeta={zeta:.3f})", alpha=0.8, color='tab:red', linewidth=1)
    
    plt.title(f"Aggregated Traffic Simulation {title_suffix}\nMean P(ON) = {tau/(tau+zeta):.2f}")
    plt.xlabel("Time (minutes)")
    plt.ylabel("Traffic Demand (Mbps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save instead of forcing interactive block
    filename = f"ipp_tune_tau{tau:.3f}_zeta{zeta:.3f}.png"
    plt.savefig(filename, bbox_inches='tight')
    print(f"Saved plot to {filename}")
    plt.close()


if __name__ == "__main__":
    df_agg = load_and_aggregate_data()
    
    # The original parameters were tau=1/15 (0.067), zeta=1/15 (0.067)
    # A smaller mean parameter (higher rate) means faster switching -> shorter ON/OFF periods -> smoother averaging over the 60s bin
    
    # 1. Original
    test_ipp_parameters(df_agg, tau=1/15, zeta=1/15, title_suffix="(Original)")
    
    # 2. Faster switching, same proportion (P(ON) = 0.5)
    test_ipp_parameters(df_agg, tau=1/2, zeta=1/2, title_suffix="(Fast Switching 1)")
    
    # 3. Even faster switching, same proportion
    test_ipp_parameters(df_agg, tau=1/0.5, zeta=1/0.5, title_suffix="(Very Fast Switching)")
    
    # 4. Stay ON longer, OFF shorter
    test_ipp_parameters(df_agg, tau=1/5, zeta=1/20, title_suffix="(P(ON)=0.8)")
    
    print("Done generating tuning plots.")
