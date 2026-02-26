import pandas as pd
import numpy as np
import os
from tqdm import tqdm

from src.data_loader import load_and_aggregate_data
from src.config import KTHParams
from src.utils import rolling_variance_proxy
from src.kth_ipp import generate_fine_series_from_coarse

def generate_augmented_data(filepath="data/histo_trafic.csv", output_path="data/augmented_data.csv"):
    print("Loading empirical data...")
    # Load raw data similar to data_loader but keeping sectors separated
    df = pd.read_csv(filepath, sep=";", encoding="latin1")
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df["tstamp_clean"] = df["tstamp"].str.replace(r"^[a-zA-Zéûîôàç]+\s+", "", regex=True)
    
    month_map = {"janvier":"January","février":"February","mars":"March","avril":"April","mai":"May","juin":"June",
                 "juillet":"July","août":"August","septembre":"September","octobre":"October","novembre":"November","décembre":"December"}
    for fr, en in month_map.items():
        df["tstamp_clean"] = df["tstamp_clean"].str.replace(fr, en, regex=False)
        
    df["tstamp"] = pd.to_datetime(df["tstamp_clean"], format="%d %B %Y")
    df = df.drop(columns="tstamp_clean")
    df = df.sort_values(["secteur", "tstamp"])
    df["trafic_mbps"] = pd.to_numeric(df["trafic_mbps"], errors="coerce")
    df = df.dropna(subset=["trafic_mbps"])
    
    # Use the tuned parameters from config.py directly
    p_macro = KTHParams(dt=60.0)
    
    print(f"Generating fine-grained data for {df['secteur'].nunique()} sectors...")
    print(f"Using tuned parameters: tau={p_macro.tau}, zeta={p_macro.zeta}")
    
    all_fine_data = []
    
    sector_groups = df.groupby("secteur")
    for sector, group in tqdm(sector_groups):
        group = group.sort_values("tstamp")
        coarse_agg = group["trafic_mbps"].to_numpy()
        
        fine_macro, _, _ = generate_fine_series_from_coarse(
            coarse_agg,
            coarse_var=rolling_variance_proxy(coarse_agg, k=6, var_floor_ratio=0.01, tau=p_macro.tau, zeta=p_macro.zeta, T=p_macro.T),
            p=p_macro,
            seed=42
        )
        
        # We also need timestamps for the fine data
        # Coarse points are every 5 mins (300s), dt is 60s
        steps = int(p_macro.T / p_macro.dt)
        n_points = len(coarse_agg) * steps
        fine_series = fine_macro[:n_points]
        
        # Create interpolated timestamps
        first_tstamp = group["tstamp"].iloc[0]
        fine_tstamps = pd.date_range(start=first_tstamp, periods=n_points, freq=f"{int(p_macro.dt)}s")
        
        sector_df = pd.DataFrame({
            "secteur": sector,
            "tstamp": fine_tstamps,
            "trafic_mbps": fine_series
        })
        all_fine_data.append(sector_df)
    
    augmented_df = pd.concat(all_fine_data, ignore_index=True)
    augmented_df.to_csv(output_path, index=False)
    print(f"Augmented dataset saved to {output_path}")

if __name__ == "__main__":
    generate_augmented_data()
