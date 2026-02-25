import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from src.config import KTHParams
from src.utils import rolling_variance_proxy
from src.kth_ipp import generate_fine_series_from_coarse

def load_and_aggregate_data(filepath="data/histo_trafic.csv"):
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
    
    df_agg = df.groupby("tstamp")["trafic_mbps"].sum().reset_index()
    df_agg = df_agg.sort_values("tstamp")
    return df_agg

def get_synthesized_data(df_agg, dt=60.0):
    coarse_agg = df_agg["trafic_mbps"].to_numpy()
    p_macro = KTHParams(
        T=300.0,
        tau=1/15,
        zeta=1/15,
        lambda_fixed=0.5, 
        dt=dt
    )
    
    fine_macro, report_macro, recon_macro = generate_fine_series_from_coarse(
        coarse_agg,
        coarse_var=rolling_variance_proxy(coarse_agg, k=6, var_floor_ratio=0.01, tau=p_macro.tau, zeta=p_macro.zeta, T=p_macro.T),
        p=p_macro,
        seed=42
    )
    return fine_macro

class TimeSeriesDataset(Dataset):
    def __init__(self, data, seq_length, pred_length):
        self.data = torch.FloatTensor(data)
        self.seq_length = seq_length
        self.pred_length = pred_length
        
    def __len__(self):
        return len(self.data) - self.seq_length - self.pred_length + 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx + self.seq_length]
        y = self.data[idx + self.seq_length:idx + self.seq_length + self.pred_length]
        return x, y

def prepare_dataloaders(data, seq_length=60, pred_length=12, batch_size=32, train_split=0.8, val_split=0.1):
    # Determine split indices
    train_size = int(len(data) * train_split)
    val_size = int(len(data) * val_split)
    
    # Scale data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))
    
    train_data = data_scaled[:train_size]
    val_data = data_scaled[train_size:train_size+val_size]
    test_data = data_scaled[train_size+val_size:]
    
    # Create datasets
    train_dataset = TimeSeriesDataset(train_data, seq_length, pred_length)
    val_dataset = TimeSeriesDataset(val_data, seq_length, pred_length)
    test_dataset = TimeSeriesDataset(test_data, seq_length, pred_length)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, scaler, train_data, val_data, test_data
