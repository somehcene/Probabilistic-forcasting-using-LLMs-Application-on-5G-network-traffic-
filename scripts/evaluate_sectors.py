# Run from anywhere: make the repo root the working dir and importable
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

from src.evaluate import calculate_metrics
from src.models.sarima_wrapper import get_sarima_forecast

# We will conditionally import chronos to avoid loading massive models if not needed instantly
from src.models.chronos_wrapper import get_chronos_forecast, get_chronos2_forecast
import torch

def import_torch_dtype():
    return torch.float32

warnings.filterwarnings("ignore")

def evaluate_models_on_sectors(filepath="data/augmented_data.csv", pred_length=1, context_length=60):
    print(f"Loading augmented dataset from {filepath}...")
    df = pd.read_csv(filepath)
    df["tstamp"] = pd.to_datetime(df["tstamp"])
    
    sectors = df["secteur"].unique()
    print(f"Found {len(sectors)} sectors. Evaluating SARIMA vs Chronos v1 vs Chronos-2 on the last {pred_length} points.")
    
    from chronos import ChronosPipeline, Chronos2Pipeline
    print("Pre-loading Chronos Pipelines...")
    pipeline_v1 = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
        device_map="cpu",
        torch_dtype=import_torch_dtype(),
    )
    pipeline_v2 = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map="cpu",
        torch_dtype=import_torch_dtype(),
    )
    
    sarima_rmses = []
    
    # Storage for batching Chronos
    all_actuals = []
    all_histories = []
    valid_sectors = []
    
    for sector in tqdm(sectors, desc="Running SARIMA and collecting histories"):
        sector_df = df[df["secteur"] == sector].sort_values("tstamp")
        series = sector_df["trafic_mbps"].to_numpy()
        
        if len(series) < context_length + pred_length:
            continue
            
        actual = series[-pred_length:]
        history = series[-(context_length + pred_length):-pred_length]
        
        # 1. SARIMA Forecast
        sarima_pred = get_sarima_forecast(history, pred_length)
        sarima_metrics = calculate_metrics(actual, sarima_pred)
        sarima_rmses.append(sarima_metrics["RMSE"])
        
        # Collect for Chronos
        all_actuals.append(actual)
        all_histories.append(history)
        valid_sectors.append(sector)
        
    print(f"\nRunning Chronos models on a batch of {len(all_histories)} sectors...")
    
    # List of 1D tensors required for Chronos2 (and Chronos1 batch)
    history_tensors_list = [torch.tensor(h, dtype=torch.float32) for h in all_histories]
    history_tensor_stacked = torch.stack(history_tensors_list)
    
    # 2. Chronos v1 Forecast (Batched)
    chronos_preds_v1 = get_chronos_forecast(history_tensor_stacked, pred_length=pred_length, model_id="amazon/chronos-t5-small", pipeline=pipeline_v1)
    
    # 3. Chronos 2 Forecast (Batched)
    chronos_preds_v2 = get_chronos2_forecast(history_tensors_list, pred_length=pred_length, pipeline=pipeline_v2)
    
    chronos_rmses_v1 = []
    chronos_rmses_v2 = []
    for i in range(len(valid_sectors)):
        actual = all_actuals[i]
        
        metrics_v1 = calculate_metrics(actual, chronos_preds_v1[i])
        chronos_rmses_v1.append(metrics_v1["RMSE"])
        
        metrics_v2 = calculate_metrics(actual, chronos_preds_v2[i])
        chronos_rmses_v2.append(metrics_v2["RMSE"])
        
    avg_sarima_rmse = np.mean(sarima_rmses) if sarima_rmses else 0
    avg_chronos_v1_rmse = np.mean(chronos_rmses_v1) if chronos_rmses_v1 else 0
    avg_chronos_v2_rmse = np.mean(chronos_rmses_v2) if chronos_rmses_v2 else 0
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS (RMSE on last measurement of all sectors)")
    print("="*50)
    print(f"Average SARIMA RMSE : {avg_sarima_rmse:.4f} (Baseline)")
    print(f"Average Chronos-T5 RMSE : {avg_chronos_v1_rmse:.4f} (V1)")
    print(f"Average Chronos-2 RMSE : {avg_chronos_v2_rmse:.4f} (State-of-the-Art)")
    print("="*50)
    
if __name__ == "__main__":
    evaluate_models_on_sectors()
