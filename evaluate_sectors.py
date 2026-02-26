import pandas as pd
import numpy as np
import warnings
from tqdm import tqdm

from src.evaluate import calculate_metrics
from src.models.sarima_wrapper import get_sarima_forecast

# We will conditionally import chronos to avoid loading massive models if not needed instantly
from src.models.chronos_wrapper import get_chronos_forecast
import torch

def import_torch_dtype():
    return torch.float32

warnings.filterwarnings("ignore")

def evaluate_models_on_sectors(filepath="data/augmented_data.csv", pred_length=12, context_length=60):
    print(f"Loading augmented dataset from {filepath}...")
    df = pd.read_csv(filepath)
    df["tstamp"] = pd.to_datetime(df["tstamp"])
    
    sectors = df["secteur"].unique()
    print(f"Found {len(sectors)} sectors. Evaluating SARIMA vs Chronos 2 on the last {pred_length} points.")
    
    # Pre-load Chronos Pipeline to avoid loading it 86 times
    from chronos import ChronosPipeline
    print("Pre-loading Chronos 2 Pipeline...")
    pipeline = ChronosPipeline.from_pretrained(
        "amazon/chronos-t5-small",
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
        
    print(f"\nRunning Chronos 2 on a batch of {len(all_histories)} sectors...")
    
    # 2. Chronos 2 Forecast (Batched)
    history_tensor = torch.tensor(np.array(all_histories), dtype=torch.float32)
    chronos_preds = get_chronos_forecast(history_tensor, pred_length=pred_length, model_id="amazon/chronos-t5-small", pipeline=pipeline)
    
    chronos_rmses = []
    for i in range(len(valid_sectors)):
        actual = all_actuals[i]
        pred = chronos_preds[i]
        chronos_metrics = calculate_metrics(actual, pred)
        chronos_rmses.append(chronos_metrics["RMSE"])
        
    avg_sarima_rmse = np.mean(sarima_rmses) if sarima_rmses else 0
    avg_chronos_rmse = np.mean(chronos_rmses) if chronos_rmses else 0
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS (RMSE on last measurement of all sectors)")
    print("="*50)
    print(f"Average SARIMA RMSE : {avg_sarima_rmse:.4f}")
    print(f"Average Chronos 2 RMSE : {avg_chronos_rmse:.4f}")
    print("="*50)
    
if __name__ == "__main__":
    evaluate_models_on_sectors()
