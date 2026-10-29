import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def calculate_metrics(y_true, y_pred, y_pred_90=None, y_pred_95=None):
    """
    Calculates MAE, RMSE, MAPE, and Pinball Loss (0.5, 0.9, 0.95).
    y_true and y_pred should be inverse-transformed to original scale.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # Avoid division by zero in MAPE
    epsilon = 1e-10
    mape = np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + epsilon))) * 100
    
    pinball_50 = np.mean(np.maximum(0.5 * (y_true - y_pred), -0.5 * (y_true - y_pred)))
    
    res = {
        "MAE": mae,
        "RMSE": rmse,
        "MAPE": mape,
        "Pinball_0.5": pinball_50
    }
    
    if y_pred_90 is not None:
        res["Pinball_0.90"] = np.mean(np.maximum(0.9 * (y_true - y_pred_90), -0.1 * (y_true - y_pred_90)))
    if y_pred_95 is not None:
        res["Pinball_0.95"] = np.mean(np.maximum(0.95 * (y_true - y_pred_95), -0.05 * (y_true - y_pred_95)))
        
    return res

def print_metrics(model_name, metrics):
    print(f"--- {model_name} Evaluation ---")
    print(f"MAE:  {metrics['MAE']:.4f}")
    print(f"RMSE: {metrics['RMSE']:.4f}")
    print(f"MAPE: {metrics['MAPE']:.4f}%")
    if 'Pinball_0.5' in metrics:
        print(f"Pinball (0.5): {metrics['Pinball_0.5']:.4f}")
    print("-" * 30)
