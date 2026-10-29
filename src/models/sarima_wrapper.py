import pmdarima as pm
import numpy as np
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.simplefilter('ignore', ConvergenceWarning)

def get_sarima_forecast(history: np.ndarray, pred_length: int) -> np.ndarray:
    """
    Fits an ARIMA(1,0,1) model on the historical data and predicts the next 'pred_length' steps.
    """
    # Ensure history is a 1D numpy array
    history = np.asarray(history).flatten()
    
    # Fit fixed ARIMA
    try:
        model = pm.ARIMA(order=(1,0,1), suppress_warnings=True)
        model.fit(history)
        
        forecast_mean, conf_90_bounds = model.predict(n_periods=pred_length, return_conf_int=True, alpha=0.20)
        _, conf_95_bounds = model.predict(n_periods=pred_length, return_conf_int=True, alpha=0.10)
        
        forecast = forecast_mean
        forecast_90 = conf_90_bounds[:, 1] # Upper bound of 80% CI is the 90th percentile
        forecast_95 = conf_95_bounds[:, 1] # Upper bound of 90% CI is the 95th percentile
        
    except Exception as e:
        print(f"SARIMA fitting failed: {e}. Falling back to naive forecast.")
        forecast = np.full(pred_length, history[-1])
        forecast_90 = forecast.copy()
        forecast_95 = forecast.copy()
        
    return {"0.5": forecast, "0.9": forecast_90, "0.95": forecast_95}
