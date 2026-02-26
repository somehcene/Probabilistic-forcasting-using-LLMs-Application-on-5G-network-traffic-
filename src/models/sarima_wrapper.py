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
        forecast = model.predict(n_periods=pred_length)
    except Exception as e:
        print(f"SARIMA fitting failed: {e}. Falling back to naive forecast.")
        forecast = np.full(pred_length, history[-1])
        
    return forecast
