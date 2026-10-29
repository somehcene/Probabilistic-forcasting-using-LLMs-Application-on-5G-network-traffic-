import torch
from chronos import ChronosPipeline

def get_chronos_forecast(context, pred_length=12, model_id="amazon/chronos-t5-small", device_map="cpu", pipeline=None, batch_size=16):
    """
    context: numpy array or torch tensor of shape (batch, seq_len)
    """
    if pipeline is None:
        pipeline = ChronosPipeline.from_pretrained(
            model_id,
            device_map=device_map,
            torch_dtype=torch.float32,
        )

    if not isinstance(context, torch.Tensor):
        context = torch.tensor(context, dtype=torch.float32)
        
    # Chronos expects shape (batch_size, context_length)
    if context.ndim == 1:
        context = context.unsqueeze(0)
    elif context.ndim == 3 and context.shape[2] == 1:
        context = context.squeeze(2)

    # forecast shape: [num_series, num_samples, prediction_length]
    if context.shape[0] > batch_size:
        forecasts = []
        from tqdm import tqdm
        for i in range(0, context.shape[0], batch_size):
            batch = context[i:i+batch_size]
            f = pipeline.predict(batch, prediction_length=pred_length, num_samples=20)
            forecasts.append(f)
        forecast = torch.cat(forecasts, dim=0)
    else:
        forecast = pipeline.predict(
            context,
            prediction_length=pred_length,
            num_samples=20,
        )
    
    # We take the median of the drawn samples as the point forecast, and upper upper limits for bounds
    forecast_50 = torch.quantile(forecast.float(), 0.5, dim=1).numpy()
    forecast_90 = torch.quantile(forecast.float(), 0.90, dim=1).numpy()
    forecast_95 = torch.quantile(forecast.float(), 0.95, dim=1).numpy()
    
    return {"0.5": forecast_50, "0.9": forecast_90, "0.95": forecast_95}

def get_chronos2_forecast(context_list, pred_length=1, pipeline=None, batch_size=16):
    """
    context_list: list of 1D torch tensors
    """
    from chronos import Chronos2Pipeline
    if pipeline is None:
        pipeline = Chronos2Pipeline.from_pretrained(
            "amazon/chronos-2",
            device_map="cpu",
            torch_dtype=torch.float32,
        )

    # forecast shape: [num_series, num_samples, prediction_length]
    if len(context_list) > batch_size:
        forecasts = []
        for i in range(0, len(context_list), batch_size):
            batch = context_list[i:i+batch_size]
            f = pipeline.predict(batch, prediction_length=pred_length)
            if isinstance(f, list):
                f = torch.cat(f, dim=0)
            forecasts.append(f)
        forecast = torch.cat(forecasts, dim=0)
    else:
        forecast = pipeline.predict(
            context_list,
            prediction_length=pred_length,
        )
        # forecast is a list of tensors of shape (1, num_samples, prediction_length) for each series
        if isinstance(forecast, list):
            forecast = torch.cat(forecast, dim=0)
    
    # We extract target quantiles
    forecast_50 = torch.quantile(forecast.float(), 0.5, dim=1).numpy()
    forecast_90 = torch.quantile(forecast.float(), 0.90, dim=1).numpy()
    forecast_95 = torch.quantile(forecast.float(), 0.95, dim=1).numpy()
    
    # After dim 1 reduction, shape is (batch, pred_length). 
    return {"0.5": forecast_50, "0.9": forecast_90, "0.95": forecast_95}
