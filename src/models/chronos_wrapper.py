import torch
from chronos import ChronosPipeline

def get_chronos_forecast(context, pred_length=12, model_id="amazon/chronos-t5-small", device_map="cpu", pipeline=None):
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
    forecast = pipeline.predict(
        context,
        prediction_length=pred_length,
        num_samples=20,
    )
    
    # We take the median of the drawn samples as the point forecast
    point_forecast = torch.quantile(forecast, 0.5, dim=1)
    
    return point_forecast.numpy()
