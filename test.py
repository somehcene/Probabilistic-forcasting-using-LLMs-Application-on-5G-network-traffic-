from src.data_loader import load_and_aggregate_data, get_synthesized_data, prepare_dataloaders
from src.models.chronos_wrapper import get_chronos_forecast

df_agg = load_and_aggregate_data()
fine_data = get_synthesized_data(df_agg, dt=60.0)
train_loader, val_loader, test_loader, scaler, _, _, _ = prepare_dataloaders(
    fine_data, seq_length=60, pred_length=12, batch_size=32)

test_seq, test_labels = next(iter(test_loader))
chronos_preds = get_chronos_forecast(test_seq.squeeze(-1), pred_length=12, model_id="amazon/chronos-t5-small")

print("test_seq shape:", test_seq.shape)
print("test_labels shape:", test_labels.shape)
print("chronos_preds type:", type(chronos_preds))
print("chronos_preds shape:", getattr(chronos_preds, "shape", len(chronos_preds)))
idx = 0
unscaled = scaler.inverse_transform(chronos_preds[idx].reshape(-1, 1))
print("unscaled shape:", unscaled.shape)
actual = scaler.inverse_transform(test_labels[idx].numpy())
print("actual shape:", actual.shape)

from src.evaluate import calculate_metrics
print("lstm calc:", calculate_metrics(actual, unscaled))
