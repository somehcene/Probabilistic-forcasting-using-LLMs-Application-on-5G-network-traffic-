#!/usr/bin/env python
# coding: utf-8

# # Forecasting Pipeline: LSTM vs Chronos2
# 
# In this notebook, we load the aggregated, fine-grained IPP synthesized 6G network traffic data, and we train a PyTorch LSTM model alongside a zero-shot evaluation on Amazon's Chronos2 Foundation Model.
# 
# We then plot the results to visually understand their forecasting capabilities on micro-bursts vs macro trends.

# In[ ]:


import torch
import matplotlib.pyplot as plt
import numpy as np
from src.data_loader import load_and_aggregate_data, get_synthesized_data, prepare_dataloaders
from src.models.lstm import TrafficLSTM, train_model
from src.models.chronos_wrapper import get_chronos_forecast
from src.evaluate import calculate_metrics, print_metrics


# ## 1. Load and Prepare 6G Traffic Data
# We use the fine-grained data mapped via Continuous-Time Markov Chain IPP logic to capture real network volatility.

# In[ ]:


# Load empirical data and synthesize fine-grained IPP data
df_agg = load_and_aggregate_data()
fine_data = get_synthesized_data(df_agg, dt=60.0)

# Set Forecasting Window Specs
SEQ_LENGTH = 60    # 60 past points 
PRED_LENGTH = 12   # 12 future points

# Create Datasets and Loaders
train_loader, val_loader, test_loader, scaler, train_scaled, val_scaled, test_scaled = prepare_dataloaders(
    fine_data, 
    seq_length=SEQ_LENGTH, 
    pred_length=PRED_LENGTH, 
    batch_size=32
)

print(f"Data Lengths -> Train: {len(train_scaled)} | Val: {len(val_scaled)} | Test: {len(test_scaled)}")


# ## 2. PyTorch LSTM Model Training
# Here we train the LSTM to identify the historical sequential dependencies.

# In[ ]:


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

lstm_model = TrafficLSTM(
    input_size=1, 
    hidden_layer_size=64, 
    num_layers=2, 
    output_size=PRED_LENGTH
)

# Train (Uncomment to train - requires a few minutes depending on CPU/GPU)
# lstm_model = train_model(
#     lstm_model, 
#     train_loader, 
#     val_loader, 
#     epochs=30, 
#     patience=5, 
#     device=device
# )


# ## 3. Evaluation and Visualization on Test Data
# We evaluate a slice of the test data (e.g., pulling a random batch) and compare LSTM and Chronos zero-shot capabilities.

# In[ ]:


# Let's take the very first batch of the test data for evaluation & plotting
lstm_model.eval()
test_seq, test_labels = next(iter(test_loader))

with torch.no_grad():
    # LSTM Prediction
    lstm_preds = lstm_model(test_seq.to(device)).cpu().numpy()

    # Chronos Prediction
    # Chronos expects the context data. Removing the feature dim (since it's only 1 feature)
    chronos_preds = get_chronos_forecast(test_seq.squeeze(-1), pred_length=PRED_LENGTH, model_id="amazon/chronos-t5-small")

    # Evaluate on the first sample in the batch
    idx = 0
    actual = scaler.inverse_transform(test_labels[idx].numpy())
    lstm_pred_unscaled = scaler.inverse_transform(lstm_preds[idx].reshape(-1, 1))
    chronos_pred_unscaled = scaler.inverse_transform(chronos_preds[idx].reshape(-1, 1))

    # Metrics
    lstm_metrics = calculate_metrics(actual, lstm_pred_unscaled)
    chronos_metrics = calculate_metrics(actual, chronos_pred_unscaled)

    print_metrics("LSTM", lstm_metrics)
    print_metrics("Chronos2", chronos_metrics)

    # ---- PLOTTING ----
    plt.figure(figsize=(14, 6))

    context_len = len(test_seq[idx])
    x_context = np.arange(context_len)
    x_pred = np.arange(context_len, context_len + PRED_LENGTH)

    # Plot Past Context
    plt.plot(x_context, scaler.inverse_transform(test_seq[idx].numpy()), label="Past Context", color='black', alpha=0.7)

    # Plot Actual Ground Truth
    plt.plot(x_pred, actual, label="Ground Truth", linestyle='--', color='black', marker='o')

    # Plot LSTM
    plt.plot(x_pred, lstm_pred_unscaled, label="LSTM Forecast", color='tab:red', marker='x')

    # Plot Chronos2
    plt.plot(x_pred, chronos_pred_unscaled, label="Chronos2 Zero-Shot", color='tab:blue', marker='s')

    plt.title("Forecasting Comparison: LSTM vs Amazon Chronos2 (Small)")
    plt.xlabel("Time Steps")
    plt.ylabel("Traffic Demand (Mbps)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

