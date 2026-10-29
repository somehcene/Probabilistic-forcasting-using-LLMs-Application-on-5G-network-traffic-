# Run from anywhere: make the repo root the working dir and importable
import os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT); sys.path.insert(0, ROOT)

import json

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# LSTM Architecture & Point Forecasting Experiments\n",
                "This notebook explores different architectures and configurations for the LSTM to improve point forecasting accuracy. Specifically, we evaluate different architectures (e.g. standard vs bidirectional LSTM) and dedicate a separate model specifically trained and optimized for 1-step forecasting, as relying on the first step of a 12-step trained model is suboptimal and yields structural inaccuracies."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import pandas as pd\n",
                "import numpy as np\n",
                "import torch\n",
                "import torch.nn as nn\n",
                "from torch.utils.data import TensorDataset, DataLoader\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.preprocessing import MinMaxScaler\n",
                "from sklearn.metrics import mean_squared_error\n",
                "\n",
                "# Load data\n",
                "df = pd.read_csv('data/augmented_data.csv')\n",
                "series = df[df['secteur'] == df['secteur'].iloc[0]]['trafic_mbps'].values\n",
                "print(f\"Data shape: {series.shape}\")"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Data preparation function\n",
                "def create_sequences(data, seq_length, pred_length):\n",
                "    xs, ys = [], []\n",
                "    for i in range(len(data) - seq_length - pred_length + 1):\n",
                "        x = data[i:(i + seq_length)]\n",
                "        y = data[(i + seq_length):(i + seq_length + pred_length)]\n",
                "        xs.append(x)\n",
                "        ys.append(y)\n",
                "    return np.array(xs), np.array(ys)\n",
                "\n",
                "scaler = MinMaxScaler(feature_range=(-1, 1))\n",
                "scaled_series = scaler.fit_transform(series.reshape(-1, 1)).flatten()\n",
                "\n",
                "# Train/Val/Test split on the scaled data\n",
                "train_size = int(len(scaled_series) * 0.7)\n",
                "val_size = int(len(scaled_series) * 0.15)\n",
                "train_data = scaled_series[:train_size]\n",
                "val_data = scaled_series[train_size:train_size+val_size]\n",
                "test_data = scaled_series[train_size+val_size:]"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Model Definitions\n",
                "class PointLSTM(nn.Module):\n",
                "    def __init__(self, hidden_layer_size=64, num_layers=2):\n",
                "        super().__init__()\n",
                "        self.lstm = nn.LSTM(1, hidden_layer_size, num_layers, batch_first=True, dropout=0.2)\n",
                "        self.linear = nn.Linear(hidden_layer_size, 1) # Predict exactly 1 step ahead\n",
                "    def forward(self, input_seq):\n",
                "        lstm_out, _ = self.lstm(input_seq)\n",
                "        return self.linear(lstm_out[:, -1, :])\n",
                "\n",
                "class PointBiLSTM(nn.Module):\n",
                "    def __init__(self, hidden_layer_size=64, num_layers=2):\n",
                "        super().__init__()\n",
                "        self.lstm = nn.LSTM(1, hidden_layer_size, num_layers, batch_first=True, dropout=0.2, bidirectional=True)\n",
                "        self.linear = nn.Linear(hidden_layer_size * 2, 1)\n",
                "    def forward(self, input_seq):\n",
                "        lstm_out, _ = self.lstm(input_seq)\n",
                "        return self.linear(lstm_out[:, -1, :])\n",
                "\n",
                "class DeepPointLSTM(nn.Module):\n",
                "    def __init__(self, hidden_layer_size=128, num_layers=4):\n",
                "        super().__init__()\n",
                "        self.lstm = nn.LSTM(1, hidden_layer_size, num_layers, batch_first=True, dropout=0.3)\n",
                "        self.linear = nn.Linear(hidden_layer_size, 1)\n",
                "    def forward(self, input_seq):\n",
                "        lstm_out, _ = self.lstm(input_seq)\n",
                "        return self.linear(lstm_out[:, -1, :])"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Training Loop\n",
                "def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=5):\n",
                "    device = \"cpu\"\n",
                "    model.to(device)\n",
                "    loss_function = nn.MSELoss()\n",
                "    optimizer = torch.optim.Adam(model.parameters(), lr=lr)\n",
                "    \n",
                "    best_val_loss = float(\"inf\")\n",
                "    early_stop_counter = 0\n",
                "    best_model_state = None\n",
                "    \n",
                "    for i in range(epochs):\n",
                "        model.train()\n",
                "        for seq, labels in train_loader:\n",
                "            seq, labels = seq.to(device), labels.to(device)\n",
                "            optimizer.zero_grad()\n",
                "            y_pred = model(seq)\n",
                "            loss = loss_function(y_pred, labels.squeeze(-1))\n",
                "            loss.backward()\n",
                "            optimizer.step()\n",
                "            \n",
                "        model.eval()\n",
                "        val_loss = 0\n",
                "        with torch.no_grad():\n",
                "            for seq, labels in val_loader:\n",
                "                seq, labels = seq.to(device), labels.to(device)\n",
                "                y_pred = model(seq)\n",
                "                val_loss += loss_function(y_pred, labels.squeeze(-1)).item()\n",
                "        val_loss /= len(val_loader)\n",
                "        \n",
                "        if val_loss < best_val_loss:\n",
                "            best_val_loss = val_loss\n",
                "            best_model_state = model.state_dict()\n",
                "            early_stop_counter = 0\n",
                "        else:\n",
                "            early_stop_counter += 1\n",
                "            if early_stop_counter >= patience:\n",
                "                break\n",
                "                \n",
                "    model.load_state_dict(best_model_state)\n",
                "    return model\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Evaluate Architectures on 1-Step Forecasting\n",
                "seq_length = 60\n",
                "pred_length = 1\n",
                "\n",
                "X_train, y_train = create_sequences(train_data, seq_length, pred_length)\n",
                "X_val, y_val = create_sequences(val_data, seq_length, pred_length)\n",
                "X_test, y_test = create_sequences(test_data, seq_length, pred_length)\n",
                "\n",
                "def to_loader(X, y):\n",
                "    return DataLoader(TensorDataset(torch.tensor(X, dtype=torch.float32).unsqueeze(-1), torch.tensor(y, dtype=torch.float32).unsqueeze(-1)), batch_size=32, shuffle=True)\n",
                "\n",
                "train_loader = to_loader(X_train, y_train)\n",
                "val_loader = to_loader(X_val, y_val)\n",
                "test_loader = to_loader(X_test, y_test)\n",
                "\n",
                "models = {\n",
                "    \"Standard LSTM\": PointLSTM(),\n",
                "    \"Bidirectional LSTM\": PointBiLSTM(),\n",
                "    \"Deep LSTM\": DeepPointLSTM()\n",
                "}\n",
                "\n",
                "results = {}\n",
                "for name, m in models.items():\n",
                "    print(f\"Training {name}...\")\n",
                "    m = train_model(m, train_loader, val_loader)\n",
                "    \n",
                "    m.eval()\n",
                "    preds = []\n",
                "    with torch.no_grad():\n",
                "        for seq, _ in test_loader:\n",
                "            preds.append(m(seq).numpy())\n",
                "    preds = np.concatenate(preds).flatten()\n",
                "    \n",
                "    # Inverse transform\n",
                "    preds_inv = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()\n",
                "    actuals_inv = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()\n",
                "    \n",
                "    rmse = np.sqrt(mean_squared_error(actuals_inv, preds_inv))\n",
                "    results[name] = {\"rmse\": rmse, \"preds\": preds_inv}\n",
                "    print(f\"{name} RMSE (1-step): {rmse:.2f}\")\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plotting\n",
                "plt.figure(figsize=(15, 6))\n",
                "plt.plot(actuals_inv[-100:], label=\"Ground Truth\", color=\"black\", marker='o')\n",
                "for name, res in results.items():\n",
                "    plt.plot(res[\"preds\"][-100:], label=f\"{name} (RMSE: {res['rmse']:.2f})\", alpha=0.8)\n",
                "plt.legend()\n",
                "plt.title(\"1-Step Forecasting Architecture Comparisons\")\n",
                "plt.xlim(0, 100)\n",
                "plt.show()\n"
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.9"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

notebook['cells'].insert(1 if notebook['cells'][0]['cell_type']=='markdown' else 0, {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':['import os, sys\n', "if os.path.basename(os.getcwd()) == 'notebooks':\n", "    os.chdir('..')\n", 'sys.path.insert(0, os.getcwd())']})

with open('notebooks/03_lstm_experiments.ipynb', 'w') as f:
    json.dump(notebook, f, indent=4)
