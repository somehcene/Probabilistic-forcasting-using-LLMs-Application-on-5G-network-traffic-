import torch
import torch.nn as nn
import numpy as np

class TrafficLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=64, num_layers=2, output_size=12, dropout=0.2):
        super(TrafficLSTM, self).__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, input_seq):
        # input_seq: (batch, seq_len, input_size)
        lstm_out, _ = self.lstm(input_seq)
        
        # Take the output of the last time step
        last_time_step = lstm_out[:, -1, :]
        predictions = self.linear(last_time_step)
        
        return predictions

def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=5, device='cpu'):
    model.to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_model_state = None
    
    for i in range(epochs):
        model.train()
        train_loss = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            
            y_pred = model(seq)
            
            # y_pred is (batch, pred_len), labels is (batch, pred_len, 1)
            loss = loss_function(y_pred, labels.squeeze(-1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seq, labels in val_loader:
                seq, labels = seq.to(device), labels.to(device)
                y_pred = model(seq)
                single_loss = loss_function(y_pred, labels.squeeze(-1))
                val_loss += single_loss.item()
                
        # Average losses
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        
        print(f'Epoch {i:3} | Train Loss: {train_loss:10.5f} | Val Loss: {val_loss:10.5f}')
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping triggered")
                break
                
    model.load_state_dict(best_model_state)
    return model
