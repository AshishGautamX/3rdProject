"""LSTM workload forecaster (PyTorch).
Predicts next HORIZON load values from a WINDOW_SIZE input sequence.
Input X : (N, window, 9 features)
Output y: (N, horizon)  — forecasted load (first feature column)
"""
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from config.settings import (
    LSTM_HIDDEN_DIM, LSTM_LAYERS, LSTM_DROPOUT,
    LSTM_LR, LSTM_EPOCHS, LSTM_BATCH_SIZE, LSTM_PATIENCE, HORIZON,
)


class LSTMForecaster(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=LSTM_HIDDEN_DIM,
                 num_layers=LSTM_LAYERS, dropout=LSTM_DROPOUT, horizon=HORIZON):
        super().__init__()
        self.horizon   = horizon
        self.input_dim = input_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc   = nn.Linear(hidden_dim, horizon)   # predicts horizon load values

    def forward(self, x):
        out, _ = self.lstm(x)
        last   = out[:, -1, :]          # (B, hidden)
        return self.fc(last)            # (B, horizon)


def train_lstm(X_train, y_train, X_val, y_val, weights_path: str = None):
    """Train with early stopping. Returns trained model."""
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[lstm] Training on {device}")

    model     = LSTMForecaster(X_train.shape[2]).to(device)
    opt       = torch.optim.Adam(model.parameters(), lr=LSTM_LR)
    criterion = nn.HuberLoss()

    train_dl = DataLoader(
        TensorDataset(torch.tensor(X_train), torch.tensor(y_train)),
        batch_size=LSTM_BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(
        TensorDataset(torch.tensor(X_val), torch.tensor(y_val)),
        batch_size=LSTM_BATCH_SIZE)

    best_val, patience_cnt = float("inf"), 0
    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        t_loss = sum(
            (lambda xb, yb: (opt.zero_grad(), loss := criterion(model(xb.to(device)), yb.to(device)),
             loss.backward(), opt.step(), loss.item())[-1])(xb, yb)
            for xb, yb in train_dl) / len(train_dl)

        model.eval()
        with torch.no_grad():
            v_loss = sum(criterion(model(xb.to(device)), yb.to(device)).item()
                         for xb, yb in val_dl) / len(val_dl)

        print(f"[lstm] Epoch {epoch:3d}/{LSTM_EPOCHS} | train={t_loss:.4f} val={v_loss:.4f}")
        if v_loss < best_val:
            best_val, patience_cnt = v_loss, 0
            if weights_path:
                os.makedirs(os.path.dirname(weights_path), exist_ok=True)
                torch.save(model.state_dict(), weights_path)
        else:
            patience_cnt += 1
            if patience_cnt >= LSTM_PATIENCE:
                print(f"[lstm] Early stopping at epoch {epoch}")
                break

    if weights_path and os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=device))
    return model


def predict(model, X: np.ndarray) -> np.ndarray:
    """Inference → predictions shape (N, horizon)."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        return model(torch.tensor(X).to(device)).cpu().numpy()


def load_lstm(weights_path: str, input_dim: int):
    """Load a saved LSTM from disk."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = LSTMForecaster(input_dim).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    print(f"[lstm] Weights loaded from {weights_path}")
    return model
