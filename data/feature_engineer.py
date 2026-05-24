"""Build sliding-window sequences and engineered features for the LSTM.

Input is a 1-D timeseries (T,). All feature ops return (T, F) where F is small.
"""
import numpy as np
import pandas as pd
from config.settings import WINDOW_SIZE, HORIZON


def add_rolling_stats(series_1d: np.ndarray, windows=(5, 15)) -> np.ndarray:
    """Append rolling mean & std columns. Output shape: (T, 1 + 2*len(windows))."""
    s   = pd.Series(series_1d.astype(np.float64))
    cols = [series_1d.reshape(-1, 1)]
    for w in windows:
        cols.append(s.rolling(w, min_periods=1).mean().values.reshape(-1, 1))
        cols.append(s.rolling(w, min_periods=1).std().fillna(0).values.reshape(-1, 1))
    return np.concatenate(cols, axis=1).astype(np.float32)


def add_time_features(series_2d: np.ndarray, mins_per_day: int = 1440) -> np.ndarray:
    """Append 4 cyclical time features (hour sin/cos, day-of-week sin/cos)."""
    T   = len(series_2d)
    idx = np.arange(T)
    hour = (idx % mins_per_day) / 60
    dow  = (idx // mins_per_day) % 7
    extra = np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  / 7),  np.cos(2 * np.pi * dow  / 7),
    ], axis=1).astype(np.float32)
    return np.concatenate([series_2d, extra], axis=1)


def build_features(series_1d: np.ndarray) -> np.ndarray:
    """Full feature pipeline: 1-D → (T, 9) float32 array.
    Columns: [load, roll5_mean, roll5_std, roll15_mean, roll15_std,
              sin_hr, cos_hr, sin_dow, cos_dow]
    """
    feat = add_rolling_stats(series_1d)    # (T, 5)
    feat = add_time_features(feat)         # (T, 9)
    return feat


def make_sequences(series_2d: np.ndarray,
                   window: int = WINDOW_SIZE,
                   horizon: int = HORIZON):
    """Slice (T, F) → X (N, window, F) and y (N, horizon) pairs.
    y contains only the first (load) column for forecasting.
    """
    X, y = [], []
    for i in range(len(series_2d) - window - horizon + 1):
        X.append(series_2d[i : i + window])
        y.append(series_2d[i + window : i + window + horizon, 0])  # load col only
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)
