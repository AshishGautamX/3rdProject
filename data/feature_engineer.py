"""Build sliding-window sequences and engineered features for the LSTM."""
import numpy as np
import pandas as pd
from config.settings import WINDOW_SIZE, HORIZON


def add_rolling_stats(series: np.ndarray, windows=(5, 15)) -> np.ndarray:
    """Append rolling mean & std for each window. Pads leading NaNs with 0."""
    df = pd.DataFrame(series)
    extras = []
    for w in windows:
        extras.append(df.rolling(w, min_periods=1).mean().values)
        extras.append(df.rolling(w, min_periods=1).std().fillna(0).values)
    return np.concatenate([series] + extras, axis=1).astype(np.float32)


def add_time_features(series: np.ndarray, mins_per_day: int = 1440) -> np.ndarray:
    """Append cyclical hour-of-day and day-of-week encodings (4 extra cols)."""
    T   = len(series)
    idx = np.arange(T)
    hour = (idx % mins_per_day) / 60
    dow  = (idx // mins_per_day) % 7
    extra = np.stack([
        np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
        np.sin(2 * np.pi * dow  / 7),  np.cos(2 * np.pi * dow  / 7),
    ], axis=1).astype(np.float32)
    return np.concatenate([series, extra], axis=1)


def make_sequences(series: np.ndarray, window: int = WINDOW_SIZE, horizon: int = HORIZON):
    """Slice (T, F) → X (N, window, F) and y (N, horizon, F) pairs."""
    X, y = [], []
    for i in range(len(series) - window - horizon + 1):
        X.append(series[i : i + window])
        y.append(series[i + window : i + window + horizon])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def build_features(series: np.ndarray) -> np.ndarray:
    """Apply all feature engineering steps in sequence."""
    series = add_rolling_stats(series)
    series = add_time_features(series)
    return series
