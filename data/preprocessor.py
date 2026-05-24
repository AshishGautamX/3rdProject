"""Normalise and split the aggregated 1-D timeseries.

Works on the small (20160,) array produced by load_aggregated_timeseries().
No large matrix operations — RAM usage is negligible.
"""
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from config.settings import TRAIN_RATIO, VAL_RATIO


def normalise(series: np.ndarray):
    """Min-Max normalise a 1-D array. Returns (scaled, scaler)."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series.reshape(-1, 1)).flatten()
    return scaled.astype(np.float32), scaler


def chronological_split(series: np.ndarray):
    """Split chronologically into train / val / test. No shuffle."""
    n         = len(series)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    return series[:train_end], series[train_end:val_end], series[val_end:]


def preprocess(timeseries: np.ndarray):
    """Normalise → split.
    Returns (train_1d, val_1d, test_1d, scaler).
    """
    ts, scaler       = normalise(timeseries)
    train, val, test = chronological_split(ts)
    print(f"[preproc] Total minutes: {len(ts)} | "
          f"train={len(train)} val={len(val)} test={len(test)}")
    return train, val, test, scaler
