"""Clean, normalise, and split Azure Functions trace data."""
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from config.settings import MIN_TOTAL_INVOCATIONS, TRAIN_RATIO, VAL_RATIO

_MINUTE_COLS = [str(i) for i in range(1, 1441)]


def _minute_cols(df: pd.DataFrame) -> list:
    return [c for c in _MINUTE_COLS if c in df.columns]


def filter_sparse(inv_df: pd.DataFrame) -> pd.DataFrame:
    """Drop functions whose total invocations fall below the threshold."""
    mcols = _minute_cols(inv_df)
    inv_df = inv_df.copy()
    inv_df["_total"] = inv_df[mcols].sum(axis=1)
    before = len(inv_df)
    inv_df = inv_df[inv_df["_total"] >= MIN_TOTAL_INVOCATIONS].drop(columns=["_total"])
    print(f"[preproc] Sparse filter: {before} → {len(inv_df)} functions")
    return inv_df.reset_index(drop=True)


def build_timeseries(inv_df: pd.DataFrame) -> np.ndarray:
    """Convert invocation DataFrame → 2-D array (total_minutes, num_functions)."""
    mcols = _minute_cols(inv_df)
    matrix = inv_df[mcols].values.astype(np.float32)   # (functions, minutes)
    return matrix.T                                      # → (minutes, functions)


def normalise(series: np.ndarray):
    """Min-Max normalise. Returns (scaled_array, fitted_scaler)."""
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(series)
    return scaled, scaler


def chronological_split(series: np.ndarray):
    """Split time-series into train / val / test (chronological, no shuffle)."""
    n = len(series)
    train_end = int(n * TRAIN_RATIO)
    val_end   = int(n * (TRAIN_RATIO + VAL_RATIO))
    return series[:train_end], series[train_end:val_end], series[val_end:]


def preprocess(inv_df: pd.DataFrame):
    """Full pipeline: filter → timeseries → normalise → split.
    Returns (train, val, test, scaler).
    """
    inv_df       = filter_sparse(inv_df)
    ts           = build_timeseries(inv_df)
    ts, scaler   = normalise(ts)
    train, val, test = chronological_split(ts)
    print(f"[preproc] Series shape: {ts.shape} | "
          f"train={len(train)} val={len(val)} test={len(test)} min")
    return train, val, test, scaler
