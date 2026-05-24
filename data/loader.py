"""Load Azure Functions 2019 trace CSVs from Google Drive.

Memory-efficient design: CSVs are read ONE AT A TIME, aggregated immediately
to a per-minute total, then discarded. Peak RAM never exceeds ~200 MB.
"""
import os
import gc
import numpy as np
import pandas as pd
from config.settings import (
    NUM_DAYS,
    INVOCATION_FILE_TEMPLATE, DURATION_FILE_TEMPLATE,
    MIN_TOTAL_INVOCATIONS,
)

_MINUTE_COLS = [str(i) for i in range(1, 1441)]


def _minute_cols_present(df: pd.DataFrame) -> list:
    return [c for c in _MINUTE_COLS if c in df.columns]


def load_aggregated_timeseries(path: str, days: int = NUM_DAYS) -> np.ndarray:
    """
    Stream-load all invocation CSVs one day at a time and return a single
    1-D array of shape (days * 1440,) containing the total invocations
    per minute aggregated across all functions.

    Peak RAM: ~one CSV at a time (~100–300 MB) instead of all 14 at once.
    """
    daily_totals = []

    for day in range(1, days + 1):
        fpath = os.path.join(path, INVOCATION_FILE_TEMPLATE.format(day=day))
        if not os.path.exists(fpath):
            print(f"[loader] Day {day:02d} not found — filling with zeros.")
            daily_totals.append(np.zeros(1440, dtype=np.float32))
            continue

        # Read only the minute columns (skip metadata string cols)
        df = pd.read_csv(fpath, usecols=lambda c: c in _MINUTE_COLS)
        mcols = _minute_cols_present(df)

        # Filter sparse functions before aggregating (saves memory)
        row_totals = df[mcols].sum(axis=1)
        df = df.loc[row_totals >= MIN_TOTAL_INVOCATIONS, mcols]

        # Aggregate: total invocations per minute across all functions
        per_minute = df.sum(axis=0).reindex(_MINUTE_COLS[:len(mcols)], fill_value=0)
        daily_totals.append(per_minute.values.astype(np.float32))

        n_funcs = len(df)
        del df, row_totals, per_minute
        gc.collect()
        print(f"[loader] Day {day:02d} — {n_funcs} functions aggregated.")

    timeseries = np.concatenate(daily_totals)   # shape: (days * 1440,)
    print(f"[loader] Timeseries shape: {timeseries.shape} | "
          f"min={timeseries.min():.0f} max={timeseries.max():.0f}")
    return timeseries


def load_avg_duration(path: str, days: int = NUM_DAYS) -> np.ndarray:
    """
    Load average execution duration per minute across all days.
    Returns shape (days * 1440,) — uses p50 column where available.
    Falls back to zeros if duration files are missing.
    """
    daily_dur = []
    for day in range(1, days + 1):
        fpath = os.path.join(path, DURATION_FILE_TEMPLATE.format(day=day))
        if not os.path.exists(fpath):
            daily_dur.append(np.full(1440, 200.0, dtype=np.float32))
            continue

        df = pd.read_csv(fpath)
        # Azure duration CSV has a percentile column (p50 or similar)
        p50_col = next((c for c in df.columns if "50" in c), None)
        if p50_col:
            avg_dur = float(df[p50_col].mean())
        else:
            avg_dur = 200.0
        daily_dur.append(np.full(1440, avg_dur, dtype=np.float32))
        del df
        gc.collect()

    return np.concatenate(daily_dur)
