"""Load Azure Functions 2019 trace CSVs from Google Drive."""
import os
import pandas as pd
from config.settings import (
    DRIVE_DATA_PATH, NUM_DAYS,
    INVOCATION_FILE_TEMPLATE, DURATION_FILE_TEMPLATE, MEMORY_FILE_TEMPLATE,
)


def _load_day_files(path: str, template: str, days: int) -> pd.DataFrame:
    frames = []
    for day in range(1, days + 1):
        fpath = os.path.join(path, template.format(day=day))
        if not os.path.exists(fpath):
            print(f"[loader] WARNING: {fpath} not found — skipping.")
            continue
        df = pd.read_csv(fpath)
        df["day"] = day
        frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No files matched '{template}' in '{path}'")
    return pd.concat(frames, ignore_index=True)


def load_invocations(path: str = DRIVE_DATA_PATH) -> pd.DataFrame:
    """Per-minute invocation counts for all functions across all days."""
    return _load_day_files(path, INVOCATION_FILE_TEMPLATE, NUM_DAYS)


def load_durations(path: str = DRIVE_DATA_PATH) -> pd.DataFrame:
    """Execution time percentile data per function per day."""
    return _load_day_files(path, DURATION_FILE_TEMPLATE, NUM_DAYS)


def load_memory(path: str = DRIVE_DATA_PATH) -> pd.DataFrame:
    """Memory allocation percentiles per app per day."""
    return _load_day_files(path, MEMORY_FILE_TEMPLATE, NUM_DAYS)


def load_all(path: str = DRIVE_DATA_PATH):
    """Convenience: load invocations, durations, memory in one call."""
    print("[loader] Loading invocations ...")
    inv = load_invocations(path)
    print(f"         Shape: {inv.shape}")

    print("[loader] Loading durations ...")
    dur = load_durations(path)
    print(f"         Shape: {dur.shape}")

    print("[loader] Loading memory ...")
    mem = load_memory(path)
    print(f"         Shape: {mem.shape}")

    return inv, dur, mem
