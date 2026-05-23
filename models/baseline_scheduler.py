"""Default FCFS / Round-Robin serverless scheduler simulation."""
import numpy as np
from typing import List, Dict


def simulate_baseline(
    invocations: np.ndarray,
    max_slots: int = 10,
    avg_duration_ms: float = 200.0,
    strategy: str = "fcfs",
) -> List[Dict]:
    """Simulate a default scheduler over the test workload.

    Parameters
    ----------
    invocations    : 1-D array — jobs arriving each minute
    max_slots      : maximum concurrent execution slots
    avg_duration_ms: average job execution time in ms
    strategy       : 'fcfs' or 'round_robin'

    Returns list of per-job records with response_ms, wait_ms, slot, queued.
    """
    records      = []
    queue        = 0
    slot_free_at = [0] * max_slots

    for minute, arrivals in enumerate(invocations):
        arrivals = max(0, int(round(float(arrivals))))
        queue   += arrivals
        t_ms     = minute * 60_000   # minute → ms

        slot_order = list(range(max_slots))
        if strategy == "round_robin":
            np.random.shuffle(slot_order)
        else:
            slot_order.sort(key=lambda s: slot_free_at[s])

        for slot in slot_order:
            if queue == 0:
                break
            start  = max(t_ms, slot_free_at[slot])
            end    = start + avg_duration_ms
            slot_free_at[slot] = end
            records.append({
                "minute":      minute,
                "slot":        slot,
                "start_ms":    start,
                "end_ms":      end,
                "wait_ms":     start - t_ms,
                "response_ms": end - t_ms,
                "queued":      queue,
            })
            queue -= 1

    return records
