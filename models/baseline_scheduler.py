"""Default FCFS / Round-Robin serverless scheduler simulation.

Tracks each job's arrival time so response_ms = completion - arrival
(captures real queuing delay across minutes, not just execution time).
"""
from collections import deque
import numpy as np
from typing import List, Dict


def simulate_baseline(
    invocations: np.ndarray,
    max_slots: int = 10,
    avg_duration_ms: float = 200.0,
    strategy: str = "fcfs",
) -> List[Dict]:
    """Simulate FCFS / Round-Robin scheduler over the test workload.

    Each job's response time is measured from its arrival minute to its
    completion, correctly capturing multi-minute queuing delays.

    Returns list of per-job records.
    """
    records      = []
    arrival_q    = deque()       # FIFO queue of arrival_time_ms per job
    slot_free_at = [0] * max_slots

    for minute, arrivals in enumerate(invocations):
        arrivals = max(0, int(round(float(arrivals))))
        t_ms     = minute * 60_000

        # Enqueue all new arrivals (they arrive at t_ms)
        for _ in range(arrivals):
            arrival_q.append(t_ms)

        slot_order = list(range(max_slots))
        if strategy == "round_robin":
            np.random.shuffle(slot_order)
        else:
            slot_order.sort(key=lambda s: slot_free_at[s])

        for slot in slot_order:
            if not arrival_q:
                break
            job_arrival_ms     = arrival_q.popleft()          # oldest job first
            start              = max(t_ms, slot_free_at[slot])
            end                = start + avg_duration_ms
            slot_free_at[slot] = end
            records.append({
                "minute":      minute,
                "slot":        slot,
                "arrival_ms":  job_arrival_ms,
                "start_ms":    start,
                "end_ms":      end,
                "wait_ms":     start - job_arrival_ms,
                "response_ms": end   - job_arrival_ms,   # ← arrival to completion
                "queued":      len(arrival_q),
                "processed":   1,
            })

    return records
