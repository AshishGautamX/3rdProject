"""Compute scheduler performance metrics from job records."""
import numpy as np
from typing import List, Dict


def compute_response_time(records: List[Dict], key: str = "response_ms") -> Dict:
    times = np.array([r[key] for r in records if key in r], dtype=np.float64)
    if len(times) == 0:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0}
    return {"mean": float(np.mean(times)),
            "p95":  float(np.percentile(times, 95)),
            "p99":  float(np.percentile(times, 99))}


def compute_throughput(records: List[Dict], total_minutes: int) -> float:
    total_jobs = sum(r.get("processed", 1) for r in records)
    return total_jobs / max(total_minutes * 60, 1)


def compute_utilisation(records: List[Dict], max_slots: int) -> float:
    """Fraction of slot-capacity used.

    Handles two record styles:
      - Per-job (baseline/reactive): one record per job, 'processed'=1
      - Per-minute (RL): one record per step, 'processed'=N jobs that step

    Groups by minute so that per-job records with a 'slots' key don't
    multiply capacity by the number of jobs in a minute.
    """
    if not records:
        return 0.0
    if "slots" in records[0]:
        # Group by minute to get correct slot capacity per minute
        by_min: Dict[int, Dict] = {}
        for r in records:
            m = r["minute"]
            if m not in by_min:
                by_min[m] = {"proc": 0, "slots": r["slots"]}
            by_min[m]["proc"] += r.get("processed", 1)
        total_proc = sum(v["proc"]  for v in by_min.values())
        total_cap  = sum(v["slots"] for v in by_min.values())
    else:
        minutes    = max(r["minute"] for r in records) + 1
        total_proc = len(records)
        total_cap  = max_slots * minutes
    return total_proc / max(total_cap, 1)


def compute_cost(records: List[Dict], max_slots: int) -> float:
    """Average slots consumed per minute (lower = cheaper)."""
    if not records:
        return 0.0
    if "slots" in records[0]:
        by_min: Dict[int, int] = {}
        for r in records:
            by_min[r["minute"]] = r["slots"]   # last write per minute is fine
        return float(np.mean(list(by_min.values()))) if by_min else 0.0
    return float(max_slots)


def summarise(records: List[Dict], total_minutes: int,
              max_slots: int, label: str,
              verbose: bool = True) -> Dict:
    """Return metrics dict. Prints summary only when verbose=True."""
    rt   = compute_response_time(records)
    thr  = compute_throughput(records, total_minutes)
    util = compute_utilisation(records, max_slots)
    cost = compute_cost(records, max_slots)

    if verbose:
        print(f"  {label:40s}  mean={rt['mean']:>9.1f}ms  "
              f"p99={rt['p99']:>9.1f}ms  thr={thr:.4f}  "
              f"util={util*100:.1f}%  slots={cost:.1f}")

    return {"response_mean": rt["mean"], "response_p95": rt["p95"],
            "response_p99":  rt["p99"], "throughput":   thr,
            "utilisation":   util,      "cost":         cost}
