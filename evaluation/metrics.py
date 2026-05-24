"""Compute scheduler performance metrics from job records."""
import numpy as np
from typing import List, Dict


def compute_response_time(records: List[Dict], key: str = "response_ms") -> Dict:
    """Return mean, p95, p99 response times in ms."""
    times = np.array([r[key] for r in records if key in r], dtype=np.float64)
    if len(times) == 0:
        return {"mean": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "mean": float(np.mean(times)),
        "p95":  float(np.percentile(times, 95)),
        "p99":  float(np.percentile(times, 99)),
    }


def compute_throughput(records: List[Dict], total_minutes: int) -> float:
    """Total jobs processed per second over the simulation window."""
    total_jobs = sum(r.get("processed", 1) for r in records)
    return total_jobs / max(total_minutes * 60, 1)


def compute_utilisation(records: List[Dict], max_slots: int) -> float:
    """Fraction of available slot-capacity actually used.

    Baseline: 1 job per record, capacity = max_slots × total_minutes.
    AI      : each record has 'slots' and 'processed' fields.
    """
    if not records:
        return 0.0
    if "slots" in records[0]:                      # AI records
        total_proc = sum(r["processed"] for r in records)
        total_cap  = sum(r["slots"]     for r in records)
    else:                                           # baseline records
        minutes    = (max(r["minute"] for r in records) + 1)
        total_proc = len(records)
        total_cap  = max_slots * minutes
    return total_proc / max(total_cap, 1)


def summarise(records: List[Dict], total_minutes: int,
              max_slots: int, label: str) -> Dict:
    """Print + return a formatted summary of all metrics."""
    rt   = compute_response_time(records)
    thr  = compute_throughput(records, total_minutes)
    util = compute_utilisation(records, max_slots)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Mean Response Time : {rt['mean']:>10.2f} ms")
    print(f"  P95  Response Time : {rt['p95']:>10.2f} ms")
    print(f"  P99  Response Time : {rt['p99']:>10.2f} ms")
    print(f"  Throughput         : {thr:>10.4f} jobs/sec")
    print(f"  Resource Util.     : {util*100:>9.2f} %")
    print(f"{'='*60}")

    return {"response_mean": rt["mean"], "response_p95": rt["p95"],
            "response_p99": rt["p99"], "throughput": thr, "utilisation": util}
