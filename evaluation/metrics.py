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
    """Jobs completed per second over the simulation window."""
    return len(records) / max(total_minutes * 60, 1)


def compute_utilisation(records: List[Dict], max_slots: int) -> float:
    """Average fraction of slots actively used."""
    vals = []
    for r in records:
        if "slot" in r:                             # baseline: 1 job / record
            vals.append(1.0)
        elif "slots" in r and "processed" in r:     # RL record
            vals.append(r["processed"] / max(r["slots"], 1))
    return float(np.mean(vals)) if vals else 0.0


def summarise(records: List[Dict], total_minutes: int,
              max_slots: int, label: str) -> Dict:
    """Print + return a formatted summary of all metrics."""
    rt   = compute_response_time(records)
    thr  = compute_throughput(records, total_minutes)
    util = compute_utilisation(records, max_slots)

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Mean Response Time : {rt['mean']:>8.2f} ms")
    print(f"  P95  Response Time : {rt['p95']:>8.2f} ms")
    print(f"  P99  Response Time : {rt['p99']:>8.2f} ms")
    print(f"  Throughput         : {thr:>8.4f} jobs/sec")
    print(f"  Resource Util.     : {util*100:>7.2f} %")
    print(f"{'='*60}")

    return {"response_mean": rt["mean"], "response_p95": rt["p95"],
            "response_p99": rt["p99"], "throughput": thr, "utilisation": util}
