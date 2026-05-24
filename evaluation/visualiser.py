"""Visualise and compare baseline vs AI scheduler results."""
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict
from config.settings import RESULTS_DIR

sns.set_theme(style="darkgrid")


def plot_metric_comparison(baseline_metrics: Dict, ai_metrics: Dict,
                            save: bool = True):
    """Three separate subplots — response time, throughput, utilisation.
    Keeps metrics on their own axis so different scales don't hide bars.
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("Scheduler Performance Comparison", fontsize=14, fontweight="bold")

    colors = {"Default (FCFS)": "#e07b54", "AI (PPO+LSTM)": "#4c8bca"}

    # ── Response time ─────────────────────────────────────────────────────────
    ax = axes[0]
    metrics_rt = {
        "Default (FCFS)": baseline_metrics["response_mean"],
        "AI (PPO+LSTM)":  ai_metrics["response_mean"],
    }
    bars = ax.bar(list(metrics_rt.keys()), list(metrics_rt.values()),
                  color=list(colors.values()), width=0.4)
    ax.bar_label(bars, fmt="%.0f", padding=3, fontsize=9)
    ax.set_title("Mean Response Time (ms)")
    ax.set_ylabel("ms")

    # ── Throughput ────────────────────────────────────────────────────────────
    ax = axes[1]
    metrics_thr = {
        "Default (FCFS)": baseline_metrics["throughput"],
        "AI (PPO+LSTM)":  ai_metrics["throughput"],
    }
    bars = ax.bar(list(metrics_thr.keys()), list(metrics_thr.values()),
                  color=list(colors.values()), width=0.4)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_title("Throughput (jobs/sec)")
    ax.set_ylabel("jobs/sec")

    # ── Utilisation ───────────────────────────────────────────────────────────
    ax = axes[2]
    metrics_util = {
        "Default (FCFS)": baseline_metrics["utilisation"] * 100,
        "AI (PPO+LSTM)":  ai_metrics["utilisation"] * 100,
    }
    bars = ax.bar(list(metrics_util.keys()), list(metrics_util.values()),
                  color=list(colors.values()), width=0.4)
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
    ax.set_title("Resource Utilisation (%)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 110)

    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "metric_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"[vis] Saved → {path}")
    plt.show()


def plot_workload_prediction(actual: np.ndarray, predicted: np.ndarray,
                             n_points: int = 500, save: bool = True):
    """Time-series plot: actual vs LSTM-predicted invocations (normalised)."""
    n = min(n_points, len(actual), len(predicted))
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, actual[:n],    label="Actual",    linewidth=1.2, color="#e07b54")
    ax.plot(t, predicted[:n], label="Predicted", linewidth=1.2,
            color="#4c8bca", linestyle="--")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Invocations (normalised)")
    ax.set_title("LSTM Workload Prediction vs Actual", fontsize=13, fontweight="bold")
    ax.legend()
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "workload_prediction.png")
        plt.savefig(path, dpi=150)
        print(f"[vis] Saved → {path}")
    plt.show()


def plot_reward_curve(rewards: list, save: bool = True):
    """Smoothed RL episode reward curve."""
    r   = np.array(rewards)
    win = max(1, len(r) // 50)
    smooth = np.convolve(r, np.ones(win) / win, mode="valid")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(smooth, color="#5cb85c", linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title("RL Training Reward Curve", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "reward_curve.png")
        plt.savefig(path, dpi=150)
        print(f"[vis] Saved → {path}")
    plt.show()
