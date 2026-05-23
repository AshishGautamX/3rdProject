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
    """Side-by-side bar chart comparing both schedulers on all 3 metrics."""
    labels   = ["Mean Resp. (ms)", "P95 Resp. (ms)", "Throughput (j/s)", "Util. (%)"]
    baseline = [
        baseline_metrics["response_mean"],
        baseline_metrics["response_p95"],
        baseline_metrics["throughput"],
        baseline_metrics["utilisation"] * 100,
    ]
    ai = [
        ai_metrics["response_mean"],
        ai_metrics["response_p95"],
        ai_metrics["throughput"],
        ai_metrics["utilisation"] * 100,
    ]

    x   = np.arange(len(labels))
    w   = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - w / 2, baseline, w, label="Default (FCFS)", color="#e07b54")
    ax.bar(x + w / 2, ai,       w, label="AI (PPO+LSTM)",  color="#4c8bca")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title("Scheduler Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(); plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "metric_comparison.png")
        plt.savefig(path, dpi=150)
        print(f"[vis] Saved → {path}")
    plt.show()


def plot_workload_prediction(actual: np.ndarray, predicted: np.ndarray,
                             n_points: int = 500, save: bool = True):
    """Time-series plot: actual vs LSTM-predicted invocations."""
    t = np.arange(min(n_points, len(actual)))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, actual[:n_points],    label="Actual",    linewidth=1.2, color="#e07b54")
    ax.plot(t, predicted[:n_points], label="Predicted", linewidth=1.2,
            color="#4c8bca", linestyle="--")
    ax.set_xlabel("Minute"); ax.set_ylabel("Invocations (normalised)")
    ax.set_title("LSTM Workload Prediction vs Actual", fontsize=13, fontweight="bold")
    ax.legend(); plt.tight_layout()

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
    ax.set_xlabel("Step"); ax.set_ylabel("Reward")
    ax.set_title("RL Training Reward Curve", fontsize=13, fontweight="bold")
    plt.tight_layout()

    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "reward_curve.png")
        plt.savefig(path, dpi=150)
        print(f"[vis] Saved → {path}")
    plt.show()
