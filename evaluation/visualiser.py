"""Visualise and compare multi-scheduler results + ablation study."""
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from typing import Dict, List
from config.settings import RESULTS_DIR

sns.set_theme(style="darkgrid")

# Colour palette — consistent across all charts
PALETTE = {
    "FCFS":              "#e07b54",
    "Round-Robin":       "#e0a854",
    "EDF":               "#c8d454",
    "Reactive":          "#54c8d4",
    "LSTM-Threshold":    "#8854d4",
    "PPO-only":          "#54d488",
    "LSTM+PPO":          "#4c8bca",
    "AI (PPO+LSTM)":     "#4c8bca",
    "Default (FCFS)":    "#e07b54",
}


def _colour(name: str) -> str:
    for k, v in PALETTE.items():
        if k in name:
            return v
    return "#888888"


# ─────────────────────────────────────────────────────────────────────────────
# 1. Multi-scheduler comparison table (printed)
# ─────────────────────────────────────────────────────────────────────────────

def print_comparison_table(all_metrics: Dict[str, Dict]) -> None:
    """Print the full comparison table reviewers expect."""
    col_w = 16
    headers = ["Scheduler", "Mean RT (ms)", "P95 (ms)", "P99 (ms)",
               "Throughput", "Util (%)", "Avg Slots"]
    sep = "=" * (col_w * len(headers))
    print(f"\n{sep}")
    print("  " + "".join(f"{h:<{col_w}}" for h in headers))
    print(sep)
    for name, m in all_metrics.items():
        row = [
            name,
            f"{m['response_mean']:.1f}",
            f"{m['response_p95']:.1f}",
            f"{m['response_p99']:.1f}",
            f"{m['throughput']:.4f}",
            f"{m['utilisation']*100:.1f}",
            f"{m.get('cost', 0):.1f}",
        ]
        print("  " + "".join(f"{v:<{col_w}}" for v in row))
    print(sep)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Multi-scheduler bar chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_multi_scheduler_comparison(all_metrics: Dict[str, Dict],
                                    save: bool = True) -> None:
    """4-panel bar chart: Mean RT, P99, Throughput, Avg Slots consumed."""
    names  = list(all_metrics.keys())
    colors = [_colour(n) for n in names]
    x      = np.arange(len(names))
    width  = 0.55

    fig, axes = plt.subplots(1, 4, figsize=(18, 5))
    fig.suptitle("Multi-Scheduler Performance Comparison", fontsize=14,
                 fontweight="bold")

    def _bar(ax, values, title, ylabel, fmt="{:.0f}", log=False):
        bars = ax.bar(x, values, width=width, color=colors, edgecolor="white",
                      linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=25, ha="right", fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt.format(val), ha="center", va="bottom", fontsize=7.5)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    _bar(axes[0], [m["response_mean"] for m in all_metrics.values()],
         "Mean Response Time (ms)", "ms", log=True)
    _bar(axes[1], [m["response_p99"] for m in all_metrics.values()],
         "P99 Response Time (ms)", "ms", log=True)
    _bar(axes[2], [m["throughput"] for m in all_metrics.values()],
         "Throughput (jobs/sec)", "j/s", fmt="{:.4f}")
    _bar(axes[3], [m.get("cost", 0) for m in all_metrics.values()],
         "Avg Slots Used (Cost)", "slots", fmt="{:.1f}")

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "multi_scheduler_comparison.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[vis] Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 3. Ablation study chart
# ─────────────────────────────────────────────────────────────────────────────

def plot_ablation(ablation_metrics: Dict[str, Dict], save: bool = True) -> None:
    """Grouped bar chart showing component contribution (ablation study)."""
    names  = list(ablation_metrics.keys())
    colors = [_colour(n) for n in names]
    x      = np.arange(len(names))
    width  = 0.55

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Ablation Study — Component Contribution", fontsize=13,
                 fontweight="bold")

    def _bar(ax, values, title, ylabel, fmt="{:.0f}", log=False):
        bars = ax.bar(x, values, width=width, color=colors, edgecolor="white",
                      linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=9)
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                    fmt.format(val), ha="center", va="bottom", fontsize=8)
        if log:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(mticker.ScalarFormatter())

    _bar(axes[0], [m["response_mean"] for m in ablation_metrics.values()],
         "Mean Response Time (ms)", "ms", log=True)
    _bar(axes[1], [m["response_p99"] for m in ablation_metrics.values()],
         "P99 Response Time (ms)", "ms", log=True)

    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "ablation_study.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"[vis] Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 4. Workload prediction plot (unchanged API)
# ─────────────────────────────────────────────────────────────────────────────

def plot_workload_prediction(actual: np.ndarray, predicted: np.ndarray,
                             n_points: int = 500, save: bool = True) -> None:
    """Time-series plot: actual vs LSTM-predicted invocations (normalised)."""
    n = min(n_points, len(actual), len(predicted))
    t = np.arange(n)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, actual[:n],    label="Actual",    linewidth=1.2, color="#e07b54")
    ax.plot(t, predicted[:n], label="Predicted", linewidth=1.2,
            color="#4c8bca", linestyle="--")
    ax.set_xlabel("Minute")
    ax.set_ylabel("Invocations (normalised)")
    ax.set_title("LSTM Workload Prediction vs Actual", fontsize=13,
                 fontweight="bold")
    ax.legend()
    plt.tight_layout()
    if save:
        os.makedirs(RESULTS_DIR, exist_ok=True)
        path = os.path.join(RESULTS_DIR, "workload_prediction.png")
        plt.savefig(path, dpi=150)
        print(f"[vis] Saved → {path}")
    plt.show()


# ─────────────────────────────────────────────────────────────────────────────
# 5. Legacy 2-scheduler comparison (kept for backward compat)
# ─────────────────────────────────────────────────────────────────────────────

def plot_metric_comparison(baseline_metrics: Dict, ai_metrics: Dict,
                           save: bool = True) -> None:
    """Backward-compatible 2-scheduler comparison (FCFS vs AI)."""
    plot_multi_scheduler_comparison(
        {"Default (FCFS)": baseline_metrics, "AI (PPO+LSTM)": ai_metrics},
        save=save,
    )
