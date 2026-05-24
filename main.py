"""
main.py — Local entry point (optional, mirrors colab_loader.py logic).
Run from the repo root: python main.py --data_path <path>
"""
import argparse
import numpy as np
import os

from data.loader import load_aggregated_timeseries
from data.preprocessor import preprocess
from data.feature_engineer import build_features, make_sequences
from models.baseline_scheduler import simulate_baseline
from models.lstm_predictor import train_lstm, predict
from models.rl_scheduler import train_rl, evaluate_rl
from evaluation.metrics import summarise
from evaluation.visualiser import plot_metric_comparison, plot_workload_prediction
from config.settings import (
    MAX_SLOTS, WINDOW_SIZE, HORIZON, WEIGHTS_DIR,
    ARRIVAL_SCALE, AVG_DURATION_MS,
)

BASELINE_SLOTS = 10


def parse_args():
    p = argparse.ArgumentParser(description="AI-Driven Serverless Scheduler")
    p.add_argument("--data_path", default="/content/drive/MyDrive/serverless_data/")
    p.add_argument("--skip_train", action="store_true",
                   help="Load saved weights instead of retraining")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Data ──────────────────────────────────────────────────────────────────
    raw_ts = load_aggregated_timeseries(args.data_path)
    train_1d, val_1d, test_1d, scaler = preprocess(raw_ts)

    # Scale normalised → integer arrivals (consistent with colab_loader)
    train_arr = np.clip(np.round(train_1d * ARRIVAL_SCALE).astype(np.float32), 0, ARRIVAL_SCALE * 3)
    test_arr  = np.clip(np.round(test_1d  * ARRIVAL_SCALE).astype(np.float32), 0, ARRIVAL_SCALE * 3)

    X_train, y_train = make_sequences(build_features(train_1d))
    X_val,   y_val   = make_sequences(build_features(val_1d))
    X_test,  _       = make_sequences(build_features(test_1d))

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_records = simulate_baseline(test_arr, max_slots=BASELINE_SLOTS,
                                         avg_duration_ms=AVG_DURATION_MS)
    baseline_metrics = summarise(baseline_records, len(test_arr), BASELINE_SLOTS,
                                 label="DEFAULT SCHEDULER (FCFS)")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    lstm_w = os.path.join(WEIGHTS_DIR, "lstm_best.pt")
    if args.skip_train and os.path.exists(lstm_w):
        from models.lstm_predictor import load_lstm
        lstm_model = load_lstm(lstm_w, X_train.shape[2])
    else:
        lstm_model = train_lstm(X_train, y_train, X_val, y_val, weights_path=lstm_w)

    preds    = predict(lstm_model, X_test)   # (N, HORIZON)
    preds_1d = preds[:, 0]                   # next-minute forecast, normalised

    aligned_norm = np.zeros_like(test_1d)
    aligned_norm[WINDOW_SIZE: WINDOW_SIZE + len(preds_1d)] = preds_1d
    aligned_arr  = np.clip(np.round(aligned_norm * ARRIVAL_SCALE).astype(np.float32),
                           0, ARRIVAL_SCALE * 3)

    preds_2d_train = np.tile(train_arr[:, None], (1, HORIZON)).astype(np.float32)
    preds_2d_test  = np.tile(aligned_arr[:, None], (1, HORIZON)).astype(np.float32)

    # ── RL ────────────────────────────────────────────────────────────────────
    rl_w = os.path.join(WEIGHTS_DIR, "ppo_policy")
    if args.skip_train and os.path.exists(rl_w + ".zip"):
        from models.rl_scheduler import load_rl
        global_max_load = float(train_arr.max())
        rl_model = load_rl(rl_w, train_arr, preds_2d_train,
                           global_max_load=global_max_load)
    else:
        rl_model, global_max_load = train_rl(train_arr, preds_2d_train,
                                              weights_path=rl_w)

    ai_records = evaluate_rl(rl_model, test_arr, preds_2d_test,
                              global_max_load=global_max_load)
    ai_metrics = summarise(ai_records, len(test_arr), MAX_SLOTS,
                           label="AI SCHEDULER (PPO + LSTM)")

    # ── Visualise ─────────────────────────────────────────────────────────────
    plot_metric_comparison(baseline_metrics, ai_metrics)
    plot_workload_prediction(test_1d, aligned_norm)


if __name__ == "__main__":
    main()
