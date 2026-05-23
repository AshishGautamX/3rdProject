"""
main.py — Local entry point (optional, mirrors colab_loader.py logic).
Run from the repo root: python main.py --data_path <path>
"""
import argparse
import numpy as np
import os

from data.loader import load_all
from data.preprocessor import preprocess
from data.feature_engineer import build_features, make_sequences
from models.baseline_scheduler import simulate_baseline
from models.lstm_predictor import train_lstm, predict
from models.rl_scheduler import train_rl, evaluate_rl
from evaluation.metrics import summarise
from evaluation.visualiser import plot_metric_comparison, plot_workload_prediction
from config.settings import MAX_SLOTS, WINDOW_SIZE, HORIZON, WEIGHTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description="AI-Driven Serverless Scheduler")
    p.add_argument("--data_path", default="/content/drive/MyDrive/serverless_data/")
    p.add_argument("--skip_train", action="store_true",
                   help="Load saved weights instead of retraining")
    return p.parse_args()


def main():
    args = parse_args()

    # ── Data ──────────────────────────────────────────────────────────────────
    inv_df, _, _ = load_all(args.data_path)
    train_raw, val_raw, test_raw, scaler = preprocess(inv_df)

    train_1d = train_raw.mean(axis=1)
    test_1d  = test_raw.mean(axis=1)

    X_train, y_train = make_sequences(build_features(train_raw))
    X_val,   y_val   = make_sequences(build_features(val_raw))
    X_test,  _       = make_sequences(build_features(test_raw))

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_records = simulate_baseline(test_1d, max_slots=10)
    baseline_metrics = summarise(baseline_records, len(test_1d), MAX_SLOTS,
                                  label="DEFAULT SCHEDULER (FCFS)")

    # ── LSTM ──────────────────────────────────────────────────────────────────
    lstm_w = os.path.join(WEIGHTS_DIR, "lstm_best.pt")
    if args.skip_train and os.path.exists(lstm_w):
        from models.lstm_predictor import load_lstm
        lstm_model = load_lstm(lstm_w, X_train.shape[2])
    else:
        lstm_model = train_lstm(X_train, y_train, X_val, y_val, weights_path=lstm_w)

    preds_1d = predict(lstm_model, X_test)[:, 0, :].mean(axis=1)
    aligned  = np.zeros_like(test_1d)
    aligned[WINDOW_SIZE: WINDOW_SIZE + len(preds_1d)] = preds_1d

    preds_2d_train = np.tile(train_1d[:, None], (1, HORIZON))
    preds_2d_test  = np.tile(aligned[:, None],  (1, HORIZON))

    # ── RL ────────────────────────────────────────────────────────────────────
    rl_w = os.path.join(WEIGHTS_DIR, "ppo_policy")
    if args.skip_train and os.path.exists(rl_w + ".zip"):
        from models.rl_scheduler import load_rl
        rl_model = load_rl(rl_w, test_1d, preds_2d_test)
    else:
        rl_model = train_rl(train_1d, preds_2d_train, weights_path=rl_w)

    ai_records = evaluate_rl(rl_model, test_1d, preds_2d_test)
    ai_metrics = summarise(ai_records, len(test_1d), MAX_SLOTS,
                            label="AI SCHEDULER (PPO + LSTM)")

    # ── Visualise ─────────────────────────────────────────────────────────────
    plot_metric_comparison(baseline_metrics, ai_metrics)
    plot_workload_prediction(test_1d, aligned)


if __name__ == "__main__":
    main()
