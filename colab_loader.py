# =============================================================================
# COLAB LOADER — AI-Driven Dynamic Job Scheduling in Serverless Platforms
# GitHub: https://github.com/AshishGautamX/3rdProject
# Run each SECTION as a separate Colab cell.
# =============================================================================

# === SECTION 0: Install Dependencies =========================================
# !pip install -q torch stable-baselines3 gymnasium pandas numpy scikit-learn \
#              matplotlib seaborn groq tqdm

# === SECTION 1: Mount Google Drive & Set Data Path ===========================
# from google.colab import drive
# drive.mount('/content/drive')
# DRIVE_PATH = "/content/drive/MyDrive/serverless_data/"
# print(f"Data path: {DRIVE_PATH}")

# === SECTION 2: Clone GitHub Repo & Add to Path ==============================
# import subprocess, sys
# subprocess.run(["git", "clone", "-q",
#                 "https://github.com/AshishGautamX/3rdProject.git",
#                 "/content/3rdProject"], check=True)
# sys.path.insert(0, "/content/3rdProject")
# print("Repo cloned and added to path.")

# === SECTION 3: (Optional) Set Groq API Key ==================================
# import os
# from getpass import getpass
# # Skip this section if you don't want the LLM hint layer.
# os.environ["GROQ_API_KEY"] = getpass("Enter Groq API key (or press Enter to skip): ")
# # Also flip USE_LLM in config/settings.py to True if you set a key.

# === SECTION 4: Load & Preprocess Data =======================================
import sys
sys.path.insert(0, "/content/3rdProject")

from data.loader import load_all
from data.preprocessor import preprocess
from data.feature_engineer import build_features, make_sequences

DRIVE_PATH = "/content/drive/MyDrive/serverless_data/"

print("Loading data from Google Drive ...")
inv_df, dur_df, mem_df = load_all(DRIVE_PATH)

print("\nPreprocessing ...")
train_raw, val_raw, test_raw, scaler = preprocess(inv_df)

# Aggregate across functions → single invocation load signal (mean per minute)
import numpy as np
train_1d = train_raw.mean(axis=1)
val_1d   = val_raw.mean(axis=1)
test_1d  = test_raw.mean(axis=1)

# Feature engineering + sequence creation
train_feat = build_features(train_raw)
val_feat   = build_features(val_raw)
test_feat  = build_features(test_raw)

X_train, y_train = make_sequences(train_feat)
X_val,   y_val   = make_sequences(val_feat)
X_test,  y_test  = make_sequences(test_feat)

print(f"\nSequence shapes — X_train: {X_train.shape} | X_test: {X_test.shape}")

# === SECTION 5: Run Baseline Scheduler =======================================
from models.baseline_scheduler import simulate_baseline
from evaluation.metrics import summarise
from config.settings import MAX_SLOTS

print("\nRunning baseline (FCFS) scheduler ...")
baseline_records = simulate_baseline(test_1d, max_slots=10, strategy="fcfs")
baseline_metrics = summarise(baseline_records, len(test_1d), MAX_SLOTS,
                              label="DEFAULT SCHEDULER (FCFS)")

# === SECTION 6: Train LSTM Predictor =========================================
import os
from models.lstm_predictor import train_lstm, predict, load_lstm
from config.settings import WEIGHTS_DIR

lstm_weights = os.path.join(WEIGHTS_DIR, "lstm_best.pt")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

if os.path.exists(lstm_weights):
    print(f"\n[lstm] Found saved weights at {lstm_weights} — loading.")
    lstm_model = load_lstm(lstm_weights, input_dim=X_train.shape[2])
else:
    print("\nTraining LSTM forecaster ...")
    lstm_model = train_lstm(X_train, y_train, X_val, y_val, weights_path=lstm_weights)

# Generate predictions on test set (first horizon step per sample)
preds_raw = predict(lstm_model, X_test)           # (N, horizon, F)
preds_1d  = preds_raw[:, 0, :].mean(axis=1)       # collapse to 1-D

# Align predictions to test_1d length
from config.settings import WINDOW_SIZE
aligned_preds = np.zeros_like(test_1d)
aligned_preds[WINDOW_SIZE: WINDOW_SIZE + len(preds_1d)] = preds_1d

print("\nLSTM training complete.")

# === SECTION 7: Train RL Scheduler ===========================================
from models.rl_scheduler import train_rl, evaluate_rl, load_rl

rl_weights = os.path.join(WEIGHTS_DIR, "ppo_policy")

# Build 2-D prediction array needed by ServerlessEnv
from config.settings import HORIZON
preds_2d_train = np.tile(train_1d[:, None], (1, HORIZON))
preds_2d_test  = np.tile(aligned_preds[:, None], (1, HORIZON))

if os.path.exists(rl_weights + ".zip"):
    print(f"\n[rl] Found saved policy — loading.")
    rl_model = load_rl(rl_weights, test_1d, preds_2d_test)
else:
    print("\nTraining PPO RL scheduler ...")
    rl_model = train_rl(train_1d, preds_2d_train, weights_path=rl_weights)

# === SECTION 8: Evaluate AI Scheduler ========================================
from models.rl_scheduler import evaluate_rl

print("\nEvaluating AI scheduler on test set ...")
ai_records = evaluate_rl(rl_model, test_1d, preds_2d_test)
ai_metrics = summarise(ai_records, len(test_1d), MAX_SLOTS,
                        label="AI SCHEDULER (PPO + LSTM)")

# === SECTION 9: Compare & Visualise ==========================================
from evaluation.visualiser import (plot_metric_comparison,
                                    plot_workload_prediction)

print("\nGenerating comparison plots ...")
plot_metric_comparison(baseline_metrics, ai_metrics, save=True)
plot_workload_prediction(test_1d, aligned_preds, n_points=500, save=True)

# Final side-by-side table
print("\n" + "=" * 62)
print(f"{'METRIC':<28} {'DEFAULT':>14} {'AI-DRIVEN':>14}")
print("=" * 62)
rows = [
    ("Mean Response (ms)",  "response_mean"),
    ("P95 Response (ms)",   "response_p95"),
    ("Throughput (j/s)",    "throughput"),
    ("Utilisation (%)",     "utilisation"),
]
for label, key in rows:
    bv = baseline_metrics[key] * (100 if key == "utilisation" else 1)
    av = ai_metrics[key]       * (100 if key == "utilisation" else 1)
    print(f"  {label:<26} {bv:>14.4f} {av:>14.4f}")
print("=" * 62)
