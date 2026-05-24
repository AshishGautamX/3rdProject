# =============================================================================
# COLAB LOADER — AI-Driven Dynamic Job Scheduling in Serverless Platforms
# GitHub: https://github.com/AshishGautamX/3rdProject
# Run each SECTION as a separate Colab cell.
# =============================================================================

# === SECTION 0: Install Dependencies =========================================
!pip install -q torch stable-baselines3 gymnasium pandas numpy scikit-learn \
             matplotlib seaborn groq tqdm

# === SECTION 1: Mount Google Drive & Extract Dataset =========================
from google.colab import drive
import os, tarfile, gc

drive.mount('/content/drive')

DRIVE_PATH   = "/content/drive/MyDrive/serverless_data/"
ARCHIVE_NAME = "azurefunctions-dataset2019.tar.xz"
ARCHIVE_PATH = os.path.join(DRIVE_PATH, ARCHIVE_NAME)
EXTRACT_PATH = "/content/azure_data/"   # Colab's fast local SSD

if not os.path.exists(ARCHIVE_PATH):
    raise FileNotFoundError(
        f"Archive not found at {ARCHIVE_PATH}\n"
        "Upload 'azurefunctions-dataset2019.tar.xz' into your Drive "
        "'serverless_data' folder and retry."
    )

if not os.path.exists(EXTRACT_PATH):
    print(f"Extracting {ARCHIVE_NAME} — this may take a few minutes ...")
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with tarfile.open(ARCHIVE_PATH, "r:xz") as tar:
        tar.extractall(EXTRACT_PATH, filter="data")
    print("Extraction complete.")
else:
    print("Archive already extracted — skipping.")

# Auto-locate the folder that contains the invocation CSVs (handles nesting)
candidates = [
    root for root, _, files in os.walk(EXTRACT_PATH)
    if any(f.startswith("invocations_per_function") for f in files)
]
if not candidates:
    raise RuntimeError(f"Could not find invocation CSVs inside {EXTRACT_PATH}")
DATA_PATH = candidates[0]
print(f"Data path: {DATA_PATH}")

# === SECTION 2: Clone GitHub Repo & Add to Path ==============================
import subprocess, sys

repo_dir = "/content/3rdProject"
if os.path.exists(repo_dir):
    print("Repo already cloned — pulling latest ...")
    subprocess.run(["git", "-C", repo_dir, "pull", "-q"], check=True)
else:
    print("Cloning repo ...")
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/AshishGautamX/3rdProject.git",
                    repo_dir], check=True)

sys.path.insert(0, repo_dir)
print("Repo ready.")

# === SECTION 3: (Optional) Set Groq API Key ==================================
# LLM hint layer is OFF by default (USE_LLM = False in config/settings.py).
# Uncomment below ONLY if you want to enable it.
# import os
# from getpass import getpass
# os.environ["GROQ_API_KEY"] = getpass("Enter Groq API key: ")
# Then set USE_LLM = True in config/settings.py before pushing to GitHub.

# === SECTION 4: Load & Preprocess Data =======================================
# Memory usage: ~200 MB peak (streams one CSV at a time, aggregates, discards)
import numpy as np

from data.loader import load_aggregated_timeseries, load_avg_duration
from data.preprocessor import preprocess
from data.feature_engineer import build_features, make_sequences

print("Loading & aggregating invocation data (one day at a time) ...")
raw_ts = load_aggregated_timeseries(DATA_PATH)   # shape: (20160,)

print("\nLoading average duration data ...")
avg_dur_ts = load_avg_duration(DATA_PATH)        # shape: (20160,)

print("\nPreprocessing ...")
train_1d, val_1d, test_1d, scaler = preprocess(raw_ts)

# Build feature matrices — shape (T, 9), memory ~1 MB each
print("\nBuilding features ...")
train_feat = build_features(train_1d)
val_feat   = build_features(val_1d)
test_feat  = build_features(test_1d)

# Sliding-window sequences for LSTM — shape (N, WINDOW, 9) / (N, HORIZON)
X_train, y_train = make_sequences(train_feat)
X_val,   y_val   = make_sequences(val_feat)
X_test,  y_test  = make_sequences(test_feat)

print(f"\nSequence shapes: X_train={X_train.shape} | X_test={X_test.shape}")
print(f"y shapes       : y_train={y_train.shape} | y_test={y_test.shape}")

gc.collect()

# === SECTION 5: Run Baseline Scheduler =======================================
from models.baseline_scheduler import simulate_baseline
from evaluation.metrics import summarise
from config.settings import MAX_SLOTS

# Use raw (un-normalised) test load for the simulator
split_idx    = int(len(raw_ts) * 0.90)
test_raw_1d  = raw_ts[split_idx:]
avg_dur_test = float(avg_dur_ts[split_idx:].mean())

print("\nRunning baseline (FCFS) scheduler ...")
baseline_records = simulate_baseline(
    test_raw_1d, max_slots=10,
    avg_duration_ms=avg_dur_test, strategy="fcfs"
)
baseline_metrics = summarise(baseline_records, len(test_raw_1d), MAX_SLOTS,
                              label="DEFAULT SCHEDULER (FCFS)")

# === SECTION 6: Train LSTM Predictor =========================================
from models.lstm_predictor import train_lstm, predict, load_lstm
from config.settings import WEIGHTS_DIR, WINDOW_SIZE

lstm_weights = os.path.join(WEIGHTS_DIR, "lstm_best.pt")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

if os.path.exists(lstm_weights):
    print(f"\n[lstm] Saved weights found — loading.")
    lstm_model = load_lstm(lstm_weights, input_dim=X_train.shape[2])
else:
    print("\nTraining LSTM forecaster ...")
    lstm_model = train_lstm(X_train, y_train, X_val, y_val,
                            weights_path=lstm_weights)

# Predictions shape: (N, HORIZON) — first step = next-minute load forecast
preds = predict(lstm_model, X_test)          # (N, HORIZON)
preds_1d = preds[:, 0]                       # (N,) — use first horizon step

# Align to test_1d length (pad leading zeros for the warmup window)
aligned_preds = np.zeros_like(test_1d)
aligned_preds[WINDOW_SIZE: WINDOW_SIZE + len(preds_1d)] = preds_1d

print("\nLSTM done.")
gc.collect()

# === SECTION 7: Train RL Scheduler ===========================================
from models.rl_scheduler import train_rl, evaluate_rl, load_rl
from config.settings import HORIZON

rl_weights = os.path.join(WEIGHTS_DIR, "ppo_policy")

# Build 2-D prediction arrays for ServerlessEnv (T, HORIZON)
preds_2d_train = np.tile(train_1d[:, None], (1, HORIZON))
preds_2d_test  = np.tile(aligned_preds[:, None], (1, HORIZON))

if os.path.exists(rl_weights + ".zip"):
    print("\n[rl] Saved policy found — loading.")
    rl_model = load_rl(rl_weights, test_1d, preds_2d_test)
else:
    print("\nTraining PPO RL scheduler ...")
    rl_model = train_rl(train_1d, preds_2d_train, weights_path=rl_weights)

gc.collect()

# === SECTION 8: Evaluate AI Scheduler ========================================
print("\nEvaluating AI scheduler on test set ...")
ai_records = evaluate_rl(rl_model, test_1d, preds_2d_test)
ai_metrics = summarise(ai_records, len(test_1d), MAX_SLOTS,
                        label="AI SCHEDULER (PPO + LSTM)")

# === SECTION 9: Compare & Visualise ==========================================
from evaluation.visualiser import plot_metric_comparison, plot_workload_prediction

print("\nGenerating comparison plots ...")
plot_metric_comparison(baseline_metrics, ai_metrics, save=True)
plot_workload_prediction(test_1d, aligned_preds, n_points=500, save=True)

# Final comparison table
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
