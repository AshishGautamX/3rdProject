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
EXTRACT_PATH = "/content/azure_data/"

if not os.path.exists(ARCHIVE_PATH):
    raise FileNotFoundError(
        f"Archive not found at {ARCHIVE_PATH}\n"
        "Upload 'azurefunctions-dataset2019.tar.xz' into your Drive "
        "'serverless_data' folder and retry."
    )

if not os.path.exists(EXTRACT_PATH):
    print(f"Extracting {ARCHIVE_NAME} ...")
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with tarfile.open(ARCHIVE_PATH, "r:xz") as tar:
        tar.extractall(EXTRACT_PATH, filter="data")
    print("Extraction complete.")
else:
    print("Already extracted — skipping.")

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
    print("Repo found — pulling latest ...")
    subprocess.run(["git", "-C", repo_dir, "pull", "-q"], check=True)
else:
    print("Cloning repo ...")
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/AshishGautamX/3rdProject.git",
                    repo_dir], check=True)
sys.path.insert(0, repo_dir)
print("Repo ready.")

# === SECTION 3: (Optional) Groq API Key =====================================
# Disabled by default. Uncomment only if you want the LLM hint layer.
# import os
# from getpass import getpass
# os.environ["GROQ_API_KEY"] = getpass("Enter Groq API key: ")

# === SECTION 4: Load & Preprocess Data =======================================
import numpy as np

from data.loader import load_aggregated_timeseries
from data.preprocessor import preprocess
from data.feature_engineer import build_features, make_sequences
from config.settings import ARRIVAL_SCALE, AVG_DURATION_MS, MAX_SLOTS, HORIZON, WINDOW_SIZE

print("Loading & aggregating data (one CSV at a time) ...")
raw_ts = load_aggregated_timeseries(DATA_PATH)       # shape (20160,) raw counts

print("\nPreprocessing (normalise + split) ...")
train_1d, val_1d, test_1d, scaler = preprocess(raw_ts)   # all in [0, 1]

# Scale normalised load → realistic integer job arrivals for the simulator.
# ARRIVAL_SCALE=50: peak ≈ 50 jobs/min vs MAX_SLOTS=20 → real queuing pressure.
train_arr = np.clip(np.round(train_1d * ARRIVAL_SCALE).astype(np.float32), 0, ARRIVAL_SCALE * 3)
val_arr   = np.clip(np.round(val_1d   * ARRIVAL_SCALE).astype(np.float32), 0, ARRIVAL_SCALE * 3)
test_arr  = np.clip(np.round(test_1d  * ARRIVAL_SCALE).astype(np.float32), 0, ARRIVAL_SCALE * 3)

print(f"\nScaled arrivals — train: {train_arr.min():.0f}–{train_arr.max():.0f} "
      f"| test: {test_arr.min():.0f}–{test_arr.max():.0f} jobs/min")

# Build LSTM features & sequences from normalised (not scaled) values
print("\nBuilding LSTM features ...")
train_feat = build_features(train_1d)
val_feat   = build_features(val_1d)
test_feat  = build_features(test_1d)

X_train, y_train = make_sequences(train_feat)
X_val,   y_val   = make_sequences(val_feat)
X_test,  y_test  = make_sequences(test_feat)

print(f"Sequence shapes: X_train={X_train.shape} | X_test={X_test.shape}")
gc.collect()

# === SECTION 5: Baseline Scheduler ===========================================
from models.baseline_scheduler import simulate_baseline
from evaluation.metrics import summarise

print(f"\nRunning baseline (FCFS, 10 slots, avg_dur={AVG_DURATION_MS}ms) ...")
baseline_records = simulate_baseline(
    test_arr, max_slots=10,
    avg_duration_ms=AVG_DURATION_MS, strategy="fcfs"
)
baseline_metrics = summarise(baseline_records, len(test_arr), MAX_SLOTS,
                              label="DEFAULT SCHEDULER (FCFS)")

# === SECTION 6: Train LSTM Predictor =========================================
from models.lstm_predictor import train_lstm, predict, load_lstm
from config.settings import WEIGHTS_DIR

lstm_weights = os.path.join(WEIGHTS_DIR, "lstm_best.pt")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

if os.path.exists(lstm_weights):
    print(f"\n[lstm] Saved weights found — loading.")
    lstm_model = load_lstm(lstm_weights, input_dim=X_train.shape[2])
else:
    print("\nTraining LSTM ...")
    lstm_model = train_lstm(X_train, y_train, X_val, y_val,
                            weights_path=lstm_weights)

preds = predict(lstm_model, X_test)     # (N, HORIZON)
preds_1d = preds[:, 0]                  # next-minute forecast (normalised)

# Align to test_arr length
aligned_norm = np.zeros_like(test_1d)
aligned_norm[WINDOW_SIZE: WINDOW_SIZE + len(preds_1d)] = preds_1d
aligned_arr  = np.clip(np.round(aligned_norm * ARRIVAL_SCALE).astype(np.float32),
                       0, ARRIVAL_SCALE * 3)

print("\nLSTM done.")
gc.collect()

# === SECTION 7: Train RL Scheduler ===========================================
from models.rl_scheduler import train_rl, evaluate_rl, load_rl

rl_weights     = os.path.join(WEIGHTS_DIR, "ppo_policy")
preds_2d_train = np.tile(train_1d[:, None], (1, HORIZON))   # (T, H) normalised
preds_2d_test  = np.tile(aligned_norm[:, None], (1, HORIZON))

# RL env receives SCALED arrivals so queue dynamics are realistic
if os.path.exists(rl_weights + ".zip"):
    print("\n[rl] Saved policy found — loading.")
    rl_model = load_rl(rl_weights, train_arr, preds_2d_train)
else:
    print("\nTraining PPO RL scheduler ...")
    rl_model = train_rl(train_arr, preds_2d_train, weights_path=rl_weights)

gc.collect()

# === SECTION 8: Evaluate AI Scheduler ========================================
print("\nEvaluating AI scheduler ...")
ai_records = evaluate_rl(rl_model, test_arr, preds_2d_test)
ai_metrics = summarise(ai_records, len(test_arr), MAX_SLOTS,
                        label="AI SCHEDULER (PPO + LSTM)")

# === SECTION 9: Compare & Visualise ==========================================
from evaluation.visualiser import plot_metric_comparison, plot_workload_prediction

print("\nGenerating plots ...")
plot_metric_comparison(baseline_metrics, ai_metrics, save=True)
plot_workload_prediction(test_1d, aligned_norm, n_points=500, save=True)

print("\n" + "=" * 62)
print(f"{'METRIC':<28} {'DEFAULT':>14} {'AI-DRIVEN':>14}")
print("=" * 62)
rows = [
    ("Mean Response (ms)",  "response_mean"),
    ("P95 Response (ms)",   "response_p95"),
    ("P99 Response (ms)",   "response_p99"),
    ("Throughput (j/s)",    "throughput"),
    ("Utilisation (%)",     "utilisation"),
]
for label, key in rows:
    bv = baseline_metrics[key] * (100 if key == "utilisation" else 1)
    av = ai_metrics[key]       * (100 if key == "utilisation" else 1)
    print(f"  {label:<26} {bv:>14.4f} {av:>14.4f}")
print("=" * 62)
