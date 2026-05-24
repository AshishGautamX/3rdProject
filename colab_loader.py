# =============================================================================
# COLAB LOADER — AI-Driven Dynamic Job Scheduling in Serverless Platforms
# Run each SECTION as a separate Colab cell.
# =============================================================================

# === SECTION 0: Install Dependencies =========================================
!pip install -q torch stable-baselines3 gymnasium pandas numpy scikit-learn \
             matplotlib seaborn groq tqdm

# === SECTION 1: Mount Drive & Extract Dataset ================================
from google.colab import drive
import os, tarfile, gc

drive.mount('/content/drive')

DRIVE_PATH   = "/content/drive/MyDrive/serverless_data/"
ARCHIVE_PATH = os.path.join(DRIVE_PATH, "azurefunctions-dataset2019.tar.xz")
EXTRACT_PATH = "/content/azure_data/"

if not os.path.exists(ARCHIVE_PATH):
    raise FileNotFoundError(
        f"Archive not found: {ARCHIVE_PATH}\n"
        "Place 'azurefunctions-dataset2019.tar.xz' in Drive/serverless_data/")

if not os.path.exists(EXTRACT_PATH):
    print("Extracting dataset ...")
    os.makedirs(EXTRACT_PATH, exist_ok=True)
    with tarfile.open(ARCHIVE_PATH, "r:xz") as tar:
        tar.extractall(EXTRACT_PATH, filter="data")
else:
    print("Dataset already extracted.")

candidates = [
    root for root, _, files in os.walk(EXTRACT_PATH)
    if any(f.startswith("invocations_per_function") for f in files)
]
if not candidates:
    raise RuntimeError(f"No invocation CSVs found in {EXTRACT_PATH}")
DATA_PATH = candidates[0]
print(f"Data path: {DATA_PATH}")

# === SECTION 2: Clone Repo ====================================================
import subprocess, sys

repo_dir = "/content/3rdProject"
if os.path.exists(repo_dir):
    subprocess.run(["git", "-C", repo_dir, "pull", "-q"], check=True)
    print("Repo updated.")
else:
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/AshishGautamX/3rdProject.git",
                    repo_dir], check=True)
    print("Repo cloned.")
sys.path.insert(0, repo_dir)

# === SECTION 3: (Optional) Groq API Key =====================================
# from getpass import getpass
# os.environ["GROQ_API_KEY"] = getpass("Groq API key: ")

# === SECTION 4: Load & Preprocess Data =======================================
import numpy as np
import warnings
warnings.filterwarnings("ignore", message=".*Gym has been unmaintained.*")
warnings.filterwarnings("ignore", category=DeprecationWarning)

from data.loader import load_aggregated_timeseries
from data.preprocessor import preprocess
from data.feature_engineer import build_features, make_sequences
from config.settings import (
    ARRIVAL_SCALE, AVG_DURATION_MS, MAX_SLOTS, HORIZON,
    WINDOW_SIZE, BASELINE_SLOTS, WEIGHTS_DIR,
)

print("Loading data ...")
raw_ts = load_aggregated_timeseries(DATA_PATH)

train_1d, val_1d, test_1d, scaler = preprocess(raw_ts)

train_arr = np.clip(np.round(train_1d * ARRIVAL_SCALE).astype(np.float32), 0, ARRIVAL_SCALE * 3)
val_arr   = np.clip(np.round(val_1d   * ARRIVAL_SCALE).astype(np.float32), 0, ARRIVAL_SCALE * 3)
test_arr  = np.clip(np.round(test_1d  * ARRIVAL_SCALE).astype(np.float32), 0, ARRIVAL_SCALE * 3)

print(f"Arrivals — train: {train_arr.min():.0f}–{train_arr.max():.0f} "
      f"| test: {test_arr.min():.0f}–{test_arr.max():.0f} jobs/min")

train_feat = build_features(train_1d)
val_feat   = build_features(val_1d)
test_feat  = build_features(test_1d)
X_train, y_train = make_sequences(train_feat)
X_val,   y_val   = make_sequences(val_feat)
X_test,  y_test  = make_sequences(test_feat)
print(f"Sequences: X_train={X_train.shape} | X_test={X_test.shape}")
gc.collect()

# === SECTION 5: Baselines ====================================================
from models.baseline_scheduler import simulate_baseline, simulate_reactive
from evaluation.metrics import summarise

print("\n── Baselines ──────────────────────────────────────────────────────────")
print(f"  {'Scheduler':<40} {'Mean RT':>9}  {'P99':>9}  {'Thr':>8}  {'Util':>6}  {'Slots':>5}")
print(f"  {'-'*90}")

fcfs_rec      = simulate_baseline(test_arr, max_slots=BASELINE_SLOTS,
                                  avg_duration_ms=AVG_DURATION_MS, strategy="fcfs")
fcfs_metrics  = summarise(fcfs_rec, len(test_arr), BASELINE_SLOTS, "FCFS (10 slots)")

rr_rec        = simulate_baseline(test_arr, max_slots=BASELINE_SLOTS,
                                  avg_duration_ms=AVG_DURATION_MS, strategy="round_robin")
rr_metrics    = summarise(rr_rec, len(test_arr), BASELINE_SLOTS, "Round-Robin (10 slots)")

edf_rec       = simulate_baseline(test_arr, max_slots=BASELINE_SLOTS,
                                  avg_duration_ms=AVG_DURATION_MS, strategy="edf")
edf_metrics   = summarise(edf_rec, len(test_arr), BASELINE_SLOTS, "EDF (10 slots)")

reactive_rec  = simulate_reactive(test_arr, avg_duration_ms=AVG_DURATION_MS)
reactive_metrics = summarise(reactive_rec, len(test_arr), MAX_SLOTS, "Reactive (HPA-style)")

all_metrics = {
    "FCFS":        fcfs_metrics,
    "Round-Robin": rr_metrics,
    "EDF":         edf_metrics,
    "Reactive":    reactive_metrics,
}
gc.collect()

# === SECTION 6: LSTM Predictor ===============================================
from models.lstm_predictor import train_lstm, predict, load_lstm

lstm_weights = os.path.join(WEIGHTS_DIR, "lstm_best.pt")
os.makedirs(WEIGHTS_DIR, exist_ok=True)

if os.path.exists(lstm_weights):
    print("\n[lstm] Loading saved weights ...")
    lstm_model = load_lstm(lstm_weights, input_dim=X_train.shape[2])
else:
    print("\n[lstm] Training ...")
    lstm_model = train_lstm(X_train, y_train, X_val, y_val,
                            weights_path=lstm_weights)

preds    = predict(lstm_model, X_test)
preds_1d = preds[:, 0]

aligned_norm = np.zeros_like(test_1d)
aligned_norm[WINDOW_SIZE: WINDOW_SIZE + len(preds_1d)] = preds_1d
aligned_arr  = np.clip(np.round(aligned_norm * ARRIVAL_SCALE).astype(np.float32),
                       0, ARRIVAL_SCALE * 3)
gc.collect()

# === SECTION 7: Train RL Scheduler ===========================================
from models.rl_scheduler import train_rl, evaluate_rl, load_rl

rl_weights = os.path.join(WEIGHTS_DIR, "ppo_policy")
preds_2d_train = np.tile(train_arr[:, None], (1, HORIZON)).astype(np.float32)
preds_2d_test  = np.tile(aligned_arr[:, None], (1, HORIZON)).astype(np.float32)

# Set True whenever env observation space changes (e.g. new state dims).
FORCE_RL_RETRAIN = True
if FORCE_RL_RETRAIN and os.path.exists(rl_weights + ".zip"):
    os.remove(rl_weights + ".zip")
    print("[rl] Old weights removed — retraining ...")

if os.path.exists(rl_weights + ".zip"):
    print("[rl] Loading saved policy ...")
    global_max_load = float(train_arr.max())
    rl_model = load_rl(rl_weights, train_arr, preds_2d_train,
                       global_max_load=global_max_load)
else:
    print("[rl] Training PPO (this takes ~3 min) ...")
    rl_model, global_max_load = train_rl(train_arr, preds_2d_train,
                                          weights_path=rl_weights)
gc.collect()

# === SECTION 8: Evaluate AI Scheduler ========================================
print("\n[rl] Evaluating on test set ...")
ai_records = evaluate_rl(rl_model, test_arr, preds_2d_test,
                          global_max_load=global_max_load)
ai_metrics = summarise(ai_records, len(test_arr), MAX_SLOTS, "LSTM+PPO (ours)")
all_metrics["LSTM+PPO"] = ai_metrics

# === SECTION 9: Ablation Study ===============================================
from evaluation.ablation import run_ablation

print("\n── Ablation Study ─────────────────────────────────────────────────────")
ablation_metrics = run_ablation(
    test_arr          = test_arr,
    lstm_preds_scaled = aligned_arr,
    train_arr         = train_arr,
    ai_metrics        = ai_metrics,
    weights_dir       = WEIGHTS_DIR,
    total_minutes     = len(test_arr),
)
gc.collect()

# === SECTION 10: Comparison Table & Plots ====================================
from evaluation.visualiser import (
    plot_multi_scheduler_comparison,
    plot_ablation,
    plot_workload_prediction,
    print_comparison_table,
)

print("\n── Final Comparison Table ─────────────────────────────────────────────")
print_comparison_table(all_metrics)

print("\n── Improvement vs FCFS ────────────────────────────────────────────────")
for name, m in all_metrics.items():
    if name == "FCFS":
        continue
    rt_imp  = (fcfs_metrics["response_mean"] - m["response_mean"]) / max(fcfs_metrics["response_mean"], 1) * 100
    p99_imp = (fcfs_metrics["response_p99"]  - m["response_p99"])  / max(fcfs_metrics["response_p99"],  1) * 100
    print(f"  {name:<20} Mean RT: {rt_imp:+.1f}%   P99: {p99_imp:+.1f}%")

print("\nGenerating plots ...")
plot_workload_prediction(test_1d, aligned_norm, n_points=500, save=True)
plot_multi_scheduler_comparison(all_metrics, save=True)
plot_ablation(ablation_metrics, save=True)
print("Done.")
