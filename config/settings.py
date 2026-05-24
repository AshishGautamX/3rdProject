"""Central configuration for the AI-Driven Serverless Scheduler."""

# ── Data ──────────────────────────────────────────────────────────────────────
DRIVE_DATA_PATH            = "/content/drive/MyDrive/serverless_data/"
INVOCATION_FILE_TEMPLATE   = "invocations_per_function_md.anon.d{day:02d}.csv"
DURATION_FILE_TEMPLATE     = "function_durations_percentiles.anon.d{day:02d}.csv"
MEMORY_FILE_TEMPLATE       = "app_memory_percentiles.anon.d{day:02d}.csv"
NUM_DAYS                   = 14

# ── Preprocessing ─────────────────────────────────────────────────────────────
MIN_TOTAL_INVOCATIONS = 100
TRAIN_RATIO           = 0.80
VAL_RATIO             = 0.10    # test = remaining 0.10

# ── LSTM ──────────────────────────────────────────────────────────────────────
WINDOW_SIZE     = 30
HORIZON         = 5
LSTM_HIDDEN_DIM = 128
LSTM_LAYERS     = 2
LSTM_DROPOUT    = 0.2
LSTM_LR         = 1e-3
LSTM_EPOCHS     = 50
LSTM_BATCH_SIZE = 64
LSTM_PATIENCE   = 7

# ── RL / Environment ──────────────────────────────────────────────────────────
MAX_SLOTS     = 20
MIN_SLOTS     = 1
ALPHA         = 1.5    # response-time penalty weight
BETA          = 0.0    # idle-slot penalty (0 = agent freely over-provisions)
PPO_TIMESTEPS = 200_000

# PPO stability hyperparameters (explicit to prevent oscillation/entropy collapse)
PPO_ENT_COEF      = 0.01    # entropy bonus — keeps policy from collapsing prematurely
PPO_N_STEPS       = 4096    # larger rollout buffer → smoother gradient estimates
PPO_BATCH_SIZE    = 128     # mini-batch size for PPO update
PPO_N_EPOCHS      = 10      # epochs per PPO update
PPO_CLIP_RANGE    = 0.2
PPO_MAX_GRAD_NORM = 0.5     # gradient clipping — prevents value_loss spikes
PPO_VF_COEF       = 0.5     # value function coefficient

# ── Simulator ─────────────────────────────────────────────────────────────────
# Scale normalised load (0-1) → realistic job arrivals per minute.
ARRIVAL_SCALE   = 25
AVG_DURATION_MS = 200.0   # ms — used in baseline AND RL env

# Baseline slots = avg expected arrivals so FCFS handles average load
# but queues during spikes. Fair and defensible static baseline.
BASELINE_SLOTS  = 10

# Reactive autoscaler thresholds (HPA-style rule-based baseline)
REACTIVE_SCALE_UP_Q   = 5   # scale up 1 slot if queue depth exceeds this
REACTIVE_SCALE_DOWN_Q = 2   # scale down 1 slot if queue depth falls below this

# ── LLM (optional) ────────────────────────────────────────────────────────────
USE_LLM          = False   # set True to enable Groq hint layer
GROQ_MODEL       = "llama-3.3-70b-versatile"
GROQ_API_KEY_ENV = "GROQ_API_KEY"
SPIKE_THRESHOLD  = 2.0     # std-devs above mean to trigger LLM

# ── GitHub ────────────────────────────────────────────────────────────────────
GITHUB_USER = "AshishGautamX"
GITHUB_REPO = "3rdProject"

# ── Output paths ──────────────────────────────────────────────────────────────
WEIGHTS_DIR = "/content/drive/MyDrive/serverless_data/weights/"
RESULTS_DIR = "/content/drive/MyDrive/serverless_data/results/"
