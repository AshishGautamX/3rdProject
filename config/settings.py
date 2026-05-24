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
ALPHA         = 0.5    # response-time penalty weight
BETA          = 0.2    # idle-slot penalty weight
PPO_TIMESTEPS = 200_000

# ── Simulator ─────────────────────────────────────────────────────────────────
# Scale normalised load (0-1) → realistic job arrivals per minute.
# With MAX_SLOTS=20, ARRIVAL_SCALE=50 means peak ≈ 2.5× capacity → queuing.
ARRIVAL_SCALE   = 50
AVG_DURATION_MS = 200.0   # ms — used consistently in baseline AND RL env

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
