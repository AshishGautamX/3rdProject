"""Custom Gymnasium environment for the serverless scheduling problem.

State  (7-dim): [norm_load, norm_pred_load, norm_queue, norm_slots,
                 norm_queue_trend, norm_time_of_day, norm_recent_avg_load]
Action         : int — concurrency slots to allocate (MIN_SLOTS … MAX_SLOTS)
Reward         : clipped(demand_served_fraction − α * response_norm)
                 Bounded to [-1, 1] for stable PPO value estimation.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from config.settings import MAX_SLOTS, MIN_SLOTS, ALPHA, BETA


class ServerlessEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, workload: np.ndarray, predictions: np.ndarray,
                 avg_duration_ms: float = 200.0, max_load: float = None):
        """
        workload    : 1-D array (T,) — actual invocations per minute (scaled)
        predictions : 2-D array (T, HORIZON) — LSTM look-ahead predictions
        max_load    : normalisation constant; pass training max to keep
                      observation scale consistent across train/test envs.
        """
        super().__init__()
        self.workload    = workload.astype(np.float32)
        self.predictions = predictions.astype(np.float32)
        self.avg_dur     = avg_duration_ms
        self.max_load    = max_load if max_load is not None else max(float(workload.max()), 1.0)
        self.T           = len(workload)

        # 7-dimensional observation space
        self.observation_space = spaces.Box(0.0, 1.0, shape=(7,), dtype=np.float32)
        self.action_space      = spaces.Discrete(MAX_SLOTS - MIN_SLOTS + 1)

        self._t          = 0
        self._queue      = 0
        self._slots      = MIN_SLOTS
        self._prev_queue = 0      # for queue-trend feature
        self._load_buf   = []     # recent loads for rolling average

    # ── Gym interface ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t, self._queue, self._slots = 0, 0, MIN_SLOTS
        self._prev_queue = 0
        self._load_buf   = []
        return self._obs(), {}

    def step(self, action: int):
        self._slots      = int(action) + MIN_SLOTS
        arrivals         = int(round(float(self.workload[self._t])))
        self._prev_queue = self._queue
        self._queue      = max(0, self._queue + arrivals)
        self._load_buf.append(arrivals)
        if len(self._load_buf) > 10:
            self._load_buf.pop(0)

        processed    = min(self._queue, self._slots)
        self._queue -= processed
        idle_slots   = self._slots - processed

        # Reward: fraction of demand served − response penalty
        # Bounded to [-1, 1] for stable value function learning
        total_demand  = processed + self._queue        # = prev_queue + arrivals
        throughput    = processed / max(total_demand, 1)
        response_norm = min(
            (self.avg_dur + (self._queue / max(self._slots, 1)) * self.avg_dur)
            / 5000.0, 1.0
        )
        idle_norm = idle_slots / MAX_SLOTS
        raw_reward = throughput - ALPHA * response_norm - BETA * idle_norm
        reward     = float(np.clip(raw_reward, -1.0, 1.0))  # bounded for stable critic

        self._t += 1
        done     = self._t >= self.T
        return self._obs(), reward, done, False, {}

    def _obs(self) -> np.ndarray:
        if self._t >= self.T:
            return np.zeros(7, dtype=np.float32)

        cur        = self.workload[self._t] / self.max_load
        pred       = (self.predictions[self._t, 0] / self.max_load
                      if self._t < len(self.predictions) else 0.0)
        q_norm     = min(self._queue / 100.0, 1.0)
        s_norm     = self._slots / MAX_SLOTS
        # Queue trend: positive = growing, negative = shrinking
        q_trend    = np.clip((self._queue - self._prev_queue) / 20.0 + 0.5, 0.0, 1.0)
        # Time-of-day: 1440 minutes in a day → periodic signal
        tod        = (self._t % 1440) / 1440.0
        # Rolling average load over last 10 minutes
        avg_load   = (np.mean(self._load_buf) / self.max_load
                      if self._load_buf else cur)
        avg_load   = float(np.clip(avg_load, 0.0, 1.0))

        return np.array([cur, pred, q_norm, s_norm, q_trend, tod, avg_load],
                        dtype=np.float32)
