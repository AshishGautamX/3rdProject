"""Custom Gymnasium environment for the serverless scheduling problem.

State  : [norm_current_load, norm_predicted_load, norm_queue_depth, norm_active_slots]
Action : int — concurrency slots to allocate (maps to MIN_SLOTS … MAX_SLOTS)
Reward : throughput − α*response_time_norm − β*idle_slots_norm
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
        workload    : 1-D array (T,) — actual invocations per minute
        predictions : 2-D array (T, HORIZON) — LSTM look-ahead predictions
        max_load    : normalisation constant; pass training max to keep
                      observation scale consistent across train/test envs.
        """
        super().__init__()
        self.workload      = workload.astype(np.float32)
        self.predictions   = predictions.astype(np.float32)
        self.avg_dur       = avg_duration_ms
        self.max_load      = max_load if max_load is not None else max(float(workload.max()), 1.0)

        self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
        self.action_space      = spaces.Discrete(MAX_SLOTS - MIN_SLOTS + 1)

        self._t     = 0
        self._queue = 0
        self._slots = MIN_SLOTS

    # ── Gym interface ──────────────────────────────────────────────────────────
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._t, self._queue, self._slots = 0, 0, MIN_SLOTS
        return self._obs(), {}

    def step(self, action: int):
        self._slots  = int(action) + MIN_SLOTS
        arrivals     = int(round(float(self.workload[self._t])))
        self._queue  = max(0, self._queue + arrivals)

        processed    = min(self._queue, self._slots)
        self._queue -= processed
        idle_slots   = self._slots - processed

        # Fraction of demand served (not slot efficiency — avoids under-allocation bias)
        total_demand  = processed + self._queue   # = old_queue + arrivals
        throughput    = processed / max(total_demand, 1)
        response_norm = min((self.avg_dur + (self._queue / max(self._slots, 1))
                             * self.avg_dur) / 5000.0, 1.0)
        idle_norm     = idle_slots / MAX_SLOTS
        reward        = throughput - ALPHA * response_norm - BETA * idle_norm

        self._t   += 1
        done       = self._t >= len(self.workload)
        return self._obs(), float(reward), done, False, {}

    def _obs(self) -> np.ndarray:
        if self._t >= len(self.workload):
            return np.zeros(4, dtype=np.float32)
        cur  = self.workload[self._t] / self.max_load
        pred = (self.predictions[self._t, 0] / self.max_load
                if self._t < len(self.predictions) else 0.0)
        return np.array([cur, pred,
                         min(self._queue / 100.0, 1.0),
                         self._slots / MAX_SLOTS], dtype=np.float32)
