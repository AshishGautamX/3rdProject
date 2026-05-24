"""Ablation study — quantifies each component's contribution.

Arms:
  1. FCFS (10 slots)          — static baseline
  2. Reactive autoscaler      — rule-based dynamic allocation
  3. LSTM + threshold         — predictive, no RL
  4. PPO only (no LSTM)       — RL without prediction features
  5. LSTM + PPO               — full system (reuses Section 8 metrics)
"""
import os
import numpy as np
from typing import Dict

from models.baseline_scheduler import simulate_baseline, simulate_reactive
from evaluation.metrics import summarise
from config.settings import (
    AVG_DURATION_MS, BASELINE_SLOTS, MAX_SLOTS, MIN_SLOTS,
    PPO_TIMESTEPS, PPO_ENT_COEF, PPO_N_STEPS, PPO_BATCH_SIZE,
    PPO_N_EPOCHS, PPO_CLIP_RANGE, PPO_MAX_GRAD_NORM, PPO_VF_COEF, ALPHA,
)


# ─── LSTM-threshold scheduler ─────────────────────────────────────────────────

def simulate_lstm_threshold(invocations: np.ndarray,
                             lstm_preds_scaled: np.ndarray,
                             avg_duration_ms: float = AVG_DURATION_MS,
                             safety_factor: float = 1.3) -> list:
    """Set slots = round(LSTM_prediction × safety_factor). No RL."""
    from collections import deque
    records   = []
    arrival_q = deque()
    slot_free = [0] * MAX_SLOTS

    for minute, arrivals in enumerate(invocations):
        arrivals = max(0, int(round(float(arrivals))))
        t_ms     = minute * 60_000
        pred     = float(lstm_preds_scaled[minute]) if minute < len(lstm_preds_scaled) else arrivals
        slots    = int(np.clip(round(pred * safety_factor), MIN_SLOTS, MAX_SLOTS))

        for _ in range(arrivals):
            arrival_q.append(t_ms)

        slot_order = sorted(range(slots), key=lambda s: slot_free[s])
        for slot in slot_order:
            if not arrival_q:
                break
            job_arr = arrival_q.popleft()
            start   = max(t_ms, slot_free[slot])
            end     = start + avg_duration_ms
            slot_free[slot] = end
            records.append({
                "minute":      minute,
                "slots":       slots,
                "arrival_ms":  job_arr,
                "start_ms":    start,
                "end_ms":      end,
                "wait_ms":     start - job_arr,
                "response_ms": end   - job_arr,
                "queued":      len(arrival_q),
                "processed":   1,
            })
    return records


# ─── PPO-only env (no LSTM features, 4-dim state) ─────────────────────────────

def _make_simple_env_class():
    """Returns a 4-dim Gym env class (no LSTM prediction in state)."""
    import gymnasium as gym
    from gymnasium import spaces

    class SimplePPOEnv(gym.Env):
        metadata = {"render_modes": []}

        def __init__(self, workload, avg_dur, max_load):
            super().__init__()
            self.workload = workload.astype(np.float32)
            self.avg_dur  = avg_dur
            self.max_load = max_load
            self.T        = len(workload)
            self.observation_space = spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32)
            self.action_space      = spaces.Discrete(MAX_SLOTS - MIN_SLOTS + 1)
            self._t = self._queue = 0
            self._slots = MIN_SLOTS

        def reset(self, *, seed=None, options=None):
            super().reset(seed=seed)
            self._t = self._queue = 0
            self._slots = MIN_SLOTS
            return self._obs(), {}

        def step(self, action):
            self._slots  = int(action) + MIN_SLOTS
            arrivals     = int(round(float(self.workload[self._t])))
            self._queue  = max(0, self._queue + arrivals)
            processed    = min(self._queue, self._slots)
            self._queue -= processed
            total_demand = processed + self._queue
            throughput   = processed / max(total_demand, 1)
            resp_norm    = min((self.avg_dur + (self._queue / max(self._slots, 1))
                                * self.avg_dur) / 5000.0, 1.0)
            reward = float(np.clip(throughput - ALPHA * resp_norm, -1.0, 1.0))
            self._t += 1
            return self._obs(), reward, self._t >= self.T, False, {}

        def _obs(self):
            if self._t >= self.T:
                return np.zeros(4, dtype=np.float32)
            return np.array([
                self.workload[self._t] / self.max_load,
                min(self._queue / 100.0, 1.0),
                self._slots / MAX_SLOTS,
                (self._t % 1440) / 1440.0,
            ], dtype=np.float32)

    return SimplePPOEnv


def train_ppo_no_lstm(workload_train: np.ndarray,
                      weights_path: str = None):
    """Train PPO with 4-dim state (no LSTM). Returns (model, max_load)."""
    from stable_baselines3 import PPO
    from stable_baselines3.common.monitor import Monitor

    SimplePPOEnv    = _make_simple_env_class()
    global_max_load = max(float(workload_train.max()), 1.0)
    env = Monitor(SimplePPOEnv(workload_train, AVG_DURATION_MS, global_max_load))
    model = PPO(
        "MlpPolicy", env, verbose=0, device="cpu",
        ent_coef=PPO_ENT_COEF, n_steps=PPO_N_STEPS,
        batch_size=PPO_BATCH_SIZE, n_epochs=PPO_N_EPOCHS,
        clip_range=PPO_CLIP_RANGE, max_grad_norm=PPO_MAX_GRAD_NORM,
        vf_coef=PPO_VF_COEF, normalize_advantage=True,
    )
    model.learn(total_timesteps=PPO_TIMESTEPS)
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        model.save(weights_path)
    return model, global_max_load


def evaluate_ppo_no_lstm(model, workload_test: np.ndarray,
                          global_max_load: float) -> list:
    """Evaluate PPO-only model (4-dim state)."""
    SimplePPOEnv = _make_simple_env_class()
    env    = SimplePPOEnv(workload_test, AVG_DURATION_MS, global_max_load)
    obs, _ = env.reset()
    records, done = [], False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        slots     = int(action) + 1
        t_pre     = env._t
        q_pre     = env._queue
        arrivals  = int(round(float(workload_test[t_pre]))) if t_pre < len(workload_test) else 0

        obs, _, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

        processed   = min(q_pre + arrivals, slots)
        response_ms = (q_pre / max(slots, 1)) * AVG_DURATION_MS + AVG_DURATION_MS

        if processed > 0:
            records.append({
                "minute":      t_pre,
                "slots":       slots,
                "arrivals":    arrivals,
                "response_ms": response_ms,
                "processed":   processed,
                "queued":      env._queue,
            })
    return records


# ─── Main ablation runner ─────────────────────────────────────────────────────

def run_ablation(test_arr: np.ndarray,
                 lstm_preds_scaled: np.ndarray,
                 train_arr: np.ndarray,
                 ai_metrics: Dict,          # pre-computed from Section 8 — reused
                 weights_dir: str,
                 total_minutes: int) -> Dict[str, Dict]:
    """Run all ablation arms. Returns dict of {arm_name: metrics_dict}."""
    results = {}

    print("[ablation] 1/5 FCFS ...")
    rec = simulate_baseline(test_arr, max_slots=BASELINE_SLOTS,
                            avg_duration_ms=AVG_DURATION_MS, strategy="fcfs")
    results["FCFS"] = summarise(rec, total_minutes, BASELINE_SLOTS,
                                "FCFS", verbose=True)

    print("[ablation] 2/5 Reactive ...")
    rec = simulate_reactive(test_arr, avg_duration_ms=AVG_DURATION_MS)
    results["Reactive"] = summarise(rec, total_minutes, MAX_SLOTS,
                                    "Reactive", verbose=True)

    print("[ablation] 3/5 LSTM+Threshold ...")
    rec = simulate_lstm_threshold(test_arr, lstm_preds_scaled)
    results["LSTM-Threshold"] = summarise(rec, total_minutes, MAX_SLOTS,
                                          "LSTM-Threshold", verbose=True)

    print("[ablation] 4/5 PPO-only (training ~3 min) ...")
    ppo_path = os.path.join(weights_dir, "ppo_only_policy")
    if os.path.exists(ppo_path + ".zip"):
        from stable_baselines3 import PPO
        SimplePPOEnv = _make_simple_env_class()
        dummy = SimplePPOEnv(test_arr, AVG_DURATION_MS,
                             max(float(test_arr.max()), 1.0))
        ppo_only = PPO.load(ppo_path, env=dummy)
    else:
        ppo_only, _ = train_ppo_no_lstm(train_arr, weights_path=ppo_path)
        print(f"  [ablation] PPO-only saved → {ppo_path}")

    global_max_load = max(float(train_arr.max()), 1.0)
    rec = evaluate_ppo_no_lstm(ppo_only, test_arr, global_max_load)
    results["PPO-only"] = summarise(rec, total_minutes, MAX_SLOTS,
                                    "PPO-only", verbose=True)

    print("[ablation] 5/5 LSTM+PPO (reusing Section 8) ...")
    results["LSTM+PPO"] = ai_metrics   # no re-compute needed

    return results
