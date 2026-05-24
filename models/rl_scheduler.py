"""PPO-based RL scheduler using Stable-Baselines3."""
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from config.settings import PPO_TIMESTEPS, AVG_DURATION_MS
from environment.serverless_env import ServerlessEnv


def train_rl(workload_train: np.ndarray, predictions_train: np.ndarray,
             weights_path: str = None):
    """Train PPO on the training workload. Returns trained model."""
    env   = Monitor(ServerlessEnv(workload_train, predictions_train,
                                  avg_duration_ms=AVG_DURATION_MS))
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")
    model.learn(total_timesteps=PPO_TIMESTEPS)
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        model.save(weights_path)
        print(f"[rl] Policy saved → {weights_path}")
    return model


def evaluate_rl(model, workload_test: np.ndarray,
                predictions_test: np.ndarray) -> list:
    """Run trained policy on test workload.
    Captures pre-step queue/arrivals for accurate response-time calculation.
    Returns list of per-step records.
    """
    env    = ServerlessEnv(workload_test, predictions_test,
                           avg_duration_ms=AVG_DURATION_MS)
    obs, _ = env.reset()
    records, done = [], False

    while not done:
        action, _  = model.predict(obs, deterministic=True)
        slots      = int(action) + 1

        # Capture pre-step state
        t_pre     = env._t
        queue_pre = env._queue
        arrivals  = (int(round(float(workload_test[t_pre])))
                     if t_pre < len(workload_test) else 0)

        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated

        total_demand = queue_pre + arrivals
        processed    = min(total_demand, slots)

        # Response = queue wait + execution time
        wait_ms      = (queue_pre / max(slots, 1)) * AVG_DURATION_MS
        response_ms  = wait_ms + AVG_DURATION_MS

        if processed > 0:
            records.append({
                "minute":      t_pre,
                "slots":       slots,
                "arrivals":    arrivals,
                "reward":      reward,
                "response_ms": response_ms,
                "processed":   processed,
                "queued":      env._queue,
            })

    return records


def load_rl(weights_path: str, workload: np.ndarray, predictions: np.ndarray):
    """Load a previously saved PPO policy."""
    env   = ServerlessEnv(workload, predictions, avg_duration_ms=AVG_DURATION_MS)
    model = PPO.load(weights_path, env=env)
    print(f"[rl] Policy loaded ← {weights_path}")
    return model
