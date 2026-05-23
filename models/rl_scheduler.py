"""PPO-based RL scheduler using Stable-Baselines3."""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from config.settings import PPO_TIMESTEPS
from environment.serverless_env import ServerlessEnv


def train_rl(workload_train: "np.ndarray", predictions_train: "np.ndarray",
             weights_path: str = None):
    """Train PPO on the training workload. Returns trained model."""
    env   = Monitor(ServerlessEnv(workload_train, predictions_train))
    model = PPO("MlpPolicy", env, verbose=1)
    model.learn(total_timesteps=PPO_TIMESTEPS)
    if weights_path:
        os.makedirs(os.path.dirname(weights_path), exist_ok=True)
        model.save(weights_path)
        print(f"[rl] Policy saved → {weights_path}")
    return model


def evaluate_rl(model, workload_test: "np.ndarray",
                predictions_test: "np.ndarray") -> list:
    """Run trained policy on test workload. Returns list of step records."""
    env    = ServerlessEnv(workload_test, predictions_test)
    obs, _ = env.reset()
    records, done = [], False

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(int(action))
        done = terminated or truncated
        t    = env._t - 1
        slots = int(action) + 1
        records.append({
            "minute":      t,
            "slots":       slots,
            "arrivals":    int(round(float(workload_test[t]))) if t < len(workload_test) else 0,
            "reward":      reward,
            "response_ms": env.avg_dur * (1 + env._queue / max(slots, 1)),
            "processed":   min(slots, env._queue + slots),
            "queued":      env._queue,
        })
    return records


def load_rl(weights_path: str, workload: "np.ndarray", predictions: "np.ndarray"):
    """Load a previously saved PPO policy."""
    env   = ServerlessEnv(workload, predictions)
    model = PPO.load(weights_path, env=env)
    print(f"[rl] Policy loaded ← {weights_path}")
    return model
