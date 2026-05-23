"""Optional LLM hint layer (Groq API). Disabled by default via USE_LLM=False.

When enabled, the LLM is only queried on predicted workload spikes.
Falls back silently if the API key is missing or the call fails.
"""
import os
import numpy as np
from config.settings import USE_LLM, GROQ_MODEL, GROQ_API_KEY_ENV, SPIKE_THRESHOLD, MAX_SLOTS


def _is_spike(value: float, history: np.ndarray) -> bool:
    if len(history) < 5:
        return False
    std = history.std()
    return std > 0 and (value - history.mean()) / std > SPIKE_THRESHOLD


def get_llm_hint(current_load: float, predicted_load: float,
                 current_slots: int, history: np.ndarray) -> "int | None":
    """Return a recommended slot count from the LLM, or None if skipped."""
    if not USE_LLM:
        return None
    if not _is_spike(predicted_load, history):
        return None

    api_key = os.environ.get(GROQ_API_KEY_ENV, "")
    if not api_key:
        print("[llm] GROQ_API_KEY not set — skipping hint.")
        return None

    try:
        from groq import Groq
        client = Groq(api_key=api_key)
        prompt = (
            f"Serverless scheduler. Current load: {current_load:.0f} jobs/min. "
            f"Predicted load: {predicted_load:.0f} jobs/min. "
            f"Current slots: {current_slots}. Max slots: {MAX_SLOTS}. "
            f"Reply with a single integer slot count only."
        )
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=8, temperature=0.0,
        )
        raw  = resp.choices[0].message.content.strip()
        hint = int("".join(filter(str.isdigit, raw)))
        hint = max(1, min(hint, MAX_SLOTS))
        print(f"[llm] Spike → hint: {hint} slots")
        return hint
    except Exception as exc:
        print(f"[llm] Error: {exc} — using RL action.")
        return None
