"""Phase 0 smoke test: every component imports and runs once. Prints ok per stage."""
import os

# --- MuJoCo + sim body ---
import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)          # required in recent versions
env = gym.make("FetchPickAndPlace-v4")         # v4 = the reset-bug-fixed version
obs, _ = env.reset(seed=0)
env.step(env.action_space.sample())
env.close()
print("ok: mujoco + gymnasium-robotics")

# --- Whisper (CPU for the smoke test everywhere) ---
from faster_whisper import WhisperModel
WhisperModel("tiny", device="cpu", compute_type="int8")   # loading proves the stack
print("ok: faster-whisper (cpu)")

# --- Local LLM via Ollama ---
import ollama
names = [m.model for m in ollama.list().models]
assert any("llama3.2" in n for n in names), f"pull llama3.2:3b — have {names}"
ollama.generate(model="llama3.2:3b", prompt="reply with: ok", options={"num_predict": 3})
print("ok: ollama local llm")

# --- Cloud LLM (guarded so a missing key doesn't crash the test) ---
if os.getenv("OPENAI_API_KEY"):
    from openai import OpenAI
    OpenAI().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "reply with: ok"}],
        max_tokens=3,
    )
    print("ok: cloud llm")
else:
    print("skip: cloud llm (set OPENAI_API_KEY)")

print("\nPhase 0 complete.")