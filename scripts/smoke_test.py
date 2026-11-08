#!/usr/bin/env python3
"""
smoke_test.py - proves the thesis environment before anything depends on it.
Run on BOTH machines (Mac dev / Windows benchmark) inside the `thesis` conda env.
Each line prints ok/FAIL independently, so a partial env is diagnosable.
Imports + cheap constructions only. Does NOT download any model weights.
"""

def check(name, fn):
    try:
        fn()
        print(f"[ ok ] {name}")
        return True
    except Exception as e:
        print(f"[FAIL] {name}: {type(e).__name__}: {e}")
        return False


results = []

# --- Sim body (Phase 2 / 2.5) ---
def _mujoco():
    import mujoco  # noqa: F401
results.append(check("mujoco", _mujoco))

def _gym_fetch():
    import gymnasium as gym
    import gymnasium_robotics
    try:
        gym.register_envs(gymnasium_robotics)  # newer API
    except Exception:
        pass  # older gymnasium-robotics auto-registers on import
    for env_id in ("FetchPickAndPlace-v4", "FetchPickAndPlace-v3"):
        try:
            env = gym.make(env_id)
            env.close()
            print(f"       -> resolved Fetch env id: {env_id}")
            return
        except Exception:
            continue
    raise RuntimeError("neither FetchPickAndPlace-v4 nor v3 is registered")
results.append(check("gymnasium + gymnasium-robotics (Fetch)", _gym_fetch))

# --- RL backend (Phase 2.5) ---
def _sb3():
    from stable_baselines3 import SAC, HerReplayBuffer  # noqa: F401
results.append(check("stable-baselines3 (SAC + HER)", _sb3))

def _sb3_contrib():
    from sb3_contrib import TQC  # noqa: F401  (zoo's tuned Fetch config)
results.append(check("sb3-contrib (TQC)", _sb3_contrib))

# --- Vision (Phase 6) ---
def _ultra():
    from ultralytics import YOLO  # noqa: F401  (import only, no weight download)
results.append(check("ultralytics (YOLO)", _ultra))

def _cv2():
    import cv2  # noqa: F401
results.append(check("opencv-python", _cv2))

# --- ASR (Phase 4) ---
def _whisper():
    from faster_whisper import WhisperModel  # noqa: F401 (no model download)
results.append(check("faster-whisper", _whisper))

# --- LLM arms (Phase 3) ---
def _openai():
    import openai  # noqa: F401
results.append(check("openai (cloud arm)", _openai))

def _ollama():
    import ollama  # noqa: F401
results.append(check("ollama (local arm)", _ollama))

def _numpy():
    import numpy  # noqa: F401
results.append(check("numpy", _numpy))

print()
passed = sum(results)
print(f"{passed}/{len(results)} components OK")
if passed != len(results):
    raise SystemExit(1)
print("Environment ready.")
