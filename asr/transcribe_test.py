# asr/transcribe_test.py
# Phase 4, step 3: transcribe asr/clip.wav in isolation. Forced English, offline model load.

import os
# Load Whisper from local cache only — never hit the network on load.
# This is what took model-load from 169s to 0.4s: the network round-trip was the tax.
# Works identically on Windows and Mac; no shell setup needed.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"   # silence the cosmetic Windows warning

import time
from pathlib import Path
from faster_whisper import WhisperModel

CLIP     = Path("asr/clip.wav")
MODEL    = "base"          # accuracy-per-ms sweet spot for short English commands
LANGUAGE = "en"            # skip auto-detect (saves latency; commands are English)

def main():
    if not CLIP.exists():
        raise SystemExit(f"{CLIP} not found — run record_test.py first.")

    # int8 on CPU: fast load, low RAM; fine for a 74M-param 'base' model.
    print(f"Loading faster-whisper '{MODEL}' (first run downloads the model)...")
    t0 = time.perf_counter()
    model = WhisperModel(MODEL, device="cpu", compute_type="int8")
    t_load = time.perf_counter() - t0
    print(f"Model loaded in {t_load:.2f}s")

    t1 = time.perf_counter()
    segments, info = model.transcribe(str(CLIP), language=LANGUAGE)
    text = " ".join(seg.text for seg in segments).strip()
    t_asr = time.perf_counter() - t1

    print(f"\n--- TRANSCRIPT ---\n{text!r}")
    print(f"--- ASR time: {t_asr:.2f}s | audio {info.duration:.2f}s ---")

if __name__ == "__main__":
    main()