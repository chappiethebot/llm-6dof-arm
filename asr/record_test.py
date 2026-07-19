# asr/record_test.py
# Phase 4, step 2 (push-to-talk): record until Enter. Cross-platform (Win now, Mac later).
# Selects mic BY NAME with default fallback; prints PEAK AMPLITUDE (proof of capture);
# stamps t_mic_stop at the Enter press (the moment the latency clock will start in step 4).

import sounddevice as sd
import numpy as np
import wave
import time
from pathlib import Path

MIC_NAME_HINT = "Seiren"   # substring; matches on Win + Mac. None = system default.
CHANNELS      = 1
PREFERRED_SR  = 16000
OUT_PATH      = Path("asr/clip.wav")


def resolve_input_device():
    if MIC_NAME_HINT:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and MIC_NAME_HINT.lower() in dev["name"].lower():
                print(f"Matched '{MIC_NAME_HINT}' -> device {idx}: {dev['name']}")
                return idx
        print(f"'{MIC_NAME_HINT}' not found; using system default input.")
    d = sd.default.device[0]
    print(f"Using default input device {d}: {sd.query_devices(d)['name']}")
    return d


def pick_samplerate(device):
    for sr in (PREFERRED_SR, 48000, 44100):
        try:
            sd.check_input_settings(device=device, samplerate=sr,
                                    channels=CHANNELS, dtype="float32")
            return sr
        except Exception:
            continue
    raise RuntimeError("No supported sample rate found for this input device.")


def main():
    device = resolve_input_device()
    sr = pick_samplerate(device)
    if sr != PREFERRED_SR:
        print(f"NOTE: recording at {sr} Hz; will resample to 16 kHz for Whisper.")

    frames = []  # collected audio chunks
    def callback(indata, frames_count, time_info, status):
        frames.append(indata.copy())

    input("Press Enter to START recording...")
    stream = sd.InputStream(samplerate=sr, channels=CHANNELS, dtype="float32",
                            device=device, callback=callback)
    with stream:
        input("Recording... speak, then press Enter to STOP.")
        t_mic_stop = time.perf_counter()   # <-- latency clock starts HERE in step 4
    print("Recording stopped.")

    audio = np.concatenate(frames, axis=0) if frames else np.zeros((1, CHANNELS), dtype="float32")
    dur = len(audio) / sr
    peak = float(np.max(np.abs(audio)))
    print(f"Captured {dur:.2f}s of audio.")
    print(f"PEAK AMPLITUDE: {peak:.4f}   (>~0.01 = real audio; ~0 = silence)")
    if peak < 0.01:
        print("  -> Near-silence. Suspect device (Win) or mic permission (Mac), NOT Whisper.")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(OUT_PATH), "wb") as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())
    print(f"Saved WAV -> {OUT_PATH.resolve()}  (sr={sr}, t_mic_stop stamped)")


if __name__ == "__main__":
    main()