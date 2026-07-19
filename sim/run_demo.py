"""sim/run_demo.py — unified demo: pick keyboard or voice input at startup.

Menu picks the INPUT front-end; everything downstream (run_once -> drive -> arm)
is shared, so both modes run identical motion code. llm_demo.py is left untouched
as the stable typed reference; this is additive.

Model + strategy are flags (voice adds no menu for them):
  python run_demo.py --model gpt-4o-mini --strategy zero_shot
  mjpython run_demo.py --model qwen2.5:3b --strategy cot   (Mac)
Run from sim/.
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys, time, wave, argparse
from pathlib import Path
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from full_grasp_backend import FullGraspBackend
from llm.prompts import load_schema
from llm.pipeline import run_once
from llm.clients import OpenAIClient, OllamaClient
from llm_demo import drive, describe, print_legend, draw_box_markers  # reuse, unchanged

# --- audio config (proven Phase 4 steps 1-3) ---
MIC_NAME_HINT = "Seiren"
CHANNELS      = 1
PREFERRED_SR  = 16000
WHISPER_MODEL = "base"
LANGUAGE      = "en"
CLIP_PATH     = Path(os.path.join(os.path.dirname(__file__), "voice_clip.wav"))


def make_client(name):
    return OpenAIClient(name) if name.startswith("gpt-") else OllamaClient(name)


# ---------- input mode selection ----------
def choose_mode():
    print("\nSelect input mode:")
    print("  1. Keyboard (type commands)")
    print("  2. Voice (speak commands)")
    while True:
        choice = input("Enter 1 or 2: ").strip()
        if choice == "1":
            return "keyboard"
        if choice == "2":
            return "voice"
        print("  Please type 1 or 2.")


# ---------- voice front-end (only built if mode 2 chosen) ----------
def resolve_input_device():
    if MIC_NAME_HINT:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and MIC_NAME_HINT.lower() in dev["name"].lower():
                return idx
    return sd.default.device[0]


def pick_samplerate(device):
    for sr in (PREFERRED_SR, 48000, 44100):
        try:
            sd.check_input_settings(device=device, samplerate=sr,
                                    channels=CHANNELS, dtype="float32")
            return sr
        except Exception:
            continue
    raise RuntimeError("No supported sample rate for this input device.")


def record_push_to_talk(device, sr):
    frames = []
    def cb(indata, n, t, status): frames.append(indata.copy())
    input("  Press Enter to START speaking...")
    with sd.InputStream(samplerate=sr, channels=CHANNELS, dtype="float32",
                        device=device, callback=cb):
        input("  Speaking... press Enter to STOP.")
        t_mic_stop = time.perf_counter()
    audio = np.concatenate(frames, axis=0) if frames else np.zeros((1, CHANNELS), "float32")
    return audio, t_mic_stop


def save_wav(audio, sr):
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(CLIP_PATH), "wb") as wf:
        wf.setnchannels(CHANNELS); wf.setsampwidth(2); wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def get_voice_command(asr, device, sr):
    """Returns (command_text, asr_ms) or (None, None) on near-silence."""
    audio, t_mic_stop = record_push_to_talk(device, sr)
    peak = float(np.max(np.abs(audio)))
    dur = len(audio) / sr
    if peak < 0.01:
        print(f"  near-silence (peak {peak:.4f}) — skipped.")
        return None, None
    save_wav(audio, sr)
    segments, info = asr.transcribe(str(CLIP_PATH), language=LANGUAGE)
    text = " ".join(s.text for s in segments).strip()
    asr_ms = (time.perf_counter() - t_mic_stop) * 1000
    print(f"  heard: {text!r}  (spoke {dur:.1f}s, lang={info.language})")
    return text, asr_ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--strategy", default="zero_shot")
    args = ap.parse_args()

    mode = choose_mode()
    client = make_client(args.model)
    strategy = args.strategy
    examples = None   # few_shot examples: data/seen.jsonl ONLY (seal rule). Wired later.

    # Voice-only setup: import + load Whisper only if mode 2 (keyboard needs neither).
    asr = device = sr = None
    if mode == "voice":
        global sd
        import sounddevice as sd
        from faster_whisper import WhisperModel
        device = resolve_input_device()
        sr = pick_samplerate(device)
        print("Loading Whisper (offline)...")
        t = time.perf_counter()
        asr = WhisperModel(WHISPER_MODEL, device="cpu", compute_type="int8")
        print(f"Whisper loaded in {time.perf_counter()-t:.2f}s")

    schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               "schema", "schema.json")
    schema = load_schema(schema_path)

    backend = FullGraspBackend(render=True, gain=3.0)
    backend.set_object([1.30, 0.72, 0.425])

    print(f"\nModel: {client.name} | Strategy: {strategy} | Input: {mode.upper()}")
    print_legend()

    markers_ok = None
    try:
        while True:
            # --- STAGE 1: get command (mode-specific) ---
            asr_ms = None
            if mode == "keyboard":
                command = input("command> ").strip()
                if command.lower() in {"quit", "exit", ""}:
                    break
            else:
                command, asr_ms = get_voice_command(asr, device, sr)
                if command is None:
                    continue

            # --- STAGE 2: LLM -> schema (raw text; identical both modes) ---
            t0 = time.perf_counter()
            r = run_once(command, client, strategy, schema, examples=examples)
            llm_ms = (time.perf_counter() - t0) * 1000
            print(f"  understood as: {r.obj}  ({'valid' if r.ok else 'FAILED at ' + r.stage})")

            # --- STAGE 3: motion (shared) ---
            if r.ok:
                describe(r.obj)
                result = drive(backend, r.obj)
                print(f"  arm result:    {result}")
            else:
                print(f"  arm did not move (reasons={r.reasons})")

            # --- latency (adapts to mode: no fake ASR figure in keyboard) ---
            if asr_ms is not None:
                print(f"  LATENCY: ASR {asr_ms:.0f} ms | LLM {llm_ms:.0f} ms "
                      f"| total-to-parse {asr_ms+llm_ms:.0f} ms")
            else:
                print(f"  LATENCY: LLM {llm_ms:.0f} ms")

            if markers_ok is None:
                markers_ok = draw_box_markers(backend)
                print(f"  [floor markers: {'ON' if markers_ok else 'unavailable'}]")
            elif markers_ok:
                draw_box_markers(backend)

            if mode == "voice":
                if input("\n  Another? [Enter=yes, q=quit]: ").strip().lower() == "q":
                    break
            print()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        backend.close()
        print("\nclosed.")


if __name__ == "__main__":
    main()