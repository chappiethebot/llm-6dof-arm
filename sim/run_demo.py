"""sim/run_demo.py — unified demo: pick keyboard or voice input at startup.

Menu picks the INPUT front-end; everything downstream (run_once -> drive -> arm)
is shared, so both modes run identical motion code. llm_demo.py is left untouched
as the stable typed reference; this is additive.

Model + strategy are flags (voice adds no menu for them):
  Windows:  python -m sim.run_demo --model gpt-4o-mini --strategy zero_shot
  Mac:      python -m sim.run_demo --model qwen2.5:3b --strategy cot

V2: run from the REPO ROOT with `python -m sim.run_demo`, NOT from inside sim/.
The backend now imports sim.workspace for the nine grid cells, and that import
only resolves when the repo root is on sys.path. Schema selection follows the
backend: if the grid loaded, the V2 schema is used, so spoken commands may use
the same phrasings as the V2 datasets ("the far left corner of the table").
V1 location names still work in both modes.

macOS note: run under plain `python`, NOT `mjpython`. mjpython runs user code on
a non-main thread, which makes gymnasium's render_mode="human" window creation
crash ('NSWindow should only be instantiated on the main thread'). Plain python
keeps the viewer on the main thread. (The backend is also built before Whisper as
a belt-and-braces main-thread measure; harmless on Windows.)

ASR AND THE GRID: Whisper transcribes speech to text, and the grid phrasings are
ordinary English words, so no ASR change is needed. But spoken "A1" or "grid A1"
will NOT work -- the schema has no such phrasing and the model was never shown
one. Say the words: "the far left corner of the table".
"""
import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

import sys, time, wave, argparse
from pathlib import Path
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
# Both are needed. _ROOT so `sim.workspace` resolves (that is how the backend
# finds the grid); _HERE so the bare `from full_grasp_backend import ...` below
# resolves. Adding both means the demo runs BOTH ways:
#     python -m sim.llm_demo      (from the repo root)
#     python llm_demo.py          (from inside sim/)
for _p in (_ROOT, _HERE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from full_grasp_backend import FullGraspBackend, HAS_GRID
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


# ---------- voice front-end (only used if mode 2 chosen) ----------
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
    """Record until Enter. t_mic_stop stamped at Enter = latency clock start."""
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
    ap.add_argument("--schema", default="auto", choices=["auto", "v1", "v2"],
                    help="auto = v2 when the grid loaded, else v1")
    args = ap.parse_args()

    mode = choose_mode()
    client = make_client(args.model)
    strategy = args.strategy
    examples = None   # few_shot examples: data/seen*.jsonl ONLY (seal rule). Wired later.

    # --- BUILD SIM VIEWER FIRST (belt-and-braces main-thread measure) ---
    # Primary macOS fix is running under plain `python` (see module docstring).
    # Building the backend before Whisper/sounddevice is an extra safeguard and
    # is order-independent on Windows.
    root = _ROOT
    if args.schema == "auto":
        name = "schema_v2.json" if HAS_GRID else "schema.json"
    else:
        name = "schema_v2.json" if args.schema == "v2" else "schema.json"
    if name.endswith("_v2.json") and not HAS_GRID:
        sys.exit("REFUSING: v2 schema requested but the backend has no grid cells "
                 "(sim/workspace.py did not import). Run from the repo root: "
                 "python -m sim.run_demo")
    schema = load_schema(os.path.join(root, "schema", name))
    print(f"schema: {name}  ({len(schema['locations'])} locations)")

    backend = FullGraspBackend(render=True, gain=3.0)   # opens viewer on main thread
    backend.set_object([1.30, 0.72, 0.425])

    # --- THEN set up voice (mode 2 only) ---
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

    print(f"\nModel: {client.name} | Strategy: {strategy} | Input: {mode.upper()}")
    print_legend()
    if mode == "voice" and HAS_GRID:
        print('  VOICE + GRID: say the WORDS, not the cell code. "the far left')
        print('  corner of the table" works; "grid A one" does not.\n')

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
                if result.get("resolved") and result["resolved"] != r.obj.get("location"):
                    print(f"  resolved to:   {result['resolved']}")
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
