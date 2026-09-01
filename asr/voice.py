"""asr/voice.py -- the one voice front-end. Mic -> audio -> Whisper -> text.

Shared by sim/run_demo.py (MuJoCo demo) and bench/llm_client.py (bridge demo).
Both had their own copy: two mic resolvers and two Whisper blocks that had
already drifted apart -- run_demo had push-to-talk and no offline-cache retry,
llm_client had a fixed 5-second window and the retry. This is the union, so
neither loses a behaviour:

  * mic selected BY NAME SUBSTRING, never by device index. Indices shift when
    USB devices are plugged in; the name does not.
  * push-to-talk by default (`seconds=None`); pass `seconds=N` for a fixed
    window. Push-to-talk is better for a demo -- the speaker decides when the
    sentence is over -- but the fixed window is kept because a scripted run has
    nobody to press Enter.
  * HF_HUB_OFFLINE is forced on, then relaxed for exactly one retry if the
    model is not cached yet. Offline is right once cached (it was a 169s -> 0.4s
    latency fix) but it blocks the first download.
  * the ASR latency clock starts when the SPEAKER stops, not when transcription
    starts, so it includes everything the user waits through.

Whisper models are cached per name for the process, so `listen()` may be called
in a loop without reloading 140 MB each turn.

NOT a measurement path. Whisper model, mic and sample rate are all demo config.
"""
import os
import time
import wave

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

MIC_NAME_HINT = "Seiren"
CHANNELS = 1
PREFERRED_SR = 16000
LANGUAGE = "en"
MIN_PEAK = 0.01          # below this the clip is silence, not a command

_WHISPER = {}            # model name -> WhisperModel


def resolve_input_device(name_hint=MIC_NAME_HINT):
    """(index, name) for the first input device matching `name_hint`.

    Falls back to the system default, so an unplugged mic degrades to "some
    other mic" rather than an exception.
    """
    import sounddevice as sd
    if name_hint:
        for idx, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0 and name_hint.lower() in dev["name"].lower():
                return idx, dev["name"]
    default = sd.default.device[0]
    return default, sd.query_devices(default)["name"]


def pick_samplerate(device, channels=CHANNELS):
    """First rate the device actually accepts. 16 kHz is what Whisper wants;
    the fallbacks exist because some Windows drivers refuse it outright."""
    import sounddevice as sd
    for sr in (PREFERRED_SR, 48000, 44100):
        try:
            sd.check_input_settings(device=device, samplerate=sr,
                                    channels=channels, dtype="float32")
            return sr
        except Exception:
            continue
    raise RuntimeError("No supported sample rate for this input device.")


def load_whisper(model="base"):
    """Cached WhisperModel. Offline first; one online retry if not yet cached."""
    if model in _WHISPER:
        return _WHISPER[model]
    from faster_whisper import WhisperModel
    try:
        m = WhisperModel(model, device="cpu", compute_type="int8")
    except Exception:
        print("  model not cached -- downloading once")
        os.environ["HF_HUB_OFFLINE"] = "0"
        try:
            m = WhisperModel(model, device="cpu", compute_type="int8")
        finally:
            os.environ["HF_HUB_OFFLINE"] = "1"
    _WHISPER[model] = m
    return m


def record_push_to_talk(device, sr, channels=CHANNELS):
    """Record until Enter. Returns (audio, t_stop) with t_stop stamped AT Enter."""
    import sounddevice as sd
    frames = []

    def cb(indata, n, t, status):
        frames.append(indata.copy())

    input("  Press Enter to START speaking...")
    with sd.InputStream(samplerate=sr, channels=channels, dtype="float32",
                        device=device, callback=cb):
        input("  Speaking... press Enter to STOP.")
        t_stop = time.perf_counter()
    audio = (np.concatenate(frames, axis=0) if frames
             else np.zeros((1, channels), "float32"))
    return audio, t_stop


def record_fixed(device, sr, seconds, channels=CHANNELS):
    """Record a fixed window. Returns (audio, t_stop)."""
    import sounddevice as sd
    print("  recording %.0fs -- speak now..." % seconds)
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=channels,
                   dtype="float32", device=device)
    sd.wait()
    return audio, time.perf_counter()


def save_wav(audio, sr, path, channels=CHANNELS):
    """Write the clip for post-hoc listening. Debugging aid, not required."""
    pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(pcm16.tobytes())


def listen(model="base", mic=MIC_NAME_HINT, seconds=None, save_to=None,
           language=LANGUAGE, quiet=False):
    """Record and transcribe once.

    seconds=None -> push-to-talk (Enter to start, Enter to stop)
    seconds=N    -> fixed N-second window

    Returns (text, asr_s). text is None when the clip was near-silence, so a
    caller can `continue` instead of sending an empty command to a model.
    asr_s is measured from the moment the speaker stopped.
    """
    asr = load_whisper(model)
    device, dev_name = resolve_input_device(mic)
    sr = pick_samplerate(device)
    if not quiet:
        print(f"  mic: {dev_name}")

    if seconds is None:
        audio, t_stop = record_push_to_talk(device, sr)
    else:
        audio, t_stop = record_fixed(device, sr, seconds)

    peak = float(np.max(np.abs(audio)))
    dur = len(audio) / sr
    if peak < MIN_PEAK:
        print(f"  near-silence (peak {peak:.4f}) -- skipped.")
        return None, time.perf_counter() - t_stop

    if save_to is not None:
        save_wav(audio, sr, save_to)

    source = str(save_to) if save_to is not None else audio.flatten()
    segments, info = asr.transcribe(source, language=language)
    text = " ".join(s.text for s in segments).strip()
    asr_s = time.perf_counter() - t_stop
    if not quiet:
        print(f"  heard: {text!r}  (spoke {dur:.1f}s, lang={info.language})")
    return (text or None), asr_s


if __name__ == "__main__":
    # Self-check: no mic needed. Proves the wiring is intact -- the module
    # imports, the Whisper cache works, and a synthetic silent clip takes the
    # near-silence branch instead of reaching the model.
    import sys
    silence = np.zeros((PREFERRED_SR, CHANNELS), "float32")
    assert float(np.max(np.abs(silence))) < MIN_PEAK
    m1 = load_whisper("base")
    m2 = load_whisper("base")
    assert m1 is m2, "whisper model was reloaded instead of cached"
    print("ok: model cached, silence gate would trigger")
    try:
        idx, name = resolve_input_device()
        print(f"ok: mic resolves to [{idx}] {name}")
    except Exception as exc:
        print(f"(no audio device here: {type(exc).__name__}: {exc})")
    sys.exit(0)
