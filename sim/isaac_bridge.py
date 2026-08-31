"""
sim/isaac_bridge.py

Sim-side half of the LLM -> Isaac bridge. Opens the Isaac window, then watches
a directory for JSON action files, executes each through IsaacBackend, and
writes a result file back.

Runs in conda env `isaac6`. The LLM side runs separately in `thesis`; the two
never share a process, so a model crash cannot kill the simulator.

Protocol -- bridge/inbox/<id>.json:
    {"verb": "pick_place", "target": "rubiks_cube", "location": "box_1"}
Result  -- bridge/outbox/<id>.json:
    {"verb": "pick_place", ..., "ok": true, "steps": 917, "error_cm": 0.4}

Run:  python sim/isaac_bridge.py
      python sim/isaac_bridge.py --headless
"""

import argparse
import json
import os
import time
import traceback
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from isaac_backend import launch, IsaacBackend


_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(_REPO_ROOT, "bridge")
INBOX = os.path.join(BRIDGE_DIR, "inbox")
OUTBOX = os.path.join(BRIDGE_DIR, "outbox")
DONE = os.path.join(BRIDGE_DIR, "done")

POLL_SECONDS = 0.25
IDLE_STEPS = 4          # keep the window alive and responsive while waiting


def _ensure_dirs():
    for d in (INBOX, OUTBOX, DONE):
        os.makedirs(d, exist_ok=True)


def _pending():
    try:
        names = sorted(
            n for n in os.listdir(INBOX) if n.lower().endswith(".json")
        )
    except FileNotFoundError:
        return []
    return names


def _read_action(path):
    """Read a command file, tolerating a partially written file."""
    for _ in range(5):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05)
    raise ValueError(f"could not read a complete JSON action from {path}")


def _write_result(name, payload):
    tmp = os.path.join(OUTBOX, name + ".tmp")
    final = os.path.join(OUTBOX, name)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, final)      # atomic: readers never see a half file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--headless", action="store_true",
        help="run without the editor window",
    )
    parser.add_argument(
        "--capture", default=None,
        help="directory to write render frames to",
    )
    args = parser.parse_args()

    _ensure_dirs()

    print(f"[bridge] launching Isaac (headless={args.headless})")
    app = launch(headless=args.headless)
    backend = IsaacBackend(capture_dir=args.capture)

    print(f"[bridge] watching {INBOX}")
    print("[bridge] drop a JSON action file there, or run the LLM client")
    print("[bridge] Ctrl+C to stop")

    try:
        while app.is_running():
            names = _pending()

            if not names:
                for _ in range(IDLE_STEPS):
                    backend.world.step(render=True)
                app.update()
                time.sleep(POLL_SECONDS)
                continue

            name = names[0]
            path = os.path.join(INBOX, name)

            try:
                action = _read_action(path)
                print(f"[bridge] -> {action}")
                result = backend.dispatch(action)
                print(f"[bridge] <- {result}")
            except KeyError as exc:
                result = {"ok": False, "error": "unknown_name_or_verb",
                          "detail": str(exc)}
                print(f"[bridge] !! {result}")
            except Exception as exc:                      # noqa: BLE001
                result = {"ok": False, "error": "execution_failed",
                          "detail": str(exc)}
                print(f"[bridge] !! {result}")
                traceback.print_exc()

            _write_result(name, result)
            os.replace(path, os.path.join(DONE, name))

    except KeyboardInterrupt:
        print("\n[bridge] stopping")
    finally:
        backend.close()


if __name__ == "__main__":
    main()