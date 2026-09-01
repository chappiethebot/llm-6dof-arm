"""
sim/mujoco_bridge.py

MuJoCo-side half of the demo bridge. Mirrors sim/isaac_bridge.py: watches a
directory for JSON action files, executes each through SimBackend, writes a
result back.

Runs in conda env `thesis` on either machine, including the Mac. Isaac is
never imported, so no GPU is required.

NOTE: this is the DEMO path. The measured rows came from the frozen harness at
commit 3000198. Never regenerate results through this script.

Run:  python sim/mujoco_bridge.py
      python sim/mujoco_bridge.py --render
"""

import argparse
import json
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sim_backend import SimBackend

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BRIDGE_DIR = os.path.join(_REPO_ROOT, "bridge")
INBOX = os.path.join(BRIDGE_DIR, "inbox")
OUTBOX = os.path.join(BRIDGE_DIR, "outbox")
DONE = os.path.join(BRIDGE_DIR, "done")

POLL_SECONDS = 0.25


def _ensure_dirs():
    for d in (INBOX, OUTBOX, DONE):
        os.makedirs(d, exist_ok=True)


def _pending():
    try:
        return sorted(n for n in os.listdir(INBOX)
                      if n.lower().endswith(".json"))
    except FileNotFoundError:
        return []


def _read_action(path):
    """Read a command file, tolerating a partially written file."""
    for _ in range(5):
        try:
            with open(path, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except (json.JSONDecodeError, PermissionError):
            time.sleep(0.05)
    raise ValueError("could not read a complete JSON action from %s" % path)


def _write_result(name, payload):
    tmp = os.path.join(OUTBOX, name + ".tmp")
    final = os.path.join(OUTBOX, name)
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, final)      # atomic: readers never see a half file


def dispatch(backend, action):
    """Execute a validated schema action dict on SimBackend."""
    verb = action.get("verb")
    if verb == "move_to":
        return backend.move_to(action["location"])
    if verb == "pick_place":
        return backend.pick_place(action["target"], action["location"])
    if verb == "fetch":
        return backend.fetch(action["target"])
    if verb == "hold":
        return backend.hold(action["target"])
    if verb == "inspect":
        return backend.inspect(action["target"])
    if verb == "stop":
        return backend.stop()
    if verb == "reject":
        return backend.reject()
    raise KeyError("unknown verb: %r" % (verb,))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true",
                        help="open the MuJoCo viewer window")
    args = parser.parse_args()

    _ensure_dirs()

    print("[mujoco] starting SimBackend (render=%s)" % args.render)
    backend = SimBackend(render=args.render)

    print("[mujoco] watching   %s" % INBOX)
    print("[mujoco] Ctrl+C to stop")

    try:
        while True:
            names = _pending()
            if not names:
                time.sleep(POLL_SECONDS)
                continue

            name = names[0]
            path = os.path.join(INBOX, name)

            try:
                action = _read_action(path)
                print("[mujoco] -> %s" % json.dumps(action))
                result = dispatch(backend, action)
                print("[mujoco] <- %s" % json.dumps(result))
            except KeyError as exc:
                result = {"ok": False, "error": "unknown_name_or_verb",
                          "detail": str(exc)}
                print("[mujoco] !! %s" % json.dumps(result))
            except Exception as exc:                          # noqa: BLE001
                result = {"ok": False, "error": "execution_failed",
                          "detail": str(exc)}
                print("[mujoco] !! %s" % json.dumps(result))
                traceback.print_exc()

            _write_result(name, result)
            os.replace(path, os.path.join(DONE, name))

    except KeyboardInterrupt:
        print("\n[mujoco] stopping")
    finally:
        backend.close()


if __name__ == "__main__":
    main()