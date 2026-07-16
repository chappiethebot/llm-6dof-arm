"""sim/llm_demo.py — ROUGH DEMO ONLY. Type a command, watch the arm move.

Chain: command -> LLM (cloud, zero-shot) -> validated verb -> Phase 2 backend -> arm moves.

NOT part of the measurement. Proves nothing about the thesis — accuracy/latency/
generalization are all decided at the validated-object stage, before the arm runs.
This is Demo 2 machinery: motivation + a deliverable, not a result.

Known sim limits (from Phase 2, not bugs): one physical block, so all targets map
to it; inspect pauses (no wrist DOF); move_to/stop/reject do little visible.

Run from the sim/ folder:   cd sim ; python llm_demo.py
(Its imports match watch_full.py, which imports full_grasp_backend directly.)
"""
import sys, os

# Make the repo root importable so we can reuse the Phase 3 pipeline unchanged.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_grasp_backend import FullGraspBackend            # same import style as watch_full.py
from llm.clients import OpenAIClient
from llm.prompts import load_schema
from llm.pipeline import run_once

CLIENT = OpenAIClient("gpt-4o-mini")
STRATEGY = "zero_shot"


def drive(backend, obj):
    """Call the backend method that matches the validated verb object."""
    act = obj.get("action")
    if act == "pick_place":
        return backend.pick_place(obj["target"], obj["location"])
    if act == "fetch":
        return backend.fetch(obj["target"])
    if act == "hold":
        return backend.hold(obj["target"])
    if act == "inspect":
        return backend.inspect(obj["target"])
    if act == "move_to":
        return backend.move_to(obj["location"])
    if act == "stop":
        return backend.stop()
    if act == "reject":
        return backend.reject()
    return {"verb": act, "ok": False, "note": "no backend method"}


def main():
    schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema", "schema.json")
    schema = load_schema(schema_path)
    backend = FullGraspBackend(render=True, gain=3.0)      # reuse watch_full.py's proven viewer settings
    backend.set_object([1.30, 0.72, 0.425])                # place the one block where watch_full.py does
    print(f"Model: {CLIENT.name} | Strategy: {STRATEGY}")
    print("Type a command, watch the arm. 'quit' to exit.\n")
    try:
        while True:
            command = input("command> ").strip()
            if command.lower() in {"quit", "exit", ""}:
                break
            r = run_once(command, CLIENT, STRATEGY, schema)
            print(f"  understood as: {r.obj}  ({'valid' if r.ok else 'FAILED at ' + r.stage})")
            if r.ok:
                result = drive(backend, r.obj)
                print(f"  arm result:    {result}\n")
            else:
                print(f"  arm did not move (reasons={r.reasons})\n")
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        backend.close()
        print("\nclosed.")


if __name__ == "__main__":
    main()