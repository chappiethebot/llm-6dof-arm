"""sim/llm_demo.py — ROUGH DEMO ONLY. Type a command, watch the arm move.

Chain: command -> LLM -> validated verb -> Phase 2 backend -> arm moves.
NOT part of the measurement. Demo 2 machinery only.

Works with ANY of the four models (cloud or local) — just change CLIENT below.
Local models occasionally return an invalid object (arm won't move that turn);
that's model unreliability, not a demo bug.

Floor markers (colored spheres at each box): best-effort. On Mac they need
mjpython AND are attempted only AFTER the first command (viewer must be fully
live on the main thread first). If they can't draw, the demo still runs fine on
the terminal legend alone.

Run from sim/:   mjpython llm_demo.py   (Mac, needed for markers)
                 python llm_demo.py     (Windows, or Mac without markers)
"""
import sys, os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from full_grasp_backend import FullGraspBackend, LOCATIONS, OBJECT_OF
from llm.prompts import load_schema
from llm.pipeline import run_once
from llm.clients import OpenAIClient, OllamaClient

# --- pick any model; all four work here ---
CLIENT = OpenAIClient("gpt-4o-mini")
# CLIENT = OllamaClient("llama3.2:3b")
# CLIENT = OllamaClient("qwen2.5:3b")
# CLIENT = OllamaClient("qwen2.5:0.5b")
STRATEGY = "zero_shot"
# ------------------------------------------

MARKER_COLORS = {
    "box_1":        (1.0, 0.2, 0.2, 1.0),   # red
    "box_2":        (0.2, 1.0, 0.2, 1.0),   # green
    "box_3":        (0.2, 0.4, 1.0, 1.0),   # blue
    "table_center": (1.0, 1.0, 0.2, 1.0),   # yellow
    "table_left":   (1.0, 0.5, 0.0, 1.0),   # orange
    "table_right":  (0.7, 0.2, 1.0, 1.0),   # purple
    "table_front":  (0.0, 1.0, 1.0, 1.0),   # cyan
    "table_back":   (1.0, 0.4, 0.7, 1.0),   # pink
}
COLOR_NAME = {"box_1":"red","box_2":"green","box_3":"blue","table_center":"yellow",
              "table_left":"orange","table_right":"purple","table_front":"cyan","table_back":"pink"}


def draw_box_markers(backend):
    """Draw a colored sphere at each box coordinate. Best-effort: returns False
    and skips cleanly if the viewer has no reachable user scene. Never crashes."""
    import mujoco
    try:
        viewer = backend.env.unwrapped.mujoco_renderer.viewer
        scn = viewer.user_scn
        scn.ngeom = 0
        for name, (x, y) in LOCATIONS.items():
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.03, 0, 0]),
                np.array([x, y, 0.42]),
                np.eye(3).flatten(),
                np.array(MARKER_COLORS.get(name, (1, 1, 1, 1)), dtype=np.float32),
            )
            scn.ngeom += 1
        return True
    except Exception:
        return False


def print_legend():
    print("\n" + "=" * 56)
    print("SCENE LEGEND  (boxes are coordinates; markers try after 1st command)")
    print("-" * 56)
    print("  Locations (x, y)  [marker color]:")
    for name, (x, y) in LOCATIONS.items():
        print(f"    {name:14} -> ({x:.2f}, {y:.2f})   [{COLOR_NAME.get(name,'?')}]")
    print("\n  ONE physical block. These names all point to it, so the arm")
    print("  grabs the same block whichever you say:")
    print(f"    {', '.join(OBJECT_OF.keys())}")
    print("=" * 56 + "\n")


def drive(backend, obj):
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


def describe(obj):
    tgt = obj.get("target")
    loc = obj.get("location")
    if tgt:
        print(f"  treating block as: {tgt}")
    if loc and loc in LOCATIONS:
        x, y = LOCATIONS[loc]
        print(f"  destination:       {loc}  ({x:.2f}, {y:.2f})  [{COLOR_NAME.get(loc,'?')}]")


def main():
    schema_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "schema", "schema.json")
    schema = load_schema(schema_path)
    backend = FullGraspBackend(render=True, gain=3.0)
    backend.set_object([1.30, 0.72, 0.425])

    print(f"Model: {CLIENT.name} | Strategy: {STRATEGY}")
    print_legend()
    print("Tip: press 'E' in the viewer to hide the coordinate-axis arrows.")
    print("(Floor markers are attempted after your first command.)\n")

    markers_ok = None  # unknown until we try, post-step, inside the loop
    try:
        while True:
            command = input("command> ").strip()
            if command.lower() in {"quit", "exit", ""}:
                break
            r = run_once(command, CLIENT, STRATEGY, schema)
            print(f"  understood as: {r.obj}  ({'valid' if r.ok else 'FAILED at ' + r.stage})")
            if r.ok:
                describe(r.obj)
                result = drive(backend, r.obj)
                print(f"  arm result:    {result}")
            else:
                print(f"  arm did not move (reasons={r.reasons})")

            # Markers attempted only AFTER real stepping (viewer live on main thread).
            if markers_ok is None:
                markers_ok = draw_box_markers(backend)
                print(f"  [floor markers: {'ON' if markers_ok else 'unavailable'}]\n")
            elif markers_ok:
                draw_box_markers(backend)
                print()
            else:
                print()
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        backend.close()
        print("\nclosed.")


if __name__ == "__main__":
    main()