"""sim/llm_demo.py -- ROUGH DEMO ONLY. Type a command, watch the arm move.

Chain: command -> LLM -> validated verb -> Phase 2 backend -> arm moves.
NOT part of the measurement. Demo machinery only.

Works with ANY of the five models (cloud or local). Local models occasionally
return an invalid object (arm won't move that turn); that's model unreliability,
not a demo bug.

V2: the backend now also knows the nine grid cells and the two runtime symbols
(see sim/workspace.py), so the demo accepts the SAME phrasings as the V2
datasets -- "put the bottle in the far left corner of the table". The V1 names
still work, so nothing that used to run has stopped running.

Floor markers (colored spheres at each location): best-effort. If they can't
draw, the demo still runs fine on the terminal legend alone.

macOS: run under plain `python`, NOT `mjpython`. mjpython runs user code on a
non-main thread, which makes gymnasium's render_mode="human" window creation
crash ('NSWindow should only be instantiated on the main thread').

Run from the REPO ROOT (not from sim/), so `sim.workspace` imports:
    python -m sim.llm_demo
"""
import sys, os
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

from full_grasp_backend import FullGraspBackend, LOCATIONS, OBJECT_OF, HAS_GRID, RUNTIME_SYMBOLS
from llm.prompts import load_schema
from llm.pipeline import run_once
from llm.clients import OpenAIClient, OllamaClient

# --- pick any model; all five work here ---
CLIENT = OpenAIClient("gpt-4o-mini")
# CLIENT = OllamaClient("llama3.2:3b")
# CLIENT = OllamaClient("qwen2.5:3b")
# CLIENT = OllamaClient("qwen2.5:0.5b")
# CLIENT = OllamaClient("deepseek-r1:7b")
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
COLOR_NAME = {"box_1": "red", "box_2": "green", "box_3": "blue", "table_center": "yellow",
              "table_left": "orange", "table_right": "purple", "table_front": "cyan",
              "table_back": "pink"}

# Grid cells get one colour per ROW so the far/middle/near axis is readable in
# the viewer -- that axis is the one models get wrong most often.
_ROW_COLOR = {"A": (0.95, 0.35, 0.25, 1.0),    # far   - warm
              "B": (0.95, 0.85, 0.25, 1.0),    # mid   - yellow
              "C": (0.30, 0.75, 0.95, 1.0)}    # near  - cool
_ROW_NAME = {"A": "far/red", "B": "mid/yellow", "C": "near/blue"}
for _c in [c for c in LOCATIONS if c.startswith("grid_")]:
    MARKER_COLORS[_c] = _ROW_COLOR.get(_c[5], (1, 1, 1, 1))
    COLOR_NAME[_c] = _ROW_NAME.get(_c[5], "?")


def draw_box_markers(backend):
    """Draw a colored sphere at each location. Best-effort: returns False and
    skips cleanly if the viewer has no reachable user scene. Never crashes."""
    import mujoco
    try:
        viewer = backend.env.unwrapped.mujoco_renderer.viewer
        scn = viewer.user_scn
        scn.ngeom = 0
        for name, (x, y) in LOCATIONS.items():
            if scn.ngeom >= scn.maxgeom:
                break
            g = scn.geoms[scn.ngeom]
            # grid cells drawn slightly smaller so overlapping V1 names stay visible
            radius = 0.022 if name.startswith("grid_") else 0.03
            mujoco.mjv_initGeom(
                g, mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([radius, 0, 0]),
                np.array([x, y, 0.42]),
                np.eye(3).flatten(),
                np.array(MARKER_COLORS.get(name, (1, 1, 1, 1)), dtype=np.float32),
            )
            scn.ngeom += 1
        return True
    except Exception:
        return False


def print_legend():
    print("\n" + "=" * 62)
    print("SCENE LEGEND   (markers are attempted after the first command)")
    print("-" * 62)
    if HAS_GRID:
        print("  V2 GRID  -- the robot sits below row C, arm reaches towards row A")
        print()
        print("            col 1 (left)   col 2 (middle)  col 3 (right)")
        for r, word in (("A", "far   "), ("B", "middle"), ("C", "near  ")):
            cells = []
            for c in "123":
                xy = LOCATIONS.get(f"grid_{r}{c}")
                cells.append(f"{r}{c} ({xy[0]:.2f},{xy[1]:.2f})" if xy else "--")
            print(f"    {word}  " + "  ".join(f"{s:16}" for s in cells))
        print(f"\n    marker colours: row A red | row B yellow | row C blue")
        print("    say it in words, e.g.  \"put the bottle in the far left corner of the table\"")
        print()
        print("  RUNTIME SYMBOLS (the backend picks the cell, not the model):")
        for s in RUNTIME_SYMBOLS:
            print(f"    {s}")
        print()
    else:
        print("  grid cells UNAVAILABLE (sim/workspace.py did not import)")
        print("  running with V1 locations only\n")
    v1 = [n for n in LOCATIONS if not n.startswith("grid_")]
    print("  V1 locations (still valid):")
    for name in v1:
        x, y = LOCATIONS[name]
        print(f"    {name:14} -> ({x:.2f}, {y:.2f})   [{COLOR_NAME.get(name,'?')}]")
    print("\n  ONE physical block. These names all point to it, so the arm")
    print("  grabs the same block whichever you say:")
    print(f"    {', '.join(OBJECT_OF.keys())}")
    print("=" * 62 + "\n")


def drive(backend, obj):
    act = obj.get("action")
    try:
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
    except KeyError as e:
        # A model can emit a location outside the schema. In the harness that is
        # a schema_failed row; in the demo it must not kill the session.
        return {"verb": act, "ok": False, "note": f"unknown location: {e}"}
    return {"verb": act, "ok": False, "note": "no backend method"}


def describe(obj):
    tgt = obj.get("target")
    loc = obj.get("location")
    if tgt:
        print(f"  treating block as: {tgt}")
    if loc in RUNTIME_SYMBOLS:
        print(f"  destination:       {loc}  (runtime symbol -- backend will choose the cell)")
    elif loc and loc in LOCATIONS:
        x, y = LOCATIONS[loc]
        extra = ""
        if loc.startswith("grid_"):
            try:
                from sim.workspace import describe_cell
                extra = f"  {describe_cell(loc)}"
            except Exception:
                pass
        print(f"  destination:       {loc}  ({x:.2f}, {y:.2f})  [{COLOR_NAME.get(loc,'?')}]{extra}")


def main():
    root = _ROOT
    name = "schema_v2.json" if HAS_GRID else "schema.json"
    schema = load_schema(os.path.join(root, "schema", name))
    print(f"schema: {name}  ({len(schema['locations'])} locations)")

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
                if result.get("resolved") and result["resolved"] != r.obj.get("location"):
                    print(f"  resolved to:       {result['resolved']}")
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
