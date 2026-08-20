"""sim/workspace.py -- the one place a location SYMBOL becomes a coordinate.

Phase-6 Step 2. Every consumer (measurement backend, demo, later the real arm)
asks this module, so `grid_B2` means the same physical point everywhere.

Coordinates are NOT hardcoded here. They are derived from schema/grid_spec.json,
whose geometry block was measured by sim/reachability.py (9x9 sweep, 3 trials per
candidate, reproduced identically on Windows and Linux). Change the grid there and
re-run; nothing in this file needs editing.

Two symbols are RUNTIME symbols: they have no fixed coordinate and are resolved
against the scene when the command runs. The LLM emits the symbol; the backend
resolves it. That split is deliberate -- the model cannot know the sweep result or
the current object layout, so scoring it against a resolved cell would mark
correct answers wrong.

  max_reach          -> the legal cell furthest from the robot base.
  nearest_free_cell  -> the unoccupied cell closest to the gripper.

BACKEND FRAMES: coordinates are in the MuJoCo FetchPickAndPlace-v4 world frame.
The igus ReBeL has a different base frame; when that backend is added, only
`frame_for()` changes, not the symbols and not the datasets.
"""
import json
import math
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
_SPEC = _ROOT / "schema" / "grid_spec.json"
_SCHEMA_V2 = _ROOT / "schema" / "schema_v2.json"

RUNTIME_SYMBOLS = ("max_reach", "nearest_free_cell")


def _load_spec():
    return json.loads(_SPEC.read_text(encoding="utf-8"))


def _lin(a, b, n):
    if n == 1:
        return [a]
    step = (b - a) / (n - 1)
    return [a + step * i for i in range(n)]


def grid_cells():
    """Return {cell_id: (x, y)} for every grid cell, from measured geometry."""
    spec = _load_spec()
    rows, cols = spec["grid"]["rows"], spec["grid"]["cols"]
    g = spec["geometry"]
    o = g.get("origin_xy_m")
    if not o:
        raise RuntimeError(
            "grid_spec.json geometry is unmeasured. Run sim/reachability.py first.")
    xs = _lin(o["row_A_x"], o["row_E_x"], len(rows))   # far -> near
    ys = _lin(o["col_1_y"], o["col_5_y"], len(cols))   # left -> right
    return {f"grid_{r}{c}": (round(xs[i], 4), round(ys[j], 4))
            for i, r in enumerate(rows) for j, c in enumerate(cols)}


def table_z():
    return _load_spec()["geometry"].get("table_z_m", 0.40)


def location_classes():
    if not _SCHEMA_V2.exists():
        return {}
    return json.loads(_SCHEMA_V2.read_text(encoding="utf-8")).get("location_classes", {})


# --- runtime symbol resolution -------------------------------------------------

def resolve(symbol, *, robot_xy=(1.05, 0.75), gripper_xy=None, occupied_xy=(),
            free_radius=0.10):
    """Resolve a location symbol to a cell id.

    Fixed cells and containers pass through untouched. Runtime symbols are
    resolved against the arguments and the result is a CELL ID, never a raw
    coordinate -- so the caller's own LOCATIONS lookup stays the single source of
    coordinates.

    robot_xy      : base position; max_reach is measured from here.
    gripper_xy    : current end-effector xy; nearest_free_cell measures from here.
    occupied_xy   : iterable of (x, y) for objects currently on the table.
    free_radius   : a cell is occupied if an object is within this many metres.
    """
    if symbol not in RUNTIME_SYMBOLS:
        return symbol
    cells = grid_cells()

    if symbol == "max_reach":
        return max(cells, key=lambda c: math.dist(cells[c], robot_xy))

    # nearest_free_cell
    origin = tuple(gripper_xy) if gripper_xy else tuple(robot_xy)
    free = [c for c, xy in cells.items()
            if all(math.dist(xy, o) > free_radius for o in occupied_xy)]
    if not free:
        free = list(cells)          # everything occupied: fall back to all cells
    return min(free, key=lambda c: math.dist(cells[c], origin))


def describe_cell(cell):
    """Human-readable position, for demo legends and terminal output."""
    spec = _load_spec()
    rows, cols = spec["grid"]["rows"], spec["grid"]["cols"]
    if not cell.startswith("grid_") or len(cell) < 7:
        return ""
    r, c = cell[5], cell[6]
    ri, ci = rows.index(r), cols.index(int(c))
    rw = ["far", "middle", "near"][ri] if len(rows) == 3 else f"row {r}"
    cw = ["left", "middle", "right"][ci] if len(cols) == 3 else f"col {c}"
    return f"{rw} row, {cw} column"


if __name__ == "__main__":
    cells = grid_cells()
    classes = location_classes()
    print(f"{len(cells)} cells, table top z={table_z()}\n")
    for c, (x, y) in cells.items():
        print(f"  {c:9} ({x:.2f}, {y:.2f})  {classes.get(c,'?'):9} {describe_cell(c)}")
    print("\nruntime symbols:")
    print(f"  max_reach          -> {resolve('max_reach')}")
    print(f"  nearest_free_cell  -> {resolve('nearest_free_cell', gripper_xy=(1.34, 0.75))}")
    print(f"  nearest_free_cell  -> {resolve('nearest_free_cell', gripper_xy=(1.34, 0.75), occupied_xy=[(1.30, 0.75)])}"
          "   (centre blocked)")
