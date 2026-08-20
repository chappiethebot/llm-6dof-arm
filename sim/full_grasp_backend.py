import numpy as np
import gymnasium as gym, gymnasium_robotics
gym.register_envs(gymnasium_robotics)
from gymnasium_robotics.utils import mujoco_utils
import mujoco

Z_CARRY = 0.55
HANDOVER = (1.22, 0.75, 0.50)

# V1 locations. Frozen: these produced the 22,520 V1 rows and must not move.
LOCATIONS = {
    "box_1": (1.24, 0.65), "box_2": (1.44, 0.65), "box_3": (1.44, 0.85),
    "table_center": (1.34, 0.75), "table_left": (1.34, 0.63), "table_right": (1.34, 0.87),
    "table_front": (1.22, 0.75), "table_back": (1.46, 0.75),
}

# V2 grid cells, merged in from sim/workspace.py (derived from the measured
# geometry in schema/grid_spec.json). Additive: every V1 name still resolves, so
# the V1 demo and any V1 replay keep working unchanged. If workspace.py or the
# spec is missing, the backend degrades to V1-only rather than failing to import.
try:
    try:
        from sim.workspace import grid_cells, resolve as resolve_symbol, RUNTIME_SYMBOLS
    except ImportError:                       # running from inside sim/
        from workspace import grid_cells, resolve as resolve_symbol, RUNTIME_SYMBOLS
    LOCATIONS.update(grid_cells())
    HAS_GRID = True
except Exception as _e:          # noqa: BLE001 - deliberate: demo must still start
    RUNTIME_SYMBOLS = ()
    HAS_GRID = False

    def resolve_symbol(symbol, **_kw):
        return symbol

# name map kept simple so 1 object -> 3 objects is a config change later, not a rewrite
OBJECT_OF = {"rubiks_cube": "object0", "bottle": "object0", "toy_car": "object0"}


class FullGraspBackend:
    """Single physical object, real grasp. Every grasp verb verifies the lift
    and retries. Structured so adding objects later is a scene+map change."""

    def __init__(self, render=False, gain=10.0, step_limit=150):
        self.env = gym.make("FetchPickAndPlace-v4", render_mode=("human" if render else None))
        self.gain, self.step_limit = gain, step_limit
        self.obs, _ = self.env.reset()
        self.u = self.env.unwrapped

    # ---- location handling -------------------------------------------------
    def _xy(self, location):
        """Resolve a location SYMBOL to (x, y).

        Runtime symbols (max_reach, nearest_free_cell) have no fixed coordinate;
        they are resolved here against the live scene. The resolved cell id is
        stored on self.last_resolved so the caller can report what it chose --
        important for the demo, and for any future execution-success logging.
        """
        self.last_resolved = location
        if location in RUNTIME_SYMBOLS:
            cell = resolve_symbol(
                location,
                gripper_xy=tuple(self._grip_pos()[:2]),
                occupied_xy=[tuple(self.obj_pos()[:2])],
            )
            self.last_resolved = cell
            location = cell
        if location not in LOCATIONS:
            raise KeyError(
                f"unknown location {location!r}. known: {sorted(LOCATIONS)}"
                + ("" if HAS_GRID else "  (grid cells unavailable: sim/workspace.py did not import)"))
        return LOCATIONS[location]

    def obj_pos(self, obj="object0"):
        return mujoco_utils.get_joint_qpos(self.u.model, self.u.data, f"{obj}:joint")[:3].copy()

    def set_object(self, xyz, obj="object0"):
        q = mujoco_utils.get_joint_qpos(self.u.model, self.u.data, f"{obj}:joint").copy()
        q[:3] = xyz; q[3:] = [1, 0, 0, 0]
        mujoco_utils.set_joint_qpos(self.u.model, self.u.data, f"{obj}:joint", q)
        mujoco.mj_forward(self.u.model, self.u.data)
        self.obs, _, _, _, _ = self.env.step(np.zeros(4))

    def randomize_object(self):
        self.obs, _ = self.env.reset()
        return self.obj_pos()

    def _grip_pos(self):
        return self.obs["observation"][0:3].copy()

    def _reach(self, target, grip, tol=0.02):
        target = np.asarray(target, float)
        for i in range(self.step_limit):
            err = target - self._grip_pos()
            if np.linalg.norm(err) < tol:
                return i
            self.obs, _, _, _, _ = self.env.step(
                np.clip(np.concatenate([self.gain * err, [grip]]), -1, 1))
        return self.step_limit

    def _grip(self, cmd, hold=20):
        for _ in range(hold):
            self.obs, _, _, _, _ = self.env.step(np.array([0, 0, 0, cmd], float))
        return hold

    def _grasp_verified(self, obj, retries=3):
        """Grasp, check the object actually lifted, retry if it missed."""
        steps, tries = 0, 0
        for attempt in range(retries):
            tries = attempt + 1
            P = self.obj_pos(obj)
            steps += self._reach([P[0], P[1], P[2] + 0.10], 1.0)   # above, open
            steps += self._reach([P[0], P[1], P[2]], 1.0, tol=0.01) # descend, open
            steps += self._grip(-1.0, 22)                           # close
            steps += self._reach([P[0], P[1], Z_CARRY], -1.0)       # lift
            if self.obj_pos(obj)[2] > 0.50:                         # verified up?
                return True, steps, tries
            steps += self._grip(1.0, 8)                             # missed: release, retry
        return False, steps, tries

    def move_to(self, location):
        x, y = self._xy(location)
        s = self._reach([x, y, 0.47], 0.0)
        return {"verb": "move_to", "location": location,
                "resolved": self.last_resolved, "ok": True, "steps": s}

    def pick_place(self, target, location):
        obj = OBJECT_OF[target]; x, y = self._xy(location)
        held, s1, tries = self._grasp_verified(obj)
        s2 = 0
        if held:
            s2 += self._reach([x, y, Z_CARRY], -1.0)
            s2 += self._reach([x, y, 0.45], -1.0)
            s2 += self._grip(1.0, 15)
            s2 += self._reach([x, y, Z_CARRY], 1.0)
        P = self.obj_pos(obj)
        gap = float(np.linalg.norm(P[:2] - np.array([x, y])))
        # z check: gap is an XY distance, so an object knocked onto the FLOOR can
        # still report a small gap. Verified in sim: a place aimed off the table
        # returned ok=True with the object at z=0.025. The table top is ~0.40.
        on_table = bool(P[2] > 0.35)
        return {"verb": "pick_place", "target": target, "location": location,
                "resolved": self.last_resolved,
                "ok": held and gap < 0.06 and on_table, "grasp_tries": tries,
                "steps": s1 + s2, "gap_cm": round(gap * 100, 1),
                "on_table": on_table}

    def fetch(self, target):
        obj = OBJECT_OF[target]
        held, s1, tries = self._grasp_verified(obj)
        s2 = 0
        if held:
            s2 += self._reach([HANDOVER[0], HANDOVER[1], Z_CARRY], -1.0)
            s2 += self._reach(list(HANDOVER), -1.0)
            s2 += self._grip(1.0, 15)
        return {"verb": "fetch", "target": target, "ok": held, "grasp_tries": tries, "steps": s1 + s2}

    def hold(self, target):
        obj = OBJECT_OF[target]
        held, s1, tries = self._grasp_verified(obj)
        s2 = self._grip(-1.0, 15) if held else 0
        return {"verb": "hold", "target": target, "ok": held, "grasp_tries": tries,
                "held_z": round(float(self.obj_pos(obj)[2]), 3), "steps": s1 + s2}

    def inspect(self, target):
        obj = OBJECT_OF[target]
        held, s1, tries = self._grasp_verified(obj)
        s2 = self._grip(-1.0, 30) if held else 0   # lift + pause (no wrist rotation in sim)
        return {"verb": "inspect", "target": target, "ok": held, "grasp_tries": tries,
                "held_z": round(float(self.obj_pos(obj)[2]), 3), "steps": s1 + s2}

    def stop(self):
        return {"verb": "stop", "ok": True, "steps": 0}

    def reject(self):
        return {"verb": "reject", "ok": True, "steps": 0}

    def close(self):
        self.env.close()


if __name__ == "__main__":
    be = FullGraspBackend(render=False)
    print(f"locations available: {len(LOCATIONS)}  (grid cells: {'yes' if HAS_GRID else 'NO'})")
    rng = np.random.default_rng(0)
    stats = {}
    for k in range(6):
        x, y = rng.uniform(1.28, 1.40), rng.uniform(0.68, 0.82)
        be.set_object([x, y, 0.425]); r = be.pick_place("rubiks_cube", "box_1"); stats.setdefault("pick_place", []).append(r["ok"]); print(r)
        be.set_object([x, y, 0.425]); r = be.fetch("bottle"); stats.setdefault("fetch", []).append(r["ok"]); print(r)
        be.set_object([x, y, 0.425]); r = be.hold("toy_car"); stats.setdefault("hold", []).append(r["ok"]); print(r)
        be.set_object([x, y, 0.425]); r = be.inspect("rubiks_cube"); stats.setdefault("inspect", []).append(r["ok"]); print(r)
    if HAS_GRID:
        print("\n== grid cells ==")
        for cell in ["grid_A1", "grid_B2", "grid_C3", "max_reach", "nearest_free_cell"]:
            be.set_object([1.34, 0.75, 0.425])
            r = be.pick_place("bottle", cell)
            stats.setdefault("grid", []).append(r["ok"])
            print(f"  {cell:18} -> {r['resolved']:10} ok={r['ok']} gap={r['gap_cm']}cm")
    print("\n== reliability ==")
    for v, oks in stats.items():
        print(f"{v:11s} {sum(oks)}/{len(oks)}")
    be.close()
