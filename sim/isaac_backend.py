"""
sim/isaac_backend.py

Isaac Sim 6.0 backend implementing the same seven schema verbs as
sim/sim_backend.py (MuJoCo), so run_once() does not change.

Differences from the MuJoCo backend, deliberate:
  * Objects are REAL rigid bodies and are physically carried.
  * Object positions are QUERIED from the live stage, so objects dragged by
    hand in the editor are picked up where you left them.
  * Coordinates re-derived for a Franka at the origin.
  * Boxes hold ONE object; a second placement is refused, not dropped on top.

Boxes sit in a straight row, not on an arc. The earlier arc layout gave
adjacent centres 0.16 m apart while each box spanned 0.172 m, so the walls
interpenetrated.

Self-test:  python sim/isaac_backend.py
"""

import os
import traceback

GRID_ROWS, GRID_COLS = 3, 3
GRID_CENTRE_X = 0.50
CELL = 0.18
TILE_THICKNESS = 0.002

OBJ_SIZE = 0.0515
OBJ_MASS = 0.03

# Straight row, clear of the grid (grid spans y +/-0.18).
BOX_POSITIONS = {
    "box_1": (0.30, 0.40),
    "box_2": (0.48, 0.40),
    "box_3": (0.66, 0.40),
}
BOX_INNER = 0.14
BOX_WALL = 0.006
BOX_HEIGHT = 0.03

HANDOVER = (0.34, -0.34, 0.05)

# Table top. Its TOP SURFACE sits at z = 0, so it is purely cosmetic: every
# coordinate in this file is measured from z = 0 and none of them change. It is
# a VisualCuboid, so it has no collider either -- objects still rest on the
# GroundPlane at z = 0, which is exactly coplanar with the table surface. That
# means adding it cannot alter a single pick height, drop height or physics
# result, and the 0.1-0.5 cm placement figure stays valid.
#
# Extent covers the grid (x 0.32-0.68, y -0.18-0.18), the three boxes
# (y ~0.32-0.48) and the handover point (0.34, -0.34), while stopping short of
# the Franka base at the origin so the robot is not embedded in the tabletop.
TABLE_CENTRE = (0.52, 0.0)
TABLE_SIZE = (0.86, 1.16)
TABLE_THICKNESS = 0.02
TABLE_COLOR = (0.58, 0.54, 0.48)

OBJECTS = {
    "rubiks_cube": {"cell": (0, 0), "color": (0.85, 0.15, 0.15)},
    "bottle":      {"cell": (1, 2), "color": (0.15, 0.45, 0.85)},
    "toy_car":     {"cell": (1, 1), "color": (0.95, 0.75, 0.10)},
}

EE_TO_OBJECT_DZ = 0.044
EE_OFFSET_XY = (0.0, 0.005)

STEP_LIMIT = 1500
LIFT_EVENT = 6

# move_to convergence. See move_to() for why the old `event >= 1` break was
# wrong and why these exist.
MOVE_TOL = 0.02          # 2 cm in XY counts as arrived
MOVE_STEP_CAP = 900      # hard ceiling; ~7x the old fixed 125
MOVE_STALL_STEPS = 150   # give up if the error stops improving for this long
MOVE_HOVER_Z = 0.25

# Runtime symbols. Values copied from sim/workspace.py, not re-chosen: a cell
# counts as occupied when an object sits within FREE_RADIUS of its centre.
# CELL is 0.18, so 0.10 blocks exactly the cell an object stands in and no
# neighbour -- the same property it has on the MuJoCo grid.
RUNTIME_SYMBOLS = ("max_reach", "nearest_free_cell")
FREE_RADIUS = 0.10

# The seven schema verbs this backend implements, checked in dispatch() BEFORE
# the try block so an unknown verb still raises KeyError rather than being
# classified as an execution failure. Mirrors schema/schema_v2.json; that file
# is frozen and is not read here on purpose -- the backend must not acquire a
# load-time dependency on the measurement schema.
DISPATCH_VERBS = ("move_to", "pick_place", "fetch", "hold", "inspect",
                  "stop", "reject")

# ---------------------------------------------------------------------------
# FAILURE TAXONOMY
#
# Every unsuccessful result carries a `failure` key from this set. It exists
# because "execution_failed" told a supervisor watching a demo nothing at all:
# an arm that cannot reach a cell, an arm that reached it and dropped the cube,
# and an Isaac exception all looked identical.
#
# ADDITIVE ONLY. `failure` appears alongside the existing ok/steps/error_cm
# keys and only when ok is False. No existing key changes name, type or
# meaning. The demo path writes no CSV, so nothing downstream is affected.
# ---------------------------------------------------------------------------
FAIL_UNREACHABLE = "unreachable"     # arm could not servo to the commanded pose
FAIL_GRASP = "grasp_failed"          # object was never picked up
FAIL_PLACEMENT = "placement_error"   # carried, but released too far from target
FAIL_BLOCKED = "blocked"             # collision or obstructed path
FAIL_TIMEOUT = "timeout"             # OUR loop hit its step cap, still trying
FAIL_CONTROL_BUDGET = "control_budget"   # the CONTROLLER's schedule expired first
FAIL_UNKNOWN = "execution_failed"    # genuine unknown; traceback is logged

# WHY control_budget IS ITS OWN LABEL AND NOT JUST timeout.
#
# Both mean "ran out of time", but they point at different things and have
# different fixes, so collapsing them would send the next person to the wrong
# place:
#
#   timeout         our loop in move_to/_transport hit MOVE_STEP_CAP or
#                   STEP_LIMIT while the arm was still working. Fix: raise
#                   the cap, or accept that the motion is genuinely slow.
#   control_budget  PickPlaceController advanced through its own fixed
#                   events_dt schedule before the arm arrived. Our cap was
#                   never reached. Raising it changes nothing, because the
#                   controller has already moved on to its grasp phase.
#                   Fix: stop driving a bare move with a pick-and-place
#                   state machine. See audit/ISAAC_MOVE_TO_LIMITATION.md.
#
# This was FAIL_UNREACHABLE until the reach test settled it: `put the rubiks
# cube in box 3` succeeds at 0.2 cm error to a point 0.7718 m from the base,
# further than cell_2_0 at 0.7034 m. The arm can reach where max_reach sends
# it. It is the controller's schedule that runs out, not the workspace.

# An object that ends within this of where it was picked up never moved, which
# means the gripper closed on nothing. Half a cube; well outside settle jitter.
GRASP_MOVED_MIN = OBJ_SIZE / 2.0

# Exception-message signatures. HEURISTIC and deliberately narrow: Isaac
# publishes no exception taxonomy, so type name and message text are the only
# route. A wrong specific label is worse than an honest generic one, so anything
# unmatched keeps FAIL_UNKNOWN and logs a traceback.
_EXC_PATTERNS = (
    (FAIL_UNREACHABLE, ("inverse kinematic", " ik ", "ik solver", "no solution",
                        "unreachable", "out of range", "joint limit")),
    (FAIL_BLOCKED, ("collision", "penetrat", "contact force", "obstructed")),
    (FAIL_TIMEOUT, ("timeout", "timed out", "deadline")),
)


def classify_exception(exc):
    """Best-effort failure label for an exception raised during motion.

    Returns FAIL_UNKNOWN when nothing matches, which is the correct answer far
    more often than a guess would be.
    """
    text = ("%s %s" % (type(exc).__name__, exc)).lower()
    for label, needles in _EXC_PATTERNS:
        if any(n in text for n in needles):
            return label
    return FAIL_UNKNOWN


CAM_POS = (0.49, -1.30, 0.95)
CAM_EULER = (0.0, 33.0, 90.0)

_APP = None
_M = {}


def cell_centre(row, col):
    """Grid cell -> world (x, y). Row 0 is nearest the robot."""
    x = GRID_CENTRE_X + (row - (GRID_ROWS - 1) / 2.0) * CELL
    y = (col - (GRID_COLS - 1) / 2.0) * CELL
    return x, y


def box_centre(name):
    return BOX_POSITIONS[name]


def launch(headless=True):
    """Start Isaac. Must be called before constructing IsaacBackend.

    headless=False opens the full editor, where objects can be dragged by hand.
    """
    global _APP
    if _APP is not None:
        return _APP

    from isaacsim import SimulationApp
    import isaacsim as _pkg

    if headless:
        _APP = SimulationApp({"headless": True})
    else:
        apps = os.path.join(os.path.dirname(_pkg.__file__), "apps")
        _APP = SimulationApp(
            {"headless": False, "width": 1600, "height": 900},
            experience=os.path.join(apps, "isaacsim.exp.full.kit"))

    _late_imports()
    return _APP


def _late_imports():
    import numpy as np
    import omni.usd
    from pxr import UsdLux, Gf

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import (
        DynamicCuboid, VisualCuboid, GroundPlane)
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        PickPlaceController)
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils

    _M.update(
        np=np, omni_usd=omni.usd, UsdLux=UsdLux, Gf=Gf,
        World=World, DynamicCuboid=DynamicCuboid, VisualCuboid=VisualCuboid,
        GroundPlane=GroundPlane, Franka=Franka,
        PickPlaceController=PickPlaceController, Camera=Camera,
        rot_utils=rot_utils)


class IsaacBackend:
    """Seven schema verbs on a Franka. Same return contract as SimBackend."""

    def __init__(self, step_limit=STEP_LIMIT, capture_dir=None):
        if _APP is None:
            raise RuntimeError("call launch() before constructing IsaacBackend")

        np = _M["np"]
        self.np = np
        self.step_limit = step_limit
        self.capture_dir = capture_dir
        self._frame_index = 0

        if capture_dir:
            os.makedirs(capture_dir, exist_ok=True)

        self.world = _M["World"](stage_units_in_meters=1.0)
        self._build_scene()

        self.franka = self.world.scene.add(
            _M["Franka"](prim_path="/World/franka", name="franka"))
        self._add_lights()
        self._add_camera()

        self.world.reset()
        self.camera.initialize()

        # Required. Without this the fingers start closed and the arm can only
        # shove objects. Isaac's own tutorials do this in post_reset().
        self.franka.gripper.set_joint_positions(
            self.franka.gripper.joint_opened_positions)

        self.articulation_controller = self.franka.get_articulation_controller()
        self._settle(30)

        # max_reach is measured from the base, so read it rather than assume
        # the Franka is at the origin.
        base, _ = self.franka.get_world_pose()
        self.robot_xy = (float(base[0]), float(base[1]))

        self.occupancy = dict((n, None) for n in BOX_POSITIONS)
        self._sync_occupancy()

    # -- scene ------------------------------------------------------------

    def _build_scene(self):
        np = self.np
        self.world.scene.add(
            _M["GroundPlane"](
                prim_path="/World/ground", name="ground", size=10.0,
                color=np.array([0.14, 0.14, 0.16])))

        self._add_table()

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x, y = cell_centre(row, col)
                shade = 0.90 if (row + col) % 2 == 0 else 0.20
                self.world.scene.add(
                    _M["VisualCuboid"](
                        prim_path="/World/tile_%d_%d" % (row, col),
                        name="tile_%d_%d" % (row, col),
                        position=np.array([x, y, TILE_THICKNESS / 2.0]),
                        scale=np.array(
                            [CELL * 0.92, CELL * 0.92, TILE_THICKNESS]),
                        color=np.array([shade, shade, shade])))

        for box_name in BOX_POSITIONS:
            self._add_box(box_name)

        for name, spec in OBJECTS.items():
            row, col = spec["cell"]
            x, y = cell_centre(row, col)
            self.world.scene.add(
                _M["DynamicCuboid"](
                    prim_path="/World/%s" % name, name=name,
                    position=np.array([x, y, OBJ_SIZE / 2.0 + 0.002]),
                    scale=np.array([OBJ_SIZE, OBJ_SIZE, OBJ_SIZE]),
                    color=np.array(spec["color"]), mass=OBJ_MASS))

    def _add_box(self, name):
        np = self.np
        cx, cy = box_centre(name)
        colour = np.array([0.35, 0.30, 0.25])
        span = BOX_INNER + 2 * BOX_WALL

        self.world.scene.add(
            _M["VisualCuboid"](
                prim_path="/World/%s_floor" % name, name="%s_floor" % name,
                position=np.array([cx, cy, BOX_WALL / 2.0]),
                scale=np.array([span, span, BOX_WALL]), color=colour))

        half = (BOX_INNER + BOX_WALL) / 2.0
        walls = [("n", cx + half, cy, BOX_WALL, span),
                 ("s", cx - half, cy, BOX_WALL, span),
                 ("e", cx, cy + half, span, BOX_WALL),
                 ("w", cx, cy - half, span, BOX_WALL)]
        for tag, wx, wy, sx, sy in walls:
            self.world.scene.add(
                _M["VisualCuboid"](
                    prim_path="/World/%s_wall_%s" % (name, tag),
                    name="%s_wall_%s" % (name, tag),
                    position=np.array([wx, wy, BOX_HEIGHT / 2.0]),
                    scale=np.array([sx, sy, BOX_HEIGHT]), color=colour))

    def _add_table(self):
        """A visual slab whose top surface is exactly z = 0.

        Deliberately a VisualCuboid, not a DynamicCuboid or a collider: the
        GroundPlane at z = 0 already carries the contact physics, and the table
        is coplanar with it. Giving the table a collider would put two contact
        surfaces at the same height and change how objects settle, which would
        invalidate the measured placement accuracy for no visual gain.
        """
        np = self.np
        cx, cy = TABLE_CENTRE
        sx, sy = TABLE_SIZE
        self.world.scene.add(
            _M["VisualCuboid"](
                prim_path="/World/table_top", name="table_top",
                position=np.array([cx, cy, -TABLE_THICKNESS / 2.0]),
                scale=np.array([sx, sy, TABLE_THICKNESS]),
                color=np.array(TABLE_COLOR)))

    def _add_lights(self):
        stage = _M["omni_usd"].get_context().get_stage()
        UsdLux, Gf = _M["UsdLux"], _M["Gf"]
        dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
        dome.CreateIntensityAttr(400.0)
        dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 1.0))
        key = UsdLux.DistantLight.Define(stage, "/World/key_light")
        key.CreateIntensityAttr(2000.0)
        key.CreateAngleAttr(1.0)
        key.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))

    def _add_camera(self):
        np = self.np
        self.camera = _M["Camera"](
            prim_path="/World/demo_camera",
            position=np.array(CAM_POS), frequency=20, resolution=(1280, 720),
            orientation=_M["rot_utils"].euler_angles_to_quats(
                np.array(CAM_EULER), degrees=True))

    # -- name resolution ---------------------------------------------------

    def _grid_cells(self):
        """{cell_id: (x, y)} for all nine cells."""
        return dict(("cell_%d_%d" % (r, c), cell_centre(r, c))
                    for r in range(GRID_ROWS) for c in range(GRID_COLS))

    def resolve_runtime(self, symbol, exclude=None):
        """Runtime symbol -> cell id, by the SAME rules as sim/workspace.py.

        Mirrors workspace.resolve() deliberately, rather than inventing an Isaac
        rule. If the two ever disagree, the same command means two different
        physical places on the two backends, with no error raised -- the
        `_GRID_COL` failure mode again.

        workspace.py's rules, reproduced:
          max_reach          -> the cell furthest from the robot base.
          nearest_free_cell  -> the unoccupied cell closest to the gripper;
                                a cell is occupied if an object sits within
                                FREE_RADIUS of it; if nothing is free, fall back
                                to the full cell list rather than failing.

        `exclude` drops one object from the occupancy test. It exists because
        "put the cube anywhere free" must not treat the cube's own current cell
        as blocked -- the cube will have been lifted before anything is placed.
        workspace.py delegates this to the caller through `occupied_xy`, so the
        convention lives in the MuJoCo caller. CHECK sim/sim_backend.py PASSES
        THE TARGET THE SAME WAY; if it does not, one of the two is wrong.
        """
        np = self.np
        cells = self._grid_cells()

        if symbol == "max_reach":
            # TIES ARE REAL HERE AND THE TIE-BREAK IS DELIBERATE. With the
            # Franka at the origin and the grid symmetric about y = 0,
            # cell_2_0 and cell_2_2 are both 0.703420 m from the base --
            # exactly equal. A bare max() would resolve that by dict insertion
            # order, which is deterministic but arbitrary and invisible.
            # Sorting on (-distance, cell id) makes the choice explicit and
            # reproducible: among equally distant cells, the lowest id wins.
            # sim/workspace.py has no tie because the MuJoCo grid is not
            # symmetric about the base, so this is an Isaac-only property and
            # belongs in the write-up as one, not as a shared rule.
            return sorted(cells,
                          key=lambda c: (-float(np.hypot(
                              cells[c][0] - self.robot_xy[0],
                              cells[c][1] - self.robot_xy[1])), c))[0]

        occupied = []
        for obj_name in OBJECTS:
            if obj_name == exclude:
                continue
            pos, _ = self.world.scene.get_object(obj_name).get_world_pose()
            occupied.append((float(pos[0]), float(pos[1])))

        ee, _ = self.franka.end_effector.get_world_pose()
        origin = (float(ee[0]), float(ee[1]))

        free = [c for c, xy in cells.items()
                if all(float(np.hypot(xy[0] - ox, xy[1] - oy)) > FREE_RADIUS
                       for ox, oy in occupied)]
        if not free:
            free = list(cells)
        return min(free, key=lambda c: float(
            np.hypot(cells[c][0] - origin[0], cells[c][1] - origin[1])))

    def resolve(self, name, exclude=None):
        """Schema name -> (x, y, z). Objects report their LIVE pose."""
        np = self.np

        if name in OBJECTS:
            pos, _ = self.world.scene.get_object(name).get_world_pose()
            return np.array(pos, dtype=float)

        if name in BOX_POSITIONS:
            cx, cy = box_centre(name)
            return np.array([cx, cy, BOX_HEIGHT + OBJ_SIZE / 2.0])

        if name == "handover":
            return np.array(HANDOVER, dtype=float)

        if name in RUNTIME_SYMBOLS:
            return self.resolve(self.resolve_runtime(name, exclude=exclude))

        if name.startswith("cell_"):
            try:
                _, r, c = name.split("_")
                x, y = cell_centre(int(r), int(c))
                return np.array([x, y, OBJ_SIZE / 2.0])
            except (ValueError, IndexError):
                pass

        raise KeyError("unknown schema name: %r" % (name,))

    # -- occupancy ---------------------------------------------------------

    def _in_box(self, obj_xy):
        """Which box, if any, contains this (x, y)?"""
        for box_name in BOX_POSITIONS:
            cx, cy = box_centre(box_name)
            half = BOX_INNER / 2.0
            if abs(obj_xy[0] - cx) <= half and abs(obj_xy[1] - cy) <= half:
                return box_name
        return None

    def _sync_occupancy(self):
        """Rebuild occupancy from live positions.

        Derived rather than tracked incrementally, so objects dragged by hand
        in the editor are accounted for.
        """
        self.occupancy = dict((n, None) for n in BOX_POSITIONS)
        for obj_name in OBJECTS:
            pos, _ = self.world.scene.get_object(obj_name).get_world_pose()
            box = self._in_box(pos)
            if box is not None:
                self.occupancy[box] = obj_name
        return self.occupancy

    # -- motion ------------------------------------------------------------

    def _settle(self, n):
        for _ in range(n):
            self.world.step(render=True)
            self._maybe_capture()

    def _maybe_capture(self):
        if not self.capture_dir:
            return
        if self._frame_index % 6 != 0:
            self._frame_index += 1
            return
        rgba = self.camera.get_rgba()
        if rgba is not None and rgba.size > 0:
            from PIL import Image
            Image.fromarray(rgba[:, :, :3].astype("uint8")).save(
                os.path.join(self.capture_dir,
                             "frame_%05d.png" % self._frame_index))
        self._frame_index += 1

    def _transport(self, target_name, place_xyz, stop_after_lift=False):
        """Grasp `target_name` and carry it to `place_xyz`.

        The pick position is sampled ONCE before motion begins. Re-reading it
        each step makes the controller chase the object after it is lifted,
        which corrupts the approach on any object that has already moved.
        """
        np = self.np
        obj = self.world.scene.get_object(target_name)

        self._settle(5)
        pick_xyz, _ = obj.get_world_pose()
        pick_xyz = np.array(pick_xyz, dtype=float)

        controller = _M["PickPlaceController"](
            name="pp_%s" % target_name, gripper=self.franka.gripper,
            robot_articulation=self.franka)
        ee_offset = np.array([EE_OFFSET_XY[0], EE_OFFSET_XY[1], 0.0])
        place_xyz = np.asarray(place_xyz, dtype=float)

        steps = 0
        hit_cap = True
        for steps in range(1, self.step_limit + 1):
            self.world.step(render=True)
            self._maybe_capture()

            action = controller.forward(
                picking_position=pick_xyz, placing_position=place_xyz,
                current_joint_positions=self.franka.get_joint_positions(),
                end_effector_offset=ee_offset)
            self.articulation_controller.apply_action(action)

            if stop_after_lift and controller.get_current_event() >= LIFT_EVENT:
                final, _ = obj.get_world_pose()
                lifted = float(final[2]) > OBJ_SIZE
                # A lift that did not lift is a grasp failure, not a near miss.
                return lifted, steps, 0.0, (None if lifted else FAIL_GRASP)

            if controller.is_done():
                hit_cap = False
                break

        final, _ = obj.get_world_pose()
        err = float(np.linalg.norm(final[:2] - place_xyz[:2]))
        ok = err < 0.06
        return ok, steps, err, self._why_transport_failed(
            ok, hit_cap, np.array(final, dtype=float), pick_xyz)

    def _why_transport_failed(self, ok, hit_cap, final_xyz, pick_xyz):
        """Which of the failure labels fits this transport outcome.

        Order matters. The object not having moved is the most informative
        answer available and outranks the step cap: an arm that spends 1500
        steps failing to close on a cube has a grasp problem, and calling that
        a timeout would send the next person to look at the wrong thing.
        """
        if ok:
            return None
        moved = float(self.np.linalg.norm(final_xyz - pick_xyz))
        if moved < GRASP_MOVED_MIN:
            return FAIL_GRASP
        if hit_cap:
            return FAIL_TIMEOUT
        return FAIL_PLACEMENT

    # -- the seven schema verbs --------------------------------------------

    def move_to(self, location, gripper=0.0):
        """Hover the gripper above a named location.

        WHY THIS LOOP IS SHAPED LIKE THIS. The previous version broke out on
        `controller.get_current_event() >= 1`, which is the moment
        PickPlaceController LEAVES its approach phase -- not the moment the arm
        arrives anywhere. Phase 0 runs for a fixed number of steps from the
        controller's own `events_dt` schedule, so every call returned exactly
        125 steps regardless of target, stopped mid-approach, and reported
        whatever XY error it happened to have (4-5 cm observed). The old
        `ok` gate of 0.08 m then passed that as success.

        Now the loop watches the actual end-effector pose and stops on one of
        three conditions, in priority order:
          * arrived      -- XY error under MOVE_TOL;
          * descended    -- the controller has run past its descent phase
                            (event >= 2), so continuing would close the gripper
                            on empty air;
          * stalled      -- the error stopped improving for MOVE_STALL_STEPS,
                            which means the target is unreachable rather than
                            slow. Reporting that honestly is better than
                            spinning to the step cap.

        `steps` is now genuinely informative: it varies with distance. A
        constant value across different targets means this broke again.
        """
        np = self.np
        resolved = location
        if location in RUNTIME_SYMBOLS:
            resolved = self.resolve_runtime(location)
        target = self.resolve(resolved)
        hover = np.array([target[0], target[1], MOVE_HOVER_Z])

        controller = _M["PickPlaceController"](
            name="pp_move", gripper=self.franka.gripper,
            robot_articulation=self.franka)

        err = float("inf")
        best = float("inf")
        stalled = 0
        reason = "step_cap"
        steps = 0
        cap = min(self.step_limit, MOVE_STEP_CAP)

        for steps in range(1, cap + 1):
            self.world.step(render=True)
            self._maybe_capture()
            action = controller.forward(
                picking_position=hover, placing_position=hover,
                current_joint_positions=self.franka.get_joint_positions(),
                end_effector_offset=np.array([0.0, 0.005, 0.0]))
            self.articulation_controller.apply_action(action)

            ee, _ = self.franka.end_effector.get_world_pose()
            err = float(np.linalg.norm(np.array(ee)[:2] - hover[:2]))

            if err < best - 1e-4:
                best = err
                stalled = 0
            else:
                stalled += 1

            if err < MOVE_TOL:
                reason = "arrived"
                break
            if controller.get_current_event() >= 2:
                reason = "descended"
                break
            if stalled >= MOVE_STALL_STEPS:
                reason = "stalled"
                break

        ok = err < MOVE_TOL
        out = {"verb": "move_to", "location": location,
               "ok": ok, "steps": steps,
               "error_cm": round(err * 100, 1),
               "best_cm": round(best * 100, 1),
               "stop": reason}
        if not ok:
            # move_to already distinguished its own stop conditions; this maps
            # them onto the shared taxonomy without discarding the finer detail,
            # which stays in `stop`.
            #   stalled    the error stopped improving for MOVE_STALL_STEPS ->
            #              the arm cannot get closer -> genuinely unreachable
            #   descended  PickPlaceController's fixed events_dt schedule
            #              advanced to its grasp phase while the arm was STILL
            #              CONVERGING (stalled never fired). That is the
            #              controller's budget expiring, not the workspace
            #              ending -- confirmed by the box_3 reach test, see
            #              audit/ISAAC_MOVE_TO_LIMITATION.md. Was mislabelled
            #              FAIL_UNREACHABLE until that test settled it.
            #   step_cap   our own MOVE_STEP_CAP hit -> a timeout
            out["failure"] = {"stalled": FAIL_UNREACHABLE,
                              "descended": FAIL_CONTROL_BUDGET,
                              "step_cap": FAIL_TIMEOUT}.get(reason, FAIL_UNKNOWN)
        if resolved != location:
            out["resolved"] = resolved
        return out

    def pick_place(self, target, location):
        """Grasp `target`, carry it, release at `location`.

        Refuses to place into an occupied box. Taking an object OUT of a box
        needs no special verb: pick_place(obj, "cell_1_1") does it, and the
        vacated box becomes free.
        """
        self.resolve(target)
        resolved = location
        if location in RUNTIME_SYMBOLS:
            resolved = self.resolve_runtime(location, exclude=target)
        dst = self.resolve(resolved)

        self._sync_occupancy()
        if location in self.occupancy:
            holder = self.occupancy[location]
            if holder is not None and holder != target:
                return {"verb": "pick_place", "target": target,
                        "location": location, "ok": False, "steps": 0,
                        "error_cm": 0.0, "refused": "box_occupied",
                        "failure": FAIL_BLOCKED, "occupant": holder}

        ok, steps, err, failure = self._transport(target, dst)
        self._sync_occupancy()

        out = {"verb": "pick_place", "target": target, "location": location,
               "ok": bool(ok), "steps": steps, "error_cm": round(err * 100, 1)}
        if failure:
            out["failure"] = failure
        if resolved != location:
            out["resolved"] = resolved
        return out

    def fetch(self, target):
        """Grasp `target` and bring it to the fixed handover point."""
        self.resolve(target)
        ok, steps, err, failure = self._transport(target, self.resolve("handover"))
        self._sync_occupancy()
        out = {"verb": "fetch", "target": target,
               "ok": bool(ok), "steps": steps, "error_cm": round(err * 100, 1)}
        if failure:
            out["failure"] = failure
        return out

    def hold(self, target):
        """Grasp `target`, lift, and keep holding it."""
        pos = self.resolve(target)
        aloft = self.np.array([pos[0], pos[1], 0.30])
        ok, steps, err, failure = self._transport(target, aloft, stop_after_lift=True)
        self._settle(40)
        out = {"verb": "hold", "target": target,
               "ok": bool(ok), "steps": steps, "error_cm": round(err * 100, 1)}
        if failure:
            out["failure"] = failure
        return out

    def inspect(self, target):
        """Grasp `target`, lift, hold for a longer pause.

        NOTE: as in the MuJoCo backend, the 'rotate slowly' part of inspect is
        not modelled; in sim it is a pause.
        """
        pos = self.resolve(target)
        aloft = self.np.array([pos[0], pos[1], 0.30])
        ok, steps, err, failure = self._transport(target, aloft, stop_after_lift=True)
        self._settle(90)
        out = {"verb": "inspect", "target": target,
               "ok": bool(ok), "steps": steps, "error_cm": round(err * 100, 1)}
        if failure:
            out["failure"] = failure
        return out

    def stop(self):
        """Halt. Nothing to interrupt in this synchronous model."""
        return {"verb": "stop", "ok": True, "steps": 0, "error_cm": 0.0}

    def reject(self):
        """Out-of-scope request acknowledged; the arm does not move."""
        return {"verb": "reject", "ok": True, "steps": 0, "error_cm": 0.0}

    # -- dispatch ----------------------------------------------------------

    def dispatch(self, action):
        """Execute a validated schema action dict.

        Exceptions raised INSIDE a verb are classified here rather than left to
        the bridge, which could only ever say "execution_failed". Two kinds are
        deliberately NOT caught:

          KeyError     resolve() raises it for an unknown schema name, and the
                       bridge already reports that as unknown_name_or_verb --
                       a different and more useful answer than any label here.
          unknown verb re-raised below, same reason.

        Everything else becomes a normal result dict with a `failure` label, so
        a failed command prints like a failed command instead of a traceback.
        An unrecognised exception keeps FAIL_UNKNOWN and logs its traceback --
        it is never swallowed.
        """
        verb = action.get("verb")
        if verb not in DISPATCH_VERBS:
            raise KeyError("unknown verb: %r" % (verb,))

        try:
            if verb == "move_to":
                return self.move_to(action["location"])
            if verb == "pick_place":
                return self.pick_place(action["target"], action["location"])
            if verb == "fetch":
                return self.fetch(action["target"])
            if verb == "hold":
                return self.hold(action["target"])
            if verb == "inspect":
                return self.inspect(action["target"])
            if verb == "stop":
                return self.stop()
            return self.reject()
        except KeyError:
            raise                      # unknown schema name; the bridge labels it
        except Exception as exc:       # noqa: BLE001 -- classified, never silent
            label = classify_exception(exc)
            if label == FAIL_UNKNOWN:
                # Genuinely unrecognised. Log it loudly; a demo failure nobody
                # can diagnose afterwards is the thing this replaced.
                print("[backend] UNCLASSIFIED failure in %r:" % (verb,))
                traceback.print_exc()
            out = {"verb": verb, "ok": False, "steps": 0, "error_cm": 0.0,
                   "failure": label,
                   "detail": "%s: %s" % (type(exc).__name__, exc)}
            for key in ("target", "location"):
                if key in action:
                    out[key] = action[key]
            return out

    # -- scene control -----------------------------------------------------
    #
    # DELIBERATELY NOT A VERB, AND DELIBERATELY NOT IN dispatch().
    #
    # reset_scene() is reachable only from the bridge's `control` channel, which
    # carries a top-level "control" key instead of "verb". A model cannot emit
    # one: bench/llm_client.py's to_backend_action() builds the outgoing dict
    # from scratch as {"verb": ...} plus the verb's field-mask keys, and REFUSES
    # any field outside that mask rather than dropping it. There is no path by
    # which model output becomes a control message.
    #
    # Adding a "reset" case to dispatch() above would break that, because
    # dispatch() is what every validated action reaches. Do not.

    def reset_scene(self):
        """Restore the start state: objects home, boxes empty, arm home.

        Objects are re-derived from OBJECTS + cell_centre(), the SAME source
        _build_scene() spawns them from -- never a second table of poses.

        Poses are set explicitly rather than trusted to world.reset(). Isaac
        does track initial state for registered scene objects, but these are
        rigid bodies that may be mid-fall or mid-bounce when the reset arrives,
        and an explicit restore costs three calls and removes the question.
        Velocities are zeroed for the same reason: a cube returned to its cell
        while still carrying momentum walks straight back off it.

        Boxes need no explicit clearing. _sync_occupancy() DERIVES occupancy
        from live positions, so moving the objects home empties the boxes by
        construction -- that is what clears a `box_occupied` refusal.
        """
        np = self.np

        self.world.reset()

        # REQUIRED after every world.reset(), not just the one in __init__.
        # Without it the fingers come back closed and the arm shoves objects
        # instead of grasping them. See the same call at __init__.
        self.franka.gripper.set_joint_positions(
            self.franka.gripper.joint_opened_positions)

        restored = []
        for name, spec in OBJECTS.items():
            row, col = spec["cell"]
            x, y = cell_centre(row, col)
            home = np.array([x, y, OBJ_SIZE / 2.0 + 0.002])
            obj = self.world.scene.get_object(name)
            obj.set_world_pose(position=home,
                               orientation=np.array([1.0, 0.0, 0.0, 0.0]))
            # Zero momentum. set_world_pose moves the body; it does not stop it.
            for setter in ("set_linear_velocity", "set_angular_velocity"):
                fn = getattr(obj, setter, None)
                if fn is not None:
                    fn(np.zeros(3))
            restored.append(name)

        self._settle(30)
        self._sync_occupancy()

        return {"ok": True, "control": "reset", "objects": restored,
                "occupancy": dict(self.occupancy), "steps": 30}

    def close(self):
        if _APP is not None:
            _APP.close()


def _self_test():
    import sys
    headless = "--window" not in sys.argv
    launch(headless=headless)
    be = IsaacBackend(capture_dir=os.path.join("out", "isaac_backend"))

    print("[backend] runtime symbols:")
    for sym in RUNTIME_SYMBOLS:
        print("  %-18s -> %s" % (sym, be.resolve_runtime(sym)))
    print("  %-18s -> %s  (rubiks_cube excluded)"
          % ("nearest_free_cell", be.resolve_runtime("nearest_free_cell",
                                                     exclude="rubiks_cube")))
    print("")

    print("[backend] reach check:")
    for name in list(OBJECTS) + list(BOX_POSITIONS) + ["handover"]:
        p = be.resolve(name)
        print("  %-14s %s  dist=%.3f"
              % (name, be.np.round(p, 3), float(be.np.linalg.norm(p[:2]))))
    print("")

    print(be.pick_place("rubiks_cube", "box_1"))
    print("occupancy:", be.occupancy)
    print(be.pick_place("toy_car", "box_1"))
    print(be.pick_place("rubiks_cube", "cell_1_1"))
    print("occupancy:", be.occupancy)
    print(be.pick_place("toy_car", "box_1"))
    print(be.hold("bottle"))
    print(be.stop())
    print(be.reject())
    try:
        be.hold("dragon")
    except KeyError as e:
        print("invalid target ->", e)
    print(be.pick_place("bottle", "box_3"))
    

    be.close()


if __name__ == "__main__":
    _self_test()