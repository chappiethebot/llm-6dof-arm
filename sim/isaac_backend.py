"""
sim/isaac_backend.py

Isaac Sim 6.0 backend implementing the same seven schema verbs as
sim/sim_backend.py (MuJoCo), so run_once() does not change.

Differences from the MuJoCo backend, deliberate:
  * Objects are REAL rigid bodies and are physically carried. The MuJoCo
    backend mimed the motion without transporting anything.
  * Object positions are QUERIED from the live stage, not hardcoded. Drag a
    cube in the editor and the verbs pick it up where you left it.
  * Coordinates are re-derived for a Franka at the origin. MuJoCo's Fetch
    frame values (1.34, 0.75, 0.43) are meaningless here.

SimulationApp must be constructed before any isaacsim import, so this module
does its imports lazily. Call launch() first, then construct IsaacBackend.

    from sim.isaac_backend import launch, IsaacBackend
    launch(headless=False)
    be = IsaacBackend()
    print(be.pick_place("rubiks_cube", "box_1"))
    be.close()

Run directly for a self-test:
    python sim/isaac_backend.py
"""

import math
import os

# ---------------------------------------------------------------------------
# Layout. All metres, Franka base at the origin, +x forward.
# ---------------------------------------------------------------------------

GRID_ROWS, GRID_COLS = 3, 3
GRID_CENTRE_X = 0.50
CELL = 0.18                 # 0.11 was too tight; the hand fouled neighbours
TILE_THICKNESS = 0.002

OBJ_SIZE = 0.0515           # the cube size Isaac's own Franka example uses
OBJ_MASS = 0.03

# Boxes on an arc at constant radius, so no box is harder to reach than
# another. Low walls: deep boxes make the hand collide on descent.
BOX_RADIUS = 0.62
BOX_ANGLES_DEG = {"box_1": 40.0, "box_2": 60.0, "box_3": 80.0}
BOX_INNER = 0.16
BOX_WALL = 0.006
BOX_HEIGHT = 0.03

HANDOVER = (0.34, -0.34, 0.05)

OBJECTS = {
    "rubiks_cube": {"cell": (0, 0), "color": (0.85, 0.15, 0.15)},
    "bottle":      {"cell": (1, 2), "color": (0.15, 0.45, 0.85)},
    "toy_car":     {"cell": (2, 1), "color": (0.95, 0.75, 0.10)},
}

# Measured from the working grasp test: distance from the panda_hand frame to
# the centre of a held object.
EE_TO_OBJECT_DZ = 0.044
EE_OFFSET_XY = (0.0, 0.005)

STEP_LIMIT = 1500
LIFT_EVENT = 6              # controller event by which the object is airborne

CAM_POS = (0.95, -1.05, 0.70)
CAM_EULER = (0.0, 22.0, 135.0)

_APP = None
_M = {}                     # late-imported modules and classes


def cell_centre(row, col):
    """Grid cell -> world (x, y). Row 0 is nearest the robot."""
    x = GRID_CENTRE_X + (row - (GRID_ROWS - 1) / 2.0) * CELL
    y = (col - (GRID_COLS - 1) / 2.0) * CELL
    return x, y


def box_centre(name):
    ang = math.radians(BOX_ANGLES_DEG[name])
    return BOX_RADIUS * math.cos(ang), BOX_RADIUS * math.sin(ang)


def launch(headless=True):
    """Start Isaac. Must be called before constructing IsaacBackend.

    headless=False opens the full editor window, where objects can be dragged
    by hand; the verbs then act on wherever you left them.
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
            experience=os.path.join(apps, "isaacsim.exp.full.kit"),
        )

    _late_imports()
    return _APP


def _late_imports():
    """Import isaacsim modules only after SimulationApp exists."""
    import numpy as np
    import omni.usd
    from pxr import UsdLux, Gf

    from isaacsim.core.api import World
    from isaacsim.core.api.objects import (
        DynamicCuboid, VisualCuboid, GroundPlane,
    )
    from isaacsim.robot.manipulators.examples.franka import Franka
    from isaacsim.robot.manipulators.examples.franka.controllers import (
        PickPlaceController,
    )
    from isaacsim.sensors.camera import Camera
    import isaacsim.core.utils.numpy.rotations as rot_utils

    _M.update(
        np=np, omni_usd=omni.usd, UsdLux=UsdLux, Gf=Gf,
        World=World, DynamicCuboid=DynamicCuboid, VisualCuboid=VisualCuboid,
        GroundPlane=GroundPlane, Franka=Franka,
        PickPlaceController=PickPlaceController, Camera=Camera,
        rot_utils=rot_utils,
    )


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
            _M["Franka"](prim_path="/World/franka", name="franka")
        )
        self._add_lights()
        self._add_camera()

        self.world.reset()
        self.camera.initialize()

        # Required: without this the fingers start closed and the arm can only
        # shove objects around. Isaac's own tutorials do it in post_reset().
        self.franka.gripper.set_joint_positions(
            self.franka.gripper.joint_opened_positions
        )

        self.articulation_controller = self.franka.get_articulation_controller()
        self._settle(30)

    # -- scene ----------------------------------------------------------

    def _build_scene(self):
        np = self.np
        self.world.scene.add(
            _M["GroundPlane"](
                prim_path="/World/ground", name="ground", size=10.0,
                color=np.array([0.14, 0.14, 0.16]),
            )
        )

        for row in range(GRID_ROWS):
            for col in range(GRID_COLS):
                x, y = cell_centre(row, col)
                shade = 0.90 if (row + col) % 2 == 0 else 0.20
                self.world.scene.add(
                    _M["VisualCuboid"](
                        prim_path=f"/World/tile_{row}_{col}",
                        name=f"tile_{row}_{col}",
                        position=np.array([x, y, TILE_THICKNESS / 2.0]),
                        scale=np.array(
                            [CELL * 0.92, CELL * 0.92, TILE_THICKNESS]
                        ),
                        color=np.array([shade, shade, shade]),
                    )
                )

        for box_name in BOX_ANGLES_DEG:
            self._add_box(box_name)

        for name, spec in OBJECTS.items():
            row, col = spec["cell"]
            x, y = cell_centre(row, col)
            self.world.scene.add(
                _M["DynamicCuboid"](
                    prim_path=f"/World/{name}", name=name,
                    position=np.array([x, y, OBJ_SIZE / 2.0 + 0.002]),
                    scale=np.array([OBJ_SIZE, OBJ_SIZE, OBJ_SIZE]),
                    color=np.array(spec["color"]), mass=OBJ_MASS,
                )
            )

    def _add_box(self, name):
        np = self.np
        cx, cy = box_centre(name)
        colour = np.array([0.35, 0.30, 0.25])
        span = BOX_INNER + 2 * BOX_WALL

        self.world.scene.add(
            _M["VisualCuboid"](
                prim_path=f"/World/{name}_floor", name=f"{name}_floor",
                position=np.array([cx, cy, BOX_WALL / 2.0]),
                scale=np.array([span, span, BOX_WALL]), color=colour,
            )
        )

        half = (BOX_INNER + BOX_WALL) / 2.0
        for tag, wx, wy, sx, sy in [
            ("n", cx + half, cy, BOX_WALL, span),
            ("s", cx - half, cy, BOX_WALL, span),
            ("e", cx, cy + half, span, BOX_WALL),
            ("w", cx, cy - half, span, BOX_WALL),
        ]:
            self.world.scene.add(
                _M["VisualCuboid"](
                    prim_path=f"/World/{name}_wall_{tag}",
                    name=f"{name}_wall_{tag}",
                    position=np.array([wx, wy, BOX_HEIGHT / 2.0]),
                    scale=np.array([sx, sy, BOX_HEIGHT]), color=colour,
                )
            )

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
                np.array(CAM_EULER), degrees=True
            ),
        )

    # -- name resolution -------------------------------------------------

    def resolve(self, name):
        """Schema name -> (x, y, z).

        Movable objects report their LIVE pose, so hand-dragged positions are
        honoured. Fixed locations are computed from the layout.
        """
        np = self.np

        if name in OBJECTS:
            obj = self.world.scene.get_object(name)
            pos, _ = obj.get_world_pose()
            return np.array(pos, dtype=float)

        if name in BOX_ANGLES_DEG:
            cx, cy = box_centre(name)
            return np.array([cx, cy, BOX_HEIGHT + OBJ_SIZE / 2.0])

        if name == "handover":
            return np.array(HANDOVER, dtype=float)

        if name.startswith("cell_"):
            try:
                _, r, c = name.split("_")
                x, y = cell_centre(int(r), int(c))
                return np.array([x, y, OBJ_SIZE / 2.0])
            except (ValueError, IndexError):
                pass

        raise KeyError(f"unknown schema name: {name!r}")

    # -- motion primitives -----------------------------------------------

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
                os.path.join(
                    self.capture_dir, f"frame_{self._frame_index:05d}.png"
                )
            )
        self._frame_index += 1

    def _transport(self, target_name, place_xyz, stop_after_lift=False):
        """Grasp `target_name` and carry it to `place_xyz`.

        stop_after_lift=True holds the object in the air instead of placing it.
        Returns (ok, steps, final_error_m).
        """
        np = self.np
        obj = self.world.scene.get_object(target_name)

        controller = _M["PickPlaceController"](
            name=f"pp_{target_name}",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
        )
        ee_offset = np.array([EE_OFFSET_XY[0], EE_OFFSET_XY[1], 0.0])
        place_xyz = np.asarray(place_xyz, dtype=float)

        steps = 0
        for steps in range(1, self.step_limit + 1):
            self.world.step(render=True)
            self._maybe_capture()

            obj_pos, _ = obj.get_world_pose()
            action = controller.forward(
                picking_position=obj_pos,
                placing_position=place_xyz,
                current_joint_positions=self.franka.get_joint_positions(),
                end_effector_offset=ee_offset,
            )
            self.articulation_controller.apply_action(action)

            if stop_after_lift and controller.get_current_event() >= LIFT_EVENT:
                final, _ = obj.get_world_pose()
                lifted = float(final[2]) > OBJ_SIZE
                return lifted, steps, 0.0

            if controller.is_done():
                break

        final, _ = obj.get_world_pose()
        err = float(np.linalg.norm(final[:2] - place_xyz[:2]))
        return err < 0.06, steps, err

    # -- the seven schema verbs -------------------------------------------

    def move_to(self, location, gripper=0.0):
        """Hover the gripper above a named location."""
        np = self.np
        try:
            target = self.resolve(location)
        except KeyError:
            raise
        hover = np.array([target[0], target[1], 0.25])

        controller = _M["PickPlaceController"](
            name="pp_move", gripper=self.franka.gripper,
            robot_articulation=self.franka,
        )
        steps = 0
        for steps in range(1, min(self.step_limit, 400) + 1):
            self.world.step(render=True)
            self._maybe_capture()
            action = controller.forward(
                picking_position=hover, placing_position=hover,
                current_joint_positions=self.franka.get_joint_positions(),
                end_effector_offset=np.array([0.0, 0.005, 0.0]),
            )
            self.articulation_controller.apply_action(action)
            if controller.get_current_event() >= 1:
                break

        ee, _ = self.franka.end_effector.get_world_pose()
        err = float(np.linalg.norm(np.array(ee)[:2] - hover[:2]))
        return {"verb": "move_to", "location": location,
                "ok": err < 0.08, "steps": steps,
                "error_cm": round(err * 100, 1)}

    def pick_place(self, target, location):
        """Grasp `target`, carry it, release at `location`."""
        self.resolve(target)                      # KeyError on bad name
        dst = self.resolve(location)
        ok, steps, err = self._transport(target, dst)
        return {"verb": "pick_place", "target": target, "location": location,
                "ok": bool(ok), "steps": steps, "error_cm": round(err * 100, 1)}

    def fetch(self, target):
        """Grasp `target` and bring it to the fixed handover point."""
        self.resolve(target)
        ok, steps, err = self._transport(target, self.resolve("handover"))
        return {"verb": "fetch", "target": target,
                "ok": bool(ok), "steps": steps, "error_cm": round(err * 100, 1)}

    def hold(self, target):
        """Grasp `target`, lift, and keep holding it."""
        self.resolve(target)
        pos = self.resolve(target)
        aloft = self.np.array([pos[0], pos[1], 0.30])
        ok, steps, err = self._transport(target, aloft, stop_after_lift=True)
        self._settle(40)
        return {"verb": "hold", "target": target,
                "ok": bool(ok), "steps": steps, "error_cm": round(err * 100, 1)}

    def inspect(self, target):
        """Grasp `target`, lift, and hold for a longer pause.

        NOTE: as in the MuJoCo backend, the 'rotate slowly' part of inspect is
        not modelled; in sim it is a pause.
        """
        self.resolve(target)
        pos = self.resolve(target)
        aloft = self.np.array([pos[0], pos[1], 0.30])
        ok, steps, err = self._transport(target, aloft, stop_after_lift=True)
        self._settle(90)
        return {"verb": "inspect", "target": target,
                "ok": bool(ok), "steps": steps, "error_cm": round(err * 100, 1)}

    def stop(self):
        """Halt. Nothing to interrupt in this synchronous model."""
        return {"verb": "stop", "ok": True, "steps": 0, "error_cm": 0.0}

    def reject(self):
        """Out-of-scope request acknowledged; the arm does not move."""
        return {"verb": "reject", "ok": True, "steps": 0, "error_cm": 0.0}

    # -- dispatch ---------------------------------------------------------

    def dispatch(self, action):
        """Execute a validated schema action dict, e.g.
        {"verb": "pick_place", "target": "rubiks_cube", "location": "box_1"}
        """
        verb = action.get("verb")
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
        if verb == "reject":
            return self.reject()
        raise KeyError(f"unknown verb: {verb!r}")

    def close(self):
        if _APP is not None:
            _APP.close()


def _self_test():
    launch(headless=True)
    be = IsaacBackend(capture_dir=os.path.join("out", "isaac_backend"))

    print("[backend] reach check:")
    for name in list(OBJECTS) + list(BOX_ANGLES_DEG) + ["handover"]:
        p = be.resolve(name)
        d = float(be.np.linalg.norm(p[:2]))
        print(f"  {name:14s} {be.np.round(p, 3)}  dist={d:.3f}")
    print("")

    print(be.pick_place("rubiks_cube", "box_1"))
    print(be.pick_place("bottle", "box_2"))
    print(be.pick_place("rubiks_cube", "box_3"))   # box -> box
    print(be.hold("toy_car"))
    print(be.stop())
    print(be.reject())
    try:
        be.hold("dragon")
    except KeyError as e:
        print("invalid target ->", e)

    be.close()


if __name__ == "__main__":
    _self_test()