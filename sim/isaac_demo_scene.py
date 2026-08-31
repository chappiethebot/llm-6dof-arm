"""
sim/isaac_demo_scene.py

Stage A of the Isaac demo: Franka Panda at the origin, a 3x3 grid in front of
it, three open-topped boxes, and three graspable objects. Renders frames so we
can confirm the layout is sane and reachable before writing any verbs.

Run:  python sim/isaac_demo_scene.py
Out:  out/isaac_demo/frame_0000.png ...
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import numpy as np
from PIL import Image

import omni.usd
from pxr import UsdLux, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid, GroundPlane
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

OUT_DIR = os.path.join("out", "isaac_demo")
N_FRAMES = 20
WARMUP_FRAMES = 30

# --- layout, all metres, Franka base at the origin, +x forward -------------
GRID_ROWS, GRID_COLS = 3, 3
GRID_CENTRE_X = 0.5
CELL = 0.18
TILE_THICKNESS = 0.002

BOX_INNER = 0.12          # must exceed Franka's open finger span
BOX_WALL = 0.006
BOX_HEIGHT = 0.05
BOX_POSITIONS = {
    "box_1": (0.40, 0.30),
    "box_2": (0.52, 0.30),
    "box_3": (0.58, 0.30),
}

OBJ_SIZE = 0.05
OBJECTS = {
    "rubiks_cube": {"cell": (0, 0), "color": (0.85, 0.15, 0.15)},
    "bottle":      {"cell": (1, 2), "color": (0.15, 0.45, 0.85)},
    "toy_car":     {"cell": (2, 1), "color": (0.95, 0.75, 0.10)},
}

CAM_POS = np.array([-0.55, -0.95, 0.85])
CAM_EULER = np.array([0.0, 27.0, 55.0])


def cell_centre(row, col):
    """Grid cell -> world (x, y). Row 0 is nearest the robot."""
    x = GRID_CENTRE_X + (row - (GRID_ROWS - 1) / 2.0) * CELL
    y = (col - (GRID_COLS - 1) / 2.0) * CELL
    return x, y


def add_lights():
    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
    dome.CreateIntensityAttr(400.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/key_light")
    key.CreateIntensityAttr(2000.0)
    key.CreateAngleAttr(1.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))


def add_grid(world):
    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x, y = cell_centre(row, col)
            shade = 0.90 if (row + col) % 2 == 0 else 0.20
            world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/tile_{row}_{col}",
                    name=f"tile_{row}_{col}",
                    position=np.array([x, y, TILE_THICKNESS / 2.0]),
                    scale=np.array([CELL * 0.92, CELL * 0.92, TILE_THICKNESS]),
                    color=np.array([shade, shade, shade]),
                )
            )


def add_box(world, name, cx, cy):
    """Open-topped container: floor plus four static walls."""
    colour = np.array([0.35, 0.30, 0.25])
    span = BOX_INNER + 2 * BOX_WALL

    world.scene.add(
        VisualCuboid(
            prim_path=f"/World/{name}_floor",
            name=f"{name}_floor",
            position=np.array([cx, cy, BOX_WALL / 2.0]),
            scale=np.array([span, span, BOX_WALL]),
            color=colour,
        )
    )

    half = (BOX_INNER + BOX_WALL) / 2.0
    walls = [
        ("n", cx + half, cy, BOX_WALL, span),
        ("s", cx - half, cy, BOX_WALL, span),
        ("e", cx, cy + half, span, BOX_WALL),
        ("w", cx, cy - half, span, BOX_WALL),
    ]
    for tag, wx, wy, sx, sy in walls:
        world.scene.add(
            VisualCuboid(
                prim_path=f"/World/{name}_wall_{tag}",
                name=f"{name}_wall_{tag}",
                position=np.array([wx, wy, BOX_HEIGHT / 2.0]),
                scale=np.array([sx, sy, BOX_HEIGHT]),
                color=colour,
            )
        )


def add_objects(world):
    for name, spec in OBJECTS.items():
        row, col = spec["cell"]
        x, y = cell_centre(row, col)
        world.scene.add(
            DynamicCuboid(
                prim_path=f"/World/{name}",
                name=name,
                position=np.array([x, y, OBJ_SIZE / 2.0 + 0.005]),
                scale=np.array([OBJ_SIZE, OBJ_SIZE, OBJ_SIZE]),
                color=np.array(spec["color"]),
                mass=0.05,
            )
        )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    world.scene.add(
        GroundPlane(
            prim_path="/World/ground",
            name="ground",
            size=10.0,
            color=np.array([0.14, 0.14, 0.16]),
        )
    )

    add_grid(world)
    for box_name, (cx, cy) in BOX_POSITIONS.items():
        add_box(world, box_name, cx, cy)
    add_objects(world)

    franka = world.scene.add(
        Franka(prim_path="/World/franka", name="franka")
    )

    add_lights()

    camera = Camera(
        prim_path="/World/demo_camera",
        position=CAM_POS,
        frequency=20,
        resolution=(1280, 720),
        orientation=rot_utils.euler_angles_to_quats(CAM_EULER, degrees=True),
    )

    world.reset()
    camera.initialize()

    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    # Reachability sanity check: distance from base to each named target.
    print("[scene] reach check (Franka usable range is roughly 0.30-0.75 m):")
    for name in OBJECTS:
        pos, _ = world.scene.get_object(name).get_world_pose()
        print(f"  {name:14s} at {np.round(pos, 3)}  dist={np.linalg.norm(pos[:2]):.3f}")
    for box_name, (cx, cy) in BOX_POSITIONS.items():
        print(f"  {box_name:14s} at [{cx:.3f} {cy:.3f}]  dist={np.hypot(cx, cy):.3f}")

    print(f"[scene] franka dof count = {franka.num_dof}")

    saved = 0
    for i in range(N_FRAMES):
        world.step(render=True)
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            continue
        Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(
            os.path.join(OUT_DIR, f"frame_{i:04d}.png")
        )
        saved += 1

    print(f"[scene] saved {saved} frames to {OUT_DIR}")
    simulation_app.close()


if __name__ == "__main__":
    main()