"""
sim/isaac_scene_headless.py

Headless Isaac Sim scene: lit ground plane, 3x3 grid, one block, fixed front
camera framing all nine cells. Renders frames to PNG.

Run:  python sim/isaac_scene_headless.py
Out:  out/isaac_frames/frame_0000.png ...
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
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

GRID_ROWS = 3
GRID_COLS = 3
CELL = 0.12
TILE_THICKNESS = 0.002
OUT_DIR = os.path.join("out", "isaac_frames")
N_FRAMES = 30
WARMUP_FRAMES = 20

# Camera: further back and higher than the first pass, which cropped row 2.
CAM_POS = np.array([0.0, -1.35, 0.65])
CAM_EULER = np.array([0.0, 24.0, 90.0])

# Low dome + strong key: a bright dome flattens the checkerboard to one shade.
DOME_INTENSITY = 350.0
KEY_INTENSITY = 1800.0

SHADE_LIGHT = 0.92
SHADE_DARK = 0.18


def cell_centre(row, col):
    x = (col - (GRID_COLS - 1) / 2.0) * CELL
    y = ((GRID_ROWS - 1) / 2.0 - row) * CELL
    return x, y


def add_lights():
    stage = omni.usd.get_context().get_stage()

    dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
    dome.CreateIntensityAttr(DOME_INTENSITY)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 1.0))

    key = UsdLux.DistantLight.Define(stage, "/World/key_light")
    key.CreateIntensityAttr(KEY_INTENSITY)
    key.CreateAngleAttr(1.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))


def build_scene(world):
    world.scene.add(
        GroundPlane(
            prim_path="/World/ground",
            name="ground",
            size=10.0,
            color=np.array([0.12, 0.12, 0.14]),
        )
    )

    for row in range(GRID_ROWS):
        for col in range(GRID_COLS):
            x, y = cell_centre(row, col)
            shade = SHADE_LIGHT if (row + col) % 2 == 0 else SHADE_DARK
            world.scene.add(
                VisualCuboid(
                    prim_path=f"/World/tile_{row}_{col}",
                    name=f"tile_{row}_{col}",
                    position=np.array([x, y, TILE_THICKNESS / 2.0]),
                    scale=np.array([CELL * 0.92, CELL * 0.92, TILE_THICKNESS]),
                    color=np.array([shade, shade, shade]),
                )
            )

    bx, by = cell_centre(0, 0)
    world.scene.add(
        DynamicCuboid(
            prim_path="/World/block_red",
            name="block_red",
            position=np.array([bx, by, 0.03]),
            scale=np.array([0.04, 0.04, 0.04]),
            color=np.array([0.9, 0.1, 0.1]),
        )
    )


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    world = World(stage_units_in_meters=1.0)
    build_scene(world)
    add_lights()

    camera = Camera(
        prim_path="/World/front_camera",
        position=CAM_POS,
        frequency=20,
        resolution=(1280, 720),
        orientation=rot_utils.euler_angles_to_quats(CAM_EULER, degrees=True),
    )

    world.reset()
    camera.initialize()

    for _ in range(WARMUP_FRAMES):
        world.step(render=True)

    saved = 0
    for i in range(N_FRAMES):
        world.step(render=True)
        rgba = camera.get_rgba()
        if rgba is None or rgba.size == 0:
            continue
        rgb = rgba[:, :, :3].astype(np.uint8)
        if i == 0:
            print(f"[isaac_scene_headless] frame 0 mean pixel = {rgb.mean():.2f}")
        Image.fromarray(rgb).save(os.path.join(OUT_DIR, f"frame_{i:04d}.png"))
        saved += 1

    print(f"[isaac_scene_headless] saved {saved} frames to {OUT_DIR}")
    simulation_app.close()


if __name__ == "__main__":
    main()