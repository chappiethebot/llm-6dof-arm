"""
sim/isaac_scene_window.py

Windowed Isaac Sim scene using the FULL Kit experience, which provides the
editor UI (viewport, stage tree, transform gizmos). The default 'base'
experience has no UI, which is why earlier windowed runs were unusable.

SimulationApp needs an ABSOLUTE path to the .kit file; the bare name only
resolves when launching via the `isaacsim` command.

Run:  python sim/isaac_scene_window.py
"""

import os

from isaacsim import SimulationApp
import isaacsim as _isaacsim_pkg

_APPS = os.path.join(os.path.dirname(_isaacsim_pkg.__file__), "apps")
_EXPERIENCE = os.path.join(_APPS, "isaacsim.exp.full.kit")

simulation_app = SimulationApp(
    {
        "headless": False,
        "width": 1600,
        "height": 900,
    },
    experience=_EXPERIENCE,
)

import numpy as np

import omni.usd
from pxr import UsdLux, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, VisualCuboid, GroundPlane
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.utils.viewports import set_camera_view

GRID_ROWS, GRID_COLS = 3, 3
GRID_CENTRE_X = 0.45
CELL = 0.11
TILE_THICKNESS = 0.002
OBJ_SIZE = 0.045

OBJECTS = {
    "rubiks_cube": {"cell": (0, 0), "color": (0.85, 0.15, 0.15)},
    "bottle":      {"cell": (1, 2), "color": (0.15, 0.45, 0.85)},
    "toy_car":     {"cell": (2, 1), "color": (0.95, 0.75, 0.10)},
}

CAM_EYE = [-0.55, -0.95, 0.85]
CAM_TARGET = [0.45, 0.0, 0.05]


def cell_centre(row, col):
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


def build_scene(world):
    world.scene.add(
        GroundPlane(
            prim_path="/World/ground",
            name="ground",
            size=10.0,
            color=np.array([0.14, 0.14, 0.16]),
        )
    )

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

    world.scene.add(Franka(prim_path="/World/franka", name="franka"))


def main():
    print(f"[window] experience = {_EXPERIENCE}")

    world = World(stage_units_in_meters=1.0)
    build_scene(world)
    add_lights()
    world.reset()

    set_camera_view(eye=CAM_EYE, target=CAM_TARGET)

    print("[window] running -- close the window to exit")
    print("[window] first launch rebuilds the shader cache for this "
          "experience; leave it alone until it paints")

    step = 0
    while simulation_app.is_running():
        world.step(render=True)
        simulation_app.update()
        step += 1
        if step % 300 == 0:
            print(f"[window] step {step}")

    simulation_app.close()


if __name__ == "__main__":
    main()