"""
sim/isaac_grasp_test.py

Stage B gate: can Franka grasp, LIFT, and place one object?

Key fix: the gripper must be explicitly opened after world.reset(). Isaac's
own tutorials do this in post_reset(); without it the fingers start closed,
the arm descends shut, and it shoves the cube instead of grasping it.

Instrumented: logs gripper finger positions and end-effector pose each
interval so failures are diagnosable from the terminal alone.

Run:  python sim/isaac_grasp_test.py
"""

from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": True})

import os
import numpy as np
from PIL import Image

import omni.usd
from pxr import UsdLux, Gf

from isaacsim.core.api import World
from isaacsim.core.api.objects import DynamicCuboid, GroundPlane
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.robot.manipulators.examples.franka.controllers import (
    PickPlaceController,
)
from isaacsim.sensors.camera import Camera
import isaacsim.core.utils.numpy.rotations as rot_utils

OUT_DIR = os.path.join("out", "isaac_grasp")
MAX_STEPS = 1500
FRAME_EVERY = 6
LOG_EVERY = 50

OBJ_SIZE = 0.0515          # matches the cube size in Isaac's own example
CUBE_MASS = 0.03
PICK_XY = (0.40, -0.10)
PLACE_XY = (0.50, 0.20)

# Isaac's Franka pick-place example uses a small +y offset.
EE_OFFSET = np.array([0.0, 0.005, 0.0])

LIFT_THRESHOLD = 0.04
PLACE_TOLERANCE = 0.06

CAM_POS = np.array([0.95, -1.05, 0.70])
CAM_EULER = np.array([0.0, 22.0, 135.0])


def add_lights():
    stage = omni.usd.get_context().get_stage()
    dome = UsdLux.DomeLight.Define(stage, "/World/dome_light")
    dome.CreateIntensityAttr(400.0)
    dome.CreateColorAttr(Gf.Vec3f(0.85, 0.88, 1.0))
    key = UsdLux.DistantLight.Define(stage, "/World/key_light")
    key.CreateIntensityAttr(2000.0)
    key.CreateAngleAttr(1.0)
    key.AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 30.0))


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

    cube = world.scene.add(
        DynamicCuboid(
            prim_path="/World/target_cube",
            name="target_cube",
            position=np.array([PICK_XY[0], PICK_XY[1], OBJ_SIZE / 2.0]),
            scale=np.array([OBJ_SIZE, OBJ_SIZE, OBJ_SIZE]),
            color=np.array([0.85, 0.15, 0.15]),
            mass=CUBE_MASS,
        )
    )

    franka = world.scene.add(Franka(prim_path="/World/franka", name="franka"))
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

    # THE FIX: open the gripper after reset. Without this the fingers start
    # closed and the arm can only shove the cube around.
    opened = franka.gripper.joint_opened_positions
    closed = franka.gripper.joint_closed_positions
    franka.gripper.set_joint_positions(opened)

    print(f"[grasp] gripper opened positions = {opened}")
    print(f"[grasp] gripper closed positions = {closed}")
    print(f"[grasp] gripper after open call  = {franka.gripper.get_joint_positions()}")

    controller = PickPlaceController(
        name="pick_place",
        gripper=franka.gripper,
        robot_articulation=franka,
    )
    articulation_controller = franka.get_articulation_controller()

    start_pos, _ = cube.get_world_pose()
    place_target = np.array([PLACE_XY[0], PLACE_XY[1], OBJ_SIZE / 2.0])

    print(f"[grasp] cube starts at {np.round(start_pos, 3)}")
    print(f"[grasp] place target   {np.round(place_target, 3)}")
    print("")

    max_z = float(start_pos[2])
    saved = 0
    done_step = None

    for step in range(MAX_STEPS):
        world.step(render=True)

        cube_pos, _ = cube.get_world_pose()
        max_z = max(max_z, float(cube_pos[2]))

        action = controller.forward(
            picking_position=cube_pos,
            placing_position=place_target,
            current_joint_positions=franka.get_joint_positions(),
            end_effector_offset=EE_OFFSET,
        )
        articulation_controller.apply_action(action)

        if step % FRAME_EVERY == 0:
            rgba = camera.get_rgba()
            if rgba is not None and rgba.size > 0:
                Image.fromarray(rgba[:, :, :3].astype(np.uint8)).save(
                    os.path.join(OUT_DIR, f"frame_{saved:04d}.png")
                )
                saved += 1

        if step % LOG_EVERY == 0:
            fingers = franka.gripper.get_joint_positions()
            ee_pos, _ = franka.end_effector.get_world_pose()
            gap = float(ee_pos[2] - cube_pos[2])
            print(
                f"[grasp] {step:4d} ph={controller.get_current_event()} "
                f"cube_z={cube_pos[2]:.3f} ee={np.round(ee_pos, 3)} "
                f"ee-cube_dz={gap:+.3f} fingers={np.round(fingers, 4)}"
            )

        if controller.is_done():
            done_step = step
            print(f"[grasp] controller done at step {step}")
            break

    end_pos, _ = cube.get_world_pose()
    err = float(np.linalg.norm(end_pos[:2] - place_target[:2]))
    lifted = max_z - float(start_pos[2])

    print("")
    print(f"[grasp] cube ended at   {np.round(end_pos, 3)}")
    print(f"[grasp] target was      {np.round(place_target, 3)}")
    print(f"[grasp] max height      {max_z:.3f} m  (lift = {lifted:.3f} m)")
    print(f"[grasp] error vs target {err:.3f} m")

    lift_ok = lifted > LIFT_THRESHOLD
    place_ok = err < PLACE_TOLERANCE
    print(f"[grasp] lift  {'PASS' if lift_ok else 'FAIL'}")
    print(f"[grasp] place {'PASS' if place_ok else 'FAIL'}")
    print(f"[grasp] GRASP {'SUCCEEDED' if (lift_ok and place_ok) else 'FAILED'}")
    print(f"[grasp] saved {saved} frames to {OUT_DIR}")

    simulation_app.close()


if __name__ == "__main__":
    main()