import numpy as np
import gymnasium as gym
import gymnasium_robotics

gym.register_envs(gymnasium_robotics)

Z_TABLE = 0.43   # gripper height to touch an object on the table
Z_CARRY = 0.55   # gripper height while moving / crossing over obstacles

COORDS = {
    "table_center": (1.34, 0.75, Z_TABLE),
    "table_left":   (1.34, 0.63, Z_TABLE),
    "table_right":  (1.34, 0.87, Z_TABLE),
    "table_front":  (1.22, 0.75, Z_TABLE),
    "table_back":   (1.46, 0.75, Z_TABLE),
    "box_1":        (1.24, 0.65, Z_TABLE),
    "box_2":        (1.44, 0.65, Z_TABLE),
    "box_3":        (1.44, 0.85, Z_TABLE),
    "rubiks_cube":  (1.30, 0.72, Z_TABLE),
    "bottle":       (1.34, 0.80, Z_TABLE),
    "toy_car":      (1.38, 0.70, Z_TABLE),
}

HANDOVER = np.array([1.22, 0.75, 0.50])   # fixed presentation point near the near edge


def _seg_point_dist_2d(a, b, p):
    """Shortest horizontal (x, y) distance from point p to the segment a->b."""
    a, b, p = a[:2], b[:2], p[:2]
    ab = b - a
    t = 0.0 if np.dot(ab, ab) < 1e-12 else np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return float(np.linalg.norm(a + t * ab - p))


class SimBackend:
    """Wraps FetchPickAndPlace-v4 and exposes the seven schema verbs on a
    proportional controller. Tabletop obstacles are cleared by lifting over
    them. Objects are coordinate-defined: verbs run the full motion, but the
    sim's single block is not physically carried (see thesis notes)."""

    def __init__(self, render=False, gain=10.0, step_limit=200):
        self.env = gym.make("FetchPickAndPlace-v4", render_mode=("human" if render else None))
        self.gain = gain
        self.step_limit = step_limit
        self.obstacles = []                 # list of (center_xyz, radius_m)
        obs, _ = self.env.reset()
        self.obs = obs

    # ---- low-level control -------------------------------------------------
    def ee_pos(self):
        # First 3 numbers of the observation are the gripper's (x, y, z).
        return self.obs["observation"][:3].copy()

    def resolve(self, name):
        """Schema name -> (x, y, z). Raises KeyError on an unknown name."""
        if name not in COORDS:
            raise KeyError(f"unknown schema name: {name!r}")
        return np.array(COORDS[name], dtype=float)

    def add_obstacle(self, center, radius):
        """Register an obstacle as a centre point and radius, in metres."""
        self.obstacles.append((np.asarray(center, float), float(radius)))

    def _step(self, action):
        self.obs, _, _, _, _ = self.env.step(np.clip(action, -1.0, 1.0))

    def move_ee_to(self, target, gripper=0.0, tol=0.02):
        """Straight proportional reach to `target` (x, y, z).
        gripper: +1 open, -1 close, 0 neutral.
        Returns (reached, steps, final_error_m)."""
        target = np.asarray(target, float)
        for i in range(self.step_limit):
            err = target - self.ee_pos()
            if np.linalg.norm(err) < tol:
                return True, i, float(np.linalg.norm(err))
            self._step(np.concatenate([self.gain * err, [gripper]]))
        return False, self.step_limit, float(np.linalg.norm(target - self.ee_pos()))

    def _path_blocked(self, start, target, margin=0.03):
        for c, r in self.obstacles:
            if _seg_point_dist_2d(start, target, c) < r + margin:
                return True
        return False

    def move_ee_to_safe(self, target, gripper=0.0, tol=0.02):
        """Reach `target`. If the straight path would clip an obstacle, detour
        over the top (lift, cross, descend). Otherwise go straight.
        Returns (reached, total_steps, final_error_m)."""
        target = np.asarray(target, float)
        start = self.ee_pos()
        if not self.obstacles or not self._path_blocked(start, target):
            return self.move_ee_to(target, gripper, tol)
        total, err = 0, None
        for wp in (np.array([start[0], start[1], Z_CARRY]),
                   np.array([target[0], target[1], Z_CARRY]),
                   target):
            ok, steps, err = self.move_ee_to(wp, gripper, tol)
            total += steps
            if not ok:
                return False, total, err
        return True, total, err

    def _actuate_gripper(self, cmd, hold=12):
        """Hold position and open (+1) or close (-1) the gripper for a few steps."""
        for _ in range(hold):
            self._step(np.array([0.0, 0.0, 0.0, cmd]))
        return hold

    def _grasp_and_lift(self, target):
        """Shared motion: approach, descend, grasp, lift to carry height.
        Returns (steps, ok, final_error_m)."""
        src = self.resolve(target)               # KeyError on invalid name
        up_src = np.array([src[0], src[1], Z_CARRY])
        steps, ok_all, err = 0, True, 0.0
        for wp, g in [(up_src, 1.0), (src, 1.0)]:
            ok, s, err = self.move_ee_to(wp, gripper=g); steps += s; ok_all &= ok
        steps += self._actuate_gripper(-1.0)     # grasp
        ok, s, err = self.move_ee_to(up_src, gripper=-1.0); steps += s; ok_all &= ok
        return steps, ok_all, err

    # ---- the seven schema verbs -------------------------------------------
    def move_to(self, location, gripper=0.0):
        """Send the gripper to a named location (obstacle-aware)."""
        ok, steps, err = self.move_ee_to_safe(self.resolve(location), gripper=gripper)
        return {"verb": "move_to", "location": location,
                "ok": ok, "steps": steps, "error_cm": round(err * 100, 1)}

    def pick_place(self, target, location):
        """Grasp at `target`, carry over, release at `location`."""
        src = self.resolve(target)
        dst = self.resolve(location)
        up_src = np.array([src[0], src[1], Z_CARRY])
        up_dst = np.array([dst[0], dst[1], Z_CARRY])
        steps, ok_all, err = 0, True, 0.0
        for wp, g in [(up_src, 1.0), (src, 1.0)]:
            ok, s, err = self.move_ee_to(wp, gripper=g); steps += s; ok_all &= ok
        steps += self._actuate_gripper(-1.0)     # grasp
        for wp in (up_src, up_dst, dst):
            ok, s, err = self.move_ee_to(wp, gripper=-1.0); steps += s; ok_all &= ok
        steps += self._actuate_gripper(1.0)      # release
        return {"verb": "pick_place", "target": target, "location": location,
                "ok": bool(ok_all), "steps": steps, "error_cm": round(err * 100, 1)}

    def fetch(self, target):
        """Grasp `target` and carry it to the fixed handover point, then release."""
        src = self.resolve(target)
        up_src = np.array([src[0], src[1], Z_CARRY])
        up_hand = np.array([HANDOVER[0], HANDOVER[1], Z_CARRY])
        steps, ok_all, err = 0, True, 0.0
        for wp, g in [(up_src, 1.0), (src, 1.0)]:
            ok, s, err = self.move_ee_to(wp, gripper=g); steps += s; ok_all &= ok
        steps += self._actuate_gripper(-1.0)     # grasp
        for wp in (up_src, up_hand, HANDOVER):
            ok, s, err = self.move_ee_to(wp, gripper=-1.0); steps += s; ok_all &= ok
        steps += self._actuate_gripper(1.0)      # release to hand
        return {"verb": "fetch", "target": target,
                "ok": bool(ok_all), "steps": steps, "error_cm": round(err * 100, 1)}

    def hold(self, target):
        """Grasp `target`, lift, and keep holding it steady."""
        steps, ok_all, err = self._grasp_and_lift(target)
        steps += self._actuate_gripper(-1.0, hold=15)   # keep holding
        return {"verb": "hold", "target": target,
                "ok": bool(ok_all), "steps": steps, "error_cm": round(err * 100, 1)}

    def inspect(self, target):
        """Grasp `target`, lift, and hold steady for a longer pause.
        NOTE: FetchPickAndPlace has no wrist rotation, so the 'rotate slowly'
        part of inspect is a RealBackend-only behavior; in sim it is a pause."""
        steps, ok_all, err = self._grasp_and_lift(target)
        steps += self._actuate_gripper(-1.0, hold=30)   # 'slow inspect' = longer pause
        return {"verb": "inspect", "target": target,
                "ok": bool(ok_all), "steps": steps, "error_cm": round(err * 100, 1)}

    def stop(self):
        """Halt. No motion in this synchronous model; nothing to interrupt."""
        return {"verb": "stop", "ok": True, "steps": 0, "error_cm": 0.0}

    def reject(self):
        """Out-of-scope request acknowledged; the arm does not move."""
        return {"verb": "reject", "ok": True, "steps": 0, "error_cm": 0.0}

    def close(self):
        self.env.close()


if __name__ == "__main__":
    be = SimBackend(render=False)
    print(be.move_to("table_left"))
    print(be.pick_place("rubiks_cube", "box_1"))
    print(be.fetch("toy_car"))
    print(be.hold("bottle"))
    print(be.inspect("rubiks_cube"))
    print(be.stop())
    print(be.reject())
    try:
        be.hold("dragon")
    except KeyError as e:
        print("invalid target ->", e)
    be.close()