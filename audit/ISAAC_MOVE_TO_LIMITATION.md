# `move_to` on Isaac: control-budget expiry, not reach

Investigation record, 14 September 2026. **No fix applied.** This belongs in the
limitations section, not in this month's work. Demo path only — Isaac produces
no measured rows, so nothing here touches a reported number.

---

## The symptom

```
> move to max_reach
  {"verb":"move_to","location":"max_reach","ok":false,"steps":325,
   "error_cm":4.5,"best_cm":4.6,"stop":"descended",
   "failure":"unreachable","resolved":"cell_2_0"}
```

`max_reach` resolves to `cell_2_0` = **(0.680, −0.180)**, 0.7034 m from the
base. `move_to` targets **(0.680, −0.180, 0.250)**, 0.7465 m in 3D. The arm
plateaued around 4.5 cm and never arrived, against `MOVE_TOL = 0.02`.

Two hypotheses: the cell is outside the workspace, or the motion is being cut
short.

## Reach is not the cause — settled by test

```
> put the rubiks cube in box 3
  error_cm: 0.2
```

`box_3` is at planar **0.7718 m**, 3D **0.7738 m** — **further than `cell_2_0`'s
hover point at 0.7465 m**, and further than `cell_2_0` itself at 0.7034 m. The
arm reaches it and places to 2 mm.

So the workspace comfortably contains where `max_reach` sends the arm.
`pick_place` gets there; `move_to` does not. The difference is not geometry.

## The cause: a pick-and-place state machine driving a bare move

`sim/isaac_backend.py:586-605`:

```python
hover = np.array([target[0], target[1], MOVE_HOVER_Z])
controller = PickPlaceController(...)
...
action = controller.forward(
    picking_position=hover, placing_position=hover,   # <-- identical
    ...)
```

`move_to` drives `PickPlaceController` with **`picking_position == placing_position`**.
It is asking a grasp state machine to perform a move by telling it to pick and
place at the same point.

That controller runs a fixed schedule from its own `events_dt`, phase by phase,
on a step budget that does not depend on how far the arm has to travel. The file
already records this for the earlier version of the loop
(`sim/isaac_backend.py:560-563`): phase 0 ran a fixed number of steps, which is
why every call used to return exactly 125 steps regardless of target.

The loop breaks when `controller.get_current_event() >= 2` — the gripper-close
phase — because continuing would close the gripper on empty air. At 325 steps
that is what happened: phases 0 and 1 were spent and the controller moved on.

**The arm was still converging when it was cut off.** `stalled` had not reached
`MOVE_STALL_STEPS = 150`, so the error had improved within the preceding 150
steps. A genuine reach limit would have stalled first and reported `stalled`,
not `descended`.

## Consequence for the failure taxonomy

`descended` was mapped to `FAIL_UNREACHABLE`. That was wrong and is now
`FAIL_CONTROL_BUDGET` (`"control_budget"`), kept distinct from `FAIL_TIMEOUT`
because the two have different fixes:

| label | meaning | fix |
|---|---|---|
| `timeout` | our loop hit `MOVE_STEP_CAP` / `STEP_LIMIT` | raise the cap |
| `control_budget` | the controller's schedule expired first; our cap was never reached | raising the cap changes nothing — stop using a grasp controller for a move |
| `unreachable` | error stopped improving for 150 steps | the pose really is out of range |

## A second consequence: `MOVE_TOL` is unreachable for the wrong reason

`MOVE_TOL = 0.02` and the best observed error is ~0.029, so `move_to` reports
`ok: false` on every call. That has been read as a tolerance that is too tight.
On this evidence it is not: the arm is not being given enough of the
controller's schedule to close the last centimetre. **Raising `MOVE_TOL` would
paper over a control problem with a reporting change.** Left alone deliberately.

## Not investigated

- Whether a longer `events_dt` phase 0, or `RMPFlow` / a direct IK target,
  converges inside `MOVE_TOL`. Either is a real fix; neither was attempted.
- Whether `pick_place`'s accuracy holds across the whole grid or only near the
  boxes. Only `box_1` and `box_3` have been measured.
- The same question on the MuJoCo backend. `sim/sim_backend.py` uses a
  proportional controller, not a phase schedule, so it is unlikely to share
  this — but it has not been checked.

## Related, and separate

The Isaac grid has **never been reachability-swept**. `sim/reachability.py`
imports `full_grasp_backend` and measures MuJoCo only; `data/reachability.json`
is the Fetch arm in its own base frame and does not transfer. The Isaac grid
geometry (`GRID_CENTRE_X = 0.50`, `CELL = 0.18`) is hand-chosen, and no reach
figure is recorded anywhere in the repo.

That did not cause this fault — the box_3 test rules it out — but `max_reach`
still means "the furthest cell of a grid nobody swept", and it should be
described that way rather than as a measured workspace bound. No document
currently overstates it: `docs/DEMO_PORTABLE.md:272`, `docs/SESSION_2026-09-07.md:233`
and `map.md:151` all state that Isaac produces no measured rows.
