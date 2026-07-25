"""
Step 1 — reachability sweep.

Sweeps candidate (x, y) placement targets across the table and records, per
candidate, whether the arm can (a) reach the point empty and (b) complete a
full pick_place to it with a real object in the gripper.

Output: JSON to --out, consumed by schema/grid_spec.json (geometry +
reachability) and later by sim/workspace.py.

Design notes:
  * Imports the REAL FullGraspBackend. Reach/grasp logic is never
    reimplemented here, or the sweep would measure a different arm than the
    measurement harness uses.
  * FullGraspBackend.pick_place looks its target up in the module-level
    LOCATIONS dict, so arbitrary coordinates are injected under a scratch key
    and removed afterwards. The backend source is not modified.
  * Table extent is read from the MuJoCo model, not hardcoded.

Usage:
    python sim/reachability.py --n 9 --trials 3 --out data/reachability.json
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

import full_grasp_backend as fgb
from full_grasp_backend import FullGraspBackend

SCRATCH = "__sweep_target__"
PLACE_Z = 0.45          # matches FullGraspBackend.pick_place release height
GAP_TOL = 0.06          # matches FullGraspBackend.pick_place ok criterion


def table_extent(backend):
    """Read the table0 geom's world-frame extent from the model."""
    m = backend.u.model
    bid = mujoco.mj_id2name  # noqa: F841  (kept for symmetry/readability)
    for i in range(m.ngeom):
        b = m.geom_bodyid[i]
        if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) == "table0":
            wp = m.body_pos[b] + m.geom_pos[i]
            hs = m.geom_size[i]
            return {
                "x_min": float(wp[0] - hs[0]), "x_max": float(wp[0] + hs[0]),
                "y_min": float(wp[1] - hs[1]), "y_max": float(wp[1] + hs[1]),
                "z_top": float(wp[2] + hs[2]),
            }
    raise RuntimeError("table0 geom not found in model")


def probe_reach(be, x, y, z):
    """Empty-gripper reach. Returns (converged, steps)."""
    steps = be._reach([x, y, z], 0.0)
    return steps < be.step_limit, steps


def probe_place(be, x, y, pickup_xy, z_rest, trials):
    """Full loaded pick_place to (x, y). Returns aggregate dict."""
    fgb.LOCATIONS[SCRATCH] = (x, y)
    oks, gaps, tries, reasons = [], [], [], []
    try:
        for _ in range(trials):
            be.set_object([pickup_xy[0], pickup_xy[1], z_rest])
            r = be.pick_place("rubiks_cube", SCRATCH)
            oks.append(bool(r["ok"]))
            gaps.append(float(r["gap_cm"]))
            tries.append(int(r["grasp_tries"]))
            if r["ok"]:
                reasons.append("ok")
            elif r["grasp_tries"] >= 3 and r["gap_cm"] > GAP_TOL * 100:
                reasons.append("grasp_failed")
            else:
                reasons.append("gap_too_large")
    finally:
        fgb.LOCATIONS.pop(SCRATCH, None)
    return {
        "place_ok_rate": sum(oks) / len(oks),
        "gap_cm_mean": round(float(np.mean(gaps)), 2),
        "gap_cm_max": round(float(np.max(gaps)), 2),
        "grasp_tries_mean": round(float(np.mean(tries)), 2),
        "reasons": reasons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=9, help="samples per axis (n x n candidates)")
    ap.add_argument("--trials", type=int, default=3, help="pick_place repeats per candidate")
    ap.add_argument("--margin", type=float, default=0.03,
                    help="inset from the table edge, metres, so targets are not on the lip")
    ap.add_argument("--pickup", type=float, nargs=2, default=[1.34, 0.75],
                    help="fixed object spawn point, so only the PLACE target varies")
    ap.add_argument("--out", default="data/reachability.json")
    args = ap.parse_args()

    be = FullGraspBackend(render=False)
    ext = table_extent(be)
    z_rest = ext["z_top"] + 0.025          # object half-height
    xs = np.linspace(ext["x_min"] + args.margin, ext["x_max"] - args.margin, args.n)
    ys = np.linspace(ext["y_min"] + args.margin, ext["y_max"] - args.margin, args.n)

    print(f"table extent: {ext}")
    print(f"sweeping {args.n}x{args.n} = {args.n**2} candidates, {args.trials} trials each")

    t0 = time.time()
    cells = []
    for x in xs:
        for y in ys:
            reached, steps = probe_reach(be, x, y, PLACE_Z)
            rec = {"x": round(float(x), 4), "y": round(float(y), 4),
                   "reach_ok": bool(reached), "reach_steps": int(steps)}
            rec.update(probe_place(be, x, y, args.pickup, z_rest, args.trials))
            rec["legal"] = bool(rec["reach_ok"] and rec["place_ok_rate"] == 1.0)
            cells.append(rec)
        print(f"  x={x:.3f} done ({time.time()-t0:.1f}s)")
    be.close()

    legal = [c for c in cells if c["legal"]]
    result = {
        "generated_by": "sim/reachability.py",
        "env": "FetchPickAndPlace-v4",
        "backend": "FullGraspBackend",
        "params": {"n": args.n, "trials": args.trials, "margin": args.margin,
                   "pickup_xy": args.pickup, "place_z": PLACE_Z, "gap_tol_m": GAP_TOL},
        "table_extent": ext,
        "summary": {
            "candidates": len(cells),
            "legal": len(legal),
            "legal_fraction": round(len(legal) / len(cells), 3),
            "legal_x_span_m": round(max(c["x"] for c in legal) - min(c["x"] for c in legal), 4) if legal else None,
            "legal_y_span_m": round(max(c["y"] for c in legal) - min(c["y"] for c in legal), 4) if legal else None,
            "wall_time_s": round(time.time() - t0, 1),
        },
        "cells": cells,
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

    s = result["summary"]
    print(f"\nwrote {out}")
    print(f"legal {s['legal']}/{s['candidates']} ({s['legal_fraction']:.1%})")
    print(f"legal span: x {s['legal_x_span_m']} m | y {s['legal_y_span_m']} m")
    print(f"max cells at 0.06 m pitch: "
          f"{int(s['legal_x_span_m']//0.06)+1} x {int(s['legal_y_span_m']//0.06)+1}"
          if legal else "no legal cells")


if __name__ == "__main__":
    main()
