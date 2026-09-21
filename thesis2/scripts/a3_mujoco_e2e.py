"""thesis2/scripts/a3_mujoco_e2e.py -- A3: end-to-end execution success in MuJoCo.

Every other number in this project scores a STRING against a string. This one
asks the only question a robotics thesis actually cares about: when the model
emits a command, does the arm do the right thing?

WHAT THE TOLERANCE MEANS HERE
-----------------------------
The brief named `MOVE_TOL = 0.02` as unreachable (best observed 0.029) and set
it to 0.05. That constant lives in sim/isaac_backend.py:73 -- the ISAAC path.
The MuJoCo path is sim/full_grasp_backend.py, and it has no such constant:

    def move_to(self, location):
        x, y = self._xy(location)
        s = self._reach([x, y, 0.47], 0.0)
        return {..., "ok": True, "steps": s}      # <-- ok is HARDCODED True

MuJoCo `move_to` never checks whether it arrived. So 0.05 is applied here as a
gate the backend does not have: after the motion, the final gripper XY error
against the resolved cell centre must be < MOVE_TOL. That is stricter than the
backend's own reporting, not looser.

`pick_place` already has a real success test of its own (verified lift, XY gap
< 6 cm, object still above the table at z > 0.35) and it is left alone -- the
0.05 ruling was about move_to. fetch/hold/inspect succeed iff the grasp
verified the object actually left the table.

sim/ IS NOT MODIFIED. The tolerance is enforced in this file.

FAIRNESS
--------
Execution outcome depends on (command_id, verb, target, location) and nothing
else, because the object's start pose is seeded from command_id alone. So every
arm meets identical physics on the same command, and each distinct tuple is
executed ONCE and cached -- five arms share most of their predictions.

FAILURE ATTRIBUTION -- the point of the exercise
------------------------------------------------
  SUCCESS            prediction == truth AND the arm completed it
  EXECUTION_FAILURE  prediction == truth but the arm could not do it
  WRONG_PREDICTION   prediction != truth (the language model's fault)
  INVALID_SYMBOL     prediction names a location/target the scene does not have

Run:  python thesis2/scripts/a3_mujoco_e2e.py
      (conda env `thesis` -- it has mujoco; thesis2gpu does not)
"""
import csv
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "sim"))
OUT = REPO / "thesis2" / "out"
csv.field_size_limit(2 ** 31 - 1)

MOVE_TOL = 0.05          # ruled by the user, 2026-09-21. See module docstring.
OBJ_Z = 0.425

import full_grasp_backend as F                                   # noqa: E402

VERBS = ["pick_place", "fetch", "hold", "move_to", "inspect", "stop", "reject"]
TARGETS = ["rubiks_cube", "bottle", "toy_car"]
ROWS, COLS = ["A", "B", "C"], ["1", "2", "3"]
CONTAINERS = ["box_1", "box_2", "box_3"]
RUNTIMES = ["max_reach", "nearest_free_cell"]
NEEDS = {"pick_place": ("target", "location"), "fetch": ("target",),
         "hold": ("target",), "move_to": ("location",), "inspect": ("target",),
         "stop": (), "reject": ()}
MOVING = {"pick_place", "fetch", "hold", "move_to", "inspect"}


def norm(d):
    """Canonical form for comparing a prediction to the truth."""
    if not isinstance(d, dict):
        return None
    v = d.get("action")
    o = {"action": v}
    if v in NEEDS:
        for f in NEEDS[v]:
            o[f] = d.get(f)
    mods = d.get("modifiers") or []
    o["modifiers"] = sorted(mods) if isinstance(mods, list) else []
    return o


def start_pose(cmd_id):
    """Object start pose, a pure function of command_id -> identical for all arms."""
    rng = np.random.default_rng(1000 + int(cmd_id))
    return [float(rng.uniform(1.28, 1.40)), float(rng.uniform(0.68, 0.82)), OBJ_Z]


def rd(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


# ---------------------------------------------------------------- arms
print("=" * 86)
print("A3 -- END-TO-END EXECUTION IN MUJOCO   (MOVE_TOL = %.2f m)" % MOVE_TOL)
print("=" * 86)

p45 = rd(OUT / "p45_predictions_held_out_v3.csv")
arms = {}
truth_of = {}

for r in p45:
    truth_of[r["command_id"]] = json.loads(r["truth_json"])

# --- arm: flat DistilBERT (the V5 rung-4 head), from the n=217 fresh pass
flat = {}
for r in p45:
    v = VERBS[int(r["flat_verb_idx"])]
    d = {"action": v}
    if "target" in NEEDS[v]:
        d["target"] = TARGETS[int(r["flat_target_idx"])]
    if "location" in NEEDS[v]:
        d["location"] = r["flat_location"]
    flat[r["command_id"]] = d
arms["flat_distilbert"] = flat

# --- arm: factorized DistilBERT
fac = {}
for r in rd(OUT / "factorized_fp16.csv"):
    fac[r["command_id"]] = json.loads(r["pred_json"])
arms["factorized"] = fac

# --- arm: P4.5 cascade, row escalated to qwen2.5-3b-lora (rebuilt, as in A6)
cas = {}
for r in p45:
    lt = int(r["fact_loc_type_idx"])
    row_i, col_i = int(r["fact_row_idx"]), int(r["fact_col_idx"])
    if lt == 0:
        e = r["salsa_qwen3blora_letters_row"]
        if e in ROWS:
            row_i = ROWS.index(e)
        sym = "grid_%s%s" % (ROWS[row_i], COLS[col_i])
    elif lt == 1:
        sym = CONTAINERS[int(r["fact_container_which_idx"])]
    else:
        sym = RUNTIMES[int(r["fact_runtime_which_idx"])]
    v = VERBS[int(r["fact_verb_idx"])]
    d = {"action": v}
    if "target" in NEEDS[v]:
        d["target"] = TARGETS[int(r["fact_target_idx"])]
    if "location" in NEEDS[v]:
        d["location"] = sym
    cas[r["command_id"]] = d
arms["cascade_row_qwen3blora"] = cas

# --- arm: BiLSTM
bil = {}
for r in rd(OUT / "a1_bilstm_held_out_v3.csv"):
    if str(r.get("is_warmup", "")).strip().lower() == "true":
        continue
    bil[r["command_id"]] = json.loads(r["pred_json"])
arms["bilstm"] = bil

# --- arm: ensemble-over-disagreement (n=216; command_id 0 was the V5 warmup)
ens = {}
for r in rd(OUT / "a1_ensemble_held_out_v3.csv"):
    ens[r["command_id"]] = json.loads(r["pred_json"])
arms["ensemble"] = ens

# --- arm: ORACLE -- execute the ground truth. Pure execution reliability.
arms["truth_oracle"] = {k: v for k, v in truth_of.items()}

print()
for name, preds in arms.items():
    agree = sum(1 for c, p in preds.items()
                if c in truth_of and norm(p) == norm(truth_of[c]))
    print("  %-24s n=%-4d string-level strict match %.4f"
          % (name, len(preds), agree / max(1, len(preds))))

# ---------------------------------------------------------------- execute
jobs = {}
for preds in arms.values():
    for cid, d in preds.items():
        if cid not in truth_of:
            continue
        v = (d or {}).get("action")
        key = (cid, v, (d or {}).get("target"), (d or {}).get("location"))
        jobs[key] = None
print()
print("  distinct (command, verb, target, location) tuples to execute: %d"
      % len(jobs))
print("  (against %d arm x command pairs -- the cache saves %d executions)"
      % (sum(len(p) for p in arms.values()),
         sum(len(p) for p in arms.values()) - len(jobs)))

be = F.FullGraspBackend(render=False)
print("  backend: FullGraspBackend, %d locations, grid=%s"
      % (len(F.LOCATIONS), F.HAS_GRID))
print()

t0 = time.time()
cache = {}
for i, key in enumerate(sorted(jobs, key=lambda k: (int(k[0]), str(k[1])))):
    cid, verb, tgt, loc = key
    rec = {"ok": False, "reason": "", "detail": ""}
    try:
        if verb in ("stop", "reject"):
            rec.update(ok=True, reason="no_motion")
        elif verb not in NEEDS:
            rec.update(ok=False, reason="INVALID_SYMBOL", detail="unknown verb %r" % verb)
        else:
            be.set_object(start_pose(cid))
            if verb == "move_to":
                r = be.move_to(loc)
                xy = np.array(F.LOCATIONS[be.last_resolved], float)
                err = float(np.linalg.norm(be._grip_pos()[:2] - xy))
                rec.update(ok=bool(err < MOVE_TOL), reason="",
                           detail="err_cm=%.1f tol_cm=%.1f resolved=%s"
                                  % (err * 100, MOVE_TOL * 100, be.last_resolved))
                if not rec["ok"]:
                    rec["reason"] = "move_tol_exceeded"
            elif verb == "pick_place":
                r = be.pick_place(tgt, loc)
                rec.update(ok=bool(r["ok"]),
                           detail="gap_cm=%s on_table=%s tries=%s"
                                  % (r["gap_cm"], r["on_table"], r["grasp_tries"]))
                if not rec["ok"]:
                    rec["reason"] = "grasp_or_place_failed"
            else:
                r = getattr(be, verb)(tgt)
                rec.update(ok=bool(r["ok"]),
                           detail="tries=%s" % r.get("grasp_tries"))
                if not rec["ok"]:
                    rec["reason"] = "grasp_failed"
    except KeyError as e:
        rec.update(ok=False, reason="INVALID_SYMBOL", detail=str(e)[:90])
    except Exception as e:                               # noqa: BLE001
        rec.update(ok=False, reason="EXCEPTION",
                   detail="%s: %s" % (type(e).__name__, str(e)[:70]))
    cache[key] = rec
    if (i + 1) % 100 == 0:
        print("    executed %4d / %d   (%.0fs)" % (i + 1, len(jobs), time.time() - t0))
be.close()
print("    executed %4d / %d   (%.0fs)  DONE" % (len(jobs), len(jobs), time.time() - t0))

# ---------------------------------------------------------------- score
rows, per_cmd = [], []
print()
print("=" * 86)
print("PER-ARM END-TO-END RESULT")
print("=" * 86)
print("  %-24s %5s %8s %8s %9s %9s %8s" %
      ("arm", "n", "e2e", "e2e_mov", "wrongpred", "execfail", "invalid"))
print("  " + "-" * 80)

for name, preds in arms.items():
    n = succ = wrong = efail = inval = 0
    nmov = smov = 0
    for cid, d in sorted(preds.items(), key=lambda kv: int(kv[0])):
        if cid not in truth_of:
            continue
        t = truth_of[cid]
        v = (d or {}).get("action")
        key = (cid, v, (d or {}).get("target"), (d or {}).get("location"))
        rec = cache[key]
        match = norm(d) == norm(t)
        if rec["reason"] in ("INVALID_SYMBOL", "EXCEPTION"):
            outcome = "INVALID_SYMBOL"
        elif not match:
            outcome = "WRONG_PREDICTION"
        elif not rec["ok"]:
            outcome = "EXECUTION_FAILURE"
        else:
            outcome = "SUCCESS"
        n += 1
        succ += outcome == "SUCCESS"
        wrong += outcome == "WRONG_PREDICTION"
        efail += outcome == "EXECUTION_FAILURE"
        inval += outcome == "INVALID_SYMBOL"
        tv = (t or {}).get("action")
        if tv in MOVING:
            nmov += 1
            smov += outcome == "SUCCESS"
        per_cmd.append(dict(arm=name, command_id=cid, truth_action=tv,
                            pred_action=v, string_match=match,
                            exec_ok=rec["ok"], outcome=outcome,
                            reason=rec["reason"], detail=rec["detail"]))
    rows.append(dict(arm=name, n=n, e2e_success=round(succ / n, 4),
                     n_moving=nmov,
                     e2e_success_moving=round(smov / nmov, 4) if nmov else "",
                     wrong_prediction=wrong, execution_failure=efail,
                     invalid_symbol=inval,
                     wrong_prediction_rate=round(wrong / n, 4),
                     execution_failure_rate=round(efail / n, 4)))
    print("  %-24s %5d %8.4f %8s %9d %9d %8d"
          % (name, n, succ / n,
             ("%.4f" % (smov / nmov)) if nmov else "n/a", wrong, efail, inval))

print()
print("  e2e      = correct command AND the arm completed it, all 7 verbs")
print("  e2e_mov  = same, restricted to commands whose TRUTH verb moves the arm")
print("             (stop/reject are free successes; this removes them)")

# ---------------------------------------------------------------- why did execution fail
print()
print("=" * 86)
print("EXECUTION FAILURES -- when the command was RIGHT but the arm could not do it")
print("=" * 86)
byreason = {}
for p in per_cmd:
    if p["outcome"] == "EXECUTION_FAILURE":
        byreason.setdefault((p["arm"], p["reason"]), []).append(p["command_id"])
if not byreason:
    print("  none -- every correctly predicted command executed successfully")
else:
    for (arm, reason), ids in sorted(byreason.items()):
        print("  %-24s %-24s %3d  e.g. command_id %s"
              % (arm, reason, len(ids), ", ".join(ids[:6])))

print()
print("  execution outcome by TRUTH verb (oracle arm -- pure execution reliability):")
ov = {}
for p in per_cmd:
    if p["arm"] != "truth_oracle":
        continue
    a, b = ov.setdefault(p["truth_action"], [0, 0])
    ov[p["truth_action"]] = [a + (p["outcome"] == "SUCCESS"), b + 1]
for v, (s, tot) in sorted(ov.items(), key=lambda kv: -kv[1][1]):
    print("    %-12s %3d/%-3d  %.4f" % (v, s, tot, s / tot))

# ---------------------------------------------------------------- write
p = OUT / "a3_mujoco_e2e.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
p2 = OUT / "a3_mujoco_e2e_percommand.csv"
with open(p2, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(per_cmd[0].keys()))
    w.writeheader()
    w.writerows(per_cmd)
(OUT / "a3_mujoco_e2e_summary.json").write_text(json.dumps(dict(
    move_tol_m=MOVE_TOL, backend="sim/full_grasp_backend.FullGraspBackend",
    env="FetchPickAndPlace-v4", dataset="held_out_v3",
    distinct_executions=len(jobs), arms=rows,
    note="MOVE_TOL enforced in thesis2/scripts/a3_mujoco_e2e.py; sim/ unmodified. "
         "MuJoCo move_to returns ok=True unconditionally, so this gate is "
         "stricter than the backend's own reporting."), indent=2), encoding="utf-8")
print()
print("wrote %s" % p.relative_to(REPO))
print("wrote %s" % p2.relative_to(REPO))
print("wrote thesis2/out/a3_mujoco_e2e_summary.json")
print("=" * 86)
