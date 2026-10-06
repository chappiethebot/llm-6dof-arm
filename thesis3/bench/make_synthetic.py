"""Synthetic campaign in exactly the SCHEMA.md layout, to test analyze.py and
verify_campaign.py without the real bench. Ground truth is printed for hand checks.

    python make_synthetic.py <out_dir>

Cases: GPU classifier (v3 + v4c), slow LLM (v3 + v4c), a CPU model fast enough to exceed
the 20 row-passes (latency-only passes), cloud with 5 repeats and failed repeats (so some
votes have no majority), a ceiling system, a run whose CPU sensor is missing (energy must
come out empty), a resume (idle window in an earlier telemetry session), warm-up rows,
model load and loaded-idle phases, an idle window reused by the system's v4c run."""
import gzip
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
root = sys.argv[1]
for d in ["calls", "latency", "passes", "outputs"]:
    os.makedirs(os.path.join(root, d), exist_ok=True)
N, ROW_PASSES = 217, 20
LOC = ([f"grid_{'ABC'[i % 3]}{1 + (i // 3) % 3}" for i in range(108)] + ["max_reach", "nearest_free_cell"] * 12
       + [f"box_{1 + i % 3}" for i in range(36)] + [None] * 49)          # 108 grid, 24 runtime, 36 container, 49 none
VERBS = ["move_to"] * 33 + ["pick_place"] * 135 + ["inspect"] * 10 + ["stop"] * 10 + ["reject"] * 12 + ["fetch"] * 9 + ["hold"] * 8
assert len(LOC) == N and len(VERBS) == N
ROW_WORD = {"held_out_v3": {"A": "upper", "B": "middle", "C": "bottom"},
            "held_out_v4c": {"A": "back", "B": "middle", "C": "front"}}
SHA = {s: hashlib.sha256(f"{s} synthetic".encode()).hexdigest() for s in ROW_WORD}
HZ, DT = 10, int(1e9 / 10)
IDLE_GPU, IDLE_CPU = 11.0, 9.0
UTC0 = 1_790_000_000_000_000_000
st = {"t": 0, "E": 0.0, "session": "s1", "idle_n": 0}
tel_rows, runs, idles, order = [], [], [], []


def emit(phase, run_id, seconds, gpu_w, cpu_w, vram):
    for _ in range(int(round(seconds * HZ))):
        g = gpu_w + rng.normal(0, 0.5)
        st["E"] += g * DT / 1e9 * 1000
        cpu = np.nan if cpu_w is None else round(cpu_w + rng.normal(0, 0.3), 1)
        tel_rows.append({"t_utc_ns": UTC0 + st["t"], "t_mono_ns": st["t"], "run_id": run_id, "phase": phase,
                         "gpu_power_w": g, "gpu_power_instant_w": g, "gpu_energy_mj": st["E"],
                         "gpu_util_pct": 90 if gpu_w > 20 else 0, "gpu_mem_util_pct": 20, "vram_used_mb": vram,
                         "gpu_temp_c": 55, "gpu_sm_clock_mhz": 2500, "gpu_mem_clock_mhz": 9000, "gpu_pstate": 0,
                         "gpu_fan_pct": 40, "cpu_pkg_power_w": cpu, "cpu_util_pct": 20, "proc_cpu_s": st["t"] / 1e9 * 0.5,
                         "proc_rss_mb": 900, "sys_ram_used_mb": 9000, "late_ms": 0.1, "_session": st["session"]})
        st["t"] += DT


def call(run_id, split, p, idx, cid, r, d, acc, llm, failed, uses_cloud, conf_ok):
    lt = LOC[cid]
    grid = lt is not None and lt.startswith("grid_")
    ok = (rng.random() < acc) and not failed
    pred = None if failed else {"verb": VERBS[cid], "loc": lt if ok else ("grid_B2" if lt else None)}
    ph = hashlib.sha1(json.dumps(pred, sort_keys=True).encode()).hexdigest()[:12]
    row = {"run_id": run_id, "pass": p, "order_idx": idx, "cmd_id": cid, "repeat": r,
           "t_start_mono_ns": None, "t_end_mono_ns": None, "t_infer_start_mono_ns": None, "t_infer_end_mono_ns": None,
           "latency_ms": d, "valid_json": 0 if failed else 1, "action_ok": int(ok), "verb_ok": 0 if failed else 1,
           "target_ok": 1 if VERBS[cid] not in ("stop", "reject", "move_to") else None,
           "loc_ok": (int(ok) if lt else None), "verb_true": VERBS[cid], "verb_pred": None if failed else VERBS[cid],
           "loc_true": lt, "loc_pred": None if failed else pred["loc"],
           "row_true": lt[5] if grid else None, "row_pred": (pred["loc"][5] if pred["loc"] else None) if grid and not failed else None,
           "col_true": lt[6] if grid else None, "col_pred": (pred["loc"][6] if pred["loc"] else None) if grid and not failed else None,
           "row_word": ROW_WORD[split][lt[5]] if grid else None, "pred_hash": None if failed else ph,
           "confidence": float(np.clip(0.6 + 0.4 * ok - rng.random() * 0.2, 0, 1)) if conf_ok else None,
           "tokens_in": 460 if llm else None, "tokens_out": 28 if llm else None, "tokens_thinking": 0 if llm else None,
           "ttft_ms": 60 if llm else None, "error": "timeout" if failed else None,
           "extra": json.dumps({"retries": 1} if failed else ({"escalated": int(rng.random() < 0.2)} if uses_cloud else {}))}
    return row, {"run_id": run_id, "pass": p, "cmd_id": cid, "repeat": r, "raw": json.dumps(pred), "parsed": pred}


def lay(rows):
    tt = st["t"]
    for c in rows:
        c["t_start_mono_ns"] = int(tt)
        c["t_infer_start_mono_ns"] = int(tt + 0.05 * c["latency_ms"] * 1e6)
        c["t_infer_end_mono_ns"] = int(tt + 0.95 * c["latency_ms"] * 1e6)
        tt += c["latency_ms"] * 1e6
        c["t_end_mono_ns"] = int(tt)
    return (tt - st["t"]) / 1e9


def take_idle(seconds=30):
    st["idle_n"] += 1
    iid = f"idle_{st['session']}_{st['idle_n']:04d}"
    a = st["t"]
    emit("idle", iid, seconds, IDLE_GPU, IDLE_CPU, 1400)
    idles.append({"idle_id": iid, "session": st["session"], "t_start_mono_ns": a, "t_end_mono_ns": st["t"] - DT,
                  "t_start_utc_ns": UTC0 + a, "t_end_utc_ns": UTC0 + st["t"] - DT})
    return iid


def one_run(run_id, iid, split, system, family, kind, device, legend, lat_ms, gpu_w, cpu_w, acc,
            repeats=1, min_s=20, uses_cloud=0, drop_repeats=0, model_load_s=None):
    llm, conf_ok = "lora" in system or "gpt" in system, legend == "none"
    warm, outs = [], []
    for i in range(3):
        r, o = call(run_id, split, 0, i, i, 1, lat_ms * 3 if i == 0 else lat_ms, acc, llm, False, uses_cloud, conf_ok)
        warm.append(r); outs.append(o)
    emit("warmup", run_id, lay(warm), gpu_w, cpu_w, 8000 if gpu_w > 20 else 1500)
    t0 = st["t"]
    rows, lats, passes, p, elapsed, idx = [], [], [], 0, 0.0, 0
    first_hash = {}
    while elapsed < min_s or p < 1:
        p += 1
        pr = []
        for cid in rng.permutation(N):
            for rep in range(1, repeats + 1):
                failed = p == 1 and rep >= repeats - 1 and repeats > 1 and cid < drop_repeats
                d = max(lat_ms + rng.normal(0, lat_ms * 0.05), 0.05)
                elapsed += d / 1000
                r, o = call(run_id, split, p, idx, int(cid), rep, d, acc, llm, failed, uses_cloud, conf_ok)
                pr.append(r); idx += 1
                if p == 1:
                    first_hash[(int(cid), rep)] = r["pred_hash"]; outs.append(o)
                elif p <= ROW_PASSES and r["pred_hash"] != first_hash.get((int(cid), rep)):
                    outs.append(o)
        rows += pr if p <= ROW_PASSES else []
        lats += [x["latency_ms"] for x in pr]
        passes.append(pr)
    # lay every call back to back (also the latency-only passes)
    allc = [x for pr in passes for x in pr]
    lay(allc)
    pass_rows = [{"pass": i + 1, "t_start_mono_ns": pr[0]["t_start_mono_ns"], "t_end_mono_ns": pr[-1]["t_end_mono_ns"],
                  "calls": len(pr), "action_ok_sum": int(sum(x["action_ok"] for x in pr))} for i, pr in enumerate(passes)]
    emit("load", run_id, (allc[-1]["t_end_mono_ns"] - t0) / 1e9 + 0.05, gpu_w, cpu_w, 8000 if gpu_w > 20 else 1500)
    t1 = allc[-1]["t_end_mono_ns"]
    emit("cooldown", run_id, 5, IDLE_GPU, IDLE_CPU, 1400)
    # write order: outputs, calls, latency, passes, then the runs.csv row (commit marker)
    with open(os.path.join(root, "outputs", f"{run_id}.jsonl"), "w", encoding="utf-8") as f:
        for o in outs:
            f.write(json.dumps(o, default=str) + "\n")
    pd.DataFrame(warm + rows).to_csv(os.path.join(root, "calls", f"{run_id}.csv.gz"), index=False, compression="gzip")
    np.save(os.path.join(root, "latency", f"{run_id}.npy"), np.asarray(lats, dtype=np.float32))
    pd.DataFrame(pass_rows).to_csv(os.path.join(root, "passes", f"{run_id}.csv"), index=False)
    runs.append({"run_id": run_id, "session": st["session"], "system": system, "family": family, "split": split, "kind": kind,
                 "device": device, "runtime": "x", "legend": legend, "dataset_sha256": SHA[split], "idle_id": iid,
                 "t_load_start_mono_ns": t0, "t_load_end_mono_ns": t1, "t_load_start_utc_ns": UTC0 + t0,
                 "t_load_end_utc_ns": UTC0 + t1, "passes": p, "row_passes": min(p, ROW_PASSES), "calls_total": len(lats),
                 "repeats": repeats, "model_sha256": "n/a", "model_load_s": model_load_s, "uses_cloud": uses_cloud,
                 "gpu_temp_start_c": 45, "notes": ""})
    order.append({"system": system, "split": split})
    n = len(lats) / repeats
    if cpu_w is None:
        print(f"{run_id} {split}: CPU sensor missing -> expect energy_complete = False")
    else:
        print(f"{run_id} {system} {split}: truth net W {gpu_w - IDLE_GPU + cpu_w - IDLE_CPU:.1f}, window {(t1 - t0)/1e9:.1f} s, "
              f"{len(lats)} calls, {p} passes -> net Wh/1k commands ~ "
              f"{((gpu_w - IDLE_GPU) + (cpu_w - IDLE_CPU)) * (t1 - t0)/1e9/3600/n*1000:.4f}")


def system(prefix, name, family, kind, device, legend, lat_ms, gpu_w, cpu_w, acc, splits=("held_out_v3",),
           session=None, **kw):
    iid = take_idle()
    if session:
        st["session"] = session                         # a resume: the system runs in a new telemetry file
    if kind == "local":
        emit("modelload", f"{prefix}a", 4, 40.0, 35.0, 5000)
        emit("loadedidle", f"{prefix}a", 6, IDLE_GPU + (3.0 if device == "gpu" else 0.0), IDLE_CPU + 0.5, 5000)
    for i, sp in enumerate(splits):
        one_run(f"{prefix}{'ab'[i]}", iid, sp, name, family, kind, device, legend, lat_ms, gpu_w, cpu_w, acc,
                model_load_s=4.0 if (kind == "local" and i == 0) else None, **kw)


system("r1", "flat_gpu", "classifier", "local", "gpu", "none", 3.0, 60.0, 20.0, 0.75, splits=("held_out_v3", "held_out_v4c"))
system("r2", "lora_greedy", "llm_local", "local", "gpu", "v5", 900.0, 103.0, 25.0, 0.95, splits=("held_out_v3", "held_out_v4c"))
system("r3", "bilstm", "baseline", "local", "cpu", "none", 0.5, 11.0, 30.0, 0.70, session="s2")
system("r4", "v5:openai:gpt-4o-mini/rag", "v5_cell", "cloud", "cloud", "v5", 1200.0, 11.0, 10.0, 0.93, repeats=5,
       min_s=1, uses_cloud=1, drop_repeats=10)
system("r5", "cascade_oracle", "ceiling", "local", "gpu", "none", 300.0, 70.0, 22.0, 1.0)
system("r6", "ensemble", "ensemble", "local", "gpu", "v5", 300.0, 70.0, None, 0.9, uses_cloud=1)

tel = pd.DataFrame(tel_rows)
for sess, d in tel.groupby("_session"):
    d.drop(columns="_session").to_csv(os.path.join(root, f"telemetry_{sess}.csv"), index=False)
pd.DataFrame(idles).to_csv(os.path.join(root, "idle.csv"), index=False)
pd.DataFrame(runs).to_csv(os.path.join(root, "runs.csv"), index=False)
pd.DataFrame(order).to_csv(os.path.join(root, "run_order.csv"), index=False)
json.dump({"gpu_name": "SYNTHETIC", "datasets": SHA, "pandas": pd.__version__, "numpy": np.__version__},
          open(os.path.join(root, "machine.json"), "w"))
print("written", root)
