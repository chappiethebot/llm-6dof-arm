"""Which GPU energy number is right? Compares, per phase, the NVML energy counter with the
integral of 10 Hz average power, 10 Hz instantaneous power and nvidia-smi's own 100 ms log.

    D:\\condaenvs\\thesis3meter\\python.exe thesis3\\bench\\diag_energy.py

About 4 min: 60 s idle, 60 s steady matmul, 60 s pulsed matmul (100 ms on / 100 ms off),
30 s idle. The load runs in thesis2gpu (D:\\condaenvs\\thesis2gpu\\python.exe diag_load.py).
Afterwards the same three telemetry numbers are printed per phase of thesis3/out/smoke.
Writes thesis3/out/diag/ (telemetry, nvidia-smi log, diag_energy.csv). Changes nothing else.
"""
from __future__ import annotations

import csv
import datetime as dt
import glob
import math
import os
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import telemetry as T  # noqa: E402

GPU_PY = r"D:\condaenvs\thesis2gpu\python.exe"
OUT = os.path.join("thesis3", "out", "diag")
WH = 1 / 3600
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
FAKE = "--fake" in sys.argv          # test mode without a GPU (development only)
SCALE = 0.1 if "--quick" in sys.argv else 1.0   # development only


def integ(t_s, p):
    t_s, p = np.asarray(t_s, float), np.asarray(p, float)
    ok = ~np.isnan(p)
    return float(_trapz(p[ok], t_s[ok])) * WH if ok.sum() > 1 else float("nan")


def start_smi(path):
    """nvidia-smi logging every 100 ms; falls back to fewer columns on older drivers."""
    for q in ["timestamp,power.draw,power.draw.average,power.draw.instant", "timestamp,power.draw"]:
        f = open(path, "w")
        p = subprocess.Popen(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader,nounits", "-lms", "100"],
                             stdout=f, stderr=subprocess.STDOUT)
        time.sleep(1.5)
        if p.poll() is None:
            return p, q.split(",")[1:]
        f.close()
    return None, []


def read_smi(path, cols):
    rows = []
    for ln in open(path, encoding="utf-8", errors="replace"):
        parts = [x.strip() for x in ln.split(",")]
        if len(parts) != len(cols) + 1:
            continue
        try:
            t = dt.datetime.strptime(parts[0], "%Y/%m/%d %H:%M:%S.%f").timestamp()
            rows.append([t] + [float(x) if x not in ("[N/A]", "N/A") else float("nan") for x in parts[1:]])
        except ValueError:
            continue
    return np.array(rows) if rows else np.zeros((0, len(cols) + 1))


def table(name, tel_rows, smi, smi_cols, a_utc, b_utc):
    w = [r for r in tel_rows if a_utc <= int(r["t_utc_ns"]) <= b_utc]
    if len(w) < 3:
        return None
    t = np.array([int(r["t_mono_ns"]) for r in w]) / 1e9
    avg = np.array([float(r["gpu_power_w"] or "nan") for r in w])
    inst = np.array([float(r["gpu_power_instant_w"] or "nan") for r in w])
    ctr = np.array([float(r["gpu_energy_mj"] or "nan") for r in w])
    out = {"phase": name, "seconds": round(t[-1] - t[0], 1),
           "counter_Wh": (ctr[-1] - ctr[0]) / 1000 * WH if not math.isnan(ctr[0]) else float("nan"),
           "avg10Hz_Wh": integ(t, avg), "inst10Hz_Wh": integ(t, inst)}
    if len(smi):
        m = (smi[:, 0] >= a_utc / 1e9) & (smi[:, 0] <= b_utc / 1e9)
        for i, c in enumerate(smi_cols, 1):
            out[f"smi_{c.replace('power.draw', 'draw')}_Wh"] = integ(smi[m, 0], smi[m, i]) if m.sum() > 2 else float("nan")
    T_s = t[-1] - t[0]
    for k in [k for k in out if k.endswith("_Wh")]:
        out[k.replace("_Wh", "_meanW")] = out[k] / WH / T_s if T_s > 0 else float("nan")
    for k in [k for k in out if k.endswith("_Wh") and k != "counter_Wh"]:
        out[k.replace("_Wh", "_vs_counter_pct")] = 100 * (out[k] / out["counter_Wh"] - 1) if out["counter_Wh"] else float("nan")
    return out


def show(rows, title):
    print(f"\n== {title} ==")
    cols = ["phase", "seconds", "counter_meanW", "avg10Hz_meanW", "inst10Hz_meanW",
            "avg10Hz_vs_counter_pct", "inst10Hz_vs_counter_pct"] + \
           sorted({k for r in rows for k in r if k.startswith("smi_") and (k.endswith("_meanW") or k.endswith("_pct"))})
    print(" | ".join(cols))
    for r in rows:
        print(" | ".join(f"{r.get(c):.2f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols))


def main():
    os.makedirs(OUT, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    tel_path = os.path.join(OUT, f"telemetry_diag_{stamp}.csv")
    smi_path = os.path.join(OUT, f"nvidia_smi_{stamp}.csv")
    if FAKE:
        fake = T.FakeGpu.__new__(T.FakeGpu)
        tel = T.Telemetry(tel_path, gpu=fake, lhm_url=None)
        fake.__init__(tel)
        smi, smi_cols = None, []
    else:
        tel = T.Telemetry(tel_path)
        smi, smi_cols = start_smi(smi_path)
        if smi is None:
            print("nvidia-smi logging could not start; continuing with NVML only")
    tel.start()
    phases = []

    def phase(name, seconds, load=None):
        seconds = seconds * SCALE
        tel.set_phase(name, "diag")
        a = time.time_ns()
        if load and not FAKE:
            r = subprocess.run([GPU_PY, os.path.join(HERE, "diag_load.py"), load, str(seconds)])
            if r.returncode != 0:
                sys.exit(f"load generator failed (exit {r.returncode}); is {GPU_PY} the thesis2gpu python?")
        else:
            time.sleep(seconds)
        phases.append((name, a, time.time_ns()))
        print(f"{name}: done")

    print("about 4 min; do not touch the PC")
    phase("idle", 60)
    phase("load" if FAKE else "steady", 60, "steady")
    phase("pulsed", 60, "pulsed")
    phase("idle_end", 30)
    tel.stop()
    if smi is not None:
        smi.terminate()
        time.sleep(0.5)
    rows = list(csv.DictReader(open(tel_path, encoding="utf-8")))
    smi_data = read_smi(smi_path, smi_cols) if smi_cols else np.zeros((0, 1))
    res = [x for x in (table(n, rows, smi_data, smi_cols, a + int(3e9 * SCALE), b - int(1e9 * SCALE)) for n, a, b in phases) if x]
    show(res, "controlled test (first 3 s of each phase skipped)")

    smoke = []
    for f in sorted(glob.glob(os.path.join("thesis3", "out", "smoke", "telemetry_*.csv"))):
        srows = list(csv.DictReader(open(f, encoding="utf-8")))
        keys = []
        for r in srows:
            k = (r["run_id"], r["phase"])
            if r["phase"] in ("idle", "loadedidle", "warmup", "load") and (not keys or keys[-1] != k):
                keys.append(k)
        for rid, ph in keys:
            sel = [r for r in srows if r["run_id"] == rid and r["phase"] == ph]
            x = table(f"{rid}/{ph}", sel, np.zeros((0, 1)), [], int(sel[0]["t_utc_ns"]), int(sel[-1]["t_utc_ns"]))
            if x:
                smoke.append(x)
    if smoke:
        show(smoke, "smoke run, per phase")
    with open(os.path.join(OUT, "diag_energy.csv"), "w", newline="", encoding="utf-8") as f:
        allk = sorted({k for r in res + smoke for k in r})
        w = csv.DictWriter(f, fieldnames=allk)
        w.writeheader()
        w.writerows(res + smoke)
    print(f"\nwritten {OUT}\\diag_energy.csv")


if __name__ == "__main__":
    main()
