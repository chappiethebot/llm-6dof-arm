"""thesis2/final/s3_energy.py -- STAGE 3: measured latency and energy.

    python thesis2/final/s3_energy.py --idle
    python thesis2/final/s3_energy.py --arm <name> --split <split>

--idle
    nvidia-smi --query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu
    --format=csv -l 1  for 300 s. Writes the idle MEDIAN to
    thesis2/out/final/idle_baseline.json and the raw samples to power_logs/.
    GATE FAIL if mean idle utilisation > 2 %, printing the GPU process list so
    the offender is visible.

--arm --split
    Refuses to start if the last idle baseline is older than 2 hours.
    3 warm-up commands (not counted, and stated in the output), then the split
    is looped until >= 120 s of wall time. Records N, duration, per-command
    latency INCLUDING tokenisation, drafter, retrieval and network, p50/p95/p99,
    energy = (mean load W - idle median W) x wall s / N, max GPU temperature,
    then a 60 s cool-down.

    CPU arms also report process CPU ms per command via psutil; GPU watts are
    the only watts measured, and CPU watts are stated as not measured.
    Cloud arms: latency only, energy "not measurable".
    Ollama arms: OLLAMA_MODELS=D:\\LLM and V5's num_ctx; the Ollama version and
    model digest are recorded and other models are unloaded first.

    CASCADES ARE MEASURED PER COMMAND FOR REAL. The modelled constants in
    p45_cascade_full.csv are not read by this script and are used nowhere.

Appends to thesis2/out/final/unified_efficiency.csv.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
import arms as A                                                        # noqa: E402

IDLE_S = 300
LOAD_MIN_S = 120
WARMUP = 3
COOLDOWN_S = 60
IDLE_MAX_UTIL = 2.0
IDLE_MAX_AGE_S = 2 * 3600
IDLE_FILE = C.OUT / "idle_baseline.json"
EFF = C.OUT / "unified_efficiency.csv"
EFF_COLS = ["arm", "split", "kind", "utc", "n_commands", "wall_s", "warmup_excluded",
            "latency_p50_ms", "latency_p95_ms", "latency_p99_ms", "mean_latency_ms",
            "mean_power_w", "idle_median_w", "delta_power_w", "joules_per_command",
            "max_gpu_temp_c", "peak_vram_mb", "cpu_ms_per_command", "size_on_disk_mb",
            "ollama_version", "model_digest", "note"]


def nvidia_smi_query():
    try:
        o = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw,utilization.gpu,memory.used,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout
        p, u, m, t = [float(x.strip()) for x in o.strip().split(",")[:4]]
        return p, u, m, t
    except Exception:
        return None


class PowerLogger(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []
        self._halt = threading.Event()

    def run(self):
        while not self._halt.is_set():
            s = nvidia_smi_query()
            if s:
                self.samples.append((time.time(),) + s)
            time.sleep(1.0)

    def stop(self):
        self._halt.set()
        self.join(timeout=5)


def gpu_processes():
    try:
        return subprocess.run(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory",
                               "--format=csv"], capture_output=True, text=True,
                              timeout=5).stdout.strip()
    except Exception as e:
        return "nvidia-smi unavailable: %s" % e


def mode_idle():
    print("  sampling the idle GPU for %d s at 1 Hz ..." % IDLE_S)
    lg = PowerLogger()
    lg.start()
    time.sleep(IDLE_S)
    lg.stop()
    if not lg.samples:
        C.gate(False, ["nvidia-smi produced no samples", gpu_processes()])
    w = [s[1] for s in lg.samples]
    u = [s[2] for s in lg.samples]
    t = [s[4] for s in lg.samples]
    med, mean_u = statistics.median(w), statistics.mean(u)
    raw = C.POWER / ("idle_%s.csv" % time.strftime("%Y%m%d_%H%M%S"))
    with raw.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["timestamp", "power_w", "util_pct", "mem_used_mb", "temp_c"])
        wr.writerows(lg.samples)
    IDLE_FILE.write_text(json.dumps(dict(
        utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), epoch=time.time(),
        idle_median_w=round(med, 3), idle_mean_w=round(statistics.mean(w), 3),
        mean_util_pct=round(mean_u, 3), max_temp_c=max(t), n_samples=len(w),
        raw=str(raw.relative_to(C.REPO))), indent=2), encoding="utf-8")
    ok = mean_u <= IDLE_MAX_UTIL
    lines = ["idle median %.2f W (mean %.2f W), %d samples" % (med, statistics.mean(w), len(w)),
             "mean idle utilisation %.2f %% (limit %.1f %%)" % (mean_u, IDLE_MAX_UTIL),
             "max temp %.0f C" % max(t), "raw: %s" % raw.relative_to(C.REPO)]
    if not ok:
        lines += ["the GPU was NOT idle -- processes holding it:", gpu_processes()]
    C.gate(ok, lines)


def dir_size_mb(p: Path):
    if p is None or not p.exists():
        return ""
    if p.is_file():
        return round(p.stat().st_size / 1e6, 2)
    return round(sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6, 2)


def ollama_facts(arm_note):
    if "ollama" not in arm_note.lower():
        return "", ""
    try:
        v = subprocess.run(["ollama", "--version"], capture_output=True, text=True,
                           timeout=10).stdout.strip()
    except Exception as e:
        v = "unavailable: %s" % e
    try:
        ls = subprocess.run(["ollama", "list"], capture_output=True, text=True,
                            timeout=10).stdout.strip()
    except Exception as e:
        ls = "unavailable: %s" % e
    return v, ls[:400]


def mode_arm(name, split):
    arm = A.get(name)
    if not IDLE_FILE.exists():
        C.gate(False, ["no idle baseline. Run --idle first."])
    idle = json.loads(IDLE_FILE.read_text(encoding="utf-8"))
    age = time.time() - idle["epoch"]
    if age > IDLE_MAX_AGE_S:
        C.gate(False, ["idle baseline is %.1f h old (limit %.1f h). Re-run --idle."
                       % (age / 3600, IDLE_MAX_AGE_S / 3600)])
    print("  idle baseline: %.2f W median, %.1f min old" % (idle["idle_median_w"], age / 60))

    rows = C.load_split(split)
    C.require_frozen(name)
    predict = arm.build()

    print("  warm-up: %d commands, NOT counted in any figure below" % WARMUP)
    for r in rows[:WARMUP]:
        predict(r["command"])

    import psutil
    proc = psutil.Process()
    cpu0 = proc.cpu_times()
    lg = PowerLogger()
    lg.start()
    time.sleep(2)
    lat, n = [], 0
    t0 = time.time()
    while time.time() - t0 < LOAD_MIN_S:
        for r in rows:
            t1 = time.perf_counter()
            predict(r["command"])
            lat.append((time.perf_counter() - t1) * 1000)
            n += 1
        print("     %d commands, %.0f s elapsed" % (n, time.time() - t0))
    wall = time.time() - t0
    time.sleep(2)
    lg.stop()
    cpu1 = proc.cpu_times()
    cpu_ms = ((cpu1.user - cpu0.user) + (cpu1.system - cpu0.system)) * 1000 / max(n, 1)

    live = [s for s in lg.samples if s[0] >= t0]
    mean_w = statistics.mean([s[1] for s in live]) if live else float("nan")
    max_t = max([s[4] for s in live]) if live else float("nan")
    peak_vram = max([s[3] for s in live]) if live else ""
    cloud = arm.kind == "cloud"
    if cloud:
        j, note = C.NOT_MEASURABLE if hasattr(C, "NOT_MEASURABLE") else "not measurable", \
            "cloud arm: latency only; GPU energy is not the cost of this arm"
        delta = ""
    else:
        delta = mean_w - idle["idle_median_w"]
        j = round(delta * wall / n, 5)
        note = "GPU watts only; CPU watts are NOT measured"

    ver, digest = ollama_facts(arm.note)
    raw = C.POWER / ("%s_%s_%s.csv" % (name, split, time.strftime("%Y%m%d_%H%M%S")))
    with raw.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["timestamp", "power_w", "util_pct", "mem_used_mb", "temp_c"])
        wr.writerows(live)

    q = lambda p: sorted(lat)[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]
    rec = dict(arm=name, split=split, kind=arm.kind,
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               n_commands=n, wall_s=round(wall, 1), warmup_excluded=WARMUP,
               latency_p50_ms=round(q(.50), 3), latency_p95_ms=round(q(.95), 3),
               latency_p99_ms=round(q(.99), 3), mean_latency_ms=round(statistics.mean(lat), 3),
               mean_power_w=round(mean_w, 2), idle_median_w=idle["idle_median_w"],
               delta_power_w=round(delta, 2) if delta != "" else "",
               joules_per_command=j, max_gpu_temp_c=max_t, peak_vram_mb=peak_vram,
               cpu_ms_per_command=round(cpu_ms, 3), size_on_disk_mb="",
               ollama_version=ver, model_digest=digest, note=note)
    new = not EFF.exists()
    with EFF.open("a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=EFF_COLS)
        if new:
            w.writeheader()
        w.writerow(rec)

    print("  cool-down %d s ..." % COOLDOWN_S)
    time.sleep(COOLDOWN_S)
    C.gate(True, [
        "N=%d commands in %.1f s (>= %d s required); %d warm-up commands excluded"
        % (n, wall, LOAD_MIN_S, WARMUP),
        "latency p50/p95/p99 %.2f / %.2f / %.2f ms" % (q(.50), q(.95), q(.99)),
        ("energy %s J/command (mean %.1f W - idle %.2f W)" % (j, mean_w, idle["idle_median_w"]))
        if not cloud else "energy: not measurable (cloud arm)",
        "max GPU temp %.0f C; peak VRAM %s MB; CPU %.2f ms/command (CPU watts not measured)"
        % (max_t, peak_vram, cpu_ms),
        "appended to %s; raw samples %s" % (EFF.name, raw.name),
    ])


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--idle", action="store_true")
    g.add_argument("--arm")
    ap.add_argument("--split", choices=list(C.SPLITS))
    a = ap.parse_args()
    C.start_log("s3_energy_%s" % ("idle" if a.idle else "%s_%s" % (a.arm, a.split)))
    if a.idle:
        mode_idle()
    else:
        if not a.split:
            C.gate(False, ["--split is required with --arm"])
        mode_arm(a.arm, a.split)


if __name__ == "__main__":
    main()
