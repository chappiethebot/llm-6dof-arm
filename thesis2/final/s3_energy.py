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

    SPECULATIVE ARMS (a8_speculative, a10_speculative, a12_smalllm) run their
    own script with --energy, which calls measure_loop below in-process after
    its byte-identity gate against lora_greedy passes. Same protocol, same
    columns, one row per arm and split.

    CASCADES ARE MEASURED PER COMMAND FOR REAL. The modelled constants in
    p45_cascade_full.csv are not read by this script and are used nowhere.

Appends to thesis2/out/final/unified_efficiency.csv.
"""
import argparse
import csv
import json
import os
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


def require_idle():
    """The idle baseline, refused if missing or older than the limit."""
    if not IDLE_FILE.exists():
        C.gate(False, ["no idle baseline. Run --idle first."])
    idle = json.loads(IDLE_FILE.read_text(encoding="utf-8"))
    age = time.time() - idle["epoch"]
    if age > IDLE_MAX_AGE_S:
        C.gate(False, ["idle baseline is %.1f h old (limit %.1f h). Re-run --idle."
                       % (age / 3600, IDLE_MAX_AGE_S / 3600)])
    print("  idle baseline: %.2f W median, %.1f min old"
          % (idle["idle_median_w"], age / 60))
    return idle


def measure_loop(name, split, predict, idle, kind="local", note="", ollama_note=""):
    """3 warm-up commands (not counted), then loop the split for >= LOAD_MIN_S.

    Used for BOTH a registry arm and a V5 Ollama cell, so the two are measured
    under identical conditions and their J/command are comparable.
    """
    rows = C.load_split(split)
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

    live = [s_ for s_ in lg.samples if s_[0] >= t0]
    mean_w = statistics.mean([s_[1] for s_ in live]) if live else float("nan")
    max_t = max([s_[4] for s_ in live]) if live else float("nan")
    peak_vram = max([s_[3] for s_ in live]) if live else ""
    if kind == "cloud":
        j, delta = "not measurable (cloud)", ""
        note = note or "cloud arm: latency only; GPU energy is not its cost"
    else:
        delta = mean_w - idle["idle_median_w"]
        j = round(delta * wall / n, 5)
        note = note or "GPU watts only; CPU watts are NOT measured"

    ver, digest = ollama_facts(ollama_note or name)
    safe = name.replace("/", "_").replace(":", "-")
    raw = C.POWER / ("%s_%s_%s.csv" % (safe, split, time.strftime("%Y%m%d_%H%M%S")))
    with raw.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.writer(fh)
        wr.writerow(["timestamp", "power_w", "util_pct", "mem_used_mb", "temp_c"])
        wr.writerows(live)

    q = lambda pp: sorted(lat)[min(len(lat) - 1, int(round(pp * (len(lat) - 1))))]
    rec = dict(arm=name, split=split, kind=kind,
               utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               n_commands=n, wall_s=round(wall, 1), warmup_excluded=WARMUP,
               latency_p50_ms=round(q(.50), 3), latency_p95_ms=round(q(.95), 3),
               latency_p99_ms=round(q(.99), 3),
               mean_latency_ms=round(statistics.mean(lat), 3),
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
    # The caller gates on these lines. measure_loop no longer exits, so the
    # speculative scripts can call it in-process and still end on their own gate.
    return [
        "%s on %s: N=%d commands in %.1f s (>= %d s required); %d warm-up excluded"
        % (name, split, n, wall, LOAD_MIN_S, WARMUP),
        "latency p50/p95/p99 %.2f / %.2f / %.2f ms" % (q(.50), q(.95), q(.99)),
        ("energy %s J/command (mean %.1f W - idle %.2f W)"
         % (j, mean_w, idle["idle_median_w"])) if kind != "cloud"
        else "energy: not measurable (cloud)",
        "max GPU temp %.0f C; peak VRAM %s MB; CPU %.2f ms/command "
        "(CPU watts not measured)" % (max_t, peak_vram, cpu_ms),
        "appended to %s; raw samples %s" % (EFF.name, raw.name),
    ]


# The speculative arms have no in-process predictor: each is its own verified
# script (arms._spec). Their energy is measured INSIDE that script, by
# measure_loop above -- the same protocol as every other arm -- after the
# script's byte-identity gate against lora_greedy has passed.
SPEC_SCRIPTS = {"a8_speculative": "a8_speculative.py",
                "a10_speculative": "a10_prefix_cache.py",
                "a12_smalllm": "a12_smalllm_drafter.py"}


def _eff_rows(name, split):
    if not EFF.exists():
        return 0
    with EFF.open(encoding="utf-8", newline="") as fh:
        return sum(1 for r in csv.DictReader(fh)
                   if r.get("arm") == name and r.get("split") == split)


def mode_speculative(name, split):
    """Run <script> --split <split> --energy and require exactly one new row.

    The script decodes the split once, runs the byte-identity gate against
    pred_lora_greedy_<split>.csv (a hard FAIL if that file is absent), then --
    only on PASS -- calls measure_loop with an end-to-end predict(command):
    prompt, tokenisation, drafter, verifier, detokenisation. It writes no
    prediction or per-command file, so it can never overwrite a v4c result.
    """
    require_idle()            # fail fast; the script re-checks just before measuring
    script = C.REPO / "thesis2" / "scripts" / SPEC_SCRIPTS[name]
    before = _eff_rows(name, split)
    cmd = [sys.executable, "-u", str(script), "--split", split, "--energy"]
    print("  running: %s" % " ".join(cmd))
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace",
                            cwd=str(C.REPO), env=env)
    for line in proc.stdout:              # into this script's log as well
        print("  | " + line.rstrip("\n"))
    rc = proc.wait()
    after = _eff_rows(name, split)
    C.gate(rc == 0 and after == before + 1, [
        "%s --split %s --energy exit code %d%s"
        % (script.name, split, rc, "" if rc == 0 else
           " -- the byte-identity or protocol gate FAILED; no energy row may be quoted"),
        "rows for (%s, %s) in %s: %d -> %d (exactly one must be appended)"
        % (name, split, EFF.name, before, after),
    ])


def mode_arm(name, split):
    arm = A.get(name)
    if name in SPEC_SCRIPTS:
        return mode_speculative(name, split)
    idle = require_idle()
    C.require_frozen(name)
    predict = arm.build()
    C.gate(True, measure_loop(name, split, predict, idle, kind=arm.kind,
                              ollama_note=arm.note))


def v5_cell_predictor(model, strategy, split):
    """A V5 cell as a predictor, so an Ollama cell can be energy-measured the
    same way an arm is. The client is the project's own, unmodified.

    Exemplars and retrieval come from V5's own code (fixed 2026-09-26). The
    earlier version of this function called `load_seen_split(DEFAULT_K)`, which
    returns a TUPLE and whose default seen_path is the V1 file, and passed that
    fixed pool to rag as well. All three faults would have corrupted the
    few_shot and rag energy figures S3-2 schedules -- the same bug that voided
    12 cells in s1_run_v5_cells.py. `bench/run_cell._load_rows` now supplies the
    pool from V5's own DATASETS table, and rag retrieves per command as V5 does
    (bench/run_cell.py:279-283), with the Retriever built once and reused.
    """
    from llm.pipeline import run_once
    from llm.clients import all_clients
    from bench.run_cell import apply_legend
    from llm import prompts as P
    from bench.dataset import DEFAULT_K
    apply_legend("v5")
    schema = P.load_schema(str(C.SCHEMA_FILE))
    roster = {c.name: c for c in all_clients()}
    if model not in roster:
        C.gate(False, ["client %r is not in llm.clients.all_clients() -- is Ollama "
                       "running with OLLAMA_MODELS=D:\LLM?" % model])
    client = roster[model]

    _, fewshot_examples, retriever = C.v5_pool(split, strategy)
    print("  exemplars: k=%d from V5's DATASETS[%r]['exemplars_from'] "
          "(%d in the fixed pool); rag retriever: %s"
          % (DEFAULT_K, split, len(fewshot_examples or []),
             "built once, reused" if retriever else "n/a"))

    def predict(cmd):
        examples, _lat, _ids, _sc = C.v5_examples_for(cmd, strategy,
                                                      fewshot_examples, retriever)
        res = run_once(cmd, client, strategy, schema, examples=examples)
        return (getattr(res, "obj", None) or {}), dict(
            latency_ms=(getattr(res, "latency_s", 0.0) or 0.0) * 1000)
    return predict


def mode_v5_cell(cell, split):
    """Energy + latency for one V5 cell. Cloud cells are refused: their work is
    off this machine, so they get latency only, from Stage 1."""
    if "/" not in cell:
        C.gate(False, ["--v5-cell takes model/strategy, e.g. "
                       "ollama:qwen2.5-3b-lora/zero_shot"])
    model, strategy = cell.rsplit("/", 1)
    if model.startswith("openai:"):
        C.gate(False, ["%s is a CLOUD cell: energy is not measurable from this "
                       "machine. Its latency comes from Stage 1." % cell])
    idle = require_idle()
    predict = v5_cell_predictor(model, strategy, split)
    C.gate(True, measure_loop("v5:%s" % cell, split, predict, idle, kind="local",
                 note="V5 Ollama cell measured through llm/pipeline.py run_once; "
                      "GPU watts only, CPU watts NOT measured"))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--idle", action="store_true")
    g.add_argument("--arm")
    g.add_argument("--v5-cell", dest="v5_cell",
                   help="model/strategy, e.g. ollama:qwen2.5-3b-lora/zero_shot")
    ap.add_argument("--split", choices=list(C.SPLITS))
    a = ap.parse_args()
    tag = ("idle" if a.idle else
           ("%s_%s" % (a.arm, a.split) if a.arm
            else "v5_%s_%s" % (a.v5_cell.replace("/", "_").replace(":", "-"), a.split)))
    C.start_log("s3_energy_%s" % tag)
    if a.idle:
        mode_idle()
    elif a.arm:
        if not a.split:
            C.gate(False, ["--split is required with --arm"])
        mode_arm(a.arm, a.split)
    else:
        if not a.split:
            C.gate(False, ["--split is required with --v5-cell"])
        mode_v5_cell(a.v5_cell, a.split)


if __name__ == "__main__":
    main()
