"""thesis3 unified bench: one constant measurement protocol for all 66 systems x 2 test sets.

    python thesis3\bench\run_bench.py --plan                          # runs, blocks, envs, legends, ETA -> PLAN.md
    python thesis3\bench\run_bench.py --smoke                         # flat_gpu + lora_greedy, held_out_v3, 30 s load
    python thesis3\bench\run_bench.py --campaign c1 --block A         # one block (A, B, C1, C2, D), resumable
    python thesis3\bench\run_bench.py --campaign c1 --redo <run_id>   # an incomplete run, again

Runs in D:\\condaenvs\\thesis3meter. This process samples power (telemetry.Telemetry, 10 Hz),
drives the protocol and writes runs.csv / idle.csv / machine.json / repro_check.csv. It loads
no model and imports nothing from thesis2: every system runs in a worker subprocess
(worker.py) started with the python.exe of the system's env. Files: SCHEMA.md.
"""
from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import registry as REG          # noqa: E402  (metadata only at import)
import telemetry as T           # noqa: E402

csv.field_size_limit(2 ** 31 - 1)
OUT = REPO / "thesis3" / "out"
T2 = REPO / "thesis2" / "out" / "final"
WORKER = HERE / "worker.py"
SPLITS = REG.SPLITS
TAG = {"held_out_v3": "v3", "held_out_v4c": "v4c"}
IDLE_S, LOADEDIDLE_S, COOLDOWN_S, MIN_LOAD_S = 300.0, 60.0, 60.0, 300.0
IDLE_REFRESH_S = 45 * 60      # a new idle window when the last ended > 45 min ago, so it is <= 60 min old at the v3 load start
TEMP_MAX_C, TEMP_WAIT_S = 50.0, 600.0
MIN_HZ, CLOUD_ERR_SHARE = 9.0, 0.02
OLLAMA = "http://127.0.0.1:11434"
RUN_COLS = ["run_id", "session", "system", "family", "kind", "device", "runtime", "split", "legend", "dataset_sha256",
            "idle_id", "t_load_start_mono_ns", "t_load_end_mono_ns", "t_load_start_utc_ns", "t_load_end_utc_ns",
            "passes", "row_passes", "calls_total", "repeats", "model_sha256", "model_load_s", "uses_cloud",
            "gpu_temp_start_c", "notes"]
IDLE_COLS = ["idle_id", "session", "t_start_mono_ns", "t_end_mono_ns", "t_start_utc_ns", "t_end_utc_ns"]
LEGEND_SOURCE = {"v5": "llm.prompts_v5 legend via bench.run_cell.apply_legend('v5')",
                 "none": "builds no prompt",
                 "salsa_row": "thesis2/scripts/salsa_row.py ROW_LEGEND (the row scorer's own legend)"}


class Stop(Exception):
    pass


def stop(msg):
    raise Stop(msg)


# ====================================================================== data
def datasets():
    extra = sorted(glob.glob(str(REPO / "data" / "held_out_v5*.jsonl")))
    if extra:
        stop("found %s. 'V5' in this thesis is the legend version, not a test set: ask the author before running." % extra)
    out = {}
    for sp in SPLITS:
        b = (REPO / "data" / ("%s.jsonl" % sp)).read_bytes()
        n = len([l for l in b.decode("utf-8").splitlines() if l.strip()])
        if n != 217:
            stop("data/%s.jsonl has %d commands, expected 217" % (sp, n))
        out[sp] = hashlib.sha256(b).hexdigest()
    return out


def registry_gate():
    bad = REG.check_against_expected()
    if bad:
        stop("registry and systems_expected.csv disagree:\n  " + "\n  ".join(bad))


def read_csv(p):
    if not Path(p).exists():
        return []
    with open(p, encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_row(p, cols, row):
    new = not Path(p).exists()
    with open(p, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        if new:
            w.writeheader()
        w.writerow(row)
        f.flush()
        os.fsync(f.fileno())


def write_run_order():
    with open(HERE / "run_order.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["system", "split", "block"])
        for n in REG.ORDER:
            for sp in SPLITS:
                w.writerow([n, sp, REG.SYSTEMS[n].block])


# ====================================================================== Ollama (HTTP only: no ollama package here)
def _http(path, body=None, timeout=10):
    req = urllib.request.Request(OLLAMA + path, data=None if body is None else json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as r:
        return json.loads(r.read().decode() or "{}")


def ollama_up():
    try:
        _http("/api/version", timeout=3)
        return True
    except Exception:                                                  # noqa: BLE001
        return False


def unload_ollama():
    """keep_alive=0 for every resident model, then wait until none is resident."""
    if not ollama_up():
        return
    for _ in range(60):
        loaded = [m["name"] for m in _http("/api/ps").get("models", [])]
        if not loaded:
            return
        for m in loaded:
            _http("/api/generate", {"model": m, "keep_alive": 0}, timeout=60)
        time.sleep(1)
    stop("Ollama still holds models after 60 s of keep_alive=0: %s" % loaded)


def _winenv(name):
    import winreg
    vals = []
    for hive, key in ((winreg.HKEY_CURRENT_USER, "Environment"),
                      (winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Session Manager\Environment")):
        try:
            with winreg.OpenKey(hive, key) as k:
                vals.append(winreg.QueryValueEx(k, name)[0])
        except OSError:
            pass
    return vals[0] if vals else None


def ollama_facts(required=True):
    if not ollama_up():
        if required:
            stop("the Ollama server is not reachable at %s: start Ollama (OLLAMA_MODELS=D:\\LLM, "
                 "OLLAMA_MAX_LOADED_MODELS=1) and run the block again" % OLLAMA)
        return {}
    env = {"OLLAMA_MODELS": _winenv("OLLAMA_MODELS"), "OLLAMA_MAX_LOADED_MODELS": _winenv("OLLAMA_MAX_LOADED_MODELS")}
    if required and (str(env["OLLAMA_MODELS"]).rstrip("\\").lower() != r"d:\llm" or str(env["OLLAMA_MAX_LOADED_MODELS"]) != "1"):
        stop("Ollama environment is %s; required OLLAMA_MODELS=D:\\LLM and OLLAMA_MAX_LOADED_MODELS=1 "
             "(set them, restart Ollama, run again)" % env)
    tags = {m["name"]: m["digest"] for m in _http("/api/tags").get("models", [])}
    need = sorted({REG.SYSTEMS[n].model[len("ollama:"):] for n in REG.SYSTEMS if REG.SYSTEMS[n].model.startswith("ollama:")})
    miss = [m for m in need if (m if ":" in m else m + ":latest") not in tags]
    if required and miss:
        stop("Ollama server lacks %s (is it serving D:\\LLM?)" % miss)
    return {"version": _http("/api/version").get("version"), "env_registry": env, "digests": tags}


def uses_ollama(s):
    return s.model.startswith("ollama:") or s.builder == "ensemble"


# ====================================================================== machine.json
def _cmd(args, timeout=60):
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except Exception as e:                                             # noqa: BLE001
        return "unavailable: %s" % e


def env_versions(env):
    try:
        p = subprocess.run([REG.PY[env], str(WORKER), "--versions"], capture_output=True, text=True, timeout=300,
                           encoding="utf-8", errors="replace", cwd=str(REPO))
        return json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:                                             # noqa: BLE001
        return {"error": repr(e)}


def probe_gate():
    p = subprocess.run([sys.executable, str(HERE / "telemetry.py"), "--probe"], capture_output=True, text=True,
                       encoding="utf-8", errors="replace", timeout=120)
    print(p.stdout.rstrip())
    sensor = next((l.split(":", 1)[1].split("<-")[0].strip() for l in p.stdout.splitlines()
                   if l.startswith("CPU sensor used:")), "")
    if p.returncode != 0 or "PROBE PASS" not in p.stdout:
        stop("telemetry.py --probe FAILED (see above): start LibreHardwareMonitor as administrator, web server on :8085, "
             "update interval <= 1 s")
    if not (sensor.lower().startswith(("/intelcpu/", "/amdcpu/")) or "cpu package" in sensor.lower()):
        stop("the probe's CPU sensor is %r, not the CPU package" % sensor)
    js = p.stdout[p.stdout.find("{"): p.stdout.rfind("}") + 1]
    try:
        return json.loads(js)
    except ValueError:
        return {"raw": p.stdout}


def machine_json(root, session, gpu, ds, probe, ollama):
    path = root / "machine.json"
    git = _cmd(["git", "-C", str(REPO), "rev-parse", "HEAD"])
    dirty = bool(_cmd(["git", "-C", str(REPO), "status", "--porcelain"]))
    sess = {"session": session, "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_commit": git,
            "git_dirty": dirty, "probe": probe, "ollama": ollama}
    if path.exists():
        m = json.load(open(path, encoding="utf-8"))
        if m.get("datasets") != ds:
            stop("machine.json datasets %s != the files on disk %s: the test data changed mid-campaign" % (m.get("datasets"), ds))
        m.setdefault("sessions", []).append(sess)
    else:
        import numpy
        import pandas
        lhm = Path(r"C:\Tools\LibreHardwareMonitor\LibreHardwareMonitor.exe")
        m = T.machine_info(gpu)
        m.update({
            "cpu_model": _cmd(["powershell", "-NoProfile", "-Command", "(Get-CimInstance Win32_Processor).Name"]),
            "windows_build": platform.version(), "windows_release": platform.release(),
            "power_plan": _cmd(["powercfg", "/getactivescheme"]),
            "lhm_version": _cmd(["powershell", "-NoProfile", "-Command",
                                 "(Get-Item '%s').VersionInfo.FileVersion" % lhm]) if lhm.exists() else "not found at %s" % lhm,
            "lhm_update_interval": "set by hand in LHM (RUNBOOK step 1); measured refresh %s s"
                                   % probe.get("cpu_sensor_refresh_s"),
            "parent": {"python": sys.version.split()[0], "executable": sys.executable, "pandas": pandas.__version__,
                       "numpy": numpy.__version__},
            "envs": {e: dict(env_versions(e), python_exe=p) for e, p in REG.PY.items()},
            "conda_envs": {"parent": r"D:\condaenvs\thesis3meter", **REG.PY},
            "ollama_version": _cmd(["ollama", "--version"]),
            "git_commit": git, "datasets": ds, "sessions": [sess]})
    tmp = path.with_suffix(".json.tmp")
    json.dump(m, open(tmp, "w", encoding="utf-8"), indent=1, default=str)
    os.replace(tmp, path)
    return m


# ====================================================================== campaign
class Campaign:
    def __init__(self, root: Path, min_load_s=MIN_LOAD_S, splits=SPLITS):
        self.root, self.min_load_s, self.splits = root, min_load_s, splits
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "logs").mkdir(exist_ok=True)
        self.session = time.strftime("%Y%m%d_%H%M%S")
        self.ds = datasets()
        self.idle_n, self.idle_id, self.idle_end = 0, None, None
        self.tel = None

    # ---- bookkeeping
    def done(self):
        return {(r["system"], r["split"]) for r in read_csv(self.root / "runs.csv")}

    def orphans(self):
        known = {r["run_id"] for r in read_csv(self.root / "runs.csv")}
        out = set()
        for sub, ext in (("calls", ".csv.gz"), ("latency", ".npy"), ("passes", ".csv"), ("outputs", ".jsonl")):
            d = self.root / sub
            if d.is_dir():
                out |= {f[: -len(ext)] for f in os.listdir(d) if f.endswith(ext) and f[: -len(ext)] not in known}
        return sorted(out)

    def start(self):
        registry_gate()
        orph = self.orphans()
        if orph:
            stop("incomplete runs (files without a runs.csv row): %s\nre-run each with: "
                 "python thesis3\\bench\\run_bench.py --campaign %s --redo <run_id>" % (orph, self.root.name))
        probe = probe_gate()
        self.tel = T.Telemetry(str(self.root / ("telemetry_%s.csv" % self.session)), hz=10)
        machine_json(self.root, self.session, self.tel.gpu, self.ds, probe, ollama_facts(required=False))
        self.tel.start()
        self.tel.set_phase("between", "")
        print("session %s: telemetry_%s.csv started" % (self.session, self.session))

    def close(self):
        if self.tel is not None:
            try:
                self.tel.set_phase("between", "")
            except Exception:                                          # noqa: BLE001
                pass
            self.tel.stop()
            print("telemetry stopped: %d samples, %d late, %d sampler errors" % (self.tel.n_samples, self.tel.n_late, self.tel.n_errors))

    # ---- protocol steps
    def gpu_temp(self):
        return float(self.tel.gpu.read()["gpu_temp_c"])

    def thermal_gate(self):
        t0 = time.monotonic()
        while True:
            t = self.gpu_temp()
            if t <= TEMP_MAX_C:
                waited = time.monotonic() - t0
                if waited >= 1:
                    print("  thermal gate: waited %.0f s, GPU %.0f C" % (waited, t))
                return waited
            if time.monotonic() - t0 > TEMP_WAIT_S:
                stop("GPU still at %.0f C after %.0f min (limit %.0f C): let it cool, then run the block again"
                     % (t, TEMP_WAIT_S / 60, TEMP_MAX_C))
            time.sleep(1)

    def ensure_idle(self):
        if self.idle_end is not None and (time.perf_counter_ns() - self.idle_end) / 1e9 <= IDLE_REFRESH_S:
            return
        self.idle_n += 1
        iid = "idle_%s_%04d" % (self.session, self.idle_n)
        print("  idle window %s: %.0f s, nothing loaded" % (iid, IDLE_S))
        self.tel.set_phase("idle", iid)
        a, ua = time.perf_counter_ns(), time.time_ns()
        time.sleep(IDLE_S)
        b, ub = time.perf_counter_ns(), time.time_ns()
        self.tel.set_phase("between", "")
        append_row(self.root / "idle.csv", IDLE_COLS, dict(idle_id=iid, session=self.session, t_start_mono_ns=a,
                                                           t_end_mono_ns=b, t_start_utc_ns=ua, t_end_utc_ns=ub))
        self.idle_id, self.idle_end = iid, b

    # ---- one system
    def run_system(self, s, todo):
        print("\n=== %s  [%s, %s, legend %s]  %s" % (s.name, s.block, s.env, s.legend, ", ".join(todo)))
        unload_ollama()
        waited = self.thermal_gate()
        self.ensure_idle()
        local = s.kind == "local"
        runs = [{"run_id": "%s-%s-%s" % (self.session, s.slug, TAG[sp]), "split": sp, "run_index": SPLITS.index(sp)}
                for sp in todo]
        args = [REG.PY[s.env], "-u", str(WORKER), "--system", s.name, "--root", str(self.root), "--runs", json.dumps(runs),
                "--min-load-s", str(self.min_load_s), "--cooldown-s", str(COOLDOWN_S),
                "--loadedidle-s", str(LOADEDIDLE_S if local else 0.0)]
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
                   TRANSFORMERS_OFFLINE="1", OLLAMA_MODELS=r"D:\LLM")
        logp = self.root / "logs" / ("%s.log" % runs[0]["run_id"])
        notes = list()
        if waited >= 1:
            notes.append("thermal gate waited %.0f s" % waited)
        if todo[0] != SPLITS[0]:
            notes.append("resumed without its held_out_v3 run: model load and standby are logged under this run")
        loaded, rows, cur, pre_wait = {}, [], {}, 0.0
        with open(logp, "w", encoding="utf-8") as log:
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=log, text=True,
                                    encoding="utf-8", errors="replace", cwd=str(REPO), env=env)
            self.tel.track_pid(proc.pid)
            try:
                for line in proc.stdout:
                    try:
                        m = json.loads(line)
                    except ValueError:
                        log.write("[stdout] " + line)
                        continue
                    k = m.get("msg")
                    if k == "thermal":                                   # after the warm-up, before the load
                        pre_wait = self.thermal_gate()
                        proc.stdin.write("ok\n")
                        proc.stdin.flush()
                    elif k == "phase":
                        self.tel.set_phase(m["phase"], m["run_id"])     # raises if the sampler died
                        if m["phase"] == "load":
                            cur = {"temp": self.gpu_temp(), "n0": self.tel.n_samples, "err0": self.tel.n_errors,
                                   "pre_wait": pre_wait}
                        elif m["phase"] == "cooldown":
                            cur["n1"] = self.tel.n_samples
                        if m["phase"] != "between":
                            print("  %s %s" % (m["phase"], m["run_id"]))
                        proc.stdin.write("ok\n")
                        proc.stdin.flush()
                    elif k == "loaded":
                        loaded = m
                        print("  loaded in %.1f s (%s)" % (m["model_load_s"], m["model_sha256"][:60]))
                    elif k == "progress":
                        print("    pass %d: %.0f s, %d calls, pass action %.4f" % (m["pass"], m["elapsed_s"], m["calls"], m["pass_acc"]))
                    elif k == "run_done":
                        rows.append(self.finish_run(s, m, loaded, cur, notes, first=not rows))
                        proc.stdin.write("ok\n")
                        proc.stdin.flush()
                rc = proc.wait()
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.wait()
                self.tel.untrack_pid(proc.pid)
                try:
                    self.tel.set_phase("between", "")
                finally:
                    unload_ollama()
        if rc != 0 or len(rows) != len(runs):
            tail = logp.read_text(encoding="utf-8", errors="replace").splitlines()[-25:]
            stop("worker for %s exited with code %s after %d of %d runs; log %s:\n    %s"
                 % (s.name, rc, len(rows), len(runs), logp, "\n    ".join(tail)))
        for r in rows:                                          # the commit markers, after a clean exit
            append_row(self.root / "runs.csv", RUN_COLS, r)
        print("  committed %s" % ", ".join(r["run_id"] for r in rows))

    def finish_run(self, s, m, loaded, cur, notes, first):
        rid, sp = m["run_id"], m["split"]
        T_s = (m["t_load_end_mono_ns"] - m["t_load_start_mono_ns"]) / 1e9
        hz = (cur.get("n1", 0) - cur.get("n0", 0)) / T_s if T_s > 0 else 0.0
        n = list(notes) + list(loaded.get("notes") or [])
        if cur.get("pre_wait", 0) >= 1:
            n.append("pre-load thermal gate (after warm-up) waited %.0f s" % cur["pre_wait"])
        e = self.tel.n_errors - cur.get("err0", self.tel.n_errors)
        if e:
            n.append("telemetry sampler errors during this run: %d" % e)
        R = s.repeats
        print("  %s: %d passes, %d calls in %.0f s, pass-1 action %.4f, %.2f Hz, pass-1 errors %d"
              % (rid, m["passes"], m["calls_total"], T_s, m["pass1_action_acc"], hz, m["pass1_errors"]))
        # ---- gates
        if m["pass1_cmds_complete"] != 217 or m["pass1_calls"] != 217 * R:
            stop("%s: pass 1 scored %d commands / %d calls, expected 217 / %d" % (rid, m["pass1_cmds_complete"], m["pass1_calls"], 217 * R))
        if s.uses_cloud and m["pass1_errors"] > CLOUD_ERR_SHARE * m["pass1_calls"]:
            stop("%s: %d of %d pass-1 calls failed (> %.0f %%); see the calls file" % (rid, m["pass1_errors"], m["pass1_calls"], 100 * CLOUD_ERR_SHARE))
        if not s.uses_cloud and m["pass1_errors"]:
            stop("%s: %d pass-1 calls failed on a local system; see %s" % (rid, m["pass1_errors"], self.root / "calls" / (rid + ".csv.gz")))
        if hz < MIN_HZ:
            stop("%s: telemetry sampled %.2f Hz over the load window (< %.0f Hz)" % (rid, hz, MIN_HZ))
        if s.builder == "spec":
            self.identity_gate(s, rid, sp)
        return dict(run_id=rid, session=self.session, system=s.name, family=s.family, kind=s.kind, device=s.device,
                    runtime=s.runtime, split=sp, legend=s.legend, dataset_sha256=self.ds[sp], idle_id=self.idle_id,
                    t_load_start_mono_ns=m["t_load_start_mono_ns"], t_load_end_mono_ns=m["t_load_end_mono_ns"],
                    t_load_start_utc_ns=m["t_load_start_utc_ns"], t_load_end_utc_ns=m["t_load_end_utc_ns"],
                    passes=m["passes"], row_passes=m["row_passes"], calls_total=m["calls_total"], repeats=R,
                    model_sha256=loaded.get("model_sha256", "n/a"),
                    model_load_s=round(loaded["model_load_s"], 3) if (first and s.kind == "local") else "",
                    uses_cloud=s.uses_cloud, gpu_temp_start_c=round(cur.get("temp", float("nan")), 1), notes="; ".join(n))

    def identity_gate(self, s, rid, sp):
        base = [r for r in read_csv(self.root / "runs.csv") if r["system"] == "lora_greedy" and r["split"] == sp]
        if not base:
            stop("%s: lora_greedy has no %s run in this campaign; block B runs it first" % (s.name, sp))
        a, b = pass1_raw(self.root, base[0]["run_id"]), pass1_raw(self.root, rid)
        diff = [c for c in range(217) if a.get(c) != b.get(c)]
        if diff:
            stop("%s on %s: pass-1 raw output differs from lora_greedy on %d commands %s" % (s.name, sp, len(diff), diff[:10]))
        print("  byte-identity gate: %s == lora_greedy on %s, 217/217" % (s.name, sp))


def pass1_raw(root, rid):
    out = {}
    with open(root / "outputs" / ("%s.jsonl" % rid), encoding="utf-8") as f:
        for ln in f:
            o = json.loads(ln)
            if o["pass"] == 1 and int(o.get("repeat", 1)) == 1:
                out[int(o["cmd_id"])] = o["raw"]
    return out


# ====================================================================== reproduction check
def repro_check(root):
    import pandas as pd
    mf = {(r["arm"], r["split"]): r for r in read_csv(T2 / "metrics_final.csv")}
    v5 = {(r["model"], r["strategy"]): r for r in read_csv(T2 / "v5cells_both_splits.csv")}
    out = []
    for r in read_csv(root / "runs.csv"):
        c = pd.read_csv(root / "calls" / ("%s.csv.gz" % r["run_id"]), usecols=["pass", "cmd_id", "action_ok"])
        c = c[c["pass"] == 1]
        R = int(r["repeats"])
        acc = float((c.groupby("cmd_id").action_ok.sum() * 2 > R).mean())
        ref, n_ref, src = "", "", ""
        if r["system"].startswith("v5:"):
            model, strat = r["system"][3:].rsplit("/", 1)
            x = v5.get((model, strat))
            if x:
                t = "v3" if r["split"] == "held_out_v3" else "v4c"
                ref, n_ref, src = x["%s_action" % t], x["%s_n_action" % t], "v5cells_both_splits.csv (%s)" % x["collapse"][:60]
        else:
            x = mf.get((r["system"], r["split"]))
            if x:
                ref, n_ref, src = x["action_match"], x["n_commands"], "metrics_final.csv"
        try:
            diff = round(100 * (acc - float(ref)), 2)
        except ValueError:
            diff = ""
        out.append(dict(system=r["system"], split=r["split"], run_id=r["run_id"], action_acc_thesis3=round(acc, 4),
                        n_thesis3=int(c.cmd_id.nunique()), action_match_thesis2=ref, n_thesis2=n_ref, diff_pts=diff,
                        thesis2_source=src))
    if out:
        pd.DataFrame(out).to_csv(root / "repro_check.csv", index=False)
    return out


def verify(root, extra=()):
    p = subprocess.run([sys.executable, str(HERE / "verify_campaign.py"), str(root)] + list(extra),
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    print(p.stdout.rstrip())
    if p.stderr.strip():
        print(p.stderr.rstrip())
    return p.returncode


# ====================================================================== plan
def _mean(xs):
    xs = [x for x in xs if x == x]
    return sum(xs) / len(xs) if xs else float("nan")


def estimates():
    """seconds per decided command, from the final run's measurements."""
    eff = {}
    for r in read_csv(T2 / "unified_efficiency.csv"):
        if r["split"] == "held_out_v3":
            eff[r["arm"]] = float(r["wall_s"]) / float(r["n_commands"])
    v5 = {}
    rows = read_csv(T2 / "results_v5cells_held_out_v4c.csv")
    for r in rows:
        if r["model"].startswith("openai:"):
            try:
                v5.setdefault(r["strategy"], []).append(float(r["latency_s"]) + float(r["retrieval_latency_s"] or 0))
            except ValueError:
                pass
    gpt_call = {k: _mean(v) for k, v in v5.items()}
    pc = {}
    for n in ("cascade_cloud_tauA", "cascade_cloud_always"):
        pc[n] = _mean([float(r["latency_ms"]) / 1000 for r in read_csv(T2 / ("pred_%s_held_out_v3.csv" % n)) if r["latency_ms"]])
    # ensemble escalation share, from the two V5 cells it pairs (held_out_v3)
    by = {}
    for r in read_csv(T2 / "results_v5cells_held_out_v3.csv"):
        if (r["model"], r["strategy"]) in (("classifier:distilbert", "classifier"), ("ollama:qwen2.5-3b-lora", "zero_shot")):
            try:
                o = json.loads(r["pred_json"] or "{}")
            except ValueError:
                o = {}
            by.setdefault(r["command_id"], []).append((o.get("action"), o.get("target"), o.get("location")))
    esc = _mean([float(len(set(v)) > 1) for v in by.values()])
    est = {}
    for n, s in REG.SYSTEMS.items():
        if n in eff:
            est[n] = eff[n]
        elif s.model == "openai:gpt-4o-mini":
            est[n] = 5 * gpt_call[s.strategy]
        elif n in pc:
            est[n] = 5 * pc[n]
        elif n == "ensemble":
            est[n] = eff["v5:classifier:distilbert/classifier"] + eff["v5:ollama:qwen2.5-3b-lora/zero_shot"] + esc * 5 * gpt_call["rag"]
    return est, esc


def load_est(s):
    if s.builder == "spec":
        return 150
    if s.model.startswith("ollama:") or s.builder == "ensemble":
        return 30 if "deepseek" in s.model else 15
    if s.runtime.startswith("transformers") or "transformers" in s.runtime:
        return 60
    return 15


def plan(min_load_s=MIN_LOAD_S):
    registry_gate()
    ds = datasets()
    write_run_order()
    est, esc = estimates()
    sys_rows, blocks = [], {}
    for n in REG.ORDER:
        s = REG.SYSTEMS[n]
        cmd_s = est[n]
        pass_s = cmd_s * 217
        load = math.ceil(min_load_s / pass_s) * pass_s if pass_s < min_load_s else pass_s
        warm = 3 * cmd_s / s.repeats
        per_run = warm + load + COOLDOWN_S + 10
        tot = 2 * per_run + load_est(s) + (LOADEDIDLE_S if s.kind == "local" else 0)
        sys_rows.append((s, cmd_s, load, tot))
        blocks[s.block] = blocks.get(s.block, 0.0) + tot
    for b in blocks:
        blocks[b] += IDLE_S * (1 + int(blocks[b] // IDLE_REFRESH_S))
    hms = lambda x: "%d h %02d min" % (x // 3600, (x % 3600) // 60)        # noqa: E731
    L = ["# thesis3 unified bench: plan", "",
         "Generated by `python thesis3\\bench\\run_bench.py --plan` on %s. 66 systems x 2 test sets = %d runs."
         % (time.strftime("%Y-%m-%d %H:%M"), 2 * len(REG.SYSTEMS)), "",
         "## Test data (identical for every system)", "",
         "| test set | commands | sha256 |", "|---|---|---|"]
    L += ["| %s | 217 | `%s` |" % (sp, ds[sp]) for sp in SPLITS]
    L += ["", "No `data/held_out_v5*.jsonl` exists (checked at every launch; V5 is the legend version).", "",
          "## Blocks (fixed order; `--block X` runs one block; resume skips any (system, test set) with a runs.csv row)", "",
          "| block | content | systems | runs | ETA |", "|---|---|---|---|---|"]
    desc = {"A": "CPU systems", "B": "transformers / torch GPU systems", "C1": "Ollama cells except deepseek-r1",
            "C2": "the 5 deepseek-r1:7b cells", "D": "gpt-4o-mini cells and systems that call gpt-4o-mini"}
    for b in REG.BLOCKS:
        k = sum(1 for s in REG.SYSTEMS.values() if s.block == b)
        L.append("| %s | %s | %d | %d | %s |" % (b, desc[b], k, 2 * k, hms(blocks[b])))
    L.append("| all | | %d | %d | %s |" % (len(REG.SYSTEMS), 2 * len(REG.SYSTEMS), hms(sum(blocks.values()))))
    L += ["", "ETA per system = 2 x (3 warm-up calls + load + 60 s cool-down + ~10 s file writing) + model load "
          "(estimate) + 60 s loaded idle (local only); load = whole passes of 217 commands until >= %.0f s, one pass "
          "taking 217 x wall_s / n_commands from `thesis2/out/final/unified_efficiency.csv` (held_out_v3; also used for "
          "held_out_v4c). gpt-4o-mini cells: 5 x mean (latency_s + retrieval_latency_s) of "
          "`results_v5cells_held_out_v4c.csv`; cloud cascades: 5 x mean latency_ms of their pred files; ensemble: "
          "classifier + qwen2.5-3b-lora/zero_shot + %.1f %% escalation x 5 gpt-4o-mini/rag calls (escalation share = "
          "disagreement of the two V5 cells on held_out_v3). Per block: + one 300 s idle window per started 45 min. "
          "The thermal-gate wait is not included." % (min_load_s, 100 * esc), "",
          "## System -> block -> env -> legend", "",
          "Legend `recorded` systems record the value the thesis2 code uses: `v5` for the 4 salsa_full systems, "
          "`salsa_row` for the 4 cascades that call a row model.", "",
          "| # | system | block | env (python.exe) | legend | legend source | kind | device | runtime | repeats | cloud | s / command | load s | ETA |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    exp = {r["system"]: r for r in read_csv(HERE / "systems_expected.csv")}
    for i, (s, cmd_s, load, tot) in enumerate(sys_rows, 1):
        le = exp[s.name]["legend_expected"]
        L.append("| %d | `%s` | %s | %s (`%s`) | %s%s | %s | %s | %s | %s | %d | %d | %.4g | %.0f | %s |"
                 % (i, s.name, s.block, s.env, REG.PY[s.env], s.legend, " (recorded)" if le == "recorded" else "",
                    LEGEND_SOURCE[s.legend], s.kind, s.device, s.runtime, s.repeats, s.uses_cloud, cmd_s, load, hms(tot)))
    L += ["", "## Ensemble wiring", "",
          "One worker in the `thesis` env, no helper process (thesis2 definition, `arms._ensemble`, METHODS_NOTES "
          "item 8), run online per command:", "",
          "1. `classifier:distilbert` (V5 cell, CPU, single-threaded) and `ollama:qwen2.5-3b-lora` zero_shot, both "
          "through `llm/pipeline.run_once` with the V5 legend;",
          "2. if they agree on (action, target, location): the classifier's object;",
          "3. otherwise `openai:gpt-4o-mini` rag x 5 (one retrieval, the V5 Retriever), every call through "
          "`common.RetryingClient` (`with_retries`); majority vote over the canonical prediction, a tie to the "
          "earliest repeat, a failed repeat votes `{}`.", "",
          "The whole command (all of the above) is one timed call. `extra`: `escalated`, `cloud_calls` (0 or 5), "
          "`retries`. The raw output stores every component's raw text as JSON. Model load = classifier weights + "
          "Ollama keep-alive load of qwen2.5-3b-lora + the Retriever.", "",
          "## Protocol (every system)", "",
          "Unload all (Ollama keep_alive=0, previous worker ended) -> thermal gate (GPU <= 50 C, poll 1 s, max 10 min) "
          "-> idle window (300 s, nothing loaded) when the last one ended > 45 min ago -> `modelload` (worker load, "
          "timed) -> `loadedidle` 60 s (local) -> per test set: `warmup` (commands 0-2) -> thermal gate again (<= 50 C, model resident) -> `load` (shuffled passes, "
          "seed 20260827 + pass + run index (0 = v3, 1 = v4c), >= 1 pass and >= 300 s, pass in progress finished; "
          "cloud: 5 repeats per command) -> `cooldown` 60 s -> files -> after the worker exits cleanly, the runs.csv rows.", "",
          "## Decisions and definitions to check", "",
          "- **Idle refresh at 45 min, not 60.** verify_campaign.py requires the idle window to be <= 60 min old at "
          "the system's held_out_v3 load start; thermal gate + model load + 60 s loaded idle come in between, so a new "
          "window is taken when the last one is > 45 min old.",
          "- **Second thermal gate, after the warm-up.** The smoke run showed the 3 warm-up calls of lora_greedy "
          "heating the GPU from 45 to 65 C in 3 s, so its load started at 57 C and verify_campaign.py (start <= 51 C) "
          "failed. The parent now waits for <= 50 C between warm-up and load (model resident, phase `between`); "
          "the wait is written in `notes`.",
          "- **A8/A10/A12** are loaded by importing their thesis2 scripts with `--limit 3 --outdir <temp dir>`; "
          "`model_load_s` includes that 3-command decode (written in `notes`). A10/A12 then cache the token prefix "
          "common to each test set's 217 prompts, as their thesis2 runs did. `first_rejection_field` uses A11's "
          "`regions()` (taken from the file without running it), after the call's timing ends.",
          "- **t_infer** = first start to last end of the wrapped model calls: classifier/net `forward`, "
          "Qwen2/T5 `forward` and `generate`, SentenceTransformer `encode`, sklearn `transform`/`predict`, "
          "`OllamaClient/OpenAIClient/DistilBertClassifierClient.complete`, OpenAI `chat.completions.create`. "
          "rules_*: no model call, so t_infer = the whole call.",
          "- **Confidence**: classifiers = product of the verb, target and location heads' max softmax (factorized: "
          "location = the weakest factor, as arms.py defines it; captured logits, computed after timing). Empty for "
          "LLMs, knn_minilm and rules.",
          "- **Tokens**: Ollama `prompt_eval_count`/`eval_count` (eval_count includes the thinking tokens); "
          "deepseek-r1 `tokens_thinking` = the trace re-tokenised with bench/backfill_tokens_v5's DeepSeek vocabulary; "
          "OpenAI `usage`; transformers from the input/output ids. Ollama `ttft_ms` = server total_duration - "
          "eval_duration; transformers `ttft_ms` = call start to the start of the LLM's second forward pass (flant5: "
          "the T5 forward); empty for A8/A10/A12 (their first token comes out of a verifier pass).",
          "- **Failed calls**: an exception or a model_error/prompt_error stage is written to `error` and scored "
          "wrong. extract_failed / schema_failed are not errors (wrong answers, `valid_json` 0).",
          "- **Ollama**: the server must run with OLLAMA_MODELS=D:\\LLM and OLLAMA_MAX_LOADED_MODELS=1 (checked in "
          "the user/machine environment, models checked on the server); version and every digest go into "
          "machine.json, the digest into `model_sha256`. num_ctx 4096, temperature 0.0 (as recorded), no num_predict.",
          "- **HF offline**: workers run with HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1, as the thesis2 energy stage.", "",
          "## Commands (PowerShell, D:\\condaenvs\\thesis3meter)", "",
          "    python thesis3\\bench\\run_bench.py --dry A",
          "    python thesis3\\bench\\run_bench.py --campaign c1 --block A",
          "    python thesis3\\bench\\verify_campaign.py thesis3\\out\\c1 --partial",
          "", "`--dry X` (a block or system names) loads each system and runs commands 0-2 of both test sets through "
          "it: no telemetry, no campaign files (logs in thesis3\\out\\dry). A cloud system costs 6 gpt-4o-mini calls "
          "(ensemble up to 30). Then B, C1, C2, D the same way. The harness runs verify --partial itself at the end "
          "of every block and writes repro_check.csv.", ""]
    (HERE / "PLAN.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L))
    print("\nwrote %s and %s" % (HERE / "PLAN.md", HERE / "run_order.csv"))


# ====================================================================== entry points
def run_block(root, systems, splits=SPLITS, min_load_s=MIN_LOAD_S, verify_args=("--partial",)):
    c = Campaign(root, min_load_s, splits)
    if any(uses_ollama(REG.SYSTEMS[n]) for n in systems):
        ollama_facts(required=True)
    try:
        c.start()
        done = c.done()
        for n in systems:
            todo = [sp for sp in splits if (n, sp) not in done]
            if not todo:
                print("skip %s: already measured" % n)
                continue
            c.run_system(REG.SYSTEMS[n], todo)
    finally:
        c.close()
        unload_ollama()
    rc = repro_check(root)
    print("\nrepro_check.csv: %d runs" % len(rc))
    if verify_args is not None and verify(root, verify_args) != 0:
        stop("verify_campaign.py FAILED (above): do not start the next block")


def smoke():
    root = OUT / "smoke"
    if root.exists():
        print("removing the previous smoke output %s" % root)
        shutil.rmtree(root)
    run_block(root, ["flat_gpu", "lora_greedy"], splits=("held_out_v3",), min_load_s=30.0, verify_args=None)
    p = subprocess.run([sys.executable, str(HERE / "analyze.py"), str(root)], text=True)
    if p.returncode:
        stop("analyze.py failed on the smoke output")
    rc = verify(root, ("--subset", "--partial", "--after-analysis", "--min-load-s", "30"))
    import pandas as pd
    pd.set_option("display.width", 200)
    s = pd.read_csv(root / "summary.csv")
    q = pd.read_csv(root / "quality.csv").merge(s[["run_id", "system"]], on="run_id")
    print("\nsummary.csv")
    print(s[["system", "split", "action_acc", "lat_p50_ms", "net_wh_per_1k", "energy_complete", "loaded_idle_net_w",
             "model_load_s"]].to_string(index=False))
    print("\nquality.csv")
    print(q[["system", "sample_hz", "gpu_counter_vs_integral_pct", "cpu_sensor_refresh_s"]].to_string(index=False))
    print("\nrepro_check.csv")
    print(pd.read_csv(root / "repro_check.csv").to_string(index=False))
    return rc


def dry(names):
    """load each system and run commands 0-2 of both test sets through it: no telemetry, no files
    in a campaign (worker logs go to thesis3/out/dry/). Costs 3 x 2 x repeats gpt-4o-mini calls
    for a cloud system."""
    registry_gate()
    logs = OUT / "dry"
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", OLLAMA_MODELS=r"D:\LLM")
    bad = []
    for n in names:
        s = REG.SYSTEMS[n]
        unload_ollama()
        logp = logs / ("%s.log" % s.slug)
        with open(logp, "w", encoding="utf-8") as log:
            p = subprocess.run([REG.PY[s.env], "-u", str(WORKER), "--system", n, "--dry"], stdout=subprocess.PIPE,
                               stderr=log, text=True, encoding="utf-8", errors="replace", cwd=str(REPO), env=env)
        print("\n=== %s  (%s) exit %d" % (n, s.env, p.returncode))
        print(p.stdout.rstrip())
        if p.returncode:
            bad.append(n)
            print("  log tail:\n    " + "\n    ".join(logp.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]))
    unload_ollama()
    print("\nDRY %s" % ("PASS (%d systems)" % len(names) if not bad else "FAIL: %s" % bad))
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--plan", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--dry", nargs="+", metavar="SYSTEM_OR_BLOCK",
                    help="load and run 3 commands per test set, no measurement (block names expand)")
    ap.add_argument("--campaign")
    ap.add_argument("--block", choices=REG.BLOCKS)
    ap.add_argument("--redo", metavar="RUN_ID")
    a = ap.parse_args()
    try:
        if a.plan:
            return plan()
        if a.smoke:
            return smoke()
        if a.dry:
            names = [n for x in a.dry for n in ([m for m in REG.ORDER if REG.SYSTEMS[m].block == x] if x in REG.BLOCKS else [x])]
            unknown = [n for n in names if n not in REG.SYSTEMS]
            if unknown:
                ap.error("unknown systems %s" % unknown)
            sys.exit(dry(names))
        if not a.campaign or not (a.block or a.redo):
            ap.error("--campaign NAME with --block X or --redo RUN_ID (or --plan / --smoke)")
        root = OUT / a.campaign
        write_run_order()
        if a.redo:
            rid = a.redo
            if any(r["run_id"] == rid for r in read_csv(root / "runs.csv")):
                stop("%s has a runs.csv row: it is complete, nothing to redo" % rid)
            slug = rid.split("-", 1)[1].rsplit("-", 1)[0] if "-" in rid else ""
            sysn = [n for n, s in REG.SYSTEMS.items() if s.slug == slug]
            if not sysn:
                stop("cannot map run id %s to a system" % rid)
            for sub, ext in (("outputs", ".jsonl"), ("calls", ".csv.gz"), ("latency", ".npy"), ("passes", ".csv")):
                p = root / sub / (rid + ext)
                if p.exists():
                    print("deleting incomplete %s" % p)
                    p.unlink()
            return run_block(root, sysn)
        run_block(root, [n for n in REG.ORDER if REG.SYSTEMS[n].block == a.block])
        print("\nblock %s complete. Next: robocopy thesis3\\out\\%s D:\\thesis3_backup\\%s /MIR" % (a.block, a.campaign, a.campaign))
    except Stop as e:
        print("\nCAMPAIGN STOPPED: %s" % e)
        sys.exit(1)


if __name__ == "__main__":
    main()
