"""thesis3 telemetry sampler: one constant measurement setup for every test.

Samples, at a fixed rate (default 10 Hz = every 100 ms, the NVIDIA power update
period reported by Yang et al., SC 2024):

  GPU  (NVML via nvidia-ml-py)
    gpu_power_w            board power, NVML's averaged reading (W)
    gpu_power_instant_w    instantaneous board power, if the driver exposes it (W)
    gpu_energy_mj          on-board cumulative energy counter (mJ since driver load).
                           This counter integrates on the device, so it has no
                           sampling gaps; it is the PRIMARY GPU energy source.
    gpu_util_pct, gpu_mem_util_pct, vram_used_mb, gpu_temp_c,
    gpu_sm_clock_mhz, gpu_mem_clock_mhz, gpu_pstate, gpu_fan_pct
  CPU  (LibreHardwareMonitor HTTP server, http://127.0.0.1:8085/data.json)
    cpu_pkg_power_w        CPU package power (W). Handles "45,3 W" (German locale).
  System / processes (psutil)
    cpu_util_pct           system-wide CPU utilisation (%)
    proc_cpu_s             cumulative CPU seconds of the tracked processes: the
                           system-under-test child process(es) registered with
                           track_pid() plus every process whose name contains
                           "ollama"; the sampler's own process is NOT counted.
                           Process list refreshed every 5 s
    proc_rss_mb            resident memory of the tracked processes (MB)
    sys_ram_used_mb        system RAM in use (MB)
  Bookkeeping
    t_utc_ns               wall clock, UTC, ns since epoch (time.time_ns)
    t_mono_ns              monotonic clock (time.perf_counter_ns). The harness stamps
                           every command with the SAME clock, so commands and power
                           samples can be joined exactly.
    run_id, phase          set by the harness: idle / warmup / load / cooldown

Not measurable in software on this machine, and therefore NOT reported as measured:
RAM (DIMM) power, motherboard, SSD, fans, PSU losses, wall-socket power.

Usage (inside the PARENT harness process; the system under test runs in a child
process, so the sampler never competes with it for the Python GIL):
    tel = Telemetry(out_csv, hz=10); tel.start()
    tel.set_phase("idle", run_id="idle_0001"); ...
    tel.track_pid(child.pid); ...; tel.untrack_pid(child.pid); tel.stop()
On Windows, start() also raises the timer resolution to 1 ms (timeBeginPeriod) and
keeps the system and the display awake (SetThreadExecutionState) while it runs, so
sleep, hibernation or the monitor switching off cannot happen mid-campaign.

Self-test without a GPU:  python telemetry.py --selftest
Live check on the bench:  python telemetry.py --probe   (prints one sample, exits 1
                          if the GPU or the CPU package sensor is missing)
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request

import psutil

FIELDS = [
    "t_utc_ns", "t_mono_ns", "run_id", "phase",
    "gpu_power_w", "gpu_power_instant_w", "gpu_energy_mj",
    "gpu_util_pct", "gpu_mem_util_pct", "vram_used_mb", "gpu_temp_c",
    "gpu_sm_clock_mhz", "gpu_mem_clock_mhz", "gpu_pstate", "gpu_fan_pct",
    "cpu_pkg_power_w", "cpu_util_pct", "proc_cpu_s", "proc_rss_mb", "sys_ram_used_mb",
    "late_ms",
]
NAN = float("nan")


# --------------------------------------------------------------------- GPU
class NvmlSource:
    def __init__(self, index: int = 0):
        import pynvml as n
        self.n = n
        n.nvmlInit()
        self.h = n.nvmlDeviceGetHandleByIndex(index)
        self.name = n.nvmlDeviceGetName(self.h)
        if isinstance(self.name, bytes):
            self.name = self.name.decode()
        drv = n.nvmlSystemGetDriverVersion()
        self.driver = drv.decode() if isinstance(drv, bytes) else drv
        self.has_instant = hasattr(n, "NVML_FI_DEV_POWER_INSTANT")
        try:
            n.nvmlDeviceGetTotalEnergyConsumption(self.h)
            self.has_energy = True
        except Exception:
            self.has_energy = False

    def _safe(self, f, *a, default=NAN):
        try:
            return f(*a)
        except Exception:
            return default

    def read(self) -> dict:
        n, h = self.n, self.h
        d = {}
        d["gpu_power_w"] = self._safe(n.nvmlDeviceGetPowerUsage, h) / 1000.0
        d["gpu_power_instant_w"] = NAN
        if self.has_instant:
            try:
                v = n.nvmlDeviceGetFieldValues(h, [n.NVML_FI_DEV_POWER_INSTANT])[0]
                if v.nvmlReturn == 0:
                    d["gpu_power_instant_w"] = v.value.uiVal / 1000.0
            except Exception:
                pass
        d["gpu_energy_mj"] = self._safe(n.nvmlDeviceGetTotalEnergyConsumption, h) if self.has_energy else NAN
        u = self._safe(n.nvmlDeviceGetUtilizationRates, h, default=None)
        d["gpu_util_pct"] = u.gpu if u is not None else NAN
        d["gpu_mem_util_pct"] = u.memory if u is not None else NAN
        m = self._safe(n.nvmlDeviceGetMemoryInfo, h, default=None)
        d["vram_used_mb"] = m.used / 2**20 if m is not None else NAN
        d["gpu_temp_c"] = self._safe(n.nvmlDeviceGetTemperature, h, n.NVML_TEMPERATURE_GPU)
        d["gpu_sm_clock_mhz"] = self._safe(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_SM)
        d["gpu_mem_clock_mhz"] = self._safe(n.nvmlDeviceGetClockInfo, h, n.NVML_CLOCK_MEM)
        d["gpu_pstate"] = self._safe(n.nvmlDeviceGetPerformanceState, h)
        d["gpu_fan_pct"] = self._safe(n.nvmlDeviceGetFanSpeed, h)
        return d

    def info(self) -> dict:
        return {"gpu_name": self.name, "driver": self.driver,
                "nvml_energy_counter": self.has_energy, "nvml_instant_power": self.has_instant}


class FakeGpu:
    """Deterministic stand-in for --selftest: 11 W idle, 100 W under load."""
    def __init__(self, tel: "Telemetry"):
        self.tel, self.e = tel, 0.0
        self.last = time.perf_counter()

    def read(self) -> dict:
        now = time.perf_counter()
        p = 100.0 if self.tel.phase == "load" else 11.0
        self.e += p * (now - self.last) * 1000.0
        self.last = now
        return {"gpu_power_w": p, "gpu_power_instant_w": p, "gpu_energy_mj": self.e,
                "gpu_util_pct": 95 if p > 50 else 0, "gpu_mem_util_pct": 30, "vram_used_mb": 8000,
                "gpu_temp_c": 60, "gpu_sm_clock_mhz": 2500, "gpu_mem_clock_mhz": 9000,
                "gpu_pstate": 0, "gpu_fan_pct": 40}

    def info(self) -> dict:
        return {"gpu_name": "FAKE", "driver": "n/a", "nvml_energy_counter": True, "nvml_instant_power": True}


# --------------------------------------------------------------------- CPU package power
_NUM = re.compile(r"-?\d+(?:[.,]\d+)?")


def _to_float(v) -> float:
    if isinstance(v, (int, float)):
        return float(v)
    m = _NUM.search(str(v))
    return float(m.group(0).replace(",", ".")) if m else NAN


class LhmSource:
    """CPU package power from LibreHardwareMonitor's HTTP server.

    Start LibreHardwareMonitor as administrator, Options -> Remote Web Server -> Run,
    port 8085. Its sensors refresh at LHM's own update interval; set it to the
    smallest value it offers. Duplicate readings between refreshes are expected and
    are flagged later by the analysis (cpu_sensor_refresh_s).

    The HTTP call runs in its own thread so a slow response can never delay a 10 Hz
    sample; the sampler reads the cached value. A value older than STALE_S is NaN, so
    a dead LHM shows up as missing CPU power (and incomplete energy), never as zero."""
    STALE_S = 2.0
    CPU_PREFIXES = ("/intelcpu/", "/amdcpu/")

    def __init__(self, url: str = "http://127.0.0.1:8085/data.json", poll_hz: float = 10.0):
        self.url = url.replace("localhost", "127.0.0.1")       # avoids slow IPv6/localhost lookups on Windows
        self._open = urllib.request.build_opener(urllib.request.ProxyHandler({})).open   # never via a proxy
        self.path = None
        self.period = 1.0 / poll_hz
        self.errors = 0
        self._val, self._t = NAN, 0.0
        self._lock = threading.Lock()
        self._halt = threading.Event()
        self._th = None

    # ---- tree helpers
    def _fetch(self):
        with self._open(self.url, timeout=0.5) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    @staticmethod
    def _walk(node, trail=()):
        yield node, trail
        for c in node.get("Children", []) or []:
            yield from LhmSource._walk(c, trail + (node.get("Text", ""),))

    def _find(self, tree):
        """Only a CPU package power sensor: SensorId under /intelcpu/ or /amdcpu/.
        Older LHM builds without SensorId: exact text "CPU Package" (Intel) or "Package"
        under a hardware node named like a CPU. Never "GPU Package"."""
        fallback = None
        for node, trail in self._walk(tree):
            text = (node.get("Text") or "").strip().lower()
            sid = (node.get("SensorId") or "").lower()
            if sid:
                if sid.startswith(self.CPU_PREFIXES) and "/power/" in sid and text in ("cpu package", "package"):
                    return node.get("SensorId")
                continue
            in_power = bool(trail) and trail[-1].strip().lower() == "powers"
            cpu_hw = any(k in " ".join(trail).lower() for k in ("intel core", "ryzen", "xeon", "epyc", "threadripper", "core ultra"))
            if in_power and fallback is None and (text == "cpu package" or (text == "package" and cpu_hw)):
                fallback = "text:" + "/".join(trail + (node.get("Text", ""),))
        return fallback

    def _match(self, node, trail):
        if self.path.startswith("text:"):
            return "text:" + "/".join(trail + (node.get("Text", ""),)) == self.path
        return node.get("SensorId") == self.path

    def read_once(self) -> float:
        tree = self._fetch()
        if self.path is None:
            self.path = self._find(tree)
            if self.path is None:
                return NAN
        for node, trail in self._walk(tree):
            if self._match(node, trail):
                v = node.get("RawValue")
                return _to_float(v if v not in (None, "") else node.get("Value"))
        return NAN

    # ---- background poller
    def _loop(self):
        while not self._halt.is_set():
            t0 = time.monotonic()
            try:
                v = self.read_once()
            except Exception:
                v = NAN
                self.errors += 1
            with self._lock:
                if not math.isnan(v):
                    self._val, self._t = v, time.monotonic()
            self._halt.wait(max(0.0, self.period - (time.monotonic() - t0)))

    def start(self):
        if self._th is None:
            self._th = threading.Thread(target=self._loop, daemon=True, name="lhm-poller")
            self._th.start()

    def stop(self):
        self._halt.set()

    def read(self) -> float:
        """cached value for the sampler (non-blocking)."""
        with self._lock:
            return self._val if time.monotonic() - self._t <= self.STALE_S else NAN


# --------------------------------------------------------------------- sampler
class Telemetry(threading.Thread):
    def __init__(self, out_csv: str, hz: float = 10.0, gpu=None, lhm_url: str | None = "http://127.0.0.1:8085/data.json",
                 track_names=("ollama",), track_self: bool = False):
        super().__init__(daemon=True)
        self.out_csv, self.period_ns = out_csv, int(1e9 / hz)
        self.hz = hz
        self.gpu = gpu if gpu is not None else NvmlSource()
        self.lhm = LhmSource(lhm_url) if lhm_url else None
        self.track_names = track_names
        self.track_pids: set[int] = set()
        self.track_self = track_self
        self._state = ("init", "")                       # (phase, run_id), swapped atomically
        self._cpu_last: dict[int, float] = {}
        self._cpu_acc = 0.0
        self._halt = threading.Event()
        self._procs, self._procs_t = [], 0.0
        self.n_samples, self.n_late, self.n_errors = 0, 0, 0
        self.last_error = ""
        psutil.cpu_percent(interval=None)

    def track_pid(self, pid: int):
        """count this process (and its children) in proc_cpu_s / proc_rss_mb from now on."""
        self.track_pids.add(int(pid))
        self._procs_t = 0.0                              # refresh the process list at the next sample

    def untrack_pid(self, pid: int):
        self.track_pids.discard(int(pid))
        self._procs_t = 0.0

    def set_phase(self, phase: str, run_id: str | None = None):
        """Also the harness's health check: raises if the sampler thread has died, so a
        run is never measured without telemetry."""
        if self.ident is not None and not self.is_alive():
            raise RuntimeError(f"telemetry sampler is not running (last error: {self.last_error})")
        self._state = (phase, self._state[1] if run_id is None else run_id)   # one assignment: never half-updated

    def _tracked(self):
        now = time.monotonic()
        if now - self._procs_t > 5.0 or not self._procs:
            procs = [psutil.Process(os.getpid())] if self.track_self else []
            for pid in list(self.track_pids):
                try:
                    pp = psutil.Process(pid)
                    procs.append(pp)
                    procs.extend(pp.children(recursive=True))
                except Exception:
                    self.track_pids.discard(pid)
            for p in psutil.process_iter(["name"]):
                try:
                    if any(t in (p.info["name"] or "").lower() for t in self.track_names):
                        procs.append(p)
                except Exception:
                    pass
            self._procs, self._procs_t = procs, now
        return self._procs

    @property
    def phase(self):
        return self._state[0]

    @property
    def run_id(self):
        return self._state[1]

    def _proc_stats(self):
        """proc_cpu_s only grows by the CPU each tracked process used since it was last seen:
        a process that joins the list adds nothing at first sight (its lifetime CPU before
        tracking is not ours), a process that exits stops contributing."""
        rss, seen = 0.0, {}
        for p in self._tracked():
            try:
                t = p.cpu_times()
                c = t.user + t.system
                if p.pid in self._cpu_last:
                    self._cpu_acc += max(0.0, c - self._cpu_last[p.pid])
                seen[p.pid] = c
                rss += p.memory_info().rss / 2**20
            except Exception:
                pass
        self._cpu_last = seen
        return self._cpu_acc, rss

    def start(self):
        if os.path.exists(self.out_csv):                  # fail loudly in the harness, before anything is measured
            raise FileExistsError(f"{self.out_csv} exists: use a new session name, never overwrite telemetry")
        if self.lhm:
            self.lhm.start()
            time.sleep(0.3)                              # first CPU value before the first sample
        super().start()

    @staticmethod
    def _windows_keep_awake(on: bool):
        if os.name != "nt":
            return
        try:
            import ctypes
            ES_CONTINUOUS, ES_SYSTEM_REQUIRED, ES_DISPLAY_REQUIRED = 0x80000000, 0x00000001, 0x00000002
            ctypes.windll.kernel32.SetThreadExecutionState(
                ES_CONTINUOUS | (ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED if on else 0))
            (ctypes.windll.winmm.timeBeginPeriod if on else ctypes.windll.winmm.timeEndPeriod)(1)
        except Exception:
            pass

    def run(self):
        self._windows_keep_awake(True)                   # this thread lives for the whole session
        try:
            self._run()
        finally:
            self._windows_keep_awake(False)

    def _run(self):
        os.makedirs(os.path.dirname(os.path.abspath(self.out_csv)), exist_ok=True)
        with open(self.out_csv, "x", newline="", encoding="utf-8") as f:   # "x": never overwrite an earlier session
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            next_t = time.perf_counter_ns()
            last_flush = time.monotonic()
            while not self._halt.is_set():
                now = time.perf_counter_ns()
                late_ms = (now - next_t) / 1e6
                phase, run_id = self._state
                row = {"t_utc_ns": time.time_ns(), "t_mono_ns": now, "run_id": run_id, "phase": phase}
                try:                                     # one bad read must never kill the sampler
                    row.update(self.gpu.read())
                    row["cpu_pkg_power_w"] = self.lhm.read() if self.lhm else NAN
                    row["cpu_util_pct"] = psutil.cpu_percent(interval=None)
                    row["proc_cpu_s"], row["proc_rss_mb"] = self._proc_stats()
                    row["sys_ram_used_mb"] = psutil.virtual_memory().used / 2**20
                except Exception as e:                   # keep the time stamp, NaN the rest
                    self.n_errors += 1
                    self.last_error = repr(e)[:200]
                    for k in FIELDS:
                        row.setdefault(k, NAN)
                row["late_ms"] = round(late_ms, 3)
                w.writerow(row)
                self.n_samples += 1
                if late_ms > self.period_ns / 1e6 / 2:
                    self.n_late += 1
                if time.monotonic() - last_flush > 1.0:
                    f.flush(); last_flush = time.monotonic()
                next_t += self.period_ns
                if next_t < time.perf_counter_ns():      # fell behind: re-anchor, never burst
                    next_t = time.perf_counter_ns() + self.period_ns
                self._halt.wait(max(0.0, (next_t - time.perf_counter_ns()) / 1e9))

    def stop(self):
        self._halt.set()
        if self.lhm:
            self.lhm.stop()
        self.join(timeout=5)


def machine_info(gpu) -> dict:
    import platform
    info = {"utc_start": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "os": platform.platform(), "python": sys.version.split()[0],
            "cpu": platform.processor(), "cpu_logical": psutil.cpu_count(),
            "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1)}
    info.update(gpu.info())
    return info


# --------------------------------------------------------------------- CLI
def _selftest():
    import tempfile
    out = os.path.join(tempfile.mkdtemp(), "selftest.csv")
    fake = FakeGpu.__new__(FakeGpu)
    tel = Telemetry(out, hz=10, gpu=fake, lhm_url=None)
    fake.__init__(tel)
    tel.start()
    tel.set_phase("idle", "selftest"); time.sleep(1.0)
    tel.set_phase("load"); time.sleep(2.0)
    tel.set_phase("cooldown"); time.sleep(0.5)
    tel.stop()
    rows = list(csv.DictReader(open(out)))
    hz = (len(rows) - 1) / ((int(rows[-1]["t_mono_ns"]) - int(rows[0]["t_mono_ns"])) / 1e9)
    load = [r for r in rows if r["phase"] == "load"]
    e = (float(load[-1]["gpu_energy_mj"]) - float(load[0]["gpu_energy_mj"])) / 1000 / 3600
    span = (int(load[-1]["t_mono_ns"]) - int(load[0]["t_mono_ns"])) / 1e9
    expect = 100.0 * span / 3.6                          # FakeGpu draws 100 W in "load": mWh over the sampled span
    print(f"samples {len(rows)}  achieved rate {hz:.2f} Hz  late {tel.n_late}  load energy {e*1000:.3f} mWh (expect {expect:.1f})")
    assert 9.0 < hz < 11.0, "sampling rate off"
    assert abs(e * 1000 - expect) / expect < 0.02, "energy integration off by more than 2 %"
    print("SELFTEST PASS:", out)


def _probe():
    gpu = NvmlSource()
    lhm = LhmSource()
    s = gpu.read()
    try:
        s["cpu_pkg_power_w"] = lhm.read_once()
    except Exception as e:
        s["cpu_pkg_power_w"] = NAN
        print("LHM not reachable:", repr(e))
    s["cpu_sensor"] = lhm.path
    vals, t_end = [], time.monotonic() + 5.0
    while time.monotonic() < t_end and lhm.path:
        try:
            vals.append(lhm.read_once())
        except Exception:
            pass
        time.sleep(0.1)
    changes = sum(1 for a, b in zip(vals, vals[1:]) if a != b)
    s["cpu_sensor_refresh_s"] = round(5.0 / changes, 2) if changes else NAN
    print(json.dumps({**machine_info(gpu), **s}, indent=1, default=str))
    ok = not math.isnan(s["gpu_power_w"]) and not math.isnan(s["cpu_pkg_power_w"])
    if ok and not (s["cpu_sensor_refresh_s"] <= 1.0):
        print(f"CPU package power refreshes every {s['cpu_sensor_refresh_s']} s: set LibreHardwareMonitor "
              "Options -> Update Interval to 500 ms or 1 s, then probe again")
        ok = False
    print("PROBE", "PASS" if ok else "FAIL (GPU power or CPU package power missing; is LibreHardwareMonitor's web server running as admin on :8085?)")
    if ok:
        print("CPU sensor used:", lhm.path, "<- must be the CPU package, not a GPU sensor")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--probe", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    elif a.probe:
        _probe()
    else:
        ap.print_help()
