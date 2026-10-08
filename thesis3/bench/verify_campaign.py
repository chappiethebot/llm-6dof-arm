"""Completeness gate for a thesis3 campaign: proves that every planned run and every
measurement exists, is complete and was taken under the same conditions, BEFORE any
number goes into the thesis.

    python verify_campaign.py thesis3/out/c1 --partial            # after each block
    python verify_campaign.py thesis3/out/c1 --after-analysis     # at the end: the final gate
    python verify_campaign.py D:/thesis3_backup/c1 --check-manifest   # is the backup identical?

The expected systems come from systems_expected.csv next to this script (66 systems with
family, kind, uses_cloud, repeats, legend), NOT from anything the harness wrote, so a
system the harness forgot is caught. Writes verify_report.txt and MANIFEST.sha256 (sha256
of every file). Exit code 0 = every check passed, 1 = at least one FAIL (listed at the end).

The defaults are the real protocol. The synthetic self-test runs with
--subset --min-load-s 20 --min-idle-s 29 --cooldown-s 4 --loadedidle-s 5."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("root")
ap.add_argument("--expected", default=os.path.join(HERE, "systems_expected.csv"))
ap.add_argument("--splits", default="held_out_v3,held_out_v4c")
ap.add_argument("--n", type=int, default=217)
ap.add_argument("--warmup", type=int, default=3)
ap.add_argument("--row-passes", type=int, default=20)
ap.add_argument("--min-load-s", type=float, default=300.0)
ap.add_argument("--min-idle-s", type=float, default=290.0)
ap.add_argument("--max-idle-age-min", type=float, default=60.0)
ap.add_argument("--cooldown-s", type=float, default=55.0)
ap.add_argument("--loadedidle-s", type=float, default=55.0)
ap.add_argument("--max-start-temp-c", type=float, default=51.0)
ap.add_argument("--min-hz", type=float, default=9.0)
ap.add_argument("--max-cloud-error-share", type=float, default=0.02)
ap.add_argument("--subset", action="store_true", help="smoke / synthetic: only the systems present are expected")
ap.add_argument("--partial", action="store_true", help="campaign still running: missing planned runs are WARN, not FAIL")
ap.add_argument("--after-analysis", action="store_true")
ap.add_argument("--check-manifest", action="store_true")
a = ap.parse_args()
R = a.root
fails, warns, lines = [], [], []
SPLITS = a.splits.split(",")


def log(level, msg):
    lines.append(f"{level:5s} {msg}")
    (fails if level == "FAIL" else warns if level == "WARN" else []).append(msg)


def sha256(path, buf=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(buf), b""):
            h.update(b)
    return h.hexdigest()


def all_files():
    for d, _, fs in os.walk(R):
        for f in fs:
            if f not in ("MANIFEST.sha256", "verify_report.txt"):
                yield os.path.relpath(os.path.join(d, f), R).replace("\\", "/")


def empty(v) -> bool:
    return v is None or (isinstance(v, float) and np.isnan(v)) or str(v).strip() in ("", "nan", "None", "<NA>")


# ------------------------------------------------------------ backup check only
if a.check_manifest:
    man = os.path.join(R, "MANIFEST.sha256")
    if not os.path.exists(man):
        sys.exit(f"FAIL no MANIFEST.sha256 in {R}")
    bad, listed = 0, set()
    for ln in open(man, encoding="utf-8"):
        h, rel = ln.rstrip("\n").split("  ", 1)
        listed.add(rel)
        p = os.path.join(R, rel)
        if not os.path.exists(p):
            print("FAIL missing", rel); bad += 1
        elif sha256(p) != h:
            print("FAIL changed", rel); bad += 1
    print("MANIFEST", "PASS" if bad == 0 else f"FAIL ({bad} files)", f"({len(listed)} files listed)")
    sys.exit(1 if bad else 0)

# ------------------------------------------------------------ files
for f in ["machine.json", "runs.csv", "idle.csv"]:
    log("PASS" if os.path.exists(os.path.join(R, f)) else "FAIL", f"file {f}")
tel_files = sorted(f for f in os.listdir(R) if f.startswith("telemetry_") and f.endswith(".csv"))
log("PASS" if tel_files else "FAIL", f"telemetry files: {len(tel_files)}")
if not os.path.exists(a.expected):
    log("FAIL", f"{a.expected} missing: cannot prove every system was measured")
if fails:
    print("\n".join(lines)); sys.exit(1)

S = {c: str for c in ["run_id", "session", "idle_id", "system", "family", "split", "kind", "legend", "dataset_sha256",
                      "uses_cloud", "error", "phase", "verb_true", "loc_true", "row_true", "col_true", "row_word",
                      "legend_expected", "repeats", "verb_pred", "loc_pred"]}
runs = pd.read_csv(os.path.join(R, "runs.csv"), dtype=S)
idle_tab = pd.read_csv(os.path.join(R, "idle.csv"), dtype=S)
exp = pd.read_csv(a.expected, dtype=str)
machine = json.load(open(os.path.join(R, "machine.json"), encoding="utf-8"))
tel = {f[len("telemetry_"):-4]: pd.read_csv(os.path.join(R, f), dtype=S,
                                            usecols=["t_mono_ns", "run_id", "phase", "gpu_power_w", "cpu_pkg_power_w", "gpu_temp_c"])
       for f in tel_files}
for t in tel.values():
    t["run_id"] = t["run_id"].fillna("")

# ------------------------------------------------------------ expected systems x test sets
want = {(s, sp) for s in exp.system for sp in SPLITS}
have = list(zip(runs.system, runs.split))
if a.subset:
    want = {k for k in want if k[0] in set(runs.system)}
dup = sorted({k for k in have if have.count(k) > 1})
missing = sorted(want - set(have))
unknown = sorted({s for s, _ in have} - set(exp.system))
log("PASS" if not dup else "FAIL", "no duplicate (system, test set)" + (f": {dup}" if dup else ""))
log("PASS" if not missing else ("WARN" if a.partial else "FAIL"),
    f"expected runs present: {len(want) - len(missing)} / {len(want)}" + (f"; missing {missing[:8]}{' ...' if len(missing) > 8 else ''}" if missing else ""))
log("PASS" if not unknown else "FAIL", "every system is in systems_expected.csv" + (f"; unknown {unknown}" if unknown else ""))
bad_split = sorted(set(runs.split) - set(SPLITS))
log("PASS" if not bad_split else "FAIL", f"only the test sets {SPLITS}" + (f"; found {bad_split}" if bad_split else ""))
dup_idle = idle_tab.idle_id[idle_tab.idle_id.duplicated()].tolist()
log("PASS" if not dup_idle else "FAIL", "idle ids unique" + (f"; repeated {dup_idle}" if dup_idle else ""))

# ------------------------------------------------------------ same data and legend everywhere
ref_sha = machine.get("datasets") or {}
for sp, d in runs.groupby("split"):
    shas = d.dataset_sha256 if "dataset_sha256" in d else pd.Series([None] * len(d))
    ok = shas.map(lambda v: not empty(v)).all() and shas.nunique() == 1 and ref_sha.get(sp) in set(shas)
    log("PASS" if ok else "FAIL", f"{sp}: every run used the dataset whose sha256 is in machine.json"
        + ("" if ok else f" (run shas: {sorted(set(shas.fillna('<empty>').astype(str)))[:3]}, machine.json: {ref_sha.get(sp)})"))
e_by = exp.set_index("system")
for r in runs.itertuples():
    if r.system not in e_by.index:
        continue
    e = e_by.loc[r.system]
    probs = []
    le, lg = str(e.get("legend_expected", "")), getattr(r, "legend", None)
    if empty(lg):
        probs.append("legend empty")
    elif le in ("v5", "none") and lg != le:
        probs.append(f"legend {lg}, expected {le}")
    for col in ["kind", "uses_cloud", "repeats", "family"]:
        if col in e and not empty(e[col]) and str(getattr(r, col, "")).replace(".0", "") != str(e[col]):
            probs.append(f"{col} {getattr(r, col, '')}, expected {e[col]}")
    if probs:
        log("FAIL", f"{r.run_id} {r.system} {r.split}: " + "; ".join(probs))
for s, d in runs.groupby("system"):
    if d.legend.nunique() > 1:
        log("FAIL", f"{s}: different legends on the two test sets {sorted(d.legend.unique())}")

# ------------------------------------------------------------ orphans (a run that crashed before its runs.csv row)
known = set(runs.run_id)
for sub, ext in [("calls", ".csv.gz"), ("latency", ".npy"), ("passes", ".csv"), ("outputs", ".jsonl")]:
    dd = os.path.join(R, sub)
    orph = sorted(f[: -len(ext)] for f in (os.listdir(dd) if os.path.isdir(dd) else []) if f.endswith(ext) and f[: -len(ext)] not in known)
    if orph:
        log("FAIL", f"{sub}/ files without a runs.csv row (incomplete run, re-run it with --redo): {orph[:8]}")

# ------------------------------------------------------------ per run
truth = {}
v3_start = {r.system: float(r.t_load_start_utc_ns) for r in runs.itertuples()
            if r.split == "held_out_v3" and not empty(getattr(r, "t_load_start_utc_ns", None))}
pass1_out = {}
for r in runs.itertuples():
    rid, tag = r.run_id, f"{r.run_id} {r.system} {r.split}"
    P = []
    try:
        rep = int(float(r.repeats))
    except (TypeError, ValueError):
        rep = 1
        P.append("repeats empty")
    cloud = str(getattr(r, "uses_cloud", "0")) in ("1", "1.0", "True") or r.kind == "cloud"
    paths = {"calls": os.path.join(R, "calls", f"{rid}.csv.gz"), "latency": os.path.join(R, "latency", f"{rid}.npy"),
             "passes": os.path.join(R, "passes", f"{rid}.csv"), "outputs": os.path.join(R, "outputs", f"{rid}.jsonl")}
    miss = [k for k, p in paths.items() if not os.path.exists(p)]
    if miss:
        log("FAIL", f"{tag}: missing files {miss}")
        continue
    c = pd.read_csv(paths["calls"], dtype=S, low_memory=False)
    lat = np.load(paths["latency"])
    ps = pd.read_csv(paths["passes"])
    p0, p1, ld = c[c["pass"] == 0], c[c["pass"] == 1], c[c["pass"] >= 1]
    # commands and repeats
    if len(p0) != a.warmup or set(p0.cmd_id.astype(int)) != set(range(a.warmup)):
        P.append(f"warm-up is not exactly commands 0..{a.warmup - 1} ({sorted(p0.cmd_id.astype(int))})")
    cnt = p1.groupby("cmd_id").size()
    if set(p1.cmd_id.astype(int)) != set(range(a.n)) or not (cnt == rep).all():
        P.append(f"pass 1 is not every command 0..{a.n - 1} exactly {rep}x ({p1.cmd_id.nunique()} commands, {len(p1)} rows)")
    for col in ["action_ok", "verb_ok", "valid_json", "verb_true"]:
        if col not in ld or ld[col].isna().any():
            P.append(f"empty {col} in {int(ld[col].isna().sum()) if col in ld else 'all'} load rows")
    # passes, latency file and calls file agree
    if list(ps["pass"]) != list(range(1, len(ps) + 1)):
        P.append("passes are not numbered 1..P")
    if int(ps.calls.sum()) != len(lat):
        P.append(f"latency file has {len(lat)} calls, passes file {int(ps.calls.sum())}")
    if not np.all(np.isfinite(lat)) or (lat <= 0).any():
        P.append("latency file has non-positive or non-finite values")
    incomplete = ps[ps.calls != a.n * rep]
    if len(incomplete):
        P.append(f"passes not complete (every pass is all {a.n} commands x {rep}): {incomplete['pass'].tolist()[:5]}")
    rp = min(len(ps), a.row_passes)
    for p in range(1, rp + 1):
        if (ld["pass"] == p).sum() != int(ps.loc[ps["pass"] == p, "calls"].sum()):
            P.append(f"pass {p}: calls file and passes file disagree"); break
    if ld["pass"].max() > a.row_passes if len(ld) else False:
        P.append(f"call rows beyond pass {a.row_passes}")
    if not empty(getattr(r, "calls_total", None)) and int(float(r.calls_total)) != len(lat):
        P.append(f"runs.csv calls_total {r.calls_total} != {len(lat)}")
    t0, t1 = float(r.t_load_start_mono_ns), float(r.t_load_end_mono_ns)
    T = (t1 - t0) / 1e9
    if ld[["t_start_mono_ns", "t_end_mono_ns"]].isna().any().any():
        P.append("calls without time stamps")
    elif ((ld.t_start_mono_ns.astype(float) < t0 - 1e6) | (ld.t_end_mono_ns.astype(float) > t1 + 1e6)).any():
        P.append("calls outside the load window")
    if T < a.min_load_s:
        P.append(f"load window {T:.0f} s < {a.min_load_s:.0f} s")
    err = int(p1.error.notna().sum()) if "error" in p1 else 0
    if (not cloud and err > 0) or (cloud and err > a.max_cloud_error_share * max(len(p1), 1)):
        P.append(f"{err} failed calls in pass 1")
    t_start = float(getattr(r, "gpu_temp_start_c", np.nan))
    if not np.isnan(t_start) and t_start > a.max_start_temp_c:
        P.append(f"started at {t_start:.0f} C")
    # telemetry
    if r.session not in tel:
        P.append(f"telemetry_{r.session}.csv missing")
    else:
        tt = tel[r.session]
        mt = tt[tt.run_id == rid]
        w = mt[(mt.phase == "load") & (mt.t_mono_ns.astype(float) >= t0) & (mt.t_mono_ns.astype(float) <= t1)]
        hz = (len(w) - 1) / T if T > 0 and len(w) > 1 else 0.0
        gaps = np.diff(w.t_mono_ns.astype(float).values) / 1e9 if len(w) > 1 else np.array([np.inf])
        if hz < a.min_hz:
            P.append(f"sampling {hz:.2f} Hz < {a.min_hz}")
        if gaps.max() > max(5.0, 0.01 * T) or (gaps > 0.2).mean() > 0.01:
            P.append(f"telemetry gaps (max {gaps.max():.2f} s, {100 * (gaps > 0.2).mean():.1f} % > 0.2 s)")
        if len(w) and w.gpu_power_w.isna().mean() > 0.01:
            P.append(f"GPU power missing in {100 * w.gpu_power_w.isna().mean():.1f} % of samples")
        if len(w) and w.cpu_pkg_power_w.isna().mean() > 0.05:
            P.append(f"CPU power missing in {100 * w.cpu_pkg_power_w.isna().mean():.1f} % of samples (energy incomplete)")
        if len(w) > 20 and w.cpu_pkg_power_w.notna().any() and w.cpu_pkg_power_w.astype(float).nunique() == 1:
            P.append("CPU power constant over the whole window (stuck sensor)")
        cd = mt[mt.phase == "cooldown"]
        if len(cd) < 2 or np.ptp(cd.t_mono_ns.astype(float)) / 1e9 < a.cooldown_s:
            P.append("cool-down missing or short")
        wdur = (p0.t_end_mono_ns.astype(float).max() - p0.t_start_mono_ns.astype(float).min()) / 1e9 if len(p0) else 0
        if wdur >= 0.2 and not (mt.phase == "warmup").any():
            P.append("no warm-up telemetry")
        if r.kind == "local" and r.split == "held_out_v3":
            if not (mt.phase == "modelload").any():
                P.append("no 'modelload' telemetry (cold start not measured)")
            li = mt[mt.phase == "loadedidle"]
            if len(li) < 2 or np.ptp(li.t_mono_ns.astype(float)) / 1e9 < a.loadedidle_s:
                P.append("'loadedidle' missing or short (standby not measured)")
            if empty(getattr(r, "model_load_s", None)):
                P.append("model_load_s empty")
    if r.kind == "local" and r.system not in v3_start:
        P.append("no held_out_v3 run for this system yet (cold start and standby come from it)")
    # idle baseline
    ir = idle_tab[idle_tab.idle_id == r.idle_id]
    if len(ir) != 1:
        P.append(f"idle window {r.idle_id} not in idle.csv")
    else:
        ir = ir.iloc[0]
        it = tel.get(ir.session)
        iw = it[(it.run_id == r.idle_id) & (it.phase == "idle")] if it is not None else pd.DataFrame()
        dur = np.ptp(iw.t_mono_ns.astype(float)) / 1e9 if len(iw) > 1 else 0.0
        if dur < a.min_idle_s:
            P.append(f"idle window {dur:.0f} s < {a.min_idle_s:.0f} s")
        start = v3_start.get(r.system, float(getattr(r, "t_load_start_utc_ns", np.nan)))
        age = (start - float(ir.t_end_utc_ns)) / 6e10 if not empty(ir.get("t_end_utc_ns")) else np.nan
        if np.isnan(age) or age > a.max_idle_age_min or age < 0:
            P.append(f"idle baseline age at the system's start {age:.0f} min (limit {a.max_idle_age_min:.0f})")
    # raw outputs: pass 0 and every pass-1 call must be re-scorable without re-running
    seen = set()
    with open(paths["outputs"], encoding="utf-8") as f:
        for ln in f:
            o = json.loads(ln)
            if o.get("pass") in (0, 1):
                seen.add((o["pass"], int(o["cmd_id"]), int(o.get("repeat", 1))))
                if o.get("pass") == 1 and int(o.get("repeat", 1)) == 1:
                    pass1_out[(rid, int(o["cmd_id"]))] = o.get("raw")
    need = {(0, int(x), 1) for x in p0.cmd_id} | {(1, int(x), int(y)) for x, y in zip(p1.cmd_id, p1["repeat"])}
    if need - seen:
        P.append(f"raw outputs missing for {len(need - seen)} pass-0/pass-1 calls")
    # the same truth for every run of a test set
    tr = p1.drop_duplicates("cmd_id").set_index("cmd_id")[[x for x in ["verb_true", "loc_true", "row_true", "col_true"] if x in p1]]
    truth.setdefault(r.split, []).append((rid, tr.sort_index().fillna("").astype(str)))
    log("PASS" if not P else "FAIL", tag + ("" if not P else ": " + "; ".join(P)))

for sp, lst in truth.items():
    base_id, base = lst[0]
    diff = [rid for rid, tr in lst[1:] if not tr.equals(base)]
    log("PASS" if not diff else "FAIL", f"{sp}: every run scored against identical ground truth"
        + (f"; differs from {base_id}: {diff[:8]}" if diff else ""))

# speculative systems must reproduce the plain LoRA exactly (byte-identical pass-1 output)
for sp in SPLITS:
    base = runs[(runs.system == "lora_greedy") & (runs.split == sp)]
    for s in ["a8_speculative", "a10_speculative", "a12_smalllm"]:
        rr = runs[(runs.system == s) & (runs.split == sp)]
        if base.empty or rr.empty:
            continue
        b, x = base.run_id.iloc[0], rr.run_id.iloc[0]
        nd = sum(pass1_out.get((b, i)) != pass1_out.get((x, i)) for i in range(a.n))
        log("PASS" if nd == 0 else "FAIL", f"{sp}: {s} byte-identical to lora_greedy" + (f" ({nd} commands differ)" if nd else ""))

# ------------------------------------------------------------ analysis outputs
if a.after_analysis:
    for f in ["summary.csv", "per_verb.csv", "per_loc_type.csv", "confusion.csv", "per_pass.csv",
              "per_command_energy.csv", "compare.csv", "quality.csv", "report.md"]:
        log("PASS" if os.path.exists(os.path.join(R, f)) else "FAIL", f"analysis file {f}")
    sp_ = os.path.join(R, "summary.csv")
    if os.path.exists(sp_):
        s = pd.read_csv(sp_, dtype={"run_id": str})
        miss = sorted(set(runs.run_id) - set(s.run_id))
        log("PASS" if not miss else "FAIL", "summary.csv covers every run" + (f"; missing {miss[:10]}" if miss else ""))
        core = ["action_acc", "action_wilson_lo", "verb_acc", "lat_p50_ms", "lat_p95_ms", "lat_p99_ms", "dec_lat_p95_ms",
                "throughput_cmd_per_s", "gpu_mean_w", "cpu_mean_w", "net_wh_per_1k", "net_kwh_per_million",
                "co2_g_per_million", "eur_per_million", "device_hours_per_million"]
        local = ["vram_peak_mb", "model_load_s", "loaded_idle_net_w", "wh_per_cmd_duty"]
        for r in s.itertuples():
            cols = core + (local if r.kind == "local" else [])
            gone = [c for c in cols if c not in s.columns or pd.isna(getattr(r, c, np.nan))]
            log("PASS" if not gone else "FAIL", f"metrics {r.run_id} {r.system} {r.split}" + (f": empty {gone}" if gone else ""))
        lt = os.path.join(R, "per_loc_type.csv")
        if os.path.exists(lt):
            pl = pd.read_csv(lt)
            for sp, d in pl.groupby("split"):
                shapes = d.pivot_table(index="run_id", columns="loc_type", values="n", aggfunc="sum").fillna(0)
                log("PASS" if len(shapes.drop_duplicates()) == 1 else "FAIL", f"{sp}: identical location-type counts in every run "
                    f"({ {k: int(v) for k, v in shapes.iloc[0].items()} })")

# ------------------------------------------------------------ manifest
files = sorted(all_files())
with open(os.path.join(R, "MANIFEST.sha256"), "w", encoding="utf-8") as f:
    for rel in files:
        f.write(f"{sha256(os.path.join(R, rel))}  {rel}\n")
log("PASS", f"MANIFEST.sha256 written ({len(files)} files)")

verdict = "PASS" if not fails else f"FAIL ({len(fails)} checks)"
lines.append(f"\nVERDICT {verdict}; {len(warns)} warnings")
open(os.path.join(R, "verify_report.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
print("\n".join(lines))
sys.exit(0 if not fails else 1)
