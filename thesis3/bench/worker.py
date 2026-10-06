"""thesis3 worker: runs ONE system in its own conda env, under the parent's telemetry.

    <env python> thesis3/bench/worker.py --system NAME --root DIR --runs JSON [...]
    <env python> thesis3/bench/worker.py --versions

Started by run_bench.py, never by hand. Loads the system (registry.build), then for each
run: warm-up (3 calls, commands 0-2), load passes (shuffled, >= 1 full pass and
>= min-load-s, the pass in progress finished), cool-down, then writes the run's files in
SCHEMA.md order. Every phase change goes to the parent as one JSON line on stdout with this
process's perf_counter_ns(); the worker waits for the parent's "ok" (telemetry phase set)
before it starts the phase. Everything else the system prints goes to stderr (the log).
"""
from __future__ import annotations

import os
import sys

# stdout is the protocol channel: keep a private handle, send every other print to stderr
_PROTO = os.fdopen(os.dup(1), "w", encoding="utf-8", buffering=1)
os.dup2(2, 1)
sys.stdout = sys.stderr

import argparse  # noqa: E402
import csv  # noqa: E402
import gc  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import registry as REG  # noqa: E402

ns = time.perf_counter_ns
SEED = 20260827
COLS = ["run_id", "pass", "order_idx", "cmd_id", "repeat", "t_start_mono_ns", "t_end_mono_ns", "latency_ms",
        "t_infer_start_mono_ns", "t_infer_end_mono_ns", "valid_json", "action_ok", "verb_ok", "target_ok", "loc_ok",
        "verb_true", "verb_pred", "loc_true", "loc_pred", "row_true", "row_pred", "col_true", "col_pred", "row_word",
        "pred_hash", "confidence", "tokens_in", "tokens_out", "tokens_thinking", "ttft_ms", "error", "extra"]
SEEN_W, UNSEEN_W = ("middle", "center"), ("upper", "bottom", "back", "front")


def send(msg: dict, ack=False):
    _PROTO.write(json.dumps(msg) + "\n")
    _PROTO.flush()
    if ack:
        line = sys.stdin.readline().strip()
        if line != "ok":
            raise SystemExit("parent did not acknowledge %s (got %r)" % (msg.get("phase") or msg.get("msg"), line))


def phase(name, run_id):
    send({"msg": "phase", "phase": name, "run_id": run_id, "t_mono_ns": ns(), "t_utc_ns": time.time_ns()}, ack=True)


# ---------------------------------------------------------------- scoring (llm/score.py)
def rowof(loc):
    return loc[5] if isinstance(loc, str) and loc.startswith("grid_") and len(loc) == 7 else None


def colof(loc):
    return loc[6] if isinstance(loc, str) and loc.startswith("grid_") and len(loc) == 7 else None


def row_word(cmd):
    low = cmd.lower()
    seen = [w for w in SEEN_W if re.search(r"\b%s\b" % w, low)]
    unseen = [w for w in UNSEEN_W if re.search(r"\b%s\b" % w, low)]
    if seen and unseen:
        return "mixed"
    return (unseen or seen or [""])[0]


def _s(v):
    return "" if v is None else (v if isinstance(v, str) else json.dumps(v, sort_keys=True))


def score_fields(obj, truth, cmd, score):
    """per-field correctness, the same definitions as thesis2/final/common.score_row"""
    tl, pl = truth.get("location"), obj.get("location")
    grid = rowof(tl) is not None
    return {"valid_json": int(bool(obj)), "action_ok": int(bool(score(obj, truth).strict_match)),
            "verb_ok": int(obj.get("action") == truth.get("action")),
            "target_ok": "" if not truth.get("target") else int(obj.get("target") == truth.get("target")),
            "loc_ok": "" if not tl else int(pl == tl),
            "verb_true": truth.get("action"), "verb_pred": _s(obj.get("action")),
            "loc_true": tl or "", "loc_pred": _s(pl),
            "row_true": rowof(tl) or "", "row_pred": (rowof(pl) or "") if grid else "",
            "col_true": colof(tl) or "", "col_pred": (colof(pl) or "") if grid else "",
            "row_word": row_word(cmd) if grid else ""}


def pred_hash(obj):
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]


def _fsync(f):
    f.flush()
    os.fsync(f.fileno())


# ---------------------------------------------------------------- one run
def run_one(s, built, meter, run, a, score):
    rid, split, ridx = run["run_id"], run["split"], run["run_index"]
    data = [json.loads(l) for l in (REG.REPO / "data" / ("%s.jsonl" % split)).read_text(encoding="utf-8").splitlines() if l.strip()]
    cmds = [r["command"] for r in data]
    truths = []
    for r in data:
        t = dict(r["expected"])
        t.setdefault("modifiers", [])
        truths.append(t)
    N, R = len(cmds), s.repeats
    if built.prepare:
        built.prepare(split, data)
    predict, finish, active = built.predict, built.finish, meter.active
    rows, outputs, first = [], [], {}
    err_text = None

    def call(cid):
        nonlocal err_text
        if active:
            meter.reset()
        t_s = ns()
        try:
            raw, obj, info = predict(cmds[cid], truths[cid])
            err_text = None
        except Exception as e:                                         # noqa: BLE001
            raw, obj, info = "", {}, None
            err_text = ("%s: %s" % (type(e).__name__, e))[:500]
        return t_s, ns(), raw, obj if isinstance(obj, dict) else {}, info

    def full(p, idx, cid, rep, t_s, t_e, raw, obj, info):
        raw_s = REG._json_raw(raw, obj)
        f = score_fields(obj, truths[cid], cmds[cid], score)
        fin = {}
        if info is not None:
            try:
                fin = finish(raw_s, obj, info, meter, t_s, t_e) or {}
            except Exception as e:                                     # noqa: BLE001
                fin = {"error": "finish: %s: %s" % (type(e).__name__, e)}
        err = err_text or fin.get("error") or ""
        if err:
            f["action_ok"] = 0                                          # a failed call is scored wrong
        ph = pred_hash(obj)
        i0, i1 = (meter.t0, meter.t1) if active and meter.t0 is not None else (t_s, t_e)
        rec = {"run_id": rid, "pass": p, "order_idx": idx, "cmd_id": cid, "repeat": rep, "t_start_mono_ns": t_s,
               "t_end_mono_ns": t_e, "latency_ms": (t_e - t_s) / 1e6, "t_infer_start_mono_ns": i0,
               "t_infer_end_mono_ns": i1, **f, "pred_hash": ph,
               **{k: fin.get(k, "") for k in ("confidence", "tokens_in", "tokens_out", "tokens_thinking", "ttft_ms")},
               "error": err, "extra": json.dumps(fin.get("extra", {}))}
        for k in ("tokens_in", "tokens_out", "tokens_thinking", "confidence", "ttft_ms"):
            if rec[k] is None:
                rec[k] = ""
        rows.append(rec)
        if p in (0, 1):
            first[(cid, rep)] = (raw if raw is not None else obj, f["action_ok"], ph, bool(err))
        if p <= 1 or ph != first[(cid, rep)][2]:
            outputs.append({"run_id": rid, "pass": p, "cmd_id": cid, "repeat": rep, "raw": raw_s, "parsed": obj})
        return f["action_ok"], err

    # ---- warm-up: commands 0, 1, 2, pass 0 (not scored by the analysis)
    phase("warmup", rid)
    for i in range(a.warmup):
        t_s, t_e, raw, obj, info = call(i)
        full(0, i, i, 1, t_s, t_e, raw, obj, info)
    first.clear()

    # ---- load: shuffled full passes until >= 1 pass and >= min_load_s
    send({"msg": "thermal", "run_id": rid}, ack=True)      # the warm-up heats the GPU: parent waits for <= 50 C
    phase("load", rid)
    t0, utc0 = ns(), time.time_ns()
    lat = np.empty(1 << 16, dtype=np.float32)
    n = 0
    passes, p, idx, t_last = [], 0, 0, t0
    p1_err = 0
    while True:
        p += 1
        keep = p <= a.row_passes
        order = np.random.default_rng(SEED + p + ridx).permutation(N).tolist()
        ok_sum, p_start = 0, None
        for cid in order:
            for rep in range(1, R + 1):
                t_s, t_e, raw, obj, info = call(cid)
                if p_start is None:
                    p_start = t_s
                if n == len(lat):
                    lat = np.concatenate([lat, np.empty(len(lat), dtype=np.float32)])
                lat[n] = (t_e - t_s) / 1e6
                n += 1
                if keep:
                    ok, err = full(p, idx, cid, rep, t_s, t_e, raw, obj, info)
                    if p == 1 and err:
                        p1_err += 1
                else:
                    f0 = first[(cid, rep)]
                    key = raw if raw is not None else obj
                    if err_text is None and not f0[3] and key == f0[0]:
                        ok = f0[1]
                    else:
                        ok = 0 if err_text else int(bool(score(obj, truths[cid]).strict_match))
                ok_sum += ok
                idx += 1
                t_last = t_e
        passes.append({"pass": p, "t_start_mono_ns": p_start, "t_end_mono_ns": t_last, "calls": N * R,
                       "action_ok_sum": ok_sum})
        el = (t_last - t0) / 1e9
        send({"msg": "progress", "run_id": rid, "pass": p, "elapsed_s": round(el, 1), "calls": n,
              "pass_acc": round(ok_sum / (N * R), 4)})
        if el >= a.min_load_s:
            break
    t1, utc1 = t_last, utc0 + (t_last - t0)

    # ---- cool-down
    phase("cooldown", rid)
    time.sleep(a.cooldown_s)
    phase("between", "")

    # ---- files, in SCHEMA.md order; the parent writes the runs.csv row (commit marker)
    root = Path(a.root)
    for d in ("outputs", "calls", "latency", "passes"):
        (root / d).mkdir(parents=True, exist_ok=True)
    with open(root / "outputs" / ("%s.jsonl" % rid), "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, default=str) + "\n")
        _fsync(f)
    with open(root / "calls" / ("%s.csv.gz" % rid), "wb") as fb:
        with gzip.open(fb, "wt", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=COLS)
            w.writeheader()
            w.writerows(rows)
        _fsync(fb)
    with open(root / "latency" / ("%s.npy" % rid), "wb") as f:
        np.save(f, lat[:n])
        _fsync(f)
    with open(root / "passes" / ("%s.csv" % rid), "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pass", "t_start_mono_ns", "t_end_mono_ns", "calls", "action_ok_sum"])
        w.writeheader()
        w.writerows(passes)
        _fsync(f)

    # pass-1 gate figures: commands fully scored, errors, majority-vote action match
    p1 = [r for r in rows if r["pass"] == 1]
    per = {}
    for r in p1:
        per.setdefault(r["cmd_id"], []).append(int(r["action_ok"]))
    acc = sum(1 for v in per.values() if sum(v) * 2 > R) / max(len(per), 1)
    send({"msg": "run_done", "run_id": rid, "split": split,
          "t_load_start_mono_ns": t0, "t_load_end_mono_ns": t1, "t_load_start_utc_ns": utc0, "t_load_end_utc_ns": utc1,
          "passes": p, "row_passes": min(p, a.row_passes), "calls_total": n,
          "pass1_calls": len(p1), "pass1_cmds_complete": sum(1 for v in per.values() if len(v) == R),
          "pass1_errors": p1_err, "pass1_action_acc": acc}, ack=True)


def dry(s, a, score):
    """--dry: load, then commands 0-2 of each test set through predict + finish; no telemetry, no files."""
    meter = REG.Meter()
    sha = REG.model_sha(s)
    t = ns()
    built = REG.build(s, meter)
    send({"msg": "dry", "system": s.name, "model_load_s": round((ns() - t) / 1e9, 2), "model_sha256": sha})
    for split in REG.SPLITS:
        data = [json.loads(l) for l in (REG.REPO / "data" / ("%s.jsonl" % split)).read_text(encoding="utf-8").splitlines() if l.strip()]
        if built.prepare:
            built.prepare(split, data)
        for cid in range(a.warmup):
            truth = dict(data[cid]["expected"], modifiers=data[cid]["expected"].get("modifiers", []))
            meter.reset()
            t_s = ns()
            raw, obj, info = built.predict(data[cid]["command"], truth)
            t_e = ns()
            raw_s = REG._json_raw(raw, obj)
            fin = built.finish(raw_s, obj, info, meter, t_s, t_e) or {}
            f = score_fields(obj, truth, data[cid]["command"], score)
            send({"msg": "dry", "split": split, "cmd_id": cid, "latency_ms": round((t_e - t_s) / 1e6, 2),
                  "infer_ms": round((meter.t1 - meter.t0) / 1e6, 2) if meter.t0 else None,
                  "action_ok": f["action_ok"], "raw": raw_s[:160], "parsed": obj,
                  **{k: v for k, v in fin.items() if v not in ("", None, {})}})


def versions():
    out = {"python": sys.version.split()[0], "executable": sys.executable}
    for m in ("torch", "transformers", "peft", "numpy", "pandas", "openai", "ollama", "sentence_transformers", "sklearn"):
        try:
            mod = __import__(m)
            out[m] = getattr(mod, "__version__", "?")
        except Exception:                                              # noqa: BLE001
            out[m] = "missing"
    try:
        import torch
        out["cuda"] = torch.version.cuda
        out["cuda_available"] = torch.cuda.is_available()
    except Exception:                                                  # noqa: BLE001
        pass
    send(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system")
    ap.add_argument("--root")
    ap.add_argument("--runs", help="JSON list of {run_id, split, run_index}")
    ap.add_argument("--min-load-s", type=float, default=300.0)
    ap.add_argument("--cooldown-s", type=float, default=60.0)
    ap.add_argument("--loadedidle-s", type=float, default=60.0)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--row-passes", type=int, default=20)
    ap.add_argument("--versions", action="store_true")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    if a.versions:
        return versions()
    csv.field_size_limit(2 ** 31 - 1)
    s = REG.SYSTEMS[a.system]
    REG._setup_paths()
    from llm.score import score
    if a.dry:
        return dry(s, a, score)
    runs = json.loads(a.runs)

    sha = REG.model_sha(s)                         # frozen-checkpoint gate + weights hash, before the timed load
    meter = REG.Meter()
    phase("modelload", runs[0]["run_id"])
    t = ns()
    built = REG.build(s, meter)
    load_s = (ns() - t) / 1e9
    send({"msg": "loaded", "model_load_s": load_s, "model_sha256": sha, "notes": built.notes})
    if a.loadedidle_s > 0:                         # standby cost, local systems only
        phase("loadedidle", runs[0]["run_id"])
        time.sleep(a.loadedidle_s)
        phase("between", "")
    for run in runs:
        run_one(s, built, meter, run, a, score)
    del built
    gc.collect()
    if "torch" in sys.modules:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    send({"msg": "done"})


if __name__ == "__main__":
    main()
