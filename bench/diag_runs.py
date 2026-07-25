#!/usr/bin/env python
"""
bench/diag_runs.py -- Diagnostic A.

Question: the seen grid and the held-out grid were collected under conditions
that differed. Local models were perfectly deterministic across 5 repeats in one
run and wildly unstable in the other, at the same recorded temperature. Until we
know what differed, the seen->held-out accuracy drop cannot be attributed to
generalization.

This script does not fix anything. It reads both existing CSVs and reports every
difference it can see in the data itself: when each run happened, how long it
took, what temperature was recorded, and exactly how unstable each model was
across repeats.

Read-only. Never writes to the CSVs.

Usage:
    python bench/diag_runs.py
    python bench/diag_runs.py --seen bench/results_seen.csv --held bench/results_heldout.csv
"""

import argparse
import collections
import csv
import datetime as dt
from pathlib import Path


def load(path):
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"not found: {p}\nPass the right paths with --seen / --held")
    with p.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f)), p


def parse_ts(s):
    try:
        return dt.datetime.fromisoformat(s)
    except Exception:
        return None


def timing(rows):
    ts = sorted(t for t in (parse_ts(r.get("ts_iso", "")) for r in rows) if t)
    if not ts:
        return None
    return {"first": ts[0], "last": ts[-1], "span_h": (ts[-1] - ts[0]).total_seconds() / 3600.0}


def gaps(rows, threshold_min=20):
    """Long pauses inside a run -- machine slept, restarted, or resumed later."""
    ts = sorted(t for t in (parse_ts(r.get("ts_iso", "")) for r in rows) if t)
    out = []
    for a, b in zip(ts, ts[1:]):
        d = (b - a).total_seconds() / 60.0
        if d >= threshold_min:
            out.append((a, b, d))
    return out


def instability(rows):
    """Per model: fraction of (command, strategy) groups whose RAW output was not
    identical across all repeats. This is the number that differed between runs."""
    groups = collections.defaultdict(set)
    for r in rows:
        key = (r.get("model", ""), r.get("strategy", ""), r.get("command_id", ""))
        groups[key].add(r.get("raw", ""))
    per_model = collections.defaultdict(lambda: [0, 0])
    for (model, _strat, _cid), raws in groups.items():
        per_model[model][1] += 1
        if len(raws) > 1:
            per_model[model][0] += 1
    return {m: (v[0], v[1], v[0] / v[1] if v[1] else 0.0) for m, v in per_model.items()}


def latency_by_model(rows):
    acc = collections.defaultdict(list)
    for r in rows:
        try:
            acc[r.get("model", "")].append(float(r["latency_s"]))
        except (ValueError, KeyError, TypeError):
            pass
    out = {}
    for m, v in acc.items():
        v.sort()
        out[m] = {"n": len(v), "median": v[len(v) // 2] if v else None,
                  "min": v[0] if v else None, "max": v[-1] if v else None}
    return out


def column_values(rows, col):
    return collections.Counter(r.get(col, "") for r in rows)


def report(name, rows, path):
    print("=" * 74)
    print(f"{name}   {path}   {len(rows):,} rows")
    print("=" * 74)

    t = timing(rows)
    if t:
        print(f"  first row : {t['first']}")
        print(f"  last row  : {t['last']}")
        print(f"  wall span : {t['span_h']:.1f} h")
    g = gaps(rows)
    if g:
        print(f"  PAUSES >=20 min inside the run: {len(g)}")
        for a, b, d in g[:6]:
            print(f"     {a} -> {b}   ({d:.0f} min)")
        if len(g) > 6:
            print(f"     ... and {len(g)-6} more")
    else:
        print("  no pauses >=20 min (run was continuous)")

    print(f"  temperature values recorded: {dict(column_values(rows, 'temperature'))}")

    print("\n  RAW-OUTPUT INSTABILITY across repeats  (0% = perfectly deterministic)")
    inst = instability(rows)
    for m in sorted(inst):
        varied, total, frac = inst[m]
        print(f"    {m:24s} {varied:5d}/{total:<5d} groups varied   {frac:6.1%}")

    print("\n  LATENCY (s)")
    for m, s in sorted(latency_by_model(rows).items()):
        print(f"    {m:24s} n={s['n']:<6d} median={s['median']:.2f}  "
              f"min={s['min']:.2f}  max={s['max']:.2f}")
    print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", default="bench/results_seen.csv")
    ap.add_argument("--held", default="bench/results_heldout.csv")
    args = ap.parse_args()

    seen, sp = load(args.seen)
    held, hp = load(args.held)

    report("SEEN GRID", seen, sp)
    report("HELD-OUT GRID", held, hp)

    print("=" * 74)
    print("SIDE BY SIDE -- instability")
    print("=" * 74)
    si, hi = instability(seen), instability(held)
    print(f"  {'model':24s} {'seen':>10s} {'held-out':>10s}   verdict")
    for m in sorted(set(si) | set(hi)):
        s = si.get(m, (0, 0, 0.0))[2]
        h = hi.get(m, (0, 0, 0.0))[2]
        flag = "SAME" if abs(s - h) < 0.05 else "DIFFERENT"
        print(f"  {m:24s} {s:9.1%} {h:9.1%}   {flag}")

    st, ht = timing(seen), timing(held)
    if st and ht:
        gap_days = abs((ht["first"] - st["last"]).total_seconds()) / 86400.0
        print(f"\n  gap between the two runs: {gap_days:.1f} days")
        print("  -> if this is more than a day, check whether Ollama was updated in between:")
        print("     ollama --version      (and compare to any note you kept)")


if __name__ == "__main__":
    main()
