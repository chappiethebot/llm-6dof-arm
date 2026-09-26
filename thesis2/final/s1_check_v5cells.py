"""thesis2/final/s1_check_v5cells.py -- audit the replicated V5 cell grid. REPORT ONLY.

    python thesis2/final/s1_check_v5cells.py [--split held_out_v3]

Reads thesis2/out/final/results_v5cells_<split>.csv and, for each local cell,
prints:

  rows                  rows written for the cell
  empty/error           rows that produced no usable object: stage != validated,
                        or raw empty, or pred_json {} -- counted separately as
                        prompt_error / model_error / extract_failed / schema_failed
  action match          recomputed here with llm/score.py, on this run's rows
  V5 action match       the SAME cell's number from bench/results_v5_*.csv, on
                        V5's own non-warmup denominator
  raw agreement         raw output identical to V5's, over the shared command_ids
  latency p50           median latency_s per row, where recorded

GATE (this script FAILS the run, it does not merely report):
  * any cell containing any `prompt_error` row -- a prompt_error is a crash
    BEFORE the model call, so those rows are not a result;
  * any cell whose empty-raw count exceeds V5's for the same cell;
  * on held_out_v3, any cell whose raw agreement with V5 is below 210/216.

Nothing is written, no model is loaded, and no held-out inference is performed:
this reads two CSV families and the scorer. s1_run_v5_cells.py invokes this
script at the end of every run and fails its own gate if this one fails.
"""
import argparse
import collections
import csv
import glob
import json
import os
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

csv.field_size_limit(2 ** 31 - 1)
V5_GLOB = str(C.BENCH / "results_v5_*.csv")
# The raw-agreement floor on held_out_v3. 216 non-warmup commands are compared;
# 210 allows six commands of sampling/server drift and no more. Set by the
# 2026-09-26 ruling on the V5-cells audit, not tuned against any result.
MIN_AGREE, MIN_SHARED = 210, 216
SKIP = ("_scored.csv", "_tokens.csv", "results_v5_all.csv", "results_v5_slim.csv",
        "results_v5_check.csv", "results_v5_smoke.csv")


def load_mine(split):
    p = C.OUT / ("results_v5cells_%s.csv" % split)
    if not p.exists():
        print("GATE FAIL: %s does not exist -- run s1_run_v5_cells.py first"
              % p.relative_to(C.REPO))
        raise SystemExit(2)
    return p, list(csv.DictReader(p.open(encoding="utf-8", newline="")))


def load_v5():
    """V5's own rows for every cell, keyed (model, strategy) -> {command_id: row}.

    Warm-up rows are kept but flagged, so the denominators can be stated rather
    than silently chosen.
    """
    out = collections.defaultdict(dict)
    warm = collections.defaultdict(set)
    for f in sorted(glob.glob(V5_GLOB)):
        b = os.path.basename(f)
        if any(b.endswith(s) or b == s for s in SKIP):
            continue
        for r in csv.DictReader(open(f, encoding="utf-8", newline="")):
            if r.get("dataset") != "held_out_v3":
                continue
            k = (r["model"], r["strategy"])
            cid = r["command_id"]
            if str(r.get("is_warmup", "")).lower() == "true":
                warm[k].add(cid)
                continue
            # V5 cloud cells have repeats; keep the FIRST repeat for a raw-text
            # comparison and record every repeat's outcome for the accuracy.
            out[k].setdefault(cid, []).append(r)
    return out, warm


def obj_of(row, col="pred_json"):
    try:
        o = json.loads(row[col] or "{}")
        return o if isinstance(o, dict) else {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", default="held_out_v3", choices=list(C.SPLITS))
    a = ap.parse_args()
    C.start_log("s1_check_v5cells_%s" % a.split)

    path, mine = load_mine(a.split)
    v5, v5warm = load_v5()
    print("  this run : %s  (%d rows)" % (path.relative_to(C.REPO), len(mine)))
    print("  V5 source: bench/results_v5_*.csv  (%d cells on held_out_v3)" % len(v5))
    if a.split != "held_out_v3":
        print("  NOTE: V5 never ran %s, so the V5 columns will read 'n/a'" % a.split)

    by_cell = collections.defaultdict(list)
    for r in mine:
        by_cell[(r["model"], r["strategy"])].append(r)

    print()
    print("=" * 152)
    print("%-26s %-11s %6s %7s %-34s %9s %9s %11s %9s"
          % ("model", "strategy", "rows", "warm", "empty/error rows (by stage)",
             "action", "V5 action", "raw agree", "lat p50 s"))
    print("=" * 152)

    totals = collections.Counter()
    flagged = []
    prompt_err_cells = []       # any prompt_error at all -> GATE FAIL
    worse_empty_cells = []      # empty raw count above V5's -> GATE FAIL
    low_agree_cells = []        # raw agreement below MIN_AGREE -> GATE FAIL (v3)
    for key in sorted(by_cell):
        rows = by_cell[key]
        warm = sum(1 for r in rows if str(r.get("is_warmup", "")).lower() == "true")
        stages = collections.Counter(r["stage"] for r in rows)
        bad = {s: n for s, n in stages.items() if s != "validated"}
        n_bad = sum(bad.values())
        # empty output is a separate, stricter count than "stage != validated"
        n_empty = sum(1 for r in rows if not (r.get("raw") or "").strip())
        n_noobj = sum(1 for r in rows if not obj_of(r))

        hits = sum(1 for r in rows
                   if bool(C.score(obj_of(r), json.loads(r["truth_json"])).strict_match))
        acc = hits / len(rows)

        vr = v5.get(key)
        if vr:
            v5_flat = [x for reps in vr.values() for x in reps]
            v5_hits = sum(1 for x in v5_flat
                          if str(x["strict_match"]).lower() == "true")
            v5_acc = "%.4f" % (v5_hits / len(v5_flat))
            shared = sorted(set(vr) & {r["command_id"] for r in rows}, key=int)
            mine_by = {r["command_id"]: r for r in rows}
            same = sum(1 for cid in shared
                       if (mine_by[cid].get("raw") or "").strip()
                       == (vr[cid][0].get("raw") or "").strip())
            agree = "%d/%d" % (same, len(shared))
            v5_empty = sum(1 for x in v5_flat if not (x.get("raw") or "").strip())
            if n_empty > v5_empty:
                worse_empty_cells.append((key, n_empty, v5_empty))
            if a.split == "held_out_v3" and shared and same < MIN_AGREE:
                low_agree_cells.append((key, same, len(shared)))
        else:
            v5_acc, agree = "n/a", "n/a"

        lats = [float(r["latency_s"]) for r in rows
                if (r.get("latency_s") or "").strip()
                and str(r["latency_s"]).replace(".", "", 1).replace("e-", "", 1).isdigit()
                or _isfloat(r.get("latency_s"))]
        lat = "%.3f" % statistics.median(lats) if lats else "not recorded"

        badstr = ", ".join("%s %d" % (s, n) for s, n in sorted(bad.items())) or "none"
        print("%-26s %-11s %6d %7d %-34s %9.4f %9s %11s %9s"
              % (key[0], key[1], len(rows), warm, badstr[:34], acc, v5_acc, agree, lat))
        totals["rows"] += len(rows)
        totals["bad"] += n_bad
        totals["empty"] += n_empty
        if n_bad == len(rows):
            flagged.append((key, sorted(bad), n_empty, n_noobj))
        n_prompt_err = stages.get("prompt_error", 0)
        if n_prompt_err:
            prompt_err_cells.append((key, n_prompt_err, len(rows)))

    print("=" * 152)
    print("  %d cells, %d rows; %d rows not 'validated'; %d rows with empty raw output"
          % (len(by_cell), totals["rows"], totals["bad"], totals["empty"]))

    if flagged:
        print()
        print("  CELLS WHERE **EVERY** ROW FAILED -- these cells carry no measurement:")
        for key, stages, n_empty, n_noobj in flagged:
            print("     %-26s %-11s stages=%s  empty raw=%d  no parsed object=%d"
                  % (key[0], key[1], stages, n_empty, n_noobj))
        ex = [r for r in mine if r["stage"] == "prompt_error"]
        if ex:
            print()
            print("  the recorded reason on those rows:")
            for reason, n in collections.Counter(r["reasons"] for r in ex).most_common(3):
                print("     %-70s x%d" % (reason[:70], n))

    print()
    print("  Denominators. This run writes %d rows per local cell at R = 1 and marks"
          % 217)
    print("  command_id 0 as the warm-up, as V5 did; V5's own cell accuracy above is")
    print("  computed on its NON-warmup rows (216 for a local cell, 1,084-1,085 for a")
    print("  cloud cell with 5 repeats). The two accuracy columns therefore use")
    print("  slightly different denominators and are comparable to within one command.")
    # ------------------------------------------------------------------ gate
    fails = []
    if prompt_err_cells:
        fails.append("%d cell(s) contain prompt_error rows -- a prompt_error means "
                     "NO MODEL CALL was made, so the cell carries no measurement:"
                     % len(prompt_err_cells))
        for key, n, tot in prompt_err_cells:
            fails.append("     %-26s %-11s %d/%d rows prompt_error" % (key + (n, tot)))
    if worse_empty_cells:
        fails.append("%d cell(s) have MORE empty-raw rows than V5 did for the same "
                     "cell:" % len(worse_empty_cells))
        for key, mine_n, v5_n in worse_empty_cells:
            fails.append("     %-26s %-11s empty raw %d > V5's %d"
                         % (key + (mine_n, v5_n)))
    if low_agree_cells:
        fails.append("%d cell(s) fall below the raw-agreement floor of %d/%d on "
                     "held_out_v3:" % (len(low_agree_cells), MIN_AGREE, MIN_SHARED))
        for key, same, shared_n in low_agree_cells:
            fails.append("     %-26s %-11s raw agreement %d/%d"
                         % (key + (same, shared_n)))
    C.gate(not fails, ([
        "REPORT ONLY: nothing written, no model loaded, no inference run",
        "%d cells, %d rows; %d not 'validated'; %d empty raw"
        % (len(by_cell), totals["rows"], totals["bad"], totals["empty"]),
        "%d of %d cells have every row failed" % (len(flagged), len(by_cell)),
        "prompt_error cells: %d (any is a FAIL)" % len(prompt_err_cells),
        "cells worse than V5 on empty output: %d (any is a FAIL)"
        % len(worse_empty_cells),
        ("cells below the %d/%d raw-agreement floor: %d (any is a FAIL)"
         % (MIN_AGREE, MIN_SHARED, len(low_agree_cells))) if a.split == "held_out_v3"
        else "raw-agreement floor: not applied (V5 never ran %s)" % a.split,
    ] + fails))


def _isfloat(x):
    try:
        float(x)
        return True
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    main()
