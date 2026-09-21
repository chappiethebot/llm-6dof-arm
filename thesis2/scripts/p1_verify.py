"""thesis2/scripts/p1_verify.py -- sanity checks on the P1.2 result.

READ-ONLY. Two questions:
  1. What is held_out_v3's true location composition? The brief says 132
     grid-cell; P1.2 counted 107.
  2. Cells showing an EXACT 0.00 row-col gap -- is that because errors are
     non-decomposable (both factors wrong together), which would weaken the
     factorization argument?
"""
import collections
import csv
import json
import os

csv.field_size_limit(2 ** 31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(ROOT)

CONTAINERS = {"box_1", "box_2", "box_3"}
RUNTIME = {"max_reach", "nearest_free_cell"}
T = lambda v: str(v).strip().lower() == "true"


def ltype(loc):
    if loc is None:
        return "no_location"
    if not isinstance(loc, str):
        # some models emit a nested object for `location`; that is a structural
        # failure, not a grid prediction
        return "other"
    if loc in CONTAINERS:
        return "container"
    if loc in RUNTIME:
        return "runtime"
    # exactly grid_<row><col>; a truncated "grid_A" is malformed, not a cell
    if loc.startswith("grid_") and len(loc) == 7:
        return "grid_cell"
    return "other"


print("=" * 78)
print("Q1. held_out_v3.jsonl location composition")
rows = [json.loads(l) for l in open("data/held_out_v3.jsonl", encoding="utf-8") if l.strip()]
c = collections.Counter(ltype(r["expected"].get("location")) for r in rows)
for k in ("grid_cell", "runtime", "container", "no_location"):
    print("   %-12s %4d" % (k, c[k]))
print("   %-12s %4d" % ("TOTAL", sum(c.values())))
print()
print("   the brief's '132 grid-cell' == grid_cell + runtime = %d + %d = %d"
      % (c["grid_cell"], c["runtime"], c["grid_cell"] + c["runtime"]))
print("   TRUE grid cells (grid_A1..grid_C3) = %d" % c["grid_cell"])

print()
print("=" * 78)
print("Q2. exact-zero row-col gap cells: are errors decomposable?")


def obj(r, k):
    try:
        o = json.loads(r[k]) if r[k] else None
        return o if isinstance(o, dict) else None
    except Exception:
        return None


def loc(o):
    return o.get("location") if o else None


allrows = []
for f in ("rung1", "rag", "lora", "cls"):
    with open("bench/results_v5_%s_scored.csv" % f, encoding="utf-8", newline="") as fh:
        allrows += list(csv.DictReader(fh))
ded = {}
for r in allrows:
    ded[(r.get("dataset", ""), r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
rows5 = [r for r in ded.values() if not T(r["is_warmup"])]

by = collections.defaultdict(list)
for r in rows5:
    by[(r["model"], r["strategy"])].append(r)

WATCH = [("ollama:qwen2.5-3b-lora", "zero_shot"),
         ("openai:gpt-4o-mini", "zero_shot"),
         ("ollama:qwen2.5-0.5b-lora", "zero_shot"),
         ("ollama:qwen2.5-0.5b-lora", "role_based"),
         ("classifier:distilbert", "classifier"),
         ("ollama:qwen2.5:3b", "cot")]

print("%-28s %-11s %6s %7s %7s %7s %7s" %
      ("model", "strategy", "gerr", "rowONLY", "colONLY", "BOTH", "offgrid"))
for key in WATCH:
    cell = by.get(key, [])
    gerr = rowonly = colonly = both = offgrid = 0
    for r in cell:
        tl, pl = loc(obj(r, "truth_json")), loc(obj(r, "pred_json"))
        if ltype(tl) != "grid_cell":
            continue
        if tl == pl:
            continue
        gerr += 1
        if ltype(pl) != "grid_cell":
            offgrid += 1
            continue
        rw, cw = pl[5] != tl[5], pl[6] != tl[6]
        if rw and cw:
            both += 1
        elif rw:
            rowonly += 1
        else:
            colonly += 1
    print("%-28s %-11s %6d %7d %7d %7d %7d" %
          (key[0], key[1], gerr, rowonly, colonly, both, offgrid))

print()
print("   rowONLY+colONLY are DECOMPOSABLE errors -- exactly what a per-factor")
print("   cascade can target. BOTH and offgrid are not: fixing one factor alone")
print("   cannot repair them.")
print()
print("=" * 78)
print("Q3. decomposability pooled over ALL 36 cells")
tot = collections.Counter()
for key, cell in by.items():
    for r in cell:
        tl, pl = loc(obj(r, "truth_json")), loc(obj(r, "pred_json"))
        if ltype(tl) != "grid_cell" or tl == pl:
            continue
        tot["grid_errors"] += 1
        if ltype(pl) != "grid_cell":
            tot["offgrid"] += 1
            continue
        rw, cw = pl[5] != tl[5], pl[6] != tl[6]
        if rw and cw:
            tot["both_axes"] += 1
        elif rw:
            tot["row_only"] += 1
        else:
            tot["col_only"] += 1
n = tot["grid_errors"]
for k in ("row_only", "col_only", "both_axes", "offgrid"):
    print("   %-12s %6d  %5.1f%% of grid-cell errors" % (k, tot[k], 100.0 * tot[k] / n))
print("   %-12s %6d" % ("TOTAL", n))
print()
print("   DECOMPOSABLE (row_only + col_only) = %.1f%%"
      % (100.0 * (tot["row_only"] + tot["col_only"]) / n))
print("=" * 78)
