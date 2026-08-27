"""Why do both LoRA adapters put ~90% of grid predictions on row B?

Two competing accounts:
  DATASET   train_v2 over-represents row B and the adapters learned the
            frequency.
  TRAINING  train_v2 is balanced and the collapse is a fine-tuning artifact.
And a third, testable against the container split:
  COPY-ELSE-DEFAULT  the adapters learned "emit the symbol when it appears in
            the command, otherwise emit the most common grid answer".
"""
import csv, json, collections, os

csv.field_size_limit(2**31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CELLS = [f"grid_{r}{c}" for r in "ABC" for c in "123"]
BOXES = ["box_1", "box_2", "box_3"]
RUNTIME = ["max_reach", "nearest_free_cell"]
CONTAINER_START = 181


def jl(name):
    p = os.path.join(ROOT, "data", name)
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


def loc_of(row):
    return (row.get("expected") or {}).get("location")


print("=" * 92)
print("1. TRAINING-SET ROW MARGINAL  (is row B over-represented?)")
print("=" * 92)
for name in ("train_v2.jsonl", "dev_v2.jsonl", "held_out_v3.jsonl"):
    rows = jl(name)
    grid = [r for r in rows if loc_of(r) in CELLS]
    c = collections.Counter(loc_of(r)[5] for r in grid)
    n = len(grid)
    tot = len(rows)
    box = sum(1 for r in rows if loc_of(r) in BOXES)
    rt = sum(1 for r in rows if loc_of(r) in RUNTIME)
    print(f"\n  {name:20} total={tot:4}  grid={n:4}  container={box:3}  runtime={rt:3}")
    if n:
        print(f"    row A {c['A']:4} = {100*c['A']/n:5.1f}%"
              f"   row B {c['B']:4} = {100*c['B']/n:5.1f}%"
              f"   row C {c['C']:4} = {100*c['C']/n:5.1f}%")
        cc = collections.Counter(loc_of(r) for r in grid)
        print("    per cell: " + "  ".join(f"{k[5:]}={cc[k]}" for k in CELLS))

print()
print("=" * 92)
print("2. PREDICTION MARGINAL, and the COPY-ELSE-DEFAULT test")
print("=" * 92)

d = {}
for tag in ("rung1", "rag", "lora", "cls"):
    p = os.path.join(ROOT, "bench", f"results_heldout_v3_{tag}_scored.csv")
    with open(p, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
rows = [r for r in d.values() if r["is_warmup"] != "True"]


def pred_loc(r):
    try:
        o = json.loads(r["pred_json"]) if r["pred_json"] else None
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(o, dict):
        return None
    v = o.get("location")
    if v is None and isinstance(o.get("action"), dict):
        v = o["action"].get("location")
    return v if isinstance(v, str) else None


def truth_loc(r):
    try:
        return json.loads(r["truth_json"] or "{}").get("location")
    except (json.JSONDecodeError, TypeError):
        return None


TARGETS = ["ollama:qwen2.5-0.5b-lora", "ollama:qwen2.5-3b-lora",
           "classifier:distilbert"]

for m in TARGETS:
    mr = [r for r in rows if r["model"] == m]
    grid = [r for r in mr if truth_loc(r) in CELLS]
    cont = [r for r in mr if truth_loc(r) in BOXES]
    rt = [r for r in mr if truth_loc(r) in RUNTIME]
    print(f"\n--- {m} ---")

    pc = collections.Counter()
    for r in grid:
        p = pred_loc(r)
        pc[p[5] if (p in CELLS) else ("box" if p in BOXES else
                                      ("runtime" if p in RUNTIME else "none/other"))] += 1
    n = len(grid)
    print(f"  on GRID commands (n={n}, truth uniform 33.3% per row):")
    for k in ("A", "B", "C", "box", "runtime", "none/other"):
        if pc[k]:
            print(f"     predicts {k:11} {pc[k]:4} = {100*pc[k]/n:5.1f}%")
    acc_g = sum(r["strict_match"] == "True" for r in grid) / n if n else 0

    nc = len(cont)
    acc_c = sum(r["strict_match"] == "True" for r in cont) / nc if nc else 0
    cpred = collections.Counter(pred_loc(r) for r in cont)
    print(f"  on CONTAINER commands (n={nc}): action match = {acc_c:.3f}")
    print(f"     predicted symbols: {dict(cpred.most_common(5))}")

    # exact-copy test: does the predicted symbol appear VERBATIM in the command?
    def verbatim(r):
        p = pred_loc(r)
        if not p:
            return False
        return p.replace("_", " ") in r["command"].lower()
    cv = sum(verbatim(r) for r in cont)
    gv = sum(verbatim(r) for r in grid)
    print(f"     prediction appears verbatim in command: "
          f"container {cv}/{nc} = {100*cv/nc if nc else 0:.1f}%   "
          f"grid {gv}/{n} = {100*gv/n if n else 0:.1f}%")
    print(f"  action match: grid {acc_g:.3f}  container {acc_c:.3f}")

    if rt:
        rc = collections.Counter(pred_loc(r) for r in rt)
        acc_r = sum(r["strict_match"] == "True" for r in rt) / len(rt)
        print(f"  on RUNTIME-SYMBOL commands (n={len(rt)}): action match = {acc_r:.3f}"
              f"  top preds {dict(rc.most_common(3))}")

print()
print("=" * 92)
print("3. WHICH SINGLE CELL do the LoRA models default to?")
print("=" * 92)
for m in TARGETS[:2]:
    grid = [r for r in rows if r["model"] == m and truth_loc(r) in CELLS]
    c = collections.Counter(pred_loc(r) for r in grid)
    n = len(grid)
    print(f"  {m}:")
    for k, v in c.most_common(4):
        print(f"     {str(k):22} {v:4} = {100*v/n:5.1f}%")
