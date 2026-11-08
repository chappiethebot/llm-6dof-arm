import csv, collections
csv.field_size_limit(2**31-1)
rows = list(csv.DictReader(open("bench/results_heldout_v3_rung1.csv", encoding="utf-8", newline="")))
print("total rows:", len(rows), "(expect 21700)")
d = {}
for r in rows: d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
print("after dedupe:", len(d), "| duplicates:", len(rows) - len(d))
T = lambda v: v in ("True", "true", "1")
cells = collections.defaultdict(list)
for r in d.values(): cells[(r["model"], r["strategy"])].append(r)
print("cells:", len(cells), "(expect 20)")
print(f"{'model':24}{'strategy':12}{'n':>6}{'ok':>7}{'strict':>8}{'p50':>7}{'p99':>8}")
for (m, s), c in cells.items():
    v = [r for r in c if not T(r["is_warmup"])]
    lat = sorted(float(r["latency_s"]) for r in v)
    p = lambda q: lat[min(int(q * len(lat)), len(lat) - 1)]
    print(f"{m:24}{s:12}{len(v):>6}{sum(T(r['ok']) for r in v)/len(v):>7.3f}"
          f"{sum(T(r['strict_match']) for r in v)/len(v):>8.3f}{p(.50):>7.2f}{p(.99):>8.2f}")
    if sum(T(r["ok"]) for r in v) == 0: print("   *** DEAD CELL ***")
