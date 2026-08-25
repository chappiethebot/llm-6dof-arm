import csv, collections
csv.field_size_limit(2**31-1)
T = lambda v: v in ("True","true","1")
FILES = ["rung1","rag","lora","cls"]
d = {}
for f in FILES:
    for r in csv.DictReader(open(f"bench/results_heldout_v3_{f}_scored.csv", encoding="utf-8", newline="")):
        d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
cells = collections.defaultdict(list)
for r in d.values(): cells[(r["model"], r["strategy"])].append(r)
print(f"cells: {len(cells)}\n")
print(f"{'model':26}{'strategy':11}{'strict':>8}{'norm':>7}{'gap':>7}{'grid':>7}{'box':>7}{'p50':>8}{'p99':>8}")
for (m,s),c in sorted(cells.items()):
    v = [r for r in c if not T(r["is_warmup"])]
    def lat(r):
        x = float(r["latency_s"])
        if s=="rag" and r.get("retrieval_latency_s"): x += float(r["retrieval_latency_s"])
        return x
    L = sorted(lat(r) for r in v); p = lambda q: L[min(int(q*len(L)),len(L)-1)]
    g = [r for r in v if int(r["command_id"]) < 181]
    b = [r for r in v if int(r["command_id"]) >= 181]
    st = sum(T(r["strict_match"]) for r in v)/len(v)
    no = sum(T(r["normalized_match"]) for r in v)/len(v)
    print(f"{m:26}{s:11}{st:>8.3f}{no:>7.3f}{no-st:>7.3f}"
          f"{sum(T(r['strict_match']) for r in g)/len(g):>7.3f}"
          f"{sum(T(r['strict_match']) for r in b)/len(b):>7.3f}{p(.50):>8.3f}{p(.99):>8.3f}")
