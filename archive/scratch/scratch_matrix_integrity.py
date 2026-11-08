import csv, json, collections
csv.field_size_limit(2**31-1)
T = lambda v: v in ("True", "true", "1")
FILES = ["results_heldout_v3_rung1.csv", "results_heldout_v3_rag.csv",
         "results_heldout_v3_lora.csv", "results_heldout_v3_cls.csv"]
d, tot = {}, 0
for fn in FILES:
    rs = list(csv.DictReader(open("bench/" + fn, encoding="utf-8", newline="")))
    tot += len(rs)
    print(f"{fn:34} {len(rs):>6} rows")
    for r in rs: d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
print(f"\ntotal {tot} (expect 30380) | after dedupe {len(d)} | dupes {tot-len(d)}")

cells = collections.defaultdict(list)
for r in d.values(): cells[(r["model"], r["strategy"])].append(r)
print(f"cells: {len(cells)} (expect 28)\n")
print(f"{'model':26}{'strategy':11}{'n':>6}{'ok':>7}{'strict':>8}{'norm':>7}{'p50':>8}{'p99':>8}{'box':>7}")
for (m, s), c in sorted(cells.items()):
    v = [r for r in c if not T(r["is_warmup"])]
    def lat(r):
        x = float(r["latency_s"])
        if s == "rag" and r.get("retrieval_latency_s"): x += float(r["retrieval_latency_s"])
        return x
    L = sorted(lat(r) for r in v)
    p = lambda q: L[min(int(q*len(L)), len(L)-1)]
    bx = [r for r in v if int(r["command_id"]) >= 181]
    print(f"{m:26}{s:11}{len(v):>6}{sum(T(r['ok']) for r in v)/len(v):>7.3f}"
          f"{sum(T(r['strict_match']) for r in v)/len(v):>8.3f}"
          f"{sum(T(r['normalized_match']) for r in v)/len(v):>7.3f}"
          f"{p(.50):>8.3f}{p(.99):>8.3f}"
          f"{sum(T(r['strict_match']) for r in bx)/len(bx):>7.3f}")
    if sum(T(r["ok"]) for r in v) == 0: print("   *** DEAD CELL ***")

print("\n--- RUNG 4 STRUCTURAL CHECK ---")
cl = [r for r in cells[("classifier:distilbert","classifier")] if not T(r["is_warmup"])]
VERBS = {"pick_place","move_to","fetch","hold","inspect","stop","reject"}
bad = collections.Counter()
for r in cl:
    try: a = json.loads(r["pred_json"]).get("action")
    except Exception: a = "<unparseable>"
    if a not in VERBS: bad[a] += 1
print("illegal/unparseable verbs:", dict(bad) or "NONE (structural guarantee holds)")
print("sample preds:", [r["pred_json"] for r in cl[:2]])
