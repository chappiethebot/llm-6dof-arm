import csv, collections
csv.field_size_limit(2**31 - 1)
rows = list(csv.DictReader(open("bench/results_cls_grid_smoke.csv", encoding="utf-8")))
print("rows:", len(rows), "| columns:", len(rows[0]))
print("stages:", dict(collections.Counter(r["stage"] for r in rows)))
print("temperature col:", repr(rows[0]["temperature"]), "| thinking_chars:",
      set(r["thinking_chars"] for r in rows))
warm = [r for r in rows if r["is_warmup"] == "True"]
cold = [float(r["latency_s"]) for r in rows if r["is_warmup"] != "True"]
cold.sort()
strict = sum(r["strict_match"] == "True" for r in rows)
norm   = sum(r["normalized_match"] == "True" for r in rows)
print(f"warmup rows: {len(warm)} | strict {strict}/{len(rows)} | normalized {norm}/{len(rows)}")
print(f"latency (warmup excluded, n={len(cold)}): p50 {cold[len(cold)//2]*1000:.1f} ms | "
      f"p95 {cold[int(0.95*len(cold))-1]*1000:.1f} ms | max {cold[-1]*1000:.1f} ms")
