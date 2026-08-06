import csv, sys
path = sys.argv[1] if len(sys.argv) > 1 else "bench/results_v2_smoke3.csv"
lo = int(sys.argv[2]) if len(sys.argv) > 2 else 68
for r in csv.DictReader(open(path, encoding="utf-8")):
    if int(r["command_id"]) < lo:
        continue
    print(f"[{r['strict_match']:5}] {r['command']}")
    print(f"        truth: {r['truth_json']}")
    print(f"        pred : {r['pred_json']}")