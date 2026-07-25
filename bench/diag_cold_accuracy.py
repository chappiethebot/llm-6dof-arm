"""Is the cold first call less accurate, or just different?"""
import csv, collections

def tally(path, label):
    acc = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            bucket = "cold(run0)" if r["run_index"] == "0" else "warm(1-4)"
            a = acc[r["model"]][bucket]
            a[1] += 1
            if r["strict_match"] == "True":
                a[0] += 1
    print("=" * 62)
    print(label)
    print(f"  {'model':24s} {'cold':>16s} {'warm':>16s}")
    for m in sorted(acc):
        c, w = acc[m]["cold(run0)"], acc[m]["warm(1-4)"]
        cs = f"{c[0]/c[1]:.1%} (n={c[1]})" if c[1] else "-"
        ws = f"{w[0]/w[1]:.1%} (n={w[1]})" if w[1] else "-"
        print(f"  {m:24s} {cs:>16s} {ws:>16s}")

tally("bench/results_grid.csv", "SEEN")
tally("bench/results_heldout.csv", "HELD-OUT")