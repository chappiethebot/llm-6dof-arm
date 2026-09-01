"""Diagnostic A: is the variation entirely a run0 (cold-start) effect?

DATA VERSION: V1 ONLY. Reads bench/results_grid.csv. Nothing here reads V2,
V3 or V4 files.
"""
import csv, collections

groups = collections.defaultdict(dict)
with open("bench/results_grid.csv", encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        groups[(r["model"], r["strategy"], r["command_id"])][r["run_index"]] = r["raw"]

tally = collections.defaultdict(collections.Counter)
for (model, _s, _c), runs in groups.items():
    if len(set(runs.values())) == 1:
        continue
    warm = {k: v for k, v in runs.items() if k != "0"}
    if len(set(warm.values())) == 1 and runs.get("0") not in set(warm.values()):
        tally[model]["run0 only"] += 1
    elif len(set(warm.values())) == 1:
        tally[model]["other"] += 1
    else:
        tally[model]["warm runs disagree too"] += 1

print(f"{'model':26s} {'run0 only':>10s} {'warm disagree':>14s} {'other':>7s}")
for m in sorted(tally):
    t = tally[m]
    print(f"{m:26s} {t['run0 only']:10d} {t['warm runs disagree too']:14d} {t['other']:7d}")