"""Diagnostic A: WHAT varied between repeats in the seen run?

DATA VERSION: V1 ONLY. Reads bench/results_grid.csv. Nothing here reads V2,
V3 or V4 files.
"""
import csv, collections, sys

PATH = "bench/results_grid.csv"
groups = collections.defaultdict(dict)

with open(PATH, encoding="utf-8", newline="") as f:
    for r in csv.DictReader(f):
        key = (r["model"], r["strategy"], r["command_id"])
        groups[key][r["run_index"]] = r["raw"]

for model_filter in ["ollama:deepseek-r1:7b", "ollama:qwen2.5:0.5b"]:
    varying = [(k, v) for k, v in groups.items()
               if k[0] == model_filter and len(set(v.values())) > 1]
    print("=" * 70)
    print(f"{model_filter}: {len(varying)} varying groups")
    print("=" * 70)
    for key, runs in varying[:3]:
        print(f"\n--- {key[1]} cid={key[2]} : {len(set(runs.values()))} distinct ---")
        for ri in sorted(runs):
            print(f"  run{ri}: {runs[ri][:200]!r}")