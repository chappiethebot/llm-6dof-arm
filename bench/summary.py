#!/usr/bin/env python
"""Quick KPI summary over a scored grid CSV. Reads the file, prints the three
headline views: strict-vs-normalized accuracy per cell, and latency percentiles
per model (warmup rows excluded). Read-only; computes nothing new, just reports."""
import argparse, csv, statistics, math
from collections import defaultdict


def pct(vals, p):
    v = sorted(vals)
    k = (len(v) - 1) * p / 100
    f = math.floor(k)
    c = math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def short(m):
    return m.split(":", 1)[1] if ":" in m else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bench/results_grid_scored.csv")
    args = ap.parse_args()
    rows = list(csv.DictReader(open(args.inp, encoding="utf-8")))

    models = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
              "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b"]
    strats = ["zero_shot", "few_shot", "cot", "role_based"]

    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        acc[r["model"]][r["strategy"]].append(
            (r["strict_match"] == "True", r["normalized_match"] == "True"))

    print("ACCURACY  strict / normalized / gap")
    print(f'{"model":16}{"strategy":11}{"strict":>8}{"norm":>8}{"gap":>7}')
    for m in models:
        for s in strats:
            cell = acc[m][s]
            st = sum(x for x, _ in cell) / len(cell)
            nm = sum(y for _, y in cell) / len(cell)
            flag = "  <==" if (nm - st) > 0.10 else ""
            print(f'{short(m):16}{s:11}{st*100:7.1f}%{nm*100:7.1f}%{(nm-st)*100:+6.1f}%{flag}')

    print()
    print("LATENCY (s)  warmup excluded, pooled across strategies")
    print(f'{"model":16}{"median":>8}{"p95":>8}{"p99":>8}{"max":>8}')
    for m in models:
        lats = [float(r["latency_s"]) for r in rows
                if r["model"] == m and r["is_warmup"] != "True" and r["latency_s"]]
        print(f'{short(m):16}{statistics.median(lats):8.2f}{pct(lats,95):8.2f}'
              f'{pct(lats,99):8.2f}{max(lats):8.2f}')


if __name__ == "__main__":
    main()
