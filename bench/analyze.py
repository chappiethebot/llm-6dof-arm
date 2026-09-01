#!/usr/bin/env python
"""
bench/analyze.py -- Phase 5 analysis. Reads the two SCORED grid CSVs
(seen + held-out), dedupes defensively, and emits the results-chapter figures
and summary tables. Read-only over the data; produces PNGs + printed tables.

Figures written to bench/figures/:
  1. accuracy_heatmap.png      -- strict accuracy, model x strategy, seen vs held-out
  2. latency_tails.png         -- per-model latency median/p95/p99 (warmup excluded, log y)
  3. generalization_gap.png    -- seen vs held-out strict accuracy, per model
  4. cot_strict_vs_norm.png    -- the "cost of thinking": strict vs normalized under cot

Usage:
    python -m bench.analyze
    python -m bench.analyze --seen bench/results_grid_scored.csv --held bench/results_heldout_scored.csv

DATA VERSION: V1/V2 ONLY -- it defaults to the V2 scored CSVs. This is the
provenance of the committed V2 tables and is not broken; it simply has no
notion of the per-model effective-n correction the V3 work introduced. For V3
see audit/figures_v3.py and audit/v4_results.py; for V4 see audit/figures_v4.py.
"""
import argparse, csv, math, statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display needed; write files
import matplotlib.pyplot as plt

MODELS = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
          "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b"]
STRATS = ["zero_shot", "few_shot", "cot", "role_based"]


def short(m):
    return m.split(":", 1)[1] if ":" in m else m


def load(path):
    """Read a scored CSV, dedupe on identity key keeping LAST (resume orphans).

    KEY INCLUDES `dataset`. command_id is an index into THAT dataset's test
    rows, so seen row 7 and held-out row 7 are different commands. Without
    `dataset` in the key a union of two files silently drops rows -- no error,
    no warning, just a smaller denominator. This function currently reads one
    file at a time, so the four-field key happened to be safe; it stops being
    safe the moment anyone passes a combined file or unions two loads.
    Matches bench/summary.py, bench/figures.py and bench/rescore.py."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    seen = {}
    for r in rows:
        seen[(r.get("dataset", ""), r["model"], r["strategy"],
              r["command_id"], r["run_index"])] = r
    return list(seen.values())


def pct(vals, p):
    v = sorted(vals)
    k = (len(v) - 1) * p / 100
    f, c = math.floor(k), math.ceil(k)
    return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)


def acc_grid(rows, col):
    """Return {model: {strat: fraction True in `col`}}."""
    tot = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    for r in rows:
        c = tot[r["model"]][r["strategy"]]
        c[0] += (r[col] == "True")
        c[1] += 1
    return {m: {s: (tot[m][s][0] / tot[m][s][1] if tot[m][s][1] else 0.0)
                for s in STRATS} for m in MODELS}


def fig_heatmap(seen, held, out):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.subplots_adjust(wspace=0.08)
    for idx, (ax, rows, title) in enumerate(
            [(axes[0], seen, "SEEN"), (axes[1], held, "HELD-OUT")]):
        g = acc_grid(rows, "strict_match")
        data = [[g[m][s] * 100 for s in STRATS] for m in MODELS]
        im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(range(len(STRATS))); ax.set_xticklabels(STRATS, rotation=30, ha="right")
        if idx == 0:
            ax.set_yticks(range(len(MODELS))); ax.set_yticklabels([short(m) for m in MODELS])
        else:
            ax.set_yticks([])  # redundant -- same rows as left panel
        ax.set_title(f"Strict accuracy (%) -- {title}")
        for i in range(len(MODELS)):
            for j in range(len(STRATS)):
                ax.text(j, i, f"{data[i][j]:.0f}", ha="center", va="center", fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.025)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_latency(seen, out):
    """Per-model latency median/p95/p99, warmup excluded. Log y (deepseek dwarfs rest)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(len(MODELS))
    meds, p95s, p99s = [], [], []
    for m in MODELS:
        lats = [float(r["latency_s"]) for r in seen
                if r["model"] == m and r["is_warmup"] != "True" and r["latency_s"]]
        meds.append(statistics.median(lats)); p95s.append(pct(lats, 95)); p99s.append(pct(lats, 99))
    w = 0.25
    ax.bar([x - w for x in xs], meds, w, label="median")
    ax.bar([x for x in xs], p95s, w, label="p95")
    ax.bar([x + w for x in xs], p99s, w, label="p99")
    ax.set_yscale("log")
    ax.set_xticks(list(xs)); ax.set_xticklabels([short(m) for m in MODELS], rotation=20, ha="right")
    ax.set_ylabel("LLM latency (s, log scale)")
    ax.set_title("Latency by model (warmup excluded) -- the p99 tail is the cost of thinking")
    ax.legend()
    for x, v in zip(xs, p99s):
        ax.text(x + w, v, f"{v:.1f}s", ha="center", va="bottom", fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_generalization(seen, held, out):
    """Seen vs held-out strict accuracy, averaged over strategies, per model."""
    gs, gh = acc_grid(seen, "strict_match"), acc_grid(held, "strict_match")
    seen_avg = [sum(gs[m][s] for s in STRATS) / len(STRATS) * 100 for m in MODELS]
    held_avg = [sum(gh[m][s] for s in STRATS) / len(STRATS) * 100 for m in MODELS]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(len(MODELS)); w = 0.38
    ax.bar([x - w / 2 for x in xs], seen_avg, w, label="seen")
    ax.bar([x + w / 2 for x in xs], held_avg, w, label="held-out")
    ax.set_xticks(list(xs)); ax.set_xticklabels([short(m) for m in MODELS], rotation=20, ha="right")
    ax.set_ylabel("mean strict accuracy (%)")
    ax.set_title("Generalization: seen vs held-out (mean over strategies)")
    ax.legend()
    for x, sv, hv in zip(xs, seen_avg, held_avg):
        ax.text(x, max(sv, hv) + 1, f"-{sv - hv:.0f}", ha="center", fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def fig_cot(seen, out):
    """cot strict vs normalized per model -- shows failures are formatting, not meaning."""
    gs, gn = acc_grid(seen, "strict_match"), acc_grid(seen, "normalized_match")
    strict = [gs[m]["cot"] * 100 for m in MODELS]
    norm = [gn[m]["cot"] * 100 for m in MODELS]
    fig, ax = plt.subplots(figsize=(9, 5))
    xs = range(len(MODELS)); w = 0.38
    ax.bar([x - w / 2 for x in xs], strict, w, label="strict")
    ax.bar([x + w / 2 for x in xs], norm, w, label="normalized")
    ax.set_xticks(list(xs)); ax.set_xticklabels([short(m) for m in MODELS], rotation=20, ha="right")
    ax.set_ylabel("cot accuracy (%)")
    ax.set_title("CoT: strict vs normalized (gap = correct answers, broken formatting)")
    ax.legend()
    for x, sv, nv in zip(xs, strict, norm):
        if nv - sv > 3:
            ax.text(x + w / 2, nv + 1, f"+{nv - sv:.0f}", ha="center", fontsize=8)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def print_tables(seen, held):
    gs = acc_grid(seen, "strict_match"); gh = acc_grid(held, "strict_match")
    print("\nSTRICT ACCURACY  seen -> held-out  (drop)")
    print(f'{"model":15}{"strategy":11}{"seen":>7}{"held":>7}{"drop":>8}')
    for m in MODELS:
        for s in STRATS:
            sv, hv = gs[m][s] * 100, gh[m][s] * 100
            print(f'{short(m):15}{s:11}{sv:6.1f}%{hv:6.1f}%{sv - hv:+7.1f}%')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", default="bench/results_grid_scored.csv")
    ap.add_argument("--held", default="bench/results_heldout_scored.csv")
    ap.add_argument("--outdir", default="bench/figures")
    args = ap.parse_args()

    seen = load(args.seen); held = load(args.held)
    print(f"loaded seen={len(seen)} held={len(held)} rows")
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)

    fig_heatmap(seen, held, outdir / "accuracy_heatmap.png")
    fig_latency(seen, outdir / "latency_tails.png")
    fig_generalization(seen, held, outdir / "generalization_gap.png")
    fig_cot(seen, outdir / "cot_strict_vs_norm.png")
    print(f"wrote 4 figures -> {outdir}/")
    print_tables(seen, held)


if __name__ == "__main__":
    main()
