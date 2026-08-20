#!/usr/bin/env python
"""
bench/figures.py -- generate the V2 thesis figures from scored result CSVs.

Read-only: reads the scored CSVs, writes PNG + PDF into bench/figures/.
PDF is included because LaTeX embeds vector figures without resampling.

Figures produced:
  fig1_cell_accuracy   3x3 grid heatmaps, per model, seen vs held-out. Shows
                       WHERE on the table each model succeeds, not just how often.
  fig2_confusion       confusion matrix for one model, seen vs held-out. Shows
                       that failures collapse onto specific cells rather than
                       scattering -- i.e. they are systematic, not guessing.
  fig3_clean_vs_amb    clean vs ambiguous phrasing accuracy, all models. The
                       phrase/legend collision result.
  fig4_latency_tail    p50 / p95 / p99 / max per model, log scale. The tail
                       decomposition that the thesis is built on.
  fig5_thinking        reasoning-trace length vs latency for the reasoning model.
                       Skipped automatically if no thinking column is populated.

Usage:
    python bench/figures.py
    python bench/figures.py --model openai:gpt-4o-mini --outdir bench/figures
"""

import argparse
import collections
import csv
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(2**31 - 1)

MODELS = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
          "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b"]
ROWS, COLS = list("ABC"), list("123")
CELLS = [f"grid_{r}{c}" for r in ROWS for c in COLS]
LEGEND_TOKENS = {"middle", "center", "centre"}


def short(m):
    return m.split(":", 1)[1] if ":" in m else m


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    seen = {}
    for r in rows:
        seen[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
    return list(seen.values())


def truth_loc(r):
    try:
        return json.loads(r["truth_json"]).get("location")
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def pred_loc(r):
    """Hashable predicted location. Some models nest the whole action object, so
    .get('location') can return a dict -- stringify before using as a key."""
    try:
        obj = json.loads(r["pred_json"]) if r["pred_json"] else None
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    v = obj.get("location")
    if v is None and isinstance(obj.get("action"), dict):
        v = obj["action"].get("location")
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def phrase_for(rows, cell):
    for r in rows:
        if truth_loc(r) == cell:
            c = r["command"]
            for pre in ("put the bottle in ", "put the car in ", "put the rubiks cube in ",
                        "move to ", "go to ", "navigate to "):
                if c.startswith(pre):
                    return c[len(pre):].rstrip("?")
            return c
    return ""


def is_ambiguous(phrase):
    return bool(set(phrase.lower().replace(",", " ").split()) & LEGEND_TOKENS)


def cell_acc(rows, model):
    acc = {c: [0, 0] for c in CELLS}
    for r in rows:
        if r["model"] != model:
            continue
        t = truth_loc(r)
        if t not in acc:
            continue
        acc[t][1] += 1
        if r["strict_match"] == "True":
            acc[t][0] += 1
    return {c: (100 * h / n if n else float("nan")) for c, (h, n) in acc.items()}


def save(fig, outdir, name):
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(outdir / f"{name}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {outdir/name}.png / .pdf")


# ---------------------------------------------------------------- fig 1
def fig_cell_accuracy(seen, held, outdir):
    sets = [("Seen", seen)] + ([("Held-out", held)] if held else [])
    fig, axes = plt.subplots(len(sets), len(MODELS),
                             figsize=(3.1 * len(MODELS), 3.4 * len(sets)),
                             squeeze=False)
    for i, (label, rows) in enumerate(sets):
        for j, m in enumerate(MODELS):
            ax = axes[i][j]
            a = cell_acc(rows, m)
            grid = np.array([[a[f"grid_{r}{c}"] for c in COLS] for r in ROWS])
            im = ax.imshow(grid, cmap="RdYlGn", vmin=0, vmax=100)
            for y in range(3):
                for x in range(3):
                    v = grid[y][x]
                    if not math.isnan(v):
                        ax.text(x, y, f"{v:.0f}", ha="center", va="center",
                                fontsize=11, fontweight="bold",
                                color="black" if 25 < v < 75 else "white")
            ax.set_xticks(range(3), ["1\nleft", "2\nmid", "3\nright"], fontsize=8)
            ax.set_yticks(range(3), ["A far", "B mid", "C near"], fontsize=8)
            if i == 0:
                ax.set_title(short(m), fontsize=10, fontweight="bold")
            if j == 0:
                ax.set_ylabel(label, fontsize=11, fontweight="bold")
    fig.suptitle("Strict accuracy per table cell (%)  —  robot is below row C",
                 fontsize=13, y=1.01)
    fig.colorbar(im, ax=axes, shrink=0.6, label="strict accuracy (%)")
    save(fig, outdir, "fig1_cell_accuracy")


# ---------------------------------------------------------------- fig 2
def fig_confusion(seen, held, model, outdir):
    sets = [("Seen", seen)] + ([("Held-out", held)] if held else [])
    fig, axes = plt.subplots(1, len(sets), figsize=(6.2 * len(sets), 5.4), squeeze=False)
    labels = [c[5:] for c in CELLS] + ["other"]
    for k, (label, rows) in enumerate(sets):
        ax = axes[0][k]
        M = np.zeros((len(CELLS), len(CELLS) + 1))
        for i, t in enumerate(CELLS):
            cnt = collections.Counter()
            n = 0
            for r in rows:
                if r["model"] != model or truth_loc(r) != t:
                    continue
                n += 1
                cnt[pred_loc(r)] += 1
            if not n:
                continue
            for j, p in enumerate(CELLS):
                M[i][j] = 100 * cnt.get(p, 0) / n
            M[i][-1] = 100 - M[i][:-1].sum()
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i][j] >= 1:
                    ax.text(j, i, f"{M[i][j]:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if M[i][j] > 55 else "black")
        ax.set_xticks(range(len(labels)), labels, fontsize=8, rotation=45)
        ax.set_yticks(range(len(CELLS)), [c[5:] for c in CELLS], fontsize=8)
        ax.set_xlabel("predicted"); ax.set_ylabel("true")
        ax.set_title(f"{label}", fontsize=11, fontweight="bold")
    fig.suptitle(f"Confusion matrix — {short(model)} (row %)", fontsize=13)
    fig.colorbar(im, ax=axes, shrink=0.7, label="% of that cell's rows")
    save(fig, outdir, "fig2_confusion")


# ---------------------------------------------------------------- fig 3
def fig_clean_vs_amb(seen, held, outdir):
    sets = [("Seen", seen)] + ([("Held-out", held)] if held else [])
    fig, axes = plt.subplots(1, len(sets), figsize=(6.4 * len(sets), 4.4),
                             squeeze=False, sharey=True)
    for k, (label, rows) in enumerate(sets):
        ax = axes[0][k]
        amb = {c: is_ambiguous(phrase_for(rows, c)) for c in CELLS}
        clean, ambig = [], []
        for m in MODELS:
            a = cell_acc(rows, m)
            cs = [a[c] for c in CELLS if not amb[c] and not math.isnan(a[c])]
            as_ = [a[c] for c in CELLS if amb[c] and not math.isnan(a[c])]
            clean.append(statistics.mean(cs) if cs else 0)
            ambig.append(statistics.mean(as_) if as_ else 0)
        x = np.arange(len(MODELS))
        ax.bar(x - 0.2, clean, 0.4, label="clean phrasing", color="#3d7d3d")
        ax.bar(x + 0.2, ambig, 0.4, label="ambiguous phrasing", color="#b5482f")
        for xi, (a, b) in enumerate(zip(clean, ambig)):
            ax.text(xi - 0.2, a + 1.5, f"{a:.0f}", ha="center", fontsize=8)
            ax.text(xi + 0.2, b + 1.5, f"{b:.0f}", ha="center", fontsize=8)
        ax.set_xticks(x, [short(m) for m in MODELS], rotation=20, ha="right", fontsize=8)
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.set_ylim(0, 105)
        ax.grid(axis="y", alpha=0.25)
        if k == 0:
            ax.set_ylabel("strict accuracy (%)")
            ax.legend(fontsize=9)
    fig.suptitle('Phrase–legend collision: phrasings containing "middle"/"center"\n'
                 "compete with the column-2 label in the prompt legend", fontsize=12)
    save(fig, outdir, "fig3_clean_vs_ambiguous")


# ---------------------------------------------------------------- fig 4
def fig_latency(seen, held, outdir):
    def pctl(v, p):
        v = sorted(v)
        k = (len(v) - 1) * p / 100
        f, c = math.floor(k), math.ceil(k)
        return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    width = 0.2
    x = np.arange(len(MODELS))
    stats = {"p50": [], "p95": [], "p99": [], "max": []}
    for m in MODELS:
        lats = [float(r["latency_s"]) for r in seen
                if r["model"] == m and r["is_warmup"] != "True" and r["latency_s"]]
        if not lats:
            for k in stats:
                stats[k].append(0)
            continue
        stats["p50"].append(statistics.median(lats))
        stats["p95"].append(pctl(lats, 95))
        stats["p99"].append(pctl(lats, 99))
        stats["max"].append(max(lats))
    colors = ["#4c72b0", "#dd8452", "#c44e52", "#8172b3"]
    for i, (k, c) in enumerate(zip(["p50", "p95", "p99", "max"], colors)):
        ax.bar(x + (i - 1.5) * width, stats[k], width, label=k, color=c)
    ax.set_yscale("log")
    ax.set_xticks(x, [short(m) for m in MODELS], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("latency (s, log scale)")
    ax.set_title("End-to-end LLM latency, warm-up excluded — tail decomposition")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.legend(fontsize=9)
    save(fig, outdir, "fig4_latency_tail")


# ---------------------------------------------------------------- fig 5
def fig_thinking(seen, held, outdir):
    pts = []
    for label, rows in [("Seen", seen)] + ([("Held-out", held)] if held else []):
        for r in rows:
            tc = r.get("thinking_chars")
            if tc in (None, "", "0") or not r.get("latency_s"):
                continue
            try:
                pts.append((label, int(tc), float(r["latency_s"])))
            except ValueError:
                continue
    if not pts:
        print("  fig5 skipped (no thinking_chars populated)")
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, colour in (("Seen", "#4c72b0"), ("Held-out", "#c44e52")):
        xs = [p[1] for p in pts if p[0] == label]
        ys = [p[2] for p in pts if p[0] == label]
        if xs:
            ax.scatter(xs, ys, s=5, alpha=0.25, c=colour, label=f"{label} (n={len(xs)})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("reasoning trace length (characters, log)")
    ax.set_ylabel("latency (s, log)")
    ax.set_title("The cost of thinking: trace length vs latency (deepseek-r1:7b)")
    ax.grid(alpha=0.25, which="both")
    ax.legend(fontsize=9)
    save(fig, outdir, "fig5_thinking_vs_latency")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", default="bench/results_grid_v2_scored.csv")
    ap.add_argument("--held", default="bench/results_heldout_v2_scored.csv")
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--outdir", default="bench/figures")
    args = ap.parse_args()

    seen = load(args.seen)
    held = load(args.held) if Path(args.held).exists() else None
    outdir = Path(args.outdir)
    print(f"seen {len(seen)} rows"
          + (f" | held-out {len(held)} rows" if held else " | no held-out file"))

    fig_cell_accuracy(seen, held, outdir)
    fig_confusion(seen, held, args.model, outdir)
    fig_clean_vs_amb(seen, held, outdir)
    fig_latency(seen, held, outdir)
    fig_thinking(seen, held, outdir)
    print(f"\nall figures in {outdir}/  (PNG for slides, PDF for LaTeX)")


if __name__ == "__main__":
    main()
