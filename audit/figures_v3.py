"""Task 2 -- publication figures over all 28 held_out_v3 cells. READ-ONLY.

Written fresh rather than extending bench/figures.py. That script is NOT broken
-- it renders correctly -- but it defaults to the V2 files, produces a different
five-figure set, and has no notion of the per-model effective-n correction these
figures require. It is also tracked in git with committed outputs, so it is left
untouched.

THE EFFECTIVE-n CORRECTION, which is the statistical point of these figures:
each cell holds 217 commands x 5 repeats = 1,085 rows, but for the seven LOCAL
models every repeat is byte-identical (measured: 100.0% across all 217 commands
in every local cell). Those repeats carry zero additional information, so the
effective n is 217 and a naive sqrt(1085) interval understates the error by
sqrt(5) = 2.24x. gpt-4o-mini is the exception: only 77.1% of its commands are
byte-identical across repeats, so its repeats are genuine draws.

Intervals are therefore computed by CLUSTERED BOOTSTRAP over command_id -- resample
commands, not rows -- which handles both cases correctly without a special case:
for a deterministic model the within-cluster variance is zero by construction.
"""
import csv, json, math, os, random, collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

csv.field_size_limit(2**31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "audit", "figures")
T = lambda v: v in ("True", "true", "1")
RNG = random.Random(20260827)
N_BOOT = 2000
CONTAINER_START = 181

RUNG = {}          # model -> rung label
def rung_of(model, strategy):
    if model.startswith("classifier:"):
        return "rung 4 classifier"
    if model.endswith("-lora"):
        return "rung 3 LoRA"
    return "rung 2 RAG" if strategy == "rag" else "rung 1 prompting"

RUNG_STYLE = {
    "rung 1 prompting": ("o", "#4C72B0"),
    "rung 2 RAG":       ("s", "#DD8452"),
    "rung 3 LoRA":      ("^", "#55A868"),
    "rung 4 classifier":("D", "#C44E52"),
}


def load_cells():
    d = {}
    for tag in ("rung1", "rag", "lora", "cls"):
        p = os.path.join(ROOT, "bench", f"results_heldout_v3_{tag}_scored.csv")
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                d[(r.get("dataset", ""), r["model"], r["strategy"],
                   r["command_id"], r["run_index"])] = r
    cells = collections.defaultdict(list)
    for r in d.values():
        if not T(r["is_warmup"]):
            cells[(r["model"], r["strategy"])].append(r)
    return cells


def clustered_ci(rows, pred=lambda r: T(r["strict_match"]), n_boot=N_BOOT, alpha=0.05):
    """Bootstrap over command_id clusters. Returns (point, lo, hi, n_clusters)."""
    by_cmd = collections.defaultdict(list)
    for r in rows:
        by_cmd[r["command_id"]].append(r)
    keys = list(by_cmd)
    if not keys:
        return 0.0, 0.0, 0.0, 0
    # cluster means, so a deterministic cluster contributes zero within-variance
    means = {k: sum(pred(x) for x in v) / len(v) for k, v in by_cmd.items()}
    point = sum(means.values()) / len(keys)
    boots = []
    n = len(keys)
    for _ in range(n_boot):
        s = 0.0
        for _ in range(n):
            s += means[keys[RNG.randrange(n)]]
        boots.append(s / n)
    boots.sort()
    lo = boots[int((alpha / 2) * n_boot)]
    hi = boots[int((1 - alpha / 2) * n_boot) - 1]
    return point, lo, hi, n


def short(m):
    return (m.replace("ollama:", "").replace("openai:", "")
             .replace("classifier:", "cls:"))


def cell_label(m, s):
    return f"{short(m)} / {s}"


def pctl(vals, q):
    v = sorted(vals)
    return v[min(int(q * len(v)), len(v) - 1)] if v else 0.0


def lat_of(r, strategy):
    x = float(r["latency_s"])
    if strategy == "rag" and r.get("retrieval_latency_s"):
        x += float(r["retrieval_latency_s"])
    return x


def save(fig, name):
    os.makedirs(OUTDIR, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=200,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote audit/figures/{name}.png / .pdf")


# ---------------------------------------------------------------- figure 1
def fig_accuracy(cells, stats):
    order = sorted(stats, key=lambda k: stats[k]["point"])
    fig, ax = plt.subplots(figsize=(9, 10))
    ys = range(len(order))
    for i, k in enumerate(order):
        st = stats[k]
        mk, col = RUNG_STYLE[rung_of(*k)]
        ax.errorbar(st["point"], i,
                    xerr=[[st["point"] - st["lo"]], [st["hi"] - st["point"]]],
                    fmt=mk, color=col, ms=6, capsize=3, lw=1.4)
    ax.set_yticks(list(ys))
    ax.set_yticklabels([cell_label(*k) for k in order], fontsize=8)
    ax.set_xlabel("strict accuracy (95% CI, bootstrap clustered on command_id)")
    ax.set_title("Strict accuracy per cell, all 28 cells\n"
                 "effective n = 217 commands (repeats are byte-identical for every local model)",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(0, 1)
    ax.legend(handles=[Line2D([], [], marker=m, color=c, ls="", label=r)
                       for r, (m, c) in RUNG_STYLE.items()],
              loc="lower right", fontsize=8)
    save(fig, "fig1_strict_accuracy_clustered_ci")


# ---------------------------------------------------------------- figure 2
def fig_grid_vs_container(cells, stats):
    order = sorted(stats, key=lambda k: stats[k]["grid"], reverse=True)
    fig, ax = plt.subplots(figsize=(11, 8))
    x = range(len(order))
    gw = 0.38
    ax.bar([i - gw/2 for i in x], [stats[k]["grid"] for k in order], gw,
           label="grid rows (n=181, 11-way, symbol never in command)",
           color="#4C72B0")
    # Accuracy is a proportion: clip the +-16 pt interval at [0,1] rather than
    # drawing bars that run off the top of the axis.
    bv = [stats[k]["box"] for k in order]
    lo_err = [min(0.16, v) for v in bv]
    hi_err = [min(0.16, 1.0 - v) for v in bv]
    ax.bar([i + gw/2 for i in x], bv, gw,
           yerr=[lo_err, hi_err], capsize=2, error_kw=dict(lw=0.8, alpha=0.6),
           label="container rows (n=36, 3-way, symbol verbatim; CI $\\pm$16 pts, clipped)",
           color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels([cell_label(*k) for k in order], rotation=90, fontsize=7)
    ax.set_ylabel("strict accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Grid versus container accuracy per cell\n"
                 "These are different tasks, not a hard and an easy subset of one task",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(fontsize=8, loc="upper right")
    save(fig, "fig2_grid_vs_container")


# ---------------------------------------------------------------- figure 3
def fig_pareto(cells, stats):
    fig, ax = plt.subplots(figsize=(9.5, 7))
    pts = []
    for k, st in stats.items():
        mk, col = RUNG_STYLE[rung_of(*k)]
        ax.scatter(st["p50"], st["point"], marker=mk, s=64, color=col,
                   edgecolor="black", lw=0.5, zorder=3)
        pts.append((st["p50"], st["point"], k))
    # Pareto frontier: no other cell is both faster AND more accurate
    front = [p for p in pts
             if not any(q[0] < p[0] and q[1] > p[1] for q in pts)]
    front.sort()
    ax.step([p[0] for p in front], [p[1] for p in front], where="post",
            color="black", lw=1.2, alpha=0.7, zorder=2, label="Pareto frontier")
    for lat, acc, k in front:
        ax.annotate(cell_label(*k), (lat, acc), fontsize=7,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xscale("log")
    ax.set_xlabel("p50 latency per command, seconds (log scale; rag includes retrieval)")
    ax.set_ylabel("strict accuracy")
    ax.set_title("Cost-accuracy Pareto frontier, all 28 cells\n"
                 "The result is a frontier of defensible operating points, not a single winner",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    h = [Line2D([], [], marker=m, color=c, ls="", label=r)
         for r, (m, c) in RUNG_STYLE.items()]
    h.append(Line2D([], [], color="black", lw=1.2, label="Pareto frontier"))
    ax.legend(handles=h, fontsize=8, loc="lower left")
    save(fig, "fig3_cost_accuracy_pareto")
    return front


# ---------------------------------------------------------------- figure 4
def fig_gap(cells, stats):
    order = sorted(stats, key=lambda k: stats[k]["gap"], reverse=True)
    fig, ax = plt.subplots(figsize=(11, 7))
    x = range(len(order))
    ax.bar(x, [stats[k]["gap"] for k in order],
           color=[RUNG_STYLE[rung_of(*k)][1] for k in order])
    ax.set_xticks(list(x))
    ax.set_xticklabels([cell_label(*k) for k in order], rotation=90, fontsize=7)
    ax.set_ylabel("normalized accuracy $-$ strict accuracy")
    ax.set_title("Cost of structural failure: normalized minus strict, per cell\n"
                 "A positive gap is output that was semantically right and structurally rejected",
                 fontsize=11)
    ax.grid(axis="y", alpha=0.3)
    ax.legend(handles=[Line2D([], [], marker="s", color=c, ls="", label=r)
                       for r, (m, c) in RUNG_STYLE.items()], fontsize=8)
    save(fig, "fig4_strict_vs_normalized_gap")


def main():
    cells = load_cells()
    print(f"cells: {len(cells)}")
    stats = {}
    for k, rows in cells.items():
        m, s = k
        point, lo, hi, ncl = clustered_ci(rows)
        grid = [r for r in rows if int(r["command_id"]) < CONTAINER_START]
        box = [r for r in rows if int(r["command_id"]) >= CONTAINER_START]
        lats = [lat_of(r, s) for r in rows]
        norm = sum(T(r["normalized_match"]) for r in rows) / len(rows)
        stats[k] = dict(
            point=point, lo=lo, hi=hi, n_clusters=ncl,
            grid=sum(T(r["strict_match"]) for r in grid) / len(grid),
            box=sum(T(r["strict_match"]) for r in box) / len(box),
            p50=pctl(lats, .50), p99=pctl(lats, .99),
            gap=norm - point, norm=norm, n=len(rows))
    fig_accuracy(cells, stats)
    fig_grid_vs_container(cells, stats)
    front = fig_pareto(cells, stats)
    fig_gap(cells, stats)

    # table dump so the numbers behind the figures are inspectable
    p = os.path.join(ROOT, "audit", "figures", "cell_stats.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "strategy", "rung", "n_rows", "n_clusters",
                    "strict", "ci_lo", "ci_hi", "ci_halfwidth_pts",
                    "grid", "container", "normalized", "gap", "p50_s", "p99_s",
                    "on_pareto_frontier"])
        fkeys = {k for _, _, k in front}
        for k in sorted(stats, key=lambda z: -stats[z]["point"]):
            st = stats[k]
            w.writerow([k[0], k[1], rung_of(*k), st["n"], st["n_clusters"],
                        f"{st['point']:.4f}", f"{st['lo']:.4f}", f"{st['hi']:.4f}",
                        f"{(st['hi']-st['lo'])/2*100:.1f}",
                        f"{st['grid']:.4f}", f"{st['box']:.4f}",
                        f"{st['norm']:.4f}", f"{st['gap']:.4f}",
                        f"{st['p50']:.3f}", f"{st['p99']:.3f}",
                        "yes" if k in fkeys else ""])
    print(f"  wrote audit/figures/cell_stats.csv")

    print("\nPARETO FRONTIER (no cell is both faster and more accurate):")
    for lat, acc, k in front:
        print(f"   {cell_label(*k):34} acc={acc:.3f}  p50={lat:.3f}s  [{rung_of(*k)}]")

    print("\nMEDIAN CI HALF-WIDTH: "
          f"{sorted((st['hi']-st['lo'])/2*100 for st in stats.values())[len(stats)//2]:.1f} pts")


if __name__ == "__main__":
    main()
