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

# Method labels are written for a reader who has never seen this project: the
# rung numbers are internal shorthand and mean nothing outside it. The number is
# kept in parentheses only where it costs no clarity.
def rung_of(model, strategy):
    if model.startswith("classifier:"):
        return "task classifier (4)"
    if model.endswith("-lora"):
        return "fine-tuned, LoRA (3)"
    return "retrieval, RAG (2)" if strategy == "rag" else "prompting (1)"

RUNG_STYLE = {
    "prompting (1)":       ("o", "#4C72B0"),
    "retrieval, RAG (2)":  ("s", "#DD8452"),
    "fine-tuned, LoRA (3)":("^", "#55A868"),
    "task classifier (4)": ("D", "#C44E52"),
}

# NEVER LABEL THE strict_match METRIC "EXACT MATCH" ON A FIGURE. It is not one.
# llm/score.py drops `modifiers` from both sides when falsy, and 8,897 of the
# 12,528 strict-correct rows in this study (71%) have pred.modifiers !=
# truth.modifiers and depend on that rule. The evaluation set carries an empty
# modifiers list on all 217 commands, so the field is not under test -- which
# makes the rule harmless, NOT makes the metric exact. Calling it exact match on
# an axis is the single most visible place to state something false, and it was
# filed as a Methods wording defect (AUDIT_REPORT.md B1-a). Use "action match"
# or "correct action", matching the drafted disclosure sentence.

# One-line caption under each figure: what it shows, and the takeaway.
CAPTIONS = {
    "fig1_strict_accuracy_clustered_ci":
        "Share of commands where the predicted action is correct, for all 28 "
        "model/method combinations on 217 unseen "
        "commands. Takeaway: the four methods do not rank cleanly — a prompted "
        "cloud model leads, but a small task classifier beats every local "
        "generative model, and many differences are inside the error bars.",
    "fig2_grid_vs_container":
        "The same 28 combinations split by command type: 181 commands needing "
        "spatial reasoning versus 36 where the answer word is already in the "
        "command. Takeaway: these are two different tasks, and pooling them "
        "inflates every headline number.",
    "fig3_cost_accuracy_pareto":
        "Accuracy against response time for all 28 combinations; the line marks "
        "the best available trade-offs. Takeaway: three options are worth "
        "considering and every other combination is beaten on both speed and "
        "accuracy — including every locally-run generative model.",
    "fig4_strict_vs_normalized_gap":
        "How much accuracy each combination recovers when output formatting "
        "errors are forgiven. Takeaway: a large bar means the model understood "
        "the command but emitted the wrong JSON shape — a fixable formatting "
        "problem, not a reasoning failure.",
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
    cap = CAPTIONS.get(name)
    if cap:
        # wrap by hand: figtext does not wrap, and wrap=True re-flows on resize
        words, lines, cur = cap.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 118:
                lines.append(cur); cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        # Position from the axes' TIGHT bbox, which includes the rotated tick
        # labels. A fixed figure-coordinate y (or supxlabel with an explicit y)
        # lands on top of them in the bar charts, where the labels are ~180 px
        # tall. bbox_inches="tight" then grows the canvas to include the text.
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        y0 = min(a.get_tightbbox(rend).transformed(fig.transFigure.inverted()).y0
                 for a in fig.get_axes())
        fig.text(0.5, y0 - 0.025, "\n".join(lines), ha="center", va="top",
                 fontsize=8.5, style="italic", color="#333333")
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
    ax.set_xlabel("action match: verb, target and location all correct\n"
                  "(95% CI, bootstrapped over commands)")
    ax.set_title("Correct actions: 28 model and method combinations\n"
                 "217 unseen commands; bars are 95% confidence intervals",
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
           label="spatial reasoning (181 commands, 11 possible answers, answer not in command)",
           color="#4C72B0")
    # Accuracy is a proportion: clip the +-16 pt interval at [0,1] rather than
    # drawing bars that run off the top of the axis.
    bv = [stats[k]["box"] for k in order]
    lo_err = [min(0.16, v) for v in bv]
    hi_err = [min(0.16, 1.0 - v) for v in bv]
    ax.bar([i + gw/2 for i in x], bv, gw,
           yerr=[lo_err, hi_err], capsize=2, error_kw=dict(lw=0.8, alpha=0.6),
           label="answer named in command (36 commands, 3 possible answers; CI $\\pm$16 pts)",
           color="#DD8452")
    ax.set_xticks(list(x))
    ax.set_xticklabels([cell_label(*k) for k in order], rotation=90, fontsize=7)
    ax.set_ylabel("action match (verb, target and location correct)")
    ax.set_ylim(0, 1.05)
    ax.set_title("Two different tasks, reported separately\n"
                 "Spatial-reasoning commands versus commands that name the answer",
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
            color="black", lw=1.2, alpha=0.7, zorder=2, label="best available trade-offs")
    # Label the frontier points ON the plot, with their numbers, so the reader
    # never has to cross-reference a legend to read the result.
    # Explicit per-point offsets: the two gpt-4o-mini points are ~0.1 s apart on
    # a log axis, so a shared offset rule overlaps their boxes.
    offsets = [(14, 16), (-8, -46), (16, -6)]     # by frontier order (ascending latency)
    aligns = ["left", "center", "left"]
    for i, (lat, acc, k) in enumerate(front):
        dx, dy = offsets[i] if i < len(offsets) else (12, 12)
        ax.annotate(f"{cell_label(*k)}\n{acc:.3f} accuracy, {lat*1000:.0f} ms",
                    (lat, acc), fontsize=8, fontweight="bold",
                    ha=aligns[i] if i < len(aligns) else "left",
                    xytext=(dx, dy), textcoords="offset points",
                    bbox=dict(boxstyle="round,pad=0.35", fc="white",
                              ec="#555555", lw=0.7, alpha=0.95),
                    arrowprops=dict(arrowstyle="-", lw=0.7, color="#555555"))
    ax.set_ylim(-0.05, 1.08)
    ax.set_xscale("log")
    ax.set_xlabel("median response time per command, seconds (log scale; retrieval time included)")
    ax.set_ylabel("action match (verb, target and location correct)")
    ax.set_title("Accuracy versus response time: 28 model and method combinations\n"
                 "The line marks the best available trade-offs; points below it are beaten on both",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    h = [Line2D([], [], marker=m, color=c, ls="", label=r)
         for r, (m, c) in RUNG_STYLE.items()]
    h.append(Line2D([], [], color="black", lw=1.2, label="best available trade-offs"))
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
    ax.set_ylabel("accuracy gained when formatting is forgiven")
    ax.set_title("Accuracy recovered by forgiving output-format errors\n"
                 "A tall bar means the command was understood but the JSON shape was wrong",
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
