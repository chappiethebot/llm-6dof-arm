r"""V2-style figures rendered against the V3 held-out results. READ-ONLY.

bench/figures.py is NOT modified: it is the provenance of the committed V2
figures. This file reproduces its visual style (same colormaps, same 3x3 cell
layout, same log-scale tail plot) against the four V3 _scored CSVs.

WHY FLAGS ALONE ARE NOT ENOUGH -- five blockers in bench/figures.py:
  1. load() takes ONE path. The V3 measurement is split across four files
     (rung1, rag, lora, cls) and there is no way to union them via --seen/--held.
  2. Every figure builds `sets = [("Seen", seen)] + ...`, so the first panel is
     always LABELLED "Seen". V3 is held-out only; the label would be false.
  3. fig_confusion renders ONE model (--model). Eight are wanted.
  4. Axis and colorbar labels say "strict accuracy", which this audit
     established is not exact match (AUDIT_REPORT.md B1-a).
  5. is_ambiguous() classifies a whole CELL from the FIRST matching row's
     phrasing. On held_out_v3 that is lossy: grid_B1 and grid_B3 each contain
     9 legend-token commands and 3 without, so a per-cell label misassigns 3
     rows in each. This file classifies PER ROW and reports both.

MODELS is also hardcoded to the 5 base models there; models_in() appends
extras, so the LoRA pair and the classifier would appear but in arbitrary
order. Here the eight are ordered explicitly.

BASIS. fig1 and fig2 are GRID-ONLY by construction (they are indexed by
grid cell), so they cover the 181 grid commands and never the 36 container
commands. fig3 is grid-only for the same reason. fig4 covers all 217. Every
caption states which.

POOLING. Like the V2 figures, a model's panel POOLS its strategies
(5 for each base model, 1 for each LoRA and the classifier). This is what
makes the V2 and V3 panels comparable; it also means a base model's panel
averages a good strategy with a bad one.
"""
import collections, csv, json, math, os, statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

csv.field_size_limit(2**31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTDIR = os.path.join(ROOT, "audit", "figures", "v3_style")

MODELS = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
          "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b",
          "ollama:qwen2.5-0.5b-lora", "ollama:qwen2.5-3b-lora",
          "classifier:distilbert"]
ROWS, COLS = list("ABC"), list("123")
CELLS = [f"grid_{r}{c}" for r in ROWS for c in COLS]
LEGEND_TOKENS = {"middle", "center", "centre"}
CONTAINER_START = 181

# The metric is strict_match. NEVER label it "exact match": llm/score.py drops
# `modifiers` when falsy and 71% of strict-correct rows depend on that rule.
METRIC = "action match (verb, target and location correct)"
METRIC_SHORT = "action match (%)"


def short(m):
    return (m.replace("openai:", "").replace("ollama:", "")
             .replace("classifier:", "cls:"))


def method_of(m):
    if m.startswith("classifier:"):
        return "task classifier"
    if m.endswith("-lora"):
        return "fine-tuned (LoRA)"
    return "prompted"


def load_all():
    d = {}
    for tag in ("rung1", "rag", "lora", "cls"):
        p = os.path.join(ROOT, "bench", f"results_heldout_v3_{tag}_scored.csv")
        with open(p, encoding="utf-8", newline="") as f:
            for r in csv.DictReader(f):
                d[(r.get("dataset", ""), r["model"], r["strategy"],
                   r["command_id"], r["run_index"])] = r
    return [r for r in d.values() if r.get("is_warmup") != "True"]


def truth_loc(r):
    try:
        return json.loads(r.get("truth_json") or "{}").get("location")
    except (json.JSONDecodeError, TypeError):
        return None


def pred_loc(r):
    try:
        obj = json.loads(r["pred_json"]) if r.get("pred_json") else None
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


def is_ambiguous_row(r):
    """PER ROW, not per cell -- grid_B1/B3 contain both kinds of phrasing."""
    c = r.get("command", "").lower().replace(",", " ")
    return bool(set(c.split()) & LEGEND_TOKENS)


def cell_acc(rows, model):
    acc = {c: [0, 0] for c in CELLS}
    for r in rows:
        if r.get("model") != model:
            continue
        t = truth_loc(r)
        if t not in acc:
            continue
        acc[t][1] += 1
        if r.get("strict_match") == "True":
            acc[t][0] += 1
    return ({c: (100 * h / n if n else float("nan")) for c, (h, n) in acc.items()},
            {c: n for c, (h, n) in acc.items()})


def save(fig, name, caption=None):
    os.makedirs(OUTDIR, exist_ok=True)
    if caption:
        words, lines, cur = caption.split(), [], ""
        for w in words:
            if len(cur) + len(w) + 1 > 130:
                lines.append(cur); cur = w
            else:
                cur = f"{cur} {w}".strip()
        lines.append(cur)
        fig.canvas.draw()
        rend = fig.canvas.get_renderer()
        y0 = min(a.get_tightbbox(rend).transformed(fig.transFigure.inverted()).y0
                 for a in fig.get_axes())
        fig.text(0.5, y0 - 0.04, "\n".join(lines), ha="center", va="top",
                 fontsize=8.5, style="italic", color="#333333")
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(OUTDIR, f"{name}.{ext}"), dpi=150,
                    bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote audit/figures/v3_style/{name}.png / .pdf")


# ------------------------------------------------------------------ fig 1
def no_location_rate(rows, model):
    """Share of a model's GRID commands producing no parseable location.

    These are folded into the heatmap cells as failures, which is correct
    arithmetic but misleading to read: a dark cell looks like the model chose
    the WRONG place, when for some models it produced nothing usable at all.
    qwen2.5:0.5b is the extreme case at ~47%. Stated per panel rather than as a
    fourth column, which would put two different quantities on one colour scale.
    """
    g = [r for r in rows if r.get("model") == model and truth_loc(r) in CELLS]
    if not g:
        return float("nan")
    return 100 * sum(1 for r in g if pred_loc(r) not in CELLS) / len(g)


def fig_cell_accuracy(rows, dump):
    fig, axes = plt.subplots(1, len(MODELS), figsize=(3.1 * len(MODELS), 4.2),
                             squeeze=False)
    im = None
    for j, m in enumerate(MODELS):
        ax = axes[0][j]
        a, n = cell_acc(rows, m)
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
        nl = no_location_rate(rows, m)
        ax.set_title(f"{short(m)}\n{method_of(m)}\n"
                     f"no location: {nl:.0f}%",
                     fontsize=8.5, fontweight="bold",
                     color="#8B0000" if nl >= 25 else "black")
        for c in CELLS:
            dump.append(dict(figure="fig1", model=m, key=c,
                             value=f"{a[c]:.1f}", n=n[c]))
        dump.append(dict(figure="fig1", model=m, key="no_location_pct",
                         value=f"{nl:.1f}", n=sum(n.values())))
    fig.suptitle(f"{METRIC.capitalize()} per table cell (%)  —  robot is below row C",
                 fontsize=13, y=1.04)
    fig.colorbar(im, ax=axes, shrink=0.72, label=METRIC_SHORT)
    save(fig, "fig1_cell_accuracy",
         "Where on the table each model succeeds. Basis: the 181 grid commands only "
         "(the 36 container commands have no grid cell). Strategies are pooled per model, "
         "as in the V2 figure. READ THE PANEL TITLES: a dark cell mixes wrong-place "
         "answers with commands that produced no usable location at all, and for some "
         "models the second is nearly half of them. Takeaway: dark rows are mostly a "
         "collapsed output distribution, not a spatial failure on those cells.")


# ------------------------------------------------------------------ fig 2
def fig_confusion(rows, dump):
    labels = [c[5:] for c in CELLS] + ["other"]
    fig, axes = plt.subplots(2, 4, figsize=(22, 11), squeeze=False)
    im = None
    for idx, m in enumerate(MODELS):
        ax = axes[idx // 4][idx % 4]
        M = np.zeros((len(CELLS), len(CELLS) + 1))
        for i, t in enumerate(CELLS):
            cnt, n = collections.Counter(), 0
            for r in rows:
                if r.get("model") != m or truth_loc(r) != t:
                    continue
                n += 1
                cnt[pred_loc(r)] += 1
            if not n:
                continue
            for j, p in enumerate(CELLS):
                M[i][j] = 100 * cnt.get(p, 0) / n
            M[i][-1] = 100 - M[i][:-1].sum()
            for j, lab in enumerate(labels):
                dump.append(dict(figure="fig2", model=m,
                                 key=f"{t[5:]}->{lab}", value=f"{M[i][j]:.1f}", n=n))
        im = ax.imshow(M, cmap="Blues", vmin=0, vmax=100)
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                if M[i][j] >= 1:
                    ax.text(j, i, f"{M[i][j]:.0f}", ha="center", va="center",
                            fontsize=7,
                            color="white" if M[i][j] > 55 else "black")
        ax.set_xticks(range(len(labels)), labels, fontsize=7, rotation=45)
        ax.set_yticks(range(len(CELLS)), [c[5:] for c in CELLS], fontsize=7)
        ax.set_xlabel("predicted", fontsize=8)
        ax.set_ylabel("true", fontsize=8)
        ax.set_title(f"{short(m)}  ({method_of(m)})", fontsize=10, fontweight="bold")
    fig.suptitle("Confusion matrices, all eight clients (row %)", fontsize=15, y=0.98)
    fig.colorbar(im, ax=axes, shrink=0.5, label="% of that true cell's rows")
    save(fig, "fig2_confusion",
         "True grid cell against predicted, as a percentage of each true cell's rows; "
         "'other' collects container symbols, runtime symbols, absent and malformed "
         "predictions. Basis: the 181 grid commands only. Takeaway: errors concentrate "
         "on specific wrong cells rather than scattering, so they are systematic.")


# ------------------------------------------------------------------ fig 3
def fig_clean_vs_amb(rows, dump):
    fig, ax = plt.subplots(figsize=(11, 5))
    clean, ambig = [], []
    for m in MODELS:
        cs = [r for r in rows if r.get("model") == m and truth_loc(r) in CELLS
              and not is_ambiguous_row(r)]
        am = [r for r in rows if r.get("model") == m and truth_loc(r) in CELLS
              and is_ambiguous_row(r)]
        f = lambda g: 100 * sum(x.get("strict_match") == "True" for x in g) / len(g) if g else 0
        clean.append(f(cs)); ambig.append(f(am))
        dump.append(dict(figure="fig3", model=m, key="clean",
                         value=f"{f(cs):.1f}", n=len(cs)))
        dump.append(dict(figure="fig3", model=m, key="ambiguous",
                         value=f"{f(am):.1f}", n=len(am)))
    x = np.arange(len(MODELS))
    ax.bar(x - 0.2, clean, 0.4, label="clean phrasing", color="#3d7d3d")
    ax.bar(x + 0.2, ambig, 0.4, label='phrasing contains "middle"/"center"',
           color="#b5482f")
    for xi, (a, b) in enumerate(zip(clean, ambig)):
        ax.text(xi - 0.2, a + 1.5, f"{a:.0f}", ha="center", fontsize=8)
        ax.text(xi + 0.2, b + 1.5, f"{b:.0f}", ha="center", fontsize=8)
    ax.set_xticks(x, [f"{short(m)}\n{method_of(m)}" for m in MODELS],
                  fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylabel(METRIC_SHORT)
    ax.legend(fontsize=9)
    fig.suptitle('Phrase-legend collision: phrasings containing "middle"/"center"\n'
                 "compete with the column-2 label in the prompt legend", fontsize=12)
    save(fig, "fig3_clean_vs_ambiguous",
         "Accuracy split by whether the command contains a legend-colliding word. "
         "Basis: the 181 grid commands only. Classified PER COMMAND, not per cell: "
         "grid_B1 and grid_B3 each contain 9 colliding and 3 clean phrasings, so the "
         "per-cell method used in the V2 figure mislabels 3 rows in each.")


# ------------------------------------------------------------------ fig 4
def fig_latency(rows, dump):
    def pctl(v, p):
        v = sorted(v)
        if not v:
            return float("nan")
        k = (len(v) - 1) * p / 100
        f, c = math.floor(k), math.ceil(k)
        return v[f] if f == c else v[f] + (v[c] - v[f]) * (k - f)

    fig, ax = plt.subplots(figsize=(11, 5.4))
    stats = ["p50", "p95", "p99", "max"]
    w = 0.2
    x = np.arange(len(MODELS))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]
    for si, s in enumerate(stats):
        vals = []
        for m in MODELS:
            lat = []
            for r in rows:
                if r.get("model") != m:
                    continue
                try:
                    v = float(r["latency_s"])
                except (TypeError, ValueError):
                    continue
                if r.get("strategy") == "rag" and r.get("retrieval_latency_s"):
                    v += float(r["retrieval_latency_s"])
                lat.append(v)
            val = max(lat) if s == "max" else pctl(lat, int(s[1:]))
            vals.append(val)
            dump.append(dict(figure="fig4", model=m, key=s,
                             value=f"{val:.4f}", n=len(lat)))
        ax.bar(x + (si - 1.5) * w, vals, w, label=s, color=colors[si])
    ax.set_yscale("log")
    ax.set_xticks(x, [f"{short(m)}\n{method_of(m)}" for m in MODELS],
                  fontsize=7.5, rotation=12, ha="right")
    ax.set_ylabel("latency per command, seconds (log scale)")
    ax.grid(axis="y", alpha=0.3, which="both")
    ax.legend(fontsize=9, ncol=4)
    # The classifier sits ~2 orders of magnitude below everything else; pin the
    # limits so its bars stay visible instead of collapsing onto the axis.
    ax.set_ylim(0.005, 400)
    fig.suptitle("Latency tail per client (warm-up row excluded, log scale)",
                 fontsize=13)
    save(fig, "fig4_latency_tail",
         "Median, 95th, 99th percentile and maximum response time. Basis: all 217 "
         "commands. Retrieval time is included for RAG rows; strategies are pooled per "
         "model. Takeaway: the tail is where the cost lives — several models have a "
         "maximum two orders of magnitude above their median.")


def main():
    rows = load_all()
    print(f"rows (warm-up excluded): {len(rows):,}")
    grid_rows = [r for r in rows if truth_loc(r) in CELLS]
    print(f"grid-cell rows: {len(grid_rows):,}  |  models: {len(MODELS)}")
    dump = []
    fig_cell_accuracy(rows, dump)
    fig_confusion(rows, dump)
    fig_clean_vs_amb(rows, dump)
    fig_latency(rows, dump)
    p = os.path.join(OUTDIR, "v3_style_numbers.csv")
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "model", "key", "value", "n"])
        w.writeheader(); w.writerows(dump)
    print(f"  wrote audit/figures/v3_style/v3_style_numbers.csv ({len(dump)} rows)")

    # ---- whole-row / whole-column zero check, the V2 finding to reproduce ----
    print("\nWHOLE-ROW / WHOLE-COLUMN ZEROS (V2 fig1 showed these; do they reproduce?)")
    print(f"{'model':26}{'zero rows':>26}{'zero cols':>26}")
    for m in MODELS:
        a, n = cell_acc(rows, m)
        zr = [r for r in ROWS
              if all((a[f"grid_{r}{c}"] == 0 or math.isnan(a[f"grid_{r}{c}"])) for c in COLS)]
        zc = [c for c in COLS
              if all((a[f"grid_{r}{c}"] == 0 or math.isnan(a[f"grid_{r}{c}"])) for r in ROWS)]
        print(f"{short(m):26}{(','.join(zr) or '-'):>26}{(','.join(zc) or '-'):>26}")


if __name__ == "__main__":
    main()
