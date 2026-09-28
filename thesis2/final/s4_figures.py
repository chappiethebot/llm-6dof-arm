"""thesis2/final/s4_figures.py -- STAGE 4: the six figures.

    python thesis2/final/s4_figures.py

Writes to thesis2/figures/final/:

  fig1_action_vs_latency.png   action match vs latency p50 (log x), one panel per
                               split. Local arms at their Stage-3 protocol latency;
                               cloud arms at their Stage-1 per-command latency
                               (network included) -- the only figure they appear in.
                               held_out_v3 also carries the 31 local V5 cells.
  fig2_action_vs_energy.png    action match vs J/command (log x). An arm with
                               J <= 0 sits on a floor marker labelled "below GPU
                               measurement floor", never dropped. No cloud panel:
                               cloud energy is not measurable. v3 carries the 31
                               local V5 cells. The frontier uses GPU-measured arms
                               only: J > 0 AND model VRAM above idle > 0. CPU-only
                               arms stay plotted but are excluded (2026-09-28).
  fig3_row_by_wordgroup.png    heatmap: row accuracy by row-word group, every arm,
                               v3 and v4c side by side in ONE fixed arm order
  fig4_row_confusion.png       heatmap: row confusion signature (A<->C mirror vs
                               default-to-B), same arm order
  fig5_a10_breakdown.png       A10 p50 total from unified_efficiency.csv (Stage 3)
                               and the drafter p50 from the same Stage-3 run's log,
                               both splits; first-rejection field on held_out_v3
                               (A11, earlier run) and held_out_v4c (A11 --split)
  fig6_action_vs_vram.png      action match vs model VRAM above idle (MB); a value
                               <= 0 plots at 0 and reads "~0"

Rules (2026-09-28):
  * every point is a NUMBERED marker; a key table beside the plot maps the number
    to the system and its values -- no free-floating text labels;
  * cascade_oracle is never a point: it is a dashed horizontal "ceiling" line;
  * rules_open is marked "vocabulary-unfair";
  * both are excluded from every Pareto frontier;
  * no filled/hollow "claimable" marking: claimability belongs to comparisons
    (final_significance.csv), not to systems.
"""
import csv
import math
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402
import numpy as np                                                      # noqa: E402

CEILING = "cascade_oracle"
UNFAIR = "rules_open"
EXCLUDED_FROM_FRONTIER = {CEILING, UNFAIR}

STYLE = {                     # marker, face, edge, size, legend label
    "local": ("o", "#4C72B0", "#1f3d6b", 46, "local arm"),
    "cloud": ("s", "#DD8452", "#8a4a1c", 50, "cloud arm (Stage-1 latency, network included)"),
    "unfair": ("X", "#9e9e9e", "#555555", 60, "rules_open -- vocabulary-unfair"),
    "v5": ("^", "#55A868", "#2d6b3c", 34, "V5 local cell (Ollama Q4_K_M)"),
    "floor": ("v", "#bbbbbb", "#555555", 46, "J <= 0: below GPU measurement floor"),
    "cpu": ("o", "white", "#4C72B0", 46, "CPU-only arm (no model on the GPU)"),
}
SPLIT_TAG = {"held_out_v3": "v3", "held_out_v4c": "v4c"}


def load(name):
    p = C.OUT / name
    if not p.exists():
        return []
    with p.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def f(v):
    try:
        x = float(v)
        return x if math.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def pareto(points):
    """Points (x, y, ...) maximising y while minimising x; extra fields ride along."""
    front, best = [], -1e9
    for p in sorted(points, key=lambda p: (p[0], -p[1])):
        if p[1] > best:
            front.append(p)
            best = p[1]
    return front


# ------------------------------------------------------------------ numbering
def numbering(metrics, v5):
    """Stable numbers: registry arms 1..N alphabetically, V5 local cells V1..V31."""
    arms = sorted({r["arm"] for r in metrics})
    num = {a: str(i + 1) for i, a in enumerate(arms)}
    cells = [r for r in v5 if r["kind"] == "local"]
    for i, r in enumerate(cells):
        num["v5:%s/%s" % (r["model"], r["strategy"])] = "V%d" % (i + 1)
    return num


def place_labels(ax, pts, fontsize=6.5):
    """Put each point's number next to it without overlapping another number
    or marker: greedy search over candidate offsets, in display coordinates."""
    fig = ax.figure
    fig.canvas.draw()
    trans = ax.transData
    placed = []
    marks = [trans.transform((x, y)) for x, y, _ in pts]
    cand = [(5, 3), (5, -9), (-5, 3), (-5, -9), (8, 10), (-8, 10), (8, -15),
            (-8, -15), (14, -3), (-14, -3), (12, 18), (-12, 18), (18, 6), (-18, 6),
            (20, -12), (-20, -12), (4, 24), (-4, 24), (4, -24), (-4, -24)]
    for (x, y, lbl), (px, py) in zip(pts, marks):
        w, h = 4.2 * len(lbl) + 2, 8
        best = None
        for dx, dy in cand:
            bx = px + dx if dx >= 0 else px + dx - w
            by = py + dy
            box = (bx, by, bx + w, by + h)
            hit = any(not (box[2] < b[0] or box[0] > b[2] or box[3] < b[1] or box[1] > b[3])
                      for b in placed)
            hit = hit or any(box[0] - 3 < mx < box[2] + 3 and box[1] - 3 < my < box[3] + 3
                             for mx, my in marks if (mx, my) != (px, py))
            if not hit:
                best = (dx, dy, box)
                break
        if best is None:
            dx, dy = cand[0]
            bx = px + dx
            best = (dx, dy, (bx, py + dy, bx + w, py + dy + h))
        dx, dy, box = best
        placed.append(box)
        far = abs(dx) > 9 or abs(dy) > 11
        ax.annotate(lbl, (x, y), xytext=(dx, dy), textcoords="offset points",
                    fontsize=fontsize, ha="left" if dx >= 0 else "right", va="bottom",
                    color="#222222",
                    arrowprops=dict(arrowstyle="-", lw=.4, color="#888888")
                    if far else None)


def key_table(ax, header, entries, title, note, n_col=3):
    """Full-width key under the plots: fixed-width monospace columns, each
    repeating the header, so no column can run into the next."""
    ax.axis("off")
    ax.set_title(title, fontsize=8.5, loc="left")
    per = math.ceil(len(entries) / n_col)
    for c in range(n_col):
        chunk = [header] + entries[c * per:(c + 1) * per]
        ax.text(c / n_col, 1.0, "\n".join(chunk), family="monospace", fontsize=6.2,
                va="top", ha="left", transform=ax.transAxes)
    ax.text(0, -0.02, note, fontsize=7, va="top", ha="left", transform=ax.transAxes,
            style="italic", color="#333333")


def fmt(v, kind):
    if v is None:
        return "-"
    if kind == "lat":
        return "%.3g ms" % v
    if kind == "j":
        return "%.3g J" % v
    return "%.0f MB" % v


# ------------------------------------------------------------------ scatters
def fig_scatter(metrics, v5, num, which, fname):
    """which: 'lat' | 'j' | 'vram'."""
    xlab = {"lat": "latency p50 (ms, log)",
            "j": "J / command (log)",
            "vram": "model VRAM above idle (MB)"}[which]
    title = {"lat": "action match vs latency",
             "j": "action match vs energy (GPU, idle-subtracted)",
             "vram": "action match vs model VRAM above idle"}[which]
    logx = which in ("lat", "j")

    fig = plt.figure(figsize=(17, 12.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.62], hspace=0.42, wspace=0.16)
    axes = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
    kax = fig.add_subplot(gs[1, :])

    # the floor for J <= 0: one decade below the smallest positive J plotted
    pos_j = [f(r.get("joules_per_command_raw")) for r in metrics]
    pos_j += [f(r.get("v3_joules_per_command_raw")) for r in v5]
    pos_j = [x for x in pos_j if x is not None and x > 0]
    j_floor = 10 ** (math.floor(math.log10(min(pos_j))) - 1) if pos_j else 1e-6

    rows_key = {}

    def floor_x(k):
        """k-th floor point on a panel: spread across the grey band (log offsets
        -0.3..+0.3) so coincident J <= 0 points stay legible. Position inside
        the band carries no meaning."""
        return j_floor * 10 ** (((k % 5) - 2) * 0.15)

    for ax, split in zip(axes, C.SPLITS):
        tag = SPLIT_TAG[split]
        n_floor = 0
        if logx:
            ax.set_xscale("log")    # before plotting: the frontier reads the limits
        pts, front_pts, labels = [], [], []
        used_styles = set()
        for r in metrics:
            if r["split"] != split or r["status"] != "valid":
                continue
            a, y = r["arm"], f(r.get("action_match"))
            if y is None:
                continue
            cloud = r.get("kind") == "cloud"
            if which == "lat":
                x = f(r.get("measured_latency_p50_ms")) if not cloud else None
                x = x if x is not None else f(r.get("latency_p50_ms"))
            elif which == "j":
                if cloud:
                    continue
                x = f(r.get("joules_per_command_raw"))
            else:
                if cloud:
                    continue
                x = f(r.get("vram_above_idle_mb_raw"))
                x = None if x is None else max(x, 0.0)      # <= 0 plots at 0
            if a == CEILING:
                ax.axhline(y, ls="--", lw=1.1, color="#7f7f7f", zorder=1)
                ax.annotate("%s -- %s ceiling, not a system (%.3f)" % (num[a], a, y),
                            (0.01, y), xycoords=("axes fraction", "data"),
                            fontsize=6.5, color="#555555", va="bottom")
                rows_key.setdefault(a, {})[tag] = (y, None)
                continue
            if x is None:
                rows_key.setdefault(a, {})[tag] = (y, None)
                continue
            style = "unfair" if a == UNFAIR else ("cloud" if cloud else "local")
            if which == "j" and x <= 0:
                style, x_plot = "floor", floor_x(n_floor)
                n_floor += 1
            else:
                x_plot = x
            rows_key.setdefault(a, {})[tag] = (y, x)
            vram = f(r.get("vram_above_idle_mb_raw"))
            cpu_only = which == "j" and vram is not None and vram <= 0
            if cpu_only and style == "local":
                style = "cpu"
            mk, fc, ec, sz, _ = STYLE[style]
            used_styles.add(style)
            ax.scatter(x_plot, y, marker=mk, s=sz, facecolors=fc, edgecolors=ec,
                       linewidths=.6, zorder=3)
            pts.append((x_plot, y, num[a]))
            # Energy frontier (2026-09-28): GPU-measured arms only -- J > 0 AND a
            # model on the GPU. Floor-band (J <= 0) and CPU-only arms stay plotted
            # but cannot be on it: their CPU energy is not measured.
            gpu_measured = not (which == "j" and (style == "floor" or cpu_only))
            if a not in EXCLUDED_FROM_FRONTIER and gpu_measured:
                front_pts.append((x_plot, y, x_plot))

        if split == "held_out_v3" and which in ("lat", "j"):
            for r in v5:
                if r["kind"] != "local":
                    continue
                key = "v5:%s/%s" % (r["model"], r["strategy"])
                y = f(r.get("v3_action"))
                x = (f(r.get("v3_latency_p50_ms")) if which == "lat"
                     else f(r.get("v3_joules_per_command_raw")))
                if y is None or x is None:
                    continue
                style = "v5"
                x_plot = x
                if which == "j" and x <= 0:
                    style, x_plot = "floor", floor_x(n_floor)
                    n_floor += 1
                vram5 = f(r.get("v3_vram_above_idle_mb_raw"))
                cpu5 = which == "j" and vram5 is not None and vram5 <= 0
                mk, fc, ec, sz, _ = STYLE[style]
                used_styles.add(style)
                ax.scatter(x_plot, y, marker=mk, s=sz, facecolors=fc, edgecolors=ec,
                           linewidths=.6, zorder=3)
                pts.append((x_plot, y, num[key]))
                if not (which == "j" and (style == "floor" or cpu5)):
                    front_pts.append((x_plot, y, x_plot))
                rows_key.setdefault(key, {})["v3"] = (y, x)

        if front_pts:
            fr = pareto(front_pts)
            # Staircase: for any budget >= x_i the best achievable accuracy is y_i,
            # so the frontier runs flat to the right edge. A frontier of ONE point
            # (v4c energy: flat_cpu dominates) is then visible, not a lone dot.
            ax.autoscale_view()
            x_right = ax.get_xlim()[1]
            xs = [q[0] for q in fr] + [x_right]
            ys = [q[1] for q in fr] + [fr[-1][1]]
            ax.plot(xs, ys, "-", lw=1.1, color="#C44E52", alpha=.8, zorder=2,
                    drawstyle="steps-post")
            ax.scatter([q[2] for q in fr], [q[1] for q in fr], s=150, facecolors="none",
                       edgecolors="#C44E52", linewidths=.9, zorder=2)
            ax.set_xlim(right=x_right)
        if which == "j":
            ax.axvspan(j_floor / 3, j_floor * 3, color="#dddddd", alpha=.6, zorder=0)
            ax.annotate("below GPU\nmeasurement floor\n(J <= 0)", (j_floor, 0.02),
                        xycoords=("data", "axes fraction"), fontsize=6.5,
                        ha="center", va="bottom", color="#444444")
        if logx:
            ax.set_xscale("log")
        if which == "vram":
            ax.axvline(0, color="#bbbbbb", lw=.8, zorder=0)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel(xlab)
        ax.set_ylabel("action match")
        ax.grid(alpha=.25)
        ax.set_title("%s -- %s" % (split, title), fontsize=10)
        handles = [plt.Line2D([], [], marker=STYLE[s][0], ls="", markerfacecolor=STYLE[s][1],
                              markeredgecolor=STYLE[s][2], label=STYLE[s][4])
                   for s in ("local", "cpu", "cloud", "unfair", "v5", "floor")
                   if s in used_styles]
        handles.append(plt.Line2D([], [], ls="-", color="#C44E52", marker="o",
                                  markerfacecolor="none", markersize=9,
                                  label="Pareto frontier, staircase; ringed = on it "
                                        "(excl. ceiling, vocabulary-unfair)"))
        handles.append(plt.Line2D([], [], ls="--", color="#7f7f7f",
                                  label="cascade_oracle ceiling (not a system)"))
        ax.legend(handles=handles, fontsize=7, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
        place_labels(ax, pts)

    # ---- key table
    kind = {"lat": "lat", "j": "j", "vram": "vram"}[which]
    header = "%-4s %-33s %-17s %-17s" % ("#", "system", "v3 acc | x", "v4c acc | x")
    ent = []
    order = sorted(rows_key, key=lambda k: (k.startswith("v5:"),
                                             int(num[k].lstrip("V"))))
    for k in order:
        cells = []
        for tag in ("v3", "v4c"):
            v = rows_key[k].get(tag)
            if not v:
                cells.append("-")
                continue
            y, x = v
            xs = fmt(x, kind)
            if which == "j" and x is not None and x <= 0:
                xs = "~0 floor"
            if which == "vram" and x is not None and x <= 0:
                xs = "~0"
            cells.append("%.3f|%s" % (y, xs))
        name = k.replace("v5:ollama:", "V5 ").replace("v5:classifier:", "V5 ")
        if k == CEILING:
            name += " (ceiling)"
        if k == UNFAIR:
            name += " (vocab-unfair)"
        ent.append("%-4s %-33s %-17s %-17s" % (num[k], name[:33], cells[0], cells[1]))
    notes = {"lat": "local arms: Stage-3 protocol p50; cloud arms: Stage-1 p50 "
                    "(network). ensemble: offline pairing, no latency.",
             "j": "CPU-only arms: CPU energy not measured; excluded from the energy "
                  "frontier. Cloud arms: energy not measurable (latency figure only). "
                  "Ensemble: offline pairing, no energy. J <= 0 points are spread "
                  "across the grey floor band for legibility; position inside it "
                  "carries no meaning.",
             "vram": "peak VRAM minus the median idle memory.used of the idle baseline "
                     "in force (power_logs/idle_*.csv); a value <= 0 plots at 0 and "
                     "reads ~0. Cloud arms omitted."}[which]
    key_table(kax, header, ent, "key: number -> system, action match | x-value "
              "(V = V5 cell, held_out_v3 only)", notes)
    fig.savefig(C.FIGS / fname, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return fname


# ------------------------------------------------------------------ heatmaps
def arm_order(metrics):
    """ONE fixed order for fig3 and fig4: by held_out_v4c action match, desc."""
    v4c = {r["arm"]: f(r.get("action_match")) or 0 for r in metrics
           if r["split"] == "held_out_v4c"}
    return sorted(v4c, key=lambda a: (-v4c[a], a))


def arm_label(a):
    return a + (" (ceiling)" if a == CEILING else "") + (" (vocab-unfair)" if a == UNFAIR else "")


def heatmap(ax, mat, rows, cols, fmt_cell, vmax=1.0, cmap="RdYlGn"):
    arr = np.array([[np.nan if v is None else v for v in r] for r in mat], float)
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels(cols, fontsize=7.5)
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7.5)
    def ink(v):
        """white on dark cells: both ends of RdYlGn, the top of Reds"""
        if v is None:
            return "#777777"
        t = v / vmax
        dark = (t < 0.15 or t > 0.85) if cmap == "RdYlGn" else t > 0.6
        return "white" if dark else "#111111"

    for i, r in enumerate(mat):
        for j, v in enumerate(r):
            ax.text(j, i, fmt_cell(v), ha="center", va="center", fontsize=6.8,
                    color=ink(v))
    for x in np.arange(-.5, len(cols), 1):
        ax.axvline(x, color="white", lw=.6)
    return im


def fig_wordgroup(metrics, fname):
    groups = ["seen", "unseen", "mixed"]
    order = arm_order(metrics)
    by = {(r["arm"], r["split"]): r for r in metrics}
    n_of, mat = {}, []
    for a in order:
        row = []
        for split in C.SPLITS:
            parsed = {}
            for part in (by.get((a, split), {}).get("row_by_word_group") or "").split(";"):
                part = part.strip()
                if "/" in part:
                    g, frac = part.rsplit(" ", 1)
                    h, t = frac.split("/")
                    parsed[g.strip()] = (int(h), int(t))
                    n_of[(split, g.strip())] = int(t)
            for g in groups:
                h_t = parsed.get(g)
                row.append(h_t[0] / h_t[1] if h_t and h_t[1] else None)
        mat.append(row)
    cols = ["%s\n%s\n(n=%s)" % (SPLIT_TAG[s], g, n_of.get((s, g), "?"))
            for s in C.SPLITS for g in groups]
    fig, ax = plt.subplots(figsize=(10, 12))
    im = heatmap(ax, mat, [arm_label(a) for a in order], cols,
                 lambda v: "n/a" if v is None else "%.2f" % v)
    ax.axvline(2.5, color="black", lw=1.6)
    fig.colorbar(im, ax=ax, fraction=.03, pad=.02, label="row accuracy")
    ax.set_title("Row accuracy by row word -- seen = middle/center; unseen = upper/bottom "
                 "(v3) or back/front (v4c);\nmixed = both. One arm order for both splits "
                 "(by v4c action match). Chance = 0.33.", fontsize=9, pad=40)
    fig.tight_layout()
    fig.savefig(C.FIGS / fname, dpi=170)
    plt.close(fig)
    return fname


def fig_confusion(metrics, fname):
    order = arm_order(metrics)
    by = {(r["arm"], r["split"]): r for r in metrics}
    mat, n_grid = [], {}
    for a in order:
        row = []
        for split in C.SPLITS:
            r = by.get((a, split), {})
            s = r.get("row_signature") or ""
            n_grid[split] = r.get("n_row") or n_grid.get(split, "?")
            try:
                m = int(s.split("A<->C mirror")[1].split(";")[0])
                b = int(s.split("default-to-B")[1])
            except (IndexError, ValueError):
                m = b = None
            row += [m, b]
        mat.append(row)
    cols = ["%s\nA<->C mirror" % SPLIT_TAG[s] if k == 0 else "%s\ndefault to B" % SPLIT_TAG[s]
            for s in C.SPLITS for k in (0, 1)]
    vmax = max([v for r in mat for v in r if v is not None] + [1])
    fig, ax = plt.subplots(figsize=(8.5, 12))
    im = heatmap(ax, mat, [arm_label(a) for a in order], cols,
                 lambda v: "n/a" if v is None else "%d" % v, vmax=vmax, cmap="Reds")
    ax.axvline(1.5, color="black", lw=1.6)
    fig.colorbar(im, ax=ax, fraction=.04, pad=.02, label="grid commands")
    ax.set_title("Row confusion signature -- counts of the %s grid commands per split\n"
                 "A<->C mirror: far and near swapped; default to B: a non-B row answered "
                 "as B" % n_grid.get("held_out_v3", "?"), fontsize=9, pad=40)
    fig.tight_layout()
    fig.savefig(C.FIGS / fname, dpi=170)
    plt.close(fig)
    return fname


def stage3_drafter_p50(split):
    """The drafter p50 printed by the Stage-3 run of A10 on this split
    ("classifier draft floor: p50 X ms" in s3_energy_a10_speculative_<split>_*.log),
    from the latest log whose run passed. None if absent."""
    import re
    for lg in sorted(C.LOGS.glob("s3_energy_a10_speculative_%s_*.log" % split), reverse=True):
        txt = lg.read_text(encoding="utf-8", errors="replace")
        m = re.search(r"classifier draft floor: p50 ([0-9.]+) ms", txt)
        if m and "GATE PASS" in txt:
            return float(m.group(1)), lg.name
    return None, None


def fig_a10(fname):
    """Left: A10 p50 latency split into drafter and verification + decode, from the
    Stage-3 source -- the total is unified_efficiency.csv latency_p50_ms (the same
    number as the efficiency table), the drafter p50 is printed by the same Stage-3
    run. Right: first-rejection field, held_out_v3 (A11, earlier run) and
    held_out_v4c (A11 --split held_out_v4c on the final run's A8 outputs)."""
    from collections import Counter
    eff = {(r["arm"], r["split"]): r for r in load("unified_efficiency.csv")}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.6))
    names, draft, rest, tot, srcs = [], [], [], [], []
    for split in C.SPLITS:
        e = eff.get(("a10_speculative", split))
        d, src = stage3_drafter_p50(split)
        if not e or d is None:
            continue
        t = float(e["latency_p50_ms"])
        names.append("A10\n%s" % split); draft.append(d); rest.append(max(t - d, 0))
        tot.append(t); srcs.append(src)
    x = range(len(names))
    axes[0].bar(x, draft, .55, color="#4C72B0", label="classifier drafter (p50)")
    axes[0].bar(x, rest, .55, bottom=draft, color="#DD8452",
                label="verification + decode (total p50 - drafter p50)")
    for i in x:
        axes[0].text(i, tot[i] * 1.02, "total p50 %.1f ms\ndrafter %.1f ms (%.0f%%)"
                     % (tot[i], draft[i], 100 * draft[i] / tot[i]),
                     ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(list(x)); axes[0].set_xticklabels(names, fontsize=8)
    axes[0].set_ylim(0, max(tot + [1]) * 1.4)
    axes[0].set_ylabel("latency (ms)")
    axes[0].set_title("A10 p50 latency -- source: Stage 3 (unified_efficiency.csv\n"
                      "latency_p50_ms; drafter p50 from the same Stage-3 run's log)",
                      fontsize=9.5)
    axes[0].legend(fontsize=7.5, loc="upper left")

    sources = [("held_out_v3", C.REPO / "thesis2" / "out" / "a11_rejection_positions.csv"),
               ("held_out_v4c", C.OUT / "a11_rejection_positions_held_out_v4c.csv")]
    counts = {}
    for split, pth in sources:
        if pth.exists():
            with pth.open(encoding="utf-8", newline="") as fh:
                counts[split] = Counter(r["region"] for r in csv.DictReader(fh)
                                        if not r["region"].startswith("(none"))
    regions = sorted({k for c in counts.values() for k in c},
                     key=lambda k: -sum(c.get(k, 0) for c in counts.values()))
    w = 0.38
    for k, (split, col) in enumerate((("held_out_v3", "#4C72B0"),
                                      ("held_out_v4c", "#C44E52"))):
        if split not in counts:
            continue
        vals = [counts[split].get(r, 0) for r in regions]
        xs = [i + (k - 0.5) * w for i in range(len(regions))]
        axes[1].bar(xs, vals, w, color=col,
                    label="%s (%d rejected of 217)" % (split, sum(vals)))
        for xx, v in zip(xs, vals):
            axes[1].text(xx, v, str(v), ha="center", va="bottom", fontsize=8)
    axes[1].set_xticks(range(len(regions)))
    axes[1].set_xticklabels(regions, rotation=15)
    top = max([v for c in counts.values() for v in c.values()] + [1])
    axes[1].set_ylim(0, top * 1.18)
    axes[1].set_ylabel("commands")
    axes[1].set_title("first-rejection field of the classifier draft (A11 method)\n"
                      "v3: a11_rejection_positions.csv (earlier run); v4c: A11 --split "
                      "held_out_v4c (final run)", fontsize=9.5)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGS / fname, dpi=170)
    plt.close(fig)
    return fname


def main():
    C.start_log("s4_figures")
    metrics = load("metrics_final.csv") or load("metrics_stage2.csv")
    v5 = load("v5cells_both_splits.csv")
    if not metrics:
        C.gate(False, ["no metrics_final.csv or metrics_stage2.csv -- run Stage 2 first"])
    print("  %d metric rows, %d V5 cells" % (len(metrics), len(v5)))
    num = numbering(metrics, v5)
    made = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        made.append(fig_scatter(metrics, v5, num, "lat", "fig1_action_vs_latency.png"))
        made.append(fig_scatter(metrics, v5, num, "j", "fig2_action_vs_energy.png"))
        made.append(fig_wordgroup(metrics, "fig3_row_by_wordgroup.png"))
        made.append(fig_confusion(metrics, "fig4_row_confusion.png"))
        made.append(fig_a10("fig5_a10_breakdown.png"))
        made.append(fig_scatter(metrics, v5, num, "vram", "fig6_action_vs_vram.png"))
    legend_warn = [w for w in caught if "No artists with labels" in str(w.message)]
    other = [str(w.message)[:100] for w in caught if w not in legend_warn]
    C.gate(not legend_warn, ["%d figures in %s" % (len(made), C.FIGS.relative_to(C.REPO))]
           + ["  %s" % m for m in made]
           + ["'No artists with labels' warnings: %d (must be 0)" % len(legend_warn),
              "other warnings: %d %s" % (len(other), other[:3]),
              "numbered markers + key table; oracle = ceiling line; rules_open = "
              "vocabulary-unfair; both excluded from every Pareto frontier",
              "no claimable marking: claimability belongs to comparisons"])


if __name__ == "__main__":
    main()
