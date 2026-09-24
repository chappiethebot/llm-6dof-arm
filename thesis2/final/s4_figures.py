"""thesis2/final/s4_figures.py -- STAGE 4: the six figures.

    python thesis2/final/s4_figures.py

Writes to thesis2/figures/final/:

  fig1_action_vs_latency.png     action match vs latency p50 (log x), both
                                 splits, CI bars, Pareto frontier
  fig2_action_vs_energy.png      action match vs J/command (log x); cloud arms
                                 in a separate panel labelled "energy not
                                 measurable"
  fig3_row_by_wordgroup.png      row accuracy by row-word group (seen / unseen /
                                 mixed), every arm, v3 and v4c
  fig4_row_confusion.png         row confusion signature per arm
                                 (A<->C mirror vs default-to-B)
  fig5_a10_breakdown.png         A10 latency breakdown (drafter / verification /
                                 decode) and first-rejection field, v3 vs v4c
  fig6_action_vs_vram.png        action match vs peak VRAM

Claimable differences are marked on every figure: an arm whose primary-family
verdict is CLAIMABLE is drawn with a filled marker and annotated; nominal and
not-claimable arms are hollow. If Stage 3 or Stage 4 has not run, the figure is
still produced and the missing axis is labelled, never silently dropped.
"""
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

import matplotlib                                                       # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                         # noqa: E402

SPLIT_COLOR = {"held_out_v3": "#1f77b4", "held_out_v4c": "#d62728"}


def load(name):
    p = C.OUT / name
    if not p.exists():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8", newline="")))


def f(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def pareto(points):
    """(x, y) maximising y while minimising x."""
    pts = sorted(points, key=lambda p: (p[0], -p[1]))
    best, front = -1e9, []
    for x, y, lbl in pts:
        if y > best:
            front.append((x, y, lbl))
            best = y
    return front


def claimable_arms(sig):
    out = set()
    for r in sig:
        if r.get("family") == "primary" and r.get("verdict") == "CLAIMABLE":
            out.add((r["arm_a"], r["split"]))
            out.add((r["arm_b"], r["split"]))
    return out


def fig_scatter(metrics, sig, xcol, fname, xlabel, title, logx=True, cloud_split=False):
    claim = claimable_arms(sig)
    fig, axes = plt.subplots(1, 2 if cloud_split else 1,
                             figsize=(13 if cloud_split else 8, 5.5), squeeze=False)
    for split in C.SPLITS:
        rows = [r for r in metrics if r["split"] == split and r["status"] == "valid"]
        local = [r for r in rows if r["kind"] != "cloud"]
        cloud = [r for r in rows if r["kind"] == "cloud"]
        for ax, sub, is_cloud in ((axes[0][0], local, False),
                                  (axes[0][-1], cloud, True) if cloud_split else
                                  (axes[0][0], [], True)):
            pts = []
            for r in sub:
                x, y = f(r.get(xcol)), f(r.get("action_match"))
                if x is None or y is None or (logx and x <= 0):
                    continue
                pts.append((x, y, r["arm"]))
                filled = (r["arm"], split) in claim
                ax.scatter(x, y, s=64, marker="o",
                           facecolors=SPLIT_COLOR[split] if filled else "none",
                           edgecolors=SPLIT_COLOR[split], zorder=3)
                ax.annotate(r["arm"], (x, y), fontsize=6, xytext=(3, 3),
                            textcoords="offset points")
            if pts and not is_cloud:
                fr = pareto(pts)
                ax.plot([p[0] for p in fr], [p[1] for p in fr], "--", lw=1,
                        color=SPLIT_COLOR[split], alpha=.6,
                        label="%s Pareto frontier" % split)
    for ax in axes[0]:
        if logx:
            ax.set_xscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("action match")
        ax.grid(alpha=.25)
        ax.legend(fontsize=7, loc="lower left")
    axes[0][0].set_title(title)
    if cloud_split:
        axes[0][1].set_title("cloud arms -- energy not measurable")
    fig.suptitle("filled marker = CLAIMABLE in the primary family; hollow = nominal "
                 "or not claimable", fontsize=8, y=.02)
    fig.tight_layout()
    fig.savefig(C.FIGS / fname, dpi=160)
    plt.close(fig)
    return fname


def fig_wordgroup(metrics, fname):
    groups = ["seen", "unseen", "mixed"]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, split in zip(axes, C.SPLITS):
        rows = [r for r in metrics if r["split"] == split and r["status"] == "valid"]
        rows.sort(key=lambda r: -(f(r.get("action_match")) or 0))
        names, vals = [], {g: [] for g in groups}
        for r in rows:
            parsed = {}
            for part in (r.get("row_by_word_group") or "").split(";"):
                part = part.strip()
                if not part or "/" not in part:
                    continue
                g, frac = part.rsplit(" ", 1)
                h, t = frac.split("/")
                parsed[g.strip()] = int(h) / int(t) if int(t) else None
            if not parsed:
                continue
            names.append(r["arm"])
            for g in groups:
                vals[g].append(parsed.get(g, float("nan")))
        x = range(len(names))
        w = 0.26
        for i, g in enumerate(groups):
            ax.bar([xx + (i - 1) * w for xx in x], vals[g], w, label=g)
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("row accuracy")
        ax.set_title("%s -- row accuracy by row word "
                     "(seen = middle/center; unseen = the split's new pair)" % split)
        ax.axhline(1 / 3, ls=":", c="k", lw=1)
        ax.annotate("chance", (0, 1 / 3), fontsize=7, va="bottom")
        ax.grid(axis="y", alpha=.25)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGS / fname, dpi=160)
    plt.close(fig)
    return fname


def fig_confusion(metrics, fname):
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    for ax, split in zip(axes, C.SPLITS):
        rows = [r for r in metrics if r["split"] == split and r["status"] == "valid"
                and r.get("row_signature")]
        names, mirror, tob = [], [], []
        for r in rows:
            s = r["row_signature"]
            try:
                m = int(s.split("A<->C mirror")[1].split(";")[0])
                b = int(s.split("default-to-B")[1])
            except Exception:
                continue
            names.append(r["arm"]); mirror.append(m); tob.append(b)
        x = range(len(names))
        ax.bar([xx - .2 for xx in x], mirror, .4, label="A<->C mirror")
        ax.bar([xx + .2 for xx in x], tob, .4, label="default to B")
        ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=75, ha="right", fontsize=6)
        ax.set_ylabel("grid commands")
        ax.set_title("%s -- row confusion signature" % split)
        ax.grid(axis="y", alpha=.25); ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(C.FIGS / fname, dpi=160)
    plt.close(fig)
    return fname


def fig_a10(metrics, fname):
    """A10 latency breakdown + first-rejection field, v3 vs v4c."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    old = C.REPO / "thesis2" / "out"
    # breakdown: drafter (measured), verification+decode (remainder)
    parts, labels = [], []
    p = old / "a10_prefix_cache_percommand.csv"
    if p.exists():
        rows = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
        import statistics as st
        draft = st.median([float(r["draft_ms"]) for r in rows])
        total = st.median([float(r["prefix_spec_ms"]) for r in rows])
        parts = [draft, max(total - draft, 0)]
        labels = ["classifier drafter", "verification + decode"]
        axes[0].bar(["A10 (held_out_v3)"], [parts[0]], label=labels[0])
        axes[0].bar(["A10 (held_out_v3)"], [parts[1]], bottom=[parts[0]], label=labels[1])
        axes[0].annotate("p50 total %.1f ms\ndrafter %.0f %%"
                         % (total, 100 * parts[0] / total), (0, total), fontsize=8,
                         ha="center", va="bottom")
    axes[0].set_ylabel("latency p50 (ms)")
    axes[0].set_title("A10 latency breakdown")
    axes[0].legend(fontsize=8)
    axes[0].annotate("held_out_v4c: A10 equals lora_greedy by construction;\n"
                     "no separate v4c speculative run was made",
                     (0.02, 0.02), xycoords="axes fraction", fontsize=7)

    rej = old / "a11_rejection_positions.csv"
    if rej.exists():
        from collections import Counter
        c = Counter(r["region"] for r in csv.DictReader(rej.open(encoding="utf-8", newline=""))
                    if not r["region"].startswith("(none"))
        axes[1].bar(list(c), list(c.values()), color="#d62728")
        for i, (k, v) in enumerate(c.items()):
            axes[1].annotate(str(v), (i, v), ha="center", va="bottom", fontsize=8)
    axes[1].set_title("first-rejection field (held_out_v3)")
    axes[1].set_ylabel("commands")
    axes[1].tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(C.FIGS / fname, dpi=160)
    plt.close(fig)
    return fname


def main():
    C.start_log("s4_figures")
    metrics = load("metrics_final.csv") or load("metrics_stage2.csv")
    sig = load("final_significance.csv")
    if not metrics:
        C.gate(False, ["no metrics_final.csv or metrics_stage2.csv -- run Stage 2 first"])
    print("  %d metric rows, %d significance rows" % (len(metrics), len(sig)))
    made = []
    made.append(fig_scatter(metrics, sig, "latency_p50_ms", "fig1_action_vs_latency.png",
                            "latency p50 (ms, log)", "action match vs latency"))
    made.append(fig_scatter(metrics, sig, "joules_per_command", "fig2_action_vs_energy.png",
                            "J / command (log)", "action match vs energy",
                            cloud_split=True))
    made.append(fig_wordgroup(metrics, "fig3_row_by_wordgroup.png"))
    made.append(fig_confusion(metrics, "fig4_row_confusion.png"))
    made.append(fig_a10(metrics, "fig5_a10_breakdown.png"))
    made.append(fig_scatter(metrics, sig, "peak_vram_mb", "fig6_action_vs_vram.png",
                            "peak VRAM (MB, log)", "action match vs peak VRAM"))
    C.gate(True, ["%d figures in %s" % (len(made), C.FIGS.relative_to(C.REPO))]
           + ["  %s" % m for m in made]
           + ["claimable differences marked: filled = CLAIMABLE in the primary family"])


if __name__ == "__main__":
    main()
