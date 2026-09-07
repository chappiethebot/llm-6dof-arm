"""audit/figures_v4_compare.py -- figures for audit/V3_VS_V4.md

Reads audit/v4_cells.csv ONLY. Runs no models, touches no sealed data, and
re-derives nothing: every accuracy and interval plotted is copied from that
file as measured under scorer commit 3000198.

    python audit/figures_v4_compare.py

Writes audit/figures/fig_v4_{1,2,3}_*.{png,pdf}

NOISE BAND. Every panel shades +/- NOISE_PTS points. That band is the median
bootstrap CI half-width across all cells in v4_cells.csv (2,000 resamples over
command_id clusters, seed 20260827). A bar inside the band is not a result.
"""
import csv
import os
import collections

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CELLS = os.path.join(HERE, "v4_cells.csv")
OUTDIR = os.path.join(HERE, "figures")

NOISE_PTS = 6.0          # median CI half-width, rounded up from 5.8
C_UP, C_DOWN, C_NOISE = "#1b7837", "#b2182b", "#9e9e9e"


def load():
    rows = []
    with open(CELLS, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("acc", "ci_lo", "ci_hi", "half_pt"):
                r[k] = float(r[k])
            r["delta"] = float(r["delta_vs_v3"]) if r["delta_vs_v3"] else None
            r["n_commands"] = int(r["n_commands"])
            rows.append(r)
    return rows


def short(model):
    return model.split(":", 1)[1] if ":" in model else model


def fig1_delta(rows):
    """Every V4a cell as a change from its own V3 baseline."""
    cells = [r for r in rows if r["condition"] == "V4a"]
    cells.sort(key=lambda r: r["delta"])
    labels = ["%s / %s" % (short(r["model"]), r["strategy"]) for r in cells]
    deltas = [r["delta"] * 100 for r in cells]

    # Colour by the NOISE BAND, not by the csv verdict column. One cell
    # (qwen2.5:0.5b/few_shot, -5.5 pts) is flagged **real** in the csv but sits
    # inside the band; see V3_VS_V4.md section 8. The band is the published
    # rule, so the figure follows the band.
    colours = [C_NOISE if abs(d) < NOISE_PTS else (C_UP if d > 0 else C_DOWN)
               for d in deltas]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.axvspan(-NOISE_PTS, NOISE_PTS, color="#000000", alpha=0.06, zorder=0)
    ax.axvline(0, color="#444444", lw=1.0, zorder=1)
    ax.barh(range(len(cells)), deltas, color=colours, zorder=2)
    ax.set_yticks(range(len(cells)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("action-match change, V4a minus V3 (percentage points)")
    ax.set_title("The legend rewrite changed almost nothing\n"
                 "20 of 27 cells inside the $\\pm$%.0f-point noise band  "
                 "(all 217 commands)" % NOISE_PTS, fontsize=11)
    ax.text(0.015, 0.015,
            "shaded band = median bootstrap CI half-width (5.8 pt, rounded up)\n"
            "moves inside it are not claimable",
            transform=ax.transAxes, fontsize=8, color="#555555")
    ax.grid(axis="x", alpha=0.25, zorder=0)
    fig.tight_layout()
    return fig, "fig_v4_1_delta_vs_v3"


def _ci_panel(ax, series, title):
    """series: [(label, acc, lo, hi, colour)] -- accuracy with its interval."""
    xs = range(len(series))
    accs = [s[1] * 100 for s in series]
    lo = [(s[1] - s[2]) * 100 for s in series]
    hi = [(s[3] - s[1]) * 100 for s in series]
    ax.bar(xs, accs, color=[s[4] for s in series], width=0.6, zorder=2)
    ax.errorbar(xs, accs, yerr=[lo, hi], fmt="none", ecolor="#222222",
                capsize=4, lw=1.2, zorder=3)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([s[0] for s in series], fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", alpha=0.25, zorder=0)


def fig2_llama_lite(rows):
    """V4a-lite puts llama back inside V3's interval."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["model"], r["strategy"])][r["condition"]] = r

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    order = [("V3", "#4d4d4d"), ("V4a", C_DOWN), ("V4a-lite", "#2166ac")]
    for ax, strat in zip(axes, ("few_shot", "rag")):
        cell = by[("ollama:llama3.2:3b", strat)]
        series = [(c, cell[c]["acc"], cell[c]["ci_lo"], cell[c]["ci_hi"], col)
                  for c, col in order if c in cell]
        _ci_panel(ax, series, "llama3.2:3b / %s" % strat)
    axes[0].set_ylabel("action match (%), all 217")
    fig.suptitle("Bug 4, isolated: removing the literal cell id and the arrows\n"
                 "recovers llama3.2 to within V3's interval", fontsize=11)
    fig.tight_layout()
    return fig, "fig_v4_2_llama_v4a_lite"


def fig3_lora_v4c(rows):
    """V4c: row words present in NEITHER split."""
    by = collections.defaultdict(dict)
    for r in rows:
        by[(r["model"], r["strategy"])][r["condition"]] = r

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.6), sharey=True)
    order = [("V3", "#4d4d4d"), ("V4a", "#7f7f7f"), ("V4c", "#762a83")]
    for ax, model in zip(axes, ("ollama:qwen2.5-0.5b-lora",
                                "ollama:qwen2.5-3b-lora")):
        cell = by[(model, "zero_shot")]
        series = [(c, cell[c]["acc"], cell[c]["ci_lo"], cell[c]["ci_hi"], col)
                  for c, col in order if c in cell]
        _ci_panel(ax, series, "%s / zero_shot" % short(model))
    axes[0].set_ylabel("action match (%), all 217")
    fig.suptitle("V4c: row vocabulary in NEITHER split.\n"
                 "Overall accuracy moves by less than the noise band for both "
                 "adapters", fontsize=11)
    fig.tight_layout()
    return fig, "fig_v4_3_lora_v4c"


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rows = load()
    for builder in (fig1_delta, fig2_llama_lite, fig3_lora_v4c):
        fig, stem = builder(rows)
        for ext in ("png", "pdf"):
            path = os.path.join(OUTDIR, "%s.%s" % (stem, ext))
            fig.savefig(path, dpi=200, bbox_inches="tight")
            print("wrote %s" % path)
        plt.close(fig)


if __name__ == "__main__":
    main()
