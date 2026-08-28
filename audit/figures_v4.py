"""V4 condition figures, same style as audit/figures_v3.py. Read-only."""
import csv, collections, os, random
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

csv.field_size_limit(2**31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "audit", "figures")
T = lambda v: v in ("True", "true", "1")
RNG = random.Random(20260827)
NB = 2000

COND_STYLE = {"V3": ("o", "#4C72B0"), "V4a": ("s", "#DD8452"),
              "V4a-lite": ("^", "#55A868"), "V4c": ("D", "#C44E52")}
FILES = {
    "V3": ["results_heldout_v3_rung1_scored.csv", "results_heldout_v3_rag_scored.csv",
           "results_heldout_v3_lora_scored.csv"],
    "V4a": ["results_v4a_local_scored.csv", "results_v4a_lora_scored.csv",
            "results_v4a_cloud_scored.csv"],
    "V4a-lite": ["results_v4alite_llama_scored.csv"],
    "V4c": ["results_v4c_lora_scored.csv"],
}


def load(c):
    d = {}
    for f in FILES[c]:
        p = os.path.join(ROOT, "bench", f)
        if os.path.exists(p):
            for r in csv.DictReader(open(p, encoding="utf-8", newline="")):
                d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
    return [r for r in d.values() if not T(r["is_warmup"])]


def ci(rows):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["command_id"]].append(r)
    k = list(by)
    m = {x: sum(T(r["strict_match"]) for r in by[x]) / len(by[x]) for x in k}
    pt = sum(m.values()) / len(k)
    n = len(k)
    b = sorted(sum(m[k[RNG.randrange(n)]] for _ in range(n)) / n for _ in range(NB))
    return pt, b[int(.025 * NB)], b[int(.975 * NB) - 1]


def short(m):
    return m.replace("ollama:", "").replace("openai:", "").replace("classifier:", "cls:")


D = {c: load(c) for c in FILES}
S = {}
for c, rows in D.items():
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["model"], r["strategy"])].append(r)
    for k, rs in g.items():
        S[(c,) + k] = ci(rs)


def save(fig, name, cap):
    os.makedirs(OUT, exist_ok=True)
    words, lines, cur = cap.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > 120:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    lines.append(cur)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    y0 = min(a.get_tightbbox(rend).transformed(fig.transFigure.inverted()).y0
             for a in fig.get_axes())
    fig.text(0.5, y0 - 0.03, "\n".join(lines), ha="center", va="top",
             fontsize=8.5, style="italic", color="#333333")
    for e in ("png", "pdf"):
        fig.savefig(os.path.join(OUT, f"{name}.{e}"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote audit/figures/{name}.png / .pdf")


# ---- fig 5: V3 vs V4a per cell, paired ----
pairs = sorted({(m, s) for (c, m, s) in S if c == "V4a" and ("V3", m, s) in S},
               key=lambda k: S[("V3",) + k][0])
fig, ax = plt.subplots(figsize=(10, 9))
for i, k in enumerate(pairs):
    for c in ("V3", "V4a"):
        pt, lo, hi = S[(c,) + k]
        mk, col = COND_STYLE[c]
        ax.errorbar(pt, i + (0.16 if c == "V4a" else -0.16),
                    xerr=[[pt - lo], [hi - pt]], fmt=mk, color=col, ms=5,
                    capsize=2.5, lw=1.2)
ax.set_yticks(range(len(pairs)))
ax.set_yticklabels([f"{short(m)} / {s}" for m, s in pairs], fontsize=8)
ax.set_xlabel("action match: verb, target and location all correct\n"
              "(95% CI, bootstrapped over commands)")
ax.set_title("Original legend vs revised legend, 27 paired cells\n"
             "217 unseen commands; overlapping intervals mean no real difference",
             fontsize=11)
ax.grid(axis="x", alpha=.3); ax.set_xlim(0, 1)
ax.legend(handles=[Line2D([], [], marker=COND_STYLE[c][0], color=COND_STYLE[c][1],
                          ls="", label=c) for c in ("V3", "V4a")],
          loc="lower right", fontsize=9)
save(fig, "fig5_v3_vs_v4a",
     "Every model and method run under both legends. Takeaway: the revised legend changes "
     "almost nothing overall - 21 of 27 cells are inside the error bars - but the six that "
     "move do so in both directions, and the largest model regresses alongside the smallest.")

# ---- fig 6: llama3.2 failure-mode decomposition ----
import json
def rates(cond, m, s):
    rows = [r for r in D[cond] if r["model"] == m and r["strategy"] == s]
    if not rows:
        return None
    mod = b2 = 0
    for r in rows:
        try:
            o = json.loads(r["pred_json"]) if r["pred_json"] else {}
        except Exception:
            o = {}
        if isinstance(o, dict):
            mod += bool(o.get("modifiers"))
            b2 += o.get("location") == "grid_B2"
    return 100 * mod / len(rows), 100 * b2 / len(rows)


fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
conds = ["V3", "V4a", "V4a-lite"]
for ax, s in zip(axes, ("few_shot", "rag")):
    v = [rates(c, "ollama:llama3.2:3b", s) for c in conds]
    x = range(len(conds))
    ax.bar([i - .2 for i in x], [q[0] for q in v], .4,
           label="spurious modifiers emitted", color="#DD8452")
    ax.bar([i + .2 for i in x], [q[1] for q in v], .4,
           label="predicted grid_B2", color="#4C72B0")
    for i, q in enumerate(v):
        ax.text(i - .2, q[0] + 1, f"{q[0]:.0f}%", ha="center", fontsize=8)
        ax.text(i + .2, q[1] + 1, f"{q[1]:.0f}%", ha="center", fontsize=8)
    ax.set_xticks(list(x), conds); ax.set_title(f"llama3.2:3b / {s}", fontsize=11)
    ax.set_ylabel("% of that cell's rows"); ax.grid(axis="y", alpha=.3)
    ax.set_ylim(0, 70)
axes[0].legend(fontsize=8)
fig.suptitle("Two failure modes with different causes", fontsize=13)
save(fig, "fig6_llama_failure_modes",
     "Left bars: how often the model emitted a spurious modifier (a correct location still "
     "scored wrong). Right bars: how often it answered the centre square. Takeaway: the plain-prose "
     "legend fully removes the centre-square anchoring but only halves the modifier problem, so the "
     "two have different causes.")

# ---- fig 7: V4c row vs column ----
CELLS = [f"grid_{r}{c}" for r in "ABC" for c in "123"]
def rc(cond, m):
    rows = [r for r in D[cond] if r["model"] == m]
    g = []
    for r in rows:
        try:
            t = json.loads(r["truth_json"] or "{}").get("location")
            o = json.loads(r["pred_json"]) if r["pred_json"] else {}
            p = o.get("location") if isinstance(o, dict) else None
        except Exception:
            continue
        if t in CELLS:
            g.append((t, p))
    if not g:
        return None
    return (100 * sum(p in CELLS and p[5] == t[5] for t, p in g) / len(g),
            100 * sum(p in CELLS and p[6] == t[6] for t, p in g) / len(g))


fig, ax = plt.subplots(figsize=(9, 4.8))
models = ["ollama:qwen2.5-0.5b-lora", "ollama:qwen2.5-3b-lora"]
labels, rows_, cols_ = [], [], []
for m in models:
    for c in ("V3", "V4c"):
        v = rc(c, m)
        if v:
            labels.append(f"{short(m)}\n{c}"); rows_.append(v[0]); cols_.append(v[1])
x = range(len(labels))
ax.bar([i - .2 for i in x], rows_, .4, label="row correct", color="#C44E52")
ax.bar([i + .2 for i in x], cols_, .4, label="column correct", color="#4C72B0")
ax.axhspan(30, 45, color="grey", alpha=.15)
ax.axhline(70, color="k", ls="--", lw=1)
ax.text(len(labels) - .5, 71, "pre-registered: learned a concept", fontsize=8, ha="right")
ax.text(len(labels) - .5, 37, "pre-registered: memorised strings", fontsize=8, ha="right")
ax.axhline(33.3, color="grey", ls=":", lw=1)
for i, (r_, c_) in enumerate(zip(rows_, cols_)):
    ax.text(i - .2, r_ + 1, f"{r_:.0f}", ha="center", fontsize=8)
    ax.text(i + .2, c_ + 1, f"{c_:.0f}", ha="center", fontsize=8)
ax.set_xticks(list(x), labels, fontsize=8)
ax.set_ylabel("% correct on that axis"); ax.set_ylim(0, 105)
ax.grid(axis="y", alpha=.3); ax.legend(fontsize=9, loc="upper left")
ax.set_title("Fine-tuned models on row vocabulary they were never trained on\n"
             "dotted line = chance on one axis", fontsize=11)
save(fig, "fig7_v4c_row_vs_column",
     "The two fine-tuned models tested on front/back, wording present in neither their training "
     "data nor the original evaluation set. Takeaway: columns stay near ceiling while rows sit at "
     "or near chance, so the adapters read the side-to-side axis and guess the front-to-back one.")
print("done")
