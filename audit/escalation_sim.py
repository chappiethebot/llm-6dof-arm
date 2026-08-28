"""Step 9b: SIMULATE field-level escalation. Offline, from existing predictions.

Keep the classifier's verb and target. Escalate ONLY the location to
gpt-4o-mini when the location head's softmax margin < threshold.

Nothing is run live. gpt-4o-mini's location comes from its already-measured
V3 rag predictions (the current frontier top). Latency and cost are the measured
per-call values for each system.
"""
import csv, collections, json, os, random

csv.field_size_limit(2**31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = lambda v: v in ("True", "true", "1")
RNG = random.Random(20260827)
NB = 2000

# measured, from audit/figures/cell_stats.csv
CLS_LAT, GPT_LAT = 0.019, 0.804          # median seconds per command
# measured cloud cost: $0.0758 for the rag cell / 1085 rows
GPT_COST = 0.0758 / 1085


def clustered_ci(vals_by_cmd):
    k = list(vals_by_cmd)
    m = {x: sum(vals_by_cmd[x]) / len(vals_by_cmd[x]) for x in k}
    pt = sum(m.values()) / len(k)
    n = len(k)
    b = sorted(sum(m[k[RNG.randrange(n)]] for _ in range(n)) / n for _ in range(NB))
    return pt, b[int(.025 * NB)], b[int(.975 * NB) - 1]


margins = {int(r["command_id"]): r for r in
           csv.DictReader(open(os.path.join(ROOT, "audit", "cls_margins.csv"),
                               encoding="utf-8"))}

# gpt-4o-mini rag predictions, keyed by command_id (5 repeats -> take run 0)
gpt = {}
for r in csv.DictReader(open(os.path.join(ROOT, "bench",
                                          "results_heldout_v3_rag_scored.csv"),
                             encoding="utf-8", newline="")):
    if r["model"] == "openai:gpt-4o-mini" and r["run_index"] == "0":
        gpt[int(r["command_id"])] = r


def loc(js):
    try:
        o = json.loads(js) if js else None
    except Exception:
        return None
    if not isinstance(o, dict):
        return None
    v = o.get("location")
    if v is None and isinstance(o.get("action"), dict):
        v = o["action"].get("location")
    return v if isinstance(v, str) else None


def match(pred_obj, truth_obj):
    """Strict, mirroring llm.score: drop falsy modifiers on both sides."""
    d = lambda o: {k: v for k, v in o.items() if not (k == "modifiers" and not v)}
    return d(pred_obj) == d(truth_obj)


rows = []
for cid, m in margins.items():
    truth = json.loads(m["truth_json"])
    cls_obj = json.loads(m["pred_json"])
    g = gpt.get(cid)
    rows.append(dict(cid=cid, margin=float(m["location_margin"]),
                     cls_obj=cls_obj, truth=truth,
                     gpt_loc=loc(g["pred_json"]) if g else None))

THRESH = [0.0, 0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40,
          0.50, 0.60, 0.75, 1.01]
print(f"{'thresh':>7}{'escal%':>8}{'accuracy':>10}{'95% CI':>20}"
      f"{'mean lat s':>12}{'cost $/1k':>11}")
res = []
for th in THRESH:
    by = collections.defaultdict(list)
    esc = 0
    for r in rows:
        obj = dict(r["cls_obj"])
        if r["margin"] < th and "location" in obj and r["gpt_loc"]:
            obj["location"] = r["gpt_loc"]
            esc += 1
        by[r["cid"]].append(match(obj, r["truth"]))
    pt, lo, hi = clustered_ci(by)
    frac = esc / len(rows)
    lat = CLS_LAT + frac * GPT_LAT
    cost = frac * GPT_COST * 1000
    res.append((th, frac, pt, lo, hi, lat, cost))
    print(f"{th:>7.2f}{frac*100:>7.1f}%{pt:>10.3f}  [{lo:.3f}, {hi:.3f}]"
          f"{lat:>12.3f}{cost:>11.4f}")

# ---- does any point dominate the measured frontier? ----
FRONTIER = [("classifier", 0.747, 0.019), ("gpt-4o-mini/role_based", 0.855, 0.699),
            ("gpt-4o-mini/rag", 0.924, 0.804)]
print("\nDOMINANCE CHECK vs the measured Pareto frontier")
print("  a simulated point dominates if it is BOTH faster AND more accurate")
any_dom = False
for th, frac, pt, lo, hi, lat, cost in res:
    for name, facc, flat in FRONTIER:
        if lat < flat and pt > facc:
            print(f"  thresh {th:.2f}: acc {pt:.3f} lat {lat:.3f}s DOMINATES {name} "
                  f"({facc:.3f}, {flat:.3f}s)")
            any_dom = True
if not any_dom:
    print("  NONE. No threshold produces a point that dominates any frontier member.")

import json as _j
with open(os.path.join(ROOT, "audit", "escalation_sim.csv"), "w", newline="",
          encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["threshold", "escalated_frac", "accuracy", "ci_lo", "ci_hi",
                "mean_latency_s", "cost_usd_per_1000"])
    for r in res:
        w.writerow([f"{r[0]:.2f}", f"{r[1]:.4f}", f"{r[2]:.4f}", f"{r[3]:.4f}",
                    f"{r[4]:.4f}", f"{r[5]:.4f}", f"{r[6]:.4f}"])
print("\nwrote audit/escalation_sim.csv")

# ---- figure ----
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
for ax, xi, xlab, logx in ((axes[0], 5, "mean response time per command, seconds", True),
                           (axes[1], 6, "cloud cost, USD per 1000 commands", False)):
    xs = [r[xi] for r in res]; ys = [r[2] for r in res]
    ax.errorbar(xs, ys, yerr=[[r[2]-r[3] for r in res], [r[4]-r[2] for r in res]],
                fmt="o-", color="#8172B2", ms=5, capsize=2, lw=1.2,
                label="escalation sweep (simulated)")
    for r in res:
        if r[0] in (0.0, 0.10, 0.30, 1.01):
            ax.annotate(f"t={r[0]:.2f}", (r[xi], r[2]), fontsize=7,
                        xytext=(4, -10), textcoords="offset points")
    if xi == 5:
        for name, facc, flat in FRONTIER:
            ax.scatter([flat], [facc], marker="D", s=70, color="#C44E52",
                       edgecolor="black", zorder=5)
            ax.annotate(name, (flat, facc), fontsize=7.5, xytext=(6, 4),
                        textcoords="offset points")
        ax.plot([f[2] for f in FRONTIER], [f[1] for f in FRONTIER], "--",
                color="#C44E52", lw=1, label="measured Pareto frontier")
        ax.set_xscale("log")
    ax.set_xlabel(xlab); ax.set_ylabel("action match")
    ax.grid(alpha=.3, which="both"); ax.legend(fontsize=8, loc="lower right")
fig.suptitle("Simulated field-level escalation: keep the classifier's verb and target,\n"
             "send only low-confidence locations to the cloud model", fontsize=12)
cap = ("Each purple point is one confidence threshold; t=0 is the classifier alone, "
       "t=1.01 escalates every location. Simulation over already-measured predictions - "
       "nothing was re-run live. Takeaway: the sweep traces a smooth trade-off but no "
       "threshold beats the existing options on both speed and accuracy at once.")
words, lines, cur = cap.split(), [], ""
for w in words:
    if len(cur) + len(w) + 1 > 125:
        lines.append(cur); cur = w
    else:
        cur = f"{cur} {w}".strip()
lines.append(cur)
fig.canvas.draw()
y0 = min(a.get_tightbbox(fig.canvas.get_renderer())
         .transformed(fig.transFigure.inverted()).y0 for a in fig.get_axes())
fig.text(0.5, y0 - 0.05, "\n".join(lines), ha="center", va="top", fontsize=8.5,
         style="italic", color="#333")
for e in ("png", "pdf"):
    fig.savefig(os.path.join(ROOT, "audit", "figures", f"fig8_escalation_sim.{e}"),
                dpi=200, bbox_inches="tight")
print("wrote audit/figures/fig8_escalation_sim.png / .pdf")
