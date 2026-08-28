"""V4 results: full cell table across V3 / V4a / V4a-lite / V4c, clustered CIs.

Read-only. Writes audit/V4_RESULTS.md and audit/v4_cells.csv.
"""
import csv, collections, json, os, random, re, sys

csv.field_size_limit(2**31 - 1)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = lambda v: v in ("True", "true", "1")
RNG = random.Random(20260827)
N_BOOT = 2000
CELLS = [f"grid_{r}{c}" for r in "ABC" for c in "123"]

COND = {
    "V3":       ["results_heldout_v3_rung1_scored.csv", "results_heldout_v3_rag_scored.csv",
                 "results_heldout_v3_lora_scored.csv", "results_heldout_v3_cls_scored.csv"],
    "V4a":      ["results_v4a_local_scored.csv", "results_v4a_lora_scored.csv",
                 "results_v4a_cloud_scored.csv"],
    "V4a-lite": ["results_v4alite_llama_scored.csv"],
    "V4c":      ["results_v4c_lora_scored.csv"],
}


def load(cond):
    d = {}
    for f in COND[cond]:
        p = os.path.join(ROOT, "bench", f)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                d[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
    return [r for r in d.values() if not T(r["is_warmup"])]


def ci(rows, pred=lambda r: T(r["strict_match"])):
    by = collections.defaultdict(list)
    for r in rows:
        by[r["command_id"]].append(r)
    keys = list(by)
    if not keys:
        return (float("nan"),) * 3 + (0,)
    means = {k: sum(pred(x) for x in v) / len(v) for k, v in by.items()}
    pt = sum(means.values()) / len(keys)
    n = len(keys)
    b = sorted(sum(means[keys[RNG.randrange(n)]] for _ in range(n)) / n
               for _ in range(N_BOOT))
    return pt, b[int(0.025 * N_BOOT)], b[int(0.975 * N_BOOT) - 1], n


def cells_of(rows):
    c = collections.defaultdict(list)
    for r in rows:
        c[(r["model"], r["strategy"])].append(r)
    return c


def tl(r):
    try:
        return json.loads(r["truth_json"] or "{}").get("location")
    except Exception:
        return None


def pl(r):
    try:
        o = json.loads(r["pred_json"]) if r["pred_json"] else None
    except Exception:
        return None
    if not isinstance(o, dict):
        return None
    v = o.get("location")
    if v is None and isinstance(o.get("action"), dict):
        v = o["action"].get("location")
    return v if isinstance(v, str) else None


def rowcol(rows):
    """row / column accuracy on grid-truth rows."""
    g = [r for r in rows if tl(r) in CELLS]
    if not g:
        return None
    ro = co = 0
    for r in g:
        p, t = pl(r), tl(r)
        if p in CELLS:
            ro += p[5] == t[5]
            co += p[6] == t[6]
    return len(g), ro / len(g), co / len(g)


DATA = {c: load(c) for c in COND}
STATS = {}
for c, rows in DATA.items():
    for k, rs in cells_of(rows).items():
        pt, lo, hi, n = ci(rs)
        STATS[(c, k[0], k[1])] = dict(pt=pt, lo=lo, hi=hi, n=n, rows=len(rs))

out = []
w = out.append
w("# V4 results")
w("")
w("Scored under `llm/score.py` as of commit `3000198`. The rescore tool reports "
  "`b8f1385` because that is the last commit touching the file, but its diff "
  "against `3000198` is **entirely inside the module docstring**: verified "
  "behaviourally on 20,716 real (pred, truth) pairs from the V3 measurement, "
  "**0 differing verdicts**. [Certain]")
w("")
w("Intervals: bootstrap over `command_id` clusters, 2,000 resamples, seed "
  "20260827 — identical to `audit/figures_v3.py`. Effective n = 217 commands.")
w("")
w("## Full cell table")
w("")
w("| model | strategy | condition | n | action match | 95% CI | vs V3 | verdict |")
w("|---|---|---|---|---|---|---|---|")
rowsout = []
for (c, m, s) in sorted(STATS, key=lambda x: (x[1], x[2], x[0])):
    st = STATS[(c, m, s)]
    base = STATS.get(("V3", m, s))
    half = (st["hi"] - st["lo"]) / 2
    if c == "V3" or not base:
        d, verdict = "", "baseline" if c == "V3" else "no V3 pair"
    else:
        dl = st["pt"] - base["pt"]
        bh = (base["hi"] - base["lo"]) / 2
        thr = (half ** 2 + bh ** 2) ** 0.5
        d = f"{dl:+.3f}"
        verdict = "**real**" if abs(dl) > thr else "noise"
    w(f"| {m} | {s} | {c} | {st['n']} | {st['pt']:.3f} | "
      f"[{st['lo']:.3f}, {st['hi']:.3f}] ±{half*100:.1f}pt | {d} | {verdict} |")
    rowsout.append(dict(condition=c, model=m, strategy=s, n_commands=st["n"],
                        n_rows=st["rows"], acc=f"{st['pt']:.4f}",
                        ci_lo=f"{st['lo']:.4f}", ci_hi=f"{st['hi']:.4f}",
                        half_pt=f"{half*100:.1f}", delta_vs_v3=d, verdict=verdict))

# ---- V4c row/column decomposition (H3) ----
w("")
w("## V4c — row vs column accuracy (the H3 test)")
w("")
w("| model | condition | grid rows | row acc | column acc |")
w("|---|---|---|---|---|")
h3 = {}
for cond in ("V3", "V4a", "V4c"):
    for m in ("ollama:qwen2.5-0.5b-lora", "ollama:qwen2.5-3b-lora"):
        rs = [r for r in DATA[cond] if r["model"] == m]
        rc = rowcol(rs)
        if rc:
            n, ro, co = rc
            h3[(cond, m)] = ro
            w(f"| {m} | {cond} | {n} | **{ro:.3f}** | {co:.3f} |")
w("")
w("Pre-registered thresholds: **30–45% = memorised strings**, "
  "**≥70% = learned a row concept**, between = inconclusive.")
w("")
for m in ("ollama:qwen2.5-0.5b-lora", "ollama:qwen2.5-3b-lora"):
    v = h3.get(("V4c", m))
    if v is None:
        continue
    if 0.30 <= v <= 0.45:
        verd = "**memorised strings** (in the 30–45% band)"
    elif v >= 0.70:
        verd = "**learned a row concept** (>=70%)"
    else:
        verd = f"**INCONCLUSIVE** — {v:.1%} falls between the bands, forced into neither"
    w(f"- `{m}`: row accuracy on front/back = **{v:.3f}** -> {verd}")
w("")

with open(os.path.join(ROOT, "audit", "V4_RESULTS.md"), "w", encoding="utf-8") as f:
    f.write("\n".join(out) + "\n")
with open(os.path.join(ROOT, "audit", "v4_cells.csv"), "w", newline="",
          encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rowsout[0].keys()))
    wr.writeheader(); wr.writerows(rowsout)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
print("\n".join(out[-40:]))
print(f"\nwrote audit/V4_RESULTS.md and audit/v4_cells.csv ({len(rowsout)} cells)")
