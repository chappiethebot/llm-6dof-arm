"""thesis2/scripts/a13_v4c_significance.py -- A13 significance: the two v4c
comparisons and the four cloud-cascade comparisons.

RECOVERED 2026-09-22 from the session scratchpad (`f1_v4c_stats.py` and
`f3_v4c_groups.py`), merged so that a13_v4c_significance.csv and
a13_v4c_cloud_significance.csv both have a generating script. Deterministic:
clustered bootstrap over command_id, 2,000 resamples, seed 20260827, exact
binomial McNemar -- the same protocol as A9.

Part 1 writes a13_v4c_significance.csv  (flat classifier vs qwen2.5-3b-lora V5)
Part 2 writes a13_v4c_cloud_significance.csv (cascade->gpt-4o-mini vs flat, at
        tau=1.0 and at the pre-registered tau_A=0.50) and prints the v4c row
        accuracy split by the row word the command uses.
Holm correction across the whole family is applied afterwards by
a16_holm_correction.py, which adds p_holm and claimable_corrected.

Run:  python thesis2/scripts/a13_v4c_significance.py
"""
import csv, json, math, re, sys, collections
import numpy as np
R = r"C:\Users\USER\llm-6dof-arm"
sys.path.insert(0, R)
from llm.score import score
csv.field_size_limit(2 ** 31 - 1)
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
sch = json.load(open(R + r"\schema\schema_v2.json"))
V = list(sch["verbs"]); T = sch["targets"]; REQ = {v: sch["verbs"][v]["required"] for v in V}
rowof = lambda l: l[5] if isinstance(l, str) and l.startswith("grid_") and len(l) == 7 else None
B, SEED = 2000, 20260827


def emit(v, t, l):
    o = {"action": v}
    if "target" in REQ[v]: o["target"] = t
    if "location" in REQ[v]: o["location"] = l
    o["modifiers"] = []
    return o


def mcnemar_exact(a, b):
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    n = bb + cc
    if n == 0: return bb, cc, 1.0
    k = min(bb, cc)
    return bb, cc, min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def clustered_bootstrap(a, b, seed=SEED, nb=B):
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    rng = np.random.default_rng(seed)
    d = np.empty(nb)
    for i in range(nb):
        idx = rng.integers(0, n, n)
        d[i] = a[idx].mean() - b[idx].mean()
    return a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


p45 = rd(R + r"\thesis2\out\p45_predictions_held_out_v4c.csv")
lora = rd(R + r"\thesis2\out\a13_v4c_lora_percommand.csv")
assert len(p45) == len(lora) == 217
for a, b in zip(p45, lora):
    assert a["command"] == b["command"]

flat_act, lora_act, flat_row, lora_row, ids_row = [], [], [], [], []
for a, b in zip(p45, lora):
    t = json.loads(a["truth_json"]); t.setdefault("modifiers", [])
    fo = emit(V[int(a["flat_verb_idx"])], T[int(a["flat_target_idx"])], a["flat_location"])
    flat_act.append(int(bool(score(fo, t).strict_match)))
    lora_act.append(int(b["strict_match"] == "True"))
    tr = rowof(t.get("location"))
    if tr:
        ids_row.append(a["command_id"])
        flat_row.append(int(rowof(fo.get("location")) == tr))
        lora_row.append(int(b["pred_row"] == tr))

print("=" * 104)
print("held_out_v4c -- flat DistilBERT classifier (A) vs qwen2.5-3b-lora V5 zero-shot (B)")
print("  clustered bootstrap over command_id, %d resamples, seed %d; McNemar exact binomial" % (B, SEED))
print("=" * 104)
print("  %-16s %5s %9s %9s %10s %20s %9s %9s %12s" %
      ("metric", "n", "A flat", "B lora", "diff pts", "95% CI (pts)", "b", "c", "McNemar p"))
rowsout = []
for name, a, b in (("action_match", flat_act, lora_act), ("row_accuracy", flat_row, lora_row)):
    pt, lo, hi = clustered_bootstrap(a, b)
    bb, cc, p = mcnemar_exact(a, b)
    sep = not (lo <= 0 <= hi)
    verdict = "CLAIMABLE" if (sep and p < 0.05) else "NOT CLAIMABLE"
    print("  %-16s %5d %9.4f %9.4f %+10.2f   [%+7.2f, %+7.2f] %9d %9d %12.3e   %s"
          % (name, len(a), np.mean(a), np.mean(b), 100 * pt, 100 * lo, 100 * hi, bb, cc, p, verdict))
    rowsout.append(dict(dataset="held_out_v4c", comparison="flat classifier vs qwen2.5-3b-lora V5 zero-shot",
                        arm_a="flat_distilbert", arm_b="qwen3b_lora_v5_prompted", metric=name,
                        n=len(a), acc_a=round(float(np.mean(a)), 4), acc_b=round(float(np.mean(b)), 4),
                        diff_pts=round(100 * pt, 2), ci_lo_pts=round(100 * lo, 2), ci_hi_pts=round(100 * hi, 2),
                        mcnemar_b=bb, mcnemar_c=cc, mcnemar_p=("%.3e" % p), ci_excludes_zero=sep,
                        verdict=verdict))
# seed robustness
print()
for name, a, b in (("action_match", flat_act, lora_act), ("row_accuracy", flat_row, lora_row)):
    cis = [clustered_bootstrap(a, b, seed=s)[1:] for s in (20260827, 1, 2, 3)]
    print("  %-14s CI across seeds: %s" % (name, "  ".join("[%+.2f,%+.2f]" % (100 * l, 100 * h) for l, h in cis)))

with open(R + r"\thesis2\out\a13_v4c_significance.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rowsout[0].keys())); w.writeheader(); w.writerows(rowsout)
print("\nwrote thesis2/out/a13_v4c_significance.csv")

# ------------------------------------------------------------- word counts
WORDS = ["back", "front", "middle", "center", "upper", "bottom", "top", "lower", "far", "near"]
print()
print("=" * 104)
print("ROW-WORD OCCURRENCES BY SPLIT  (whole-word, case-insensitive; commands containing the word / total occurrences)")
print("=" * 104)
splits = {}
for nm in ("train_v2", "dev_v2", "held_out_v3", "held_out_v4c"):
    rows = [json.loads(l) for l in open(R + r"\data\%s.jsonl" % nm, encoding="utf-8") if l.strip()]
    splits[nm] = rows
print("  %-10s %6s %s" % ("split", "n", " ".join("%10s" % w for w in WORDS)))
for nm, rows in splits.items():
    cnt = []
    for w in WORDS:
        c = sum(1 for r in rows if re.search(r"\b%s\b" % w, r["command"].lower()))
        o = sum(len(re.findall(r"\b%s\b" % w, r["command"].lower())) for r in rows)
        cnt.append("%4d/%-5d" % (c, o))
    print("  %-10s %6d %s" % (nm, len(rows), " ".join("%10s" % x for x in cnt)))
print()
print("  (cells are: commands containing the word / total occurrences)")
print()
print("  train_v2 + dev_v2 combined, words with ZERO occurrences:",
      [w for w in WORDS
       if not any(re.search(r"\b%s\b" % w, r["command"].lower())
                  for nm in ("train_v2", "dev_v2") for r in splits[nm])])


print()
print('#' * 104)
print('PART 2 -- row-word split and the cloud-cascade comparisons')
print('#' * 104)

import csv, json, math, re, sys, collections
import numpy as np
R = r"C:\Users\USER\llm-6dof-arm"
sys.path.insert(0, R)
from llm.score import score
csv.field_size_limit(2 ** 31 - 1)
rd = lambda p: list(csv.DictReader(open(p, encoding="utf-8", newline="")))
sch = json.load(open(R + r"\schema\schema_v2.json"))
V = list(sch["verbs"]); T = sch["targets"]; REQ = {v: sch["verbs"][v]["required"] for v in V}
ROWS = ["A", "B", "C"]; COLS = ["1", "2", "3"]
CONT = ["box_1", "box_2", "box_3"]; RT = ["max_reach", "nearest_free_cell"]
rowof = lambda l: l[5] if isinstance(l, str) and l.startswith("grid_") and len(l) == 7 else None
B, SEED = 2000, 20260827


def emit(v, t, l):
    o = {"action": v}
    if "target" in REQ[v]: o["target"] = t
    if "location" in REQ[v]: o["location"] = l
    o["modifiers"] = []
    return o


def mcn(a, b):
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    n = bb + cc
    if n == 0: return bb, cc, 1.0
    k = min(bb, cc)
    return bb, cc, min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def boot(a, b, seed=SEED, nb=B):
    a = np.asarray(a, float); b = np.asarray(b, float); n = len(a)
    rng = np.random.default_rng(seed)
    d = np.array([a[i].mean() - b[i].mean() for i in (rng.integers(0, n, n) for _ in range(nb))])
    return a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


p45 = rd(R + r"\thesis2\out\p45_predictions_held_out_v4c.csv")
lora = rd(R + r"\thesis2\out\a13_v4c_lora_percommand.csv")
for a, b in zip(p45, lora):
    assert a["command"] == b["command"]

# ---------------------------------------------------------------- (1) groups
def group(cmd):
    low = cmd.lower()
    bf = bool(re.search(r"\bback\b|\bfront\b", low))
    mid = bool(re.search(r"\bmiddle\b", low))
    cen = bool(re.search(r"\bcenter\b", low))
    if bf and not (mid or cen): return "back/front only"
    if bf and (mid or cen): return "back/front + middle/center"
    if mid and cen: return "middle+center"
    if mid: return "middle only"
    if cen: return "center only"
    return "no row word"


print("=" * 100)
print("(1) held_out_v4c ROW ACCURACY BY THE ROW WORD THE COMMAND USES (grid commands only, n=108)")
print("=" * 100)
recs = []
for a, b in zip(p45, lora):
    t = json.loads(a["truth_json"]); t.setdefault("modifiers", [])
    tr = rowof(t.get("location"))
    if not tr: continue
    fo = emit(V[int(a["flat_verb_idx"])], T[int(a["flat_target_idx"])], a["flat_location"])
    recs.append(dict(cmd=a["command"], g=group(a["command"]), truth_row=tr,
                     flat_row=rowof(fo.get("location")), lora_row=b["pred_row"] or None,
                     cloud_row=a["cloud_row"], fact_row=ROWS[int(a["fact_row_idx"])]))
g = collections.Counter(r["g"] for r in recs)
print("  group sizes: %s   (total %d)" % (dict(g), len(recs)))
print()
print("  %-28s %5s %18s %18s" % ("group", "n", "flat correct", "LoRA-V5 correct"))
for name in sorted(g):
    sub = [r for r in recs if r["g"] == name]
    fc = sum(r["flat_row"] == r["truth_row"] for r in sub)
    lc = sum(r["lora_row"] == r["truth_row"] for r in sub)
    print("  %-28s %5d %10d (%.4f) %10d (%.4f)"
          % (name, len(sub), fc, fc / len(sub), lc, lc / len(sub)))
print()
print("  by TRUTH ROW:")
print("  %-10s %5s %18s %18s" % ("truth row", "n", "flat correct", "LoRA correct"))
for tr in ROWS:
    sub = [r for r in recs if r["truth_row"] == tr]
    fc = sum(r["flat_row"] == tr for r in sub); lc = sum(r["lora_row"] == tr for r in sub)
    print("  %-10s %5d %10d (%.4f) %10d (%.4f)" % (tr, len(sub), fc, fc / len(sub), lc, lc / len(sub)))
print()
print("  flat row confusion (truth -> pred): %s"
      % dict(sorted(collections.Counter((r["truth_row"], r["flat_row"] or "-") for r in recs).items())))
print("  LoRA row confusion (truth -> pred): %s"
      % dict(sorted(collections.Counter((r["truth_row"], r["lora_row"] or "-") for r in recs).items())))

# ------------------------------------------------- (2) cascade-to-cloud arm
print()
print("=" * 100)
print("(2) WHICH TAU IS 'cascade-to-cloud 0.7788 / 0.6759'?")
print("=" * 100)
cf = rd(R + r"\thesis2\out\p45_cascade_full.csv")
for tau in (0.5, 1.0):
    r = [x for x in cf if x["dataset"] == "held_out_v4c" and x["signal"] == "factorized_row_conf"
         and x["target"] == "cloud" and abs(float(x["tau"]) - tau) < 1e-9][0]
    print("  tau=%.2f  escalation_rate %s (%s of 217)  action %s  row %s"
          % (tau, r["escalation_rate"], r["escalated"], r["action_match"], r["row_acc"]))

# rebuild the tau=1.0 cloud cascade per command, then test vs flat
flat_act, cloud_act, flat_row, cloud_row = [], [], [], []
for a in p45:
    t = json.loads(a["truth_json"]); t.setdefault("modifiers", [])
    lt = int(a["fact_loc_type_idx"]); ri = int(a["fact_row_idx"])
    esc = (lt == 0 and a["cloud_row"] in ROWS)          # tau = 1.0: always escalate grid-typed
    if lt == 0:
        sym = "grid_%s%s" % (a["cloud_row"] if esc else ROWS[ri], COLS[int(a["fact_col_idx"])])
    elif lt == 1:
        sym = CONT[int(a["fact_container_which_idx"])]
    else:
        sym = RT[int(a["fact_runtime_which_idx"])]
    co = emit(V[int(a["fact_verb_idx"])], T[int(a["fact_target_idx"])], sym)
    fo = emit(V[int(a["flat_verb_idx"])], T[int(a["flat_target_idx"])], a["flat_location"])
    cloud_act.append(int(bool(score(co, t).strict_match)))
    flat_act.append(int(bool(score(fo, t).strict_match)))
    tr = rowof(t.get("location"))
    if tr:
        cloud_row.append(int(rowof(co.get("location")) == tr))
        flat_row.append(int(rowof(fo.get("location")) == tr))
print("  rebuilt tau=1.0 cloud cascade: action %.4f  row %.4f  (target file says 0.7788 / 0.6759)"
      % (np.mean(cloud_act), np.mean(cloud_row)))

print()
print("  cascade-to-cloud (tau=1.0) vs flat classifier, held_out_v4c")
print("  %-14s %5s %9s %9s %10s %22s %7s %7s %12s %s"
      % ("metric", "n", "cloud", "flat", "diff pts", "95% CI", "b", "c", "McNemar p", "verdict"))
out = []
for nm, a, b in (("action_match", cloud_act, flat_act), ("row_accuracy", cloud_row, flat_row)):
    pt, lo, hi = boot(a, b)
    bb, cc, pv = mcn(a, b)
    sep = not (lo <= 0 <= hi)
    verdict = "CLAIMABLE" if (sep and pv < 0.05) else "NOT CLAIMABLE"
    print("  %-14s %5d %9.4f %9.4f %+10.2f   [%+7.2f, %+7.2f] %7d %7d %12.4g   %s"
          % (nm, len(a), np.mean(a), np.mean(b), 100 * pt, 100 * lo, 100 * hi, bb, cc, pv, verdict))
    out.append(dict(dataset="held_out_v4c",
                    comparison="cascade row->gpt-4o-mini (tau=1.0, always escalate) vs flat classifier",
                    arm_a="cascade_row_cloud_tau1.0", arm_b="flat_distilbert", metric=nm, n=len(a),
                    acc_a=round(float(np.mean(a)), 4), acc_b=round(float(np.mean(b)), 4),
                    diff_pts=round(100 * pt, 2), ci_lo_pts=round(100 * lo, 2), ci_hi_pts=round(100 * hi, 2),
                    mcnemar_b=bb, mcnemar_c=cc, mcnemar_p=("%.4g" % pv), ci_excludes_zero=sep, verdict=verdict))
print()
for nm, a, b in (("action_match", cloud_act, flat_act), ("row_accuracy", cloud_row, flat_row)):
    print("  %-14s CI across seeds: %s" % (nm, "  ".join(
        "[%+.2f,%+.2f]" % (100 * boot(a, b, seed=s)[1], 100 * boot(a, b, seed=s)[2]) for s in (20260827, 1, 2, 3))))

# also tau_A = 0.50 cloud, for completeness
flat2, cl2, fr2, cr2 = [], [], [], []
for a in p45:
    t = json.loads(a["truth_json"]); t.setdefault("modifiers", [])
    lt = int(a["fact_loc_type_idx"]); ri = int(a["fact_row_idx"])
    esc = (lt == 0 and float(a["fact_row_conf"]) < 0.5 and a["cloud_row"] in ROWS)
    if lt == 0:
        sym = "grid_%s%s" % (a["cloud_row"] if esc else ROWS[ri], COLS[int(a["fact_col_idx"])])
    elif lt == 1:
        sym = CONT[int(a["fact_container_which_idx"])]
    else:
        sym = RT[int(a["fact_runtime_which_idx"])]
    co = emit(V[int(a["fact_verb_idx"])], T[int(a["fact_target_idx"])], sym)
    cl2.append(int(bool(score(co, t).strict_match)))
    tr = rowof(t.get("location"))
    if tr: cr2.append(int(rowof(co.get("location")) == tr))
pt, lo, hi = boot(cl2, flat_act); bb, cc, pv = mcn(cl2, flat_act)
print()
print("  [for completeness] tau_A=0.50 cloud cascade vs flat, action: %.4f vs %.4f  %+.2f pts  CI [%+.2f,%+.2f]  b=%d c=%d p=%.4g  %s"
      % (np.mean(cl2), np.mean(flat_act), 100 * pt, 100 * lo, 100 * hi, bb, cc, pv,
         "CLAIMABLE" if (not (lo <= 0 <= hi) and pv < 0.05) else "NOT CLAIMABLE"))
ptr, lor, hir = boot(cr2, flat_row); bbr, ccr, pvr = mcn(cr2, flat_row)
print("  [for completeness] tau_A=0.50 cloud cascade vs flat, row   : %.4f vs %.4f  %+.2f pts  CI [%+.2f,%+.2f]  b=%d c=%d p=%.4g  %s"
      % (np.mean(cr2), np.mean(flat_row), 100 * ptr, 100 * lor, 100 * hir, bbr, ccr, pvr,
         "CLAIMABLE" if (not (lor <= 0 <= hir) and pvr < 0.05) else "NOT CLAIMABLE"))

with open(R + r"\thesis2\out\a13_v4c_cloud_significance.csv", "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(out[0].keys())); w.writeheader(); w.writerows(out)
print("\nwrote thesis2/out/a13_v4c_cloud_significance.csv")
