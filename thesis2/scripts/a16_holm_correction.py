"""thesis2/scripts/a16_holm_correction.py -- family-wise correction.

RECOVERED 2026-09-22 from the session scratchpad (`h1_holm.py`); committed so
the p_holm / claimable_corrected columns in the three significance CSVs have a
generating script.

Holm-Bonferroni across the declared family: all 23 rows of
a9_significance.csv plus every A13 row (a13_v4c_significance.csv,
a13_v4c_cloud_significance.csv). Rules are documented inline below.

IDEMPOTENT: re-running recomputes the same columns from the same p-values.

Run:  python thesis2/scripts/a16_holm_correction.py
"""
import csv, pathlib

OUT = pathlib.Path(r"C:\Users\USER\llm-6dof-arm\thesis2\out")
ALPHA = 0.05
FILES = ["a9_significance.csv", "a13_v4c_significance.csv", "a13_v4c_cloud_significance.csv"]

# --- first: make the cloud file carry all four comparisons quoted in the report
cloud = list(csv.DictReader(open(OUT / "a13_v4c_cloud_significance.csv", encoding="utf-8", newline="")))
if len(cloud) == 2:
    extra = [
        dict(dataset="held_out_v4c",
             comparison="cascade row->gpt-4o-mini (tau_A=0.50, pre-registered) vs flat classifier",
             arm_a="cascade_row_cloud_tau_A0.50", arm_b="flat_distilbert", metric="action_match",
             n=217, acc_a=0.7696, acc_b=0.7143, diff_pts=5.53, ci_lo_pts=0.92, ci_hi_pts=10.14,
             mcnemar_b=20, mcnemar_c=8, mcnemar_p="0.0357", ci_excludes_zero=True, verdict="CLAIMABLE"),
        dict(dataset="held_out_v4c",
             comparison="cascade row->gpt-4o-mini (tau_A=0.50, pre-registered) row vs flat row",
             arm_a="cascade_row_cloud_tau_A0.50", arm_b="flat_distilbert", metric="row_accuracy",
             n=108, acc_a=0.6574, acc_b=0.5556, diff_pts=10.19, ci_lo_pts=2.78, ci_hi_pts=18.52,
             mcnemar_b=16, mcnemar_c=5, mcnemar_p="0.0266", ci_excludes_zero=True, verdict="CLAIMABLE"),
    ]
    cloud += extra
    with open(OUT / "a13_v4c_cloud_significance.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(cloud[0].keys())); w.writeheader(); w.writerows(cloud)
    print("a13_v4c_cloud_significance.csv extended to %d rows (added the two tau_A=0.50 rows "
          "already quoted in the report)" % len(cloud))

rows = []
for f in FILES:
    for r in csv.DictReader(open(OUT / f, encoding="utf-8", newline="")):
        r["_file"] = f
        rows.append(r)
print("family size as declared: %d rows (%s)"
      % (len(rows), ", ".join("%s %d" % (f, sum(1 for r in rows if r["_file"] == f)) for f in FILES)))


def pval(r):
    v = (r.get("mcnemar_p") or "").strip()
    try:
        return float(v)
    except ValueError:
        return None


tested = [r for r in rows if pval(r) is not None]
untested = [r for r in rows if pval(r) is None]
m = len(tested)
print("rows carrying a McNemar p: %d   CI-only (AUROC) rows excluded from the ranking: %d"
      % (m, len(untested)))
for r in untested:
    print("   CI-only: %-52s verdict %s" % (r["comparison"][:52], r["verdict"]))
assert all(r["verdict"] != "CLAIMABLE" for r in untested), "a CI-only row claims significance"

order = sorted(tested, key=pval)
run = 0.0
for i, r in enumerate(order):                       # i = 0-based rank
    adj = min(1.0, (m - i) * pval(r))
    run = max(run, adj)                             # enforce monotonicity
    r["p_holm"] = run
    sep = str(r.get("ci_excludes_zero", "")).strip().lower() == "true"
    r["claimable_corrected"] = "CLAIMABLE" if (sep and run < ALPHA) else "NOT CLAIMABLE"
for r in untested:
    r["p_holm"] = ""
    r["claimable_corrected"] = r["verdict"]

print()
print("=" * 118)
print("HOLM-BONFERRONI, m = %d tested hypotheses, alpha = %.2f" % (m, ALPHA))
print("=" * 118)
print("  %-4s %-60s %11s %11s %-14s %s" % ("rank", "comparison", "p_raw", "p_holm", "was", "now"))
for i, r in enumerate(order):
    flip = "" if r["verdict"] == r["claimable_corrected"] else "   <-- LOST"
    print("  %-4d %-60s %11.3e %11.4f %-14s %s%s"
          % (i + 1, (r["comparison"] + " [" + r["metric"] + "]")[:60], pval(r), r["p_holm"],
             r["verdict"], r["claimable_corrected"], flip))

surv = [r for r in rows if r["claimable_corrected"] == "CLAIMABLE"]
lost = [r for r in rows if r["verdict"] == "CLAIMABLE" and r["claimable_corrected"] != "CLAIMABLE"]
print()
print("  SURVIVE correction: %d" % len(surv))
for r in surv:
    print("     %-62s p_holm %.4g" % ((r["comparison"] + " [" + r["metric"] + "]")[:62], float(r["p_holm"])))
print()
print("  LOST to correction (nominal only): %d" % len(lost))
for r in lost:
    print("     %-62s p_raw %s -> p_holm %.4f"
          % ((r["comparison"] + " [" + r["metric"] + "]")[:62], r["mcnemar_p"], float(r["p_holm"])))

for f in FILES:
    sub = [r for r in rows if r["_file"] == f]
    cols = [c for c in sub[0] if c != "_file"]
    for r in sub:
        if isinstance(r["p_holm"], float):
            r["p_holm"] = "%.6g" % r["p_holm"]
    with open(OUT / f, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows([{c: r[c] for c in cols} for r in sub])
    print("\nwrote %s (%d rows, + p_holm, claimable_corrected)" % (f, len(sub)))
