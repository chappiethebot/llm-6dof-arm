"""thesis2/final/s4_significance.py -- STAGE 4: the declared families.

    python thesis2/final/s4_significance.py

PRIMARY FAMILY -- 8 hypotheses, DECLARED 22 Sep 2026, BEFORE this run, and
hard-coded below. Holm-Bonferroni WITHIN these 8.

TRANSPARENCY NOTE, printed in the output and written to the CSV: H1, H2, H4 and
H5 were measured once before this family was declared. The old 26-hypothesis
Holm values are reported alongside, read from the earlier significance files, so
the reader can see both.

EXPLORATORY FAMILY -- every other pairwise comparison between arms that have
predictions on the same split, Holm across all of them. Reported, never
headlined.

PROTOCOL: clustered bootstrap over command_id, 2,000 resamples, seed 20260827;
McNemar's exact binomial test; claimable only if the 95 % CI excludes 0 AND
p_holm < 0.05.

Output: thesis2/out/final/final_significance.csv with a `family` column.
"""
import csv
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402
import arms as A                                                        # noqa: E402

import numpy as np                                                      # noqa: E402

B, ALPHA = 2000, 0.05

# ------------------------------------------------- the primary family, fixed
PRIMARY = [
    ("H1", "held_out_v4c", "flat_gpu", "lora_greedy", "action"),
    ("H2", "held_out_v4c", "flat_gpu", "lora_greedy", "row"),
    ("H3", "held_out_v3", "lora_greedy", "flat_gpu", "action"),
    ("H4", "held_out_v3", "factorized_fp16", "flat_gpu", "action"),
    ("H5", "held_out_v4c", "cascade_cloud_tauA", "flat_gpu", "action"),
    ("H6", "held_out_v4c", "grpo_base", "lora_greedy", "action"),
    ("H7", "held_out_v3", "salsa_full_qwen2.5-3b-lora", "lora_greedy", "action"),
    ("H8", "held_out_v4c", "textcnn", "flat_gpu", "action"),
]
PRE_MEASURED = {"H1", "H2", "H4", "H5"}
DECLARED = "2026-09-22"

# The old 26-hypothesis Holm values, for the transparency note.
OLD_FILES = ["a9_significance.csv", "a13_v4c_significance.csv",
             "a13_v4c_cloud_significance.csv"]


def old_holm():
    out = []
    for fn in OLD_FILES:
        p = C.REPO / "thesis2" / "out" / fn
        if p.exists():
            out += list(csv.DictReader(p.open(encoding="utf-8", newline="")))
    return out


def outcomes(arm_name, split, metric):
    """{command_id: bool} for one arm, one split, one metric."""
    rows = C.read_preds(arm_name, split)
    if not rows:
        return None
    arm = A.REGISTRY.get(arm_name)
    by = defaultdict(list)
    for r in rows:
        by[r["command_id"]].append(r)
    out = {}
    for cid, rs in by.items():
        rs.sort(key=lambda r: int(r["repeat"]))
        col = "correct" if metric == "action" else "row_correct"
        vals = [r[col] for r in rs]
        if metric == "row" and vals[0] in ("", None):
            continue
        bools = [v in (True, "True") for v in vals]
        out[cid] = (bools[0] if arm and arm.determinism == "deterministic"
                    else C.majority(bools))
    return out


def mcnemar_exact(a, b):
    bb = sum(1 for x, y in zip(a, b) if x and not y)
    cc = sum(1 for x, y in zip(a, b) if y and not x)
    n = bb + cc
    if n == 0:
        return bb, cc, 1.0
    k = min(bb, cc)
    return bb, cc, min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / 2.0 ** n)


def clustered_bootstrap(a, b, seed=C.SEED, n_boot=B):
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    n = len(a)
    d = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)          # clusters = command_ids
        d[i] = a[idx].mean() - b[idx].mean()
    return a.mean() - b.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)


def compare(hid, family, split, arm_a, arm_b, metric):
    da, db = outcomes(arm_a, split, metric), outcomes(arm_b, split, metric)
    base = dict(id=hid, family=family, split=split, arm_a=arm_a, arm_b=arm_b,
                metric=metric)
    if da is None or db is None:
        miss = [n for n, d in ((arm_a, da), (arm_b, db)) if d is None]
        return dict(base, n="", acc_a="", acc_b="", diff_pts="", ci_lo_pts="",
                    ci_hi_pts="", mcnemar_b="", mcnemar_c="", mcnemar_p="",
                    p_holm="", verdict="NOT RUN",
                    note="no predictions for: %s" % ", ".join(miss))
    keys = sorted(set(da) & set(db), key=int)
    a = [int(da[k]) for k in keys]
    b = [int(db[k]) for k in keys]
    pt, lo, hi = clustered_bootstrap(a, b)
    bb, cc, p = mcnemar_exact(a, b)
    return dict(base, n=len(keys), acc_a=round(float(np.mean(a)), 4),
                acc_b=round(float(np.mean(b)), 4), diff_pts=round(100 * pt, 2),
                ci_lo_pts=round(100 * lo, 2), ci_hi_pts=round(100 * hi, 2),
                mcnemar_b=bb, mcnemar_c=cc, mcnemar_p="%.6g" % p,
                p_holm="", verdict="", note="")


def holm(rows):
    """Holm-Bonferroni within one family; sets p_holm and verdict in place."""
    tested = [r for r in rows if r["mcnemar_p"] != ""]
    m = len(tested)
    run = 0.0
    for i, r in enumerate(sorted(tested, key=lambda r: float(r["mcnemar_p"]))):
        adj = min(1.0, (m - i) * float(r["mcnemar_p"]))
        run = max(run, adj)
        r["p_holm"] = "%.6g" % run
        sep = not (float(r["ci_lo_pts"]) <= 0 <= float(r["ci_hi_pts"]))
        r["verdict"] = "CLAIMABLE" if (sep and run < ALPHA) else (
            "nominal (does not survive family-wise correction)" if sep and float(r["mcnemar_p"]) < ALPHA
            else "NOT CLAIMABLE")
    return m


def main():
    C.start_log("s4_significance")
    print("  protocol: clustered bootstrap over command_id, %d resamples, seed %d;"
          % (B, C.SEED))
    print("            exact McNemar; claimable iff CI excludes 0 AND p_holm < %.2f" % ALPHA)
    print("  primary family DECLARED %s, before this run; 8 hypotheses, hard-coded." % DECLARED)

    rows = [compare(hid, "primary", split, a, b, m) for hid, split, a, b, m in PRIMARY]
    m_primary = holm(rows)

    print()
    print("  PRIMARY FAMILY (Holm within %d tested of 8 declared)" % m_primary)
    print("  %-4s %-14s %-28s %-8s %6s %9s %9s %18s %10s %s"
          % ("id", "split", "comparison", "metric", "n", "A", "B", "95% CI", "p_holm", "verdict"))
    for r in rows:
        print("  %-4s %-14s %-28s %-8s %6s %9s %9s %18s %10s %s"
              % (r["id"], r["split"], ("%s vs %s" % (r["arm_a"], r["arm_b"]))[:28],
                 r["metric"], r["n"], r["acc_a"], r["acc_b"],
                 ("[%s, %s]" % (r["ci_lo_pts"], r["ci_hi_pts"])) if r["n"] else "-",
                 r["p_holm"] or "-", r["verdict"] or r.get("note", "")))

    print()
    print("  TRANSPARENCY NOTE")
    print("    H1, H2, H4 and H5 were measured ONCE before this family was declared")
    print("    (%s). They are not new measurements. The 26-hypothesis Holm values" % DECLARED)
    print("    from that earlier family are reported alongside:")
    old = old_holm()
    # (substring of the old `comparison`, required old `metric`) -- both are
    # needed: the v4c action and row rows share one comparison string, and the
    # two cloud cascades differ only by the tau in the string.
    want = {
        "H1": ("flat classifier vs qwen2.5-3b-lora", "action_match"),
        "H2": ("flat classifier vs qwen2.5-3b-lora", "row_accuracy"),
        "H4": ("factorized vs flat", "accuracy"),
        "H5": ("tau_A=0.50", "action_match"),
    }
    for hid in ("H1", "H2", "H4", "H5"):
        frag, metric = want[hid]
        hit = [o for o in old
               if frag.lower() in o.get("comparison", "").lower()
               and o.get("metric", "") == metric]
        if hit:
            o = hit[0]
            print("      %-3s old family: %-56s p_raw %-10s p_holm %-10s -> %s"
                  % (hid, o["comparison"][:56], o.get("mcnemar_p", ""),
                     o.get("p_holm", ""), o.get("claimable_corrected", "")))
        else:
            print("      %-3s old family: no row with comparison~%r and metric=%r"
                  % (hid, frag, metric))

    # ------------------------------------------------------ exploratory
    print()
    print("  EXPLORATORY FAMILY -- every other pair with predictions on the same split")
    have = {}
    for split in C.SPLITS:
        for name in sorted(A.REGISTRY):
            if C.read_preds(name, split):
                have.setdefault(split, []).append(name)
    primary_keys = {(s, a, b, m) for _, s, a, b, m in PRIMARY}
    ex = []
    for split, names in have.items():
        for a, b in itertools.combinations(names, 2):
            for metric in ("action", "row"):
                if (split, a, b, metric) in primary_keys or (split, b, a, metric) in primary_keys:
                    continue
                ex.append(compare("E%03d" % (len(ex) + 1), "exploratory", split, a, b, metric))
    m_ex = holm(ex)
    print("    %d exploratory comparisons, Holm across all of them" % m_ex)
    claim = [r for r in ex if r["verdict"] == "CLAIMABLE"]
    print("    %d survive correction; reported, never headlined" % len(claim))
    for r in claim[:15]:
        print("      %-14s %-40s %s pts, p_holm %s"
              % (r["split"], "%s vs %s (%s)" % (r["arm_a"], r["arm_b"], r["metric"]),
                 r["diff_pts"], r["p_holm"]))

    allrows = rows + ex
    cols = ["id", "family", "split", "arm_a", "arm_b", "metric", "n", "acc_a", "acc_b",
            "diff_pts", "ci_lo_pts", "ci_hi_pts", "mcnemar_b", "mcnemar_c",
            "mcnemar_p", "p_holm", "verdict", "note"]
    p = C.OUT / "final_significance.csv"
    with p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in allrows:
            w.writerow({c: r.get(c, "") for c in cols})

    notrun = [r["id"] for r in rows if r["verdict"] == "NOT RUN"]
    C.gate(True, [
        "primary: %d of 8 hypotheses tested, %d CLAIMABLE after Holm within the family"
        % (m_primary, sum(1 for r in rows if r["verdict"] == "CLAIMABLE")),
        "primary hypotheses NOT RUN (missing arms): %s" % (notrun or "none"),
        "exploratory: %d comparisons, %d claimable, reported never headlined" % (m_ex, len(claim)),
        "H1/H2/H4/H5 pre-date the family declaration; old Holm values printed above",
        "output: %s" % p.relative_to(C.REPO),
    ])


if __name__ == "__main__":
    main()
