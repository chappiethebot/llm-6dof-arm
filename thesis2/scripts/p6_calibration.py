"""thesis2/scripts/p6_calibration.py -- P6.4: calibration of the SALSA logit softmax.

Offline, from the cached predictions. Compares the calibration of the
single-pass logit distribution over A/B/C against the DistilBERT factor heads
on the same commands and the same 3-way question.
"""
import csv
import json
import os
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
OUT = REPO / "thesis2" / "out"


def ece(pairs, bins=10):
    if not pairs:
        return float("nan")
    tot = 0.0
    for i in range(bins):
        a, b = i / bins, (i + 1) / bins
        sub = [p for p in pairs if a <= p[0] < b or (i == bins - 1 and p[0] >= a)]
        if not sub:
            continue
        conf = sum(c for c, _ in sub) / len(sub)
        acc = sum(1.0 for _, k in sub if k) / len(sub)
        tot += (len(sub) / len(pairs)) * abs(conf - acc)
    return tot


def brier(pairs):
    return (sum((c - (1.0 if k else 0.0)) ** 2 for c, k in pairs) / len(pairs)
            if pairs else float("nan"))


rows = []
print("=" * 86)
print("P6.4 -- calibration on the 3-way ROW question, same commands, same denominator")
print("=" * 86)
print("%-14s %-22s %5s %8s %9s %8s %8s" %
      ("dataset", "scorer", "n", "acc", "mean_conf", "ECE", "Brier"))
for ds in ("held_out_v3", "held_out_v4c"):
    with open(OUT / ("p45_predictions_%s.csv" % ds), encoding="utf-8", newline="") as fh:
        recs = [r for r in csv.DictReader(fh) if r["truth_row"]]
    arms = [("distilbert_factor_row", "fact_row_conf", "fact_row")]
    for tag in ("qwen3b", "qwen3blora"):
        for mp in ("letters", "digits"):
            arms.append(("salsa_%s_%s" % (tag, mp),
                         "salsa_%s_%s_conf" % (tag, mp),
                         "salsa_%s_%s_row" % (tag, mp)))
    for name, confcol, predcol in arms:
        pairs = [(float(r[confcol]), r[predcol] == r["truth_row"]) for r in recs
                 if r.get(confcol)]
        if not pairs:
            continue
        acc = sum(1 for _, k in pairs if k) / len(pairs)
        mc = sum(c for c, _ in pairs) / len(pairs)
        e, b = ece(pairs), brier(pairs)
        print("%-14s %-22s %5d %8.4f %9.4f %8.4f %8.4f"
              % (ds, name, len(pairs), acc, mc, e, b))
        rows.append(dict(dataset=ds, scorer=name, n=len(pairs),
                         accuracy=round(acc, 4), mean_confidence=round(mc, 4),
                         ECE=round(e, 4), Brier=round(b, 4),
                         overconfident=round(mc - acc, 4)))

p = OUT / "logit_calibration.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print()
print("  `overconfident` = mean_confidence - accuracy. Positive means the scorer")
print("  claims more certainty than it earns.")
for r in rows:
    if r["overconfident"] > 0.25:
        print("    %-14s %-22s +%.3f  <-- badly overconfident"
              % (r["dataset"], r["scorer"], r["overconfident"]))
print()
print("wrote %s" % p.relative_to(REPO))
print("=" * 86)
