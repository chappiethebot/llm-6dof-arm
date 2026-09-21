"""thesis2/scripts/p2_row_confidence.py -- can a confidence gate SEE the row failure?

The row head is 0/72 on updown and 36/36 on neither. The question that decides
whether per-factor escalation (P4.5) can work at all:

  is the head UNCONFIDENT when it is wrong (a gate fires, escalation rescues
  those commands) or CONFIDENT when it is wrong (a gate is blind and no
  escalation policy can help)?
"""
import json
import os
import statistics
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from p2_train_factorized import (FactorizedClassifier, build_label_space,
                                 factorize, IGNORE, ROWS, MAX_LEN)
from transformers import AutoTokenizer

UPDOWN = ("upper", "bottom", "top", "lower")
FARNEAR = ("far", "near", "farthest", "nearest")


def buck(cmd):
    low = " %s " % " ".join(cmd.lower().replace(",", " ").split())
    fn = any(" %s " % w in low for w in FARNEAR)
    ud = any(" %s " % w in low for w in UPDOWN)
    return "updown" if (ud and not fn) else ("farnear" if (fn and not ud) else
                                             ("both" if fn and ud else "neither"))


tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
space = build_label_space()
m = FactorizedClassifier(space)
m.load_state_dict(torch.load(
    REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt",
    map_location="cpu"))
m.eval()

rows = [json.loads(l) for l in open(REPO / "data" / "held_out_v3.jsonl",
                                    encoding="utf-8") if l.strip()]
groups = {"updown": [], "neither": []}
with torch.no_grad():
    for r in rows:
        lt, rr, cc, cw, rw = factorize(r["expected"].get("location"))
        if rr == IGNORE:
            continue
        enc = tok(r["command"], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt")
        lg = m(enc["input_ids"], enc["attention_mask"])
        p = torch.softmax(lg["row"], dim=-1)[0]
        v, i = torch.sort(p, descending=True)
        groups[buck(r["command"])].append(
            dict(conf=float(v[0]), margin=float(v[0] - v[1]),
                 correct=int(i[0]) == rr, pred=ROWS[int(i[0])], truth=ROWS[rr]))

print("=" * 74)
print("ROW-HEAD CONFIDENCE ON held_out_v3, by vocabulary bucket")
print("=" * 74)
print("%-9s %4s %8s %9s %9s %9s %9s" %
      ("bucket", "n", "acc", "mean_cnf", "med_cnf", "min_cnf", "mean_mrg"))
for b, g in sorted(groups.items()):
    if not g:
        continue
    acc = sum(x["correct"] for x in g) / len(g)
    cf = [x["conf"] for x in g]
    mg = [x["margin"] for x in g]
    print("%-9s %4d %8.4f %9.4f %9.4f %9.4f %9.4f" %
          (b, len(g), acc, statistics.mean(cf), statistics.median(cf),
           min(cf), statistics.mean(mg)))

print()
print("Can a threshold separate the two buckets?")
allrows = groups["updown"] + groups["neither"]
for tau in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99):
    fire = [x for x in allrows if x["conf"] < tau]
    ud_fire = sum(1 for x in fire if not x["correct"])
    ok_fire = sum(1 for x in fire if x["correct"])
    wrong_total = sum(1 for x in allrows if not x["correct"])
    print("  tau=%.2f  escalate %3d/%d (%.0f%%)  catches %d/%d wrong (%.0f%% recall)"
          "  wastes %d on already-correct"
          % (tau, len(fire), len(allrows), 100 * len(fire) / len(allrows),
             ud_fire, wrong_total, 100 * ud_fire / wrong_total, ok_fire))

print()
print("Confidence distribution of WRONG row predictions (all updown):")
w = sorted(x["conf"] for x in allrows if not x["correct"])
print("  n=%d  min %.4f  p25 %.4f  median %.4f  p75 %.4f  max %.4f"
      % (len(w), w[0], w[len(w) // 4], statistics.median(w),
         w[3 * len(w) // 4], w[-1]))
c = sorted(x["conf"] for x in allrows if x["correct"])
print("Confidence distribution of CORRECT row predictions:")
print("  n=%d  min %.4f  p25 %.4f  median %.4f  p75 %.4f  max %.4f"
      % (len(c), c[0], c[len(c) // 4], statistics.median(c),
         c[3 * len(c) // 4], c[-1]))
print("=" * 74)
