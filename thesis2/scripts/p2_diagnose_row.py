"""thesis2/scripts/p2_diagnose_row.py -- is the row-head collapse real or a bug?

Three checks:
  1. Label mapping: does factorize() agree with the raw symbol on held-out?
  2. Prediction distribution: is the row head constant, or inverted?
  3. Train/dev vs held-out: does it work on the vocabulary it was trained on?

If the head is right on far/near and wrong on upper/bottom, the collapse is a
vocabulary-generalization failure, not a wiring error.
"""
import collections
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from p2_train_factorized import (FactorizedClassifier, build_label_space,
                                 factorize, IGNORE, ROWS, MAX_LEN)
from ft.train_classifier import MultiHeadClassifier, build_label_space as flat_space
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
space = build_label_space()
fact = FactorizedClassifier(space)
fact.load_state_dict(torch.load(
    REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt",
    map_location="cpu"))
fact.eval()
flat = MultiHeadClassifier(flat_space())
flat.load_state_dict(torch.load(
    REPO / "ft" / "runs" / "distilbert_cls" / "epoch_09" / "model.pt", map_location="cpu"))
flat.eval()
fsp = flat_space()

UPDOWN = ("upper", "bottom", "top", "lower")
FARNEAR = ("far", "near", "farthest", "nearest")


def buck(cmd):
    low = " %s " % " ".join(cmd.lower().replace(",", " ").split())
    fn = any(" %s " % w in low for w in FARNEAR)
    ud = any(" %s " % w in low for w in UPDOWN)
    return "updown" if (ud and not fn) else ("farnear" if (fn and not ud) else
                                             ("both" if fn and ud else "neither"))


def rows_of(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


@torch.no_grad()
def predict(rowsrc, label):
    print("=" * 74)
    print("%s  (n=%d)" % (label, len(rowsrc)))
    conf = collections.Counter()
    bybuck = collections.defaultdict(lambda: [0, 0])
    flatconf = collections.Counter()
    for r in rowsrc:
        loc = r["expected"].get("location")
        lt, rr, cc, cw, rw = factorize(loc)
        if rr == IGNORE:
            continue
        enc = tok(r["command"], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt")
        lg = fact(enc["input_ids"], enc["attention_mask"])
        pr = int(lg["row"].argmax(-1).item())
        conf[(ROWS[rr], ROWS[pr])] += 1
        b = buck(r["command"])
        bybuck[b][1] += 1
        if pr == rr:
            bybuck[b][0] += 1
        _v, _t, ll = flat(enc["input_ids"], enc["attention_mask"])
        ploc = fsp["locations"][int(ll.argmax(-1).item())]
        prow = ploc[5] if ploc.startswith("grid_") and len(ploc) == 7 else "?"
        flatconf[(ROWS[rr], prow)] += 1

    print("  FACTORIZED row confusion (truth -> pred):")
    for t in ROWS:
        line = "     %s ->" % t
        for p in ROWS:
            line += "  %s:%-4d" % (p, conf[(t, p)])
        print(line)
    print("  FLAT head's symbol, row component (truth -> pred):")
    for t in ROWS:
        line = "     %s ->" % t
        for p in ROWS + ["?"]:
            line += "  %s:%-4d" % (p, flatconf[(t, p)])
        print(line)
    print("  factorized row accuracy by vocabulary bucket:")
    for b, (h, n) in sorted(bybuck.items()):
        print("     %-9s %3d/%-3d = %.4f" % (b, h, n, h / n))
    print()


predict(rows_of(REPO / "data" / "train_v2.jsonl"), "TRAIN (seen during training)")
predict(rows_of(REPO / "data" / "dev_v2.jsonl"), "DEV (selection only)")
predict(rows_of(REPO / "data" / "held_out_v3.jsonl"), "HELD_OUT_V3 (never seen)")

print("=" * 74)
print("row-word vocabulary present in each split (grid rows only):")
for name in ("train_v2", "dev_v2", "held_out_v3"):
    rs = rows_of(REPO / "data" / ("%s.jsonl" % name))
    c = collections.Counter()
    for r in rs:
        if factorize(r["expected"].get("location"))[1] == IGNORE:
            continue
        c[buck(r["command"])] += 1
    print("   %-12s %s" % (name, dict(c)))
print("=" * 74)
