"""thesis2/scripts/a1_bilstm_check.py -- why are v3 and v4c identical for the BiLSTM?

The BiLSTM scored exactly 0.7005 / 0.7857 / 0.7976 / 0.6852 on BOTH held_out_v3
and held_out_v4c. Those sets differ in 72 commands' row wording. Identical
metrics across them is either a bug (wrong file loaded) or a real property
(the row words are all out-of-vocabulary and collapse to <unk>).

Checks:
  1. are the two prediction CSVs literally identical in pred_json?
  2. do the two datasets actually differ in their command strings?
  3. what fraction of the differing tokens are OOV for a train_v2 vocabulary?
  4. where does 0.6852 row accuracy come from if the row word is unreadable?
"""
import collections
import csv
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))
from p2_train_factorized import factorize, IGNORE, ROWS          # noqa: E402

OUT = REPO / "thesis2" / "out"


def load(n):
    return [json.loads(l) for l in open(REPO / "data" / ("%s.jsonl" % n),
                                        encoding="utf-8") if l.strip()]


def rd(p):
    with open(p, encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


v3, v4c = load("held_out_v3"), load("held_out_v4c")
p3 = rd(OUT / "a1_bilstm_held_out_v3.csv")
p4 = rd(OUT / "a1_bilstm_held_out_v4c.csv")

print("=" * 78)
print("1. ARE THE PREDICTIONS IDENTICAL?")
same = sum(1 for a, b in zip(p3, p4) if a["pred_json"] == b["pred_json"])
print("   identical pred_json rows: %d / %d" % (same, len(p3)))

print()
print("2. DO THE DATASETS DIFFER?")
diff = [(a["command"], b["command"]) for a, b in zip(v3, v4c)
        if a["command"] != b["command"]]
print("   commands that differ: %d / %d" % (len(diff), len(v3)))
for a, b in diff[:3]:
    print("      v3 : %s" % a)
    print("      v4c: %s" % b)

print()
print("3. ARE THE DIFFERING TOKENS OUT OF VOCABULARY?")
vocab = set()
for r in load("train_v2"):
    vocab.update(r["command"].lower().split())
print("   train_v2 vocabulary: %d types" % len(vocab))
changed = collections.Counter()
for a, b in diff:
    wa, wb = set(a.lower().split()), set(b.lower().split())
    for w in (wa ^ wb):
        changed[w] += 1
print("   tokens that change between v3 and v4c, and their OOV status:")
for w, n in changed.most_common():
    print("      %-10s x%-4d %s" % (w, n, "IN VOCAB" if w in vocab else "OOV -> <unk>"))
allo = all(w not in vocab for w in changed)
print("   every differing token is OOV: %s" % allo)

print()
print("4. WHERE DOES 0.6852 ROW ACCURACY COME FROM?")
by = collections.Counter()
rowcm = collections.Counter()
for rec, r in zip(p3, v3):
    tf = dict(zip(["loc_type", "row", "col", "cw", "rw"],
                  factorize(r["expected"].get("location"))))
    if tf["row"] == IGNORE:
        continue
    pl = json.loads(rec["pred_json"]).get("location")
    pf = factorize(pl)[1] if pl else IGNORE
    truth = ROWS[tf["row"]]
    pred = ROWS[pf] if pf != IGNORE else "?"
    rowcm[(truth, pred)] += 1
    by[truth] += 1
print("   row confusion (truth -> pred), n=%d grid commands:" % sum(by.values()))
for t in ROWS:
    line = "      %s ->" % t
    for p in ROWS + ["?"]:
        line += "  %s:%-3d" % (p, rowcm[(t, p)])
    print(line + "   (truth count %d)" % by[t])
corr = sum(rowcm[(t, t)] for t in ROWS)
print("   correct %d / %d = %.4f" % (corr, sum(by.values()), corr / sum(by.values())))
print()
print("   INTERPRETATION: if the row WORD is <unk>, any row accuracy above")
print("   chance must come from another cue in the sentence, not from the row")
print("   word itself. Check which rows are over-predicted above.")
print("=" * 78)
