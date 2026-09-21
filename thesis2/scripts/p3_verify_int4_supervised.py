"""thesis2/scripts/p3_verify_int4_supervised.py -- agreement where the head is USED.

The first verification measured prediction agreement over all 217 commands.
That over-counts: emit() discards a head's output unless the predicted verb
requires that field, so a container_which prediction on a `stop` command is
never used and its flip is invisible to the metric.

This recomputes agreement restricted to the commands where each head is
actually supervised -- i.e. where its output reaches the emitted object. That
is the number that says whether INT4 changes what the system DOES.
"""
import json
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from p2_train_factorized import (FactorizedClassifier, FACTORS,     # noqa: E402
                                 build_label_space, factorize, IGNORE, MAX_LEN)
from p3_quantization_gate import PrecisionSwitch                    # noqa: E402
from transformers import AutoTokenizer                              # noqa: E402

CK = REPO / "thesis2" / "models" / "distilbert_factorized" / "epoch_11" / "model.pt"
OUT = REPO / "thesis2" / "out"
HEADS = ["verb", "target"] + FACTORS

device = "cuda" if torch.cuda.is_available() else "cpu"
space = build_label_space()
tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
rows = [json.loads(l) for l in open(REPO / "data" / "held_out_v3.jsonl",
                                    encoding="utf-8") if l.strip()]
model = FactorizedClassifier(space)
model.load_state_dict(torch.load(CK, map_location="cpu"))
model.eval().to(device)
if device == "cuda":
    model.half()
sw = PrecisionSwitch(model)

enc = [tok(r["command"], truncation=True, max_length=MAX_LEN,
           padding="max_length", return_tensors="pt").to(device) for r in rows]
pred = {}
for mode in ("fp16", "int4"):
    sw.use(mode)
    store = {h: [] for h in HEADS}
    with torch.no_grad():
        for e in enc:
            lg = model(e["input_ids"], e["attention_mask"])
            for h in HEADS:
                store[h].append(int(lg[h].argmax(-1).item()))
    pred[mode] = store

# which commands actually USE each head
used = {h: [] for h in HEADS}
for i, r in enumerate(rows):
    exp = r["expected"]
    used["verb"].append(i)
    if exp.get("target"):
        used["target"].append(i)
    tf = dict(zip(FACTORS, factorize(exp.get("location"))))
    for f in FACTORS:
        if tf[f] != IGNORE:
            used[f].append(i)

print("=" * 86)
print("AGREEMENT RESTRICTED TO COMMANDS WHERE THE HEAD'S OUTPUT IS USED")
print("=" * 86)
print("%-18s %6s %9s %11s   %6s %9s %11s" %
      ("head", "n_all", "agree_all", "pct_all", "n_used", "agree_use", "pct_use"))
recs = []
for h in HEADS:
    a_all = sum(1 for i in range(len(rows)) if pred["fp16"][h][i] == pred["int4"][h][i])
    u = used[h]
    a_use = sum(1 for i in u if pred["fp16"][h][i] == pred["int4"][h][i])
    print("%-18s %6d %9d %10.2f%%   %6d %9d %10.2f%%" %
          (h, len(rows), a_all, 100.0 * a_all / len(rows),
           len(u), a_use, 100.0 * a_use / len(u) if u else float("nan")))
    recs.append(dict(head=h, n_all=len(rows), agree_all=a_all,
                     pct_all=round(100.0 * a_all / len(rows), 2),
                     n_used=len(u), agree_used=a_use,
                     pct_used=round(100.0 * a_use / len(u), 2) if u else ""))

print()
print("Reading: `pct_all` counts flips on commands where emit() throws the")
print("head's output away (a container_which prediction on a `stop` command).")
print("`pct_used` is the number that says whether INT4 changes what the system")
print("actually emits.")

import csv
p = OUT / "p3_int4_agreement_supervised.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=list(recs[0]))
    w.writeheader()
    w.writerows(recs)
print()
print("wrote %s" % p.relative_to(REPO))
print("=" * 86)
