"""thesis2/scripts/a5_massive_constrained.py -- A5: does constrained reconstruction fix it?

Phase 7 reported that the factorized head emits an ILLEGAL scenario_action
combination on 1.98% of MASSIVE test rows (only 60 of 828 combinations exist).
Two questions, both answerable offline from the saved predictions:

  1. Does constrained reconstruction -- picking the highest-scoring LEGAL
     combination instead of the independent argmaxes -- recover that 1.98%?
  2. Does it change the per-factor vs flat ROUTER comparison, which is the
     actual Idea 1 evidence?

The saved per-row file carries only the argmax and its confidence, not the full
factor distributions, so the constrained decode is done two ways:

  (a) EXACT, re-running the model to get full scenario/action distributions and
      maximising p(scenario)*p(action) over the 60 legal pairs;
  (b) a cheap fallback if the model cannot be re-run: snap an illegal pair to
      the legal pair sharing its scenario with the most training mass.

(a) is attempted first and is what gets reported if it succeeds.
"""
import collections
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))
OUT = REPO / "thesis2" / "out"


def auroc(pairs):
    pos = np.sort(np.array([s for s, k in pairs if k], dtype=float))
    neg = np.array([s for s, k in pairs if not k], dtype=float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    r1 = np.searchsorted(pos, neg, side="left")
    r2 = np.searchsorted(pos, neg, side="right")
    return float(np.sum(len(pos) - r2) + 0.5 * np.sum(r2 - r1)) / (len(pos) * len(neg))


print("=" * 80)
print("A5 -- MASSIVE illegal-combination rate under constrained reconstruction")
print("=" * 80)

from p7_massive import Model, BASE, MAX_LEN, SEED, set_seed      # noqa: E402
from datasets import load_dataset                                 # noqa: E402
from transformers import AutoTokenizer                            # noqa: E402

set_seed(SEED)
d = load_dataset("mteb/amazon_massive_intent", "en")
tr, te = d["train"], d["test"]
intents = sorted(set(tr["label_text"]))
scen = sorted({i.split("_")[0] for i in intents})
acts = sorted({i.split("_", 1)[1] for i in intents})
legal = set(intents)
print("  intents %d  scenarios %d  actions %d  legal pairs %d of %d"
      % (len(intents), len(scen), len(acts), len(legal), len(scen) * len(acts)))

# The Phase 7 script did not persist the trained weights, so the model is
# retrained with the identical seed and recipe. Stated plainly rather than
# pretending the original run was reloaded.
print()
print("  NOTE: p7_massive.py did not save its checkpoint, so this RETRAINS with")
print("  the same seed (%d) and recipe. Numbers may differ in the last decimal"
      % SEED)
print("  from the Phase 7 report if cuDNN picks different kernels.")

device = "cuda" if torch.cuda.is_available() else "cpu"
tok = AutoTokenizer.from_pretrained(BASE)
i2i = {s: k for k, s in enumerate(intents)}
s2i = {s: k for k, s in enumerate(scen)}
a2i = {s: k for k, s in enumerate(acts)}

import torch.nn as nn                                            # noqa: E402
from torch.utils.data import DataLoader, Dataset                 # noqa: E402
import math                                                      # noqa: E402


class DS(Dataset):
    def __init__(self, split):
        self.t = split["text"]
        self.f = [i2i[x] for x in split["label_text"]]
        self.s = [s2i[x.split("_")[0]] for x in split["label_text"]]
        self.a = [a2i[x.split("_", 1)[1]] for x in split["label_text"]]

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = tok(self.t[i], truncation=True, max_length=MAX_LEN,
                padding="max_length", return_tensors="pt")
        return dict(input_ids=e["input_ids"].squeeze(0),
                    attention_mask=e["attention_mask"].squeeze(0),
                    flat=torch.tensor(self.f[i]), scen=torch.tensor(self.s[i]),
                    act=torch.tensor(self.a[i]))


dtr, dte = DS(tr), DS(te)
dl = DataLoader(dtr, batch_size=32, shuffle=True,
                generator=torch.Generator().manual_seed(SEED))
m = Model(len(intents), len(scen), len(acts)).to(device)
opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=0.01)
total = 4 * math.ceil(len(dtr) / 32)


def lam(step):
    return (step / 100) if step < 100 else max(0.0, 1.0 - (step - 100) / max(1, total - 100))


sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
ce = nn.CrossEntropyLoss()
for ep in range(4):
    m.train()
    for b in dl:
        fl, sc, ac = m(b["input_ids"].to(device), b["attention_mask"].to(device))
        loss = (ce(fl, b["flat"].to(device)) + ce(sc, b["scen"].to(device))
                + ce(ac, b["act"].to(device)))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); sch.step()
    print("    epoch %d done" % (ep + 1))

LEGAL_PAIRS = [(s2i[i.split("_")[0]], a2i[i.split("_", 1)[1]]) for i in intents]

m.eval()
res = []
with torch.no_grad():
    for b in DataLoader(dte, batch_size=128):
        fl, sc, ac = m(b["input_ids"].to(device), b["attention_mask"].to(device))
        pf = torch.softmax(fl, -1); ps = torch.softmax(sc, -1); pa = torch.softmax(ac, -1)
        for i in range(fl.shape[0]):
            si, ai = int(ps[i].argmax()), int(pa[i].argmax())
            free = "%s_%s" % (scen[si], acts[ai])
            lp = ps[i].float().log(); la = pa[i].float().log()
            best = max(LEGAL_PAIRS, key=lambda p: float(lp[p[0]] + la[p[1]]))
            constrained = "%s_%s" % (scen[best[0]], acts[best[1]])
            res.append(dict(truth=intents[int(b["flat"][i])],
                            flat_pred=intents[int(pf[i].argmax())],
                            flat_conf=float(pf[i].max()),
                            free_sym=free, free_legal=free in legal,
                            constrained_sym=constrained,
                            scen_conf=float(ps[i].max()), act_conf=float(pa[i].max()),
                            constrained_conf=float(torch.exp(lp[best[0]] + la[best[1]]))))

n = len(res)
flat_acc = sum(1 for r in res if r["flat_pred"] == r["truth"]) / n
free_acc = sum(1 for r in res if r["free_sym"] == r["truth"]) / n
con_acc = sum(1 for r in res if r["constrained_sym"] == r["truth"]) / n
illegal = sum(1 for r in res if not r["free_legal"]) / n
illegal_n = sum(1 for r in res if not r["free_legal"])
fixed = sum(1 for r in res if not r["free_legal"] and r["constrained_sym"] == r["truth"])

print()
print("  TEST n=%d" % n)
print("  flat 60-way                          %.4f" % flat_acc)
print("  factorized, free argmax              %.4f   (illegal %.2f%%, %d rows)"
      % (free_acc, 100 * illegal, illegal_n))
print("  factorized, CONSTRAINED to 60 legal  %.4f   (illegal 0.00%% by construction)"
      % con_acc)
print("  constrained - free                   %+.2f pts" % (100 * (con_acc - free_acc)))
print("  of the %d illegal rows, constrained decode got %d right (%.1f%%)"
      % (illegal_n, fixed, 100.0 * fixed / illegal_n if illegal_n else 0))

print()
print("  ROUTER COMPARISON -- does constraining change the Idea 1 evidence?")
rows = []
for label, symkey, confkey in (("free", "free_sym", None),
                               ("constrained", "constrained_sym", "constrained_conf")):
    minf = [(min(r["scen_conf"], r["act_conf"]), r[symkey] == r["truth"]) for r in res]
    flt = [(r["flat_conf"], r[symkey] == r["truth"]) for r in res]
    a_f, a_l = auroc(minf), auroc(flt)
    print("    %-12s AUROC per-factor %.4f | flat %.4f | gap %+.4f"
          % (label, a_f, a_l, a_f - a_l))
    for budget in (0.05, 0.10, 0.20, 0.30, 0.50):
        line = "      budget %3.0f%%" % (100 * budget)
        for signame, scores in (("per_factor", minf), ("flat", flt)):
            order = sorted(range(n), key=lambda i: scores[i][0])
            k = int(round(budget * n))
            esc = set(order[:k])
            hit = 0
            for i, r in enumerate(res):
                if i in esc:
                    if r["scen_conf"] <= r["act_conf"]:
                        sym = "%s_%s" % (r["truth"].split("_")[0],
                                         r[symkey].split("_", 1)[1])
                    else:
                        sym = "%s_%s" % (r[symkey].split("_")[0],
                                         r["truth"].split("_", 1)[1])
                else:
                    sym = r[symkey]
                hit += int(sym == r["truth"])
            line += "  %-10s %.4f" % (signame, hit / n)
            rows.append(dict(decode=label, signal=signame, budget=budget,
                             accuracy=round(hit / n, 4)))
        print(line)

p = OUT / "a5_massive_constrained.csv"
with open(p, "w", newline="", encoding="utf-8") as fh:
    w = csv.DictWriter(fh, fieldnames=["decode", "signal", "budget", "accuracy"])
    w.writeheader(); w.writerows(rows)
(OUT / "a5_massive_constrained_summary.json").write_text(json.dumps(dict(
    n_test=n, flat_accuracy=round(flat_acc, 4),
    factorized_free=round(free_acc, 4), factorized_constrained=round(con_acc, 4),
    illegal_rate_free=round(illegal, 4), illegal_rows=illegal_n,
    illegal_rows_fixed_by_constraint=fixed,
    delta_constrained_minus_free_pts=round(100 * (con_acc - free_acc), 2),
    retrained=True, seed=SEED), indent=2), encoding="utf-8")
print()
print("wrote %s and _summary.json" % p.relative_to(REPO))
print("=" * 80)
