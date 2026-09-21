"""thesis2/scripts/p7_massive.py -- PHASE 7, generalization beyond the robot arm.

MASSIVE (en-US, via mteb/amazon_massive_intent). 11,514 train / 2,033 dev /
2,974 test. Sixty intents, and every intent label is literally
`<scenario>_<action>` -- a COMPOSITE label with a natural two-factor
decomposition, exactly the structure `grid_<row><col>` has.

  flat         60-way intent head
  factorized   scenario head (18-way) x action head (K-way), symbol
               reconstructed as scenario_action

WHY THIS IS THE DECISIVE TEST. MASSIVE's train/dev/test are a random split of
one distribution. There is NO vocabulary disjunction. `held_out_v3` has a total
row-vocabulary disjunction from `train_v2`, which is what drove the Phase 2
AUROC to 1.000. If the per-factor router signal still beats the flat single
confidence here, the result is not an artifact of that disjunction.

Escalation target is an ORACLE. The question Phase 7 asks is about the ROUTER
SIGNAL, not about who answers; using an oracle isolates the signal and costs
nothing. Both signals escalate the same factor to the same oracle.
"""
import argparse
import collections
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
OUT = REPO / "thesis2" / "out"
MODELS = REPO / "thesis2" / "models"

SEED = 20260820
BASE = "distilbert-base-uncased"
MAX_LEN = 48
BATCH = 32
EPOCHS = 4
LR = 5e-5
WARMUP = 100
DROPOUT = 0.1


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


class DS(Dataset):
    def __init__(self, texts, flat, scen, act, tok):
        self.t, self.f, self.s, self.a, self.tok = texts, flat, scen, act, tok

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        e = self.tok(self.t[i], truncation=True, max_length=MAX_LEN,
                     padding="max_length", return_tensors="pt")
        return dict(input_ids=e["input_ids"].squeeze(0),
                    attention_mask=e["attention_mask"].squeeze(0),
                    flat=torch.tensor(self.f[i]), scen=torch.tensor(self.s[i]),
                    act=torch.tensor(self.a[i]))


class Model(nn.Module):
    """Both heads on ONE backbone so flat and factorized share a substrate."""

    def __init__(self, n_flat, n_scen, n_act):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(BASE)
        h = getattr(self.enc.config, "dim", None) or self.enc.config.hidden_size
        self.do = nn.Dropout(DROPOUT)
        self.flat = nn.Linear(h, n_flat)
        self.scen = nn.Linear(h, n_scen)
        self.act = nn.Linear(h, n_act)

    def forward(self, ids, mask):
        cls = self.do(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return self.flat(cls), self.scen(cls), self.act(cls)


def auroc(pairs):
    pos = [s for s, k in pairs if k]
    neg = [s for s, k in pairs if not k]
    if not pos or not neg:
        return float("nan")
    pos = np.sort(np.array(pos)); neg = np.array(neg)
    r = np.searchsorted(pos, neg, side="left")
    r2 = np.searchsorted(pos, neg, side="right")
    wins = float(np.sum(len(pos) - r2) + 0.5 * np.sum(r2 - r))
    return wins / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    from datasets import load_dataset
    from transformers import AutoTokenizer

    d = load_dataset("mteb/amazon_massive_intent", "en")
    tr, dv, te = d["train"], d["validation"], d["test"]
    intents = sorted(set(tr["label_text"]))
    scen = sorted({i.split("_")[0] for i in intents})
    acts = sorted({i.split("_", 1)[1] for i in intents})
    legal = set(intents)
    i2i = {s: k for k, s in enumerate(intents)}
    s2i = {s: k for k, s in enumerate(scen)}
    a2i = {s: k for k, s in enumerate(acts)}

    print("=" * 78)
    print("PHASE 7 -- MASSIVE en-US")
    print("=" * 78)
    print("  train %d  dev %d  test %d" % (len(tr), len(dv), len(te)))
    print("  intents %d  scenarios %d  actions %d  (legal combos %d of %d possible)"
          % (len(intents), len(scen), len(acts), len(legal), len(scen) * len(acts)))
    print("  device %s" % device)
    print("  NO vocabulary disjunction: train/dev/test are a random split of one "
          "distribution")

    tok = AutoTokenizer.from_pretrained(BASE)

    def prep(split):
        f = [i2i[x] for x in split["label_text"]]
        s = [s2i[x.split("_")[0]] for x in split["label_text"]]
        a = [a2i[x.split("_", 1)[1]] for x in split["label_text"]]
        return DS(split["text"], f, s, a, tok)

    dtr, ddv, dte = prep(tr), prep(dv), prep(te)
    dl = DataLoader(dtr, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(SEED))

    m = Model(len(intents), len(scen), len(acts)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    total = args.epochs * math.ceil(len(dtr) / BATCH)

    def lam(step):
        if step < WARMUP:
            return step / max(1, WARMUP)
        return max(0.0, 1.0 - (step - WARMUP) / max(1, total - WARMUP))

    sch = torch.optim.lr_scheduler.LambdaLR(opt, lam)
    ce = nn.CrossEntropyLoss()

    @torch.no_grad()
    def infer(ds):
        m.eval()
        out = []
        loader = DataLoader(ds, batch_size=128)
        for b in loader:
            fl, sc, ac = m(b["input_ids"].to(device), b["attention_mask"].to(device))
            pf = torch.softmax(fl, -1); ps = torch.softmax(sc, -1); pa = torch.softmax(ac, -1)
            for i in range(fl.shape[0]):
                si, ai = int(ps[i].argmax()), int(pa[i].argmax())
                sym = "%s_%s" % (scen[si], acts[ai])
                out.append(dict(
                    truth=intents[int(b["flat"][i])],
                    truth_scen=scen[int(b["scen"][i])], truth_act=acts[int(b["act"][i])],
                    flat_pred=intents[int(pf[i].argmax())], flat_conf=float(pf[i].max()),
                    scen_pred=scen[si], scen_conf=float(ps[i].max()),
                    act_pred=acts[ai], act_conf=float(pa[i].max()),
                    fact_sym=sym, fact_legal=sym in legal))
        m.train()
        return out

    print()
    print("%5s %11s %10s %10s" % ("epoch", "loss", "dev_flat", "dev_fact"))
    for ep in range(1, args.epochs + 1):
        run, nb = 0.0, 0
        for b in dl:
            fl, sc, ac = m(b["input_ids"].to(device), b["attention_mask"].to(device))
            loss = (ce(fl, b["flat"].to(device)) + ce(sc, b["scen"].to(device))
                    + ce(ac, b["act"].to(device)))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sch.step()
            run += loss.item(); nb += 1
        o = infer(ddv)
        df = sum(1 for r in o if r["flat_pred"] == r["truth"]) / len(o)
        dx = sum(1 for r in o if r["fact_sym"] == r["truth"]) / len(o)
        print("%5d %11.4f %10.4f %10.4f" % (ep, run / nb, df, dx))

    res = infer(dte)
    n = len(res)
    flat_acc = sum(1 for r in res if r["flat_pred"] == r["truth"]) / n
    fact_acc = sum(1 for r in res if r["fact_sym"] == r["truth"]) / n
    scen_acc = sum(1 for r in res if r["scen_pred"] == r["truth_scen"]) / n
    act_acc = sum(1 for r in res if r["act_pred"] == r["truth_act"]) / n
    illegal = sum(1 for r in res if not r["fact_legal"]) / n

    print()
    print("=" * 78)
    print("TEST, n=%d  (MASSIVE en-US test split)" % n)
    print("  flat 60-way intent accuracy        %.4f" % flat_acc)
    print("  factorized reconstructed accuracy  %.4f   (delta %+.2f pts)"
          % (fact_acc, 100 * (fact_acc - flat_acc)))
    print("  scenario factor accuracy           %.4f" % scen_acc)
    print("  action   factor accuracy           %.4f" % act_acc)
    print("  factorized produced an ILLEGAL combo on %.2f%% of rows" % (100 * illegal))

    # ---- router signal comparison at matched budget, oracle escalation ----
    print()
    print("  ROUTER SIGNAL -- AUROC for ranking 'factorized symbol correct'")
    minfact = [(min(r["scen_conf"], r["act_conf"]), r["fact_sym"] == r["truth"]) for r in res]
    flatsig = [(r["flat_conf"], r["fact_sym"] == r["truth"]) for r in res]
    a_fact, a_flat = auroc(minfact), auroc(flatsig)
    print("    per-factor min(scen,act) confidence  AUROC %.4f" % a_fact)
    print("    flat single intent confidence        AUROC %.4f" % a_flat)

    rows = []
    print()
    print("  MATCHED BUDGET, escalate the WEAKEST factor to an oracle")
    print("  %-8s %-26s %-26s" % ("budget", "per-factor signal", "flat signal"))
    for budget in (0.05, 0.10, 0.20, 0.30, 0.40, 0.50):
        line = "  %-8.0f%%" % (100 * budget)
        for signame, scores in (("per_factor", minfact), ("flat", flatsig)):
            order = sorted(range(n), key=lambda i: scores[i][0])
            k = int(round(budget * n))
            esc = set(order[:k])
            hit = 0
            for i, r in enumerate(res):
                if i in esc:
                    # escalate ONLY the weaker factor; the other stays local
                    if r["scen_conf"] <= r["act_conf"]:
                        sym = "%s_%s" % (r["truth_scen"], r["act_pred"])
                    else:
                        sym = "%s_%s" % (r["scen_pred"], r["truth_act"])
                else:
                    sym = r["fact_sym"]
                hit += int(sym == r["truth"])
            acc = hit / n
            rows.append(dict(dataset="massive_en", signal=signame, budget=budget,
                             escalated=k, accuracy=round(acc, 4)))
            line += " %-26s" % ("acc %.4f @ %d rows" % (acc, k))
        print(line)

    p = OUT / "generalization_massive.csv"
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "signal", "budget",
                                           "escalated", "accuracy"])
        w.writeheader(); w.writerows(rows)

    summary = dict(dataset="massive_en_US", n_test=n, n_intents=len(intents),
                   n_scenarios=len(scen), n_actions=len(acts),
                   flat_accuracy=round(flat_acc, 4),
                   factorized_accuracy=round(fact_acc, 4),
                   delta_pts=round(100 * (fact_acc - flat_acc), 2),
                   scenario_acc=round(scen_acc, 4), action_acc=round(act_acc, 4),
                   illegal_combo_rate=round(illegal, 4),
                   auroc_per_factor=round(a_fact, 4), auroc_flat=round(a_flat, 4),
                   epochs=args.epochs, seed=SEED)
    (OUT / "generalization_massive_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8")
    with open(OUT / "generalization_massive_preds.csv", "w", newline="",
              encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0]))
        w.writeheader(); w.writerows(res)
    print()
    print("wrote %s, _summary.json, _preds.csv" % p.relative_to(REPO))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
