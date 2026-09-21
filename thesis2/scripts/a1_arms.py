"""thesis2/scripts/a1_arms.py -- A1 arms 1, 2, 4: frozen probe, TinyBERT-4, BiLSTM.

All three train on train_v2 ONLY, select on dev_v2 ONLY, evaluate on
held_out_v3 and held_out_v4c. held_out_* is never opened before evaluation --
the loader for it is called only inside evaluate().

  --arm frozen_probe   MiniLM frozen, 3 linear heads (verb/target/location flat)
  --arm tinybert       TinyBERT-4 (14.35 M) fine-tuned, FACTORIZED head
  --arm bilstm         BiLSTM joint intent+slot, from scratch, no pretraining

Shared: the same emit() schema mask as every other arm, so schema validity is
1.0 by construction; the same llm/score.py for action match; the same
dev-only selection. NOTE (Part B review, S-7): the code LOADS THE ARGMAX dev
epoch (`best`), not the plateau-earliest one it prints as `chosen`; the two
can differ by one epoch. Selection is dev_v2 only either way.
"""
import argparse
import csv
import json
import math
import os
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[2]
os.chdir(REPO)
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "thesis2" / "scripts"))

from llm.score import score                                        # noqa: E402
from p2_train_factorized import (build_label_space, emit, factorize,  # noqa: E402
                                 reconstruct, IGNORE, ROWS, COLS,
                                 CONTAINERS, RUNTIMES, FACTORS)

OUT = REPO / "thesis2" / "out"
SEED = 20260820
SPACE = build_label_space()
V, T, L = SPACE["verbs"], SPACE["targets"], SPACE["locations"]
PLATEAU_TOL = 1


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def load(name):
    return [json.loads(l) for l in
            open(REPO / "data" / ("%s.jsonl" % name), encoding="utf-8") if l.strip()]


def ece(pairs, bins=10):
    if not pairs:
        return float("nan")
    t = 0.0
    for i in range(bins):
        a, b = i / bins, (i + 1) / bins
        s = [p for p in pairs if a <= p[0] < b or (i == bins - 1 and p[0] >= a)]
        if not s:
            continue
        t += (len(s) / len(pairs)) * abs(sum(c for c, _ in s) / len(s)
                                         - sum(1.0 for _, k in s if k) / len(s))
    return t


def brier(pairs):
    return (sum((c - (1.0 if k else 0.0)) ** 2 for c, k in pairs) / len(pairs)
            if pairs else float("nan"))


def evaluate(arm, predict, dataset, resident_mb, joules=""):
    """predict(command) -> (obj, {head: (conf, correct_idx_or_None)})."""
    rows = load(dataset)
    lat, recs = [], []
    hits = dict(action=0, joint=0, verb=0, target=0, location=0)
    den = dict(action=0, joint=0, verb=0, target=0, location=0)
    fh = {f: [0, 0] for f in FACTORS}
    cal = {}
    for ci, r in enumerate(rows):
        truth = dict(r["expected"])
        truth.setdefault("modifiers", [])
        t0 = time.perf_counter()
        obj, conf = predict(r["command"])
        lat.append((time.perf_counter() - t0) * 1000)
        sm = bool(score(obj, truth).strict_match)
        den["action"] += 1; hits["action"] += int(sm)
        den["verb"] += 1
        hits["verb"] += int(obj.get("action") == truth.get("action"))
        if truth.get("target"):
            den["target"] += 1
            hits["target"] += int(obj.get("target") == truth.get("target"))
        if truth.get("location"):
            den["location"] += 1
            hits["location"] += int(obj.get("location") == truth.get("location"))
        if truth.get("target") and truth.get("location"):
            den["joint"] += 1
            hits["joint"] += int(all(obj.get(k) == truth.get(k)
                                     for k in ("action", "target", "location")))
        tf = dict(zip(FACTORS, factorize(truth.get("location"))))
        pf = dict(zip(FACTORS, factorize(obj.get("location"))
                      if obj.get("location") else (IGNORE,) * 5))
        for f in FACTORS:
            if tf[f] != IGNORE:
                fh[f][1] += 1
                fh[f][0] += int(pf[f] == tf[f])
        for h, (c, ok) in conf.items():
            cal.setdefault(h, []).append((c, ok))
        recs.append(dict(model=arm, strategy="arm", dataset=dataset,
                         command_id=ci, command=r["command"], run_index=0,
                         is_warmup=False, stage="validated", ok=True,
                         strict_match=sm, pred_json=json.dumps(obj),
                         truth_json=json.dumps(truth),
                         latency_s=round(lat[-1] / 1000.0, 8)))
    with open(OUT / ("a1_%s_%s.csv" % (arm, dataset)), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(recs[0])); w.writeheader(); w.writerows(recs)
    lat.sort()
    q = lambda p: lat[min(len(lat) - 1, int(round(p * (len(lat) - 1))))]
    rej = [r for r in rows if r["expected"]["action"] == "reject"]
    rejhit = sum(1 for r in rej if predict(r["command"])[0]["action"] == "reject")
    s = dict(arm=arm, dataset=dataset, n_commands=den["action"],
             action_match=round(hits["action"] / den["action"], 4),
             joint_correct=round(hits["joint"] / den["joint"], 4),
             verb_acc=round(hits["verb"] / den["verb"], 4),
             target_acc=round(hits["target"] / den["target"], 4),
             location_acc=round(hits["location"] / den["location"], 4),
             n_location=den["location"], n_joint=den["joint"],
             schema_validity=1.0,
             reject_recall=round(rejhit / len(rej), 4) if rej else "",
             n_reject=len(rej),
             latency_p50_ms=round(q(.50), 3), latency_p95_ms=round(q(.95), 3),
             latency_p99_ms=round(q(.99), 3), joules_per_command=joules,
             resident_mb=resident_mb, escalation_rate=0)
    for f in FACTORS:
        s["factor_%s" % f] = (round(fh[f][0] / fh[f][1], 4) if fh[f][1] else "")
        s["n_%s" % f] = fh[f][1]
    for h, p in cal.items():
        s["ECE_%s" % h] = round(ece(p), 4)
        s["Brier_%s" % h] = round(brier(p), 4)
    print("  %-16s %-14s action %.4f  joint %.4f  loc %.4f  row %s  p50 %.1f ms"
          % (arm, dataset, s["action_match"], s["joint_correct"],
             s["location_acc"], s["factor_row"], s["latency_p50_ms"]))
    return s


# ------------------------------------------------------------- arm 1 --------
def run_frozen_probe():
    """MiniLM FROZEN, three linear heads on the pooled embedding."""
    from sentence_transformers import SentenceTransformer
    set_seed(SEED)
    enc = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    for p in enc.parameters():
        p.requires_grad = False
    tr, dv = load("train_v2"), load("dev_v2")

    def embed(rows):
        return torch.tensor(enc.encode([r["command"] for r in rows],
                                       normalize_embeddings=True,
                                       show_progress_bar=False))
    Xtr, Xdv = embed(tr), embed(dv)
    h = Xtr.shape[1]
    heads = nn.ModuleDict(dict(verb=nn.Linear(h, len(V)),
                               target=nn.Linear(h, len(T)),
                               location=nn.Linear(h, len(L))))
    ytr = {"verb": torch.tensor([V.index(r["expected"]["action"]) for r in tr]),
           "target": torch.tensor([T.index(r["expected"]["target"])
                                   if r["expected"].get("target") else IGNORE for r in tr]),
           "location": torch.tensor([L.index(r["expected"]["location"])
                                     if r["expected"].get("location") else IGNORE for r in tr])}
    opt = torch.optim.AdamW(heads.parameters(), lr=1e-2, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    best, hist = None, []
    for ep in range(1, 201):
        heads.train(); opt.zero_grad()
        loss = sum(ce(heads[k](Xtr), ytr[k]) for k in heads)
        loss.backward(); opt.step()
        if ep % 10:
            continue
        heads.eval()
        with torch.no_grad():
            hit = 0
            for i, r in enumerate(dv):
                o = emit(int(heads["verb"](Xdv[i:i+1]).argmax()),
                         int(heads["target"](Xdv[i:i+1]).argmax()),
                         L[int(heads["location"](Xdv[i:i+1]).argmax())], SPACE)
                t = dict(r["expected"]); t.setdefault("modifiers", [])
                hit += int(o == t)
        hist.append((ep, hit))
        if best is None or hit > best[1]:
            best = (ep, hit, {k: {n: p.clone() for n, p in heads[k].state_dict().items()}
                              for k in heads})
    bh = max(h2 for _e, h2 in hist)
    chosen = min([(e, h2) for e, h2 in hist if h2 >= bh - PLATEAU_TOL])
    print("  dev plateau %d/%d, plateau-earliest epoch %d; LOADED argmax epoch %d"
          % (bh, len(dv), chosen[0], best[0]))
    for k in heads:
        heads[k].load_state_dict(best[2][k])
    heads.eval()

    def predict(cmd):
        with torch.no_grad():
            x = torch.tensor(enc.encode([cmd], normalize_embeddings=True,
                                        show_progress_bar=False))
            lg = {k: heads[k](x) for k in heads}
            pr = {k: torch.softmax(lg[k], -1)[0] for k in lg}
            iv, it, il = (int(pr["verb"].argmax()), int(pr["target"].argmax()),
                          int(pr["location"].argmax()))
        obj = emit(iv, it, L[il], SPACE)
        return obj, {"verb": (float(pr["verb"].max()), True),
                     "target": (float(pr["target"].max()), True),
                     "location": (float(pr["location"].max()), True)}
    return predict, 87.0


# ------------------------------------------------------------- arm 2 --------
class TinyFactorized(nn.Module):
    def __init__(self, base="huawei-noah/TinyBERT_General_4L_312D"):
        super().__init__()
        from transformers import AutoModel
        self.enc = AutoModel.from_pretrained(base)
        h = self.enc.config.hidden_size
        self.do = nn.Dropout(0.1)
        self.verb = nn.Linear(h, len(V)); self.target = nn.Linear(h, len(T))
        self.loc_type = nn.Linear(h, 3); self.row = nn.Linear(h, 3)
        self.col = nn.Linear(h, 3); self.container_which = nn.Linear(h, 3)
        self.runtime_which = nn.Linear(h, 2)

    def forward(self, ids, mask):
        c = self.do(self.enc(input_ids=ids, attention_mask=mask).last_hidden_state[:, 0])
        return {k: getattr(self, k)(c) for k in
                ["verb", "target"] + FACTORS}


def run_tinybert():
    from transformers import AutoTokenizer
    set_seed(SEED)
    base = "huawei-noah/TinyBERT_General_4L_312D"
    tok = AutoTokenizer.from_pretrained(base)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr, dv = load("train_v2"), load("dev_v2")

    def enc_rows(rows):
        e = tok([r["command"] for r in rows], truncation=True, max_length=64,
                padding="max_length", return_tensors="pt")
        lab = {"verb": torch.tensor([V.index(r["expected"]["action"]) for r in rows]),
               "target": torch.tensor([T.index(r["expected"]["target"])
                                       if r["expected"].get("target") else IGNORE
                                       for r in rows])}
        fs = [factorize(r["expected"].get("location")) for r in rows]
        for i, f in enumerate(FACTORS):
            lab[f] = torch.tensor([x[i] for x in fs])
        return e, lab
    etr, ltr = enc_rows(tr)
    m = TinyFactorized(base).to(dev)
    print("  TinyBERT parameters: %.2f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
    opt = torch.optim.AdamW(m.parameters(), lr=5e-5, weight_decay=0.01)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    n, bs, EP = len(tr), 16, 30
    hist, best = [], None
    for ep in range(1, EP + 1):
        m.train()
        perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lg = m(etr["input_ids"][idx].to(dev), etr["attention_mask"][idx].to(dev))
            loss = 0
            for k in lg:
                lab = ltr[k][idx].to(dev)
                if (lab != IGNORE).any():
                    loss = loss + ce(lg[k], lab)
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); hit = 0
        with torch.no_grad():
            for r in dv:
                e = tok(r["command"], truncation=True, max_length=64,
                        padding="max_length", return_tensors="pt").to(dev)
                lg = m(e["input_ids"], e["attention_mask"])
                ix = {k: int(lg[k].argmax(-1)) for k in lg}
                o = emit(ix["verb"], ix["target"],
                         reconstruct(ix["loc_type"], ix["row"], ix["col"],
                                     ix["container_which"], ix["runtime_which"]), SPACE)
                t = dict(r["expected"]); t.setdefault("modifiers", [])
                hit += int(o == t)
        hist.append((ep, hit))
        if best is None or hit > best[1]:
            best = (ep, hit, {k: v.clone() for k, v in m.state_dict().items()})
    bh = max(h for _e, h in hist)
    chosen = min([(e, h) for e, h in hist if h >= bh - PLATEAU_TOL])
    print("  dev plateau %d/%d, plateau-earliest epoch %d; LOADED argmax epoch %d"
          % (bh, len(dv), chosen[0], best[0]))
    m.load_state_dict(best[2]); m.eval()

    def predict(cmd):
        with torch.no_grad():
            e = tok(cmd, truncation=True, max_length=64, padding="max_length",
                    return_tensors="pt").to(dev)
            lg = m(e["input_ids"], e["attention_mask"])
            pr = {k: torch.softmax(lg[k], -1)[0] for k in lg}
            ix = {k: int(pr[k].argmax()) for k in pr}
        obj = emit(ix["verb"], ix["target"],
                   reconstruct(ix["loc_type"], ix["row"], ix["col"],
                               ix["container_which"], ix["runtime_which"]), SPACE)
        return obj, {k: (float(pr[k].max()), True) for k in pr}
    return predict, round(sum(p.numel() for p in m.parameters()) * 4 / 1e6, 1)


# ------------------------------------------------------------- arm 4 --------
class BiLSTMJoint(nn.Module):
    """From scratch: embedding + BiLSTM + 3 heads. No pretrained weights."""

    def __init__(self, vocab, emb=128, hid=192):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.lstm = nn.LSTM(emb, hid, batch_first=True, bidirectional=True)
        self.do = nn.Dropout(0.2)
        self.verb = nn.Linear(2 * hid, len(V)); self.target = nn.Linear(2 * hid, len(T))
        self.location = nn.Linear(2 * hid, len(L))

    def forward(self, ids):
        x, _ = self.lstm(self.emb(ids))
        pooled = self.do(x.max(dim=1).values)
        return dict(verb=self.verb(pooled), target=self.target(pooled),
                    location=self.location(pooled))


def run_bilstm():
    set_seed(SEED)
    tr, dv = load("train_v2"), load("dev_v2")
    # vocabulary from TRAIN ONLY -- an OOV word at eval maps to <unk>
    vocab = {"<pad>": 0, "<unk>": 1}
    for r in tr:
        for w in r["command"].lower().split():
            vocab.setdefault(w, len(vocab))
    print("  vocabulary built from train_v2 only: %d types" % len(vocab))
    MAXL = 24

    def ids_of(cmd):
        t = [vocab.get(w, 1) for w in cmd.lower().split()][:MAXL]
        return t + [0] * (MAXL - len(t))
    Xtr = torch.tensor([ids_of(r["command"]) for r in tr])
    ytr = {"verb": torch.tensor([V.index(r["expected"]["action"]) for r in tr]),
           "target": torch.tensor([T.index(r["expected"]["target"])
                                   if r["expected"].get("target") else IGNORE for r in tr]),
           "location": torch.tensor([L.index(r["expected"]["location"])
                                     if r["expected"].get("location") else IGNORE for r in tr])}
    m = BiLSTMJoint(len(vocab))
    print("  BiLSTM parameters: %.2f M" % (sum(p.numel() for p in m.parameters()) / 1e6))
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, weight_decay=1e-4)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)
    hist, best, n, bs = [], None, len(tr), 16
    for ep in range(1, 61):
        m.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            lg = m(Xtr[idx])
            loss = sum(ce(lg[k], ytr[k][idx]) for k in lg
                       if (ytr[k][idx] != IGNORE).any())
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval(); hit = 0
        with torch.no_grad():
            for r in dv:
                lg = m(torch.tensor([ids_of(r["command"])]))
                o = emit(int(lg["verb"].argmax()), int(lg["target"].argmax()),
                         L[int(lg["location"].argmax())], SPACE)
                t = dict(r["expected"]); t.setdefault("modifiers", [])
                hit += int(o == t)
        hist.append((ep, hit))
        if best is None or hit > best[1]:
            best = (ep, hit, {k: v.clone() for k, v in m.state_dict().items()})
    bh = max(h for _e, h in hist)
    chosen = min([(e, h) for e, h in hist if h >= bh - PLATEAU_TOL])
    print("  dev plateau %d/%d, plateau-earliest epoch %d; LOADED argmax epoch %d"
          % (bh, len(dv), chosen[0], best[0]))
    m.load_state_dict(best[2]); m.eval()

    def predict(cmd):
        with torch.no_grad():
            lg = m(torch.tensor([ids_of(cmd)]))
            pr = {k: torch.softmax(lg[k], -1)[0] for k in lg}
        obj = emit(int(pr["verb"].argmax()), int(pr["target"].argmax()),
                   L[int(pr["location"].argmax())], SPACE)
        return obj, {k: (float(pr[k].max()), True) for k in pr}
    return predict, round(sum(p.numel() for p in m.parameters()) * 4 / 1e6, 1)


ARMS = {"frozen_probe": run_frozen_probe, "tinybert": run_tinybert,
        "bilstm": run_bilstm}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=sorted(ARMS))
    a = ap.parse_args()
    print("=" * 78)
    print("A1 ARM: %s   (train_v2 only, dev_v2 selection, held_out never seen "
          "before evaluate)" % a.arm)
    print("=" * 78)
    predict, mb = ARMS[a.arm]()
    summ = [evaluate(a.arm, predict, ds, mb)
            for ds in ("held_out_v3", "held_out_v4c")]
    p = OUT / ("a1_%s_summary.csv" % a.arm)
    with open(p, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0]))
        w.writeheader(); w.writerows(summ)
    print("  wrote %s" % p.relative_to(REPO))
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
