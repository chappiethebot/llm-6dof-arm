"""thesis2/final/s0_train_mlpbow.py -- STAGE 0: MLP bag-of-words arm (train_v2 only).

Same tokeniser and vocabulary as the BiLSTM/TextCNN (lowercase whitespace split,
train_v2 only, OOV -> <unk>). Binary bag-of-words over that vocabulary:

  BoW(|V|) -> Linear 256 -> ReLU -> dropout 0.3 -> three heads
  heads: verb(7), target(3), location(14); loss = sum of three masked CEs,
  masked exactly as the flat classifier masks them.

  Adam lr 1e-3, batch 32, 30 epochs, seed 20260827.  No sweeps, no flags.
  Selection: ARGMAX dev_v2 action match -> thesis2/out/final/selection_mlpbow.json

No held-out file is opened here.

Run (thesis2gpu):  python thesis2/final/s0_train_mlpbow.py
"""
import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

import numpy as np                                                      # noqa: E402
import torch                                                            # noqa: E402
import torch.nn as nn                                                   # noqa: E402

IGNORE = -100
HIDDEN = 256
DROPOUT = 0.3
LR = 1e-3
BATCH = 32
EPOCHS = 30
ARM = "mlpbow"


class MLPBoW(nn.Module):
    def __init__(self, vocab, n_verb, n_target, n_loc):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(vocab, HIDDEN), nn.ReLU(), nn.Dropout(DROPOUT))
        self.verb = nn.Linear(HIDDEN, n_verb)
        self.target = nn.Linear(HIDDEN, n_target)
        self.location = nn.Linear(HIDDEN, n_loc)

    def forward(self, x):
        h = self.body(x)
        return self.verb(h), self.target(h), self.location(h)


def masked_ce(logits, labels, fn):
    if (labels != IGNORE).any():
        return fn(logits, labels)
    return logits.sum() * 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--smoke", type=int, default=0,
                    help="train on N train_v2 commands for 1 epoch (syntax check only)")
    a = ap.parse_args()
    C.start_log("s0_train_mlpbow")
    C.env_banner()

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    torch.cuda.manual_seed_all(C.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, dv = C.load_split(C.TRAIN), C.load_split(C.DEV)
    if a.smoke:
        tr, dv = tr[:a.smoke], dv[:a.smoke]
        print("  SMOKE MODE: %d train / %d dev commands, 1 epoch" % (len(tr), len(dv)))

    vocab = {"<pad>": 0, "<unk>": 1}
    for r in tr:
        for w in r["command"].lower().split():
            vocab.setdefault(w, len(vocab))
    print("  vocabulary from %s only: %d types" % (C.TRAIN, len(vocab)))
    C.assert_vocab_from_train_only(vocab, where="the MLP-BoW vocabulary")

    def bow_of(cmd):
        v = torch.zeros(len(vocab))
        for w in cmd.lower().split():
            v[vocab.get(w, 1)] = 1.0            # BINARY bag of words
        return v

    def labels_of(rows):
        vv = torch.tensor([C.VERBS.index(r["expected"]["action"]) for r in rows])
        t, l = [], []
        for r in rows:
            e = r["expected"]
            need = C.VERB_FIELDS[e["action"]]
            t.append(C.TARGETS.index(e["target"]) if "target" in need else IGNORE)
            l.append(C.LOCATIONS.index(e["location"]) if "location" in need else IGNORE)
        return vv, torch.tensor(t), torch.tensor(l)

    X = torch.stack([bow_of(r["command"]) for r in tr])
    yv, yt, yl = labels_of(tr)
    print("  supervised rows: verb %d  target %d  location %d"
          % (len(tr), int((yt != IGNORE).sum()), int((yl != IGNORE).sum())))

    model = MLPBoW(len(vocab), len(C.VERBS), len(C.TARGETS), len(C.LOCATIONS)).to(dev)
    print("  parameters: %.3f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    @torch.no_grad()
    def dev_action_match(rows):
        model.eval()
        x = torch.stack([bow_of(r["command"]) for r in rows]).to(dev)
        lv, lt, ll = model(x)
        hits = 0
        for i, r in enumerate(rows):
            obj = C.emit(int(lv[i].argmax()), int(lt[i].argmax()), int(ll[i].argmax()), C.SPACE)
            hits += int(bool(C.score(obj, C.truth_of(r)).strict_match))
        model.train()
        return hits / len(rows), hits

    n_epochs = 1 if a.smoke else EPOCHS
    hist, best = [], None
    print()
    print("%5s %11s %11s %8s %7s" % ("epoch", "train_loss", "dev_exact", "hits", "secs"))
    print("-" * 46)
    for ep in range(1, n_epochs + 1):
        t0 = time.time()
        perm = torch.randperm(len(tr), generator=torch.Generator().manual_seed(C.SEED + ep))
        run, nb = 0.0, 0
        for i in range(0, len(tr), BATCH):
            idx = perm[i:i + BATCH]
            lv, lt, ll = model(X[idx].to(dev))
            loss = (ce(lv, yv[idx].to(dev))
                    + masked_ce(lt, yt[idx].to(dev), ce)
                    + masked_ce(ll, yl[idx].to(dev), ce))
            opt.zero_grad(); loss.backward(); opt.step()
            run += float(loss); nb += 1
        acc, hits = dev_action_match(dv)
        hist.append(dict(epoch=ep, train_loss=run / nb, dev_action_match=acc, dev_hits=hits))
        print("%5d %11.4f %11.4f %4d/%-3d %7.1f" % (ep, run / nb, acc, hits, len(dv), time.time() - t0))
        if best is None or hits > best[1]:
            best = (ep, hits, {k: v.detach().clone() for k, v in model.state_dict().items()})

    sel_ep, sel_hits, sd = best
    ckpt = C.MODELS / "mlpbow.pt"
    torch.save(dict(state_dict=sd, vocab=vocab,
                    arch=dict(hidden=HIDDEN, dropout=DROPOUT)), ckpt)
    rec = dict(arm=ARM, rule="argmax dev_v2 action match", selected_epoch=sel_ep,
               dev_hits=sel_hits, dev_total=len(dv),
               dev_action_match=round(sel_hits / len(dv), 4), epochs_run=n_epochs,
               seed=C.SEED, lr=LR, batch=BATCH, vocab_size=len(vocab),
               vocab_source=C.TRAIN, checkpoint=str(ckpt.relative_to(C.REPO)),
               history=hist, smoke=bool(a.smoke))
    (C.OUT / "selection_mlpbow.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    C.gate(bool(a.smoke) or sel_hits > 0, [
        "SELECTED epoch %d by argmax dev_v2 action match: %d/%d = %.4f"
        % (sel_ep, sel_hits, len(dv), sel_hits / len(dv)),
        "checkpoint: %s" % ckpt.relative_to(C.REPO),
        "selection record: %s" % (C.OUT / "selection_mlpbow.json").relative_to(C.REPO),
        "vocabulary: %d types from %s only; held-out words absent" % (len(vocab), C.TRAIN),
        "SMOKE RUN -- not a trained arm" if a.smoke else "next: s0_salsa_full.py --check",
    ])


if __name__ == "__main__":
    main()
