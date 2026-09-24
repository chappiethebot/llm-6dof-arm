"""thesis2/final/s0_train_textcnn.py -- STAGE 0: TextCNN arm (train_v2 only).

Tokeniser and vocabulary are EXACTLY the BiLSTM's in thesis2/scripts/a1_arms.py:
lowercase whitespace split, vocabulary built from train_v2 only, OOV -> <unk>,
truncate/pad to 24. Nothing from held_out_v3 or held_out_v4c is opened here, and
the vocabulary guard at startup refuses to run if any held-out row word reached
the vocabulary.

  embedding 128 (from scratch) -> Conv1d widths 3/4/5, 100 filters each, ReLU,
  max-over-time pool -> concat 300 -> dropout 0.5 -> three heads
  heads: verb(7), target(3), location(14)
  loss = sum of three cross-entropies, masked exactly as the flat classifier
         masks them (ignore_index=-100 where the verb does not require the field)

  Adam lr 1e-3, batch 32, 30 epochs, seed 20260827.   No sweeps, no flags.
  Selection: ARGMAX dev_v2 action match (stated rule for this arm), persisted to
  thesis2/out/final/selection_textcnn.json.

Run (thesis2gpu):  python thesis2/final/s0_train_textcnn.py
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
MAXL = 24
EMB = 128
WIDTHS = (3, 4, 5)
NFILT = 100
DROPOUT = 0.5
LR = 1e-3
BATCH = 32
EPOCHS = 30
ARM = "textcnn"


class TextCNN(nn.Module):
    def __init__(self, vocab, n_verb, n_target, n_loc):
        super().__init__()
        self.emb = nn.Embedding(vocab, EMB, padding_idx=0)
        self.convs = nn.ModuleList([nn.Conv1d(EMB, NFILT, w, padding=w - 1) for w in WIDTHS])
        self.drop = nn.Dropout(DROPOUT)
        h = NFILT * len(WIDTHS)
        self.verb = nn.Linear(h, n_verb)
        self.target = nn.Linear(h, n_target)
        self.location = nn.Linear(h, n_loc)

    def forward(self, ids):
        x = self.emb(ids).transpose(1, 2)                  # (B, EMB, T)
        feats = [torch.relu(c(x)).max(dim=2).values for c in self.convs]
        h = self.drop(torch.cat(feats, dim=1))
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
    C.start_log("s0_train_textcnn")
    C.env_banner()

    random.seed(C.SEED); np.random.seed(C.SEED); torch.manual_seed(C.SEED)
    torch.cuda.manual_seed_all(C.SEED)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    tr, dv = C.load_split(C.TRAIN), C.load_split(C.DEV)
    if a.smoke:
        tr, dv = tr[:a.smoke], dv[:a.smoke]
        print("  SMOKE MODE: %d train / %d dev commands, 1 epoch" % (len(tr), len(dv)))

    # ---- vocabulary: train_v2 only, identical construction to the BiLSTM
    vocab = {"<pad>": 0, "<unk>": 1}
    for r in tr:
        for w in r["command"].lower().split():
            vocab.setdefault(w, len(vocab))
    print("  vocabulary from %s only: %d types" % (C.TRAIN, len(vocab)))
    C.assert_vocab_from_train_only(vocab, where="the TextCNN vocabulary")

    def ids_of(cmd):
        t = [vocab.get(w, 1) for w in cmd.lower().split()][:MAXL]
        return t + [0] * (MAXL - len(t))

    def labels_of(rows):
        v = torch.tensor([C.VERBS.index(r["expected"]["action"]) for r in rows])
        t, l = [], []
        for r in rows:
            e = r["expected"]
            need = C.VERB_FIELDS[e["action"]]
            t.append(C.TARGETS.index(e["target"]) if "target" in need else IGNORE)
            l.append(C.LOCATIONS.index(e["location"]) if "location" in need else IGNORE)
        return v, torch.tensor(t), torch.tensor(l)

    X = torch.tensor([ids_of(r["command"]) for r in tr])
    yv, yt, yl = labels_of(tr)
    print("  supervised rows: verb %d  target %d  location %d"
          % (len(tr), int((yt != IGNORE).sum()), int((yl != IGNORE).sum())))

    model = TextCNN(len(vocab), len(C.VERBS), len(C.TARGETS), len(C.LOCATIONS)).to(dev)
    print("  parameters: %.3f M" % (sum(p.numel() for p in model.parameters()) / 1e6))
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    ce = nn.CrossEntropyLoss(ignore_index=IGNORE)

    @torch.no_grad()
    def dev_action_match(rows):
        model.eval()
        ids = torch.tensor([ids_of(r["command"]) for r in rows]).to(dev)
        lv, lt, ll = model(ids)
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
    ckpt = C.MODELS / "textcnn.pt"
    torch.save(dict(state_dict=sd, vocab=vocab, maxl=MAXL, arch=dict(
        emb=EMB, widths=list(WIDTHS), nfilt=NFILT, dropout=DROPOUT)), ckpt)
    rec = dict(arm=ARM, rule="argmax dev_v2 action match", selected_epoch=sel_ep,
               dev_hits=sel_hits, dev_total=len(dv),
               dev_action_match=round(sel_hits / len(dv), 4), epochs_run=n_epochs,
               seed=C.SEED, lr=LR, batch=BATCH, vocab_size=len(vocab),
               vocab_source=C.TRAIN, checkpoint=str(ckpt.relative_to(C.REPO)),
               history=hist, smoke=bool(a.smoke))
    (C.OUT / "selection_textcnn.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")

    ok = (not a.smoke) and sel_hits > 0
    C.gate(ok or bool(a.smoke), [
        "SELECTED epoch %d by argmax dev_v2 action match: %d/%d = %.4f"
        % (sel_ep, sel_hits, len(dv), sel_hits / len(dv)),
        "checkpoint: %s" % ckpt.relative_to(C.REPO),
        "selection record: %s" % (C.OUT / "selection_textcnn.json").relative_to(C.REPO),
        "vocabulary: %d types from %s only; held-out words absent" % (len(vocab), C.TRAIN),
        "SMOKE RUN -- not a trained arm" if a.smoke else "next: s0_train_mlpbow.py",
    ])


if __name__ == "__main__":
    main()
