"""thesis2/scripts/p2_train_factorized.py -- PHASE 2.1, factorized location head.

Replaces the flat 14-way location head with five factor heads on the SAME
DistilBERT backbone. Verb and target heads are unchanged.

    loc_type          3-way   grid / container / runtime      (all location rows)
    row               3-way   A / B / C                       (grid rows only)
    col               3-way   1 / 2 / 3                       (grid rows only)
    container_which   3-way   box_1 / box_2 / box_3           (container rows only)
    runtime_which     2-way   max_reach / nearest_free_cell   (runtime rows only)

PARAMETER COUNT IS IDENTICAL TO THE FLAT HEAD. 3+3+3+3+2 = 14 output units,
the same 14 the flat head has. Factorization here buys per-factor confidence,
not capacity -- so a difference in accuracy cannot be attributed to size.

HYPERPARAMETERS ARE COPIED FROM ft/train_classifier.py AND NOT SWEPT. Same
seed, lr, batch size, epochs, warmup, dropout, max_len, weight decay, and the
same checkpoint rule (EARLIEST epoch within PLATEAU_TOL dev rows of the best,
not argmax). 57 dev rows cannot resolve a hyperparameter search. Deviations
are listed under DEVIATIONS below.

DEVIATIONS from ft/train_classifier.py, all forced by factorization:
  1. Five location heads instead of one; five masked CE terms instead of one.
     The loss is therefore a sum of 7 terms, not 3. No term is reweighted --
     plain sum, matching the original's plain sum over 3.
  2. Dev selection uses reconstructed-symbol exact match, so the metric is
     comparable to the original's, but the reconstruction step is new.
  3. Writes to thesis2/models/, never ft/runs/. The existing tree is read-only.

TRAINS ON train_v2 ONLY. SELECTS ON dev_v2 ONLY. held_out_v3 is never opened
by this script.
"""
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ---- copied verbatim from ft/train_classifier.py, deliberately not swept ----
SEED = 20260820
BASE_MODEL = "distilbert-base-uncased"
MAX_LEN = 64
BATCH_SIZE = 16
EPOCHS = 15
LR = 3e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 10
DROPOUT = 0.1
PLATEAU_TOL = 1

REPO = Path(__file__).resolve().parents[2]
SCHEMA = REPO / "schema" / "schema_v2.json"
TRAIN_FILE = REPO / "data" / "train_v2.jsonl"
DEV_FILE = REPO / "data" / "dev_v2.jsonl"
OUT_DIR = REPO / "thesis2" / "models" / "distilbert_factorized"

IGNORE = -100

ROWS = ["A", "B", "C"]
COLS = ["1", "2", "3"]
CONTAINERS = ["box_1", "box_2", "box_3"]
RUNTIMES = ["max_reach", "nearest_free_cell"]
LOC_TYPES = ["grid", "container", "runtime"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_label_space():
    sch = json.loads(SCHEMA.read_text(encoding="utf-8"))
    return {
        "schema_version": sch.get("schema_version"),
        "verbs": list(sch["verbs"].keys()),
        "targets": list(sch["targets"]),
        "locations": list(sch["locations"]),
        "verb_fields": {v: list(sch["verbs"][v]["required"]) for v in sch["verbs"]},
        "loc_types": LOC_TYPES, "rows": ROWS, "cols": COLS,
        "containers": CONTAINERS, "runtimes": RUNTIMES,
    }


# --------------------------------------------------------------- factors ----
def factorize(loc):
    """14-class symbol -> (loc_type, row, col, container_which, runtime_which).

    Unsupervised factors come back as IGNORE so masked CE skips them.
    """
    if loc is None:
        return IGNORE, IGNORE, IGNORE, IGNORE, IGNORE
    if loc in CONTAINERS:
        return 1, IGNORE, IGNORE, CONTAINERS.index(loc), IGNORE
    if loc in RUNTIMES:
        return 2, IGNORE, IGNORE, IGNORE, RUNTIMES.index(loc)
    if loc.startswith("grid_") and len(loc) == 7:
        return 0, ROWS.index(loc[5]), COLS.index(loc[6]), IGNORE, IGNORE
    raise ValueError("unfactorizable location: %r" % (loc,))


def reconstruct(lt_i, row_i, col_i, cw_i, rw_i):
    """Factors -> the 14-class symbol. Inverse of factorize()."""
    if lt_i == 0:
        return "grid_%s%s" % (ROWS[row_i], COLS[col_i])
    if lt_i == 1:
        return CONTAINERS[cw_i]
    return RUNTIMES[rw_i]


def check_roundtrip(space):
    """factorize -> reconstruct must be the identity on all 14 symbols."""
    bad = []
    for loc in space["locations"]:
        lt, r, c, cw, rw = factorize(loc)
        back = reconstruct(lt, 0 if r == IGNORE else r, 0 if c == IGNORE else c,
                           0 if cw == IGNORE else cw, 0 if rw == IGNORE else rw)
        if back != loc:
            bad.append((loc, back))
    return bad


# --------------------------------------------------------------- dataset ----
class CommandDataset(Dataset):
    def __init__(self, rows, tok, space):
        self.rows = rows
        self.tok = tok
        self.v2i = {s: i for i, s in enumerate(space["verbs"])}
        self.t2i = {s: i for i, s in enumerate(space["targets"])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        exp = row["expected"]
        enc = self.tok(row["command"], truncation=True, max_length=MAX_LEN,
                       padding="max_length", return_tensors="pt")
        tgt = exp.get("target")
        lt, rr, cc, cw, rw = factorize(exp.get("location"))
        L = lambda v: torch.tensor(v, dtype=torch.long)
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "verb": L(self.v2i[exp["action"]]),
            "target": L(self.t2i[tgt] if tgt is not None else IGNORE),
            "loc_type": L(lt), "row": L(rr), "col": L(cc),
            "container_which": L(cw), "runtime_which": L(rw),
        }


# ----------------------------------------------------------------- model ----
FACTORS = ["loc_type", "row", "col", "container_which", "runtime_which"]


class FactorizedClassifier(nn.Module):
    """One shared DistilBERT encoder; verb, target and FIVE location factors."""

    def __init__(self, space, base_model=BASE_MODEL, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        cfg = self.encoder.config
        h = getattr(cfg, "dim", None) or getattr(cfg, "hidden_size", None)
        if h is None:
            raise RuntimeError("cannot find hidden size on %s" % type(cfg).__name__)
        self.dropout = nn.Dropout(dropout)
        self.verb_head = nn.Linear(h, len(space["verbs"]))
        self.target_head = nn.Linear(h, len(space["targets"]))
        self.loc_type_head = nn.Linear(h, len(LOC_TYPES))
        self.row_head = nn.Linear(h, len(ROWS))
        self.col_head = nn.Linear(h, len(COLS))
        self.container_which_head = nn.Linear(h, len(CONTAINERS))
        self.runtime_which_head = nn.Linear(h, len(RUNTIMES))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])
        return {
            "verb": self.verb_head(cls),
            "target": self.target_head(cls),
            "loc_type": self.loc_type_head(cls),
            "row": self.row_head(cls),
            "col": self.col_head(cls),
            "container_which": self.container_which_head(cls),
            "runtime_which": self.runtime_which_head(cls),
        }

    def heads_from_cls(self, cls):
        """Run ONLY the heads on a cached [CLS] vector.

        Phase 4 needs this: escalating one factor must not re-run the encoder
        if the head alone can be re-run. Kept here so the two phases cannot
        disagree about what a head is.
        """
        return {
            "verb": self.verb_head(cls), "target": self.target_head(cls),
            "loc_type": self.loc_type_head(cls), "row": self.row_head(cls),
            "col": self.col_head(cls),
            "container_which": self.container_which_head(cls),
            "runtime_which": self.runtime_which_head(cls),
        }


def masked_ce(logits, labels, loss_fn):
    """CE that returns 0.0 rather than NaN when no row in the batch is
    supervised for this head. With five conditionally-supervised heads and
    batch 16 this fires often -- a batch with no container row is common."""
    if (labels != IGNORE).any():
        return loss_fn(logits, labels)
    return logits.sum() * 0.0


def emit(verb_idx, target_idx, loc_symbol, space):
    """Build the action object from the schema's required-field list.

    Identical in shape and guarantee to ft/train_classifier.py's emit(): the
    location is DISCARDED unless the predicted verb requires it.
    """
    verb = space["verbs"][verb_idx]
    obj = {"action": verb}
    req = space["verb_fields"][verb]
    if "target" in req:
        obj["target"] = space["targets"][target_idx]
    if "location" in req:
        obj["location"] = loc_symbol
    obj["modifiers"] = []
    return obj


def predict_symbol(logits):
    """argmax each factor, then reconstruct. Returns (symbol, per-factor idx)."""
    idx = {k: int(logits[k].argmax(-1).item()) for k in FACTORS}
    return reconstruct(idx["loc_type"], idx["row"], idx["col"],
                       idx["container_which"], idx["runtime_which"]), idx


@torch.no_grad()
def dev_exact_match(model, rows, tok, space, device):
    model.eval()
    hits = 0
    for row in rows:
        enc = tok(row["command"], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt").to(device)
        lg = model(enc["input_ids"], enc["attention_mask"])
        sym, _ = predict_symbol(lg)
        obj = emit(int(lg["verb"].argmax(-1).item()),
                   int(lg["target"].argmax(-1).item()), sym, space)
        truth = dict(row["expected"])
        truth.setdefault("modifiers", [])
        if obj == truth:
            hits += 1
    model.train()
    return hits / len(rows), hits


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    space = build_label_space()

    bad = check_roundtrip(space)
    print("=" * 72)
    print("PHASE 2.1 -- factorized location head")
    print("=" * 72)
    print("  factorize/reconstruct round-trip over all %d symbols: %s"
          % (len(space["locations"]), "EXACT" if not bad else "FAILED %s" % bad))
    if bad:
        return 1

    train_rows = load_jsonl(TRAIN_FILE)
    dev_rows = load_jsonl(DEV_FILE)
    print("  device        = %s" % device)
    print("  torch         = %s" % torch.__version__)
    print("  base model    = %s" % BASE_MODEL)
    print("  train / dev   = %d / %d   (held_out_v3 NEVER opened here)"
          % (len(train_rows), len(dev_rows)))
    print("  seed=%d lr=%s bs=%d epochs=%d warmup=%d max_len=%d dropout=%s"
          % (SEED, LR, BATCH_SIZE, EPOCHS, WARMUP_STEPS, MAX_LEN, DROPOUT))

    # supervision counts per factor, so the masked-loss design is auditable
    counts = {f: 0 for f in FACTORS}
    for r in train_rows:
        lt, rr, cc, cw, rw = factorize(r["expected"].get("location"))
        for f, v in zip(FACTORS, (lt, rr, cc, cw, rw)):
            if v != IGNORE:
                counts[f] += 1
    print("  supervised train rows per factor: %s" % counts)

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = CommandDataset(train_rows, tok, space)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))

    model = FactorizedClassifier(space).to(device)
    nparam = sum(p.numel() for p in model.parameters())
    headp = sum(p.numel() for n, p in model.named_parameters() if "head" in n)
    print("  parameters    = %.1f M  (heads %d)" % (nparam / 1e6, headp))

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * math.ceil(len(train_ds) / BATCH_SIZE)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return max(0.0, 1.0 - prog)

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "label_maps.json").write_text(json.dumps(space, indent=2),
                                             encoding="utf-8")
    print("  total steps   = %d" % total_steps)
    print("  output        = %s\n" % OUT_DIR)
    print("%5s %11s %10s %8s %7s" % ("epoch", "train_loss", "dev_exact", "hits", "secs"))
    print("-" * 46)

    history = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        running, nb = 0.0, 0
        for batch in train_dl:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            lg = model(ids, mask)
            loss = loss_fn(lg["verb"], batch["verb"].to(device))
            loss = loss + masked_ce(lg["target"], batch["target"].to(device), loss_fn)
            for f in FACTORS:
                loss = loss + masked_ce(lg[f], batch[f].to(device), loss_fn)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            nb += 1
        acc, hits = dev_exact_match(model, dev_rows, tok, space, device)
        dt = time.time() - t0
        history.append({"epoch": epoch, "train_loss": running / nb,
                        "dev_exact": acc, "dev_hits": hits, "secs": round(dt, 1)})
        print("%5d %11.4f %10.4f %4d/%-3d %7.1f"
              % (epoch, running / nb, acc, hits, len(dev_rows), dt))
        ck = OUT_DIR / ("epoch_%02d" % epoch)
        ck.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ck / "model.pt")
        tok.save_pretrained(ck)

    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    best = max(h["dev_hits"] for h in history)
    plateau = [h for h in history if h["dev_hits"] >= best - PLATEAU_TOL]
    chosen = min(plateau, key=lambda h: h["epoch"])
    argmax = max(history, key=lambda h: (h["dev_hits"], -h["epoch"]))
    print("\n" + "=" * 72)
    print("CHECKPOINT SELECTION -- earliest epoch at the dev plateau, not argmax")
    print("  best dev hits : %d/%d" % (best, len(dev_rows)))
    print("  plateau       : epochs %s" % [h["epoch"] for h in plateau])
    print("  SELECTED      : epoch %d  (%d/%d = %.1f%%)"
          % (chosen["epoch"], chosen["dev_hits"], len(dev_rows),
             100 * chosen["dev_exact"]))
    print("  argmax would  : epoch %d" % argmax["epoch"])
    (OUT_DIR / "selected.json").write_text(json.dumps(
        {"selected_epoch": chosen["epoch"], "dev_hits": chosen["dev_hits"],
         "dev_total": len(dev_rows), "rule": "earliest epoch within %d dev row "
         "of best, matching ft/train_classifier.py" % PLATEAU_TOL,
         "plateau_epochs": [h["epoch"] for h in plateau],
         "argmax_epoch": argmax["epoch"]}, indent=2), encoding="utf-8")

    # keep the selected checkpoint, drop the rest (265 MB each, 15 of them)
    kept = OUT_DIR / ("epoch_%02d" % chosen["epoch"])
    freed = 0
    for d in sorted(OUT_DIR.glob("epoch_*")):
        if d != kept:
            freed += sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
            shutil.rmtree(d)
    print("  kept %s, removed the other checkpoints (%.0f MB freed)"
          % (kept.name, freed / 1e6))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
