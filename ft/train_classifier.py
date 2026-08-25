#!/usr/bin/env python
"""
ft/train_classifier.py -- Rung 4. DistilBERT with three classification heads.

Maps a command string to the SAME JSON action object the LLMs emit, but by
classification rather than generation. Structural validity is guaranteed by
construction: the emitted object is BUILT from schema_v2.json's `required`
list for the predicted verb, so it cannot carry a field the validator rejects
and cannot contain a verb outside the 7-symbol enum.

DESIGN DECISIONS (pre-registered, not swept)
--------------------------------------------
1. NO `none` CLASS. Field presence is a deterministic function of the verb --
   verified zero exceptions across 468 rows (train 230, dev 57, held_out_v3 181).
   A `none` class would make part of the guarantee LEARNED, and a learned `none`
   can only ever disagree with the mask (which wins), so it is dead capacity.
   Instead: masked loss during training (ignore_index=-100) + emit-time masking
   read from the schema.

2. VERB_FIELDS IS READ FROM schema_v2.json, NEVER HARD-CODED. The validator in
   llm/pipeline.py reads the same file, so the mask provably agrees with the
   validator. One source of truth.

3. PLAIN CROSS-ENTROPY, NO CLASS WEIGHTS. The location head's imbalance is
   1.7:1 (box_2 12 rows .. grid_A1 7 rows across 132 location-bearing train
   rows). That is not an imbalance problem. Stated in methods so an examiner
   does not assume it was ignored.

4. PLAIN PYTORCH LOOP, NOT transformers.Trainer. The multi-head masked loss is
   custom, and Trainer's 5.x surface produced four API bugs in the rung-3
   session. This file touches only AutoTokenizer / AutoModel / last_hidden_state.

5. `modifiers` IS A CONSTANT `[]`, NOT A HEAD. Present on every row in all three
   splits, non-empty on none of them.

CHECKPOINT SELECTION -- same rule as rung 3
-------------------------------------------
EARLIEST epoch reaching the dev plateau, NOT argmax. Plateau = within one dev
row (1/57 = 1.75 points) of the best dev exact-match. Rationale: 57 dev rows is
+-13 points at 95% confidence, so argmax on a one-row difference is selection
noise. This script PRINTS the recommendation and saves every epoch; it does not
silently pick for you.

The dev metric here is FOR EPOCH SELECTION ONLY. It is not a reported number.
All reported numbers come from the bench harness at frozen commit 3000198.

USAGE (conda env `train`, NOT `thesis`)
---------------------------------------
    conda activate train
    cd C:\\Users\\USER\\llm-6dof-arm
    python -m ft.train_classifier

First run needs the DistilBERT weights from the hub, so make sure
HF_HUB_OFFLINE is NOT set in that shell:
    Remove-Item Env:\\HF_HUB_OFFLINE -ErrorAction SilentlyContinue

Writes ft/runs/distilbert_cls/epoch_XX/ (one dir per epoch, ~265 MB each,
~4 GB total for 15 epochs) plus label_maps.json and history.json.
"""
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel

# ---------------------------------------------------------------- config ----
# PRE-REGISTERED. Do not sweep. 57 dev rows cannot resolve a hyperparameter
# search -- any difference it finds is noise.
SEED          = 20260820          # same seed as rung 3
BASE_MODEL    = "distilbert-base-uncased"
MAX_LEN       = 64                # commands are short; longest is well under this
BATCH_SIZE    = 16
EPOCHS        = 15                # 230/16 = 15 steps/epoch -> 225 total steps
LR            = 3e-5
WEIGHT_DECAY  = 0.01
WARMUP_STEPS  = 10                # matches rung 3's warmup_steps style
DROPOUT       = 0.1
PLATEAU_TOL   = 1                 # dev rows; 1 row = 1.75 points on 57

REPO       = Path(__file__).resolve().parents[1]
SCHEMA     = REPO / "schema" / "schema_v2.json"
TRAIN_FILE = REPO / "data" / "train_v2.jsonl"
DEV_FILE   = REPO / "data" / "dev_v2.jsonl"
OUT_DIR    = REPO / "ft" / "runs" / "distilbert_cls"

IGNORE = -100   # torch's CrossEntropyLoss ignore_index


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ------------------------------------------------------------ label space ----
def build_label_space():
    """Read the enums and the per-verb required-field map from the schema.

    Order comes from the schema file, which is generated and version-controlled,
    so the index<->symbol mapping is reproducible across runs and machines."""
    sch = json.loads(SCHEMA.read_text(encoding="utf-8"))
    verbs     = list(sch["verbs"].keys())
    targets   = list(sch["targets"])
    locations = list(sch["locations"])
    verb_fields = {v: list(sch["verbs"][v]["required"]) for v in verbs}
    return {
        "schema_version": sch.get("schema_version"),
        "verbs": verbs,
        "targets": targets,
        "locations": locations,
        "verb_fields": verb_fields,
    }


# ---------------------------------------------------------------- dataset ----
class CommandDataset(Dataset):
    """One row -> (tokenised command, verb label, target label, location label).

    A label is IGNORE (-100) when the verb does not require that field. Torch's
    CrossEntropyLoss skips those positions, so the head is never trained to
    predict anything for a field that will be masked away at emit time."""

    def __init__(self, rows, tok, space):
        self.rows = rows
        self.tok = tok
        self.v2i = {s: i for i, s in enumerate(space["verbs"])}
        self.t2i = {s: i for i, s in enumerate(space["targets"])}
        self.l2i = {s: i for i, s in enumerate(space["locations"])}

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        exp = row["expected"]
        enc = self.tok(
            row["command"],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        tgt = exp.get("target")
        loc = exp.get("location")
        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "verb":     torch.tensor(self.v2i[exp["action"]], dtype=torch.long),
            "target":   torch.tensor(self.t2i[tgt] if tgt is not None else IGNORE, dtype=torch.long),
            "location": torch.tensor(self.l2i[loc] if loc is not None else IGNORE, dtype=torch.long),
        }


# ------------------------------------------------------------------ model ----
class MultiHeadClassifier(nn.Module):
    """One shared DistilBERT encoder, three linear heads on the [CLS] token."""

    def __init__(self, space, base_model=BASE_MODEL, dropout=DROPOUT):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(base_model)
        # DistilBERT's config names the hidden size `dim`; other configs (and
        # some transformers versions) expose `hidden_size`. Try both rather than
        # betting on one -- this is the only version-sensitive line in the file.
        cfg = self.encoder.config
        h = getattr(cfg, "dim", None) or getattr(cfg, "hidden_size", None)
        if h is None:
            raise RuntimeError(f"cannot find hidden size on {type(cfg).__name__}: "
                               f"{[k for k in vars(cfg) if 'dim' in k or 'size' in k]}")
        self.dropout = nn.Dropout(dropout)
        self.verb_head     = nn.Linear(h, len(space["verbs"]))
        self.target_head   = nn.Linear(h, len(space["targets"]))
        self.location_head = nn.Linear(h, len(space["locations"]))

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls = self.dropout(out.last_hidden_state[:, 0])   # [CLS] pooled
        return self.verb_head(cls), self.target_head(cls), self.location_head(cls)


def masked_ce(logits, labels, loss_fn):
    """Cross-entropy that returns 0.0 instead of NaN when a batch has no
    supervised rows for this head.

    Non-obvious: torch's CrossEntropyLoss with ignore_index averages over the
    NON-ignored positions. If every position in the batch is ignored -- e.g. a
    batch that happens to be all `stop`/`reject`, which have no target and no
    location -- the denominator is zero and the loss is NaN. One NaN poisons
    every gradient in the model. With 20 stop + 18 reject rows and batch 16
    this is not hypothetical."""
    if (labels != IGNORE).any():
        return loss_fn(logits, labels)
    return logits.sum() * 0.0    # keeps the graph connected, contributes nothing


# ------------------------------------------------------------------ emit -----
def emit(verb_idx, target_idx, location_idx, space):
    """Build the action object from the SCHEMA's required-field list.

    This is where the structural guarantee lives. The target and location
    predictions are DISCARDED unless the predicted verb requires them, so a
    `stop` can never carry a target no matter what the target head said."""
    verb = space["verbs"][verb_idx]
    obj = {"action": verb}
    required = space["verb_fields"][verb]
    if "target" in required:
        obj["target"] = space["targets"][target_idx]
    if "location" in required:
        obj["location"] = space["locations"][location_idx]
    obj["modifiers"] = []          # constant: empty on all 468 rows, all 3 splits
    return obj


@torch.no_grad()
def dev_exact_match(model, rows, tok, space, device):
    """Fraction of dev rows whose emitted object equals the ground truth.

    EPOCH SELECTION ONLY. Never a reported number -- reported numbers come from
    the bench harness at frozen commit 3000198."""
    model.eval()
    hits = 0
    for row in rows:
        enc = tok(row["command"], truncation=True, max_length=MAX_LEN,
                  padding="max_length", return_tensors="pt").to(device)
        vl, tl, ll = model(enc["input_ids"], enc["attention_mask"])
        obj = emit(vl.argmax(-1).item(), tl.argmax(-1).item(), ll.argmax(-1).item(), space)
        truth = dict(row["expected"])
        truth.setdefault("modifiers", [])
        if obj == truth:
            hits += 1
    model.train()
    return hits / len(rows), hits


# ------------------------------------------------------------------ train ----
def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    space = build_label_space()
    train_rows = load_jsonl(TRAIN_FILE)
    dev_rows   = load_jsonl(DEV_FILE)

    print("=" * 68)
    print("RUNG 4 -- DistilBERT multi-head classifier")
    print("=" * 68)
    print(f"  device        = {device}  ({torch.cuda.get_device_name(0) if device == 'cuda' else 'cpu'})")
    print(f"  torch         = {torch.__version__}")
    print(f"  base model    = {BASE_MODEL}")
    print(f"  schema        = schema_v2.json (version {space['schema_version']})")
    print(f"  heads         = verb {len(space['verbs'])} | target {len(space['targets'])} | location {len(space['locations'])}")
    print(f"  train / dev   = {len(train_rows)} / {len(dev_rows)}")
    print(f"  seed={SEED} lr={LR} bs={BATCH_SIZE} epochs={EPOCHS} "
          f"warmup_steps={WARMUP_STEPS} max_len={MAX_LEN} dropout={DROPOUT}")
    print("  verb -> required fields (from schema, NOT hard-coded):")
    for v, req in space["verb_fields"].items():
        print(f"      {v:12} {req if req else '(none)'}")
    print()

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    train_ds = CommandDataset(train_rows, tok, space)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          generator=torch.Generator().manual_seed(SEED))

    model = MultiHeadClassifier(space).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters    = {n_params/1e6:.1f} M")

    optim = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    total_steps = EPOCHS * math.ceil(len(train_ds) / BATCH_SIZE)

    def lr_lambda(step):
        if step < WARMUP_STEPS:
            return step / max(1, WARMUP_STEPS)
        prog = (step - WARMUP_STEPS) / max(1, total_steps - WARMUP_STEPS)
        return max(0.0, 1.0 - prog)          # linear decay to 0

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    loss_fn = nn.CrossEntropyLoss(ignore_index=IGNORE)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "label_maps.json").write_text(
        json.dumps(space, indent=2), encoding="utf-8")

    print(f"  total steps   = {total_steps}")
    print(f"  output        = {OUT_DIR}\n")
    print(f"{'epoch':>5} {'train_loss':>11} {'dev_exact':>10} {'hits':>8}")
    print("-" * 38)

    history = []
    for epoch in range(1, EPOCHS + 1):
        running, nb = 0.0, 0
        for batch in train_dl:
            ids  = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            vlab = batch["verb"].to(device)
            tlab = batch["target"].to(device)
            llab = batch["location"].to(device)

            vl, tl, ll = model(ids, mask)
            loss = (loss_fn(vl, vlab)
                    + masked_ce(tl, tlab, loss_fn)
                    + masked_ce(ll, llab, loss_fn))

            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()

            running += loss.item()
            nb += 1

        acc, hits = dev_exact_match(model, dev_rows, tok, space, device)
        history.append({"epoch": epoch, "train_loss": running / nb,
                        "dev_exact": acc, "dev_hits": hits})
        print(f"{epoch:>5} {running/nb:>11.4f} {acc:>10.4f} {hits:>4}/{len(dev_rows)}")

        ck = OUT_DIR / f"epoch_{epoch:02d}"
        ck.mkdir(exist_ok=True)
        torch.save(model.state_dict(), ck / "model.pt")
        tok.save_pretrained(ck)

    (OUT_DIR / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")

    # ---- checkpoint selection: EARLIEST epoch at the plateau, not argmax ----
    best_hits = max(h["dev_hits"] for h in history)
    plateau = [h for h in history if h["dev_hits"] >= best_hits - PLATEAU_TOL]
    chosen = min(plateau, key=lambda h: h["epoch"])
    argmax = max(history, key=lambda h: (h["dev_hits"], -h["epoch"]))

    print("\n" + "=" * 68)
    print("CHECKPOINT SELECTION -- earliest epoch reaching the dev plateau")
    print("=" * 68)
    print(f"  best dev hits        : {best_hits}/{len(dev_rows)}")
    print(f"  plateau (within {PLATEAU_TOL} row): epochs "
          f"{[h['epoch'] for h in plateau]}")
    print(f"  SELECTED             : epoch {chosen['epoch']}  "
          f"({chosen['dev_hits']}/{len(dev_rows)} = {chosen['dev_exact']:.1%})")
    print(f"  (argmax would pick   : epoch {argmax['epoch']} -- "
          f"{'same' if argmax['epoch'] == chosen['epoch'] else 'DIFFERENT, use the rule not argmax'})")
    chosen_dir = OUT_DIR / f"epoch_{chosen['epoch']:02d}"
    print(f"\n  checkpoint dir       : {chosen_dir}")
    print("\n  Dev numbers above are FOR EPOCH SELECTION ONLY. Not reportable.")
    print("  Reported numbers come from the bench harness at commit 3000198.")


if __name__ == "__main__":
    main()
