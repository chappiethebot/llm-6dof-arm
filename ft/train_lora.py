"""ft/train_lora.py — Rung 3: LoRA fine-tuning on Qwen2.5 Instruct.

API notes for transformers 5.15.1 (verified by probe, not assumed):
  - Trainer takes `processing_class`, NOT `tokenizer`.
  - from_pretrained takes `dtype`, NOT `torch_dtype`.
  - TrainingArguments has `warmup_steps` and `eval_strategy`; `warmup_ratio`
    and `evaluation_strategy` do not exist.

LOSS MASKING: labels are -100 over the entire prompt, real token ids over the
target only. The script DUMPS one masked example before training and will
refuse to proceed if the mask does not start exactly at the target boundary.

DEV METRIC: `json_match` applies the SAME leniency as the frozen scorer at
commit 3000198 — an empty `modifiers` field may be present or absent on either
side. `json_exact` is the unforgiving version, printed alongside so the gap is
visible. Epoch selection uses `json_match`. Neither is a reported thesis
number; all reported numbers come from the bench harness at 3000198.

Config is FIXED, not swept. 230 train rows / 57 dev rows cannot support a
hyperparameter search. See the thesis methods chapter.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)
from peft import LoraConfig, get_peft_model

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "ft" / "data"
RUNS_DIR = ROOT / "ft" / "runs"

MODELS = {
    "0.5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "3b":   "Qwen/Qwen2.5-3B-Instruct",
}

# ---- FIXED CONFIG. Pre-registered. Do not sweep. ----
SEED            = 20260820
LORA_R          = 16
LORA_ALPHA      = 32
LORA_DROPOUT    = 0.05
TARGET_MODULES  = ["q_proj", "k_proj", "v_proj", "o_proj",
                   "gate_proj", "up_proj", "down_proj"]   # lm_head deliberately excluded
LEARNING_RATE   = 2e-4
EPOCHS          = 8
BATCH_SIZE      = 8
MAX_SEQ_LEN     = 384        # measured max total = 335
WARMUP_STEPS    = 10
SCHEDULER       = "cosine"
MAX_NEW_TOKENS  = 64         # measured max target = 29


# ---------------------------------------------------------------- data

def read_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


class SFTDataset(Dataset):
    """Prompt tokens masked to -100. Target tokens are the only labels."""

    def __init__(self, rows, tok, max_len):
        self.items = []
        self.n_truncated = 0
        for r in rows:
            p_ids = tok(r["prompt_text"], add_special_tokens=False)["input_ids"]
            t_ids = tok(r["target_text"], add_special_tokens=False)["input_ids"]
            ids = p_ids + t_ids
            labels = [-100] * len(p_ids) + list(t_ids)
            if len(ids) > max_len:
                self.n_truncated += 1
                ids = ids[:max_len]
                labels = labels[:max_len]
            self.items.append({
                "input_ids": ids,
                "labels": labels,
                "n_prompt": len(p_ids),
                "command": r["command"],
                "prompt_text": r["prompt_text"],
                "target_text": r["target_text"],
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        return {"input_ids": it["input_ids"], "labels": it["labels"]}


def make_collator(pad_id):
    def collate(batch):
        longest = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            pad = longest - len(b["input_ids"])
            input_ids.append(b["input_ids"] + [pad_id] * pad)
            labels.append(b["labels"] + [-100] * pad)
            attn.append([1] * len(b["input_ids"]) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }
    return collate


# ---------------------------------------------------------------- mask audit

def dump_mask(ds, tok, idx=0, max_tokens=400):
    """Show exactly which tokens carry loss. '·' = masked, literal = learned."""
    it = ds.items[idx]
    ids, labels = it["input_ids"], it["labels"]
    print("=" * 72)
    print(f"MASK AUDIT — row {idx}: {it['command']!r}")
    print(f"  total tokens {len(ids)}  |  prompt {it['n_prompt']}  |  "
          f"learned {sum(1 for l in labels if l != -100)}")
    print("-" * 72)
    parts = []
    for tid, lab in list(zip(ids, labels))[:max_tokens]:
        parts.append("·" if lab == -100 else tok.decode([tid]))
    print("".join(parts))
    print("-" * 72)
    learned = tok.decode([t for t, l in zip(ids, labels) if l != -100])
    print("LEARNED TEXT:", repr(learned))
    print("EXPECTED    :", repr(it["target_text"]))
    ok = learned == it["target_text"]
    print("MASK CORRECT:", ok)
    print("=" * 72)
    if not ok:
        raise SystemExit("ABORT: loss mask does not match the target exactly.")


# ---------------------------------------------------------------- dev eval

def _drop_empty_modifiers(obj):
    """Frozen-scorer rule (commit 3000198): an EMPTY `modifiers` field may be
    present or absent on either side. A NON-empty one is semantic and stays."""
    if isinstance(obj, dict) and obj.get("modifiers") == []:
        obj = {k: v for k, v in obj.items() if k != "modifiers"}
    return obj


@torch.no_grad()
def dev_exact_match(model, tok, rows, device, verbose_n=0):
    """Greedy generation, matching the eval-time temperature of 0.0.
    Selection metric only. Not a reported number."""
    model.eval()
    n_str, n_lenient, n_exact, n_parse = 0, 0, 0, 0
    shown = 0
    for r in rows:
        enc = tok(r["prompt_text"], return_tensors="pt",
                  add_special_tokens=False).to(device)
        out = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tok.convert_tokens_to_ids("<|im_end|>"),
            pad_token_id=tok.pad_token_id,
        )
        gen = tok.decode(out[0][enc["input_ids"].shape[1]:],
                         skip_special_tokens=True).strip()
        want = r["target_text"].replace("<|im_end|>", "").strip()

        if gen == want:
            n_str += 1
        try:
            g, w = json.loads(gen), json.loads(want)
            n_parse += 1
            if g == w:
                n_exact += 1
            if _drop_empty_modifiers(g) == _drop_empty_modifiers(w):
                n_lenient += 1
        except Exception:
            pass

        if shown < verbose_n:
            print(f"    [{r['command'][:44]!r:48s}] got={gen!r}")
            shown += 1

    model.train()
    n = len(rows)
    return {
        "string_match": n_str / n,
        "json_match":   n_lenient / n,   # frozen-scorer leniency; SELECTION METRIC
        "json_exact":   n_exact / n,     # unforgiving, for contrast
        "parse_rate":   n_parse / n,
        "n":            n,
    }


class DevEvalCallback(TrainerCallback):
    def __init__(self, model, tok, rows, device, log_path):
        self.model, self.tok, self.rows = model, tok, rows
        self.device, self.log_path = device, log_path
        self.history = []

    def on_epoch_end(self, args, state, control, **kw):
        ep = int(round(state.epoch))
        m = dev_exact_match(self.model, self.tok, self.rows, self.device,
                            verbose_n=3 if ep == 1 else 0)
        m["epoch"] = ep
        self.history.append(m)
        print(f"  >> DEV epoch {ep}: json_match {m['json_match']*100:.1f}%  "
              f"(exact {m['json_exact']*100:.1f}%)  "
              f"string {m['string_match']*100:.1f}%  "
              f"parse {m['parse_rate']*100:.1f}%  (n={m['n']})")
        with open(self.log_path, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2)
        return control


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--dry-run", action="store_true",
                    help="mask audit + one dev eval on the BASE model, then exit")
    args = ap.parse_args()

    tag = args.model
    model_id = MODELS[tag]
    out_dir = RUNS_DIR / f"qwen{tag}_lora"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(SEED)

    print("=" * 72)
    print(f"MODEL      : {model_id}")
    print(f"OUT        : {out_dir}")
    print(f"CONFIG     : r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT} "
          f"lr={LEARNING_RATE} epochs={args.epochs} bs={BATCH_SIZE} seed={SEED}")
    print("=" * 72)

    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "right"

    train_rows = read_jsonl(DATA_DIR / f"train_sft_{tag}.jsonl")
    dev_rows   = read_jsonl(DATA_DIR / f"dev_sft_{tag}.jsonl")
    print(f"train rows {len(train_rows)}  |  dev rows {len(dev_rows)}")

    train_ds = SFTDataset(train_rows, tok, MAX_SEQ_LEN)
    dev_ds   = SFTDataset(dev_rows,   tok, MAX_SEQ_LEN)
    if train_ds.n_truncated or dev_ds.n_truncated:
        raise SystemExit(f"ABORT: truncation occurred "
                         f"(train {train_ds.n_truncated}, dev {dev_ds.n_truncated}). "
                         f"Raise MAX_SEQ_LEN.")
    print("truncated rows: 0")

    dump_mask(train_ds, tok, idx=0)

    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    model.config.use_cache = False

    lcfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        bias="none", task_type="CAUSAL_LM", target_modules=TARGET_MODULES,
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    if args.dry_run:
        print("\nDRY RUN — base+untrained-adapter dev eval (expect low):")
        m = dev_exact_match(model, tok, dev_rows, device, verbose_n=5)
        print(f"  json_match {m['json_match']*100:.1f}%  "
              f"(exact {m['json_exact']*100:.1f}%)  "
              f"string {m['string_match']*100:.1f}%  "
              f"parse {m['parse_rate']*100:.1f}%")
        print("\nDRY RUN COMPLETE — no training performed.")
        return

    targs = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=SCHEDULER,
        warmup_steps=WARMUP_STEPS,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to="none",
        seed=SEED,
    )

    cb = DevEvalCallback(model, tok, dev_rows, device,
                         out_dir / "dev_metrics.json")

    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=make_collator(tok.pad_token_id),
        processing_class=tok,
        callbacks=[cb],
    )

    trainer.train()

    print("\n" + "=" * 72)
    print("PER-EPOCH DEV (selection metric only, NOT a reported number)")
    print(f"  {'epoch':>6}  {'json_match':>11}  {'exact':>8}  {'string':>8}  {'parse':>7}")
    for m in cb.history:
        print(f"  {m['epoch']:>6}  {m['json_match']*100:>10.1f}%  "
              f"{m['json_exact']*100:>7.1f}%  {m['string_match']*100:>7.1f}%  "
              f"{m['parse_rate']*100:>6.1f}%")
    if cb.history:
        best = max(cb.history, key=lambda x: x["json_match"])
        print(f"\nBEST EPOCH: {best['epoch']}  "
              f"(json_match {best['json_match']*100:.1f}%)")
        print(f"Checkpoint: {out_dir / ('checkpoint-*')}  — pick the one for epoch "
              f"{best['epoch']}")
    print("=" * 72)


if __name__ == "__main__":
    main()