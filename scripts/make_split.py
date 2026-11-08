"""
make_split.py -- carve train_v2 / dev_v2 out of seen_v2.jsonl.

WHY
---
Rungs 2, 3 and 4 need training data and a selection signal.
held_out_v2.jsonl must never provide that signal -- it is the generalization KPI.

    seen_v2.jsonl  ->  train_v2.jsonl (~80%)  +  dev_v2.jsonl (~20%)

From here on, every design choice (LoRA rank, epochs, early stopping, RAG corpus)
uses train_v2 / dev_v2 ONLY.

WHAT IT DOES
------------
1. Loads seen_v2.jsonl through bench/dataset.py, which removes the 3 few-shot
   exemplars. Those 3 rows go into NEITHER split -- they are a prompting
   artifact, not training data.
2. Splits stratified by (verb, ambiguity class), not randomly. A plain random
   split can leave `stop` or `reject` with zero dev rows.
3. Prints a per-stratum table before writing anything.
4. Refuses to overwrite existing output files.

AMBIGUITY CLASS
---------------
A row is "ambiguous" if its command contains middle / center / centre AND the
correct answer is not grid_B2. Those are the phrasings that collide with the
prompt legend's own word for column 2. Keeping the classes balanced across
train and dev means dev accuracy is not moved by an accident of the draw.

USAGE (PowerShell, from repo root)
----------------------------------
    conda activate thesis
    python scripts/make_split.py

    python scripts/make_split.py --yes     skip the confirmation prompt

SAFETY
------
Never modifies seen_v2.jsonl. Never touches held_out_v2.jsonl.
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

SEEN_PATH = os.path.join("data", "seen_v2.jsonl")
TRAIN_PATH = os.path.join("data", "train_v2.jsonl")
DEV_PATH = os.path.join("data", "dev_v2.jsonl")

DEV_FRACTION = 0.20
SEED = 20260820  # fixed. Record this number in the methods chapter.

_AMBIG_WORDS = ("middle", "center", "centre")
_CENTRE_CELL = "grid_B2"


# ---------------------------------------------------------------- exemplars

def load_rows_minus_exemplars():
    """Use bench/dataset.py so the exemplar choice lives in exactly one place."""
    bench_dir = str(Path("bench").resolve())
    if bench_dir not in sys.path:
        sys.path.insert(0, bench_dir)
    try:
        import dataset  # bench/dataset.py
    except ImportError as e:
        sys.exit(f"FATAL: could not import bench/dataset.py ({e}). "
                 f"Run this script from the repo root.")

    exemplars, test_rows = dataset.load_seen_split(seen_path=SEEN_PATH)
    return exemplars, test_rows


# ---------------------------------------------------------------- helpers

def verb_of(row):
    v = row.get("expected", {}).get("action")
    if not v:
        sys.exit(f"FATAL: row has no expected.action: {row}")
    return v


def ambiguity_of(row):
    cmd = (row.get("command") or "").lower()
    loc = row.get("expected", {}).get("location")
    if any(w in cmd for w in _AMBIG_WORDS) and loc != _CENTRE_CELL:
        return "ambiguous"
    return "clean"


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- split

def stratified_split(rows):
    """Return (train, dev). Deterministic: sorted then shuffled with a fixed seed."""
    buckets = defaultdict(list)
    for r in rows:
        buckets[(verb_of(r), ambiguity_of(r))].append(r)

    rng = random.Random(SEED)
    train, dev = [], []
    table = []

    for key in sorted(buckets.keys()):
        group = sorted(buckets[key], key=lambda r: r.get("command", ""))
        rng.shuffle(group)
        n = len(group)

        if n == 1:
            n_dev = 0          # a single row cannot be split; keep it trainable
        else:
            n_dev = max(1, round(n * DEV_FRACTION))
            n_dev = min(n_dev, n - 1)   # never leave a stratum with zero train rows

        dev.extend(group[:n_dev])
        train.extend(group[n_dev:])
        table.append((key[0], key[1], n, n - n_dev, n_dev))

    return train, dev, table


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="write without confirming")
    args = ap.parse_args()

    if not Path(SEEN_PATH).exists():
        sys.exit(f"FATAL: {SEEN_PATH} not found. Run from the repo root.")

    for p in (TRAIN_PATH, DEV_PATH):
        if Path(p).exists():
            sys.exit(f"FATAL: {p} already exists. Rename or delete it first.\n"
                     f"Re-splitting after training has started silently "
                     f"contaminates everything downstream.")

    exemplars, rows = load_rows_minus_exemplars()

    print(f"\nseen file      : {SEEN_PATH}")
    print(f"exemplars held : {len(exemplars)}  (in neither split)")
    for e in exemplars:
        print(f"    - {e.get('command')}")
    print(f"splittable rows: {len(rows)}")
    print(f"seed           : {SEED}   dev fraction: {DEV_FRACTION}")

    train, dev, table = stratified_split(rows)

    print("\n  verb          class        total   train   dev")
    print("  " + "-" * 46)
    for verb, cls, n, n_tr, n_dv in table:
        flag = "   <-- dev too small" if n_dv <= 2 else ""
        print(f"  {verb:<13} {cls:<11} {n:>5}   {n_tr:>5}  {n_dv:>4}{flag}")
    print("  " + "-" * 46)
    print(f"  {'TOTAL':<25} {len(rows):>5}   {len(train):>5}  {len(dev):>4}")

    print("\n  train by verb:", dict(Counter(verb_of(r) for r in train)))
    print("  dev   by verb:", dict(Counter(verb_of(r) for r in dev)))

    # leakage check: the same command string in both splits inflates dev accuracy
    overlap = {r.get("command") for r in train} & {r.get("command") for r in dev}
    if overlap:
        print(f"\n  LEAKAGE: {len(overlap)} command string(s) in BOTH splits:")
        for c in list(overlap)[:5]:
            print(f"    - {c}")
        print("  These are duplicates inside seen_v2.jsonl. Fix before training.")
    else:
        print("\n  leakage check: no command appears in both splits   OK")

    ex_cmds = {e.get("command") for e in exemplars}
    bleed = ex_cmds & ({r.get("command") for r in train} | {r.get("command") for r in dev})
    if bleed:
        print(f"  WARNING: {len(bleed)} exemplar command(s) also appear as data rows.")
    else:
        print("  exemplar check: no exemplar appears in either split   OK")

    if not args.yes:
        if input("\nwrite these files? [y/N] ").strip().lower() != "y":
            sys.exit("aborted, nothing written")

    write_jsonl(TRAIN_PATH, train)
    write_jsonl(DEV_PATH, dev)
    print(f"\nwrote {TRAIN_PATH}  ({len(train)} rows)")
    print(f"wrote {DEV_PATH}    ({len(dev)} rows)")
    print("held_out_v2.jsonl was NOT touched.")


if __name__ == "__main__":
    main()
