#!/usr/bin/env python
"""
ft/eval_classifier.py -- Rung 4 diagnostic. NOT a reported number.

Loads the SELECTED checkpoint and answers three questions the training log
cannot:

  1. WHICH head is wrong on the dev misses? If the verb head is perfect and
     every miss is a location, that is direct evidence for the structural
     guarantee -- the model never picks an illegal verb, it picks the wrong
     square. Different claim, different chapter.

  2. Does the emitted object EVER violate the schema? It should be impossible
     by construction. This asserts it rather than assuming it.

  3. Roughly what does one CPU forward pass cost? Sanity only -- the reportable
     latency comes from the bench harness, which times the same call inside
     run_once. If this number and the harness number disagree by more than a
     little, something is wrong with one of them.

USAGE (conda env `train` or `thesis` -- both have CPU torch)
    python -m ft.eval_classifier
    python -m ft.eval_classifier --epoch 10        # compare against argmax
"""
import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from ft.train_classifier import (MultiHeadClassifier, build_label_space,
                                 load_jsonl, emit, MAX_LEN, OUT_DIR,
                                 DEV_FILE, SCHEMA)
from transformers import AutoTokenizer

# Single-threaded, matching the deployment story stated in the methods chapter.
torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epoch", type=int, default=9,
                    help="checkpoint epoch to load (default: the selected one)")
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    ck = OUT_DIR / f"epoch_{args.epoch:02d}"
    if not ck.exists():
        raise SystemExit(f"no checkpoint at {ck}")

    space = build_label_space()
    rows = load_jsonl(DEV_FILE)
    sch = json.loads(SCHEMA.read_text(encoding="utf-8"))

    tok = AutoTokenizer.from_pretrained(str(ck))
    model = MultiHeadClassifier(space)
    model.load_state_dict(torch.load(ck / "model.pt", map_location="cpu"))
    model.eval()

    print(f"checkpoint : {ck}")
    print(f"dev rows   : {len(rows)}   threads: {torch.get_num_threads()}\n")

    per_head = {"action": [0, 0], "target": [0, 0], "location": [0, 0]}  # [hit, total]
    exact = 0
    misses = []
    latencies = []

    with torch.no_grad():
        for i, row in enumerate(rows):
            truth = dict(row["expected"])
            truth.setdefault("modifiers", [])

            t0 = time.perf_counter()
            enc = tok(row["command"], truncation=True, max_length=MAX_LEN,
                    padding=False, return_tensors="pt")
            vl, tl, ll = model(enc["input_ids"], enc["attention_mask"])
            obj = emit(vl.argmax(-1).item(), tl.argmax(-1).item(),
                       ll.argmax(-1).item(), space)
            dt = time.perf_counter() - t0
            if i >= args.warmup:
                latencies.append(dt)

            # ---- structural guarantee: assert, do not assume -----------------
            assert obj["action"] in sch["verbs"], f"illegal verb {obj['action']}"
            required = set(sch["verbs"][obj["action"]]["required"])
            present = {k for k in ("target", "location") if k in obj}
            assert present == required, (
                f"field mask violated for {obj['action']}: "
                f"emitted {present}, schema requires {required}")
            if "target" in obj:
                assert obj["target"] in sch["targets"]
            if "location" in obj:
                assert obj["location"] in sch["locations"]

            # ---- per-head scoring, only where the field is required ----------
            for key in ("action", "target", "location"):
                if key in truth:
                    per_head[key][1] += 1
                    if obj.get(key) == truth[key]:
                        per_head[key][0] += 1

            if obj == truth:
                exact += 1
            else:
                wrong = [k for k in ("action", "target", "location")
                         if truth.get(k) != obj.get(k)]
                misses.append((row["command"], truth, obj, wrong))

    print("STRUCTURAL GUARANTEE: no illegal verb, no illegal symbol, no field-mask")
    print(f"  violation in {len(rows)}/{len(rows)} rows. Asserted, not assumed.\n")

    print("PER-HEAD ACCURACY (scored only on rows where the field is required)")
    for key, (hit, tot) in per_head.items():
        pct = hit / tot if tot else float("nan")
        print(f"  {key:9} {hit:3}/{tot:<3} = {pct:6.1%}")
    print(f"\n  exact object match: {exact}/{len(rows)} = {exact/len(rows):.1%}")

    if misses:
        print(f"\nMISSES ({len(misses)}) -- which head failed:")
        for cmd, truth, obj, wrong in misses:
            print(f"  head={','.join(wrong):18} {cmd!r}")
            print(f"      truth: {truth}")
            print(f"      pred : {obj}")

    if latencies:
        ms = sorted(x * 1000 for x in latencies)
        print(f"\nCPU LATENCY -- SANITY ONLY, not reportable "
              f"(n={len(ms)}, {args.warmup} warmup excluded, 1 thread)")
        print(f"  p50 {statistics.median(ms):6.2f} ms | "
              f"p95 {ms[int(0.95*len(ms))-1]:6.2f} ms | "
              f"max {ms[-1]:6.2f} ms")
        print("  Reportable latency comes from the bench harness at commit 3000198.")


if __name__ == "__main__":
    main()
