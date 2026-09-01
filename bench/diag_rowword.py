#!/usr/bin/env python
"""
bench/diag_rowword.py -- Is the seen -> held-out difference a VOCABULARY effect
rather than generalization?

Background. seen_v2 and held_out_v2 label the SAME nine squares, but with
different words:

    cell   seen                        held-out
    A1     the far left corner         the upper left corner
    C1     the near left corner        the bottom left corner
    B1     the left side middle edge   the left side edge from the table center

So a seen -> held-out accuracy change could be either (a) genuine failure to
generalize to unseen phrasings -- the KPI -- or (b) one vocabulary simply suiting
the models better than the other. Pooled numbers cannot separate these.

This script breaks strict accuracy down by ROW LETTER, per model. Rows are the
axis the vocabulary actually changes (far/near vs upper/bottom); columns use
left/middle/right in BOTH sets and act as a control -- if columns move as much as
rows, the effect is not about the row words at all.

Read-only. Touches no dataset and no results file.

Usage:
    python bench/diag_rowword.py
    python bench/diag_rowword.py --seen bench/results_grid_v2_scored.csv \
                                 --held bench/results_heldout_v2_scored.csv

DATA VERSION: V2 ONLY. Reads the two V2 scored CSVs. The same question was
re-asked against V3 with a controlled dataset rather than an inference --
see audit/lora_vocab_test.py, audit/build_v4b.py and audit/build_v4c.py.
"""

import argparse
import collections
import csv
import json

csv.field_size_limit(2**31 - 1)

MODELS = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
          "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b"]


def short(m):
    return m.split(":", 1)[1] if ":" in m else m


def load(path):
    """Read a scored CSV, dedupe on identity key keeping LAST (resume orphans)."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    seen = {}
    for r in rows:
        seen[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
    return list(seen.values())


def breakdown(rows, model, axis):
    """axis 0 = row letter (A/B/C), axis 1 = column digit (1/2/3).
    Returns {key: (hits, n)} over grid_* rows only."""
    acc = collections.defaultdict(lambda: [0, 0])
    for r in rows:
        if r["model"] != model:
            continue
        try:
            loc = json.loads(r["truth_json"]).get("location", "")
        except (json.JSONDecodeError, TypeError, KeyError):
            continue
        if not isinstance(loc, str) or not loc.startswith("grid_"):
            continue
        key = loc[5 + axis]
        acc[key][1] += 1
        if r["strict_match"] == "True":
            acc[key][0] += 1
    return acc


def show(seen_rows, held_rows, axis, keys, title, note):
    print("=" * 74)
    print(title)
    print(note)
    print("=" * 74)
    print(f'{"model":16}{"set":10}' + "".join(f"{k:>9}" for k in keys) + f'{"all":>9}')
    for m in MODELS:
        lines = {}
        for label, rows in (("seen", seen_rows), ("held-out", held_rows)):
            acc = breakdown(rows, m, axis)
            if not acc:
                continue
            cells, tot_h, tot_n = [], 0, 0
            for k in keys:
                h, n = acc.get(k, [0, 0])
                tot_h += h
                tot_n += n
                cells.append(f"{100*h/n:8.1f}%" if n else f'{"-":>9}')
            cells.append(f"{100*tot_h/tot_n:8.1f}%" if tot_n else f'{"-":>9}')
            lines[label] = cells
        if not lines:
            continue
        for label in ("seen", "held-out"):
            if label in lines:
                print(f'{short(m):16}{label:10}' + "".join(lines[label]))
        if "seen" in lines and "held-out" in lines:
            delta = []
            for a, b in zip(lines["seen"], lines["held-out"]):
                try:
                    delta.append(f"{float(b.strip('% ')) - float(a.strip('% ')):+8.1f} ")
                except ValueError:
                    delta.append(f'{"-":>9}')
            print(f'{"":16}{"change":10}' + "".join(delta))
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", default="bench/results_grid_v2_scored.csv")
    ap.add_argument("--held", default="bench/results_heldout_v2_scored.csv")
    args = ap.parse_args()

    seen_rows = load(args.seen)
    held_rows = load(args.held)
    print(f"seen: {len(seen_rows)} rows | held-out: {len(held_rows)} rows\n")

    show(seen_rows, held_rows, 0, list("ABC"),
         "STRICT ACCURACY BY ROW LETTER  (grid_* rows only)",
         "seen says far/middle/near.  held-out says upper/middle/bottom.")

    show(seen_rows, held_rows, 1, list("123"),
         "CONTROL: STRICT ACCURACY BY COLUMN DIGIT",
         "BOTH sets say left/middle/right. Large changes here mean the effect is\n"
         "NOT specific to the row vocabulary.")

    print("=" * 74)
    print("READING IT")
    print("=" * 74)
    print("""  Rows change a lot, columns barely move
      -> the row VOCABULARY is doing the work, not generalization. The
         seen->held-out drop cannot be reported as a generalization gap
         without saying so.

  Rows and columns move together
      -> a general difficulty difference between the two sets, not a
         row-word effect. Look elsewhere for the cause.

  Strong models patterned, weak models flat and near zero
      -> the weak models are not grounding space at all, so their `drop`
         numbers are noise around a floor and should not be averaged in.

  Row B worst in both sets
      -> expected. `middle` carries no directional cue, and B1/B3 were the
         two cells whose held-out phrasings reuse seen wording.""")


if __name__ == "__main__":
    main()