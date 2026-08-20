#!/usr/bin/env python
"""
bench/diag_cells.py -- per-cell accuracy and confusion, for the V2 grid.

Why this exists. Pooled grid accuracy conceals a phrase/legend collision: four of
the nine location phrasings contain the token "middle", which also labels COLUMN 2
in the prompt legend. Models resolve those phrasings to the centre cell instead of
the intended edge, so pooled accuracy mixes "the model cannot ground space" with
"the phrase was ambiguous". This script separates them:

  1. per-cell accuracy, cloud vs local, with the phrase printed alongside
  2. the dominant confusion per cell -- what the model said instead
  3. a clean-vs-ambiguous split, where ambiguous = phrase contains a token that
     also appears in the legend as an axis label ("middle", "center", "centre")
  4. an optional confusion matrix per model

Read-only. Touches no dataset and no results file.

Usage:
    python bench/diag_cells.py
    python bench/diag_cells.py --model openai:gpt-4o-mini --matrix
    python bench/diag_cells.py --held bench/results_heldout_v2_scored.csv
"""

import argparse
import collections
import csv
import json

csv.field_size_limit(2**31 - 1)

MODELS = ["openai:gpt-4o-mini", "ollama:llama3.2:3b", "ollama:qwen2.5:3b",
          "ollama:qwen2.5:0.5b", "ollama:deepseek-r1:7b"]

CELLS = [f"grid_{r}{c}" for r in "ABC" for c in "123"]

# Tokens that the prompt legend uses as AXIS LABELS. A location phrase containing
# one of these competes with the legend for the model's attention.
LEGEND_TOKENS = {"middle", "center", "centre"}


def short(m):
    return m.split(":", 1)[1] if ":" in m else m


def load(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    seen = {}
    for r in rows:
        seen[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
    return list(seen.values())


def truth_loc(r):
    try:
        return json.loads(r["truth_json"]).get("location")
    except (json.JSONDecodeError, TypeError, KeyError):
        return None


def pred_loc(r):
    """Predicted location as a HASHABLE value. Some models nest the whole action
    object, so .get("location") can return a dict or a list -- those must be
    stringified before they are used as a Counter key."""
    try:
        obj = json.loads(r["pred_json"]) if r["pred_json"] else None
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None
    v = obj.get("location")
    if v is None:
        act = obj.get("action")
        if isinstance(act, dict):          # nested: {"action": {"verb":..., "location":...}}
            v = act.get("location")
    if v is None:
        return None
    return v if isinstance(v, str) else json.dumps(v, sort_keys=True)


def phrase_for(rows, cell):
    """Recover the location phrase from a command, by stripping the known
    template prefix. Best effort -- used for display only."""
    for r in rows:
        if truth_loc(r) == cell:
            c = r["command"]
            for pre in ("put the bottle in ", "put the car in ", "put the rubiks cube in ",
                        "move to ", "go to ", "navigate to "):
                if c.startswith(pre):
                    return c[len(pre):].rstrip("?")
            return c
    return "?"


def is_ambiguous(phrase):
    toks = set(phrase.lower().replace(",", " ").split())
    return bool(toks & LEGEND_TOKENS)


def per_cell(rows, model):
    acc = {c: [0, 0] for c in CELLS}
    conf = {c: collections.Counter() for c in CELLS}
    for r in rows:
        if r["model"] != model:
            continue
        t = truth_loc(r)
        if t not in acc:
            continue
        acc[t][1] += 1
        p = pred_loc(r)
        if r["strict_match"] == "True":
            acc[t][0] += 1
        else:
            conf[t][p] += 1
    return acc, conf


def section_cells(rows, label, models):
    print("=" * 88)
    print(f"PER-CELL STRICT ACCURACY -- {label}")
    print("=" * 88)
    phrases = {c: phrase_for(rows, c) for c in CELLS}
    amb = {c: is_ambiguous(phrases[c]) for c in CELLS}

    print(f'{"cell":9}{"amb":5}{"phrase":42}' + "".join(f"{short(m)[:11]:>12}" for m in models))
    for c in CELLS:
        line = f'{c:9}{"YES" if amb[c] else "":5}{phrases[c][:40]:42}'
        for m in models:
            acc, _ = per_cell(rows, m)
            h, n = acc[c]
            line += f"{100*h/n:11.1f}%" if n else f'{"-":>12}'
        print(line)

    print()
    print(f'{"":9}{"":5}{"CLEAN phrases (no legend token)":42}', end="")
    for m in models:
        acc, _ = per_cell(rows, m)
        h = sum(acc[c][0] for c in CELLS if not amb[c])
        n = sum(acc[c][1] for c in CELLS if not amb[c])
        print(f"{100*h/n:11.1f}%" if n else f'{"-":>12}', end="")
    print()
    print(f'{"":9}{"":5}{"AMBIGUOUS phrases":42}', end="")
    for m in models:
        acc, _ = per_cell(rows, m)
        h = sum(acc[c][0] for c in CELLS if amb[c])
        n = sum(acc[c][1] for c in CELLS if amb[c])
        print(f"{100*h/n:11.1f}%" if n else f'{"-":>12}', end="")
    print("\n")


def section_confusion(rows, model):
    print("=" * 88)
    print(f"DOMINANT CONFUSION -- {model}  (what it said when it was wrong)")
    print("=" * 88)
    acc, conf = per_cell(rows, model)
    print(f'{"cell":9}{"correct":>9}{"n":>7}   most common wrong answers')
    for c in CELLS:
        h, n = acc[c]
        if not n:
            continue
        top = ", ".join(f"{k or '<none>'} x{v}" for k, v in conf[c].most_common(3))
        print(f'{c:9}{100*h/n:8.1f}%{n:7d}   {top}')
    print()


def section_matrix(rows, model):
    print("=" * 88)
    print(f"CONFUSION MATRIX -- {model}   rows = truth, cols = prediction (%)")
    print("=" * 88)
    print(f'{"truth":9}' + "".join(f"{c[5:]:>7}" for c in CELLS) + f'{"other":>8}')
    for t in CELLS:
        counts = collections.Counter()
        n = 0
        for r in rows:
            if r["model"] != model or truth_loc(r) != t:
                continue
            n += 1
            counts[pred_loc(r)] += 1
        if not n:
            continue
        line = f'{t:9}'
        shown = 0
        for p in CELLS:
            v = counts.get(p, 0)
            shown += v
            line += f"{100*v/n:7.0f}" if v else f'{".":>7}'
        line += f"{100*(n-shown)/n:8.0f}"
        print(line)
    print("   `other` = a non-grid location, a missing location, or no parseable object.\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seen", default="bench/results_grid_v2_scored.csv")
    ap.add_argument("--held", default=None, help="also break down the held-out file")
    ap.add_argument("--model", default="openai:gpt-4o-mini",
                    help="model for the confusion sections")
    ap.add_argument("--matrix", action="store_true", help="print the full confusion matrix")
    args = ap.parse_args()

    seen = load(args.seen)
    print(f"seen: {len(seen)} rows from {args.seen}\n")
    section_cells(seen, "SEEN", MODELS)
    section_confusion(seen, args.model)
    if args.matrix:
        section_matrix(seen, args.model)

    if args.held:
        held = load(args.held)
        print(f"held-out: {len(held)} rows from {args.held}\n")
        section_cells(held, "HELD-OUT", MODELS)
        section_confusion(held, args.model)
        if args.matrix:
            section_matrix(held, args.model)

    print("=" * 88)
    print("READING IT")
    print("=" * 88)
    print("""  CLEAN >> AMBIGUOUS for a model
      -> that model CAN ground the grid; the ambiguous phrasings, not the
         model, are producing the failures. Report the split, not the pooled
         number.

  CLEAN == AMBIGUOUS and both low
      -> the model is not grounding space at all. Phrase quality is not the
         limiting factor for it.

  Confusion concentrated on ONE wrong cell
      -> a systematic mapping error (e.g. everything -> the centre cell),
         which is a finding. Scattered confusion is closer to guessing.""")


if __name__ == "__main__":
    main()
