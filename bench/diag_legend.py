#!/usr/bin/env python
"""
bench/diag_legend.py -- Which legend wording lets the model ground the grid?

Observed in round 1 (qwen2.5:3b, 12 commands): with the one-line legend that
prompts.py ships today, ROW accuracy was 0% and COLUMN accuracy 83%. The model
answered grid_C1 for every "far left corner" -- side correct, depth inverted.

Spelling the rows out over four lines moved ROW to 100%, but COLUMN fell to 58%.
The columns line was byte-identical in both, so the drop was not caused by the
columns wording itself.

This script swaps ONLY the legend text and re-runs the same commands. It does
not touch prompts.py, the datasets, or any results file. Read-only diagnostic.

Variants:
  current       -- exactly what prompts.py ships today (one terse rows line).
  explicit      -- rows spelled out over four lines; short columns line kept.
  explicit_both -- rows AND columns both spelled out. Tests whether the column
                   drop was the short columns line being crowded out.
  anchored      -- same mapping, phrased as distance rather than far/near.

The `flipped` variant from round 1 was REMOVED. It inverted the legend and the
gold answers together, so it could not distinguish "the model read the words"
from "the model ignored the legend and I moved the target to meet it". Round 1
showed the latter: predictions were identical under both legends, only the
scoring moved. A variant that cannot fail is not a test.

Usage:
    python -m bench.diag_legend
    python -m bench.diag_legend --model ollama:qwen2.5:3b --n 12
    python -m bench.diag_legend --model openai:gpt-4o-mini
    python -m bench.diag_legend --n 27      # 3 commands per grid cell
"""

import argparse
import collections
import json
import re
import time
from pathlib import Path

from llm.clients import all_clients
from llm.extract import extract_json

SCHEMA_PATH = Path("schema/schema_v2.json")
DATA_PATH = Path("data/seen_v2.jsonl")

# The line prompts.py ships today, verbatim -- the baseline to beat.
CURRENT = "Rows:    A = far from the robot,  B = middle,  C = near the robot."
COLS_LINE = "Columns: 1 = left,  2 = middle,  3 = right."

RUNTIME_LINES = (
    "max_reach = the furthest point the arm can reach; resolved by the robot.\n"
    "nearest_free_cell = whichever square is currently empty; resolved by the robot."
)

VARIANTS = {
    "current": {
        "text": CURRENT,
        "own_columns": False,
    },
    "explicit": {
        "text": (
            "Rows run from the far side of the table to the near side.\n"
            "Row A is the row FURTHEST from the robot.\n"
            "Row B is the middle row.\n"
            "Row C is the row CLOSEST to the robot."
        ),
        "own_columns": False,
    },
    "explicit_both": {
        "text": (
            "Rows run from the far side of the table to the near side.\n"
            "Row A is the row FURTHEST from the robot.\n"
            "Row B is the middle row.\n"
            "Row C is the row CLOSEST to the robot.\n"
            "Columns run from the robot's left to the robot's right.\n"
            "Column 1 is the LEFTMOST column.\n"
            "Column 2 is the middle column.\n"
            "Column 3 is the RIGHTMOST column."
        ),
        "own_columns": True,   # supplies its own columns text; short line omitted
    },
    "anchored": {
        "text": (
            "Rows are ordered by distance from the robot base:\n"
            "A = furthest away (back of the table), "
            "B = halfway, "
            "C = nearest (front of the table, next to the robot)."
        ),
        "own_columns": False,
    },
}

_OUTPUT_RULE = (
    'Output ONLY a single JSON object with an "action" field plus that verb\'s '
    "required fields. No explanation, no markdown, no code fences."
)


def schema_spec(schema, legend_text, own_columns=False):
    lines = ["Verbs (with required fields):"]
    for verb, spec in schema["verbs"].items():
        req = spec["required"]
        lines.append(f"- {verb}: requires {', '.join(req)}" if req else f"- {verb}: (no fields)")
    lines += [
        "",
        "Allowed targets: " + ", ".join(schema["targets"]),
        "Allowed locations: " + ", ".join(schema["locations"]),
        "",
        "Grid locations are squares on the table surface, named grid_<ROW><COL>.",
        legend_text,
    ]
    if not own_columns:
        lines.append(COLS_LINE)
    lines += [RUNTIME_LINES, ""]
    return "\n".join(lines)


def build(command, schema, legend_text, own_columns=False):
    return (f"Convert the robot command into a schema JSON object.\n\n"
            f"{schema_spec(schema, legend_text, own_columns)}\n"
            f"{_OUTPUT_RULE}\n\nCommand: {command}\nJSON:")


def load_grid_rows(n):
    rows = [json.loads(l) for l in DATA_PATH.open(encoding="utf-8") if l.strip()]
    grid = [r for r in rows if str(r["expected"].get("location", "")).startswith("grid_")]
    return grid[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ollama:qwen2.5:3b")
    ap.add_argument("--n", type=int, default=12, help="grid commands to test per variant")
    args = ap.parse_args()

    clients = {c.name: c for c in all_clients()}
    if args.model not in clients:
        raise SystemExit(f"unknown model. available: {list(clients)}")
    client = clients[args.model]

    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    rows = load_grid_rows(args.n)
    if not rows:
        raise SystemExit("no grid rows found in data/seen_v2.jsonl")

    cells = sorted({r["expected"]["location"] for r in rows})
    print(f"model: {client.name}   commands: {len(rows)}   cells covered: {cells}\n")
    summary = {}

    for vname, v in VARIANTS.items():
        loc_hits = row_hits = col_hits = 0
        confusion = collections.Counter()
        t0 = time.perf_counter()

        for r in rows:
            truth_loc = r["expected"]["location"]
            raw = client.complete(build(r["command"], schema, v["text"],
                                        v.get("own_columns", False)))
            ext = extract_json(raw)
            pred_loc = ""
            if ext.ok and isinstance(ext.data, dict):
                pred_loc = str(ext.data.get("location", ""))

            if pred_loc == truth_loc:
                loc_hits += 1
            pm = re.fullmatch(r"grid_([ABC])([0-9])", pred_loc)
            tm = re.fullmatch(r"grid_([ABC])([0-9])", truth_loc)
            if pm and tm:
                if pm.group(1) == tm.group(1):
                    row_hits += 1
                if pm.group(2) == tm.group(2):
                    col_hits += 1
            confusion[f"{truth_loc} -> {pred_loc or '<none>'}"] += 1

        n = len(rows)
        summary[vname] = (loc_hits / n, row_hits / n, col_hits / n)
        print("=" * 66)
        print(f"{vname}   ({time.perf_counter()-t0:.0f}s)")
        print(f"  first legend line: {v['text'].splitlines()[0]}")
        print(f"  exact location : {loc_hits}/{n}  {loc_hits/n:.0%}")
        print(f"  ROW correct    : {row_hits}/{n}  {row_hits/n:.0%}")
        print(f"  COL correct    : {col_hits}/{n}  {col_hits/n:.0%}")
        for k, c in confusion.most_common(4):
            print(f"     {c:2d}x  {k}")
        print()

    print("=" * 66)
    print("SUMMARY")
    print(f"  {'variant':14s} {'exact':>7s} {'row':>7s} {'col':>7s}")
    best = max(summary, key=lambda k: summary[k][0])
    for k, (e, r_, c_) in summary.items():
        mark = "  <-- best exact" if k == best else ""
        print(f"  {k:14s} {e:6.0%} {r_:6.0%} {c_:6.0%}{mark}")
    print("""
READING IT:
  Row AND col both near 100%  -> that is your legend. Adopt it in prompts.py.
  Row high, col low           -> spelling out rows crowded out the columns text.
  Row low                     -> the model is not reading that wording at all.
  All variants low            -> the model cannot ground far/near on this grid.
                                 That IS the result. Report it, do not fix it.

  n=12 covers ONE cell (grid_A1). A wording that wins here may only have learned
  one corner. Re-run the winner with --n 27 before locking it in.
""")


if __name__ == "__main__":
    main()
