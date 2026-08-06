#!/usr/bin/env python
"""
bench/run_grid.py -- Phase 5 grid driver. Runs the full 5x4 = 20-cell grid by
calling the SAME run_cell() the single-cell runner uses (one measurement path).

RESUME RULE (append-only, never delete):
  A cell is "done" only if the output CSV already holds its FULL expected row
  count = len(test_rows) x repeats. On restart:
    - full cell   -> skip.
    - missing     -> run it.
    - PARTIAL cell (crashed mid-run) -> run the whole cell again, APPENDING.
  We never delete rows. A re-run partial cell leaves orphan rows behind; those
  are removed at ANALYSIS time by de-duplicating on
  (model, strategy, command_id, run_index), keeping the LAST occurrence.
  Rationale: a deletion bug corrupts irreplaceable measurement data; an append
  "bug" only leaves recoverable duplicates. Never-delete is the safe choice.

deepseek-r1 is slow + RAM-heavy, so we set OLLAMA_MAX_LOADED_MODELS=1 and
OLLAMA_KEEP_ALIVE=0 in-process before any Ollama call, so only one model is
resident at a time. Both use setdefault, so an environment value already set by
the user wins -- and the effective values are PRINTED at startup, because
Diagnostic A showed keep-alive is the single biggest determinant of both latency
and cold/warm output stability. It is an experimental condition, not a setting.

Grid output goes to bench/results_grid.csv by default (separate from smoke;
gitignored). Sealed datasets require --i-understand-heldout-is-sealed and should
be written to a SEPARATE file.

V2: --dataset now accepts seen_v2 / held_out_v2. The schema follows the dataset
via run_cell.DATASETS, so a v2 run can never be scored against the v1 schema.
V2 rows carry two extra columns (thinking, thinking_chars), so they MUST go to a
new output file; run_cell refuses to append them to a v1 CSV.

Usage:
    # prove the wrapper on a small slice first (2 models x 2 strategies, 5 cmds):
    python -m bench.run_grid --dataset seen_v2 --models "openai:gpt-4o-mini,ollama:qwen2.5:3b" --strategies "zero_shot,few_shot" --limit 5 --repeats 1 --out bench/results_v2_slice.csv

    # the real V2 seen run (all 20 cells, full test set, 5 repeats):
    python -m bench.run_grid --dataset seen_v2 --repeats 5 --out bench/results_grid_v2.csv

    # the SINGLE sealed V2 held-out pass:
    python -m bench.run_grid --dataset held_out_v2 --repeats 5 --i-understand-heldout-is-sealed --out bench/results_heldout_v2.csv
"""
import argparse, csv, os, sys
from collections import Counter
from pathlib import Path

# Set Ollama limits BEFORE anything imports/constructs a client.
# setdefault: an existing environment value wins. Printed at startup either way.
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "0")

from bench.run_cell import run_cell, resolve_clients, _load_rows, DATASETS, SEALED
from bench.dataset import DEFAULT_K
from llm.pipeline import load_schema

ALL_MODELS = [
    "openai:gpt-4o-mini",
    "ollama:llama3.2:3b",
    "ollama:qwen2.5:3b",
    "ollama:qwen2.5:0.5b",
    "ollama:deepseek-r1:7b",
]
ALL_STRATEGIES = ["zero_shot", "few_shot", "cot", "role_based"]


def count_existing(out_path):
    """Return a Counter of (model, strategy) -> number of rows already in the CSV."""
    counts = Counter()
    p = Path(out_path)
    if not p.exists():
        return counts
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[(row["model"], row["strategy"])] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help="comma-separated model names (default: all 5)")
    ap.add_argument("--strategies", default=",".join(ALL_STRATEGIES),
                    help="comma-separated strategies (default: all 4)")
    ap.add_argument("--dataset", default="seen", choices=sorted(DATASETS),
                    help="sealed datasets require --i-understand-heldout-is-sealed")
    ap.add_argument("--i-understand-heldout-is-sealed", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="first N commands (0 = full test set)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--out", default="bench/results_grid.csv")
    args = ap.parse_args()

    if args.dataset in SEALED and not args.i_understand_heldout_is_sealed:
        sys.exit(f"REFUSING: {args.dataset} is sealed. This is the SINGLE sealed pass. "
                 "Pass --i-understand-heldout-is-sealed only when you truly mean it.")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    clients = resolve_clients()
    unknown = [m for m in models if m not in clients]
    if unknown:
        sys.exit(f"unknown model(s): {unknown}. available: {list(clients.keys())}")

    # How many rows a complete cell must have.
    test_rows, exemplars = _load_rows(args.dataset, args.k)
    n_test = len(test_rows) if not (args.limit and args.limit > 0) else min(args.limit, len(test_rows))
    expected_per_cell = n_test * args.repeats

    # Schema follows the dataset. Never inferred: a v2 dataset scored against the
    # v1 schema would mark every grid location invalid.
    cfg = DATASETS[args.dataset]
    schema = load_schema(cfg["schema"])

    existing = count_existing(args.out)
    cells = [(m, s) for m in models for s in strategies]

    print("run conditions (record these in the methods chapter):")
    print(f"  OLLAMA_KEEP_ALIVE       = {os.environ.get('OLLAMA_KEEP_ALIVE')}")
    print(f"  OLLAMA_MAX_LOADED_MODELS= {os.environ.get('OLLAMA_MAX_LOADED_MODELS')}")
    print(f"  dataset  = {args.dataset}  ({cfg['file']})")
    print(f"  schema   = {cfg['schema']}  (version {schema.get('schema_version', 1)}, "
          f"{len(schema['locations'])} locations)")
    print(f"  exemplars= {cfg['exemplars_from']}  k={args.k}")
    for e in exemplars:
        print(f"      {e['command']}  ->  {e['expected'].get('location', '-')}")
    print(f"\ngrid: {len(cells)} cells | {n_test} test rows x {args.repeats} repeats "
          f"= {expected_per_cell} rows/cell | total {expected_per_cell*len(cells):,} calls")
    print(f"out={args.out}\n")

    ran, skipped = 0, 0
    for i, (model, strategy) in enumerate(cells, 1):
        have = existing.get((model, strategy), 0)
        tag = f"cell {i}/{len(cells)}  {model}/{strategy}"
        if have >= expected_per_cell:
            print(f"[skip]  {tag}  (already {have} rows)")
            skipped += 1
            continue
        if have > 0:
            print(f"[resume]{tag}  (had {have} partial rows; re-running whole cell, orphans deduped in analysis)")
        else:
            print(f"[run]   {tag}")

        written = run_cell(clients[model], strategy, args.out,
                           dataset=args.dataset, k=args.k,
                           limit=args.limit, repeats=args.repeats,
                           schema=schema, quiet=True)
        print(f"        -> wrote {written} rows")
        ran += 1

    print(f"\ndone. ran {ran} cell(s), skipped {skipped} already-complete cell(s).")
    print(f"output: {args.out}")
    print("REMEMBER: dedupe on (model,strategy,command_id,run_index) keeping LAST before computing any metric, in case a cell was resumed.")


if __name__ == "__main__":
    main()
