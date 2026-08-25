#!/usr/bin/env python
"""
bench/run_grid.py -- Phase 5 grid driver. Runs a grid of (model x strategy) cells
by calling the SAME run_cell() the single-cell runner uses (one measurement path).

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

Sealed datasets require --i-understand-heldout-is-sealed and should be written to
a SEPARATE file.

V2: --dataset accepts seen_v2 / held_out_v2. The schema follows the dataset via
run_cell.DATASETS, so a v2 run can never be scored against the v1 schema.

V3 (rung 2, retrieval) -- READ BEFORE RUNNING
---------------------------------------------
--strategies now accepts "rag". Three things differ for rag cells:

1. THE RETRIEVER IS BUILT ONCE, here, before the cell loop, and reused across
   every rag cell. Building it inside run_cell() would add a ~2-3 s model load
   and a corpus encode to each of the 5 cells for no reason.

2. RAG ON seen_v2 IS CONTAMINATED AND REFUSED BY DEFAULT. The corpus is
   data/train_v2.jsonl, which is 230 of the 287 seen test rows. Retrieval hands
   the model its own answer, or a same-template sibling. Those numbers are a
   sanity check, never a result. Override with
   --i-understand-rag-on-seen-is-contaminated if you deliberately want the
   sanity run. Report rag on held_out_v3.

3. RETRIEVER SETTINGS ARE EXPERIMENTAL CONDITIONS and are printed at startup
   alongside the Ollama ones: embedding model, corpus path + row count, k, and
   the leave-one-out flag. Record them in the methods chapter.

Set $env:HF_HUB_OFFLINE=1 before an unattended pass so a network hiccup at hour
six cannot kill the run. The MiniLM weights are cached after the first use.

Usage:
    # prove the wrapper on a small slice first:
    python -m bench.run_grid --dataset seen_v2 --models "openai:gpt-4o-mini,ollama:qwen2.5:3b" --strategies "zero_shot,few_shot" --limit 5 --repeats 1 --out bench/results_v3_slice.csv

    # rung 1 re-run on the new held-out set (the cost-curve baseline):
    python -m bench.run_grid --dataset held_out_v3 --repeats 5 --i-understand-heldout-is-sealed --out bench/results_heldout_v3_rung1.csv

    # rung 2, the 5 rag cells:
    python -m bench.run_grid --dataset held_out_v3 --strategies rag --repeats 5 --i-understand-heldout-is-sealed --out bench/results_heldout_v3_rag.csv

    # rung 2 sanity check on dev (NOT a result, but not contaminated either):
    python -m bench.run_grid --dataset dev_v2 --strategies rag --repeats 1 --out bench/results_dev_rag.csv
"""
import argparse, csv, os, sys
from collections import Counter
from pathlib import Path

# Set Ollama limits BEFORE anything imports/constructs a client.
# setdefault: an existing environment value wins. Printed at startup either way.
os.environ.setdefault("OLLAMA_MAX_LOADED_MODELS", "1")
os.environ.setdefault("OLLAMA_KEEP_ALIVE", "0")

from bench.run_cell import (run_cell, resolve_clients, _load_rows,
                            DATASETS, SEALED, RAG_CONTAMINATED)
from bench.dataset import DEFAULT_K
from llm.pipeline import load_schema
# Single source of truth for the strategy lists -- do not re-declare them here.
from llm.prompts import STRATEGIES as RUNG1_STRATEGIES, ALL_STRATEGIES

ALL_MODELS = [
    "openai:gpt-4o-mini",
    "ollama:llama3.2:3b",
    "ollama:qwen2.5:3b",
    "ollama:qwen2.5:0.5b",
    "ollama:deepseek-r1:7b",
]

# Default remains the frozen 20-cell rung-1 base grid. "rag" must be asked for
# explicitly (--strategies rag), so no existing command line silently grows to 25.
DEFAULT_STRATEGIES = RUNG1_STRATEGIES


def count_existing(out_path):
    """Return a Counter of (model, strategy) -> number of rows already in the CSV."""
    counts = Counter()
    p = Path(out_path)
    if not p.exists():
        return counts
    # `thinking` traces exceed the 128 KB default field cap.
    csv.field_size_limit(2**31 - 1)
    with p.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            counts[(row["model"], row["strategy"])] += 1
    return counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(ALL_MODELS),
                    help="comma-separated model names (default: all 5)")
    ap.add_argument("--strategies", default=",".join(DEFAULT_STRATEGIES),
                    help=f"comma-separated. known: {','.join(ALL_STRATEGIES)} "
                         f"(default: the 4 rung-1 strategies)")
    ap.add_argument("--dataset", default="seen_v2", choices=sorted(DATASETS),
                    help="sealed datasets require --i-understand-heldout-is-sealed")
    ap.add_argument("--i-understand-heldout-is-sealed", action="store_true")
    ap.add_argument("--i-understand-rag-on-seen-is-contaminated", action="store_true",
                    help="allow rag cells on a dataset that overlaps the retrieval corpus")
    ap.add_argument("--limit", type=int, default=0, help="first N commands (0 = full test set)")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="in-context examples: fixed exemplars (few_shot) or retrieved (rag)")
    ap.add_argument("--out", default="bench/results_grid.csv")
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    unknown_s = [s for s in strategies if s not in ALL_STRATEGIES]
    if unknown_s:
        sys.exit(f"unknown strategy/strategies: {unknown_s}. known: {ALL_STRATEGIES}")

    if args.dataset in SEALED and not args.i_understand_heldout_is_sealed:
        sys.exit(f"REFUSING: {args.dataset} is sealed. This is the SINGLE sealed pass. "
                 "Pass --i-understand-heldout-is-sealed only when you truly mean it.")

    needs_retriever = "rag" in strategies
    if needs_retriever and args.dataset in RAG_CONTAMINATED \
            and not args.i_understand_rag_on_seen_is_contaminated:
        sys.exit(
            f"REFUSING: rag on dataset={args.dataset}.\n"
            f"  Its rows overlap the retrieval corpus (data/train_v2.jsonl), so retrieval\n"
            f"  returns the query's own answer or a same-template sibling. The resulting\n"
            f"  accuracy is not reportable, and this pass costs hours.\n"
            f"  Report rag on held_out_v3. For a deliberate sanity run use dev_v2, or pass\n"
            f"  --i-understand-rag-on-seen-is-contaminated."
        )

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

    # ---- build the retriever ONCE, before the loop -------------------------
    retriever = None
    if needs_retriever:
        from llm.retrieve import Retriever, CORPUS_PATH, MODEL_NAME
        print("building retriever (once, reused across all rag cells)...")
        retriever = Retriever(quiet=False)

    existing = count_existing(args.out)
    cells = [(m, s) for m in models for s in strategies]

    print("\nrun conditions (record these in the methods chapter):")
    print(f"  OLLAMA_KEEP_ALIVE       = {os.environ.get('OLLAMA_KEEP_ALIVE')}")
    print(f"  OLLAMA_MAX_LOADED_MODELS= {os.environ.get('OLLAMA_MAX_LOADED_MODELS')}")
    print(f"  HF_HUB_OFFLINE          = {os.environ.get('HF_HUB_OFFLINE', '(unset)')}")
    print(f"  dataset  = {args.dataset}  ({cfg['file']})")
    print(f"  schema   = {cfg['schema']}  (version {schema.get('schema_version', 1)}, "
          f"{len(schema['locations'])} locations)")
    print(f"  exemplars= {cfg['exemplars_from']}  k={args.k}")
    for e in exemplars:
        print(f"      {e['command']}  ->  {e['expected'].get('location', '-')}")
    if retriever is not None:
        print(f"  RETRIEVAL (rung 2):")
        print(f"      embedding model = {retriever.model_name}")
        print(f"      corpus          = {retriever.corpus_path}  "
              f"({len(retriever.rows)} rows)")
        print(f"      dim             = {retriever.corpus_emb.shape[1]}   "
              f"device = {retriever.device}")
        print(f"      k               = {args.k}   leave-one-out = True")
        if args.dataset in RAG_CONTAMINATED:
            print(f"      *** CONTAMINATED: {args.dataset} overlaps the corpus. "
                  f"Sanity check only. ***")

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
                           schema=schema, quiet=True,
                           retriever=retriever if strategy == "rag" else None)
        print(f"        -> wrote {written} rows")
        ran += 1

    print(f"\ndone. ran {ran} cell(s), skipped {skipped} already-complete cell(s).")
    print(f"output: {args.out}")
    print("REMEMBER: dedupe on (model,strategy,command_id,run_index) keeping LAST before computing any metric, in case a cell was resumed.")
    if needs_retriever:
        print("REMEMBER: rag end-to-end latency = latency_s + retrieval_latency_s. "
              "Never report latency_s alone for a rag cell.")


if __name__ == "__main__":
    main()
