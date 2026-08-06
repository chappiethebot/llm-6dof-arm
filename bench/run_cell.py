#!/usr/bin/env python
"""
bench/run_cell.py -- Phase 5 measurement runner (single cell), and the shared
run_cell() the grid wrapper reuses so there is ONE measurement code path.

Runs ONE (model x strategy) cell over a command dataset, scores each output
against ground truth, and appends one CSV row PER RUN (not per cell).

Row granularity is deliberate: p95/p99 latency tails and accuracy variance are
computed later in analysis, from these raw per-run rows. Aggregating to a mean
here would throw the tail away -- and the tail is the novelty hook. So: exactly
one row per (command, run_index).

Usage:
    python -m bench.run_cell --list-models
    python -m bench.run_cell --model "openai:gpt-4o-mini" --strategy zero_shot --limit 5 --repeats 1
    python -m bench.run_cell --model "openai:gpt-4o-mini" --strategy few_shot  --limit 5 --repeats 1

few-shot draws k fixed exemplars from seen.jsonl (via bench/dataset.py); those k
rows are excluded from the seen test set. For held_out, few-shot uses those SAME
seen exemplars (held_out is never an example, only a test item -- seal intact).
held_out is SEALED and refused by the CLI without an explicit flag.

V2 CHANGE (schema version 2): two columns appended, `thinking` and
`thinking_chars`. Ollama returns reasoning models' traces in a SEPARATE
`thinking` field; V1 read only `response`, so every deepseek-r1 trace was
generated and discarded (0 traces in 22,520 rows). These columns capture it.

Because this changes the row schema, V2 rows MUST go to NEW csv files. Appending
to a V1 results_*.csv will raise ValueError from DictWriter -- deliberately.
"""
import argparse, csv, json, sys, datetime
from pathlib import Path

from llm.pipeline import run_once, load_schema
from llm.score import score
from llm.clients import all_clients
from bench.dataset import load_seen_split, DEFAULT_K

# ---- Row schema: fixed NOW so no column is ever added after rows exist. ----
# V1 froze at 23 columns. V2 is a SECOND EXPERIMENT with its own files, so the
# two new columns below do not violate that lock -- V1 csvs are never reopened.
FIELDNAMES = [
    "ts_iso",           # wall-clock start of this run (sortable)
    "model",            # client.name
    "strategy",
    "dataset",          # seen | held_out
    "source",           # typed | whisper  (reserved; grid = typed)
    "command_id",       # index into the TEST rows (positional -- see note below)
    "command",          # raw command text (redundant but priceless for debugging)
    "run_index",        # 0..repeats-1  (the >=5-run repeat lives here)
    "is_warmup",        # True on the cell's first call (cid==0 and run_index==0);
                        # cold-start inflates latency -- filter on this for clean tails
    "latency_s",        # LLM-call latency ONLY (extract/validate are outside the timer)
    "stage",            # validated | extract_failed | schema_failed | model_error | ...
    "ok",               # pipeline reached a valid schema object
    "strict_match",     # exact schema match vs ground truth
    "normalized_match", # lenient match
    "category",         # score category, or invalid:<stage> on pipeline failure
    "pred_json",        # predicted object (blank on failure)
    "truth_json",       # ground-truth object
    "reasons",          # failure reasons, joined (debugging)
    "raw",              # raw LLM text -- REQUIRED for post-hoc token backfill
    # ---- V2: reasoning trace ----
    "thinking",         # reasoning trace, "" for non-reasoning models. THE thesis variable.
    "thinking_chars",   # len(thinking) -- pre-computed so analysis never parses the trace
    # ---- reserved nullable columns (backfilled later; reserving is free) ----
    "tokens_in",        # cloud cost axis -- recompute from prompt via tiktoken
    "tokens_out",       # cloud cost axis -- recompute from raw
    "cost_usd",         # cloud $/command
    "temperature",      # recorded from client.temperature -- honest methods chapter
]

# NOTE on command_id: 0-based index into the TEST rows (seen minus exemplars, or
# the full held_out file). Stable as long as the dataset file is append-only and
# k is unchanged. Reorder the file or change k and these ids shift.

# NOTE on is_warmup: flagged per invocation of run_cell(), on the first command's
# first repeat. The grid wrapper runs each cell start-to-finish in one call, so a
# cell gets exactly one warmup row. (Resume happens at CELL boundaries, never
# mid-cell, so this stays true.)


def resolve_clients():
    cs = all_clients() if callable(all_clients) else all_clients
    if isinstance(cs, dict):
        cs = list(cs.values())
    return {c.name: c for c in cs}


# V2 datasets live alongside V1. `seen` -> data/seen.jsonl (schema v1),
# `seen_v2` -> data/seen_v2.jsonl (schema v2). Kept explicit so a V2 run can
# never silently read a V1 file.
DATASETS = {
    "seen":        {"file": "data/seen.jsonl",        "schema": "schema/schema.json",   "exemplars_from": "data/seen.jsonl"},
    "held_out":    {"file": "data/held_out.jsonl",    "schema": "schema/schema.json",   "exemplars_from": "data/seen.jsonl"},
    "seen_v2":     {"file": "data/seen_v2.jsonl",     "schema": "schema/schema_v2.json","exemplars_from": "data/seen_v2.jsonl"},
    "held_out_v2": {"file": "data/held_out_v2.jsonl", "schema": "schema/schema_v2.json","exemplars_from": "data/seen_v2.jsonl"},
}
SEALED = {"held_out", "held_out_v2"}


def _load_rows(dataset, k):
    """Return (rows, fewshot_pool).
    seen:     test rows = seen minus k exemplars; exemplar pool = those k.
    held_out: test rows = the FULL held-out file (sealed, read once); exemplar
              pool = the SAME k seen exemplars. Few-shot on held-out is allowed
              and clean: exemplars come from seen ONLY, held-out is never an
              example -- only a test item -- so the seal and the generalization
              KPI both stay intact."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}. known: {sorted(DATASETS)}")
    cfg = DATASETS[dataset]
    seen_path = cfg["exemplars_from"]

    if dataset not in SEALED:
        exemplars, rows = load_seen_split(k=k, seen_path=seen_path)
        return rows, exemplars
    # held-out: test on ALL its rows; exemplars come from the matching seen file.
    data_path = Path(cfg["file"])
    rows = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
    seen_exemplars, _ = load_seen_split(k=k, seen_path=seen_path)
    return rows, seen_exemplars


def _check_header(out_path):
    """Refuse to append V2 rows to a V1 csv. Silent when the file is new."""
    out_path = Path(out_path)
    if not out_path.exists() or out_path.stat().st_size == 0:
        return
    with out_path.open("r", newline="", encoding="utf-8") as f:
        existing = next(csv.reader(f), [])
    if existing and existing != FIELDNAMES:
        missing = [c for c in FIELDNAMES if c not in existing]
        sys.exit(
            f"REFUSING: {out_path} has {len(existing)} columns, this run writes {len(FIELDNAMES)}.\n"
            f"  missing there: {missing}\n"
            f"  V2 rows must go to a NEW file. Use --out bench/results_v2_*.csv"
        )


def run_cell(client, strategy, out_path, *, dataset="seen", k=DEFAULT_K,
             limit=0, repeats=1, schema=None, quiet=False):
    """Run ONE (model x strategy) cell and append rows to out_path.
    Returns the number of rows written. This is the shared path used by both the
    CLI below and the grid wrapper -- so there is exactly one scoring/timing code."""
    if schema is None:
        # Schema follows the dataset. A v2 dataset scored against the v1 schema
        # would mark every grid location invalid, so this is never inferred.
        schema = load_schema(DATASETS[dataset]["schema"])

    rows, exemplars = _load_rows(dataset, k)

    if strategy == "few_shot" and not exemplars:
        raise ValueError("few_shot requires seen exemplars; not available for this dataset.")
    fewshot_examples = exemplars if strategy == "few_shot" else None

    if limit and limit > 0:
        rows = rows[:limit]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _check_header(out_path)
    new_file = not out_path.exists()
    written = 0

    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            w.writeheader()

        for cid, row in enumerate(rows):
            command = row["command"]
            truth = row["expected"]
            for run_index in range(repeats):
                is_warmup = (cid == 0 and run_index == 0)
                r = run_once(command, client, strategy, schema, examples=fewshot_examples)

                if r.ok and r.stage == "validated" and r.obj is not None:
                    sc = score(r.obj, truth)
                    strict, norm, cat = sc.strict_match, sc.normalized_match, sc.category
                else:
                    strict, norm, cat = False, False, f"invalid:{r.stage}"

                lat = getattr(r, "latency_s", None)
                thinking = getattr(r, "thinking", "") or ""
                w.writerow({
                    "ts_iso": datetime.datetime.now().isoformat(timespec="seconds"),
                    "model": getattr(r, "model", client.name),
                    "strategy": strategy,
                    "dataset": dataset,
                    "source": "typed",
                    "command_id": cid,
                    "command": command,
                    "run_index": run_index,
                    "is_warmup": is_warmup,
                    "latency_s": lat,
                    "stage": r.stage,
                    "ok": r.ok,
                    "strict_match": strict,
                    "normalized_match": norm,
                    "category": cat,
                    "pred_json": json.dumps(r.obj) if r.obj is not None else "",
                    "truth_json": json.dumps(truth),
                    "reasons": "; ".join(r.reasons) if getattr(r, "reasons", None) else "",
                    "raw": getattr(r, "raw", "") or "",
                    "thinking": thinking,
                    "thinking_chars": len(thinking),
                    "tokens_in": "", "tokens_out": "", "cost_usd": "",
                    "temperature": getattr(client, "temperature", ""),
                })
                f.flush()  # survive a crash mid-cell (deepseek-r1 is slow + RAM-heavy)

                written += 1
                if not quiet:
                    lat_str = f"{lat:.3f}s" if isinstance(lat, (int, float)) else "  n/a"
                    warm = " [warmup]" if is_warmup else ""
                    think = f" think={len(thinking)}ch" if thinking else ""
                    print(f"[{written}] {client.name}/{strategy} cid={cid} "
                          f"run={run_index} {r.stage} strict={strict} {lat_str}{warm}{think}")

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--strategy", default="zero_shot")
    ap.add_argument("--dataset", default="seen", choices=sorted(DATASETS))
    ap.add_argument("--limit", type=int, default=5, help="first N commands (smoke). 0 = all")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--k", type=int, default=DEFAULT_K, help="few-shot exemplars reserved from seen")
    ap.add_argument("--out", default="bench/results_smoke.csv")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--i-understand-heldout-is-sealed", action="store_true")
    args = ap.parse_args()

    clients = resolve_clients()

    if args.list_models:
        print("available model names:", list(clients.keys()))
        print("strategies expected:  zero_shot, few_shot, cot, role_based")
        return

    if args.dataset in SEALED and not args.i_understand_heldout_is_sealed:
        sys.exit(f"REFUSING: {args.dataset} is sealed. This flag is the single sealed pass; "
                 "pass --i-understand-heldout-is-sealed only when you truly mean it.")

    if args.model not in clients:
        sys.exit(f"unknown model '{args.model}'. available: {list(clients.keys())}")
    client = clients[args.model]

    cfg = DATASETS[args.dataset]
    print(f"dataset={args.dataset}  file={cfg['file']}  schema={cfg['schema']}  "
          f"exemplars={cfg['exemplars_from']}  k={args.k}")

    written = run_cell(client, args.strategy, args.out,
                       dataset=args.dataset, k=args.k,
                       limit=args.limit, repeats=args.repeats)
    print(f"\nwrote {written} rows -> {args.out}")


if __name__ == "__main__":
    main()
