#!/usr/bin/env python
"""
bench/run_cell.py -- Phase 5 single-cell measurement runner.

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
rows are excluded from the test set for ALL strategies, so every strategy is
scored on the identical test set. held_out is SEALED and refused without a flag.
"""
import argparse, csv, json, sys, datetime
from pathlib import Path

from llm.pipeline import run_once, load_schema
from llm.score import score
from llm.clients import all_clients
from bench.dataset import load_seen_split, DEFAULT_K

# ---- Row schema: fixed NOW so no column is ever added after rows exist. ----
FIELDNAMES = [
    "ts_iso",           # wall-clock start of this run (sortable)
    "model",            # client.name
    "strategy",
    "dataset",          # seen | held_out
    "source",           # typed | whisper  (reserved; grid = typed)
    "command_id",       # index into the TEST rows (positional -- see note below)
    "command",          # raw command text (redundant but priceless for debugging)
    "run_index",        # 0..repeats-1  (the >=5-run repeat lives here)
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
    # ---- reserved nullable columns (backfilled later; reserving is free) ----
    "tokens_in",        # cloud cost axis -- recompute from prompt via tiktoken
    "tokens_out",       # cloud cost axis -- recompute from raw
    "cost_usd",         # cloud $/command
    "temperature",      # recorded from client.temperature -- honest methods chapter
]

# NOTE on command_id: 0-based index into the TEST rows (seen minus exemplars).
# Stable as long as seen.jsonl is append-only and k is unchanged. Reorder the
# file or change k and these ids shift -- treat the dataset as append-only.


def resolve_clients():
    cs = all_clients() if callable(all_clients) else all_clients
    if isinstance(cs, dict):
        cs = list(cs.values())
    return {c.name: c for c in cs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--strategy", default="zero_shot")
    ap.add_argument("--dataset", default="seen", choices=["seen", "held_out"])
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

    # --- SEAL GUARD: held_out gets read exactly once, at the real sealed pass. ---
    if args.dataset == "held_out" and not args.i_understand_heldout_is_sealed:
        sys.exit("REFUSING: held_out is sealed. This flag is the single sealed pass; "
                 "pass --i-understand-heldout-is-sealed only when you truly mean it.")

    if args.model not in clients:
        sys.exit(f"unknown model '{args.model}'. available: {list(clients.keys())}")
    client = clients[args.model]

    # load_schema() -- try no-arg first, fall back to explicit path.
    try:
        schema = load_schema()
    except TypeError:
        schema = load_schema("schema/schema.json")

    # --- Dataset split: exemplars are few-shot shots ONLY; test_rows are scored
    # for ALL strategies (identical denominator = fair comparison). held_out is
    # not handled by the split helper; if you ever run held_out it reads the raw
    # file with NO exemplars (few-shot on held_out is banned anyway).
    if args.dataset == "seen":
        exemplars, rows = load_seen_split(k=args.k)
    else:
        data_path = Path("data") / f"{args.dataset}.jsonl"
        rows = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
        exemplars = []

    # few_shot needs exemplars, which only exist for seen. On held_out, refuse.
    if args.strategy == "few_shot" and not exemplars:
        sys.exit("few_shot requires seen exemplars; not available for this dataset.")

    fewshot_examples = exemplars if args.strategy == "few_shot" else None

    if args.limit and args.limit > 0:
        rows = rows[:args.limit]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_path.exists()
    written = 0

    with out_path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if new_file:
            w.writeheader()

        for cid, row in enumerate(rows):
            command = row["command"]
            truth = row["expected"]
            for run_index in range(args.repeats):
                r = run_once(command, client, args.strategy, schema, examples=fewshot_examples)

                if r.ok and r.stage == "validated" and r.obj is not None:
                    sc = score(r.obj, truth)
                    strict, norm, cat = sc.strict_match, sc.normalized_match, sc.category
                else:
                    strict, norm, cat = False, False, f"invalid:{r.stage}"

                lat = getattr(r, "latency_s", None)
                w.writerow({
                    "ts_iso": datetime.datetime.now().isoformat(timespec="seconds"),
                    "model": getattr(r, "model", args.model),
                    "strategy": args.strategy,
                    "dataset": args.dataset,
                    "source": "typed",
                    "command_id": cid,
                    "command": command,
                    "run_index": run_index,
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
                    "tokens_in": "", "tokens_out": "", "cost_usd": "",
                    "temperature": getattr(client, "temperature", ""),
                })
                f.flush()  # survive a crash mid-grid (deepseek-r1 is slow + RAM-heavy)

                written += 1
                lat_str = f"{lat:.3f}s" if isinstance(lat, (int, float)) else "  n/a"
                print(f"[{written}] {args.model}/{args.strategy} cid={cid} "
                      f"run={run_index} {r.stage} strict={strict} {lat_str}")

    print(f"\nwrote {written} rows -> {out_path}")


if __name__ == "__main__":
    main()
