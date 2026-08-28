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
    python -m bench.run_cell --model "openai:gpt-4o-mini" --strategy zero_shot --dataset seen_v2 --limit 5
    python -m bench.run_cell --model "openai:gpt-4o-mini" --strategy rag --dataset dev_v2 --limit 5

few-shot draws k fixed exemplars from the seen file (via bench/dataset.py); those
k rows are excluded from the seen test set. For held_out, few-shot uses those SAME
seen exemplars (held_out is never an example, only a test item -- seal intact).
Sealed datasets are refused by the CLI without an explicit flag.

V2 CHANGE (schema version 2): two columns appended, `thinking` and
`thinking_chars`. Ollama returns reasoning models' traces in a SEPARATE
`thinking` field; V1 read only `response`, so every deepseek-r1 trace was
generated and discarded (0 traces in 22,520 rows). These columns capture it.

V3 CHANGE (rung 2, retrieval) -- READ THIS BEFORE RUNNING
---------------------------------------------------------
Three columns appended: `retrieval_latency_s`, `retrieved_ids`, `retrieved_scores`.
They are blank for every non-rag strategy, but they are part of the header, so
this file's row schema is 28 columns and CANNOT be appended to the 25-column V2
csvs. That is deliberate and enforced by _check_header().

Consequence: EVERY run from here on -- including the rung-1 re-run on
held_out_v3 -- writes 28-column files. The old V2 csvs stay 25-column, are never
reopened, and are read separately by the analysis layer.

`retrieved_ids` is not optional bookkeeping. It is the only way to explain, after
the fact, WHY a rag cell failed on a given command. Without it a bad rag number
is uninterpretable.

LATENCY DECOMPOSITION: `latency_s` remains the LLM call ONLY, unchanged from V2,
so it stays comparable across rungs. Retrieval is timed separately in
`retrieval_latency_s`. End-to-end for a rag cell is the SUM of the two -- never
report `latency_s` alone as rag's latency.
"""
import argparse, csv, json, sys, datetime
from pathlib import Path

from llm.pipeline import run_once, load_schema
from llm.prompts import ALL_STRATEGIES


def apply_legend(version: str) -> str:
    """Select the prompt legend for this run. Returns the text now in force.

    Implemented as a one-line rebinding of llm.prompts._grid_legend rather than
    a new argument threaded through run_once/build_prompt. _schema_spec() looks
    that name up as a module global at CALL time, so rebinding it reaches every
    downstream build_prompt with no change to the measurement path -- which
    matters because llm/prompts.py and llm/pipeline.py are the provenance of the
    30,380 measured rows and must not be edited.

    v3 is the default and is a no-op, so an unflagged run is byte-identical to
    what produced those rows.
    """
    if version == "v3":
        return "v3 (unchanged; matches the 30,380 measured rows)"
    from llm import prompts as _p, prompts_v4 as _p4
    if version == "v4lite":
        _p._grid_legend = _p4._grid_legend_v4lite
        return "v4lite (V4 rule in prose; no literal cell id, no arrow notation)"
    _p._grid_legend = _p4._grid_legend_v4
    return "v4 (row B vs column 2 disambiguated; runtime symbols redefined)"
from llm.score import score
from llm.clients import all_clients
from bench.dataset import load_seen_split, DEFAULT_K

# deepseek `thinking` traces blow past Python's 128 KB default field cap.
# 2**31-1, NOT sys.maxsize, which overflows on Windows.
csv.field_size_limit(2**31 - 1)

# ---- Row schema: fixed NOW so no column is ever added after rows exist. ----
# V1 froze at 23 columns, V2 at 25. V3 is a THIRD experiment with its own files,
# so the three new columns below do not violate that lock -- V1 and V2 csvs are
# never reopened.
FIELDNAMES = [
    "ts_iso",           # wall-clock start of this run (sortable)
    "model",            # client.name
    "strategy",
    "dataset",          # seen_v2 | dev_v2 | held_out_v3 | ...
    "source",           # typed | whisper  (reserved; grid = typed)
    "command_id",       # index into the TEST rows (positional -- see note below)
    "command",          # raw command text (redundant but priceless for debugging)
    "run_index",        # 0..repeats-1  (the >=5-run repeat lives here)
    "is_warmup",        # True on the cell's first call (cid==0 and run_index==0);
                        # cold-start inflates latency -- filter on this for clean tails
    "latency_s",        # LLM-call latency ONLY (retrieval/extract/validate are outside)
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
    # ---- V3: retrieval stage (rung 2). Blank for every non-rag strategy. ----
    "retrieval_latency_s",  # encode + search. NOT included in latency_s.
    "retrieved_ids",        # JSON list of corpus row indices -- the audit trail
    "retrieved_scores",     # JSON list of cosine similarities
    # ---- reserved nullable columns (backfilled later; reserving is free) ----
    "tokens_in",        # cloud cost axis -- recompute from prompt via tiktoken
    "tokens_out",       # cloud cost axis -- recompute from raw
    "cost_usd",         # cloud $/command
    "temperature",      # recorded from client.temperature -- honest methods chapter
]

# NOTE on command_id: 0-based index into the TEST rows (seen minus exemplars, or
# the full file for non-carved datasets). Stable as long as the dataset file is
# append-only and k is unchanged. Reorder the file or change k and these ids shift.

# NOTE on is_warmup: flagged per invocation of run_cell(), on the first command's
# first repeat. The grid wrapper runs each cell start-to-finish in one call, so a
# cell gets exactly one warmup row. (Resume happens at CELL boundaries, never
# mid-cell, so this stays true.)

# NOTE on retrieved_ids: indices into the RETRIEVAL CORPUS (data/train_v2.jsonl),
# NOT into the test set. Different file, different numbering. Do not join them to
# command_id.


def resolve_clients():
    cs = all_clients() if callable(all_clients) else all_clients
    if isinstance(cs, dict):
        cs = list(cs.values())
    return {c.name: c for c in cs}


# V2/V3 datasets live alongside V1. Kept explicit so a V2 run can never silently
# read a V1 file.
#   carve=True  -> test rows are the file MINUS the k few-shot exemplars
#   carve=False -> test rows are the whole file; exemplars come from the seen file
DATASETS = {
    "seen":        {"file": "data/seen.jsonl",        "schema": "schema/schema.json",    "exemplars_from": "data/seen.jsonl",    "carve": True},
    "held_out":    {"file": "data/held_out.jsonl",    "schema": "schema/schema.json",    "exemplars_from": "data/seen.jsonl",    "carve": False},
    "seen_v2":     {"file": "data/seen_v2.jsonl",     "schema": "schema/schema_v2.json", "exemplars_from": "data/seen_v2.jsonl", "carve": True},
    "held_out_v2": {"file": "data/held_out_v2.jsonl", "schema": "schema/schema_v2.json", "exemplars_from": "data/seen_v2.jsonl", "carve": False},
    # V3 additions
    "dev_v2":      {"file": "data/dev_v2.jsonl",      "schema": "schema/schema_v2.json", "exemplars_from": "data/seen_v2.jsonl", "carve": False},
    "held_out_v3": {"file": "data/held_out_v3.jsonl", "schema": "schema/schema_v2.json", "exemplars_from": "data/seen_v2.jsonl", "carve": False},
    # held_out_v4b: held_out_v3 with the row vocabulary aligned to training
    # (upper->far, bottom->near). Ground truth identical; 72 of 217 commands
    # reworded. DELIBERATELY NOT IN `SEALED`: 72 of its commands are VERBATIM
    # training data after the reword (56 in train_v2, 14 in dev_v2), so it is a
    # memorisation probe, not a held-out KPI. See audit/build_v4b.py.
    "held_out_v4b": {"file": "data/held_out_v4b.jsonl", "schema": "schema/schema_v2.json", "exemplars_from": "data/seen_v2.jsonl", "carve": False},
    # held_out_v4c: row words replaced with front/back -- vocabulary in NEITHER
    # training nor held_out_v3. Ground truth identical; 72 of 217 reworded.
    # Verified 0 verbatim collisions with seen_v2/train_v2/dev_v2, so unlike
    # v4b this is a genuine generalisation test. SEALED-equivalent in intent:
    # it is derived from the sealed set and must not be used for tuning.
    "held_out_v4c": {"file": "data/held_out_v4c.jsonl", "schema": "schema/schema_v2.json", "exemplars_from": "data/seen_v2.jsonl", "carve": False},
}

# Sealed = run once, never used for a design decision. held_out_v2 is already
# spent; held_out_v3 is the substrate for the cost curve.
SEALED = {"held_out", "held_out_v2", "held_out_v3"}

# Datasets whose rows overlap the RAG corpus (data/train_v2.jsonl). Retrieval on
# these hands the model near-answers, so the numbers are not reportable.
RAG_CONTAMINATED = {"seen_v2"}


def _load_rows(dataset, k):
    """Return (rows, fewshot_pool).

    carve=True  (seen files): test rows = file minus k exemplars; pool = those k.
    carve=False (dev/held-out): test rows = the FULL file; pool = the SAME k seen
        exemplars. Few-shot on held-out is clean: exemplars come from seen ONLY,
        held-out is never an example -- only a test item -- so the seal and the
        generalization KPI both stay intact."""
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}. known: {sorted(DATASETS)}")
    cfg = DATASETS[dataset]
    seen_path = cfg["exemplars_from"]

    if cfg["carve"]:
        exemplars, rows = load_seen_split(k=k, seen_path=seen_path)
        return rows, exemplars

    data_path = Path(cfg["file"])
    if not data_path.exists():
        raise FileNotFoundError(f"{data_path} not found (dataset={dataset})")
    rows = [json.loads(line) for line in data_path.open(encoding="utf-8") if line.strip()]
    seen_exemplars, _ = load_seen_split(k=k, seen_path=seen_path)
    return rows, seen_exemplars


def _check_header(out_path):
    """Refuse to append V3 rows to a V1/V2 csv. Silent when the file is new."""
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
            f"  V3 rows must go to a NEW file. Use --out bench/results_v3_*.csv"
        )


def run_cell(client, strategy, out_path, *, dataset="seen_v2", k=DEFAULT_K,
             limit=0, repeats=1, schema=None, quiet=False, retriever=None):
    """Run ONE (model x strategy) cell and append rows to out_path.

    Returns the number of rows written. This is the shared path used by both the
    CLI below and the grid wrapper -- so there is exactly one scoring/timing code.

    strategy == "rag": pass a built llm.retrieve.Retriever, or one is constructed
    here. Build it ONCE and reuse it across cells -- constructing it per cell puts
    a 2-3 second model load and a corpus encode into every cell for no reason."""
    if schema is None:
        # Schema follows the dataset. A v2 dataset scored against the v1 schema
        # would mark every grid location invalid, so this is never inferred.
        schema = load_schema(DATASETS[dataset]["schema"])

    rows, exemplars = _load_rows(dataset, k)

    if strategy == "few_shot" and not exemplars:
        raise ValueError("few_shot requires seen exemplars; not available for this dataset.")
    fewshot_examples = exemplars if strategy == "few_shot" else None

    if strategy == "rag":
        if retriever is None:
            from llm.retrieve import Retriever   # imported lazily: pulls torch
            retriever = Retriever(quiet=quiet)
        if dataset in RAG_CONTAMINATED and not quiet:
            print(f"\n  *** WARNING: dataset={dataset} overlaps the RAG corpus "
                  f"(data/train_v2.jsonl).\n"
                  f"  *** Retrieval returns near-duplicates of the query. These numbers\n"
                  f"  *** are a sanity check, NOT a reportable result. Report rag on\n"
                  f"  *** held_out_v3.\n")

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

                # ---- retrieval stage (rag only), timed separately ----------
                ret_lat, ret_ids, ret_scores = "", "", ""
                examples = fewshot_examples
                if strategy == "rag":
                    res = retriever.retrieve(command, k=k, exclude_exact=True)
                    examples = res.examples
                    ret_lat = res.latency_s
                    ret_ids = json.dumps(res.ids)
                    ret_scores = json.dumps([round(s, 4) for s in res.scores])

                r = run_once(command, client, strategy, schema, examples=examples)

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
                    "retrieval_latency_s": ret_lat,
                    "retrieved_ids": ret_ids,
                    "retrieved_scores": ret_scores,
                    "tokens_in": "", "tokens_out": "", "cost_usd": "",
                    "temperature": getattr(client, "temperature", ""),
                })
                f.flush()  # survive a crash mid-cell (deepseek-r1 is slow + RAM-heavy)

                written += 1
                if not quiet:
                    lat_str = f"{lat:.3f}s" if isinstance(lat, (int, float)) else "  n/a"
                    warm = " [warmup]" if is_warmup else ""
                    think = f" think={len(thinking)}ch" if thinking else ""
                    rtv = f" ret={ret_lat*1000:.1f}ms" if isinstance(ret_lat, float) else ""
                    print(f"[{written}] {client.name}/{strategy} cid={cid} "
                          f"run={run_index} {r.stage} strict={strict} {lat_str}{warm}{think}{rtv}")

    return written


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai:gpt-4o-mini")
    ap.add_argument("--strategy", default="zero_shot", choices=ALL_STRATEGIES)
    ap.add_argument("--dataset", default="seen_v2", choices=sorted(DATASETS))
    ap.add_argument("--legend", default="v3", choices=["v3", "v4", "v4lite"],
                    help="prompt legend version. v3 (default) is what produced "
                         "the 30,380 measured rows; v4 disambiguates row B from "
                         "column 2 and rewrites the runtime-symbol definitions. "
                         "V4 runs are NOT comparable with the V3 measurement.")
    ap.add_argument("--limit", type=int, default=5, help="first N commands (smoke). 0 = all")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--k", type=int, default=DEFAULT_K,
                    help="in-context examples: fixed exemplars (few_shot) or retrieved (rag)")
    ap.add_argument("--out", default="bench/results_smoke_v3.csv")
    ap.add_argument("--list-models", action="store_true")
    ap.add_argument("--i-understand-heldout-is-sealed", action="store_true")
    args = ap.parse_args()

    apply_legend(args.legend)

    clients = resolve_clients()

    if args.list_models:
        print("available model names:", list(clients.keys()))
        print("strategies expected:  zero_shot, few_shot, cot, role_based, rag")
        print("datasets:            ", sorted(DATASETS))
        return

    if args.dataset in SEALED and not args.i_understand_heldout_is_sealed:
        sys.exit(f"REFUSING: {args.dataset} is sealed. This flag is the single sealed pass; "
                 "pass --i-understand-heldout-is-sealed only when you truly mean it.")

    if args.model not in clients:
        sys.exit(f"unknown model '{args.model}'. available: {list(clients.keys())}")
    client = clients[args.model]

    cfg = DATASETS[args.dataset]
    print(f"dataset={args.dataset}  file={cfg['file']}  schema={cfg['schema']}  "
          f"exemplars={cfg['exemplars_from']}  carve={cfg['carve']}  k={args.k}")

    written = run_cell(client, args.strategy, args.out,
                       dataset=args.dataset, k=args.k,
                       limit=args.limit, repeats=args.repeats)
    print(f"\nwrote {written} rows -> {args.out}")


if __name__ == "__main__":
    main()
