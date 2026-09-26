"""thesis2/final/s1_run_v5_cells.py -- STAGE 1: re-run the V5 cell grid.

    python thesis2/final/s1_run_v5_cells.py --split held_out_v3 [--kind local|cloud]
    python thesis2/final/s1_run_v5_cells.py --split held_out_v3 --cells "ollama:qwen2.5:3b:few_shot,ollama:qwen2.5:3b:rag" --replace-cells

The cell list and the repeat count per cell are READ from the V5 manifests in
bench/ and from bench/results_v5_*.csv. Nothing is invented and nothing dropped.
At start the script prints every cell, its repeat count and the expected total
row count, and GATE FAILs if the planned count differs from V5's own.

V5's structure, measured from bench/ (see common.v5_census):
    31 local cells x 217 commands x R=1  =  6,727 rows
     5 cloud cells x 217 commands x R=5  =  5,425 rows
                                   total = 12,152 rows on held_out_v3

Effective n stays 217 per split. Repeats collapse to one outcome per command by
MAJORITY VOTE for significance, exactly as A9 did. This run is 12,152 rows over
217 commands -- never describe it as 12,000 commands.

EXEMPLARS AND RETRIEVAL COME FROM V5's OWN CODE (fixed 2026-09-26)
-----------------------------------------------------------------
The first version of this script called `load_seen_split(DEFAULT_K)` and passed
the result as `examples` for both few_shot and rag. That was wrong three ways,
and it silently voided 12 of the 31 local cells (all few_shot and all rag) with
`TypeError: list indices must be integers or slices, not str` recorded as
`prompt_error`, 217/217 rows each, no model call:

  1. `load_seen_split` returns a TUPLE `(exemplars, test_rows)`, not a list;
  2. its default `seen_path` is `data/seen.jsonl` -- the V1 file. V5 draws from
     `data/seen_v2.jsonl`;
  3. rag does not use the fixed exemplar pool AT ALL. V5 builds an
     `llm.retrieve.Retriever` and retrieves per command
     (`bench/run_cell.py:279-283`).

All three are now taken from V5's own code rather than restated here:
`bench/run_cell.py::_load_rows(dataset, k)` supplies the rows and the exemplar
pool, reading the path from V5's `DATASETS` table, and the Retriever is built
ONCE and reused across cells, as run_cell's docstring requires.

Output: thesis2/out/final/results_v5cells_<split>.csv in the V5 row format.
Resumable per cell: a complete (cell, command, repeat) triple is skipped and
logged. `--replace-cells` removes only the named cells' rows first and is
`held_out_v3` ONLY -- the v4c no-overwrite rule stands.

s1_check_v5cells.py runs automatically at the end of every run.
"""
import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

V5_ROW_COLS = ["ts_iso", "model", "strategy", "dataset", "source", "command_id",
               "command", "run_index", "is_warmup", "latency_s", "stage", "ok",
               "strict_match", "normalized_match", "category", "pred_json",
               "truth_json", "reasons", "raw", "retrieval_latency_s",
               "retrieved_ids", "retrieved_scores", "tokens_in", "tokens_out",
               "cost_usd", "temperature",
               # not in V5's format: gpt-4o-mini retries for this row ("" = local)
               "retries"]


def out_path(split):
    return C.OUT / ("results_v5cells_%s.csv" % split)


def parse_cells(spec):
    """'model:strategy,model:strategy' -> {(model, strategy)}.

    A model name contains colons (ollama:qwen2.5:3b), so the STRATEGY is taken
    as the text after the LAST colon.
    """
    out = set()
    for item in (spec or "").split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            C.gate(False, ["--cells entry %r has no ':' -- use model:strategy" % item])
        model, strategy = item.rsplit(":", 1)
        out.add((model, strategy))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=list(C.SPLITS))
    ap.add_argument("--kind", choices=["local", "cloud"], default=None,
                    help="restrict to local or cloud cells (both if omitted)")
    ap.add_argument("--cells", default=None,
                    help="only these cells: 'model:strategy,...' "
                         "(strategy is the text after the last colon)")
    ap.add_argument("--replace-cells", dest="replace_cells", action="store_true",
                    help="delete the named cells' existing rows before re-running; "
                         "held_out_v3 ONLY; every use is logged")
    ap.add_argument("--retry-errors", dest="retry_errors", action="store_true",
                    help="remove and re-run ONLY rows whose stage is model_error, in "
                         "the selected cells; allowed on both splits; logged")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the cell plan and the row-count gate, run nothing")
    ap.add_argument("--no-check", action="store_true",
                    help="skip the automatic s1_check_v5cells.py at the end")
    a = ap.parse_args()
    C.start_log("s1_run_v5_cells_%s_%s" % (a.split, a.kind or "all"))

    cells = C.v5_census()
    rows_in = C.load_split(a.split)
    n_cmd = len(rows_in)
    want = parse_cells(a.cells) if a.cells else None
    if want:
        unknown = sorted(want - set(cells))
        if unknown:
            C.gate(False, ["--cells names cells that are not in the V5 census: %s"
                           % unknown,
                           "known cells: %s" % sorted("%s:%s" % k for k in cells)[:5]])

    if a.replace_cells and a.retry_errors:
        C.gate(False, ["--replace-cells and --retry-errors are exclusive"])

    # ---- rule: --replace-cells is held_out_v3 only
    if a.replace_cells:
        if a.split != "held_out_v3":
            C.gate(False, [
                "--replace-cells is held_out_v3 ONLY. The v4c no-overwrite rule "
                "stands: each arm and cell runs once on held_out_v4c.",
            ])
        if not want:
            C.gate(False, ["--replace-cells requires --cells: it must never "
                           "replace the whole grid"])

    print()
    print("  V5 CELL PLAN (read from bench/, not chosen)")
    print("  %-34s %-12s %-6s %8s %10s" % ("model", "strategy", "R", "n_cmd", "rows"))
    planned = 0
    chosen = []
    for (model, strategy), v in sorted(cells.items()):
        rows = n_cmd * v["repeats"]
        planned += rows
        skip = []
        if a.kind and v["kind"] != a.kind:
            skip.append("--kind %s" % a.kind)
        if want and (model, strategy) not in want:
            skip.append("--cells")
        if not skip:
            chosen.append((model, strategy, v))
        print("  %-34s %-12s %-6d %8d %10d%s"
              % (model, strategy, v["repeats"], n_cmd, rows,
                 "   (skipped: %s)" % ", ".join(skip) if skip else ""))
    print("  %-34s %-12s %-6s %8s %10d" % ("TOTAL (all cells)", "", "", "", planned))
    print("  V5's own held_out_v3 total: %d rows" % C.V5_EXPECTED_ROWS_V3)

    if a.split == "held_out_v3" and planned != C.V5_EXPECTED_ROWS_V3:
        C.gate(False, ["planned %d rows but V5's held_out_v3 total is %d"
                       % (planned, C.V5_EXPECTED_ROWS_V3),
                       "the cell list or the repeat counts do not match V5"])
    print("  row-count gate: PASS (%d == %d)" % (planned, C.V5_EXPECTED_ROWS_V3)
          if a.split == "held_out_v3" else
          "  row-count gate: held_out_v4c uses the same cells and repeats -> %d rows"
          % planned)
    print("  this run will execute %d cell(s)" % len(chosen))
    print("  effective n per cell: %d commands (repeats collapse by majority vote)"
          % n_cmd)

    if a.plan_only:
        C.gate(True, ["plan only; nothing was run",
                      "%d cells selected, %d rows planned"
                      % (len(chosen), sum(n_cmd * v["repeats"] for _, _, v in chosen))])

    # ---------------------------------------------------------------- setup
    from llm.pipeline import run_once
    from llm.clients import all_clients
    from bench.run_cell import apply_legend
    from bench.dataset import DEFAULT_K
    from llm import prompts as P

    apply_legend("v5")
    schema = P.load_schema(str(C.SCHEMA_FILE))
    roster = {c.name: c for c in all_clients()}

    # EXEMPLARS AND RETRIEVAL: common.v5_pool wraps bench/run_cell._load_rows,
    # so the exemplar path and k are V5's own, and rag gets a per-command
    # Retriever rather than the fixed pool. s3_energy.py and
    # s1_prompt_equality.py call the same two helpers.
    # "few_shot" only when a few_shot cell is actually selected: v5_pool gates on
    # an empty exemplar pool, and a zero_shot-only run must not be held to that.
    need_fs = any(st == "few_shot" for _, st, _ in chosen)
    v5_rows, exemplars, _probe = C.v5_pool(a.split,
                                           "few_shot" if need_fs else "zero_shot")
    if [r["command"] for r in v5_rows] != [r["command"] for r in rows_in]:
        C.gate(False, ["bench/run_cell._load_rows(%r) and common.load_split(%r) "
                       "disagree on the command list -- refusing to run"
                       % (a.split, a.split)])
    print()
    print("  exemplars: k=%d from V5's DATASETS['%s']['exemplars_from'] via "
          "bench/run_cell._load_rows" % (DEFAULT_K, a.split))
    print("     %s"
          % ("%d exemplar(s): %s" % (len(exemplars),
                                     [e["command"][:40] for e in exemplars])
             if exemplars else
             "no few_shot cell in this run, so the fixed pool is not loaded"))

    # The rag Retriever is built ONCE here and reused across every rag cell.
    retriever = None
    if any(st == "rag" for _, st, _ in chosen):
        print("  building the rag Retriever ONCE for reuse across cells")
        retriever = C.v5_retriever(quiet=True)

    p = out_path(a.split)
    done, existing = set(), []
    # --retry-errors: the model_error rows being retried. They stay in the file on
    # disk until their replacement is written, so a crash part-way loses nothing.
    pending_err = {}
    key = lambda r: (r["model"], r["strategy"], str(r["command_id"]), str(r["run_index"]))
    if a.retry_errors and not p.exists():
        C.gate(False, ["--retry-errors: %s does not exist -- nothing to retry"
                       % p.relative_to(C.REPO)])
    if p.exists():
        existing = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
        if a.replace_cells:
            before = len(existing)
            existing = [r for r in existing
                        if (r["model"], r["strategy"]) not in want]
            removed = before - len(existing)
            log = C.OUT / "force_rerun_log.csv"
            new = not log.exists()
            with log.open("a", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                if new:
                    w.writerow(["utc", "arm", "split", "rows_discarded"])
                for cell in sorted(want):
                    w.writerow([time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                "v5cell:%s/%s" % cell, a.split,
                                sum(1 for r in csv.DictReader(
                                    p.open(encoding="utf-8", newline=""))
                                    if (r["model"], r["strategy"]) == cell)])
            print("  *** --replace-cells: removed %d rows for %d cell(s); logged to %s ***"
                  % (removed, len(want), log.name))
        if a.retry_errors:
            # Allowed on held_out_v4c as well: a connection failure produced no
            # answer, so re-running it re-measures nothing.
            sel = {(m, st) for m, st, _ in chosen}
            pending_err = {key(r): r for r in existing
                           if (r["model"], r["strategy"]) in sel
                           and r.get("stage") == "model_error"}
            existing = [r for r in existing if key(r) not in pending_err]
            per_cell = {}
            for k in pending_err:
                per_cell[k[:2]] = per_cell.get(k[:2], 0) + 1
            for cell, n in sorted(per_cell.items()):
                C.log_rerun("v5cell:%s/%s [--retry-errors]" % cell, a.split, n)
            if not per_cell:
                C.log_rerun("v5cells:%d selected cell(s) [--retry-errors]" % len(sel),
                            a.split, 0)
            print("  *** --retry-errors: %d model_error row(s) in %d cell(s) will be "
                  "re-run; nothing else is; logged to force_rerun_log.csv ***"
                  % (len(pending_err), len(per_cell)))
            for cell, n in sorted(per_cell.items()):
                print("       %-34s %-12s %d row(s)" % (cell + (n,)))
            if not pending_err:
                C.gate(True, ["--retry-errors: 0 model_error rows in the %d selected "
                              "cell(s) -- nothing re-run" % len(sel)])
        done = {key(r) for r in existing}
        print("  RESUMING: %d rows kept; complete (cell, command, repeat) triples "
              "are skipped" % len(existing))

    out_rows = list(existing)
    t0 = time.time()
    missing_clients = []
    for model, strategy, v in chosen:
        if model not in roster:
            missing_clients.append(model)
            print("  *** client %r not in llm.clients.all_clients() -- cell SKIPPED "
                  "and recorded as not run ***" % model)
            continue
        client = roster[model]
        is_cloud = model.startswith("openai:")
        if is_cloud:
            # gpt-4o-mini: retried on connection errors, timeouts, 429 and 5xx,
            # up to 6 attempts from 2 s. Prompt, model and temperature unchanged.
            client = C.RetryingClient(client)
        # V5's rule, bench/run_cell.py:244 -- the fixed pool is for few_shot ONLY.
        fewshot_examples = exemplars if strategy == "few_shot" else None
        n_new, t_cell = 0, time.time()
        for cid, row in enumerate(rows_in):
            truth = C.truth_of(row)
            for rep in range(v["repeats"]):
                k = (model, strategy, str(cid), str(rep))
                if k in done:
                    continue
                if a.retry_errors and k not in pending_err:
                    continue                  # --retry-errors touches nothing else
                # V5's rag path, bench/run_cell.py:279-283, via common
                examples, ret_lat, ret_ids, ret_scores = C.v5_examples_for(
                    row["command"], strategy, fewshot_examples, retriever)
                r = run_once(row["command"], client, strategy, schema,
                             examples=examples)
                n_retry = client.last_retries if is_cloud else ""
                pending_err.pop(k, None)
                obj = getattr(r, "obj", None) or {}
                sc = C.score(obj, truth)
                out_rows.append({
                    "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": model, "strategy": strategy, "dataset": a.split,
                    "source": "typed", "command_id": cid, "command": row["command"],
                    "run_index": rep,
                    "is_warmup": (cid == 0 and rep == 0),
                    "latency_s": getattr(r, "latency_s", ""),
                    "stage": getattr(r, "stage", ""), "ok": getattr(r, "ok", ""),
                    "strict_match": bool(sc.strict_match),
                    "normalized_match": bool(sc.normalized_match),
                    "category": sc.category,
                    "pred_json": json.dumps(obj), "truth_json": json.dumps(truth),
                    "reasons": ";".join(getattr(r, "reasons", []) or []),
                    "raw": getattr(r, "raw", ""),
                    "retrieval_latency_s": ret_lat, "retrieved_ids": ret_ids,
                    "retrieved_scores": ret_scores,
                    "tokens_in": "", "tokens_out": "", "cost_usd": "",
                    "temperature": getattr(client, "temperature", ""),
                    "retries": n_retry,
                })
                n_new += 1
        print("     %-34s %-12s +%d rows (cell %.0fs, run %.0fs)"
              % (model, strategy, n_new, time.time() - t_cell, time.time() - t0))
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=V5_ROW_COLS, extrasaction="ignore")
            w.writeheader()
            w.writerows(out_rows + list(pending_err.values()))

    ran_cells = {(r["model"], r["strategy"]) for r in out_rows}
    print()
    print("  cells in the file: %d ; rows: %d" % (len(ran_cells), len(out_rows)))
    print("  output: %s" % p.relative_to(C.REPO))

    # -------------------------------------------- the audit, always, at the end
    if not a.no_check:
        print()
        print("=" * 96)
        print("RUNNING s1_check_v5cells.py -- the audit is part of every run")
        print("=" * 96)
        sys.stdout.flush()
        rc = subprocess.call([sys.executable,
                              str(Path(__file__).resolve().parent / "s1_check_v5cells.py"),
                              "--split", a.split])
        print()
        print("  s1_check_v5cells.py exit code: %d" % rc)
        if rc != 0:
            C.gate(False, ["the audit FAILED (exit %d) -- see its gate lines above" % rc,
                           "rows written: %d ; cells: %d" % (len(out_rows), len(ran_cells))])

    C.gate(not missing_clients, [
        "cells written: %d of the %d selected" % (len(ran_cells), len(chosen)),
        "rows written: %d" % len(out_rows),
        "effective n per cell: %d commands; repeats collapse by majority vote (A9)"
        % n_cmd,
        "clients missing from llm.clients: %s" % (missing_clients or "none"),
        "the audit ran and passed" if not a.no_check else "audit skipped (--no-check)",
        "output: %s" % p.relative_to(C.REPO),
    ])


if __name__ == "__main__":
    main()
