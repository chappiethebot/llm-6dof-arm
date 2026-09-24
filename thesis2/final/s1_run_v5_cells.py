"""thesis2/final/s1_run_v5_cells.py -- STAGE 1: re-run the V5 cell grid.

    python thesis2/final/s1_run_v5_cells.py --split held_out_v3 [--kind local|cloud]

The cell list and the repeat count per cell are READ from the V5 manifests in
bench/ and from bench/results_v5_*.csv. Nothing is invented and nothing is
dropped. At start the script prints every cell, its repeat count and the
expected total row count, and it GATE FAILs if the planned count differs from
V5's own.

V5's structure, as measured from bench/ (see common.v5_census):
    31 local cells  x 217 commands x R=1  =  6,727 rows
     5 cloud cells  x 217 commands x R=5  =  5,425 rows
                                    total = 12,152 rows on held_out_v3

held_out_v4c uses the SAME cells and the SAME repeats, so the same scale.

Effective n stays 217 per split. Repeats collapse to one outcome per command by
MAJORITY VOTE for significance, exactly as A9 did. This run is 12,152 rows over
217 commands -- never describe it as 12,000 commands.

Output: thesis2/out/final/results_v5cells_<split>.csv in the V5 row format.
Resumable per cell: a cell already complete in the output is skipped and logged.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C                                                      # noqa: E402

V5_ROW_COLS = ["ts_iso", "model", "strategy", "dataset", "source", "command_id",
               "command", "run_index", "is_warmup", "latency_s", "stage", "ok",
               "strict_match", "normalized_match", "category", "pred_json",
               "truth_json", "reasons", "raw", "tokens_in", "tokens_out",
               "cost_usd", "temperature"]


def out_path(split):
    return C.OUT / ("results_v5cells_%s.csv" % split)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--split", required=True, choices=list(C.SPLITS))
    ap.add_argument("--kind", choices=["local", "cloud"], default=None,
                    help="restrict to local or cloud cells (both if omitted)")
    ap.add_argument("--plan-only", action="store_true",
                    help="print the cell plan and the row-count gate, run nothing")
    ap.add_argument("--force-rerun-after-crash", action="store_true")
    a = ap.parse_args()
    C.start_log("s1_run_v5_cells_%s_%s" % (a.split, a.kind or "all"))

    cells = C.v5_census()
    rows_in = C.load_split(a.split)
    n_cmd = len(rows_in)

    print()
    print("  V5 CELL PLAN (read from bench/, not chosen)")
    print("  %-34s %-12s %-6s %8s %10s" % ("model", "strategy", "R", "n_cmd", "rows"))
    planned = 0
    chosen = []
    for (model, strategy), v in sorted(cells.items()):
        rows = n_cmd * v["repeats"]
        planned += rows
        mark = ""
        if a.kind and v["kind"] != a.kind:
            mark = "   (skipped: --kind %s)" % a.kind
        else:
            chosen.append((model, strategy, v))
        print("  %-34s %-12s %-6d %8d %10d%s"
              % (model, strategy, v["repeats"], n_cmd, rows, mark))
    print("  %-34s %-12s %-6s %8s %10d" % ("TOTAL (all cells)", "", "", "", planned))
    print("  V5's own held_out_v3 total: %d rows" % C.V5_EXPECTED_ROWS_V3)

    if a.split == "held_out_v3" and planned != C.V5_EXPECTED_ROWS_V3:
        C.gate(False, [
            "planned %d rows but V5's held_out_v3 total is %d"
            % (planned, C.V5_EXPECTED_ROWS_V3),
            "the cell list or the repeat counts do not match V5 -- do not run",
        ])
    print("  row-count gate: PASS (%d == %d)" % (planned, C.V5_EXPECTED_ROWS_V3)
          if a.split == "held_out_v3" else
          "  row-count gate: held_out_v4c uses the same cells and repeats -> %d rows" % planned)
    print("  effective n for every cell: %d commands (repeats collapse by majority vote)" % n_cmd)

    if a.plan_only:
        C.gate(True, ["plan only; nothing was run",
                      "%d cells selected, %d rows planned" % (len(chosen), sum(
                          n_cmd * v["repeats"] for _, _, v in chosen))])

    # ---------------------------------------------------------------- run
    from llm.pipeline import run_once
    from llm.clients import all_clients
    from bench.run_cell import apply_legend
    from llm import prompts as P
    from bench.dataset import load_seen_split, DEFAULT_K

    apply_legend("v5")
    schema = P.load_schema(str(C.SCHEMA_FILE))
    roster = {c.name: c for c in all_clients()}
    try:
        examples = load_seen_split(DEFAULT_K)
    except Exception:
        examples = None

    p = out_path(a.split)
    done = set()
    existing = []
    if p.exists() and not a.force_rerun_after_crash:
        existing = list(csv.DictReader(p.open(encoding="utf-8", newline="")))
        done = {(r["model"], r["strategy"], r["command_id"], r["run_index"])
                for r in existing}
        print("  RESUMING: %d rows present; complete (cell, command, repeat) "
              "triples are skipped" % len(existing))

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
        n_new = 0
        for cid, row in enumerate(rows_in):
            truth = C.truth_of(row)
            for rep in range(v["repeats"]):
                key = (model, strategy, str(cid), str(rep))
                if key in done:
                    continue
                res = run_once(row["command"], client, strategy, schema,
                               examples=examples if strategy in ("few_shot", "rag") else None)
                obj = getattr(res, "obj", None) or {}
                sc = C.score(obj, truth)
                out_rows.append({
                    "ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "model": model, "strategy": strategy, "dataset": a.split,
                    "source": "typed", "command_id": cid, "command": row["command"],
                    "run_index": rep,
                    # V5 marked command_id 0 of each cell as the warm-up row.
                    "is_warmup": (cid == 0 and rep == 0),
                    "latency_s": getattr(res, "latency_s", ""),
                    "stage": getattr(res, "stage", ""), "ok": getattr(res, "ok", ""),
                    "strict_match": bool(sc.strict_match),
                    "normalized_match": bool(sc.normalized_match),
                    "category": sc.category,
                    "pred_json": json.dumps(obj), "truth_json": json.dumps(truth),
                    "reasons": ";".join(getattr(res, "reasons", []) or []),
                    "raw": getattr(res, "raw", ""),
                    "tokens_in": "", "tokens_out": "", "cost_usd": "",
                    "temperature": getattr(client, "temperature", ""),
                })
                n_new += 1
        print("     %-34s %-12s +%d rows (%.0fs)" % (model, strategy, n_new, time.time() - t0))
        with p.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=V5_ROW_COLS)
            w.writeheader(); w.writerows(out_rows)

    ran_cells = {(r["model"], r["strategy"]) for r in out_rows}
    C.gate(not missing_clients, [
        "cells written: %d of the %d planned" % (len(ran_cells), len(chosen)),
        "rows written: %d" % len(out_rows),
        "effective n per cell: %d commands; repeats collapse by majority vote (A9)" % n_cmd,
        "clients missing from llm.clients: %s" % (missing_clients or "none"),
        "output: %s" % p.relative_to(C.REPO),
    ])


if __name__ == "__main__":
    main()
