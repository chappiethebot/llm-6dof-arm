#!/usr/bin/env python
"""
bench/rescore.py -- recompute score columns on an existing grid CSV using the
CURRENT llm.score.score(), without re-running any model. Reads pred_json /
truth_json from each row, recomputes strict/normalized/category, and writes a
new CSV. Also DEDUPES on (model, strategy, command_id, run_index) keeping the
LAST occurrence, so resumed-cell orphans are dropped.

Why this exists: scoring is a pure function of (pred, truth), both of which we
stored on every row. So a scorer fix is a re-SCORE (seconds), never a re-RUN.

FIELD SIZE: the V2 schema added a `thinking` column holding deepseek-r1's full
reasoning trace. Python's csv module caps a single field at 128 KB by default,
and at least one trace exceeds that -- a runaway reasoning loop, which is itself
a measurement, not corruption. The limit is raised below. sys.maxsize is NOT
used: on Windows it overflows the underlying C long and raises OverflowError.

Usage:
    python -m bench.rescore --in bench/results_grid.csv --out bench/results_grid_scored.csv
"""
import argparse, csv, json
from pathlib import Path

from llm.score import score

# 2**31 - 1: comfortably larger than any trace, and safe for the C long on Windows.
csv.field_size_limit(2**31 - 1)


def parse(s):
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="bench/results_grid.csv")
    ap.add_argument("--out", default="bench/results_grid_scored.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(args.inp, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"no rows in {args.inp}")

    # dedupe: keep LAST occurrence per identity key (resumed-cell orphan removal)
    seen = {}
    for r in rows:
        seen[(r["model"], r["strategy"], r["command_id"], r["run_index"])] = r
    deduped = list(seen.values())
    print(f"read {len(rows)} rows, {len(deduped)} after dedupe "
          f"({len(rows) - len(deduped)} orphans dropped)")

    fieldnames = list(rows[0].keys())
    changed = 0
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in deduped:
            pred = parse(r["pred_json"])
            truth = parse(r["truth_json"])
            # strict requires a genuinely validated object AND a match; a row that
            # never validated cannot be strict-correct. normalized/category may
            # still rescue a mis-formatted-but-correct object.
            if pred is not None and truth is not None:
                sc = score(pred, truth)
                new_strict = bool(sc.strict_match) and (r["stage"] == "validated")
                new_norm = bool(sc.normalized_match)
                new_cat = sc.category
            else:
                new_strict, new_norm, new_cat = False, False, f"invalid:{r['stage']}"
            if (str(new_strict) != r["strict_match"] or str(new_norm) != r["normalized_match"]
                    or new_cat != r["category"]):
                changed += 1
            r["strict_match"] = new_strict
            r["normalized_match"] = new_norm
            r["category"] = new_cat
            w.writerow(r)
    print(f"wrote {args.out} | {changed} rows had at least one score column change")


if __name__ == "__main__":
    main()
